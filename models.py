"""Graph encoders, temporal modules and heads (JAX).

Backbone = per-slot edge-gated message passing over the streaming graph, followed by a
temporal module applied to each node's sequence of slot embeddings:
  'ssm'  selective state-space recurrence with input-dependent step, input and output maps
  'gru'  gated recurrent unit
  'attn' one causal multi-head self-attention block over the window
  'none' no temporal mixing (each slot classified alone)
Graph = 'gnn' (edge-gated message passing) or 'mlp' (node features only).
"""
import jax
import jax.numpy as jnp
import numpy as np

D_MODEL = 48
D_INNER = 64
N_STATE = 8
L_WIN = 32
N_HEADS = 4


def _dense(key, i, o, scale=1.0):
    return {"w": jax.random.normal(key, (i, o)) * scale / np.sqrt(i), "b": jnp.zeros(o)}


def _ln_init(d):
    return {"g": jnp.ones(d), "b": jnp.zeros(d)}


def ln(p, x):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * p["g"] + p["b"]


def lin(p, x):
    return x @ p["w"] + p["b"]


def init_params(key, feat_dim, edge_dim, n_classes, graph="gnn", temporal="ssm", n_layers=2):
    ks = iter(jax.random.split(key, 64))
    d = D_MODEL
    P = {"inp": _dense(next(ks), feat_dim, d), "inp_ln": _ln_init(d), "mask_tok": jnp.zeros(feat_dim)}
    if graph == "gnn":
        P["gate"] = _dense(next(ks), edge_dim, n_layers * N_HEADS)
        P["gnn"] = [{"msg": _dense(next(ks), d, d),
                     "up1": _dense(next(ks), 2 * d, 2 * d), "up2": _dense(next(ks), 2 * d, d, 0.5),
                     "ln": _ln_init(d)} for _ in range(n_layers)]
    else:
        P["mlp"] = [{"up1": _dense(next(ks), d, 2 * d), "up2": _dense(next(ks), 2 * d, d, 0.5),
                     "ln": _ln_init(d)} for _ in range(n_layers)]
    if temporal == "ssm":
        A = np.tile(np.arange(1, N_STATE + 1, dtype=np.float32), (D_INNER, 1))
        P["ssm"] = {"in": _dense(next(ks), d, 2 * D_INNER), "dt": _dense(next(ks), D_INNER, D_INNER, 0.1),
                    "B": _dense(next(ks), D_INNER, N_STATE), "C": _dense(next(ks), D_INNER, N_STATE),
                    "A_log": jnp.log(jnp.asarray(A)) - 2.0, "D": jnp.ones(D_INNER),
                    "out": _dense(next(ks), D_INNER, d, 0.5), "ln": _ln_init(d)}
        P["ssm"]["dt"]["b"] = jnp.full(D_INNER, -2.0)
    elif temporal == "gru":
        P["gru"] = {"x": _dense(next(ks), d, 3 * d), "h": _dense(next(ks), d, 3 * d), "ln": _ln_init(d)}
    elif temporal == "attn":
        P["attn"] = {"qkv": _dense(next(ks), d, 3 * d), "o": _dense(next(ks), d, d, 0.5),
                     "pos": jax.random.normal(next(ks), (L_WIN, d)) * 0.02, "ln1": _ln_init(d),
                     "f1": _dense(next(ks), d, 2 * d), "f2": _dense(next(ks), 2 * d, d, 0.5), "ln2": _ln_init(d)}
    P["head_ln"] = _ln_init(d)
    P["head"] = _dense(next(ks), d, n_classes, 0.5)
    # self-supervised decoders (masked feature and masked topology reconstruction)
    P["dec_feat"] = _dense(next(ks), d, feat_dim)
    P["dec_edge"] = {"w": jax.random.normal(next(ks), (d, d)) * 0.1}
    return P


# ---------------------------------------------------------------- graph encoder
def graph_encode(P, X, E, M, node_mask=None):
    """X [G,N,F], E [G,N,N,Fe] (E[u,v]: edge u->v), M [G,N,N] -> H [G,N,d]."""
    if node_mask is not None:
        X = jnp.where(node_mask[..., None], P["mask_tok"], X)
    h = ln(P["inp_ln"], jax.nn.gelu(lin(P["inp"], X)))
    if "gnn" in P:
        Mf = M.astype(h.dtype)
        deg = Mf.sum(1)[..., None] + 1.0                     # in-degree of v
        nl = len(P["gnn"])
        # edge gates: one sigmoid per edge, layer and head (heads split the channels)
        gates = jax.nn.sigmoid(lin(P["gate"], E)) * Mf[..., None]   # [G,N,N,nl*H]
        G_, N_ = h.shape[0], h.shape[1]
        for li, lp in enumerate(P["gnn"]):
            g = gates[..., li * N_HEADS:(li + 1) * N_HEADS]
            hm = lin(lp["msg"], h).reshape(G_, N_, N_HEADS, -1)      # [G,N,H,d/H] (source u)
            m = jnp.einsum("guvh,guhe->gvhe", g, hm).reshape(G_, N_, -1) / deg
            up = lin(lp["up2"], jax.nn.gelu(lin(lp["up1"], jnp.concatenate([h, m], -1))))
            h = ln(lp["ln"], h + up)
    else:
        for lp in P["mlp"]:
            h = ln(lp["ln"], h + lin(lp["up2"], jax.nn.gelu(lin(lp["up1"], h))))
    return h


# ---------------------------------------------------------------- temporal modules
def ssm_seq(p, x, h0=None):
    """Selective state-space block. x [S,L,d] -> y [S,L,d], final state."""
    uz = lin(p["in"], x)
    u, z = jnp.split(uz, 2, -1)
    u = jax.nn.silu(u)
    dt = jax.nn.softplus(lin(p["dt"], u))                  # [S,L,Di]  input-dependent step
    Bt = lin(p["B"], u)                                     # [S,L,n]   input-dependent input map
    Ct = lin(p["C"], u)                                     # [S,L,n]   input-dependent output map
    A = -jnp.exp(p["A_log"])                                # [Di,n]
    S = x.shape[0]
    h = jnp.zeros((S, D_INNER, N_STATE)) if h0 is None else h0

    def step(h, inp):
        dt_t, b_t, c_t, u_t = inp
        h = jnp.exp(dt_t[..., None] * A) * h + (dt_t * u_t)[..., None] * b_t[:, None, :]
        return h, jnp.einsum("sdn,sn->sd", h, c_t)

    xs = tuple(jnp.swapaxes(v, 0, 1) for v in (dt, Bt, Ct, u))
    h, ys = jax.lax.scan(step, h, xs)                      # causal recurrence, O(L) per sequence
    ys = jnp.swapaxes(ys, 0, 1)
    y = ys + p["D"] * u
    out = lin(p["out"], y * jax.nn.silu(z))
    return ln(p["ln"], x + out), h


def gru_seq(p, x, h0=None):
    S, L, d = x.shape
    xg = lin(p["x"], x)
    h = jnp.zeros((S, d)) if h0 is None else h0

    def step(h, xt):
        hg = lin(p["h"], h)
        r = jax.nn.sigmoid(xt[:, :d] + hg[:, :d])
        zz = jax.nn.sigmoid(xt[:, d:2 * d] + hg[:, d:2 * d])
        n = jnp.tanh(xt[:, 2 * d:] + r * hg[:, 2 * d:])
        h = (1 - zz) * n + zz * h
        return h, h

    h, ys = jax.lax.scan(step, h, jnp.swapaxes(xg, 0, 1))
    return ln(p["ln"], x + jnp.swapaxes(ys, 0, 1)), h


def attn_seq(p, x):
    S, L, d = x.shape
    xp = x + p["pos"][-L:]
    q, k, v = jnp.split(lin(p["qkv"], xp), 3, -1)
    hd = d // N_HEADS
    q = q.reshape(S, L, N_HEADS, hd); k = k.reshape(S, L, N_HEADS, hd); v = v.reshape(S, L, N_HEADS, hd)
    att = jnp.einsum("slhe,sthe->shlt", q, k) / np.sqrt(hd)
    causal = jnp.tril(jnp.ones((L, L), bool))
    att = jnp.where(causal, att, -1e9)
    att = jax.nn.softmax(att, -1)
    o = jnp.einsum("shlt,sthe->slhe", att, v).reshape(S, L, d)
    x = ln(p["ln1"], x + lin(p["o"], o))
    return ln(p["ln2"], x + lin(p["f2"], jax.nn.gelu(lin(p["f1"], x))))


def temporal(P, h):
    """h [B,L,N,d] -> [B,L,N,d] (causal along L)."""
    B, L, N, d = h.shape
    s = jnp.transpose(h, (0, 2, 1, 3)).reshape(B * N, L, d)
    if "ssm" in P:
        s, _ = ssm_seq(P["ssm"], s)
    elif "gru" in P:
        s, _ = gru_seq(P["gru"], s)
    elif "attn" in P:
        s = attn_seq(P["attn"], s)
    return jnp.transpose(s.reshape(B, N, L, d), (0, 2, 1, 3))


def forward(P, X, E, M, class_mask):
    """X [B,L,N,F] ... -> logits [B,L,N,K], embedding [B,L,N,d]."""
    B, L, N, F = X.shape
    h = graph_encode(P, X.reshape(B * L, N, F), E.reshape((B * L, N, N, -1)), M.reshape(B * L, N, N))
    h = temporal(P, h.reshape(B, L, N, -1))
    z = ln(P["head_ln"], h)
    logits = lin(P["head"], z)
    logits = jnp.where(class_mask, logits, -1e9)
    return logits, z


# ---------------------------------------------------------------- losses
def sup_loss(P, X, E, M, Y, W, class_mask, cls_w):
    logits, _ = forward(P, X, E, M, class_mask)
    lp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]
    w = W * cls_w[Y]
    return (nll * w).sum() / (w.sum() + 1e-6)


def ssl_loss(P, X, E, M, key, p_node=0.3, p_edge=0.3):
    """Masked feature + masked topology reconstruction on single slots. X [G,N,F]."""
    k1, k2, k3 = jax.random.split(key, 3)
    G, N, F = X.shape
    nm = jax.random.uniform(k1, (G, N)) < p_node
    radio = E[..., 0] > 0.5
    em = (jax.random.uniform(k2, (G, N, N)) < p_edge) & radio
    em = em | jnp.swapaxes(em, 1, 2)
    M2 = M & ~em
    E2 = jnp.where(em[..., None], 0.0, E)
    h = graph_encode(P, X, E2, M2, node_mask=nm)
    rec = lin(P["dec_feat"], h)
    xn = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + 1e-6)
    rn = rec / (jnp.linalg.norm(rec, axis=-1, keepdims=True) + 1e-6)
    cos_err = (1 - (xn * rn).sum(-1)) ** 2                 # scaled cosine error (gamma = 2)
    l_feat = (cos_err * nm).sum() / (nm.sum() + 1e-6)
    s = jnp.einsum("gud,de,gve->guv", h, P["dec_edge"]["w"], h)
    neg = (jax.random.uniform(k3, (G, N, N)) < 0.05) & ~radio
    pos_l = -jax.nn.log_sigmoid(s) * em
    neg_l = -jax.nn.log_sigmoid(-s) * neg
    l_edge = pos_l.sum() / (em.sum() + 1e-6) + neg_l.sum() / (neg.sum() + 1e-6)
    return l_feat + l_edge
