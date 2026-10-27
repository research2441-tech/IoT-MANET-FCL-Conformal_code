"""Computational cost: parameters, training-step time, and per-slot streaming latency as the
temporal context grows (state-space recurrence and GRU carry a state; attention re-reads the window)."""
import json, os, time
import numpy as np
import jax, jax.numpy as jnp
import models as Mdl

RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
F, N = 90, 40


def nparams(P, drop=("dec_feat", "dec_edge", "mask_tok")):
    return int(sum(np.size(x) for k, v in P.items() if k not in drop for x in jax.tree_util.tree_leaves(v)))


def timeit(f, *a, reps=30):
    r = f(*a); jax.block_until_ready(r)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); r = f(*a); jax.block_until_ready(r); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def main():
    out = {"params": {}, "step_ms": {}, "stream_ms": {}, "window_ms": {}}
    key = jax.random.PRNGKey(0)
    B, L = 4, Mdl.L_WIN
    X = jax.random.normal(key, (B, L, N, F)); E = jax.random.normal(key, (B, L, N, N, 9)) * 0.3
    M = jax.random.uniform(key, (B, L, N, N)) < 0.3
    Y = jnp.zeros((B, L, N), int); W = jnp.ones((B, L, N)); cm = jnp.ones(8, bool); cw = jnp.ones(8)
    cfgs = {"mlp": ("mlp", "none"), "gru": ("mlp", "gru"), "attn": ("mlp", "attn"), "gnn": ("gnn", "none"),
            "gnn_gru": ("gnn", "gru"), "gnn_attn": ("gnn", "attn"), "proposed": ("gnn", "ssm")}
    for name, (g, tm) in cfgs.items():
        P = Mdl.init_params(key, F, 9, 8, g, tm)
        out["params"][name] = nparams(P)
        grad = jax.jit(jax.grad(Mdl.sup_loss))
        out["step_ms"][name] = timeit(grad, P, X, E, M, Y, W, cm, cw, reps=10)
        print(name, out["params"][name], round(out["step_ms"][name], 1), flush=True)
    # per-slot temporal cost for one domain (N sequences), excluding the shared graph encoder
    d = Mdl.D_MODEL
    Pssm = Mdl.init_params(key, F, 9, 8, "gnn", "ssm")["ssm"]
    Pgru = Mdl.init_params(key, F, 9, 8, "gnn", "gru")["gru"]
    Patt = Mdl.init_params(key, F, 9, 8, "gnn", "attn")["attn"]
    x1 = jax.random.normal(key, (N, 1, d))
    s_step = jax.jit(lambda p, x, h: Mdl.ssm_seq(p, x, h))
    g_step = jax.jit(lambda p, x, h: Mdl.gru_seq(p, x, h))
    h_s = jnp.zeros((N, Mdl.D_INNER, Mdl.N_STATE)); h_g = jnp.zeros((N, d))
    out["stream_ms"]["ssm"] = timeit(s_step, Pssm, x1, h_s)
    out["stream_ms"]["gru"] = timeit(g_step, Pgru, x1, h_g)
    for Lw in [16, 32, 64, 128, 256, 512, 1024]:
        Pa = dict(Patt); Pa["pos"] = jax.random.normal(key, (Lw, d)) * 0.02
        xw = jax.random.normal(key, (N, Lw, d))
        att = jax.jit(lambda p, x: Mdl.attn_seq(p, x)[:, -1])
        ssw = jax.jit(lambda p, x: Mdl.ssm_seq(p, x)[0][:, -1])
        out["window_ms"][str(Lw)] = {"attn": timeit(att, Pa, xw, reps=15), "ssm_window": timeit(ssw, Pssm, xw, reps=15)}
        print(Lw, out["window_ms"][str(Lw)], flush=True)
    json.dump(out, open(os.path.join(RES, "efficiency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
