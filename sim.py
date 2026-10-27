"""Mobile IoT-MANET stream simulator driven by real benchmark records.

A world holds several MANET domains. Each domain is a set of mobile IoT nodes moving under
random waypoint mobility. In every slot a node emits a small bag of traffic records drawn
from the benchmark: benign records from its device profile, attack records from the family of
an ongoing compromise. Compromised nodes are on-off (bursty) and may also drop forwarded
packets. The simulator returns a streaming attributed graph per domain:
  X   [T, N, Fn]   node-window features
  E   [T, N, N, Fe] directed edge features
  M   [T, N, N]    edge mask (radio links or observed communication)
plus ground truth (family label, activity, drop rate) and the radio state used by routing.
"""
import numpy as np
from data import get_pools, N_PHASE

FLOOD = {"CICIoT2023": ["DDoS", "DoS"], "CIC-DIAD2024": ["DDoS", "DoS"], "ToN-IoT": ["ddos", "dos"]}
SCAN = {"CICIoT2023": ["Recon", "Mirai"], "CIC-DIAD2024": ["Recon", "Mirai"], "ToN-IoT": ["scanning"]}
ARRIVALS = {7: [3, 2, 1, 1], 8: [3, 2, 2, 1]}


class WorldConfig:
    def __init__(self, dataset="CICIoT2023", n_domains=6, n_nodes=40, T=600, seed=0,
                 radio_range=250.0, watchdog_noise=1.0, speed_scale=1.0, holdout=None,
                 presence=0.65, feature_missing=0.0):
        self.__dict__.update(locals())
        self.__dict__.pop("self")


def _rwp_step(pos, wp, speed, pause, side, rng, dt=10.0, vmin=0.5, vmax=10.0):
    d = wp - pos
    dist = np.linalg.norm(d, axis=1) + 1e-9
    moving = pause <= 0
    step = np.minimum(speed * dt, dist)
    newpos = pos + (d / dist[:, None]) * (step * moving)[:, None]
    arrived = moving & (dist <= speed * dt)
    pause = np.where(arrived, rng.uniform(0, 30, len(pos)), pause - dt)
    wp = np.where(arrived[:, None], rng.uniform(0, side, (len(pos), 2)), wp)
    speed = np.where(arrived, rng.uniform(vmin, vmax, len(pos)), speed)
    v_eff = step * moving / dt
    return newpos, wp, speed, pause, v_eff


def simulate_domain(cfg, d, fam_arrival, rng):
    P = get_pools(cfg.dataset)
    N, T = cfg.n_nodes, cfg.T
    F = P.dim
    fams = P.families
    flood = {P.fam_index[f] for f in FLOOD[cfg.dataset] if f in P.fam_index}
    scan = {P.fam_index[f] for f in SCAN[cfg.dataset] if f in P.fam_index}
    phase_len = T // N_PHASE
    side = rng.uniform(700, 1100) * np.sqrt(N / 40.0)
    vmax = [2, 5, 8, 12, 16, 20][d % 6] * cfg.speed_scale
    prof_mix = rng.dirichlet(np.full(P.n_profiles, 0.4))
    profile = rng.choice(P.n_profiles, N, p=prof_mix)
    lam = rng.uniform(2, 6, N)
    peers = np.stack([rng.choice(np.delete(np.arange(N), i), 2, replace=False) for i in range(N)])
    pos = rng.uniform(0, side, (N, 2))
    wp = rng.uniform(0, side, (N, 2))
    speed = rng.uniform(0.5, vmax, N)
    pause = np.zeros(N)
    energy = rng.uniform(0.6, 1.0, N)
    trust = np.full(N, 0.8)
    # ---- compromise schedule -------------------------------------------------------------
    fam_of = np.zeros((T, N), int)          # 0 = benign
    victim_of = np.full((T, N), -1)
    rho_t = np.zeros((T, N)); drop_t = np.zeros((T, N)); poff_t = np.zeros((T, N)); pon_t = np.ones((T, N))
    episodes = []
    for f, a in fam_arrival.items():
        t = a
        while t < T - 10:
            k = int(rng.integers(3, 6)) if f in flood else int(rng.integers(1, 3))
            dur = int(rng.integers(30, 121))
            episodes.append((t, f, k, dur))
            t += dur + int(rng.integers(40, 160))
    episodes.sort()
    for (t0, f, k, dur) in episodes:
        t1 = min(T, t0 + dur)
        free = np.where((fam_of[t0:t1] == 0).all(0))[0]
        if len(free) < k + 1:
            continue
        chosen = rng.choice(free, k, replace=False)
        tgt = rng.choice(np.setdiff1d(free, chosen))
        fam_of[t0:t1, chosen] = f
        victim_of[t0:t1, chosen] = tgt
        for v in chosen:
            u = rng.random()
            rho_t[t0:t1, v] = rng.uniform(0.15, 0.9)
            drop_t[t0:t1, v] = 0.0 if u < 0.4 else (rng.uniform(0.3, 0.8) if u < 0.8 else 1.0)
            poff_t[t0:t1, v] = rng.uniform(0.05, 0.3)
            pon_t[t0:t1, v] = rng.uniform(0.1, 0.5)
    # ---- outputs ---------------------------------------------------------------------------
    Fn = 2 * F + 16
    X = np.zeros((T, N, Fn), np.float32)
    E = np.zeros((T, N, N, 9), np.float32)
    Mk = np.zeros((T, N, N), bool)
    PDR = np.zeros((T, N, N), np.float32)
    POS = np.zeros((T, N, 2), np.float32)
    ENER = np.zeros((T, N), np.float32)
    active = np.zeros((T, N), bool)
    act = np.zeros(N, bool)
    prev_nb = None
    prev_dist = None
    prev_deg = np.zeros(N)
    for t in range(T):
        phase = min(t // phase_len, N_PHASE - 1)
        pos, wp, speed, pause, v_eff = _rwp_step(pos, wp, speed, pause, side, rng, vmax=vmax)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        pdr = 1.0 / (1.0 + np.exp((dist - 0.85 * cfg.radio_range) / 25.0))
        pdr = pdr * (1 - rng.uniform(0, 0.15, (N, N)))
        pdr = np.minimum(pdr, pdr.T)
        np.fill_diagonal(pdr, 0)
        radio = pdr > 0.1
        deg = radio.sum(1)
        comp = fam_of[t] > 0
        # on-off activity of compromised nodes
        flip_off = rng.random(N) < poff_t[t]
        flip_on = rng.random(N) < pon_t[t]
        act = np.where(comp, np.where(act, ~flip_off, flip_on), False)
        newly = comp & (t > 0) & (fam_of[max(t - 1, 0)] == 0)
        act = act | newly
        active[t] = act
        # ---- traffic records ----
        n_rec = 1 + rng.poisson(lam)
        sent_rec = []
        src_l, dst_l = [], []
        for v in range(N):
            n = n_rec[v]
            is_att = np.zeros(n, bool)
            if act[v]:
                is_att = rng.random(n) < rho_t[t, v]
                is_att[0] = True
            ids = np.empty(n, int)
            nb = int((~is_att).sum())
            if nb:
                ids[~is_att] = P.sample_benign(rng, profile[v], phase, nb)
            if is_att.any():
                ids[is_att] = P.sample_attack(rng, fam_of[t, v], phase, int(is_att.sum()))
            dst = peers[v, rng.integers(0, 2, n)]
            if is_att.any():
                f = fam_of[t, v]
                na = int(is_att.sum())
                if f in flood:
                    dst[is_att] = victim_of[t, v]
                elif f in scan:
                    dst[is_att] = rng.choice(np.delete(np.arange(N), v), na)
                else:
                    dst[is_att] = victim_of[t, v]
            sent_rec.append(ids)
            src_l.append(np.full(n, v)); dst_l.append(dst)
        src = np.concatenate(src_l); dst = np.concatenate(dst_l); rid = np.concatenate(sent_rec)
        R = P.X[rid]
        Z = P.Z[rid]
        # per-node aggregates of sent records
        cnt = np.bincount(src, minlength=N).astype(np.float32)
        sumR = np.zeros((N, F), np.float32); np.add.at(sumR, src, R)
        meanR = sumR / cnt[:, None]
        maxR = np.full((N, F), -8.0, np.float32); np.maximum.at(maxR, src, R)
        pair = src * N + dst
        n_dest = np.array([len(np.unique(dst[src == v])) for v in range(N)], np.float32)
        rcnt = np.bincount(dst, minlength=N).astype(np.float32)
        sumZin = np.zeros((N, 4), np.float32); np.add.at(sumZin, dst, Z)
        meanZin = sumZin / np.maximum(rcnt, 1)[:, None]
        pc = np.bincount(pair, minlength=N * N).reshape(N, N).astype(np.float32)
        sumZp = np.zeros((N * N, 4), np.float32); np.add.at(sumZp, pair, Z)
        meanZp = (sumZp / np.maximum(pc.reshape(-1), 1)[:, None]).reshape(N, N, 4)
        # ---- forwarding evidence (watchdog) ----
        asked = rng.poisson(1.0 + 0.4 * deg)
        dropr = np.where(comp, drop_t[t], 0.0)
        cong = np.clip(0.02 * (cnt + rcnt) / 6.0, 0, 0.3)
        fwd = rng.binomial(asked, np.clip((1 - dropr) * (1 - cong), 0, 1))
        q = np.where(deg > 0, (pdr * radio).sum(1) / np.maximum(deg, 1), 0.3)
        q = np.clip(q ** cfg.watchdog_noise, 0.05, 1)
        obs = rng.binomial(fwd, q)
        ratio = (obs + 1.0) / (asked + 2.0)
        trust = 0.9 * trust + 0.1 * ratio
        energy = np.clip(energy - 2e-4 - 6e-5 * (cnt + rcnt + asked), 0, 1)
        nbset = radio
        churn = np.zeros(N) if prev_nb is None else (nbset ^ prev_nb).sum(1) / np.maximum((nbset | prev_nb).sum(1), 1)
        ddist = np.zeros((N, N)) if prev_dist is None else (dist - prev_dist) / 100.0
        node_misc = np.stack([
            np.log1p(cnt), n_dest / 5.0, np.log1p(rcnt), v_eff / 25.0, deg / 10.0,
            np.abs(deg - prev_deg) / 5.0, churn, energy, q, np.log1p(asked), ratio, trust], 1)
        X[t] = np.concatenate([meanR, maxR, meanZin, node_misc], 1)
        e = np.zeros((N, N, 9), np.float32)
        e[..., 0] = radio
        e[..., 1] = pdr
        e[..., 2] = np.clip(ddist, -3, 3)
        e[..., 3] = np.log1p(pc)
        e[..., 4] = np.log1p(pc.T)
        e[..., 5:9] = meanZp
        E[t] = e
        Mk[t] = radio | (pc > 0) | (pc.T > 0)
        np.fill_diagonal(Mk[t], False)
        PDR[t] = pdr * radio
        POS[t] = pos
        ENER[t] = energy
        prev_nb, prev_dist, prev_deg = nbset, dist, deg
    if cfg.feature_missing > 0:
        miss = rng.random(X.shape[:2] + (1,)) < cfg.feature_missing
        X = np.where(miss & (np.arange(Fn) < 2 * F), 0.0, X).astype(np.float32)
    return dict(X=X, E=E, M=Mk, y=fam_of.astype(np.int32), active=active, drop=drop_t.astype(np.float32),
                pdr=PDR, pos=POS, energy=ENER, side=side, vmax=vmax)




def make_world(cfg):
    P = get_pools(cfg.dataset)
    rng = np.random.default_rng(1000 + cfg.seed)
    K = len(P.families) - 1
    order = rng.permutation(np.arange(1, K + 1))
    counts = ARRIVALS[K]
    phase_of = {}
    i = 0
    for ph, c in enumerate(counts):
        for f in order[i:i + c]:
            phase_of[int(f)] = ph
        i += c
    phase_len = cfg.T // N_PHASE
    doms = []
    for d in range(cfg.n_domains):
        drng = np.random.default_rng(cfg.seed * 100 + d)
        arr = {}
        for f, ph in phase_of.items():
            if ph == 0 or drng.random() < cfg.presence:
                start = ph * phase_len
                arr[f] = int(start + (drng.integers(0, 15) if ph == 0 else drng.integers(0, int(0.9 * phase_len))))
        dom = simulate_domain(cfg, d, {f: a for f, a in arr.items() if a < cfg.T - 20}, drng)
        dom["arrival"] = arr
        doms.append(dom)
    return dict(domains=doms, phase_of=phase_of, families=P.families, n_classes=len(P.families),
                feat_dim=doms[0]["X"].shape[-1], cfg=cfg)
