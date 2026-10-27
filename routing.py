"""Packet-level routing over the simulated IoT-MANET using per-slot detector outputs.

Each slot carries `n_pkts` packets between random legitimate endpoints. A route is chosen at
the source with Dijkstra on a policy-specific cost, then simulated hop by hop: a hop succeeds
with probability 1 - (1 - pdr)^3 (MAC retries), a compromised relay drops with its drop rate,
and per-hop delay grows with the relay's load in that slot.
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra

BIG = 1e3


def path_from(pred, s, t):
    path = [t]
    while path[-1] != s:
        p = pred[path[-1]]
        if p < 0:
            return None
        path.append(p)
    return path[::-1]


def simulate(dom, slots, policy, info=None, n_pkts=20, seed=0, w_risk=1.0, w_energy=0.0, w_load=0.0,
             e_min=0.15, drop_prior=0.6, exclude=True):
    """policy in {'sp','etx','trust','hard','conf_hard','risk','oracle'}; info: dict with per-slot
    arrays dec [S,N] and risk [S,N] aligned with `slots`."""
    rng = np.random.default_rng(seed)
    N = dom["pdr"].shape[1]
    sent = deliv = 0
    hops_l, delay_l = [], []
    energy = 0.0
    excl_benign = []
    relay_tx = 0
    relay_low = 0
    for i, t in enumerate(slots):
        pdr = dom["pdr"][t]
        y = dom["y"][t]
        drop = dom["drop"][t]
        link = pdr > 0.1
        etx = np.where(link, 1.0 / np.maximum(1 - (1 - pdr) ** 3, 1e-3), np.inf)
        node_cost = np.zeros(N)
        excluded = np.zeros(N, bool)
        if policy == "sp":
            base = np.where(link, 1.0, np.inf)
        else:
            base = etx.copy()
        if policy == "trust":
            trust = dom["X"][t, :, -1]
            node_cost += 4.0 * np.clip(0.8 - trust, 0, 1)
            excluded = trust < 0.5
        elif policy in ("hard", "conf_hard", "risk"):
            dec = info["dec"][i]
            excluded = (dec == 1) | (dec == 2)
            if policy == "risk" and not exclude:
                excluded = np.zeros(N, bool)
            if policy == "risk":
                pi = np.clip(info["risk"][i], 0, 0.999)
                node_cost += w_risk * -np.log(1 - pi * drop_prior)
                node_cost += w_energy * np.clip(e_min - dom["energy"][t], 0, None) / e_min * 5
        elif policy == "oracle":
            excluded = (y > 0) & (drop > 0)
        node_cost = node_cost + excluded * BIG
        excl_benign.append(excluded[y == 0].mean())
        load = np.zeros(N)
        legit = np.where(y == 0)[0]
        static_cost = w_load == 0 or policy != "risk"
        if static_cost:
            cost = np.where(np.isfinite(base), base + node_cost[None, :], 0)
            _, pred_all = dijkstra(cost, directed=True, return_predecessors=True)
        for _ in range(n_pkts):
            s, dst = rng.choice(legit, 2, replace=False)
            if static_cost:
                pred = pred_all[s]
            else:
                cost = base + node_cost[None, :] + w_load * load[None, :]
                cost = np.where(np.isfinite(base), cost, 0)
                _, pred = dijkstra(cost, directed=True, indices=s, return_predecessors=True)
            sent += 1
            path = path_from(pred, s, dst)
            if path is None:
                continue
            ok = True
            d_pkt = 0.0
            for a, b in zip(path[:-1], path[1:]):
                if a != s:
                    relay_tx += 1
                    relay_low += dom["energy"][t][a] < e_min
                if a != s and y[a] > 0 and rng.random() < drop[a]:
                    ok = False
                    break
                tries = 0
                success = False
                while tries < 3:
                    tries += 1
                    energy += 1.0
                    if rng.random() < pdr[a, b]:
                        success = True
                        break
                energy += 0.5 * success
                load[b] += 1
                d_pkt += tries * (1.0 + load[b] / 5.0)
                if not success:
                    ok = False
                    break
            if ok:
                deliv += 1
                hops_l.append(len(path) - 1)
                delay_l.append(d_pkt)
    return dict(pdr=deliv / max(sent, 1), hops=float(np.mean(hops_l)) if hops_l else np.nan,
                delay=float(np.mean(delay_l)) if delay_l else np.nan,
                energy_per_pkt=energy / max(deliv, 1), benign_excl=float(np.mean(excl_benign)),
                low_energy_relay=relay_low / max(relay_tx, 1))
