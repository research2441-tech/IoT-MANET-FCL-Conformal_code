"""Routing experiment: packet delivery under each policy, driven by stored detector outputs."""
import json, os, sys
import numpy as np
from sim import WorldConfig, make_world
from routing import simulate

RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))


def load(method, seed, tag="main"):
    return np.load(os.path.join(RES, f"{tag}__CICIoT2023__{method}__s{seed}.npz"))


def main(seed):
    out_f = os.path.join(RES, f"routing__s{seed}.json")
    if os.path.exists(out_f):
        return
    w = make_world(WorldConfig(seed=seed))
    det = {m: load(m, seed) for m in ["proposed", "mlp", "gnn_gru"]}
    res = {}
    policies = [("sp", None, None, {}), ("etx", None, None, {}), ("trust", None, None, {}),
                ("hard", "proposed", "argmax", {}), ("conf_hard", "proposed", "full", {}),
                ("risk", "proposed", "full", {}), ("risk_energy", "proposed", "full", {"w_energy": 0.3}),
                ("risk_soft", "proposed", "full", {"exclude": False}),
                ("risk_soft_raw", "proposed", "raw", {"exclude": False}),
                ("risk_soft_mlp", "mlp", "full", {"exclude": False}),
                ("risk_soft_gnngru", "gnn_gru", "full", {"exclude": False}), ("oracle", None, None, {})]
    for di, dom in enumerate(w["domains"]):
        for name, m, var, kw in policies:
            info = None
            slots = det["proposed"][f"slots_{di}"].astype(int)
            if m is not None:
                z = det[m]
                if var == "argmax":
                    info = dict(dec=z[f"argmax_dec_{di}"], risk=z[f"va_{di}"].astype(float))
                elif var == "raw":
                    info = dict(dec=z[f"full_dec_{di}"], risk=z[f"praw_{di}"].astype(float))
                else:
                    info = dict(dec=z[f"full_dec_{di}"], risk=z[f"va_{di}"].astype(float))
            pol = "risk" if name.startswith("risk") else name
            r = simulate(dom, slots, pol, info, seed=seed * 10 + di, **kw)
            res.setdefault(name, []).append(r)
        print("domain", di, {k: round(v[-1]["pdr"], 3) for k, v in res.items()}, flush=True)
    json.dump(res, open(out_f, "w"))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
