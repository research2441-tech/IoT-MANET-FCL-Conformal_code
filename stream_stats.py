"""Descriptive statistics of the benchmarks and the derived streams (Table I)."""
import json, os
import numpy as np
from data import get_pools
from sim import WorldConfig, make_world

RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
out = {}
for ds, seeds in [("CICIoT2023", [1, 2, 3, 4, 5]), ("CIC-DIAD2024", [1, 2, 3]), ("ToN-IoT", [1, 2, 3])]:
    P = get_pools(ds)
    r = dict(records=int(len(P.y)), families=len(P.families) - 1, record_dim=int(P.dim),
             node_dim=int(2 * P.dim + 16), per_family={f: int((P.y == i).sum()) for i, f in enumerate(P.families)})
    comp, dorm, fam_dom, drop, att_rec = [], [], [], [], []
    for s in seeds:
        w = make_world(WorldConfig(dataset=ds, seed=s))
        for d in w["domains"]:
            y = d["y"]; c = y > 0
            comp.append(c.mean()); dorm.append((~d["active"][c]).mean())
            fam_dom.append(len(np.unique(y[y > 0])))
            drop.append((d["drop"][c] > 0).mean())
    r.update(comp=float(np.mean(comp)), dormant=float(np.mean(dorm)), fam_per_domain=float(np.mean(fam_dom)),
             dropping=float(np.mean(drop)), worlds=len(seeds))
    out[ds] = r
    print(ds, r, flush=True)
json.dump(out, open(os.path.join(RES, "stream_stats.json"), "w"), indent=1)
