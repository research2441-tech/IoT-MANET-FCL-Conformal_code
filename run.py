"""Run one (dataset, method, seed) configuration and save metrics and per-slot outputs."""
import argparse, json, os, time
import numpy as np
from sim import WorldConfig, make_world
from fcl import Protocol, METHODS
from evaluate import evaluate_domain, metrics, VARIANTS

RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CICIoT2023")
    ap.add_argument("--method", default="proposed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout", default=None)          # family name never labelled
    ap.add_argument("--tag", default="main")
    ap.add_argument("--deploy", default=None)           # e.g. n_nodes=80 or watchdog_noise=3
    ap.add_argument("--save_slots", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    name = f"{a.tag}__{a.dataset}__{a.method}__s{a.seed}" + (f"__ho-{a.holdout}" if a.holdout else "")
    out_json = os.path.join(RES, name + ".json")
    if os.path.exists(out_json):
        print("exists", out_json); return
    t0 = time.time()
    w = make_world(WorldConfig(dataset=a.dataset, seed=a.seed))
    hold = None
    if a.holdout:
        hold = w["families"].index(a.holdout)
    pr = Protocol(w, METHODS[a.method], seed=a.seed, holdout=hold, verbose=True)
    log = pr.run()
    res = dict(dataset=a.dataset, method=a.method, seed=a.seed, holdout=a.holdout, tag=a.tag,
               timing=pr.timing, comm=pr.comm, log=log, trig=[d.get('trig_log', []) for d in pr.doms], families=w["families"],
               phase_of={str(k): v for k, v in w["phase_of"].items()}, wall=time.time() - t0)

    def score(doms, prefix):
        out = {}
        arrays = {}
        for var in (VARIANTS if not METHODS[a.method].lgbm else ["full", "argmax", "noaci"]):
            per = []
            for di, d in enumerate(doms):
                o = evaluate_domain(d, d["store"], pr.K, variant=var, delay=pr.delay)
                m = metrics(o)
                if hold is not None:
                    hz = o["y"] == hold
                    alert = (o["dec"] == 1) | (o["dec"] == 2)
                    m["ho_n"] = int(hz.sum())
                    m["ho_alert"] = float(alert[hz].mean()) if hz.any() else float("nan")
                    m["ho_unknown"] = float((o["dec"][hz] == 2).mean()) if hz.any() else float("nan")
                per.append(m)
                if var in ("full", "argmax"):
                    arrays[f"{var}_dec_{di}"] = o["dec"].astype(np.int8)
                    if var == "full":
                        arrays[f"va_{di}"] = o["va"].astype(np.float16)
                        arrays[f"praw_{di}"] = (1 - o["prob"][..., 0]).astype(np.float16)
                        arrays[f"slots_{di}"] = o["slots"].astype(np.int16)
                        arrays[f"y_{di}"] = o["y"].astype(np.int8)
            out[var] = per
        res[prefix] = out
        return arrays

    arrays = score(pr.doms, "metrics")
    if a.save_slots:
        np.savez_compressed(os.path.join(RES, name + ".npz"), **arrays)
    if a.deploy:
        res["deploy"] = {}
        for spec in a.deploy.split(";"):
            kv = dict(x.split("=") for x in spec.split(","))
            kw = {k: (int(v) if k in ("n_nodes", "n_domains") else float(v)) for k, v in kv.items()}
            kw.setdefault("n_domains", 2 if kw.get("n_nodes", 40) >= 100 else 3)
            w2 = make_world(WorldConfig(dataset=a.dataset, seed=a.seed + 500, **kw))
            pr.deploy(w2)
            sub = {}
            for var in ["full", "argmax"]:
                sub[var] = [metrics(evaluate_domain(d, d["store"], pr.K, variant=var, delay=pr.delay)) for d in pr.doms]
            res["deploy"][spec] = sub
            print("deploy", spec, np.nanmean([m["far"] for m in sub["full"]]), np.nanmean([m["tpr"] for m in sub["full"]]))
    json.dump(res, open(out_json, "w"))
    f = res["metrics"]["full"]
    print("DONE", name, f"{res['wall']:.0f}s", {k: round(float(np.nanmean([m.get(k, np.nan) for m in f])), 3)
                                               for k in ["far", "tpr", "tpr_dormant", "zd_alert", "zd_unknown",
                                                         "macro_f1", "auprc", "ece_va", "ece_raw"]})


if __name__ == "__main__":
    main()
