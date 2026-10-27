"""Compute all experimental numbers into numbers.json (no prose)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, units, boot_ci, paired, holm, friedman

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json")
KEYS = ["far", "tpr", "tpr_active", "tpr_dormant", "tpr_flag", "zd_alert", "zd_unknown", "false_unknown",
        "macro_f1", "auprc", "auroc", "ece_va", "ece_raw", "brier_va", "brier_raw", "abstain_benign",
        "coverage_known", "set_size", "known_tpr", "n_zd"]


def summ(method, seeds=None, variant="full", tag="main", dataset="CICIoT2023", section="metrics", holdout=None):
    runs = load(tag=tag, dataset=dataset, method=method, seeds=seeds, holdout=holdout)
    o = {"n": len(runs), "seeds": sorted(runs)}
    for k in KEYS + ["ho_alert", "ho_unknown", "ho_n"]:
        u = units(runs, k, variant, section) if runs else np.array([])
        o[k] = list(boot_ci(u)) if len(u) else [np.nan] * 3
        o[k + "_u"] = [float(x) for x in u]
    o["uploads"] = float(np.mean([r["comm"]["uploads"] for r in runs.values()])) if runs else np.nan
    o["wall"] = float(np.mean([r["wall"] for r in runs.values()])) if runs else np.nan
    o["train_s"] = float(np.mean([r["timing"]["train"] for r in runs.values()])) if runs else np.nan
    return o


def test(a, b, key):
    """Paired test on the (seed, domain) units that both summaries share."""
    ua, ub = np.array(a[key + "_u"]), np.array(b[key + "_u"])
    if len(ua) != len(ub):
        common = sorted(set(a["seeds"]) & set(b["seeds"]))
        da, db = len(ua) // max(1, len(a["seeds"])), len(ub) // max(1, len(b["seeds"]))
        ua = np.concatenate([ua[a["seeds"].index(s) * da:(a["seeds"].index(s) + 1) * da] for s in common])
        ub = np.concatenate([ub[b["seeds"].index(s) * db:(b["seeds"].index(s) + 1) * db] for s in common])
    p, r, n = paired(ua, ub)
    return p, r, n


def forgetting(method, seeds=None):
    """TPR on phase-0 families during the last phase, and during the first scored phase."""
    early, late = [], []
    for f in sorted(glob.glob(os.path.join(RES, f"main__CICIoT2023__{method}__s*.npz"))):
        s = int(f.split("__s")[-1].split(".")[0])
        if seeds and s not in seeds:
            continue
        if not os.path.exists(f.replace(".npz", ".json")):
            continue
        js = json.load(open(f.replace(".npz", ".json")))
        p0 = [int(k) for k, v in js["phase_of"].items() if v == 0]
        z = np.load(f)
        nd = len([k for k in z.files if k.startswith("y_")])
        for di in range(nd):
            y = z[f"y_{di}"].astype(int); dec = z[f"full_dec_{di}"]; sl = z[f"slots_{di}"].astype(int)
            alert = (dec == 1) | (dec == 2)
            m0 = np.isin(y, p0)
            e = m0 & (sl[:, None] < 300); l_ = m0 & (sl[:, None] >= 450)
            early.append(alert[e].mean() if e.any() else np.nan)
            late.append(alert[l_].mean() if l_.any() else np.nan)
    return np.array(early), np.array(late)


def main():
    N = {}
    back = ["mlp", "gru", "attn", "gnn", "gnn_gru", "gnn_attn", "lgbm", "proposed"]
    N["back"] = {m: summ(m) for m in back}
    # paired tests of proposed vs every other backbone, Holm within metric
    N["back_tests"] = {}
    for key in ["tpr", "tpr_dormant", "auprc", "macro_f1", "zd_alert", "zd_unknown", "far"]:
        ps, rs = [], []
        for m in back[:-1]:
            p, r, n = test(N["back"]["proposed"], N["back"][m], key)
            ps.append(p); rs.append(r)
        N["back_tests"][key] = {m: [float(p), float(r)] for m, p, r in zip(back[:-1], holm(ps), rs)}
    # GNN+GRU vs GRU (does message passing add anything on top of memory?)
    for key in ["tpr", "tpr_dormant", "auprc", "zd_alert"]:
        N.setdefault("graph_vs_nograph", {})[key] = list(test(N["back"]["gnn_gru"], N["back"]["gru"], key))
        N.setdefault("ssm_vs_gru", {})[key] = list(test(N["back"]["proposed"], N["back"]["gnn_gru"], key))
        N.setdefault("gnn_vs_mlp", {})[key] = list(test(N["back"]["gnn"], N["back"]["mlp"], key))
    # Friedman on AUPRC over backbones
    lens = {len(N["back"][m]["auprc_u"]) for m in back}
    mat = np.array([N["back"][m]["auprc_u"] for m in back]).T if len(lens) == 1 else np.zeros((0, 0))
    if mat.shape[0] > 2:
        chi, p, ranks = friedman(mat)
        N["friedman_auprc"] = dict(chi2=chi, p=p, ranks={m: float(r) for m, r in zip(back, ranks)})
    mat = np.array([N["back"][m]["tpr_u"] for m in back]).T if len(lens) == 1 else np.zeros((0, 0))
    if mat.shape[0] > 2:
        chi, p, ranks = friedman(mat)
        N["friedman_tpr"] = dict(chi2=chi, p=p, ranks={m: float(r) for m, r in zip(back, ranks)})
    # conformal variants on the proposed runs
    N["variants"] = {v: summ("proposed", variant=v) for v in ["full", "nomaha", "nomondrian", "nonovelty", "noaci",
                                                               "argmax"]}
    N["variants_gru"] = {v: summ("gnn_gru", variant=v) for v in ["full", "noaci", "argmax"]}
    # federated and continual strategies (seeds 1-3) against the proposed (triggered split) on the same seeds
    s3 = [1, 2]
    fl = ["fl_local", "fl_fedavg", "fl_fedprox", "fl_fedrep", "fl_split", "fl_central", "cl_static", "cl_finetune",
          "ab_nossl", "ab_noevid", "ab_globalnorm"]
    N["fl"] = {m: summ(m, seeds=s3) for m in fl}
    N["fl"]["proposed"] = summ("proposed", seeds=s3)
    N["fl_tests"] = {}
    for key in ["tpr", "auprc", "zd_alert", "macro_f1", "far", "tpr_dormant"]:
        ps, rs, names = [], [], []
        for m in fl:
            if N["fl"][m]["n"] == 0:
                continue
            p, r, n = test(N["fl"]["proposed"], N["fl"][m], key)
            ps.append(p); rs.append(r); names.append(m)
        if ps:
            N["fl_tests"][key] = {m: [float(p), float(r)] for m, p, r in zip(names, holm(ps), rs)}
    for m in ["proposed", "fl_split", "cl_finetune", "cl_static"]:
        e, l_ = forgetting(m, seeds=s3)
        N.setdefault("forget", {})[m] = dict(early=float(np.nanmean(e)), late=float(np.nanmean(l_)))
    # deployment shift
    N["deploy"] = {}
    for m in ["proposed", "mlp", "gru", "gnn_gru"]:
        runs = load(method=m)
        for s, r in runs.items():
            for spec, v in r.get("deploy", {}).items():
                for var in ["full"]:
                    for x in v[var]:
                        for k in ["far", "tpr", "auprc", "tpr_dormant"]:
                            N["deploy"].setdefault(m, {}).setdefault(spec, {}).setdefault(k, []).append(x.get(k, np.nan))
    raw = json.loads(json.dumps(N["deploy"]))
    N["deploy_tests"] = {}
    for spec in ["n_nodes=80", "watchdog_noise=3", "feature_missing=0.3"]:
        for a, b in [("gnn_gru", "gru"), ("gnn_gru", "proposed"), ("gru", "mlp"), ("proposed", "gru")]:
            for k in ["tpr", "tpr_dormant", "auprc"]:
                ua = np.array(raw[a][spec][k][-15:], float); ub = np.array(raw[b][spec][k][-15:], float)
                p, r, n = paired(ua, ub)
                N["deploy_tests"].setdefault(spec, {})[f"{a}|{b}|{k}"] = [p, r, n]
    for m in N["deploy"]:
        for spec in N["deploy"][m]:
            for k in list(N["deploy"][m][spec]):
                u = np.array(N["deploy"][m][spec][k], float)
                if spec in ("n_nodes=80", "watchdog_noise=3", "feature_missing=0.3"):
                    u = u[-15:]          # seeds 1-5 x 3 domains (seed 1 also ran two extra shifts)
                N["deploy"][m][spec][k] = list(boot_ci(u)) + [len(u)]
    # strict zero-day
    N["loafo"] = {}
    for fam in ["BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]:
        for m in ["proposed", "mlp"]:
            s = summ(m, tag="loafo", holdout=fam)
            if s["n"]:
                N["loafo"].setdefault(fam, {})[m] = {k: s[k][0] for k in ["ho_alert", "ho_unknown", "far", "tpr"]}
    # replication corpora
    N["xdata"] = {}
    for ds in ["CIC-DIAD2024", "ToN-IoT"]:
        for m in ["proposed", "mlp", "gnn_gru", "gru", "lgbm"]:
            s = summ(m, tag="xdata", dataset=ds)
            if s["n"]:
                N["xdata"].setdefault(ds, {})[m] = s
    # benign false-alarm rate per phase (phases 1-3; phase 0 is warm-up) for the stored GNN+SSM decisions
    N["far_phase"] = {}
    for ds, f in [("CICIoT2023", "main__CICIoT2023__proposed__s1.npz"),
                  ("CIC-DIAD2024", "xdata__CIC-DIAD2024__proposed__s1.npz"),
                  ("ToN-IoT", "xdata__ToN-IoT__proposed__s1.npz")]:
        fp = os.path.join(RES, f)
        if not os.path.exists(fp):
            continue
        z = np.load(fp); num = np.zeros(4); den = np.zeros(4)
        for di in range(6):
            y = z[f"y_{di}"]; d = z[f"full_dec_{di}"]; ph = np.minimum(z[f"slots_{di}"] // 150, 3)
            al = (d == 1) | (d == 2)
            for p in range(1, 4):
                m = (ph == p)[:, None] & (y == 0)
                num[p] += al[m].sum(); den[p] += m.sum()
        N["far_phase"][ds] = [float(num[p] / den[p]) for p in range(1, 4)]
    # routing
    rr = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "routing__s*.json")))]
    if rr:
        N["routing"] = {}
        for p in rr[0]:
            d = {}
            for k in ["pdr", "delay", "energy_per_pkt", "benign_excl", "low_energy_relay", "hops"]:
                u = np.array([x[k] for r in rr for x in r[p]], float)
                d[k] = list(boot_ci(u)); d[k + "_u"] = [float(v) for v in u]
            N["routing"][p] = d
        ps, names = [], []
        for p in N["routing"]:
            if p == "risk_soft":
                continue
            pv, r, n = paired(np.array(N["routing"]["risk_soft"]["pdr_u"]), np.array(N["routing"][p]["pdr_u"]))
            ps.append(pv); names.append(p)
        N["routing_tests_soft"] = {m: float(p) for m, p in zip(names, holm(ps))}
        ps, names = [], []
        for p in N["routing"]:
            if p == "conf_hard":
                continue
            pv, r, n = paired(np.array(N["routing"]["conf_hard"]["pdr_u"]), np.array(N["routing"][p]["pdr_u"]))
            ps.append(pv); names.append(p)
        N["routing_tests_hard"] = {m: float(p) for m, p in zip(names, holm(ps))}
    for f in ["pq_bench.json", "efficiency.json", "stream_stats.json"]:
        fp = os.path.join(RES, f)
        if os.path.exists(fp):
            N[f.split(".")[0]] = json.load(open(fp))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    json.dump(clean(N), open(OUT, "w"), indent=0)
    print("ok")


if __name__ == "__main__":
    main()
