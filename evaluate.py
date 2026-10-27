"""Offline evaluation of stored predictions: conformal decisions, ACI, Venn-Abers, metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from conformal import class_pvalues, decide_dual, pvalues, VennAbers, ece

VARIANTS = {
    "full": dict(maha=True, mondrian=True, aci=True),
    "nomaha": dict(maha=False, mondrian=True, aci=True),
    "nomondrian": dict(maha=True, mondrian=False, aci=True),
    "nonovelty": dict(maha=True, mondrian=True, aci=True, novelty=False),
    "noaci": dict(maha=True, mondrian=True, aci=False),
    "argmax": None,
}


def cal_sets(cals, di, K, mondrian, pooled):
    out = {}
    for c in range(K):
        src = [cals[di]] if (c == 0 and mondrian) or (c > 0 and not pooled) else cals
        s = np.concatenate([x["soft"][x["y"] == c, c] for x in src])
        m = np.concatenate([x["maha"][x["y"] == c, c] for x in src])
        out[c] = (s, m)
    return out


def evaluate_domain(d, stores, K, alpha=0.05, variant="full", delay=30, gamma=0.005, pooled=True):
    v = VARIANTS[variant]
    rows = []
    alpha_t = alpha
    fb = {}  # slot -> list of (p_benign) of audited benign entries, released after delay
    out = dict(slots=[], y=[], dec=[], prob=[], va=[], pv=[], known=[], active_state=[], alpha=[])
    for st in stores:
        slots = st["slots"]
        S, N, _ = st["prob"].shape
        act = st["active"]
        if variant == "argmax":
            pred = np.argmax(np.where(act, st["prob"], -1), -1)
            dec = np.where(pred == 0, 0, 1)
            P = None
        else:
            cal = cal_sets(st["cal"], st["di"], K, v["mondrian"], pooled)
            P = class_pvalues(st["soft"].reshape(S * N, K),
                              None if st["maha"] is None or not v["maha"] else st["maha"].reshape(S * N, K),
                              cal, act, use_maha=v["maha"] and st["maha"] is not None).reshape(S, N, K)
            # one-class novelty: smallest Mahalanobis distance to any known class, calibrated on
            # known calibration entries (benign of this domain, attack entries pooled)
            pn = None
            if st["maha"] is not None and v.get("novelty", True):
                actk = np.where(act)[0]
                tm = st["maha"][..., actk].min(-1).ravel()
                cl = []
                for j, x in enumerate(st["cal"]):
                    keep = np.isin(x["y"], actk) & ((x["y"] > 0) | (j == st["di"]))
                    cl.append(x["maha"][keep][:, actk].min(-1))
                pn = pvalues(tm, np.concatenate(cl)).reshape(S, N)
            dec = np.zeros((S, N), int)
            for i, t in enumerate(slots):
                # ACI update from feedback that became available at slot t
                if v["aci"]:
                    for pb in fb.pop(t - delay, []):
                        err = float(pb <= alpha_t)
                        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err), 1e-3, 0.3))
                a = alpha_t if v["aci"] else alpha
                dec[i] = decide_dual(P[i], None if pn is None else pn[i], a)
                out["alpha"].append(a)
                aud = (d["lab_train"][t] | d["lab_cal"][t]) & (d["y"][t] == 0)
                if aud.any():
                    fb[t] = list(P[i, aud, 0])
        # Venn-Abers on the softmax log-odds of compromise, calibrated on this domain's entries
        cd = st["cal"][st["di"]]
        def logit(pb):
            pb = np.clip(pb, 1e-6, 1 - 1e-6)
            return np.log(1 - pb) - np.log(pb)
        sc = logit(np.exp(-cd["soft"][:, 0]))
        yc = (cd["y"] > 0).astype(float)
        if (yc == 1).sum() < 15:
            pool_s = np.concatenate([logit(np.exp(-x["soft"][:, 0])) for x in st["cal"]])
            pool_y = np.concatenate([(x["y"] > 0).astype(float) for x in st["cal"]])
            sc, yc = pool_s, pool_y
        va = VennAbers(sc, yc).predict(logit(st["prob"][..., 0]))[0]
        out["slots"].append(slots); out["y"].append(d["y"][slots]); out["dec"].append(dec)
        out["prob"].append(st["prob"]); out["va"].append(va)
        out["known"].append(act[d["y"][slots]])
        out["active_state"].append(d["active"][slots])
        if P is not None:
            out["pv"].append(P)
    for k in ("slots", "y", "dec", "prob", "va", "known", "active_state"):
        out[k] = np.concatenate(out[k])
    if out["pv"]:
        out["pv"] = np.concatenate(out["pv"])
    return out


def metrics(o):
    y = o["y"].ravel(); dec = o["dec"].ravel(); comp = y > 0
    known = o["known"].ravel(); actv = o["active_state"].ravel()
    pb = o["prob"][..., 0].ravel(); va = o["va"].ravel()
    alert = (dec == 1) | (dec == 2)
    r = {}
    r["far"] = alert[~comp].mean()
    r["abstain_benign"] = (dec[~comp] == 3).mean()
    r["abstain_all"] = (dec == 3).mean()
    r["tpr"] = alert[comp].mean() if comp.any() else np.nan
    r["tpr_flag"] = (alert | (dec == 3))[comp].mean() if comp.any() else np.nan
    r["tpr_active"] = alert[comp & actv].mean() if (comp & actv).any() else np.nan
    r["tpr_dormant"] = alert[comp & ~actv].mean() if (comp & ~actv).any() else np.nan
    zd = comp & ~known
    r["n_zd"] = int(zd.sum())
    r["zd_alert"] = alert[zd].mean() if zd.any() else np.nan
    r["zd_unknown"] = (dec[zd] == 2).mean() if zd.any() else np.nan
    kn = comp & known
    r["known_tpr"] = alert[kn].mean() if kn.any() else np.nan
    r["false_unknown"] = (dec[~comp] == 2).mean()
    s = 1 - pb
    if comp.any() and (~comp).any():
        r["auroc"] = roc_auc_score(comp, s)
        r["auprc"] = average_precision_score(comp, s)
        r["auprc_va"] = average_precision_score(comp, va)
    # closed-set macro F1 on windows whose class was known when predicted
    kk = known
    pred = np.argmax(o["prob"].reshape(-1, o["prob"].shape[-1]), -1)
    if kk.any():
        r["macro_f1"] = f1_score(y[kk], pred[kk], average="macro", labels=np.unique(y[kk]))
        r["f1_attack"] = f1_score(y[kk] > 0, pred[kk] > 0)
    # calibration of compromise probability
    raw = 1 - pb
    r["ece_va"] = ece(va, comp); r["ece_raw"] = ece(raw, comp)
    r["brier_va"] = np.mean((va - comp) ** 2); r["brier_raw"] = np.mean((raw - comp) ** 2)
    if len(o["pv"]):
        P = o["pv"].reshape(-1, o["pv"].shape[-1])
        inset = P > 0.05
        r["coverage_known"] = inset[np.where(known)[0], y[known]].mean() if known.any() else np.nan
        r["set_size"] = inset.sum(1).mean()
    return {k: float(v) for k, v in r.items()}
