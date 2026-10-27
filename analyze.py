"""Aggregate result JSONs into numbers for the manuscript: means, bootstrap CIs, paired tests."""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

RES = os.environ.get("CFG_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))


def load(tag="main", dataset="CICIoT2023", method="proposed", seeds=None, holdout=None):
    out = {}
    pat = f"{tag}__{dataset}__{method}__s*" + (f"__ho-{holdout}" if holdout else "") + ".json"
    for f in sorted(glob.glob(os.path.join(RES, pat))):
        if holdout is None and "__ho-" in f:
            continue
        r = json.load(open(f))
        if seeds is None or r["seed"] in seeds:
            out[r["seed"]] = r
    return out


def units(runs, key, variant="full", section="metrics"):
    """Per (seed, domain) values sorted by seed."""
    v = []
    for s in sorted(runs):
        for m in runs[s][section][variant]:
            v.append(m.get(key, np.nan))
    return np.array(v, float)


def boot_ci(x, n=10000, seed=0):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(a, b):
    """Wilcoxon signed-rank p and matched-pairs rank-biserial r (a vs b)."""
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    d = a - b
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0, 0.0, len(d)
    p = wilcoxon(a, b, zero_method="wilcox").pvalue
    nz = d[d != 0]
    rk = rankdata(np.abs(nz))
    r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    return float(p), float(r), len(d)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    m = len(ps)
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def friedman(mat):
    """mat [units, methods] (higher is better) -> chi2, p, mean ranks (1 = best)."""
    m = mat[~np.isnan(mat).any(1)]
    st = friedmanchisquare(*[m[:, j] for j in range(m.shape[1])])
    ranks = np.array([rankdata(-row) for row in m]).mean(0)
    return float(st.statistic), float(st.pvalue), ranks


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
