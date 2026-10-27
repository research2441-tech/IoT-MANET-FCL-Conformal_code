"""Conformal layer: Mondrian p-values, prediction sets, abstention, adaptive alpha,
Venn-Abers compromise probabilities and conformal test martingales."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def pvalues(test_nc, cal_nc):
    """Conformal p-values (smoothed-free, conservative) of test scores against calibration scores
    of the same class. test_nc [n], cal_nc [m] -> [n]."""
    cal = np.sort(cal_nc)
    m = len(cal)
    if m == 0:
        return np.ones_like(test_nc)
    ge = m - np.searchsorted(cal, test_nc, side="left")
    return (ge + 1.0) / (m + 1.0)


def class_pvalues(test_soft, test_maha, cal, active, use_maha=True):
    """test_soft/test_maha [n,K] nonconformity; cal[c] = (soft_scores, maha_scores) arrays.
    Returns p [n,K]; inactive classes get p = 0. Softmax and distance p-values are merged by
    the Bonferroni rule p = min(1, 2 min(p_soft, p_maha)), which keeps validity."""
    n, K = test_soft.shape
    P = np.zeros((n, K))
    for c in range(K):
        if not active[c] or c not in cal or len(cal[c][0]) == 0:
            continue
        ps = pvalues(test_soft[:, c], cal[c][0])
        if use_maha and test_maha is not None:
            pm = pvalues(test_maha[:, c], cal[c][1])
            P[:, c] = np.minimum(1.0, 2.0 * np.minimum(ps, pm))
        else:
            P[:, c] = ps
    return P


def decide_dual(P, p_nov, alpha, alpha_nov=0.05, kappa=3.0):
    """Dual decision. Alert when the benign class is rejected (p_benign <= alpha). An alert is a
    zero-day alert when the window is novel with respect to every known class (p_nov <= alpha_nov)
    or no known attack family conforms; otherwise it is a known-family alert. A window that is not
    rejected but has p_benign <= kappa * alpha while some attack family conforms is abstained.
    Codes: 0 benign, 1 known-attack alert, 2 zero-day alert, 3 abstain."""
    pb = P[..., 0]
    pa = P[..., 1:].max(-1)
    alert = pb <= alpha
    novel = (p_nov <= alpha_nov) if p_nov is not None else np.zeros_like(alert)
    unknown = alert & (novel | (pa <= alpha))
    known = alert & ~unknown
    abst = ~alert & (pb <= kappa * alpha) & (pa > alpha)
    return np.where(known, 1, np.where(unknown, 2, np.where(abst, 3, 0)))


def decide(P, alpha):
    """0 benign, 1 known-attack alert, 2 zero-day (empty set), 3 abstain (benign + attack in set)."""
    inset = P > alpha[..., None] if np.ndim(alpha) else P > alpha
    has_b = inset[..., 0]
    has_a = inset[..., 1:].any(-1)
    dec = np.where(has_b & ~has_a, 0, np.where(~has_b & has_a, 1, np.where(~has_b & ~has_a, 2, 3)))
    return dec, inset


class VennAbers:
    """Inductive Venn-Abers predictor on a scalar score (larger = more likely compromised).
    p0(s), p1(s) are tabulated on a score grid by refitting isotonic regression with the test
    point added under label 0 and label 1; the merged probability is p1 / (1 - p0 + p1)."""

    def __init__(self, scores, labels, grid=120):
        self.ok = len(np.unique(labels)) == 2 and len(labels) >= 10
        if not self.ok:
            self.const = float(np.mean(labels)) if len(labels) else 0.0
            return
        s = np.asarray(scores, float); y = np.asarray(labels, float)
        if len(s) > 1500:
            idx = np.random.default_rng(0).choice(len(s), 1500, replace=False)
            s, y = s[idx], y[idx]
        qs = np.unique(np.quantile(np.concatenate([s, [s.min() - 1, s.max() + 1]]), np.linspace(0, 1, grid)))
        p0 = np.zeros(len(qs)); p1 = np.zeros(len(qs))
        for i, q in enumerate(qs):
            for lab, out in ((0.0, p0), (1.0, p1)):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                ir.fit(np.append(s, q), np.append(y, lab))
                out[i] = ir.predict([q])[0]
        self.qs, self.p0, self.p1 = qs, p0, p1

    def predict(self, scores):
        if not self.ok:
            return np.full(np.shape(scores), self.const), np.full(np.shape(scores), self.const), np.full(np.shape(scores), self.const)
        p0 = np.interp(scores, self.qs, self.p0)
        p1 = np.interp(scores, self.qs, self.p1)
        return p1 / (1 - p0 + p1 + 1e-12), p0, p1


class JumperMartingale:
    """Simple Jumper conformal test martingale (betting functions 1 + eps (p - 1/2),
    eps in {-1, 0, 1}, jump rate J). Large values indicate the p-values are not uniform."""

    def __init__(self, J=0.01):
        self.J = J
        self.reset()

    def reset(self):
        self.C = {-1: 1 / 3, 0: 1 / 3, 1: 1 / 3}
        self.logM = 0.0

    def update(self, p):
        tot = sum(self.C.values())
        for e in self.C:
            self.C[e] = (1 - self.J) * self.C[e] + self.J * tot / 3
        for e in self.C:
            self.C[e] *= 1 + e * (p - 0.5)
        tot2 = sum(self.C.values())
        self.logM += np.log(tot2 / tot)
        s = tot2
        for e in self.C:
            self.C[e] /= s
        return self.logM


def ece(prob, y, bins=15):
    prob = np.asarray(prob); y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return e
