import os
"""Result figures (grayscale). Reads result JSON/NPZ files from the experiment directory."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, units, boot_ci

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "hatch.linewidth": 0.5})
NAMES = {"mlp": "MLP", "gru": "GRU", "attn": "Transformer", "gnn": "GNN", "gnn_gru": "GNN+GRU",
         "gnn_attn": "GNN+Attn.", "lgbm": "LightGBM", "proposed": "GNN+SSM"}


def fig_backbones(order):
    keys = [("tpr_active", "TPR (active)"), ("tpr_dormant", "TPR (dormant)"), ("auprc", "AUPRC")]
    fills = ["0.25", "0.65", "white"]
    hatches = [None, None, "////"]
    fig, ax = plt.subplots(figsize=(3.45, 2.15), dpi=600)
    x = np.arange(len(order)); w = 0.26
    for j, (k, lab) in enumerate(keys):
        m, lo, hi = [], [], []
        for meth in order:
            mu, a, b = boot_ci(units(load(method=meth), k))
            m.append(mu); lo.append(mu - a); hi.append(b - mu)
        ax.bar(x + (j - 1) * w, m, w * 0.92, color=fills[j], edgecolor="black", lw=0.5, hatch=hatches[j], label=lab,
               yerr=[lo, hi], error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5))
    ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in order], rotation=30, ha="right", fontsize=6.3)
    ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("score", fontsize=7)
    ax.legend(loc="upper center", ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.02),
              handlelength=1.4, columnspacing=1.0)
    ax.grid(axis="y", color="0.85", lw=0.4); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig3_backbones.png"), dpi=600)


def fig_reliability():
    zs = [np.load(f) for f in sorted(glob.glob(os.path.join(RES, "main__CICIoT2023__proposed__s*.npz")))]
    va = np.concatenate([z[k].astype(float).ravel() for z in zs for k in z.files if k.startswith("va_")])
    pr = np.concatenate([z[k].astype(float).ravel() for z in zs for k in z.files if k.startswith("praw_")])
    y = np.concatenate([(z[k] > 0).ravel() for z in zs for k in z.files if k.startswith("y_")]).astype(float)
    bins = np.linspace(0, 1, 11)
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), dpi=600, gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
    for p, lab, mk, ls in [(pr, "softmax", "o", "--"), (va, "Venn-Abers", "s", "-")]:
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        cx, cy = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 30:
                cx.append(p[m].mean()); cy.append(y[m].mean())
        ax.plot(cx, cy, marker=mk, ms=2.8, lw=0.8, ls=ls, color="black", mfc="white" if mk == "o" else "black",
                label=lab)
    ax.set_xlabel("predicted probability of compromise", fontsize=6.5)
    ax.set_ylabel("observed frequency", fontsize=6.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax = axs[1]
    hp, _ = np.histogram(pr, bins=bins); hv, _ = np.histogram(va, bins=bins)
    cen = (bins[:-1] + bins[1:]) / 2
    ax.bar(cen - 0.024, hp, 0.046, color="white", edgecolor="black", lw=0.5, hatch="////", label="softmax", log=True)
    ax.bar(cen + 0.024, hv, 0.046, color="0.4", edgecolor="black", lw=0.5, label="Venn-Abers", log=True)
    ax.set_ylim(500, 4e7)
    ax.set_xlabel("probability", fontsize=6.5); ax.set_ylabel("node slots", fontsize=6.5)
    ax.tick_params(labelsize=6); ax.legend(fontsize=5.8, frameon=False, loc="upper center")
    for a in axs:
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig4_reliability.png"), dpi=600)


def first_seen_curves(method, maxlag=200, step=25):
    """Alert rate on compromised slots versus slots elapsed since the family first appeared in the domain."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    hits = {}
    for f in sorted(glob.glob(os.path.join(RES, f"main__CICIoT2023__{method}__s*.npz"))):
        z = np.load(f)
        ndom = len([k for k in z.files if k.startswith("y_")])
        for di in range(ndom):
            y = z[f"y_{di}"].astype(int); dec = z[f"full_dec_{di}"]; sl = z[f"slots_{di}"].astype(int)
            for fam in np.unique(y[y > 0]):
                t0 = sl[np.where((y == fam).any(1))[0][0]]
                if t0 <= sl[0] + 10:          # family already present when scoring started
                    continue
                for i, t in enumerate(sl):
                    m = y[i] == fam
                    if m.any():
                        lag = (t - t0) // step
                        if lag * step < maxlag:
                            hits.setdefault(lag, []).extend(list(((dec[i][m] == 1) | (dec[i][m] == 2)).astype(float)))
    lags = sorted(hits)
    return np.array(lags) * step + step / 2, np.array([np.mean(hits[l]) for l in lags])


def fig_firstseen(methods, labels, fname="fig5_firstseen.png"):
    fig, ax = plt.subplots(figsize=(3.45, 1.9), dpi=600)
    styles = [("-", "o", "black"), ("--", "s", "black"), ("-.", "^", "0.45"), (":", "D", "0.45"), ("-", "v", "0.6")]
    for (m, lab), (ls, mk, c) in zip(zip(methods, labels), styles):
        x, yv = first_seen_curves(m)
        if len(x):
            ax.plot(x, yv, ls=ls, marker=mk, ms=2.8, lw=0.8, color=c, mfc="white", label=lab)
    ax.set_xlabel("slots since the family first appeared in the domain", fontsize=6.5)
    ax.set_ylabel("alert rate (K or Z)", fontsize=6.5)
    ax.set_ylim(0.3, 1.0); ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    ax.grid(color="0.88", lw=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, fname), dpi=600)


def fig_routing(policies, labels, fname="fig6_routing.png"):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "routing__s*.json")))]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), dpi=600, gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    y = np.arange(len(policies))[::-1]
    for i, p in enumerate(policies):
        v = np.array([u["pdr"] for r in runs for u in r[p]])
        m, lo, hi = boot_ci(v)
        ax.barh(y[i], m, 0.62, color="0.35" if p.startswith("risk") else ("white" if p != "oracle" else "0.8"),
                edgecolor="black", lw=0.5, xerr=[[m - lo], [hi - m]], error_kw=dict(lw=0.5, capsize=1.2))
        ax.text(m + 0.012, y[i], f"{m:.3f}", va="center", fontsize=5.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
    lo_x = min(np.mean([u["pdr"] for r in runs for u in r[p]]) for p in policies)
    ax.set_xlim(max(0, lo_x - 0.08), 1.0); ax.set_xlabel("packet delivery ratio", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax = axs[1]
    for i, p in enumerate(policies):
        ex = np.mean([u["benign_excl"] for r in runs for u in r[p]])
        ax.barh(y[i], ex * 100, 0.62, color="white", edgecolor="black", lw=0.5, hatch="////")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("benign relays excluded (%)", fontsize=6.5); ax.tick_params(labelsize=6)
    for a in axs:
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(OUT, fname), dpi=600)


def fig_efficiency(fname="fig7_efficiency.png"):
    e = json.load(open(os.path.join(RES, "efficiency.json")))
    Ls = sorted(int(k) for k in e["window_ms"])
    fig, ax = plt.subplots(figsize=(3.45, 1.9), dpi=600)
    ax.plot(Ls, [e["window_ms"][str(L)]["attn"] for L in Ls], "-o", color="black", ms=2.8, lw=0.8, mfc="white",
            label="causal attention, window re-read")
    ax.plot(Ls, [e["window_ms"][str(L)]["ssm_window"] for L in Ls], "--s", color="0.4", ms=2.8, lw=0.8,
            mfc="white", label="state-space, window re-read")
    ax.axhline(e["stream_ms"]["ssm"], color="black", lw=0.8, ls=":", label="state-space, streaming state")
    ax.axhline(e["stream_ms"]["gru"], color="0.5", lw=0.8, ls="-.", label="GRU, streaming state")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("temporal context $L$ (slots)", fontsize=6.5)
    ax.set_ylabel("ms per slot (40 nodes)", fontsize=6.5)
    ax.tick_params(labelsize=6); ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, fname), dpi=600)


def fig_loafo(fname="fig7_loafo.png"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    fams = ["BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
    vals = {}
    for m in ["proposed", "mlp"]:
        for f in fams:
            runs = load(tag="loafo", method=m, holdout=f)
            if runs:
                vals[(m, f)] = (np.nanmean(units(runs, "ho_alert")), np.nanmean(units(runs, "ho_unknown")))
    fams = [f for f in fams if ("proposed", f) in vals and ("mlp", f) in vals]
    fig, ax = plt.subplots(figsize=(3.45, 2.0), dpi=600)
    x = np.arange(len(fams)); w = 0.2
    specs = [("proposed", 0, "GNN+SSM, any alert", "0.25", None), ("proposed", 1, "GNN+SSM, recognised as Z", "white", "////"),
             ("mlp", 0, "MLP, any alert", "0.65", None), ("mlp", 1, "MLP, recognised as Z", "white", "...")]
    for j, (m, i, lab, c, h) in enumerate(specs):
        ax.bar(x + (j - 1.5) * w, [vals[(m, f)][i] for f in fams], w * 0.9, color=c, hatch=h, edgecolor="black",
               lw=0.5, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=6.2, rotation=20, ha="right")
    ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("rate on held-out family", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.8)
    ax.grid(axis="y", color="0.88", lw=0.4); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, fname), dpi=600)


if __name__ == "__main__":
    what = sys.argv[1:] or ["backbones", "reliability", "firstseen"]
    if "backbones" in what:
        fig_backbones(["mlp", "gru", "attn", "gnn", "gnn_gru", "gnn_attn", "lgbm", "proposed"])
    if "reliability" in what:
        fig_reliability()
    if "firstseen" in what:
        fig_firstseen(["fl_local", "fl_fedrep", "fl_fedavg", "fl_split"], ["local only", "FedRep", "FedAvg", "split (ours)"])
