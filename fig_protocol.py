import os
"""Fig. 2: prequential protocol (a) and family arrival schedule of evaluation world 1 (b)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_pools, N_PHASE
from sim import ARRIVALS

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7})


def arrivals(seed, T=600, n_domains=6, presence=0.65):
    P = get_pools("CICIoT2023")
    rng = np.random.default_rng(1000 + seed)
    K = len(P.families) - 1
    order = rng.permutation(np.arange(1, K + 1))
    phase_of, i = {}, 0
    for ph, c in enumerate(ARRIVALS[K]):
        for f in order[i:i + c]:
            phase_of[int(f)] = ph
        i += c
    pl = T // N_PHASE
    out = []
    for d in range(n_domains):
        drng = np.random.default_rng(seed * 100 + d)
        arr = {}
        for f, ph in phase_of.items():
            if ph == 0 or drng.random() < presence:
                arr[f] = int(ph * pl + (drng.integers(0, 15) if ph == 0 else drng.integers(0, int(0.9 * pl))))
        out.append({f: a for f, a in arr.items() if a < T - 20})
    return P.families, phase_of, out


fams, phase_of, arr = arrivals(1)
fig = plt.figure(figsize=(3.5, 3.05), dpi=600)
ax = fig.add_axes([0.13, 0.66, 0.84, 0.30])
for k in range(4):
    ax.add_patch(Rectangle((k * 150, 0), 150, 1, fc=["0.80", "0.90", "0.80", "0.90"][k], ec="none"))
    ax.text(k * 150 + 75, 0.86, f"phase {k}", ha="center", va="center", fontsize=6.5)
ax.text(75, 0.62, "warm-up, not scored;\nself-supervised pretraining", ha="center", va="center", fontsize=6)
for tc in range(150, 600, 50):
    ax.plot([tc, tc], [0.22, 0.42], color="black", lw=0.6)
    ax.plot(tc, 0.42, marker="v", color="black", ms=3)
tc = 350
ax.add_patch(Rectangle((0, 0.06), tc - 34, 0.12, fc="white", ec="black", lw=0.6, hatch="////"))
ax.add_patch(Rectangle((tc, 0.06), 50, 0.12, fc="black", ec="black", lw=0.6))
ax.annotate("", xy=(tc - 30, 0.12), xytext=(tc, 0.12),
            arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
ax.text(tc - 15, 0.19, r"$\Delta$", ha="center", va="bottom", fontsize=6.5)
ax.text(70, 0.30, "audited labels", ha="center", va="center", fontsize=6)
ax.text(tc + 56, 0.12, "predicted\nblock", ha="left", va="center", fontsize=6)
ax.text(400, 0.62, r"checkpoints $t_c$ every 50 slots", ha="left", va="center", fontsize=6)
ax.set_xlim(0, 600); ax.set_ylim(0, 1)
ax.set_yticks([]); ax.set_xticks(range(0, 601, 100))
ax.tick_params(labelsize=6.5, length=2)
ax.set_title("(a) Prequential protocol", fontsize=7, loc="left", pad=2)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)

ax2 = fig.add_axes([0.13, 0.13, 0.84, 0.38])
markers = ["o", "s", "^", "D", "v", "P", "X"]
names = fams[1:]
for d, a in enumerate(arr):
    for f, t in a.items():
        off = {2: -0.3, 3: 0.0, 5: 0.3, 4: -0.2, 7: 0.2}.get(f, 0.0)
        ax2.plot(t, d + off, marker=markers[f - 1], ms=3.2, mfc="white", mec="black", mew=0.6, ls="none")
for k in range(1, 4):
    ax2.axvline(k * 150, color="0.5", lw=0.5, ls=":")
ax2.set_yticks(range(6)); ax2.set_yticklabels([f"$d={i+1}$" for i in range(6)], fontsize=6.5)
ax2.set_xlim(-18, 600); ax2.set_ylim(-0.7, 5.7); ax2.invert_yaxis()
ax2.set_xlabel("slot (10 s)", fontsize=6.5, labelpad=1)
ax2.tick_params(labelsize=6.5, length=2)
ax2.set_title("(b) First compromise of each family per domain (world 1)", fontsize=7, loc="left", pad=2)
handles = [plt.Line2D([], [], marker=markers[i], ls="none", mfc="white", mec="black", mew=0.6, ms=4)
           for i in range(len(names))]
fig.legend(handles, names, loc="lower center", ncol=7, fontsize=5.6, frameon=False, handletextpad=0.15,
           columnspacing=0.6, bbox_to_anchor=(0.55, -0.005))
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig2_protocol.png"), dpi=600)
print({fams[f]: p for f, p in phase_of.items()}, arr)
