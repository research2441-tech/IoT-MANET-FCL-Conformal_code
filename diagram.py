import os
"""Small grayscale diagram engine with geometric checks.

Boxes are axis-aligned rectangles in inches. Arrows are orthogonal polylines between
explicit points. check() fails if (a) an arrow segment passes through a box other than its
declared source/target, (b) two labels or a label and a box overlap, or (c) box text does
not fit inside its box (measured with the renderer)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6.5})


class Diagram:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h), dpi=600)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.W, self.H = w, h
        self.boxes = {}
        self.frames = []
        self.arrows = []
        self.labels = []
        self.texts = []

    def frame(self, x0, y0, x1, y1, title, title_xy):
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, fc="0.965", ec="0.55", lw=0.6,
                                    ls=(0, (4, 2)), zorder=0))
        t = self.ax.text(*title_xy, title, ha="left", va="center", fontsize=6.6, style="italic", zorder=3)
        self.labels.append(("frame:" + title, t))
        self.frames.append((x0, y0, x1, y1))

    def box(self, name, x0, y0, x1, y1, text, fill="white", bold_first=False, fs=6.4, lw=0.8):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.035",
                                         fc=fill, ec="black", lw=lw, zorder=2))
        lines = text.split("\n")
        if bold_first:
            t1 = self.ax.text((x0 + x1) / 2, y1 - 0.02 - 0.055, lines[0], ha="center", va="center", fontsize=fs,
                              weight="bold", zorder=3)
            self.texts.append((name, t1))
            rest = "\n".join(lines[1:])
            if rest:
                t = self.ax.text((x0 + x1) / 2, (y0 + y1 - 0.11) / 2 - 0.01, rest, ha="center", va="center",
                                 fontsize=fs - 0.2, linespacing=1.15, zorder=3)
                self.texts.append((name, t))
        else:
            t = self.ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
                             linespacing=1.15, zorder=3)
            self.texts.append((name, t))
        self.boxes[name] = (x0, y0, x1, y1)

    def anchor(self, name, side, frac=0.5):
        x0, y0, x1, y1 = self.boxes[name]
        return {"left": (x0, y0 + frac * (y1 - y0)), "right": (x1, y0 + frac * (y1 - y0)),
                "top": (x0 + frac * (x1 - x0), y1), "bottom": (x0 + frac * (x1 - x0), y0)}[side]

    def arrow(self, pts, src=None, dst=None, dashed=False, both=False, lw=0.8):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        for (a, b) in zip(pts[:-2], pts[1:-1]):
            pass
        ls = (0, (3, 1.6)) if dashed else "-"
        self.ax.plot(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1]], color="black", lw=lw, ls=ls, zorder=1,
                     solid_capstyle="butt", dash_capstyle="butt")
        # arrow head on the last segment
        tail = pts[-2]
        if dashed:  # head only, so the dashed shaft stays visible
            (x1, y1), (x0, y0) = pts[-1], pts[-2]
            L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            tail = (x1 - (x1 - x0) / L * 0.06, y1 - (y1 - y0) / L * 0.06)
        self.ax.annotate("", xy=pts[-1], xytext=tail,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color="black", lw=lw,
                                         ls="-", shrinkA=0, shrinkB=0), zorder=4)
        if both:
            self.ax.annotate("", xy=pts[0], xytext=pts[1],
                             arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color="black", lw=lw,
                                             shrinkA=0, shrinkB=0), zorder=4)
        self.arrows.append((pts, src, dst, dashed))

    def label(self, x, y, text, ha="center", va="center", fs=6.0, italic=False):
        t = self.ax.text(x, y, text, ha=ha, va=va, fontsize=fs, style="italic" if italic else "normal", zorder=5,
                         bbox=dict(fc="white", ec="none", pad=0.6) if False else None)
        self.labels.append((text, t))
        return t

    # ------------------------------------------------------------------ checks
    def _bb(self, t, r):
        bb = t.get_window_extent(renderer=r)
        inv = self.ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return x0, y0, x1, y1

    @staticmethod
    def _ov(a, b, pad=0.0):
        return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])

    @staticmethod
    def _seg_hits(p, q, b, eps=0.012):
        x0, y0, x1, y1 = b[0] + eps, b[1] + eps, b[2] - eps, b[3] - eps
        if abs(p[0] - q[0]) < 1e-9:      # vertical
            x = p[0]; ya, yb = sorted([p[1], q[1]])
            return x0 < x < x1 and not (yb <= y0 or ya >= y1)
        y = p[1]; xa, xb = sorted([p[0], q[0]])
        return y0 < y < y1 and not (xb <= x0 or xa >= x1)

    def check(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        problems = []
        for name, t in self.texts:
            bb = self._bb(t, r); b = self.boxes[name]
            if bb[0] < b[0] + 0.01 or bb[2] > b[2] - 0.01 or bb[1] < b[1] + 0.008 or bb[3] > b[3] - 0.008:
                problems.append(f"text overflows box {name}: {bb} vs {b}")
        for pts, src, dst, dashed in self.arrows:
            for p, q in zip(pts[:-1], pts[1:]):
                if abs(p[0] - q[0]) > 1e-9 and abs(p[1] - q[1]) > 1e-9:
                    problems.append(f"non-orthogonal segment {p}->{q}")
                for name, b in self.boxes.items():
                    if name in (src, dst):
                        continue
                    if self._seg_hits(p, q, b):
                        problems.append(f"arrow {src}->{dst} crosses box {name}")
        lbb = [(txt, self._bb(t, r)) for txt, t in self.labels]
        for i, (ta, a) in enumerate(lbb):
            for name, b in self.boxes.items():
                if self._ov(a, b, 0.01):
                    problems.append(f"label '{ta}' overlaps box {name}")
            for tb, b in lbb[i + 1:]:
                if self._ov(a, b, 0.01):
                    problems.append(f"labels overlap: '{ta}' / '{tb}'")
            for pts, src, dst, dashed in self.arrows:
                for p, q in zip(pts[:-1], pts[1:]):
                    seg = (min(p[0], q[0]) - 0.004, min(p[1], q[1]) - 0.004, max(p[0], q[0]) + 0.004, max(p[1], q[1]) + 0.004)
                    if self._ov(a, seg):
                        problems.append(f"label '{ta}' touches arrow {src}->{dst}")
            if a[0] < 0 or a[2] > self.W or a[1] < 0 or a[3] > self.H:
                problems.append(f"label '{ta}' outside canvas")
        names = list(self.boxes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if self._ov(self.boxes[a], self.boxes[b], 0.05):
                    problems.append(f"boxes too close/overlap: {a} {b}")
        return problems

    def save(self, path):
        probs = self.check()
        if probs:
            raise SystemExit("LAYOUT PROBLEMS:\n" + "\n".join(probs))
        self.fig.savefig(path, dpi=600)
        self.fig.savefig(path.replace(".png", ".pdf"))
