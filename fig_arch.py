import os
"""Fig. 1: overall framework. Solid arrows = forward data flow; dashed = reverse-mode gradient."""
from diagram import Diagram

D = Diagram(7.16, 3.95)
# ---------------------------------------------------------------- local plane
D.frame(0.05, 2.02, 7.11, 3.90, "MANET domain gateway $d$ (raw records stay local)", (0.12, 3.77))
y0, y1 = 2.62, 3.30
xs = [0.15 + i * 0.995 for i in range(7)]
w = 0.86
txt = ["IoT-MANET stream\nrecords, mobility,\nwatchdog evidence",
       "Streaming graph\n$\\mathcal{G}_t$ with node and\nedge states",
       "Edge-gated\nmessage passing\nencoder $f_{\\theta}$",
       "Recurrent node\nmemory (selective\nSSM or GRU)",
       "Dual head:\nsoftmax and\nnovelty distance",
       "Mondrian conformal\n$p$-values with ACI;\nset, abstain, zero-day",
       "Venn-Abers\nprobability $\\pi_v^t$;\nrisk-priced routing"]
names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
fills = ["white", "white", "0.88", "0.88", "0.88", "0.80", "0.80"]
for n, x, t, f in zip(names, xs, txt, fills):
    D.box(n, x, y0, x + w, y1, t, fill=f)
for a, b in zip(names[:-1], names[1:]):
    p = D.anchor(a, "right"); q = D.anchor(b, "left")
    D.arrow([p, q], a, b)
# self-supervised pretraining feeding the encoder
D.box("SSL", 2.02, 3.43, 3.12, 3.70, "Masked feature and\ntopology pretraining", fill="white")
D.arrow([D.anchor("SSL", "bottom"), (2.57, y1)], "SSL", "B3")
# supervised loss on audited labels and reverse-mode gradient
D.box("LOSS", 4.13, 2.12, 4.99, 2.45, "Audited labels,\ncross-entropy $\\ell$", fill="white")
D.arrow([D.anchor("B5", "bottom"), D.anchor("LOSS", "top")], "B5", "LOSS")
yg = 2.285
D.arrow([(4.13, yg), (3.565, yg), (3.565, y0)], "LOSS", "B4", dashed=True)
D.arrow([(3.565, yg), (2.57, yg), (2.57, y0)], "LOSS", "B3", dashed=True)
D.label(3.25, 2.155, "reverse-mode gradient $\\nabla_{\\theta}\\ell$", fs=5.8, italic=True)
# ---------------------------------------------------------------- federation and trust plane
D.frame(0.05, 0.05, 7.11, 1.86, "Federation and trust plane", (4.13, 0.40))
D.box("LED", 0.15, 0.86, 1.55, 1.50, "Verifiable trust ledger\nhash chain, Merkle roots,\nML-DSA-44 signatures", fill="0.92")
D.box("KEM", 0.15, 0.16, 1.55, 0.62, "ML-KEM-768 key\nestablishment, AES-GCM", fill="0.92")
D.box("AGG", 1.90, 0.86, 3.70, 1.50,
      "Split aggregation across domains\nshared: encoder, SSM, attack rows\nlocal: benign row, scaler, calibration",
      fill="0.88")
D.box("REP", 4.13, 0.86, 4.99, 1.50, "Reservoir replay\nclass-balanced\nmemory", fill="white")
D.box("TRG", 5.125, 0.86, 5.985, 1.50, "Drift trigger:\nnew family or\nbenign martingale", fill="white")
D.box("OUT", 6.12, 0.86, 6.98, 1.50, "Routes avoid\nalerted relays;\nprice uncertain ones", fill="white")
# arrows between planes
D.arrow([D.anchor("B6", "bottom"), D.anchor("TRG", "top", (5.555 - 5.125) / 0.86)], "B6", "TRG")
D.label(5.60, 1.68, "audited $p$-values", ha="left", fs=5.8)
D.arrow([D.anchor("TRG", "left"), D.anchor("REP", "right")], "TRG", "REP")
D.arrow([D.anchor("REP", "top"), D.anchor("LOSS", "bottom")], "REP", "LOSS")
D.label(4.61, 1.68, "replay batches", ha="left", fs=5.8)
D.arrow([D.anchor("B7", "bottom"), D.anchor("OUT", "top")], "B7", "OUT")
D.arrow([(2.25, y0), (2.25, 1.50)], "B3", "AGG")
D.arrow([(2.45, 1.50), (2.45, y0)], "AGG", "B3")
D.label(2.19, 1.68, "signed update $\\Delta\\theta_d$", ha="right", fs=5.8)
D.label(2.51, 1.68, "shared $\\theta$", ha="left", fs=5.8)
D.arrow([D.anchor("AGG", "left"), D.anchor("LED", "right")], "AGG", "LED")
D.label(1.725, 1.30, "digest", fs=5.6)
D.arrow([D.anchor("KEM", "right"), (2.80, 0.39), (2.80, 0.86)], "KEM", "AGG")
D.label(2.17, 0.30, "session key", fs=5.6)
D.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig1_framework.png"))
print("ok")
