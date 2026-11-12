"""Which postures are recovered for an unseen driver?  Pooled leave-one-driver-out confusion for no calibration, the 60-s label-free baseline
(randomly ordered stream, three shuffles pooled) and the oracle baseline.   GPU LDA.   usage: [DATASET=clean] python 42_per_posture.py
Outputs results/per_posture_recall.csv, results/confusion_{none,run60,oracle}.csv (counts, rows = true posture) and figures/F10_per_posture_recall.png
"""
import guard  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpu_models as G
from common import FIG, RES, tag
from ladder import Stream

st = Stream(f=4)
rows = st.rows(5.0)
y, u = st.fr.label.values[rows], st.fr.u.values[rows]
runs = {"none": [None], "run60": [0, 1, 2], "oracle": [None]}
out, accs = {}, {}
for base, seeds in runs.items():
    C = np.zeros((20, 20))
    for seed in seeds:
        X = st.features(rows, 3.0, base) if seed is None else st.features_labelled_stream(rows, 3.0, 60.0, seed)
        d, yy, uu, pp = G.loso_eval("gLDA", X, y, u, {"pca": 150}, tag=f"{base} per-posture", keep_pred=True)
        for a, b in zip(yy, pp):
            C[a - 1, b - 1] += 1
    pd.DataFrame(C, index=range(1, 21), columns=range(1, 21)).to_csv(RES / f"confusion_{base}{tag()}.csv")
    out[base] = np.diag(C) / C.sum(axis=1)
    accs[base] = np.trace(C) / C.sum()
    print(base, "pooled accuracy", round(accs[base], 4), "| postures with recall > 2x chance (10 %):", int((out[base] > 0.10).sum()),
          "| > 4x chance (20 %):", int((out[base] > 0.20).sum()), flush=True)
d = pd.DataFrame({"posture": range(1, 21), "recall_none": out["none"], "recall_run60": out["run60"], "recall_oracle": out["oracle"]})
d.to_csv(RES / f"per_posture_recall{tag()}.csv", index=False)
print(d.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = np.arange(20)
ax.bar(x - 0.27, 100 * d.recall_none, 0.27, color="#7f7f7f", label="no calibration")
ax.bar(x, 100 * d.recall_run60, 0.27, color="#2ca02c", label="60 s running baseline (random segment order)")
ax.bar(x + 0.27, 100 * d.recall_oracle, 0.27, color="#d62728", label="oracle baseline")
ax.axhline(5, color="k", lw=0.8, ls=":", label="chance (5 %)")
ax.set_xticks(x); ax.set_xticklabels([f"P{i}" for i in range(1, 21)], fontsize=7)
ax.set_ylabel("recall for unseen drivers (%)"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / f"F10_per_posture_recall{tag()}.png", dpi=200); plt.close(fig)
