"""Side check (privacy remark only): how identifiable is a driver from ONE integrated seat map in a posture never used for enrolment?

Train on frames of postures 1-10, test on frames of postures 11-20 (and vice versa); driver identification among the cohort drivers (GPU LDA).
Resolutions 16x24, 8x12, 4x6 cells (columns 64-160).   usage: [DATASET=clean] python 38_identifiability.py
"""
import guard  # noqa: F401
import numpy as np
import pandas as pd

import gpu_models as G
from common import RES, tag
from ladder import Stream

res = []
for f in (4, 8, 16):
    st = Stream(f=f)
    rows = st.rows(5.0)
    Z = st.features(rows, 3.0, "none")
    y, u = st.fr.label.values[rows], st.fr.u.values[rows]
    n_dr = len(np.unique(u))
    for name, (a, b) in {"train P1-10 -> test P11-20": (y <= 10, y > 10), "train P11-20 -> test P1-10": (y > 10, y <= 10)}.items():
        m = G.GpuLDA().prepare(Z[a]).fit_k(u[a], min(60, Z.shape[1] - 1), "auto")
        acc = float(np.mean(m.predict(Z[b]) == u[b]))
        res.append(dict(f=f, cells=Z.shape[1], split=name, id_acc=acc, n_drivers=n_dr))
        print(f"f={f:>2} ({Z.shape[1]:>4} cells) {name}: driver-ID accuracy {acc:.3f} (chance {1 / n_dr:.3f})", flush=True)
pd.DataFrame(res).to_csv(RES / f"gpu_identifiability{tag()}.csv", index=False)
