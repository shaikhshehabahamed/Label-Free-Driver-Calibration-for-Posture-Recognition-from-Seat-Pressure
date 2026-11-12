"""Interpretable physical descriptors vs pooled pixels (ablation).

Descriptors per strip (backrest / seat pan) on the 16x24 pooled map of columns 64-160:
  load F (N), contact area A (cm2), centre of pressure (row, column), spread (row, column), correlation, upper/lower-half load fraction  -> 8 x 2 = 16,
  plus the backrest share of the total load (1) and the global centre of pressure (2) = 19 numbers.
The driver-relative version is desc(Z_t) - desc(B_t)  (descriptor of the integrated map minus descriptor of the baseline map).
Centre-of-pressure / contact-area / pressure-ratio features are the ones used for driver posture by Zhao et al. (2021) and for seated postures
by Mutlu et al. (2007) and Zemp et al. (2016).      usage: [DATASET=clean] python 36_descriptors.py
Pixel results (same rows) are in the ladder / learners experiments and are not recomputed here.
"""
import guard  # noqa: F401
import numpy as np
import pandas as pd

import gpu_models as G
from common import RES, tag
from ladder import Stream

AREA = 16 * 1.27 ** 2                       # cm2 per pooled cell
H, W = 16, 24
REG = {"back": (0, 12), "seat": (12, 24)}   # pooled columns = mat columns 64-112 and 112-160


def descriptors(M):
    """M (n, 16, 24) -> (n, 19)"""
    rr = np.arange(H)[None, :, None]
    cc = np.arange(W)[None, None, :]
    out, loads = [], []
    for name, (c0, c1) in REG.items():
        S = M[:, :, c0:c1]
        w = S.sum(axis=(1, 2)) + 1e-9
        F = w * AREA
        A = (S > 0.13).sum(axis=(1, 2)) * AREA
        cr = (S * rr).sum(axis=(1, 2)) / w
        cl = (S * (cc[:, :, c0:c1] - c0)).sum(axis=(1, 2)) / w
        sr = np.sqrt((S * (rr - cr[:, None, None]) ** 2).sum(axis=(1, 2)) / w)
        sl = np.sqrt((S * ((cc[:, :, c0:c1] - c0) - cl[:, None, None]) ** 2).sum(axis=(1, 2)) / w)
        rho = (S * (rr - cr[:, None, None]) * ((cc[:, :, c0:c1] - c0) - cl[:, None, None])).sum(axis=(1, 2)) / w / (sr * sl + 1e-6)
        lr = S[:, :H // 2].sum(axis=(1, 2)) / w
        out += [F, A, cr, cl, sr, sl, rho, lr]
        loads.append(F)
    tot = sum(loads) + 1e-9
    out += [loads[0] / tot]
    allw = M.sum(axis=(1, 2)) + 1e-9
    out += [(M * rr).sum(axis=(1, 2)) / allw, (M * cc).sum(axis=(1, 2)) / allw]
    return np.stack(out, axis=1).astype(np.float32)


st = Stream(f=4)
rows = st.rows(5.0)
y, u = st.fr.label.values[rows], st.fr.u.values[rows]
out = []
for base in ("none", "run60", "oracle"):
    if base == "run60":
        Z, B = st.features_labelled_stream(rows, 3.0, 60.0, 0, parts=True)          # random segment order (deployment-faithful)
    else:
        Z, B = st.features(rows, 3.0, base, parts=True)
    dz = descriptors(Z.reshape(-1, H, W))
    dd = dz if base == "none" else dz - descriptors(B.reshape(-1, H, W))
    for model in ("gLDA", "XGB"):
        kw = {"pca": min(150, dd.shape[1] - 1)} if model == "gLDA" else {}
        d = G.loso_eval(model, dd, y, u, kw, tag=f"descriptors|{base}")
        d.insert(0, "config", f"descriptors|{base}")
        d.insert(0, "model", model)
        out.append(d)
        print(f"{model:>5} descriptors | {base:>6}: acc {d.acc.mean():.3f} (sd {d.acc.std():.3f}) f1 {d.f1.mean():.3f}  features {dd.shape[1]}", flush=True)
        pd.concat(out).to_csv(RES / f"gpu_descriptors{tag()}.csv", index=False)
