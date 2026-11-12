"""Nested leave-one-driver-out hyper-parameter tuning (no information from the held-out driver is used for any choice).

outer: LOSO over the analysis cohort (22 drivers).   inner: GroupKFold(5) over the training drivers (groups = drivers).
stage 1  pipeline  : integration window Wi in {1,3,5} s  x  baseline window Wb in {30,60,120} s (label-free, computed on a RANDOMLY ORDERED
                     stream so that the choice cannot exploit the fixed posture order); the control tunes Wi only
stage 2  classifier: PCA components in {50,100,200} x LDA shrinkage in {auto, 0.3}
GPU LDA: the eigendecomposition is computed once per inner fold and shared by all six (k, shrinkage) settings.
usage: python 34_nested_tuning.py run|none
"""
import guard  # noqa: F401
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import gpu_models as G
from common import RES, tag
from ladder import Stream

mode = sys.argv[1]
SETTLE = float(os.environ.get("SETTLE", "5.0"))
WI = (1.0, 3.0, 5.0)
WB = (30, 60, 120) if mode == "run" else (None,)
KS = (50, 100, 200)
SHR = ("auto", 0.3)

st = Stream(f=4)
rows = st.rows(SETTLE)
y, u = st.fr.label.values[rows], st.fr.u.values[rows]
FS = {}
for wi in WI:
    for wb in WB:
        FS[(wi, wb)] = st.features(rows, wi, "none") if wb is None else st.features_labelled_stream(rows, wi, float(wb), 0)
        print("features", wi, wb, FS[(wi, wb)].shape, flush=True)

out = []
t0 = time.time()
for s in np.unique(u):
    tr = np.where(u != s)[0]
    te = np.where(u == s)[0]
    folds = list(GroupKFold(5).split(tr, groups=u[tr]))
    s1 = {}
    for key, X in FS.items():
        accs = []
        for a, b in folds:
            m = G.GpuLDA().prepare(X[tr[a]]).fit_k(y[tr[a]], 100, "auto")
            accs.append(np.mean(m.predict(X[tr[b]]) == y[tr[b]]))
        s1[key] = float(np.mean(accs))
    kbest = max(s1, key=s1.get)
    X = FS[kbest]
    s2 = {(k, h): [] for k in KS for h in SHR}
    for a, b in folds:
        P = G.GpuLDA().prepare(X[tr[a]])
        for k in KS:
            for h in SHR:
                P.fit_k(y[tr[a]], k, h)
                s2[(k, h)].append(np.mean(P.predict(X[tr[b]]) == y[tr[b]]))
    best = max(s2, key=lambda kk: np.mean(s2[kk]))
    m = G.GpuLDA().prepare(X[tr]).fit_k(y[tr], best[0], best[1])
    pred = m.predict(X[te])
    out.append(dict(mode=mode, u=int(s), acc=float(np.mean(pred == y[te])), f1=f1_score(y[te], pred, average="macro", zero_division=0),
                    Wi=kbest[0], Wb=kbest[1], pca=best[0], shrinkage=str(best[1]), inner_acc=float(np.mean(s2[best])), stage1_best=s1[kbest]))
    guard.cooldown(tag=f"nested {mode} driver {s}")
    print(f"driver {s:2d}: acc {out[-1]['acc']:.3f}  chose Wi={kbest[0]} Wb={kbest[1]} k={best[0]} shr={best[1]}  [{time.time()-t0:.0f}s]", flush=True)
d = pd.DataFrame(out)
d.to_csv(RES / f"gpu_nested_{mode}{tag()}{'' if SETTLE == 5.0 else f'_s{SETTLE:g}'}.csv", index=False)
print(f"\nnested {mode}: acc {d.acc.mean():.4f} (sd {d.acc.std():.4f}) f1 {d.f1.mean():.4f}")
print("chosen Wi:", d.Wi.value_counts().to_dict(), "| Wb:", d.Wb.value_counts().to_dict(), "| pca:", d.pca.value_counts().to_dict(), "| shrinkage:", d.shrinkage.value_counts().to_dict())
