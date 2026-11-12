"""Regression test for the GPU pipeline.

1. GpuLDA vs scikit-learn (StandardScaler -> PCA -> LDA lsqr, shrinkage auto / 0.3) on identical folds: prediction agreement + accuracy.
2. Re-run of two earlier CPU results with the new memory-light features: LDA none = 18.6 %, run60 = 26.6 % (rows >= 5 s, 30 drivers).
Light: 5 sklearn folds + 2 x 30 GPU folds.
"""
import time

import guard  # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gpu_models as G
from ladder import Stream

st = Stream(f=4)
rows = st.rows(5.0)
y = st.fr.label.values[rows]
u = st.fr.u.values[rows]
print("rows", len(rows), "| device", G.DEV, flush=True)

for base in ("none", "run60"):
    X = st.features(rows, 3.0, base)
    print(f"\n[{base}] features {X.shape}", flush=True)
    for s in (3, 11, 20, 28, 30):
        te = u == s
        for k, shr in ((150, "auto"), (60, 0.3)):
            t0 = time.time()
            sk = make_pipeline(StandardScaler(), PCA(k, random_state=0), LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shr))
            p_sk = sk.fit(X[~te], y[~te]).predict(X[te])
            t_sk = time.time() - t0
            t0 = time.time()
            p_g = G.GpuLDA().prepare(X[~te]).fit_k(y[~te], k, shr).predict(X[te])
            t_g = time.time() - t0
            print(f"  driver {s:2d} k={k:3d} shr={str(shr):>4}: sklearn acc {np.mean(p_sk == y[te]):.3f} ({t_sk:4.1f}s) | gpu acc {np.mean(p_g == y[te]):.3f} ({t_g:4.1f}s) | agreement {np.mean(p_sk == p_g):.4f}", flush=True)
    d = G.loso_eval("gLDA", X, y, u, tag=base)
    print(f"  gLDA LOSO {base}: acc {d.acc.mean():.4f} (sd {d.acc.std():.4f}) f1 {d.f1.mean():.4f}   [earlier sklearn run: {'0.186' if base == 'none' else '0.266'}]", flush=True)
