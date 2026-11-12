"""Does a small CNN change the picture?  Grouped 6-fold cross-validation over drivers (3-4 held-out drivers per fold).

Same rows, same folds, three learners: shrinkage-LDA (GPU), boosted trees (XGBoost-GPU), tiny CNN (3 conv layers) on the 16x24
driver-relative maps (integration 3 s, baseline none / 60-s running baseline on a randomly ordered stream).   usage: [DATASET=clean] python 37_cnn.py
"""
import guard  # noqa: F401
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

import gpu_models as G
from common import RES, tag
from ladder import Stream

SETTLE = float(os.environ.get("SETTLE", "5.0"))
st = Stream(f=4)
rows = st.rows(SETTLE)
y, u = st.fr.label.values[rows], st.fr.u.values[rows]
folds = list(GroupKFold(6).split(rows, groups=u))
out = []
path = RES / f"gpu_cnn_grouped6{tag()}{'' if SETTLE == 5.0 else f'_s{SETTLE:g}'}.csv"
for base in ("none", "run60"):
    X = st.features(rows, 3.0, "none") if base == "none" else st.features_labelled_stream(rows, 3.0, 60.0, 0)
    for model in ("gLDA", "XGB", "CNN"):
        t0 = time.time()
        pred = np.empty(len(rows), int)
        for tr, te in folds:
            kw = {"pca": 150} if model == "gLDA" else ({"shape": (16, 24)} if model == "CNN" else {})
            pred[te] = G.predict_fold(model, X[tr], y[tr], X[te], kw)
            guard.cooldown(tag=f"{model} {base}")
        for s in np.unique(u):
            k = u == s
            out.append(dict(model=model, base=base, u=int(s), acc=accuracy_score(y[k], pred[k]), f1=f1_score(y[k], pred[k], average="macro", zero_division=0)))
        d = pd.DataFrame(out)
        d = d[(d.model == model) & (d.base == base)]
        print(f"{model:>5} | {base:>5}: acc {d.acc.mean():.3f} (sd {d.acc.std():.3f}) f1 {d.f1.mean():.3f} [{time.time()-t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)
        pd.DataFrame(out).to_csv(path, index=False)
