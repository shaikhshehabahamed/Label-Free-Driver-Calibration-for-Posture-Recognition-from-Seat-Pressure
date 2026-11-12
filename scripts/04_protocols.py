"""E1: how much does the evaluation protocol change the answer?  random frame split vs blocked split vs LOSO.

16x24 mean-pooled maps of the two informative strips (columns 64-160), every 6th frame of single-labelled segments of the analysis cohort,
20-way posture classification.
"""
import guard  # noqa: F401  (thread caps / low priority)
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RES, model_data, tag

X, meta = model_data(f=4, stride=6, trim=0)
n = len(X)
Xf = X.reshape(n, -1)
y = meta.label.values
u = meta.u.values
print(f"n frames {n} | users {len(np.unique(u))} | classes {len(np.unique(y))}", flush=True)

MODELS = {
    "LDA": lambda: make_pipeline(StandardScaler(), PCA(150, random_state=0), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "LogReg": lambda: make_pipeline(StandardScaler(), PCA(100, random_state=0), LogisticRegression(C=0.5, max_iter=300)),
    "kNN": lambda: make_pipeline(StandardScaler(), PCA(50, random_state=0), KNeighborsClassifier(15, weights="distance")),
    "RF": lambda: RandomForestClassifier(150, max_features="sqrt", min_samples_leaf=2, n_jobs=1, random_state=0),
}


def folds_random(k=5):
    return [(tr, te) for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(Xf, y)]


def folds_blocked(k=5, buffer=40):
    rel = (meta.frame_id - meta.seg_start) / (meta.seg_end - meta.seg_start + 1)
    blk = np.minimum((rel * k).astype(int), k - 1).values
    out = []
    for b in range(k):
        te = blk == b
        keep = np.ones(n, bool)
        for s, d in meta[te].groupby("seg"):
            lo, hi = d.frame_id.min() - buffer, d.frame_id.max() + buffer
            idx = meta.index[(meta.seg == s)]
            fid = meta.loc[idx, "frame_id"].values
            keep[idx[(fid >= lo) & (fid <= hi)]] = False
        keep |= te
        out.append((np.where(keep & ~te)[0], np.where(te)[0]))
    return out


def folds_loso():
    return [(np.where(u != s)[0], np.where(u == s)[0]) for s in np.unique(u)]


def run_fold(name, tr, te):
    m = MODELS[name]()
    m.fit(Xf[tr], y[tr])
    p = m.predict(Xf[te])
    return accuracy_score(y[te], p), balanced_accuracy_score(y[te], p), f1_score(y[te], p, average="macro"), len(te)


rows = []
prot = {"random5": folds_random(), "blocked5": folds_blocked(), "LOSO": folds_loso()}
names = sys.argv[1:] or ["LDA", "kNN", "RF"]          # LogReg is available (python 04_protocols.py LogReg) but adds little and is the slowest
for name in names:
    for pname, fl in prot.items():
        t0 = time.time()
        r = Parallel(n_jobs=2)(delayed(run_fold)(name, tr, te) for tr, te in fl)
        r = np.array(r)
        w = r[:, 3] / r[:, 3].sum()
        rows.append(dict(model=name, protocol=pname, acc=float((r[:, 0] * w).sum()), acc_fold_mean=r[:, 0].mean(),
                         acc_sd=r[:, 0].std(ddof=1), bal_acc=r[:, 1].mean(), macro_f1=r[:, 2].mean(),
                         folds=len(fl), secs=time.time() - t0))
        print(rows[-1], flush=True)
        if pname == "LOSO":
            pd.DataFrame({"u": np.unique(u), "acc": r[:, 0]}).to_csv(RES / f"protocols_loso_by_user_{name}{tag()}.csv", index=False)
res = pd.DataFrame(rows)
res.to_csv(RES / f"protocols_{'_'.join(names)}{tag()}.csv", index=False)
print(res.round(3).to_string(index=False))
