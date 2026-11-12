"""Diagnostic behind decision 5 of the study record: the GPU LDA is fragile on features with (near-)null directions - per-cell scaling (CMVN) amplifies cells that are
(almost) never loaded - so every follow-up experiment uses a numerical-rank tolerance (PCA directions with eigenvalue < 1e-6 x the largest are dropped, the analogue of
scikit-learn LDA's `tol`).  Here: all four operators, 96 cells, 60-s causal window, 22 drivers, 3 shuffles, with and without the tolerance.   -> results/e2_rank_tol_check.csv
"""
import guard  # noqa: F401
import numpy as np
import pandas as pd

import calib_lab as L
import gpu_models as G
from common import RES

drivers = L.load_drivers(8)
y = np.concatenate([d.y for d in drivers])
u = np.concatenate([np.full(len(d.rows), d.u) for d in drivers])
ops = [("sub", 0.0), ("cmvn", 0.0), ("wh", 0.5), ("ea", 0.1)]
rows = []
for seed in (2001, 2002, 2003):
    feats = [L.features_multi(d, ops, "run", 60, np.random.default_rng([seed, d.u])) for d in drivers]
    for op, a in ops:
        X = np.concatenate([f[(op, a)] for f in feats])
        for tol in (None, 1e-6):
            r = G.loso_eval("gLDA", X, y, u, {"pca": 95, "rank_tol": tol})
            rows.append(dict(op=op, alpha=a, seed=seed, rank_tol=("none" if tol is None else tol), acc=r.acc.mean() * 100, n_zero_var_cells=int((X.std(0) == 0).sum())))
            print(rows[-1], flush=True)
d = pd.DataFrame(rows)
d.to_csv(RES / "e2_rank_tol_check.csv", index=False)
print(d.groupby(["op", "rank_tol"]).acc.mean().unstack().round(2))
