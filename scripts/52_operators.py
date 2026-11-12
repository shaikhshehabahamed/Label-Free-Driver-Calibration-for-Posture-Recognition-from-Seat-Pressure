"""E2 - which label-free, driver-relative operator?  Candidate approaches from the normalisation / alignment literature, on identical random-order streams.

  none      Z                                  no calibration
  sub       Z - m                              mean subtraction (sliding cepstral mean normalisation; the pilot's baseline)
  cmvn      (Z - m)/sd                         + per-cell scaling (sliding CMVN, Kaldi apply-cmvn-sliding)
  wh        C_a^{-1/2}(Z - m)                  + full whitening (CORAL, Sun et al. 2016); a = 0.1, 0.5 (CORAL's official `cov + eye` on z-scored data == a = 0.5)
  ea        R_a^{-1/2} Z                       Euclidean alignment as published (He & Wu 2020), uncentred second moment, a = 0.1
Scopes: run (causal window W = 60 / 120 s, labelled segments in random order, 5 shuffles shared by all operators = common random numbers), session
(non-causal, whole recording, occupied frames), oracle (mean of the 20 posture means, needs labels), full (no operator).  Learner: GPU shrinkage LDA
(validated against scikit-learn); 22 drivers, leave-one-driver-out.  Representations: 24 cells (f = 16) and 96 cells (f = 8).
usage: python 52_operators.py [smoke]      results -> results/e2_operators.csv (resumable)
"""
import guard  # noqa: F401
import sys
import time

import numpy as np
import pandas as pd

import calib_lab as L
import gpu_models as G
from common import RES

SEEDS = (2001, 2002, 2003, 2004, 2005)
OPS = [("sub", 0.0), ("cmvn", 0.0), ("wh", 0.1), ("wh", 0.5), ("ea", 0.1)]                 # the pre-specified family (decision rule in 52b_summarize.py)
SENS_OPS = [("cmvn", 0.03), ("cmvn", 0.3), ("wh", 0.25)]                                   # hyperparameter sensitivity (variance floor 0.03 / 0.3 of the mean SD; shrinkage 0.25): NOT part of the decision family
WINDOWS = (60, 120)


def cid(f, scope, op, a, W, seed):
    return f"f{f}|{scope}|{op}|a{a:g}|W{W}|s{seed}"


if __name__ == "__main__":
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    sens = len(sys.argv) > 1 and sys.argv[1] == "sens"                                      # python 52_operators.py sens  -> adds the sensitivity operators to the same file
    out = RES / ("e2_operators_smoke.csv" if smoke else "e2_operators.csv")
    if sens:
        OPS = SENS_OPS
    done = set(pd.read_csv(out).config) if out.exists() else set()
    t0 = time.time()
    for f in (16, 8):
        drivers = L.load_drivers(f)
        if smoke:
            drivers = drivers[:6]
        y = np.concatenate([d.y for d in drivers])
        u = np.concatenate([np.full(len(d.rows), d.u) for d in drivers])
        kw = {"pca": min(150, drivers[0].d - 1), "rank_tol": 1e-6}     # numerical-rank tolerance for ALL operators: per-cell scaling creates near-null directions the LDA would amplify
        print(f"f={f}: {len(drivers)} drivers, {drivers[0].d} cells, rows {len(y)}", flush=True)
        plan = [] if sens else [("full", [("none", 0.0)], 0, -1), ("oracle", [("sub", 0.0)], 0, -1)]
        plan += [("session", OPS, 0, -1)]
        plan += [("run", OPS, W, s) for W in WINDOWS for s in (SEEDS[:1] if smoke else SEEDS)]
        for scope, ops, W, seed in plan:
            todo = [(op, a) for op, a in ops if cid(f, scope, op, a, W, seed) not in done]
            if not todo:
                continue
            t1 = time.time()
            feats = [L.features_multi(dr, todo, scope, W, np.random.default_rng([seed, dr.u]) if scope == "run" else None) for dr in drivers]
            t_feat = time.time() - t1
            for op, a in todo:
                key = cid(f, scope, op, a, W, seed)
                X = np.concatenate([ft[(op, a)] for ft in feats])
                r = G.loso_eval("gLDA", X, y, u, kw, tag=key)
                for k, v in dict(f=f, scope=scope, op=op, alpha=a, W=W, seed=seed).items():
                    r[k] = v
                r.insert(0, "config", key)
                L.append_rows(out, r)
                print(f"  {key:44s} acc {r.acc.mean():.4f}  [features {t_feat:.0f}s | total {time.time() - t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)
