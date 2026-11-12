"""CNN under the SAME protocol as the other learners (leave-one-driver-out, 22 folds) with two training seeds per fold.

37_cnn.py evaluated the CNN with grouped 6-fold (3-4 held-out drivers) and one seed.  Here every held-out driver is predicted by a model trained on all other
cohort drivers, and each fold is trained twice (seeds 0, 1; per-driver accuracy = mean of the two).  Three configurations, all declared before the run and all reported:
  CNN|none          input = integrated 3-s map Z                                  (16 x 24, 1 channel)
  CNN|run60         input = Z - b, b = 60-s label-free baseline on a randomly ordered stream (as in the LDA / XGB rows; shuffle seed 0)
  CNN2ch|run60      input = [Z, Z - b] (two channels): the network may use the absolute map as well as the driver-relative one
usage: [DATASET=clean] python 37b_cnn_loso.py        (resumable per configuration)
"""
import guard  # noqa: F401
import time

import numpy as np
import pandas as pd

import gpu_models as G
from common import RES, tag
from ladder import Stream

SEEDS = (0, 1)
SHAPE = (16, 24)
path = RES / f"gpu_cnn_loso{tag()}.csv"
done = set(pd.read_csv(path).config) if path.exists() else set()

st = Stream(f=4)
rows = st.rows(5.0)
y, u = st.fr.label.values[rows], st.fr.u.values[rows]
Z0 = st.features(rows, 3.0, "none")
Zr, Br = st.features_labelled_stream(rows, 3.0, 60.0, 0, parts=True)
D = Zr - Br
INPUTS = {"CNN|none": (Z0, 1), "CNN|run60": (D, 1), "CNN2ch|run60": (np.concatenate([Zr, D], axis=1), 2)}

for cfg, (X, ch) in INPUTS.items():
    if cfg in done:
        print("skip (done):", cfg, flush=True)
        continue
    t0 = time.time()
    out = []
    for s in np.unique(u):
        te = u == s
        accs = []
        for seed in SEEDS:
            pred = G.cnn_predict(X[~te], y[~te], X[te], shape=SHAPE, seed=seed, in_ch=ch)
            accs.append(float(np.mean(pred == y[te])))
        out.append(dict(config=cfg, u=int(s), acc=float(np.mean(accs)), acc_seed0=accs[0], acc_seed1=accs[1], n=int(te.sum())))
        guard.cooldown(tag=f"{cfg} driver {s}")
    d = pd.DataFrame(out)
    d.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"{cfg}: acc {d.acc.mean():.4f} (sd {d.acc.std():.4f}) over {len(d)} drivers [{time.time() - t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)
