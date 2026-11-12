"""Learning curve over the NUMBER OF TRAINING DRIVERS (is the unseen-driver gap a data-quantity problem?).

For each held-out driver, n drivers are drawn at random from the others (3 repeats; the largest n uses all of them).  GPU LDA, 3 s integration.
Configurations: no calibration vs the label-free 60-s baseline computed on a randomly ordered stream (deployment-faithful, seed 0).
usage: [DATASET=clean] python 33_learning_curve.py
"""
import guard  # noqa: F401
import os
import time

import numpy as np
import pandas as pd

import gpu_models as G
from common import RES, tag
from ladder import Stream

SETTLE = float(os.environ.get("SETTLE", "5.0"))
REPS = 3

st = Stream(f=4)
rows = st.rows(SETTLE)
y, u = st.fr.label.values[rows], st.fr.u.values[rows]
drivers = np.unique(u)
SIZES = tuple(n for n in (3, 6, 9, 12, 15, 18, 21, 29) if n < len(drivers)) + (len(drivers) - 1,)
print("drivers:", len(drivers), "sizes:", SIZES, flush=True)
out = []
t0 = time.time()
for base in ("none", "run60"):
    X = st.features(rows, 3.0, "none") if base == "none" else st.features_labelled_stream(rows, 3.0, 60.0, 0)
    for n in SIZES:
        for rep in range(1 if n == SIZES[-1] else REPS):
            for s in drivers:
                rng = np.random.default_rng(1000 * n + 10 * rep + int(s))
                others = drivers[drivers != s]
                tr_d = others if n >= len(others) else rng.choice(others, n, replace=False)
                tr, te = np.isin(u, tr_d), u == s
                m = G.GpuLDA().prepare(X[tr]).fit_k(y[tr], min(150, int(tr.sum()) - 1), "auto")
                out.append(dict(base=base, n_train=n, rep=rep, u=int(s), acc=float(np.mean(m.predict(X[te]) == y[te]))))
            guard.cooldown(tag=f"learning curve {base} n={n}")
            print(f"{base} n={n} rep={rep} done [{time.time()-t0:.0f}s]", flush=True)
d = pd.DataFrame(out)
d.to_csv(RES / f"gpu_learning_curve{tag()}{'' if SETTLE == 5.0 else f'_s{SETTLE:g}'}.csv", index=False)
g = d.groupby(["base", "n_train", "u"]).acc.mean().reset_index().groupby(["base", "n_train"]).acc.agg(["mean", "std"])
print(g.round(3).to_string())
