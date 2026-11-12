"""E4 - the deep-network comparator: AdaBN (Li et al. 2018) against the driver-relative input, with 3 training seeds, leave-one-driver-out, 22 drivers.

Per fold and seed, two networks are trained (3-layer CNN, 384 cells = 16 x 24 maps, the pilot's architecture and recipe):
   input Z            (no calibration)            evaluated as  none      (standard BN)    and  none+AdaBN  (BN statistics re-estimated on the held-out driver's own rows)
   input Z - b_60 s   (label-free baseline, random-order stream, shuffle seed 4000 + training seed)   evaluated as  run60  and  run60+AdaBN
AdaBN is a non-causal, per-driver (whole-session) adaptation of the BN statistics - an upper-effort deep-learning analogue of mean/variance alignment.
usage: python 56_cnn_adabn.py      results -> results/e4_cnn_adabn.csv (resumable per (driver, seed))
"""
import guard  # noqa: F401
import time

import numpy as np
import pandas as pd

import calib_lab as L
import gpu_models as G
from common import RES

SEEDS = (0, 1, 2)                   # 3 training seeds (each training costs ~13 s plus thermal pauses on the GTX 1650)
OUT = RES / "e4_cnn_adabn.csv"
SHAPE = (16, 24)

if __name__ == "__main__":
    t0 = time.time()
    drivers = L.load_drivers(4)
    y = np.concatenate([d.y for d in drivers])
    u = np.concatenate([np.full(len(d.rows), d.u) for d in drivers])
    X0 = np.concatenate([d.Zfull[d.rows] for d in drivers])
    XR = {s: np.concatenate([L.features_multi(d, [("sub", 0.0)], "run", 60, np.random.default_rng([4000 + s, d.u]))[("sub", 0.0)] for d in drivers]) for s in SEEDS}
    done = set()
    if OUT.exists():
        o = pd.read_csv(OUT)
        done = set(zip(o.u, o.seed))
    for s in SEEDS:
        for dr in drivers:
            if (dr.u, s) in done:
                continue
            te = u == dr.u
            rec = dict(u=dr.u, seed=s, n=int(te.sum()))
            p_std, p_ada = G.cnn_predict_adabn(X0[~te], y[~te], X0[te], shape=SHAPE, seed=s)
            rec["none"], rec["none_adabn"] = float(np.mean(p_std == y[te])), float(np.mean(p_ada == y[te]))
            p_std, p_ada = G.cnn_predict_adabn(XR[s][~te], y[~te], XR[s][te], shape=SHAPE, seed=s)
            rec["run60"], rec["run60_adabn"] = float(np.mean(p_std == y[te])), float(np.mean(p_ada == y[te]))
            pd.DataFrame([rec]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            guard.cooldown(tag=f"driver {dr.u} seed {s}")
        o = pd.read_csv(OUT)
        m = o[o.seed == s][["none", "none_adabn", "run60", "run60_adabn"]].mean()
        print(f"seed {s}: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()) + f"   [{time.time() - t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)
