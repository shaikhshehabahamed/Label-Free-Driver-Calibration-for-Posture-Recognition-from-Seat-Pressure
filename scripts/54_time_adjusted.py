"""E7 - does seat pressure carry posture information BEYOND the protocol clock?

Every driver performed P1..P20 in the same order, so elapsed time alone predicts the label (the pilot's order-only floor: random forest 29.5 %, above every
uncalibrated pressure model).  The sharper validity question is incremental: with elapsed time available, does pressure still add accuracy, and does the
label-free baseline add more?   Boosted trees (GPU XGBoost, fixed small model, the pilot's non-linear learner) on 96 cells (f = 8), LOSO, 22 drivers:
    time            t only (seconds since the first labelled frame)
    p_none          pressure, no calibration                              p_run60   pressure, label-free 60-s baseline on randomly ordered streams
    p_none+t        pressure + t                                          p_run60+t pressure (random-order baseline) + t
Paired contrasts per driver: (p_none+t) - time and (p_run60+t) - time = incremental value of pressure; (p_run60+t) - (p_none+t) = value of calibration given time.
3 model seeds per condition (the random-order conditions use the matching shuffle seed).   results -> results/e7_time_adjusted.csv (resumable)
"""
import guard  # noqa: F401
import time

import numpy as np
import pandas as pd

import calib_lab as L
import gpu_models as G
from common import RES

SEEDS = (0, 1, 2)                   # 3 model seeds (resumable: a pass with fewer seeds can be topped up later)
OUT = RES / "e7_time_adjusted.csv"

if __name__ == "__main__":
    t0 = time.time()
    drivers = L.load_drivers(8)
    y = np.concatenate([d.y for d in drivers])
    u = np.concatenate([np.full(len(d.rows), d.u) for d in drivers])
    t = np.concatenate([d.t_s for d in drivers])[:, None].astype(np.float32)
    done = set(pd.read_csv(OUT).config) if OUT.exists() else set()
    X0 = np.concatenate([d.Zfull[d.rows] for d in drivers])
    for cond in ("time", "p_none", "p_none+t", "p_run60", "p_run60+t"):
        for s in SEEDS:
            key = f"{cond}|s{s}"
            if key in done:
                continue
            t1 = time.time()
            if cond.startswith("p_run60"):
                Xp = np.concatenate([L.features_multi(d, [("sub", 0.0)], "run", 60, np.random.default_rng([4000 + s, d.u]))[("sub", 0.0)] for d in drivers])
            else:
                Xp = X0
            X = t if cond == "time" else (np.hstack([Xp, t]) if cond.endswith("+t") else Xp)
            r = G.loso_eval("XGB", X, y, u, {"seed": s}, tag=key)
            r.insert(0, "cond", cond)
            r.insert(0, "config", key)
            r["seed"] = s
            L.append_rows(OUT, r)
            print(f"  {key:16s} acc {r.acc.mean():.4f}  [{time.time() - t1:.0f}s, total {time.time() - t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)
