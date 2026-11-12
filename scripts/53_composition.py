"""E3 - what does the label-free baseline need: SECONDS of data or DISTINCT POSTURES?

In a running window the two are inseparable (a longer window spans more postures).  Here the baseline is built explicitly from k distinct postures of the
driver's own unlabelled data, n seconds (a random contiguous chunk) from each, k in {1,2,3,5,8,12,19}, n in {2,5,10} s, so that
    total seconds = k * n    and    diversity = k
can be varied independently.  The row's own posture is excluded from the draw ('excl', isolating diversity from self-subtraction); a variant that
samples from all 20 postures ('incl', current posture included with probability k/20) is run at n = 5 s.  Reference rows: no calibration, the causal
running window W = 15..300 s on randomly ordered streams, the whole-session mean and the oracle, all with the same LDA, rows and 5 random seeds.
Also written: the number of distinct postures (and the effective number, exp of the entropy of the posture shares) spanned by a running window.
Learner: GPU shrinkage LDA, 22 drivers, LOSO, 384 cells (f = 4, comparable with the pilot's ladder).
usage: python 53_composition.py [f]      results -> results/e3_composition.csv, results/e3_window_composition.csv (resumable)
"""
import guard  # noqa: F401
import sys
import time

import numpy as np
import pandas as pd

import calib_lab as L
import gpu_models as G
from common import RES

SEEDS = (3001, 3002, 3003, 3004, 3005)                # bag baselines: 5 random draws
SEEDS_REF = tuple(range(3001, 3011))                   # reference ladder (running windows): 10 shuffles - the cohort-mean SD over 3 shuffles was ~0.6 pp (E1)
KS = (1, 2, 3, 5, 8, 12, 19)
NS = (2, 5, 10)
INCL = ((1, 5), (3, 5), (5, 5), (12, 5), (19, 5))
WINDOWS = (15, 30, 60, 120, 300)


def prep_bag(dr):
    posts = np.array([p for p in range(1, 21) if (dr.lab == p).any()])
    NQ = np.zeros(21, int)
    CUM = np.zeros((21, int(max((dr.lab == p).sum() for p in posts)) + 1, dr.d))
    for p in posts:
        X = dr.P[dr.lab == p].astype(np.float64)
        NQ[p] = len(X)
        CUM[p, 1:len(X) + 1] = np.cumsum(X, 0)
        CUM[p, len(X) + 1:] = CUM[p, len(X)]
    pool_excl = np.array([[q for q in posts if q != p] for p in dr.y])
    return posts, CUM, NQ, pool_excl


def bag_baseline(dr, prep, k, n_s, own, rng):
    """b (rows, d): mean over k postures of one random contiguous n_s-second chunk each (equal weights)"""
    posts, CUM, NQ, pool_excl = prep
    pool = pool_excl if own == "excl" else np.tile(posts, (len(dr.rows), 1))
    picks = np.take_along_axis(pool, np.argsort(rng.random(pool.shape), axis=1)[:, :k], axis=1)      # (rows, k) posture ids
    Lc = np.minimum(int(round(n_s * L.FPS)), NQ[picks])
    s = np.floor(rng.random(picks.shape) * (NQ[picks] - Lc + 1)).astype(int)
    cm = (CUM[picks, s + Lc] - CUM[picks, s]) / Lc[..., None]
    return cm.mean(1).astype(np.float32)


def window_composition(dr, rng):
    idx, prow = dr.shuffled_stream(rng)
    oh = np.zeros((len(idx) + 1, 21), np.int32)
    oh[1:] = np.cumsum(np.eye(21, dtype=np.int32)[dr.lab[idx]], axis=0)
    out = {}
    for W in WINDOWS:
        w = int(round(W * L.FPS))
        cnt = oh[prow + 1] - oh[np.maximum(prow + 1 - w, 0)]
        p = cnt[:, 1:] / np.maximum(cnt[:, 1:].sum(1, keepdims=True), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.exp(-np.nansum(np.where(p > 0, p * np.log(p), 0), axis=1))
        out[W] = (float((cnt[:, 1:] > 0).sum(1).mean()), float(eff.mean()))
    return out


def cid(c):
    return "|".join(f"{k}={c[k]}" for k in ("kind", "k", "n", "own", "W", "seed"))


if __name__ == "__main__":
    f = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tagf = "" if f == 4 else f"_f{f}"
    out = RES / f"e3_composition{tagf}.csv"
    done = set(pd.read_csv(out).config) if out.exists() else set()
    t0 = time.time()
    drivers = L.load_drivers(f)
    y = np.concatenate([d.y for d in drivers])
    u = np.concatenate([np.full(len(d.rows), d.u) for d in drivers])
    kw = {"pca": min(150, drivers[0].d - 1), "rank_tol": 1e-6}
    preps = {d.u: prep_bag(d) for d in drivers}
    print(f"f={f}: {len(drivers)} drivers, {drivers[0].d} cells, rows {len(y)}", flush=True)

    # side table: posture diversity of a running window
    wc = RES / "e3_window_composition.csv"
    if not wc.exists():
        rows = []
        for s in SEEDS[:3]:
            for d in drivers:
                for W, (nd, eff) in window_composition(d, np.random.default_rng([s, d.u])).items():
                    rows.append(dict(seed=s, u=d.u, W=W, distinct=nd, effective=eff))
        pd.DataFrame(rows).groupby("W")[["distinct", "effective"]].mean().to_csv(wc)
        print(pd.read_csv(wc).round(2).to_string(index=False), flush=True)

    def run(c, make):
        key = cid(c)
        if key in done:
            return
        t1 = time.time()
        X = np.concatenate([make(d) for d in drivers])
        r = G.loso_eval("gLDA", X, y, u, kw, tag=key)
        for k, v in c.items():
            r[k] = v
        r.insert(0, "config", key)
        L.append_rows(out, r)
        print(f"  {key:48s} acc {r.acc.mean():.4f}  [{time.time() - t1:.0f}s, total {time.time() - t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)

    base = dict(k=0, n=0, own="", W=0, seed=-1)
    run({**base, "kind": "none"}, lambda d: d.Zfull[d.rows])
    run({**base, "kind": "session"}, lambda d: L.features_multi(d, [("sub", 0.0)], "session")[("sub", 0.0)])
    run({**base, "kind": "oracle"}, lambda d: L.features_multi(d, [("sub", 0.0)], "oracle")[("sub", 0.0)])
    for W in WINDOWS:
        for s in SEEDS_REF:
            run({**base, "kind": "run", "W": W, "seed": s},
                lambda d: L.features_multi(d, [("sub", 0.0)], "run", W, np.random.default_rng([s, d.u]))[("sub", 0.0)])
    grid = [(k, n, "excl") for n in NS for k in KS] + [(k, n, "incl") for k, n in INCL]
    for k, n, own in grid:
        for s in SEEDS:
            run({**base, "kind": "bag", "k": k, "n": n, "own": own, "seed": s},
                lambda d: d.Zfull[d.rows] - bag_baseline(d, preps[d.u], k, n, own, np.random.default_rng([s, d.u, k, n])))
