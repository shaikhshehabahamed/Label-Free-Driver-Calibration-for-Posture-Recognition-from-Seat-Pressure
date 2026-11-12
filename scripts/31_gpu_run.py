"""Resumable GPU experiments (leave-one-driver-out).   usage: [DATASET=clean|all|orig] python 31_gpu_run.py EXPERIMENT [SETTLE]

Each configuration is appended to results/gpu_{EXPERIMENT}{tag}{_sS}.csv as soon as it finishes; configurations already present are skipped,
so an interrupted run (laptop crash, thermal pause) can simply be restarted.  tag = '' (primary 22-driver cohort), '_all' or '_orig'.
Config keys: model (gLDA|XGB|CNN), f, cols, Wi, base, op, trim, mode (full | rec | shuf), seed.
  mode full  features from the driver's complete recorded stream (pre-posture, transitions included), real order
  mode rec   labelled segments only, recorded order            (baseline can see the previous posture -> order information)
  mode shuf  labelled segments only, RANDOM order per driver   (deployment-faithful: no information about the protocol order)
Windows: base 'run{W}' uses a W-second causal running mean in every mode.
"""
import guard  # noqa: F401
import sys
import time

import numpy as np
import pandas as pd

import gpu_models as G
from common import DEFAULT_COLS, RES, tag
from ladder import Stream

SEEDS = (0, 1, 2)
REG5 = ((16, 64), (64, 112), (112, 160), (64, 160), (0, 160))


def shuf(base="run60", seeds=SEEDS, **kw):
    return [dict(base=base, mode="shuf", seed=s, **kw) for s in seeds]


EXPS = {
    # calibration ladder: label-free windows in random order (deployment-faithful) and in recorded order, plus the non-causal / labelled references
    "ladder": [dict(base=b) for b in ("none", "session", "oracle", "ref1_5", "ref1_10", "ref10_5", "ref10_10")]
              + [c for W in (15, 30, 60, 120, 300) for c in shuf(f"run{W}")] + [dict(base=f"run{W}", mode="rec") for W in (15, 30, 60, 120, 300)],
    # order control for the 60-s baseline: five random orders, recorded order, and the real stream (with gaps and the pre-posture phase)
    "order": [dict(base="run60", mode="rec"), dict(base="run60", mode="full")] + shuf("run60", range(5)),
    "window": [dict(Wi=w, base="none") for w in (0, 1, 3, 5)] + [c for w in (0, 1, 3, 5) for c in shuf("run60", Wi=w)],
    "op": [dict(op=o, base="oracle") for o in ("sub", "sub_scaled", "logratio")] + [c for o in ("sub", "sub_scaled", "logratio") for c in shuf("run60", op=o)],
    "region": [dict(cols=c, base="none") for c in REG5] + [x for c in REG5 for x in shuf("run60", cols=c)],
    "trim": [dict(trim=t, base="none") for t in (0, 14, 41)] + [x for t in (0, 14, 41) for x in shuf("run60", trim=t)],
    "res": [dict(f=f, base="none") for f in (16, 8, 4, 2)] + [x for f in (16, 8, 4, 2) for x in shuf("run60", f=f)],
    # label-set / cohort sensitivity (run with DATASET=all and DATASET=orig) and settle-time sensitivity (SETTLE=8)
    "sens": [dict(base=b) for b in ("none", "session", "oracle")] + shuf("run60"),
    # best pilot configuration: coarse maps with a longer label-free baseline, against the same-resolution session / oracle bounds
    "combo": [x for f in (16, 8, 4) for x in shuf("run120", f=f)] + shuf("run300", f=16) + [dict(f=16, base=b) for b in ("session", "oracle")],
    # learner families on the same rows (f=8: 96 cells keeps boosted trees affordable)
    "learners": [dict(model=m, f=8, base="none") for m in ("gLDA", "XGB")] + [dict(model=m, f=8, base="oracle") for m in ("gLDA", "XGB")]
                + [dict(model=m, f=8, base="run60", mode="shuf", seed=s) for m in ("gLDA", "XGB") for s in (0, 1)]
                + [dict(model=m, f=8, base="run60", mode="rec") for m in ("gLDA", "XGB")],
}
DEFAULT = dict(model="gLDA", f=4, cols=DEFAULT_COLS, Wi=3.0, base="run60", op="sub", trim=0, mode="full", seed=-1)


def cid(c):
    return f"{c['model']}|f{c['f']}|{c['cols']}|Wi{c['Wi']:g}|{c['base']}|{c['op']}|trim{c['trim']}|{c['mode']}{c['seed']}"


def main(exp, settle):
    path = RES / f"gpu_{exp}{tag()}{'' if settle == 5.0 else f'_s{settle:g}'}.csv"
    done = set(pd.read_csv(path).config) if path.exists() else set()
    streams = {}
    for cfg in EXPS[exp]:
        c = {**DEFAULT, **cfg}
        key = cid(c)
        if key in done:
            print("skip (done):", key, flush=True)
            continue
        sk = (c["f"], tuple(c["cols"]))
        if sk not in streams:
            streams = {sk: Stream(*sk)}
        st = streams[sk]
        t0 = time.time()
        rows = st.rows(settle, c["trim"])
        y, u = st.fr.label.values[rows], st.fr.u.values[rows]
        if c["mode"] == "full":
            X = st.features(rows, c["Wi"], c["base"], c["op"])
        else:
            X = st.features_labelled_stream(rows, c["Wi"], float(c["base"][3:]), None if c["mode"] == "rec" else c["seed"], c["op"])
        width = (c["cols"][1] - c["cols"][0]) // c["f"]
        kw = {"pca": min(150, X.shape[1] - 1)} if c["model"] == "gLDA" else ({"shape": (64 // c["f"], width)} if c["model"] == "CNN" else {})
        d = G.loso_eval(c["model"], X, y, u, kw, tag=key)
        for k, v in c.items():
            d[k] = str(v) if k == "cols" else v
        d.insert(0, "config", key)
        d.insert(0, "exp", exp)
        d.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"[{exp}{tag()}] {key}: acc {d.acc.mean():.4f} (sd {d.acc.std():.4f}) f1 {d.f1.mean():.4f} drivers {len(d)} rows {len(rows)} [{time.time()-t0:.0f}s] GPU {guard.gpu_temp()} C", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0)
