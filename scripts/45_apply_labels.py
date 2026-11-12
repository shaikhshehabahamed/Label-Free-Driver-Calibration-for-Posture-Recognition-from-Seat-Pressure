"""Rebuild the label-dependent parts of the cache from results/labels_repaired.csv (run after 43_repair_labels.py).

  cache/frames.csv            label, n_cover, seg_start, seg_end, dist_edge, phase recomputed (other columns untouched)
  cache/frames_orig_labels.csv  backup of the table built from the first cleaning (labels_clean.csv); kept for the label-set sensitivity analysis
  cache/segmean_fullres.npy   mean full-resolution map per (driver, posture) over single-labelled frames
The pooled arrays, forces and every sensor-derived column are unchanged (frame order is identical).  Marker presses are masked at load time (common.load_frames).
"""
import shutil

import numpy as np
import pandas as pd

from common import CACHE, load_labels, marker_mask

AREA = 1.27 ** 2

bak = CACHE / "frames_orig_labels.csv"
if not bak.exists():
    shutil.copy(CACHE / "frames.csv", bak)
    print("backed up the original-label frame table")
fr = pd.read_csv(bak)                      # always rebuild from the backup so the script is idempotent
labels = load_labels("repaired")
MK = marker_mask(fr)                       # frames inside marker presses: never enter the per-posture mean maps (the frame table keeps the plain labels)

lab_col = np.zeros(len(fr), np.int16)
cov_col = np.zeros(len(fr), np.int16)
s0_col = np.full(len(fr), -1, np.int64)
s1_col = np.full(len(fr), -1, np.int64)
edge_col = np.full(len(fr), -1, np.int64)
phase_col = np.empty(len(fr), object)
seg = np.zeros((30, 21, 64, 160), np.float32)

for u in range(1, 31):
    g = fr[fr.u == u]
    ix = g.index.values
    fid = g.frame_id.values
    T = len(fid)
    lab = labels[labels.u == u]
    cover = np.zeros(T, np.int16)
    lbl = np.zeros(T, np.int16)
    s0 = np.full(T, -1, np.int64)
    s1 = np.full(T, -1, np.int64)
    for r in lab.itertuples():
        m = (fid >= r.start) & (fid <= r.end)
        cover[m] += 1
        lbl[m] = r.p
        s0[m] = r.start
        s1[m] = r.end
    lbl[cover > 1] = -1
    dist_edge = np.where(cover == 1, np.minimum(fid - s0, s1 - fid), -1)
    first, last = lab.start.min(), lab.end.max()
    phase = np.where(fid < first, "pre", np.where(fid > last, "post", np.where(cover == 0, "gap", "seg")))
    lab_col[ix], cov_col[ix], s0_col[ix], s1_col[ix], edge_col[ix], phase_col[ix] = lbl, cover, s0, s1, dist_edge, phase
    # segment mean maps (streamed); marker-press frames excluded
    lbl_eff = np.where(MK[ix] & (lbl > 0), -2, lbl)
    maps = np.load(CACHE / f"User{u}_maps.npy", mmap_mode="r")
    sums = np.zeros((21, 64, 160), np.float64)
    cnt = np.zeros(21, np.int64)
    for s in range(0, T, 400):
        l = lbl_eff[s:s + 400]
        if (l > 0).any():
            x = np.asarray(maps[s:s + len(l)])
            for p in np.unique(l[l > 0]):
                sums[p] += x[l == p].sum(axis=0, dtype=np.float64)
                cnt[p] += int((l == p).sum())
    for p in range(1, 21):
        if cnt[p] > 0:
            seg[u - 1, p] = sums[p] / cnt[p]
    print(f"user {u:2d}: single-labelled {int((lbl > 0).sum()):5d}  overlap {int((lbl == -1).sum()):4d}  postures with data {len(np.unique(lbl[lbl > 0]))}", flush=True)

fr["label"], fr["n_cover"], fr["seg_start"], fr["seg_end"], fr["dist_edge"], fr["phase"] = lab_col, cov_col, s0_col, s1_col, edge_col, phase_col
fr.to_csv(CACHE / "frames.csv", index=False)
np.save(CACHE / "segmean_fullres.npy", seg)
old = pd.read_csv(bak)
print("\nsingle-labelled frames: %d (first cleaning) -> %d (repaired)" % ((old.label > 0).sum(), (fr.label > 0).sum()))
print("overlap-excluded frames: %d -> %d" % ((old.label == -1).sum(), (fr.label == -1).sum()))
print("segments with data: %d -> %d" % (old[old.label > 0].groupby(["u", "label"]).ngroups, fr[fr.label > 0].groupby(["u", "label"]).ngroups))
