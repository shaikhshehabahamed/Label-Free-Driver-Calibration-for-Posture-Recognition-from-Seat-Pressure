"""Marker events: the dataset README says the participant and the operator pressed four highlighted points on the mat before each posture.
Saturation maps (cache/sensel_stats.npz) show four fixed, mirror-symmetric sites that carry almost all saturation; this module scores each frame by how
many of the four sites are pressed hard and groups consecutive marker frames into events.

    site_scores(u)        -> (frame_ids, S)   S[t, k] = max pressure (N/cm2) in a (2R+1)^2 window around site k
    marker_events(u, ..)  -> DataFrame of events (start, end, n_frames, peak, n_sites)
"""
import numpy as np
import pandas as pd

from common import CACHE

SITES = {"A": (10, 133), "B": (56, 133), "C": (13, 77), "D": (53, 78)}      # (row, col) from the saturation map
R = 4


def site_scores(u):
    maps = np.load(CACHE / f"User{u}_maps.npy", mmap_mode="r")
    fid = np.load(CACHE / f"User{u}_meta.npz")["frame_id"]
    T = len(fid)
    S = np.zeros((T, len(SITES)), np.float32)
    for k, (r, c) in enumerate(SITES.values()):
        for s in range(0, T, 600):
            x = np.asarray(maps[s:s + 600, max(r - R, 0):r + R + 1, max(c - R, 0):c + R + 1], dtype=np.float32)
            S[s:s + len(x), k] = x.max(axis=(1, 2))
    return fid.astype(int), S


def marker_events(fid, S, thr=2.5, min_sites=3, merge=12, min_len=3):
    on = (S > thr).sum(axis=1) >= min_sites
    idx = np.where(on)[0]
    ev = []
    if len(idx):
        a = b = idx[0]
        for i in idx[1:]:
            if i - b > merge:
                ev.append((a, b))
                a = i
            b = i
        ev.append((a, b))
    rows = []
    for a, b in ev:
        if b - a + 1 < min_len:
            continue
        seg = S[a:b + 1]
        rows.append(dict(start=int(fid[a]), end=int(fid[b]), n_frames=int(b - a + 1), peak=float(seg.max()), n_sites=int(((seg > thr).any(axis=0)).sum())))
    return pd.DataFrame(rows, columns=["start", "end", "n_frames", "peak", "n_sites"])
