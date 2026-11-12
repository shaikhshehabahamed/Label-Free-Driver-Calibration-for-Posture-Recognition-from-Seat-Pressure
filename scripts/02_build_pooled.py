"""Build the analysis cache from the parsed full-resolution maps.

Outputs (analysis/cache):
  frames.csv               one row per recorded frame: user, frame_id, time_s, label (0 none, -1 overlap, 1..20),
                           n_cover, seg_start, seg_end, dist_edge (frames to nearest segment edge), phase, total force, saturation counts
  pooled_f{2,4,8,16}.npy   non-overlapping mean-pooled maps (N, 64/f, 160/f); f=2 stored float16
  fullres_s5.npy           float16 full-resolution maps of single-labelled frames with frame_id % 5 == 0
  fullres_s5_rows.npy      row index into frames.csv for each fullres_s5 map
  segmean_fullres.npy      (30, 21, 64, 160) mean map per (user, posture) over single-labelled frames
  sensel_stats.npz         per-sensel active / saturated frame counts over all frames
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "analysis" / "cache"
RES = ROOT / "analysis" / "results"
FACTORS = (2, 4, 8, 16)
AREA = 1.27 ** 2          # cm^2 per sensel
CAP = 3.41                # sensor saturation (N/cm^2): max value seen in every file
SAT_THR = CAP - 0.005
CHUNK = 400

labels = pd.read_csv(RES / "labels_clean.csv")


def pool_mean(x, f):
    t = x.shape[0]
    return x.reshape(t, 64 // f, f, 160 // f, f).mean(axis=(2, 4), dtype=np.float32)


def process(u):
    maps = np.load(CACHE / f"User{u}_maps.npy", mmap_mode="r")
    meta = np.load(CACHE / f"User{u}_meta.npz")
    fid = meta["frame_id"]
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

    pooled = {f: np.empty((T, 64 // f, 160 // f), np.float16 if f == 2 else np.float32) for f in FACTORS}
    force = np.empty(T, np.float32)
    n_act = np.empty(T, np.int32)
    n_sat = np.empty(T, np.int32)
    act_cnt = np.zeros((64, 160), np.int64)
    sat_cnt = np.zeros((64, 160), np.int64)
    segsum = np.zeros((21, 64, 160), np.float64)
    segn = np.zeros(21, np.int64)
    keep_rows, keep_maps = [], []
    for s in range(0, T, CHUNK):
        x = np.asarray(maps[s:s + CHUNK])
        e = s + len(x)
        for f in FACTORS:
            pooled[f][s:e] = pool_mean(x, f)
        force[s:e] = x.sum(axis=(1, 2)) * AREA
        a = x > 0
        sat = x >= SAT_THR
        n_act[s:e] = a.sum(axis=(1, 2))
        n_sat[s:e] = sat.sum(axis=(1, 2))
        act_cnt += a.sum(axis=0)
        sat_cnt += sat.sum(axis=0)
        l = lbl[s:e]
        for p in np.unique(l[l > 0]):
            m = l == p
            segsum[p] += x[m].sum(axis=0, dtype=np.float64)
            segn[p] += int(m.sum())
        m5 = (l > 0) & (fid[s:e] % 5 == 0)
        if m5.any():
            keep_rows.append(np.arange(s, e)[m5])
            keep_maps.append(x[m5].astype(np.float16))
    rows = np.concatenate(keep_rows) if keep_rows else np.empty(0, int)
    fr = pd.DataFrame(dict(u=u, row_in_user=np.arange(T), frame_id=fid, time_s=meta["time_s"], label=lbl,
                           n_cover=cover, seg_start=s0, seg_end=s1, dist_edge=dist_edge, phase=phase,
                           force_N=force, n_active=n_act, n_sat=n_sat,
                           avg_p=meta["avg_p"], peak_p=meta["peak_p"], contact_cm2=meta["contact_cm2"]))
    return dict(u=u, fr=fr, pooled=pooled, act=act_cnt, sat=sat_cnt, segsum=segsum, segn=segn,
                rows=rows, maps5=np.concatenate(keep_maps) if keep_maps else np.empty((0, 64, 160), np.float16))


if __name__ == "__main__":
    users = list(range(1, 31))
    with Pool(3) as pool:
        out = pool.map(process, users)
    out.sort(key=lambda d: d["u"])
    frames = pd.concat([d["fr"] for d in out], ignore_index=True)
    offs = np.cumsum([0] + [len(d["fr"]) for d in out])
    frames.to_csv(CACHE / "frames.csv", index=False)
    for f in FACTORS:
        np.save(CACHE / f"pooled_f{f}.npy", np.concatenate([d["pooled"][f] for d in out], axis=0))
    np.save(CACHE / "fullres_s5.npy", np.concatenate([d["maps5"] for d in out], axis=0))
    np.save(CACHE / "fullres_s5_rows.npy", np.concatenate([d["rows"] + o for d, o in zip(out, offs[:-1])]))
    seg = np.zeros((30, 21, 64, 160), np.float32)
    for d in out:
        for p in range(1, 21):
            if d["segn"][p] > 0:
                seg[d["u"] - 1, p] = d["segsum"][p] / d["segn"][p]
    np.save(CACHE / "segmean_fullres.npy", seg)
    np.savez(CACHE / "sensel_stats.npz", act=sum(d["act"] for d in out), sat=sum(d["sat"] for d in out),
             n_frames=len(frames))
    print("frames:", len(frames), "| labelled single:", int((frames.label > 0).sum()),
          "| overlap:", int((frames.label == -1).sum()), "| unlabelled:", int((frames.label == 0).sum()))
    print(frames.groupby("phase").size().to_dict())
