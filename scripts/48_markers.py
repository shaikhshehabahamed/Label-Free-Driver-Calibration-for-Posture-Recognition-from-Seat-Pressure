"""Marker events and what they say about the label table (no information from the dataset authors needed).

The dataset README states that the participant and the operator pressed four highlighted points on the mat before assuming each posture.  Those presses are
visible as simultaneous, near-saturating pressure at four fixed, mirror-symmetric sites (marker_lib.SITES).  This script

  1. detects the marker events of all 30 drivers                                  -> results/marker_events.csv
  2. calibrates the label/marker relation on the 22 drivers whose label tables are internally consistent
        start_k - end of the preceding marker   (OFF_S),   start of the following marker - end_k   (OFF_E)
     and measures how well a marker-anchored boundary reproduces the published one                 -> results/marker_validation.csv, marker_calibration.json
  3. audits every label range against the markers (interior marker = a boundary inside the range; late start = start long after its marker; unmatched)
                                                                                                    -> results/marker_label_flags.csv
usage: python 48_markers.py
"""
import json

import numpy as np
import pandas as pd

import marker_lib as ml
from common import RES, load_frames

THR, MIN_SITES = 1.0, 3
lab = pd.read_csv(RES / "labels_clean.csv")                      # published ranges (as parsed)
_fr0 = load_frames("frames_orig_labels.csv")
_last = _fr0.groupby("u").frame_id.max()


def consistent(u):
    """label table internally consistent (complete, strictly increasing in protocol order) and inside the recording"""
    L = lab[lab.u == u].sort_values("p")
    return len(L) == 20 and bool((L.start.values[1:] > L.end.values[:-1]).all()) and int(L.end.max()) <= int(_last[u])


clean = [u for u in range(1, 31) if consistent(u)]
print("drivers with a consistent label table:", len(clean))

# ---- 1. events -----------------------------------------------------------------------------------------------------------------------------------------
EV = {}
rows = []
for u in range(1, 31):
    fid, S = ml.site_scores(u)
    ev = ml.marker_events(fid, S, thr=THR, min_sites=MIN_SITES)
    EV[u] = ev
    ev = ev.assign(u=u)
    rows.append(ev)
    print(f"driver {u:2d}: {len(ev):2d} marker events", flush=True)
pd.concat(rows)[["u", "start", "end", "n_frames", "peak", "n_sites"]].to_csv(RES / "marker_events.csv", index=False)

# ---- 2. calibration and validation on the clean drivers ---------------------------------------------------------------------------------
val = []
for u in clean:
    ev = EV[u]
    L = lab[lab.u == u].sort_values("p").reset_index(drop=True)
    for k in range(len(L)):
        s, e = int(L.start[k]), int(L.end[k])
        pre = ev[(ev.end >= s - 400) & (ev.end <= s + 30)]
        nxt_lo = e - 60
        post = ev[(ev.start >= nxt_lo) & (ev.start <= e + 400)]
        r = dict(u=u, p=int(L.p[k]), start=s, end=e)
        if len(pre):
            m = pre.loc[(s - pre.end).abs().idxmin()]
            r["pre_end"] = int(m.end)
            r["off_s"] = s - int(m.end)
        if len(post) and k < len(L) - 1:
            m2 = post.loc[(post.start - e).abs().idxmin()]
            r["post_start"] = int(m2.start)
            r["off_e"] = int(m2.start) - e
        val.append(r)
V = pd.DataFrame(val)
off_s, off_e = V.off_s.dropna(), V.off_e.dropna()
OFF_S, OFF_E = int(round(off_s.median())), int(round(off_e.median()))
mad = lambda x: float((x - x.median()).abs().median())
cal = dict(OFF_S=OFF_S, OFF_E=OFF_E, n_start=int(len(off_s)), n_end=int(len(off_e)), mad_s=mad(off_s), mad_e=mad(off_e),
           p90_s=float(off_s.quantile(.9)), p90_e=float(off_e.quantile(.9)), frac_start_found=float(V.off_s.notna().mean()),
           frac_start_within_60=float(((V.off_s - OFF_S).abs() <= 60).sum() / V.off_s.notna().sum()),
           frac_end_within_60=float(((V.off_e - OFF_E).abs() <= 60).sum() / V.off_e.notna().sum()),
           n_events_median=float(np.median([len(EV[u]) for u in clean])), thr=THR, min_sites=MIN_SITES)
V.to_csv(RES / "marker_validation.csv", index=False)
json.dump(cal, open(RES / "marker_calibration.json", "w"), indent=1)
print("\ncalibration on the clean drivers:", json.dumps(cal, indent=1))

# ---- 3. audit of every label range ----------------------------------------------------------------------------------------------------------------
flags = []
for u in range(1, 31):
    ev = EV[u]
    L = lab[lab.u == u].sort_values("start").reset_index(drop=True)
    for r in L.itertuples():
        pre = ev[(ev.end >= r.start - 400) & (ev.end <= r.start + 30)]
        pre_idx, off = None, None
        if len(pre):
            pre_idx = (r.start - pre.end).abs().idxmin()
            off = int(r.start - ev.loc[pre_idx, "end"])
        inter = ev[(ev.start >= r.start + 60) & (ev.start <= r.end - 40)]
        inter = inter[np.array(inter.index) != (-1 if pre_idx is None else pre_idx)]
        kinds = []
        if len(inter):
            kinds.append("interior_marker")
        if off is None:
            kinds.append("no_preceding_marker")
        elif off > 200:
            kinds.append("late_start")
        if kinds:
            flags.append(dict(u=u, p=int(r.p), start=int(r.start), end=int(r.end), length=int(r.end - r.start + 1), pre_marker_offset=off, kinds="+".join(kinds),
                              interior=str([(int(a), int(b)) for a, b in zip(inter.start, inter.end)]), clean_driver=u in clean))
F = pd.DataFrame(flags)
F.to_csv(RES / "marker_label_flags.csv", index=False)
print("\nflagged label ranges: %d (clean drivers: %d; interior markers in clean drivers: %d)" % (len(F), int(F.clean_driver.sum()),
      int(((F.clean_driver) & F.kinds.str.contains("interior")).sum())))
print(F[F.clean_driver & F.kinds.str.contains("interior")].to_string(index=False))

# ---- 4. how many labelled frames of the analysis cohort lie inside a marker press? -----------------------------------------------------------------
fr = load_frames()
n_in, n_all, n_rows_in, n_rows = 0, 0, 0, 0
PAD = 10
for u in clean:
    g = fr[(fr.u == u) & (fr.label > 0)]
    fid = g.frame_id.values
    inside = np.zeros(len(fid), bool)
    for e in EV[u].itertuples():
        inside |= (fid >= e.start - PAD) & (fid <= e.end + PAD)
    n_in += int(inside.sum())
    n_all += len(fid)
    rows_ = (g.frame_id % 6 == 0) & (g.frame_id - g.seg_start >= 68)
    n_rows += int(rows_.sum())
    n_rows_in += int((inside & rows_.values).sum())
cal.update(frames_in_marker=n_in, frames_labelled=n_all, rows_in_marker=n_rows_in, rows=n_rows)
json.dump(cal, open(RES / "marker_calibration.json", "w"), indent=1)
print(f"labelled frames of the cohort inside a marker press (+-{PAD} frames): {n_in} of {n_all} ({100*n_in/n_all:.2f} %); evaluation rows: {n_rows_in} of {n_rows} ({100*n_rows_in/n_rows:.2f} %)")
