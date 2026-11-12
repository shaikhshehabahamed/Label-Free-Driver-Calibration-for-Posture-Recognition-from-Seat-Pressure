"""Figure: the marker presses that precede each posture, and how published labels sit relative to them (22 consistent drivers).
Needs results/marker_events.csv and results/marker_validation.csv (48_markers.py)."""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import marker_lib as ml
from common import FIG, RES

lab = pd.read_csv(RES / "labels_clean.csv")
ev = pd.read_csv(RES / "marker_events.csv")
val = pd.read_csv(RES / "marker_validation.csv")
cal = json.load(open(RES / "marker_calibration.json"))
u = 12
fid, S = ml.site_scores(u)
fig, axs = plt.subplots(1, 3, figsize=(11.0, 2.8), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
ax = axs[0]
ax.plot(fid, (S > 1.0).sum(axis=1), lw=0.6, color="k")
for r in lab[lab.u == u].itertuples():
    ax.axvspan(r.start, r.end, color=plt.cm.tab20((r.p - 1) % 20), alpha=0.25)
for e in ev[ev.u == u].itertuples():
    ax.axvline(e.start, color="#d62728", lw=0.8)
ax.set_xlim(fid.min(), fid.max()); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel("frame"); ax.set_ylabel("marker sites pressed (of 4)"); ax.set_title(f"(a) driver {u}: marker presses (red) fall between the labelled postures (shaded)", fontsize=8)
for ax, col, ttl, lab_ in ((axs[1], "off_s", "(b) label start - end of preceding marker", "frames"), (axs[2], "off_e", "(c) start of next marker - label end", "frames")):
    v = val[col].dropna()
    ax.hist(v.clip(-60, 300), bins=np.arange(-60, 301, 10), color="#4292c6")
    ax.axvline(v.median(), color="#d62728", lw=1.2)
    ax.set_xlabel(lab_); ax.set_title(f"{ttl}\nmedian {v.median():.0f}, MAD {(v - v.median()).abs().median():.0f} (n = {len(v)})", fontsize=8)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "F16_markers.png", dpi=200, bbox_inches="tight")
print("saved F16_markers.png")
