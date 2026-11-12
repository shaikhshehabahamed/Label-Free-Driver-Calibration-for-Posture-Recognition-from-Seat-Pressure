"""Figure: label ranges before and after the repair for the drivers whose label table was inconsistent or truncated."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import FIG, RES, load_frames

rep = pd.read_csv(RES / "labels_repaired.csv")
fr = load_frames()
last = fr.groupby("u").frame_id.max()
COL = {"ok": "#9ecae1", "repaired": "#fd8d3c", "recovered": "#31a354", "partial_overlap": "#de2d26", "order_deviation": "#756bb1", "no_data": "#ffffff", "truncated": "#fdd0a2", "marker_anchored": "#41b6c4"}
LAB = {"ok": "unchanged", "repaired": "bound corrected (R1)", "recovered": "kept, no longer lost to overlap (R1)", "partial_overlap": "overlap, shared frames dropped (R2)",
       "order_deviation": "performed out of protocol order (R3)", "no_data": "no recorded data (R4)", "truncated": "recording ends inside (R4)", "marker_anchored": "rebuilt from the marker presses (M3)"}
users = [2, 3, 4, 6, 8, 22, 24, 28]
fig, axs = plt.subplots(len(users), 1, figsize=(7.2, 1.15 * len(users)), sharex=False)
for ax, u in zip(axs, users):
    g = rep[rep.u == u].sort_values("p")
    for r in g.itertuples():
        ax.barh(1, r.orig_end - r.orig_start + 1, left=r.orig_start, height=0.7, color="#d9d9d9", edgecolor="k", lw=0.3)
        ax.text((r.orig_start + r.orig_end) / 2, 1, str(r.p), ha="center", va="center", fontsize=4.5)
        ax.barh(0, r.end - r.start + 1, left=r.start, height=0.7, color=COL.get(r.status, "#9ecae1"), edgecolor="k", lw=0.3, alpha=1.0 if r.use else 0.35)
        ax.text((r.start + r.end) / 2, 0, str(r.p), ha="center", va="center", fontsize=4.5)
    ax.axvline(last[u], color="k", lw=0.8, ls="--")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["repaired", "as published"], fontsize=6)
    ax.set_ylabel(f"driver {u}", fontsize=7, rotation=0, labelpad=26, va="center")
    ax.tick_params(axis="x", labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(g.orig_end.max(), g.end.max(), last[u]) + 200)
axs[-1].set_xlabel("frame index (dashed line = last recorded frame)", fontsize=7)
handles = [Patch(facecolor=COL[k], edgecolor="k", lw=0.3, label=LAB[k]) for k in ("ok", "repaired", "recovered", "partial_overlap", "order_deviation", "marker_anchored")]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "F12_label_repair.png", dpi=200, bbox_inches="tight")
print("saved F12_label_repair.png")
