"""Which parts of the 64 x 160 mat carry posture information?  Evidence from the data only (the mat's physical layout is undocumented).

Column profile of (i) the share of frames in which a sensel is active, (ii) mean pressure over singly labelled frames of the cohort, and (iii) the share of total force;
contiguous active bands are reported with their column ranges.  Writes results/geometry_audit.csv and figures/F13_geometry.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CACHE, DEFAULT_COLS, FIG, REGIONS, RES, cohort_users, load_frames, load_pooled

st = np.load(CACHE / "sensel_stats.npz")
act = st["act"] / int(st["n_frames"])                                  # (64, 160) share of frames with pressure > 0
colact = act.mean(axis=0)
fr = load_frames()
p4 = load_pooled(4)
m = (fr.label > 0) & fr.u.isin(cohort_users()) & (fr.frame_id % 12 == 0)
Xp = np.asarray(p4[np.where(m.values)[0]], dtype=np.float32)            # (n, 16, 40) mean-pooled, 4x4
col_mean = Xp.mean(axis=(0, 1))                                         # (40,) mean pressure per pooled column
colp = np.repeat(col_mean, 4)                                           # back to 160 columns
share = colp / colp.sum()

# contiguous bands where mean pressure exceeds 10 % of its maximum
on = colp > 0.10 * colp.max()
bands, s = [], None
for c in range(160):
    if on[c] and s is None:
        s = c
    if (not on[c] or c == 159) and s is not None:
        e = c if not on[c] else c + 1
        if e - s >= 8:
            bands.append((s, e))
        s = None
rows = []
for a, b in bands:
    rows.append(dict(band=f"{a}-{b}", cols=f"{a}-{b}", share_of_force=float(share[a:b].sum()), mean_pressure=float(colp[a:b].mean()), active_share=float(colact[a:b].mean())))
for name, (a, b) in {**REGIONS, "analysed (64-160)": DEFAULT_COLS}.items():
    rows.append(dict(band=name, cols=f"{a}-{b}", share_of_force=float(share[a:b].sum()), mean_pressure=float(colp[a:b].mean()), active_share=float(colact[a:b].mean())))
out = pd.DataFrame(rows)
out.to_csv(RES / "geometry_audit.csv", index=False)
print(out.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10.0, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
im = axs[0].imshow(act, aspect="equal", cmap="viridis")
axs[0].axvline(DEFAULT_COLS[0] - 0.5, color="w", lw=1, ls="--")
axs[0].set_xlabel("column (160)"); axs[0].set_ylabel("row (64)"); axs[0].set_title("(a) share of frames in which a sensel is active", fontsize=9)
plt.colorbar(im, ax=axs[0], fraction=0.025)
ax = axs[1]
ax.plot(np.arange(160), colp, color="#1f77b4")
ax.axvspan(0, DEFAULT_COLS[0], color="#999999", alpha=0.18, label="excluded (columns 0-64)")
ax.axvspan(*REGIONS["back"], color="#2ca02c", alpha=0.10, label="backrest-like strip")
ax.axvspan(*REGIONS["seat"], color="#ff7f0e", alpha=0.10, label="seat-pan-like strip")
ax.set_xlabel("column"); ax.set_ylabel("mean pressure (N/cm$^2$)"); ax.legend(frameon=False, fontsize=7); ax.set_title("(b) mean pressure by column, labelled frames", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "F13_geometry.png", dpi=200, bbox_inches="tight")
