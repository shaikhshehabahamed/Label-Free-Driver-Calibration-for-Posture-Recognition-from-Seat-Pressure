"""Exploratory data analysis: coverage, sensor geometry, saturation, load-vs-weight, posture mean maps."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, FIG, RES, load_frames, load_users

users = load_users()
fr = load_frames()
st = np.load(CACHE / "sensel_stats.npz")
N = int(st["n_frames"])

# ---- 1. per-user coverage ------------------------------------------------------------------------
g = fr.groupby("u")
tab = pd.DataFrame({
    "frames": g.size(),
    "minutes": g.time_s.apply(lambda s: (s.max() - s.min()) / 60),
    "fps": g.size() / g.time_s.apply(lambda s: s.max() - s.min()),
    "labelled": g.label.apply(lambda s: int((s > 0).sum())),
    "overlap": g.label.apply(lambda s: int((s == -1).sum())),
    "marker_frames": g.label.apply(lambda s: int((s == -2).sum())),
    "unlabelled": g.label.apply(lambda s: int((s == 0).sum())),
    "postures_with_data": g.label.apply(lambda s: s[s > 0].nunique()),
}).reset_index().merge(users, on="u")
tab["label_frac"] = tab.labelled / tab.frames
tab.to_csv(RES / "eda_users.csv", index=False)
print("== coverage ==")
print(f"frames {N} | labelled(single) {int((fr.label>0).sum())} | overlap {int((fr.label==-1).sum())} | marker presses inside labels {int((fr.label==-2).sum())} | unlabelled {int((fr.label==0).sum())}")
print(f"fps: median {tab.fps.median():.2f} (min {tab.fps.min():.2f}, max {tab.fps.max():.2f}); minutes median {tab.minutes.median():.1f}")
print("postures with data per user != 20:", tab.loc[tab.postures_with_data != 20, ["u", "postures_with_data"]].values.tolist())
print("phase counts:", fr.groupby("phase").size().to_dict())

# ---- 2. sensor geometry ---------------------------------------------------------------------------
act = st["act"] / N
sat = st["sat"] / N
rows = np.where((act > 0.05).any(axis=1))[0]
cols = np.where((act > 0.05).any(axis=0))[0]
print("\n== geometry ==")
print(f"sensels active in >5% of frames: {(act>0.05).sum()} of {act.size}  | row range {rows.min()}-{rows.max()}  col range {cols.min()}-{cols.max()}")
print(f"sensels ever active: {(act>0).sum()}, never active: {(act==0).sum()}")
print(f"saturated (>=cap) in >1% of frames: {(sat>0.01).sum()} sensels; max sat fraction {sat.max():.3f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for a, m, t in zip(ax, (act, sat), ("fraction of frames active (>0)", "fraction of frames saturated (3.41 N/cm2)")):
    im = a.imshow(m, aspect="equal", cmap="viridis")
    a.set_title(t, fontsize=10)
    plt.colorbar(im, ax=a, fraction=0.025)
    a.set_xlabel("column (160)")
    a.set_ylabel("row (64)")
fig.tight_layout()
fig.savefig(FIG / "eda_sensel_activity.png", dpi=130)
plt.close(fig)

# ---- 3. load vs body weight -------------------------------------------------------------------------
lab = fr[fr.label > 0]
per_u = lab.groupby("u").agg(force_med=("force_N", "median"), force_p10=("force_N", lambda s: s.quantile(.1)),
                             force_p90=("force_N", lambda s: s.quantile(.9)),
                             sat_med=("n_sat", "median"), act_med=("n_active", "median"),
                             contact_med=("contact_cm2", "median")).reset_index().merge(users, on="u")
per_u["weight_N"] = per_u.weight_kg * 9.81
per_u["load_ratio"] = per_u.force_med / per_u.weight_N
per_u["sat_frac"] = per_u.sat_med / per_u.act_med
per_u.to_csv(RES / "eda_load_vs_weight.csv", index=False)
r, p = stats.pearsonr(per_u.weight_kg, per_u.force_med)
rs, ps = stats.spearmanr(per_u.weight_kg, per_u.force_med)
print("\n== recorded load vs body weight (per-user median over labelled frames) ==")
print(f"load ratio (sensed force / body weight): mean {per_u.load_ratio.mean():.2f}, min {per_u.load_ratio.min():.2f}, max {per_u.load_ratio.max():.2f}")
print(f"Pearson r(weight, force) = {r:.2f} (p={p:.3g}); Spearman = {rs:.2f} (p={ps:.3g})")
r2, p2 = stats.spearmanr(per_u.bmi, per_u.load_ratio)
print(f"Spearman(BMI, load ratio) = {r2:.2f} (p={p2:.3g})")
r3, p3 = stats.spearmanr(per_u.weight_kg, per_u.sat_frac)
print(f"saturated share of active sensels: median {per_u.sat_frac.median():.3f}, max {per_u.sat_frac.max():.3f}; Spearman(weight, sat share) = {r3:.2f} (p={p3:.3g})")
print(per_u.sort_values("weight_kg")[["u", "sex", "height_cm", "weight_kg", "bmi", "force_med", "load_ratio", "sat_frac"]].round(2).tail(6).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(per_u.weight_kg, per_u.force_med, c=np.where(per_u.sex == "F", "C3", "C0"))
ax[0].set_xlabel("body weight (kg)"); ax[0].set_ylabel("median sensed force (N)")
ax[1].scatter(per_u.weight_kg, 100 * per_u.sat_frac, c=np.where(per_u.sex == "F", "C3", "C0"))
ax[1].set_xlabel("body weight (kg)"); ax[1].set_ylabel("saturated sensels (% of active)")
fig.tight_layout(); fig.savefig(FIG / "eda_load_saturation.png", dpi=130); plt.close(fig)

# ---- 4. mean posture maps -----------------------------------------------------------------------------
seg = np.load(CACHE / "segmean_fullres.npy")                   # (30, 21, 64, 160)
r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
fig, axs = plt.subplots(4, 5, figsize=(15, 5.6))
for p in range(1, 21):
    m = seg[:, p][seg[:, p].sum(axis=(1, 2)) > 0].mean(axis=0)
    a = axs[(p - 1) // 5, (p - 1) % 5]
    a.imshow(m[r0:r1, c0:c1], cmap="magma", vmin=0, vmax=1.6, aspect="equal")
    a.set_title(f"P{p}", fontsize=9); a.axis("off")
fig.tight_layout(); fig.savefig(FIG / "eda_posture_mean_maps.png", dpi=120); plt.close(fig)

# ---- 5. within-segment redundancy ------------------------------------------------------------------------
p4 = np.load(CACHE / "pooled_f4.npy", mmap_mode="r")
rng = np.random.default_rng(0)
cors = []
for (u, p), d in fr[fr.label > 0].groupby(["u", "label"]):
    idx = d.index.values
    if len(idx) < 40:
        continue
    X = np.asarray(p4[idx]).reshape(len(idx), -1)
    X = X - X.mean(axis=0)
    for lag in (1, 5, 13, 40):
        a_, b_ = X[:-lag], X[lag:]
        den = np.sqrt((a_ ** 2).sum(axis=1) * (b_ ** 2).sum(axis=1)) + 1e-9
        cors.append((u, p, lag, float(((a_ * b_).sum(axis=1) / den).mean())))
cdf = pd.DataFrame(cors, columns=["u", "p", "lag", "corr"])
print("\n== within-segment frame similarity (centred cosine of 16x40 maps, mean over segments) ==")
print(cdf.groupby("lag")["corr"].describe()[["mean", "25%", "50%", "75%"]].round(3).to_string())
print("done")
