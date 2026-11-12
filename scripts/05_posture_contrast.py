"""What do the 20 postures physically change?  Driver-centred contrast maps + variance decomposition (analysis cohort, balanced: all 20 postures).

Contrast map = posture mean map minus the same driver's mean over its postures, averaged over drivers (all columns shown, so that the excluded outer strip is visible).
Variance decomposition = driver / posture / interaction-residual share of the sum of squares of the driver x posture mean maps (4x4-pooled), per strip.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CACHE, DEFAULT_COLS, FIG, REGIONS, RES, cohort_users, tag

seg = np.load(CACHE / "segmean_fullres.npy")          # (30, 21, 64, 160); zeros where posture missing
have = seg.sum(axis=(2, 3)) > 0                        # (30, 21)
have[:, 0] = False
keep = np.zeros(30, bool)
keep[[u - 1 for u in cohort_users()]] = True
have[~keep] = False
print("drivers:", int(keep.sum()), "| segments with data:", int(have.sum()), "of", int(keep.sum()) * 20)

# ---- driver-centred contrast ------------------------------------------------------------------------------------------------------------------
subj_mean = np.zeros((30, 64, 160), np.float32)
for s in range(30):
    if keep[s]:
        subj_mean[s] = seg[s, have[s]].mean(axis=0)
cm = np.zeros((21, 64, 160), np.float32)
for p in range(1, 21):
    sel = [s for s in range(30) if have[s, p]]
    cm[p] = np.mean([seg[s, p] - subj_mean[s] for s in sel], axis=0)

c0 = 16
fig, axs = plt.subplots(4, 5, figsize=(17, 5.8))
vmax = np.percentile(np.abs(cm[1:]), 99.5)
for p in range(1, 21):
    a = axs[(p - 1) // 5, (p - 1) % 5]
    a.imshow(cm[p][:, c0:], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    a.axvline(DEFAULT_COLS[0] - c0 - 0.5, color="k", lw=0.8, ls="--")
    a.set_title(f"P{p}", fontsize=9)
    a.axis("off")
fig.suptitle("Posture mean minus the same driver's mean over all postures (N/cm2), averaged over drivers; dashed line = start of the analysed area (column 64)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "posture_contrast_maps.png", dpi=120)
plt.close(fig)


# ---- variance decomposition on 4x4-pooled segment means ------------------------------------------------------------------------------------------
def pool(a, f, cols):
    a = a[..., cols[0]:cols[1]]
    return a.reshape(*a.shape[:-2], 64 // f, f, (cols[1] - cols[0]) // f, f).mean(axis=(-3, -1))


bal = [s for s in range(30) if have[s, 1:].all()]
print("balanced drivers:", len(bal))
vd = []
for name, cols in (("all", DEFAULT_COLS), *REGIONS.items()):
    S = pool(seg[np.ix_(bal, range(1, 21))], 4, cols).reshape(len(bal), 20, -1)
    g = S.mean(axis=(0, 1))
    st = ((S - g) ** 2).sum()
    ssu = 20 * ((S.mean(axis=1) - g) ** 2).sum()
    ssp = len(bal) * ((S.mean(axis=0) - g) ** 2).sum()
    print(f"  {name:>5} cols {cols}: driver {100*ssu/st:5.1f}% | posture {100*ssp/st:5.1f}% | interaction {100*(st-ssu-ssp)/st:5.1f}%")
    vd.append(dict(region=name, cols=str(cols), n_drivers=len(bal), subject=100 * ssu / st, posture=100 * ssp / st, interaction=100 * (st - ssu - ssp) / st))
pd.DataFrame(vd).to_csv(RES / f"variance_decomposition{tag()}.csv", index=False)
