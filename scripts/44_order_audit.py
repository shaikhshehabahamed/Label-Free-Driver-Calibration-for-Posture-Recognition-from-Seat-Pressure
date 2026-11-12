"""Order audit: what do the data themselves say about the fixed posture order, drift and time structure?  (No input from the dataset authors needed.)

 A  order           Kendall tau between posture index and start frame, per driver, in the raw label table and in the repaired table
 B  load drift      driver-normalised total load per protocol slot (analysis cohort): does the load trend with elapsed time?
 C  smoothness      share of the between-slot variance of the driver-centred slot-mean maps that a low-order polynomial in slot index explains,
                    against a null in which the slot order is permuted.  Excess smoothness = drift-like (monotone) structure; none = slot signatures are
                    idiosyncratic, as expected for postures
 D  adjacency       share of unseen-driver errors that go to the neighbouring slot (|i-j| = 1) against the uniform expectation (needs confusion_*.csv)
 E  reversed pair   driver 3 (P19 before P18): template test from 43_repair_labels.py
Writes results/order_audit.csv and figures/F11_order_audit.png.      usage: [DATASET=clean] python 44_order_audit.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CACHE, DEFAULT_COLS, FIG, RES, cohort_users, tag
from stats_utils import boot_ci

facts = []


def add(key, value, note="", **extra):
    facts.append(dict(key=key, value=value, note=note, **extra))
    print(f"{key:44s} {value}  {note}", flush=True)


# ---- A: order ------------------------------------------------------------------------------------------------------------------------------
raw = pd.read_csv(RES / "labels_clean.csv")
rep = pd.read_csv(RES / "labels_repaired.csv")
for name, tab in (("raw", raw), ("repaired", rep[rep.use])):
    taus = {u: stats.kendalltau(g.p, g.start)[0] for u, g in tab.groupby("u")}
    off = sorted(u for u, t in taus.items() if t < 0.9999)
    add(f"A.drivers_in_protocol_order_{name}", int(sum(t > 0.9999 for t in taus.values())), f"of {len(taus)}; deviating: {off}")
sw = RES / "repair_swap_test.csv"
if sw.exists():
    s = pd.read_csv(sw).iloc[0]
    add("E.driver3_delta", round(float(s.delta), 3), f"template score; p under protocol order = {s.p_if_protocol_order:.3f}")

# ---- B/C: drift and smoothness on the analysis cohort ---------------------------------------------------------------------------------------------
seg = np.load(CACHE / "segmean_fullres.npy")                                  # (30, 21, 64, 160)
users = [u for u in cohort_users()]
c0, c1 = DEFAULT_COLS
A = seg[[u - 1 for u in users]][:, 1:, :, c0:c1]                              # (n, 20, 64, 96)
F = A.reshape(len(users), 20, 8, 8, 12, 8).mean(axis=(3, 5)).reshape(len(users), 20, -1)    # 8x12 pooled -> 96 features
load = A.sum(axis=(2, 3))                                                     # relative total load per segment
rel_load = load / load.mean(axis=1, keepdims=True)                            # driver-normalised
prof = rel_load.mean(axis=0)
lo, hi = np.percentile([rel_load[np.random.default_rng(b).integers(0, len(users), len(users))].mean(axis=0) for b in range(2000)], [2.5, 97.5], axis=0)
slot = np.arange(1, 21)
per_driver_rho = np.array([stats.spearmanr(slot, rel_load[i])[0] for i in range(len(users))])
add("B.load_profile_range_pct", round(100 * (prof.max() - prof.min()), 1), "max-min of the mean driver-normalised total load over the 20 slots")
add("B.load_trend_rho_median", round(float(np.median(per_driver_rho)), 3), f"median per-driver Spearman(slot, load); Wilcoxon p = {stats.wilcoxon(per_driver_rho).pvalue:.3f}", p=float(stats.wilcoxon(per_driver_rho).pvalue))

C = F - F.mean(axis=1, keepdims=True)                                         # remove each driver's own mean (driver offset)
M = C.mean(axis=0)                                                            # (20, 96) common slot signature
M = M - M.mean(axis=0)


def poly_r2(M, deg):
    X = np.vander((slot - slot.mean()) / slot.std(), deg + 1, increasing=True)[:, 1:]       # centred, intercept removed (M is centred)
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    fit = X @ beta
    return float((fit ** 2).sum() / (M ** 2).sum())


rng = np.random.default_rng(0)
res = {}
for deg in (1, 2, 3):
    obs = poly_r2(M, deg)
    null = np.array([poly_r2(M[rng.permutation(20)], deg) for _ in range(5000)])
    boots = []
    for b in range(500):
        ix = rng.integers(0, len(users), len(users))
        Mb = C[ix].mean(axis=0)
        Mb = Mb - Mb.mean(axis=0)
        boots.append(poly_r2(Mb, deg))
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    res[deg] = (obs, null, p)
    add(f"C.smooth_R2_degree{deg}", round(100 * obs, 1), f"% of between-slot variance; permutation null mean {100 * null.mean():.1f}%, p = {p:.4f}; bootstrap 95% CI "
        f"[{100 * np.percentile(boots, 2.5):.1f}, {100 * np.percentile(boots, 97.5):.1f}]", null_mean=round(100 * null.mean(), 1), p=p,
        ci_lo=round(100 * np.percentile(boots, 2.5), 1), ci_hi=round(100 * np.percentile(boots, 97.5), 1))

# ---- D: adjacency of errors --------------------------------------------------------------------------------------------------------------------------
adj = {}
for base in ("none", "run60", "oracle"):
    p_ = RES / f"confusion_{base}{tag()}.csv"
    if p_.exists():
        Cm = pd.read_csv(p_, index_col=0).values.astype(float)
        err = Cm.sum() - np.trace(Cm)
        lag = {d: sum(Cm[i, j] for i in range(20) for j in range(20) if abs(i - j) == d) for d in range(1, 20)}
        share = {d: lag[d] / err for d in lag}
        uni = {d: 2 * (20 - d) / (20 * 19) for d in range(1, 20)}
        adj[base] = (share, uni)
        add(f"D.errors_to_adjacent_slot_{base}", round(100 * share[1], 1), f"% of errors; uniform expectation {100 * uni[1]:.1f}%; within +-2 slots {100 * (share[1] + share[2]):.1f}% (uniform {100 * (uni[1] + uni[2]):.1f}%)",
            uniform=round(100 * uni[1], 1), within2=round(100 * (share[1] + share[2]), 1), within2_uniform=round(100 * (uni[1] + uni[2]), 1))

pd.DataFrame(facts).to_csv(RES / f"order_audit{tag()}.csv", index=False)

# ---- figure -------------------------------------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
ax = axs[0]
ax.plot(slot, 100 * prof, color="#1f77b4", marker="o", ms=3)
ax.fill_between(slot, 100 * lo, 100 * hi, color="#1f77b4", alpha=0.2)
ax.axhline(100, color="k", lw=0.6, ls=":")
ax.set_xlabel("protocol slot (posture index)"); ax.set_ylabel("total load, % of driver's mean"); ax.set_title("(a) load across the session", fontsize=9)
ax = axs[1]
obs, null, p = res[2]
ax.hist(100 * null, bins=40, color="#bbbbbb")
ax.axvline(100 * obs, color="#d62728", lw=1.5)
ax.set_xlabel("between-slot variance explained by a\nquadratic trend in slot (%)"); ax.set_ylabel("permuted slot orders"); ax.set_title(f"(b) smooth drift? observed {100 * obs:.0f}%, " + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}"), fontsize=9)
ax = axs[2]
if adj:
    d = np.arange(1, 8)
    w = 0.38
    share, uni = adj.get("none", next(iter(adj.values())))
    ax.bar(d - w / 2, [100 * share[k] for k in d], w, color="#7f7f7f", label="observed")
    ax.bar(d + w / 2, [100 * uni[k] for k in d], w, color="#bdbdbd", hatch="//", label="uniform confusions")
    ax.set_xlabel("slot distance |i - j| of the wrong posture"); ax.set_ylabel("share of errors (%)"); ax.legend(frameon=False, fontsize=7)
ax.set_title("(c) where unseen-driver errors go", fontsize=9)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / f"F11_order_audit{tag()}.png", dpi=200, bbox_inches="tight")
plt.close(fig)
