"""Summary of E3 (seconds vs distinct postures).  Per-driver accuracy = mean over the random draws; gain = accuracy - no calibration (paired by driver);
95 % bootstrap CIs over drivers; the seconds-vs-diversity question is answered with paired contrasts and a random-intercept mixed model on the
per-driver gains (gain ~ log2 k + log2 n + (1 | driver)); the running-window ladder is tabulated against the number of distinct postures it spans.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import stats_utils as su
from common import RES

d = pd.read_csv(RES / "e3_composition.csv")
pdv = d.groupby(["kind", "k", "n", "own", "W", "u"], dropna=False).acc.mean().reset_index()
none = pdv[pdv.kind == "none"].set_index("u").acc


def gain(sel):
    g = pdv[sel].set_index("u").acc
    return (g - none).dropna() * 100


def ci(v):
    lo, hi = su.boot_ci(v.values)
    return v.mean(), lo, hi


out = []
print("running window (random-order streams, 10 shuffles) vs the posture diversity it spans")
wc = pd.read_csv(RES / "e3_window_composition.csv").set_index("W")
for W in (15, 30, 60, 120, 300):
    v = gain((pdv.kind == "run") & (pdv.W == W))
    m, lo, hi = ci(v)
    out.append(dict(block="run", k=np.nan, n=np.nan, own="", W=W, gain=m, lo=lo, hi=hi, p=su.wilcoxon(v.values), distinct=wc.loc[W, "distinct"], effective=wc.loc[W, "effective"]))
    print(f"  W={W:3d} s: gain {m:+.2f} [{lo:+.2f}, {hi:+.2f}] pp   spans {wc.loc[W, 'distinct']:.1f} distinct postures (effective {wc.loc[W, 'effective']:.1f})")
for kind in ("session", "oracle"):
    v = gain(pdv.kind == kind)
    m, lo, hi = ci(v)
    out.append(dict(block=kind, k=np.nan, n=np.nan, own="", W=0, gain=m, lo=lo, hi=hi, p=su.wilcoxon(v.values)))
    print(f"  {kind:8s}: gain {m:+.2f} [{lo:+.2f}, {hi:+.2f}] pp")

print("\nbaseline built from k distinct postures x n seconds each (own posture excluded) - gain over no calibration, pp [95 % CI]")
G = {}
for own in ("excl", "incl"):
    for (k, n), _ in pdv[(pdv.kind == "bag") & (pdv.own == own)].groupby(["k", "n"]):
        v = gain((pdv.kind == "bag") & (pdv.own == own) & (pdv.k == k) & (pdv.n == n))
        G[(own, int(k), int(n))] = v
        m, lo, hi = ci(v)
        out.append(dict(block="bag_" + own, k=k, n=n, own=own, W=0, gain=m, lo=lo, hi=hi, p=su.wilcoxon(v.values), total_s=k * n))
tab = pd.DataFrame([dict(k=k, n=n, own=o, **dict(zip(("gain", "lo", "hi"), ci(v)))) for (o, k, n), v in G.items()])
for own in ("excl", "incl"):
    t = tab[tab.own == own]
    if len(t):
        print(f" own posture {own}:")
        piv = t.assign(txt=t.apply(lambda r: f"{r.gain:+.1f} [{r.lo:+.1f},{r.hi:+.1f}]", axis=1)).pivot(index="k", columns="n", values="txt")
        print(piv.to_string())

CONTR = []


def contrast(a, b, label):
    dd = (G[a] - G[b]).dropna()
    m, lo, hi = ci(dd)
    CONTR.append(dict(block="contrast", gain=m, lo=lo, hi=hi, p=su.wilcoxon(dd.values), label=label))


for n in (2, 5, 10):
    contrast(("excl", 5, n), ("excl", 1, n), f"diversity: k=5 vs k=1 at n={n} s")
    contrast(("excl", 19, n), ("excl", 5, n), f"diversity: k=19 vs k=5 at n={n} s")
for k in (1, 3, 5, 12, 19):
    contrast(("excl", k, 10), ("excl", k, 2), f"seconds per posture: n=10 vs n=2 at k={k}")
contrast(("excl", 1, 10), ("excl", 5, 2), "same 10 s in total: 1 posture x 10 s vs 5 postures x 2 s")
contrast(("excl", 5, 2), ("excl", 1, 10), "same 10 s in total: 5 postures x 2 s vs 1 posture x 10 s")
contrast(("excl", 3, 10), ("excl", 12, 2), "30 s vs 24 s in total: 3 x 10 s vs 12 x 2 s")
if ("incl", 5, 5) in G:
    contrast(("incl", 5, 5), ("excl", 5, 5), "current posture may be drawn (incl) vs excluded, k=5, n=5")
    contrast(("incl", 19, 5), ("excl", 19, 5), "current posture drawn (incl, k=19 of 20) vs excluded, k=19, n=5")

print("\npaired contrasts (pp, 95 % CI over drivers; bootstrap, own posture excluded unless stated; Holm over this family)")
cdf = pd.DataFrame(CONTR)
cdf["p_holm"] = su.holm(cdf.p.values)
for _, r in cdf.iterrows():
    print(f"  {r.label:62s} {r.gain:+6.2f} [{r.lo:+6.2f}, {r.hi:+6.2f}]  p={r.p:.4f}  Holm {r.p_holm:.4f}")
out += cdf.to_dict("records")

# mixed model on per-driver gains: gain ~ log2 k + log2 n + (1 | driver)
rows = []
for (o, k, n), v in G.items():
    if o == "excl":
        for u_, g_ in v.items():
            rows.append(dict(u=u_, k=k, n=n, gain=g_))
L = pd.DataFrame(rows)
L["lk"], L["ln"] = np.log2(L.k), np.log2(L.n)
mm = smf.mixedlm("gain ~ lk + ln", L, groups=L["u"]).fit(reml=True)
print("\nmixed model on per-driver gains (own posture excluded), pp per doubling:")
print(mm.summary().tables[1].to_string())
mm2 = smf.mixedlm("gain ~ lk * ln", L, groups=L["u"]).fit(reml=True)
print("with interaction: lk x ln coef %.3f (p=%.3f)" % (mm2.params["lk:ln"], mm2.pvalues["lk:ln"]))
# saturation: fit gain ~ lk within n = 5 only
L5 = L[L.n == 5]
mm5 = smf.mixedlm("gain ~ lk", L5, groups=L5["u"]).fit(reml=True)
print("n=5 only: gain per doubling of k = %.2f pp (SE %.2f)" % (mm5.params["lk"], mm5.bse["lk"]))
res = pd.DataFrame(out)
res.to_csv(RES / "e3_summary.csv", index=False)
pd.DataFrame({"term": mm.params.index, "coef": mm.params.values, "se": mm.bse.values, "p": mm.pvalues.values}).to_csv(RES / "e3_mixed_model.csv", index=False)
