"""Summary of E2 (adaptation operators): per-driver accuracies (mean over the 5 shuffles), bootstrap CIs over drivers, paired differences against
no calibration and against mean subtraction at the same scope/window, Wilcoxon + Holm within each family, and the pre-specified decision rule.

DECISION RULE (fixed when only the first 24-cell runs of the first seeds had been seen, not changed afterwards): an operator replaces mean subtraction as
the primary operator only if, against mean subtraction with the same causal window, its mean paired gain is >= 1.0 pp with Holm-adjusted Wilcoxon p < 0.05
at BOTH windows (60 s, 120 s) and BOTH representations (24 and 96 cells).  Otherwise the parsimonious operator (mean subtraction) stays primary.
"""
import numpy as np
import pandas as pd

import stats_utils as su
from common import RES

d = pd.read_csv(RES / "e2_operators.csv")
MAIN = ["cmvn", "wh(a=0.1)", "wh(a=0.5)", "ea(a=0.1)"]          # the pre-specified family; everything else is hyperparameter sensitivity (raw p, exploratory)


def _name(r):
    if r.op in ("wh", "ea"):
        return f"{r.op}(a={r.alpha})"
    if r.op == "cmvn" and r.alpha > 0:
        return f"cmvn(floor={r.alpha})"
    return r.op


d["name"] = d.apply(_name, axis=1)
d["group"] = np.where(d.scope == "run", "run" + d.W.astype(str), d.scope)
# per-driver accuracy: mean over the shuffles
pdv = d.groupby(["f", "group", "name", "u"]).acc.mean().reset_index()

rows, rule = [], []
for f, g in pdv.groupby("f"):
    none = g[(g.group == "full") & (g.name == "none")].set_index("u").acc
    for grp, h in g.groupby("group"):
        if grp == "full":
            continue
        wide = h.pivot(index="u", columns="name", values="acc")
        ref = "sub" if "sub" in wide.columns else None
        fam = [c for c in wide.columns if c in MAIN]
        # operators vs none (descriptive) and vs mean subtraction (family, Holm)
        pv = su.paired_vs_reference(wide.assign(none=none), "sub").set_index("config") if ref else pd.DataFrame()
        for c in wide.columns:
            v = wide[c].dropna().values * 100
            lo, hi = su.boot_ci(v)
            dn = (wide[c] - none).values * 100
            nlo, nhi = su.boot_ci(dn)
            r = dict(f=f, group=grp, op=c, acc=v.mean(), acc_lo=lo, acc_hi=hi, gain_vs_none=dn.mean(), gvn_lo=nlo, gvn_hi=nhi)
            if ref and c != "sub":
                p = pv.loc[c]
                r.update(diff_vs_sub=p.mean_diff, dvs_lo=p.ci_lo, dvs_hi=p.ci_hi, p_wilcoxon=p.p)
            rows.append(r)
        # Holm over the operator family only (exclude the 'none' comparison)
        if ref:
            idx = [i for i, r in enumerate(rows) if r["f"] == f and r["group"] == grp and "p_wilcoxon" in r and r["op"] in fam]
            ph = su.holm([rows[i]["p_wilcoxon"] for i in idx])
            for i, p_ in zip(idx, ph):
                rows[i]["p_holm"] = p_
out = pd.DataFrame(rows)
out.to_csv(RES / "e2_summary.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
show = out[["f", "group", "op", "acc", "acc_lo", "acc_hi", "gain_vs_none", "diff_vs_sub", "dvs_lo", "dvs_hi", "p_holm"]]
print(show.round(2).to_string(index=False))

# is the label-using first-order oracle a ceiling?  paired contrast: whole-session operator - oracle (per driver), Holm over the five session operators of each resolution
orc_rows = []
for f, g in pdv.groupby("f"):
    orc = g[g.group == "oracle"].set_index("u").acc
    sess = g[g.group == "session"]
    fam = []
    for op, h in sess.groupby("name"):
        if op not in ["sub"] + MAIN:
            continue
        dd = ((h.set_index("u").acc - orc).dropna() * 100).values
        lo, hi = su.boot_ci(dd)
        fam.append(dict(f=f, op=op, diff_vs_oracle=dd.mean(), lo=lo, hi=hi, p=su.wilcoxon(dd)))
    ph = su.holm([r["p"] for r in fam])
    for r, p_ in zip(fam, ph):
        r["p_holm"] = p_
    orc_rows += fam
oc = pd.DataFrame(orc_rows)
oc.to_csv(RES / "e2_oracle_check.csv", index=False)
print("\nORACLE CHECK (whole-session operator minus the label-using first-order oracle, pp):")
print(oc.round(3).to_string(index=False))

# decision rule
print("\nDECISION RULE: mean paired gain over mean subtraction >= 1.0 pp with Holm p < 0.05 at W = 60 s and 120 s, at 24 and 96 cells")
for op in out[out.op.isin(["cmvn", "wh(a=0.1)", "wh(a=0.5)", "ea(a=0.1)"])].op.unique():
    ok = True
    parts = []
    for f in (16, 8):
        for grp in ("run60", "run120"):
            r = out[(out.f == f) & (out.group == grp) & (out.op == op)].iloc[0]
            this = bool(r.diff_vs_sub >= 1.0 and r.p_holm < 0.05)
            ok &= this
            parts.append(f"f{f}/{grp}: {r.diff_vs_sub:+.1f} pp p={r.p_holm:.3f}{'*' if this else ''}")
    rule.append(dict(op=op, replaces_mean_subtraction=ok, detail="; ".join(parts)))
    print(f"  {op:10s} replaces mean subtraction: {ok}   ({'; '.join(parts)})")
pd.DataFrame(rule).to_csv(RES / "e2_decision_rule.csv", index=False)

# shuffle noise: SD over seeds of the cohort-mean accuracy
sd = d[d.scope == "run"].groupby(["f", "name", "W", "seed"]).acc.mean().groupby(["f", "name", "W"]).std()
print("\nSD over the 5 shuffles of the cohort-mean accuracy (pp): max %.2f, median %.2f" % (sd.max() * 100, sd.median() * 100))
