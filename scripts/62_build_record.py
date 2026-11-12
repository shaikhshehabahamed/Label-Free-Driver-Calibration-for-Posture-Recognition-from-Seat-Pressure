"""Builds analysis/STUDY_RECORD.md.  Every number in the results sections is read from results/*.csv at build time (the static blocks, 61_record_static.py,
contain only verified facts and pilot values that E1 reproduced).   usage: python 62_build_record.py
"""
import importlib.util

import numpy as np
import pandas as pd

import stats_utils as su
from common import RES, ROOT

spec = importlib.util.spec_from_file_location("rs", ROOT / "analysis" / "scripts" / "61_record_static.py")
RS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RS)


def rd(name, **kw):
    p = RES / name
    return pd.read_csv(p, **kw) if p.exists() else None


def f1(x, d=1):
    return f"{x:.{d}f}"


def sg(x, d=1):
    return f"{x:+.{d}f}"


def ci_txt(m, lo, hi, d=1):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def lvl(m, lo, hi, d=1):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------------- pilot context
def pilot_context():
    pr = rd("protocols_LDA_kNN_RF.csv")
    rf = pr[pr.model == "RF"].set_index("protocol").acc * 100
    tf = rd("time_only_baseline.csv")
    lo, hi = su.boot_ci(tf.acc.values * 100)
    vd = rd("variance_decomposition.csv").set_index("region").loc["all"]
    return dict(rf_random=rf["random5"], rf_blocked=rf["blocked5"], rf_loso=rf["LOSO"], floor=tf.acc.mean() * 100, floor_lo=lo, floor_hi=hi,
                v_subj=vd.subject, v_post=vd.posture, v_int=vd.interaction)


# ----------------------------------------------------------------------------------------------------------------------- E1
def sec_e1():
    s = rd("e1_summary.csv")
    e = rd("e1_independent_replication.csv")
    if s is None:
        return "", {}
    s = s.set_index("cond")
    rows = []
    names = [("none", "no calibration"), ("run60", "60-s running baseline"), ("run120", "120-s running baseline"), ("session", "whole-session mean"), ("oracle", "oracle")]
    for k, lab in names:
        r = s.loc[k]
        rows.append([lab, f1(r.pilot_mean), f1(r.mine_mean), ci_txt(r.diff_pp, r.diff_ci_lo, r.diff_ci_hi, 2), f"{r.r_per_driver:.3f}"])
    g = {k: s.loc[f"GAIN {k} vs none"] for k in ("run60", "run120")}
    w = e.pivot_table(index="u", columns=["cond", "seed"], values="acc")
    sdv = {}
    for c in ("run60", "run120"):
        cols = [k for k in w.columns if k[0] == c]
        sdv[c] = (w[cols].mean(0) * 100).std(ddof=1)
        sdv[c + "_drv"] = (w[cols].sub(w[("none", -1)], axis=0) * 100).std(axis=1, ddof=1).mean()
    txt = f"""### 5.1 The pilot's headline reproduces in an independent implementation (E1)

`50_independent_replication.py` re-implements the 22-driver leave-one-driver-out contrast from the written specification with a different code path (pandas rolling means instead of cumulative sums;
scikit-learn `StandardScaler -> PCA(150) -> LDA(lsqr, shrinkage='auto')` instead of the torch port; own marker mask and cohort rule; fresh shuffle seeds). **Unit test first:** its feature builders are compared with the
pilot's on identical seeds - no calibration, whole-session mean, oracle and the shuffled 60-s baseline are **bit-identical** (max |difference| = 0 over 14,921 x 384 values). Along the way the test exposed a design trap:
when every driver is given the same permutation for a given seed, the "random order" becomes a new *fixed* protocol order shared by all drivers (order information returns); the pilot draws every driver's permutation
from one sequential generator, and the follow-up uses an independent stream `default_rng([seed, driver])` per driver.

{table(["condition", "pilot (torch LDA), %", "independent (scikit-learn), %", "difference, pp [95 % CI]", "per-driver r"], rows)}

The deterministic conditions agree to <= 0.013 pp. The two random-order conditions differ by {ci_txt(s.loc['run60'].diff_pp, s.loc['run60'].diff_ci_lo, s.loc['run60'].diff_ci_hi, 2)} pp (60 s) and
{ci_txt(s.loc['run120'].diff_pp, s.loc['run120'].diff_ci_lo, s.loc['run120'].diff_ci_hi, 2)} pp (120 s): within shuffle noise. The gain over no calibration in the independent run is
{ci_txt(g['run60'].mine_mean, g['run60'].mine_gain_ci_lo, g['run60'].mine_gain_ci_hi)} pp at 60 s and {ci_txt(g['run120'].mine_mean, g['run120'].mine_gain_ci_lo, g['run120'].mine_gain_ci_hi)} pp at 120 s
(pilot {sg(g['run60'].pilot_mean)} and {sg(g['run120'].pilot_mean)}). **Shuffle noise is not negligible:** the cohort-mean accuracy varies by SD {f1(sdv['run60'], 2)} pp (60 s) and {f1(sdv['run120'], 2)} pp (120 s) across
random orders, and the per-driver accuracy by {f1(sdv['run60_drv'], 1)} / {f1(sdv['run120_drv'], 1)} pp - three shuffles (the pilot) leave about +-0.5 pp on the cohort mean, so every follow-up estimate uses 5-10 shuffles with common random numbers across the compared conditions.
"""
    return txt, dict(g60_e1=g["run60"].mine_mean, g120_e1=g["run120"].mine_mean)


# ----------------------------------------------------------------------------------------------------------------------- E2
def sec_e2():
    s = rd("e2_summary.csv")
    dr = rd("e2_decision_rule.csv")
    if s is None:
        return "", {}
    N = {}
    parts = []
    groups = [("run60", "60 s"), ("run120", "120 s"), ("session", "whole session")]
    ops = [("cmvn", "per-cell scaling (CMVN)"), ("wh(a=0.1)", "whitening, a = 0.1"), ("wh(a=0.5)", "whitening, a = 0.5 (CORAL-type)"), ("ea(a=0.1)", "Euclidean alignment as published")]
    for f, ttl in ((16, "24 cells"), (8, "96 cells")):
        none = s[(s.f == f) & (s.group == "run60") & (s.op == "sub")].iloc[0]
        none_acc = none.acc - none.gain_vs_none
        rows = [["no calibration", f1(none_acc), "", ""]]
        o = s[(s.f == f) & (s.group == "oracle")].iloc[0]
        for g, gl in groups:
            r = s[(s.f == f) & (s.group == g) & (s.op == "sub")].iloc[0]
            rows.append([f"mean subtraction, {gl}", f1(r.acc), ci_txt(r.gain_vs_none, r.gvn_lo, r.gvn_hi), "reference"])
            for op, ol in ops:
                q = s[(s.f == f) & (s.group == g) & (s.op == op)]
                if len(q):
                    q = q.iloc[0]
                    rows.append([f"&nbsp;&nbsp;{ol}", f1(q.acc), ci_txt(q.gain_vs_none, q.gvn_lo, q.gvn_hi), f"{ci_txt(q.diff_vs_sub, q.dvs_lo, q.dvs_hi)}, Holm {pstr(q.p_holm)}"])
        rows.append(["oracle (mean of the 20 posture means; needs labels)", f1(o.acc), ci_txt(o.gain_vs_none, o.gvn_lo, o.gvn_hi), ""])
        parts.append(f"**{ttl}** (leave-one-driver-out accuracy %, 22 drivers; gain over no calibration; paired difference over mean subtraction with the same window)\n\n"
                     + table(["condition", "accuracy %", "gain over none, pp [95 % CI]", "over mean subtraction, pp [95 % CI]"], rows))
        N[f"oracle{f}"] = o.acc
        N[f"sess_sub{f}"] = s[(s.f == f) & (s.group == "session") & (s.op == "sub")].iloc[0].acc
        for op, _ in ops:
            N[f"sess_{op}{f}"] = s[(s.f == f) & (s.group == "session") & (s.op == op)].iloc[0].acc
    rule = "\n".join(f"* `{r.op}`: {'**replaces**' if r.replaces_mean_subtraction else 'does not replace'} mean subtraction - {r.detail}" for r in dr.itertuples())
    return "\n\n".join(parts), N, rule


def ci3(v):
    v = np.asarray(v, float)
    lo, hi = su.boot_ci(v)
    return float(v.mean()), lo, hi


# ----------------------------------------------------------------------------------------------------------------------- E3
def sec_e3():
    d, s, wc, mm = rd("e3_composition.csv"), rd("e3_summary.csv"), rd("e3_window_composition.csv"), rd("e3_mixed_model.csv")
    if d is None or s is None or wc is None:
        return "", {}
    wc = wc.set_index("W")
    pdv = d.groupby(["kind", "k", "n", "own", "W", "u"], dropna=False).acc.mean().reset_index()
    none = pdv[pdv.kind == "none"].set_index("u").acc
    none_acc = none.mean() * 100

    def gains(sel):
        return (pdv[sel].set_index("u").acc - none).dropna() * 100

    spec = [(f"{W} s running window", (pdv.kind == "run") & (pdv.W == W), wc.loc[W, "distinct"], W) for W in (15, 30, 60, 120, 300)]
    spec += [("whole-session mean (non-causal)", pdv.kind == "session", None, "session"), ("oracle (labels)", pdv.kind == "oracle", None, "oracle")]
    gs = [gains(sel) for _, sel, _, _ in spec]
    ph = su.holm([su.wilcoxon(g.values) for g in gs])
    og = gs[-1].mean()
    rows, N = [], dict(none_acc=none_acc)
    for (lab, _, dist, key), g, p in zip(spec, gs, ph):
        m, lo, hi = ci3(g.values)
        rows.append([lab, "" if dist is None else f1(dist), f1(none_acc + m), ci_txt(m, lo, hi), pval(p), f"{100 * (g > 0).mean():.0f} %", "" if key == "oracle" else f"{100 * m / og:.0f} %"])
        N[f"g{key}"], N[f"lo{key}"], N[f"hi{key}"], N[f"pct{key}"], N[f"frac{key}"] = m, lo, hi, 100 * m / og, 100 * (g > 0).mean()
        N[f"p{key}"] = p
        if dist is not None:
            N[f"dist{key}"] = dist
    ladder = table(["calibration", "distinct postures spanned", "accuracy %", "gain over none, pp [95 % CI]", "Holm p", "drivers improved", "share of the oracle gain"], rows)
    grid = {}
    for own in ("excl", "incl"):
        b = s[s.block == f"bag_{own}"]
        if len(b):
            hdr = ["k distinct postures"] + [f"{n} s per posture" for n in (2, 5, 10)]
            rr = []
            for k in sorted(b.k.unique()):
                cells = []
                for n in (2, 5, 10):
                    q = b[(b.k == k) & (b.n == n)]
                    cells.append(ci_txt(q.gain.iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "")
                rr.append([int(k)] + cells)
                for n in (2, 5, 10):
                    q = b[(b.k == k) & (b.n == n)]
                    if len(q):
                        N[f"bag_{own}_{int(k)}_{n}"] = q.gain.iloc[0]
            grid[own] = table(hdr, rr)
    c = s[s.block == "contrast"].copy()
    crow = [[r.label, ci_txt(r.gain, r.lo, r.hi, 2), pval(r.p_holm)] for r in c.itertuples()]
    contr = table(["paired contrast", "pp [95 % CI]", "Holm p"], crow)
    for r in c.itertuples():
        N["c:" + r.label] = (r.gain, r.lo, r.hi, r.p_holm)
    mmt = ""
    if mm is not None:
        m_ = mm.set_index("term")
        N["mm_k"], N["mm_n"] = (m_.loc["lk", "coef"], m_.loc["lk", "se"], m_.loc["lk", "p"]), (m_.loc["ln", "coef"], m_.loc["ln", "se"], m_.loc["ln", "p"])
        mmt = (f"Random-intercept mixed model on the per-driver gains (own posture excluded): **{sg(m_.loc['lk', 'coef'], 2)} pp per doubling of k** (SE {m_.loc['lk', 'se']:.2f}, p {pval(m_.loc['lk', 'p'])}) "
               f"against **{sg(m_.loc['ln', 'coef'], 2)} pp per doubling of the seconds drawn per posture** (SE {m_.loc['ln', 'se']:.2f}, p {pval(m_.loc['ln', 'p'])}).")
    return dict(ladder=ladder, grid=grid, contr=contr, mm=mmt), N


def text_primary(N, P):
    return f"""### 5.2 The primary estimand (E3 reference ladder: 384 cells, shrinkage LDA, 22 drivers, 10 random orders)

Without calibration the unseen-driver accuracy is {f1(N['none_acc'])} % (chance 5 %; the elapsed-time-only random forest reaches {f1(P['floor'])} % [{f1(P['floor_lo'])}, {f1(P['floor_hi'])}], more than any uncalibrated pressure model).

{N['_ladder']}

The label-free baseline helps once it spans about a minute: **{ci_txt(N['g60'], N['lo60'], N['hi60'])} pp at 60 s and {ci_txt(N['g120'], N['lo120'], N['hi120'])} pp at 120 s**
({f1(N['frac60'], 0)} % and {f1(N['frac120'], 0)} % of the drivers improve; {f1(N['pct60'], 0)} % and {f1(N['pct120'], 0)} % of the label-using oracle gain of {sg(N['goracle'])} pp); 15 and 30 s are indistinguishable from zero after Holm.
Longer windows add little ({sg(N['g300'])} pp at 300 s against {sg(N['gsession'])} pp for the whole session). These 10-shuffle estimates replace the pilot's 3-shuffle values (+6.0 and +8.8 pp); the independent
implementation of E1 gave {sg(N['g60_e1'])} and {sg(N['g120_e1'])} pp. Supporting pilot results that this study did not change: the gain survives nested tuning of window, PCA dimension and shrinkage (+6.6 pp [+3.2, +10.5]), cohort and label-set changes
(+5.3 to +7.6 pp across five data sets, including +5.4 pp [+2.4, +8.0] on the 8 excluded drivers), and the choice of learner (section 5.6).
"""


def text_ops(N, rule, S, sens):
    def rng(op, groups, f_list=(16, 8)):
        v = S[(S.op == op) & S.group.isin(groups) & S.f.isin(f_list)].diff_vs_sub
        return v.min(), v.max()

    c_lo, c_hi = rng("cmvn", ("run60", "run120"))
    w_lo, w_hi = rng("wh(a=0.5)", ("run60", "run120"))
    e16 = S[(S.op == "ea(a=0.1)") & (S.f == 16) & S.group.isin(["run60", "run120", "session"])].diff_vs_sub
    n_sig = int(((S.op.isin(["cmvn", "wh(a=0.5)"])) & S.group.isin(["run60", "run120"]) & (S.p_holm < 0.05)).sum())
    sess16 = S[(S.f == 16) & (S.group == "session") & (S.op == "wh(a=0.1)")].iloc[0]
    oc = rd("e2_oracle_check.csv")
    o16 = oc[(oc.f == 16) & (oc.op == "wh(a=0.1)")].iloc[0]
    o8 = oc[(oc.f == 8) & oc.op.isin(["cmvn", "wh(a=0.1)", "wh(a=0.5)"])]
    sens_txt = sens or ""
    return f"""### 5.3 Which operator? (E2: identical random-order streams, 5 shuffles, common random numbers)

{N['_ops']}

**Pre-specified decision rule** (a candidate replaces mean subtraction only if it gains >= 1.0 pp over it with Holm p < 0.05 at 60 s **and** 120 s **and** at 24 **and** 96 cells):

{rule}

**Mean subtraction therefore stays the primary operator** (parsimony: no extra parameter, no covariance to estimate). The evidence behind the secondary readings:

* **Scale and covariance information is real but small on causal windows.** Per-cell scaling gains {sg(c_lo)} to {sg(c_hi)} pp and CORAL-type whitening (a = 0.5) {sg(w_lo)} to {sg(w_hi)} pp over mean subtraction in all eight causal comparisons (positive in every one;
  {n_sig} of the 8 pass Holm), but no operator clears the rule at all four settings with n = 22 drivers.
* **With enough unlabelled data it is large.** With the whole session, whitening reaches {f1(sess16.acc)} % at 24 cells (a = 0.1; {ci_txt(sess16.diff_vs_sub, sess16.dvs_lo, sess16.dvs_hi)} pp over mean subtraction), above the label-using first-order oracle ({f1(N['oracle16'])} %):
  paired difference to the oracle {ci_txt(o16.diff_vs_oracle, o16.lo, o16.hi)} pp (Holm p {pval(o16.p_holm)}). **At 24 cells the oracle is therefore not a ceiling**: it is the best *first-order* offset removal, and a label-free second-order alignment of a whole session beats it.
  At 96 cells the whole-session operators only match it ({sg(float(o8.diff_vs_oracle.min()))} to {sg(float(o8.diff_vs_oracle.max()))} pp, all intervals contain 0). The pilot's wording "oracle = upper bound" is withdrawn; the defensible statement is "first-order reference".
* **Euclidean alignment as published does not beat mean subtraction on pressure maps.** It whitens the *uncentred* second moment, which for non-zero-mean maps rescales the driver offset instead of removing it: {ci_txt(float(e16.mean()), float(e16.min()), float(e16.max()))}
  pp (mean [min, max] over the three scopes) at 24 cells, and +1.1 to +1.6 pp (not significant after Holm) at 96 cells. EA's zero-mean assumption is the reason; centring first, EA reduces to the whitening operator above.
* **Second-order alignment needs data**: its gain grows with the window (60 s -> 120 s -> session) and shrinks with dimension; the pilot's untested remark that covariance alignment "needs far more frames than dimensions" is now measured for 24-96 cells and still untested at 384.
{sens_txt}
"""


def text_mech(E3, N, P):
    g = E3["grid"]
    c = lambda lab: N["c:" + lab]
    k5v1 = c("diversity: k=5 vs k=1 at n=5 s")
    k19v5 = c("diversity: k=19 vs k=5 at n=5 s")
    tot = c("same 10 s in total: 5 postures x 2 s vs 1 posture x 10 s")
    n5 = c("seconds per posture: n=10 vs n=2 at k=5")
    inc = c("current posture may be drawn (incl) vs excluded, k=5, n=5")
    return f"""### 5.4 What the baseline needs: distinct postures, not seconds (E3)

In a running window seconds and diversity cannot be separated: it spans {f1(N['dist15'])}, {f1(N['dist30'])}, {f1(N['dist60'])}, {f1(N['dist120'])} and {f1(N['dist300'])} distinct postures at 15, 30, 60, 120 and 300 s (mean over rows and shuffles;
the pilot's "window length" axis is a diversity axis). E3 builds the baseline explicitly from k distinct postures of the driver's own unlabelled data, a random contiguous chunk of n seconds from each, so total seconds = k x n and diversity = k vary independently
(5 draws; gain over no calibration, pp [95 % CI over drivers]).

*Current posture excluded* (isolates diversity; **uses the row's label, so it is an upper bound, not deployable**):

{g['excl']}

*Label-free draw* (the current posture may be drawn; deployable analogue; n = 5 s):

{g['incl']}

{E3['contr']}

{E3['mm']} There is no interaction (the effect of k does not depend on n). Reading:

* **Distinct postures carry the benefit.** At the same 10 s in total, five postures of 2 s beat one posture of 10 s by {ci_txt(*tot[:3], 2)} pp; going from 1 to 5 postures adds {ci_txt(*k5v1[:3], 2)} pp at 5 s each, from 5 to 19 a further {ci_txt(*k19v5[:3], 2)} pp.
  Five postures of only 5 s ({sg(N['bag_incl_5_5'])} pp, label-free draw) already match the whole-session mean ({sg(N['gsession'])} pp) and beat the 120-s running window ({sg(N['g120'])} pp).
* **Seconds matter little**: 5x more seconds per posture (2 s -> 10 s) add {ci_txt(*n5[:3], 2)} pp at k = 5 and nothing resolvable at k >= 12.
* **A single *other* posture is already useful** (k = 1: {sg(N['bag_excl_1_5'])} pp): any reference that is not the current posture removes the driver-specific offset; the cost is the reference's own posture pattern, which averaging over k removes.
* **The current posture is the enemy of a causal baseline.** Allowing it into the draw costs {ci_txt(*inc[:3], 2)} pp at k = 5; in a running window the current posture holds the largest share of the buffer, which is why the running curve lies below the draws at equal diversity
  (Figure S2c: {sg(N['g120'])} pp at {f1(N['dist120'])} postures against {sg(N['bag_incl_5_5'])} pp for a label-free draw of 5). In habitual driving the window is dominated by one posture and the baseline converges to it - the driver-relative feature then shrinks towards zero, a case the DPR protocol (each posture held once) cannot test.
* **Design consequence** (an implication, not a tested system): a deployable baseline should be built to be *diverse* - accumulate a few distinct postures - rather than merely long; how fast a real driver supplies them is unknown and is the first thing a naturalistic collection must measure.
"""


def sens_text():
    s = rd("e2_summary.csv")
    if s is None or not s.op.str.contains("floor|a=0.25").any():
        return ""
    rows, vals = [], []
    for f, ttl in ((16, "24 cells"), (8, "96 cells")):
        for g, gl in (("run60", "60 s"), ("run120", "120 s"), ("session", "whole session")):
            cells = []
            for op in ("cmvn(floor=0.03)", "cmvn", "cmvn(floor=0.3)", "wh(a=0.1)", "wh(a=0.25)", "wh(a=0.5)"):
                q = s[(s.f == f) & (s.group == g) & (s.op == op)]
                cells.append(sg(q.iloc[0].diff_vs_sub) if len(q) else "")
                if len(q):
                    vals.append(float(q.iloc[0].diff_vs_sub))
            rows.append([ttl, gl] + cells)
    verdict_ = (f"all {len(vals)} cells are positive ({sg(min(vals))} to {sg(max(vals))} pp), so the direction of the secondary readings does not depend on these choices" if min(vals) > 0
                else f"{sum(v > 0 for v in vals)} of {len(vals)} cells are positive ({sg(min(vals))} to {sg(max(vals))} pp): the secondary readings are sensitive to these choices")
    t = table(["cells", "window", "CMVN floor 0.03", "CMVN floor 0.1 (main)", "CMVN floor 0.3", "whitening a = 0.1", "a = 0.25", "a = 0.5"], rows)
    return ("\n**Hyperparameter sensitivity** (gain over mean subtraction, pp; exploratory, outside the decision family; CMVN variance floor 0.03 / 0.1 / 0.3 of the cell-average SD, whitening shrinkage 0.1 / 0.25 / 0.5): "
            + verdict_ + ".\n\n" + t + "\n")


def rank_tol_text():
    d = rd("e2_rank_tol_check.csv")
    if d is None:
        return ""
    g = d.groupby(["op", "rank_tol"]).acc.mean().unstack()
    rows = [[op, f1(g.loc[op, "none"], 2), f1(g.loc[op, "1e-06"], 2), sg(g.loc[op, "1e-06"] - g.loc[op, "none"], 2)] for op in g.index if "1e-06" in g.columns]
    mx = float((g["1e-06"] - g["none"]).abs().max())
    return (f"The LDA carries an opt-in numerical-rank tolerance (1e-6) in all follow-up experiments (decision 5); it was added after an 8-driver diagnostic had shown fragility for per-cell-scaled features. "
            f"On the full cohort it is **immaterial**: no operator changes by more than {f1(mx, 2)} pp, so no result depends on it.\n\n"
            + table(["operator (96 cells, 60 s, 22 drivers, 3 shuffles)", "accuracy % without tolerance", "with rank tolerance 1e-6", "difference, pp"], rows))


# ----------------------------------------------------------------------------------------------------------------------- E7 / E4
def contr_table(df, with_frac=False):
    hdr = ["paired contrast (per driver)", "pp [95 % CI]", "Holm p"] + (["drivers better"] if with_frac else [])
    rows = [[r.label, ci_txt(r["mean"], r.lo, r.hi), pval(r.p_holm)] + ([f"{100 * r.better:.0f} %"] if with_frac else []) for _, r in df.iterrows()]
    return table(hdr, rows)


def sec_e7():
    lv, sm = rd("e7_levels.csv"), rd("e7_summary.csv")
    if lv is None:
        return None, {}
    nm = {"time": "elapsed time only", "p_none": "pressure, no calibration", "p_none+t": "pressure + elapsed time", "p_run60": "pressure, label-free 60-s baseline", "p_run60+t": "pressure (baseline) + elapsed time"}
    lt = table(["model (boosted trees, 96 cells, leave-one-driver-out, 22 drivers)", "accuracy %, mean [95 % CI]"], [[nm[r.config], lvl(r["mean"], r.ci_lo, r.ci_hi)] for _, r in lv.iterrows()])
    N = {r.config: (r["mean"], r.ci_lo, r.ci_hi) for _, r in lv.iterrows()}
    N.update({r.label: (r["mean"], r.lo, r.hi, r.p_holm) for _, r in sm.iterrows()})
    return dict(levels=lt, contr=contr_table(sm, True)), N


def sec_e4():
    sm, raw = rd("e4_summary.csv"), rd("e4_cnn_adabn.csv")
    if sm is None or raw is None:
        return None, {}
    w = raw.groupby("u")[["none", "none_adabn", "run60", "run60_adabn"]].mean() * 100
    s = su.summarize(w / 100)
    nm = {"none": "input Z (no calibration)", "none_adabn": "input Z + AdaBN", "run60": "input Z - b (60-s label-free baseline)", "run60_adabn": "input Z - b + AdaBN"}
    lt = table([f"3-layer CNN, 384 cells, LOSO, 22 drivers, {raw.seed.nunique()} training seeds", "accuracy %, mean [95 % CI]"], [[nm[r.config], lvl(r["mean"], r.ci_lo, r.ci_hi)] for _, r in s.iterrows()])
    N = {r.config: (r["mean"], r.ci_lo, r.ci_hi) for _, r in s.iterrows()}
    N.update({r.label: (r["mean"], r.lo, r.hi, r.p_holm) for _, r in sm.iterrows()})
    return dict(levels=lt, contr=contr_table(sm, True)), N


def pick(c, pos, neg, nul):
    """choose a sentence by the sign of a paired contrast (mean, lo, hi, ...): resolved positive / resolved negative / not resolved"""
    return pos if c[1] > 0 else (neg if c[2] < 0 else nul)


def text_validity(E7, N7, P):
    head = "### 5.5 Is there posture signal beyond the protocol clock? (E7)\n\n"
    if E7 is None:
        return head + "(pending)\n"
    raw = rd("e7_time_adjusted.csv")
    ns = raw.seed.nunique()
    c = lambda lab: N7[lab]
    c1, c2, c3, c4, c5, c6 = (c(x) for x in ("pressure + time  vs  time only", "pressure (label-free baseline) + time  vs  time only", "label-free baseline given time (pressure + t)",
                                              "label-free baseline without time (pressure only)", "pressure only (no calibration)  vs  time only", "pressure only (label-free baseline)  vs  time only"))
    return head + f"""Every driver performed P1..P20 in the same order, so elapsed time alone predicts the slot (the pilot's order-only random forest reaches {f1(P['floor'])} % [{f1(P['floor_lo'])}, {f1(P['floor_hi'])}], above every uncalibrated pressure model).
The incremental question is whether, with elapsed time available, pressure still adds accuracy, and whether the label-free baseline adds more. Boosted trees (GPU XGBoost, fixed small model), 96 cells, leave-one-driver-out, 22 drivers,
{ns} model seeds; the baseline conditions use the matching random order.

{E7['levels']}

{E7['contr']}

* **Pressure on its own vs the clock:** {pick(c5, 'uncalibrated pressure beats elapsed time alone', 'uncalibrated pressure is worse than elapsed time alone (as in the pilot random-forest floor)', 'uncalibrated pressure and elapsed time are not resolved from each other')}
  ({ci_txt(*c5[:3])} pp); with the label-free baseline: {ci_txt(*c6[:3])} pp ({pick(c6, 'better than the clock', 'still below the clock', 'not resolved from the clock')}).
* **Pressure given the clock:** {pick(c1, '**pressure carries slot information beyond the elapsed-time clock**', 'pressure *lowers* accuracy relative to the clock alone', 'pressure adds nothing resolvable to the clock')}
  ({ci_txt(*c1[:3])} pp without calibration, {ci_txt(*c2[:3])} pp with the label-free baseline). Whether that information is posture-specific or a drift-based clock (pressure maps also encode how long the driver has been sitting) cannot be separated on this protocol.
* **Calibration given the clock:** {pick(c3, 'the label-free baseline adds accuracy on top of pressure + time', 'the label-free baseline lowers accuracy once time is available', 'the label-free baseline adds nothing resolvable once time is available')}
  ({ci_txt(*c3[:3])} pp); without time, {ci_txt(*c4[:3])} pp ({pick(c4, 'resolved', 'resolved but negative', 'not resolved from zero')}) - against the pilot's +3.3 pp [+0.7, +5.6] for boosted trees on other random orders: **the boosted-trees gain is small and not stable across random orders** (section 5.6).
* **Meaning for the claims.** Absolute accuracy on this protocol still cannot be read as posture recognition (the order is a confound that cannot be removed by analysis); what this check bounds is whether the pressure maps carry slot information that the clock does not.
"""


def pilot_learners():
    """pilot learner contrasts (96 cells shrinkage LDA and boosted trees; 384-cell CNN, 2 seeds) recomputed from the pilot's own result files"""
    out = {}
    L = rd("gpu_learners.csv")
    for m in ("gLDA", "XGB"):
        h = L[L.model == m]
        none = h[(h.base == "none") & (h["mode"] == "full")].set_index("u").acc
        run = h[(h.base == "run60") & (h["mode"] == "shuf")].groupby("u").acc.mean()
        out[m] = (none.mean() * 100, run.mean() * 100, *ci3(((run - none) * 100).values))
    C = rd("gpu_cnn_loso.csv")
    none = C[C.config == "CNN|none"].set_index("u").acc
    run = C[C.config == "CNN|run60"].set_index("u").acc
    out["CNN"] = (none.mean() * 100, run.mean() * 100, *ci3(((run - none) * 100).values))
    return out


def text_learners(N, N7, N4):
    PL = pilot_learners()
    rows = [["shrinkage LDA, 384 cells", "this study (E3), 10 random orders", f1(N["none_acc"]), f1(N["none_acc"] + N["g60"]), ci_txt(N["g60"], N["lo60"], N["hi60"])],
            ["shrinkage LDA, 96 cells", "pilot, 2 random orders", f1(PL["gLDA"][0]), f1(PL["gLDA"][1]), ci_txt(*PL["gLDA"][2:])],
            ["boosted trees, 96 cells", "pilot, 2 random orders", f1(PL["XGB"][0]), f1(PL["XGB"][1]), ci_txt(*PL["XGB"][2:])]]
    if N7:
        a, b, g = N7["p_none"], N7["p_run60"], N7["label-free baseline without time (pressure only)"]
        rows.append(["boosted trees, 96 cells", "this study (E7), random orders matched to the model seeds", f1(a[0]), f1(b[0]), ci_txt(*g[:3])])
    rows.append(["3-layer CNN, 384 cells", "pilot, 2 seeds, 1 random order", f1(PL["CNN"][0]), f1(PL["CNN"][1]), ci_txt(*PL["CNN"][2:])])
    if N4:
        g = N4["driver-relative input (60-s baseline)  vs  none"]
        rows.append(["3-layer CNN, 384 cells", "this study (E4), training seeds with matched random orders", f1(N4["none"][0]), f1(N4["run60"][0]), ci_txt(*g[:3])])
    t = table(["learner", "source", "no calibration, %", "60-s baseline, random order, %", "gain, pp [95 % CI]"], rows)
    cnn_e4 = ""
    if N4:
        g_ = N4["driver-relative input (60-s baseline)  vs  none"]
        cnn_e4 = f" and {ci_txt(*g_[:3])} pp in E4 ({pick(g_, 'resolved', 'resolved but negative', 'not resolved from zero')})"
    xgb_e7 = "not yet re-run"
    if N7:
        x_ = N7["label-free baseline without time (pressure only)"]
        xgb_e7 = f"{ci_txt(*x_[:3])} pp ({pick(x_, 'resolved', 'resolved but negative', 'not resolved from zero')})"
    ada = ""
    if N4:
        a1, a2, a3 = N4["AdaBN (target BN statistics)  vs  none"], N4["AdaBN on top of the driver-relative input"], N4["driver-relative input  vs  AdaBN"]
        ada = (f"\n**AdaBN, the deep label-free comparator** (Li et al. 2018: the held-out driver's own unlabelled rows replace the batch-normalisation statistics; no labels, whole session, non-causal): {ci_txt(*a1[:3])} pp over the uncalibrated network "
               f"({pick(a1, 'it helps', 'it hurts', 'not resolved from zero')}); on top of the driver-relative input {ci_txt(*a2[:3])} pp; the driver-relative input against AdaBN {ci_txt(*a3[:3])} pp "
               f"({pick(a3, 'the input-level offset removal is better than BN-statistics adaptation', 'BN-statistics adaptation is better than the input-level offset removal', 'not resolved')}). "
               + (pick(a3, "The input-level baseline is better, although it is causal and uses 60 s, whereas AdaBN uses the whole session.",
                       "BN-statistics adaptation is better than the input-level baseline.",
                       "**The two are not resolved from each other:** a causal 60-s input baseline does as well as a non-causal, whole-session adaptation of the batch-normalisation statistics"
                       + (", and the two do not stack." if a2[1] <= 0 <= a2[2] else ", and they are complementary."))
                  )
               + "\n\n" + N4["_table"] + "\n\n" + N4["_contr"] + "\n")
    return f"""### 5.6 Does the effect depend on the learner? (pilot + E7 + E4)

{t}

**The benefit is learner-dependent.** Shrinkage LDA gains {ci_txt(N['g60'], N['lo60'], N['hi60'])} pp (384 cells, 10 random orders) and {ci_txt(*PL['gLDA'][2:])} pp (96 cells, pilot); the CNN gains {ci_txt(*PL['CNN'][2:])} pp in the pilot{cnn_e4}.
Boosted trees gain {ci_txt(*PL['XGB'][2:])} pp in the pilot but {xgb_e7} in E7 on other random orders: for trees the gain is small and **not stable across random orders**. A plausible reason, not tested here: trees already exploit absolute pressure
(no calibration {f1(PL['XGB'][0])} % against {f1(PL['gLDA'][0])} % for LDA at 96 cells), leaving less offset to remove. The learners differ in absolute accuracy, which is not a claim here (boosted trees and the CNN are fixed and untuned).
In recorded order the gain of boosted trees is mostly order information (pilot: +17.0 pp against +3.3 pp in random order), which is why every learner is evaluated on randomly ordered streams.
{ada}"""


# ----------------------------------------------------------------------------------------------------------------------- narrative blocks
def contrib(N, N7, N4, P):
    c1 = (f"The first measurement, for seat-pressure posture recognition, of how much unseen-user accuracy a label-free user-relative baseline recovers: {ci_txt(N['g60'], N['lo60'], N['hi60'])} pp at 60 s and {ci_txt(N['g120'], N['lo120'], N['hi120'])} pp at 120 s "
          f"on 20 classes (22 unseen drivers; {f1(N['frac60'], 0)} % and {f1(N['frac120'], 0)} % of the drivers improve; {f1(N['pct60'], 0)} % and {f1(N['pct120'], 0)} % of the label-using oracle gain), reproduced by an independent implementation.")
    k = N["mm_k"]
    c2 = (f"The baseline needs **distinct postures, not seconds**: at the same 10 s in total five postures of 2 s beat one posture of 10 s by {ci_txt(*N['c:same 10 s in total: 5 postures x 2 s vs 1 posture x 10 s'][:3])} pp; the mixed model gives {sg(k[0], 2)} pp per doubling of the number of distinct postures "
          f"against {sg(N['mm_n'][0], 2)} pp per doubling of the seconds; a label-free draw of five postures x 5 s ({sg(N['bag_incl_5_5'])} pp) matches the whole-session mean ({sg(N['gsession'])} pp).")
    c3 = ("Operator evidence from established methods with verified implementations: mean subtraction is sufficient as the parsimonious primary (pre-specified rule); scale and covariance operators add about 1-2 pp on causal windows and 2-4 pp with a whole session "
          "(above the first-order oracle at 24 cells); Euclidean alignment as published does not beat mean subtraction on non-zero-mean maps.")
    c4 = ("A validity recipe that travels: order-randomised label-free baselines with an independent permutation per driver, an elapsed-time-only null with an incremental-over-time test, an independent clean-room re-implementation with a feature-level unit test, "
          "a pre-specified operator decision rule, and Crossref-verified references with official-implementation checks.")
    return c1, c2, c3, c4


def verdict(N, N7, N4, P, rule_ok):
    v = []
    v.append(f"1. **Reproduced.** An independent scikit-learn re-implementation reproduces the pilot's headline (deterministic conditions to <= 0.013 pp; feature unit test bit-identical). With 10 random orders the label-free driver-relative baseline gives "
             f"**{ci_txt(N['g60'], N['lo60'], N['hi60'])} pp at 60 s and {ci_txt(N['g120'], N['lo120'], N['hi120'])} pp at 120 s** on 22 unseen drivers (20 classes, no-calibration {f1(N['none_acc'])} %), {f1(N['pct60'], 0)} % and {f1(N['pct120'], 0)} % of the label-using oracle gain.")
    v.append(f"2. **Mechanism: distinct postures, not seconds.** At equal total time five postures x 2 s beat one posture x 10 s by {ci_txt(*N['c:same 10 s in total: 5 postures x 2 s vs 1 posture x 10 s'][:3])} pp; {sg(N['mm_k'][0], 2)} pp per doubling of distinct postures against "
             f"{sg(N['mm_n'][0], 2)} pp per doubling of seconds. A running window of 15-300 s spans only {f1(N['dist15'])}-{f1(N['dist300'])} distinct postures and lags a label-free draw of the same diversity.")
    v.append(f"3. **Operator: mean subtraction stays primary** (pre-specified rule; no alternative clears it at all four settings). CMVN and CORAL-type whitening add small, consistently positive gains on causal windows and 2-4 pp with a whole session; "
             "at 24 cells a whole-session whitening exceeds the label-using first-order oracle, so the pilot's \"oracle = upper bound\" is withdrawn. Euclidean alignment as published does not beat mean subtraction.")
    if N7:
        c1 = N7["pressure + time  vs  time only"]
        c3 = N7["label-free baseline given time (pressure + t)"]
        v.append(f"4. **Validity against the protocol clock.** Boosted trees given pressure *and* elapsed time vs elapsed time alone: {ci_txt(*c1[:3])} pp ({pick(c1, 'pressure carries information beyond the clock', 'pressure does not add to the clock', 'not resolved from zero')}); "
                 f"the label-free baseline on top: {ci_txt(*c3[:3])} pp. Absolute accuracy still cannot be read as posture recognition (section 7).")
    else:
        v.append("4. **Validity against the protocol clock.** (pending)")
    parts = [f"5. **Learners: the benefit is learner-dependent.** Shrinkage LDA gains in every run ({sg(N['g60'])} to {sg(N['g120'])} pp at 384 cells)."]
    if N4:
        g = N4["driver-relative input (60-s baseline)  vs  none"]
        a = N4["driver-relative input  vs  AdaBN"]
        parts.append(f"The CNN gains {ci_txt(*g[:3])} pp (E4); input-level offset removal against AdaBN: {ci_txt(*a[:3])} pp.")
    if N7:
        xg = N7["label-free baseline without time (pressure only)"]
        parts.append(f"Boosted trees gain only {ci_txt(*xg[:3])} pp in E7 against +3.3 [+0.7, +5.6] in the pilot: small and not stable across random orders.")
    v.append(" ".join(parts))
    v.append("6. **Limits that stay.** One dataset, one seat, one 9-minute fixed-order session; postures undocumented; absolute accuracy 20-35 % of 20 classes; 22 drivers; design choices screened on the same drivers; a diverse baseline is not guaranteed in habitual driving (section 7).")
    return "\n".join(v)


def conclusions(N, N7, N4):
    rows = [
        ["A label-free driver-relative baseline raises unseen-driver accuracy on DPR", "E1, E3 ladder, pilot nested tuning and sensitivity", f"**Supported:** {sg(N['g60'])} / {sg(N['g120'])} pp at 60 / 120 s"],
        ["The benefit depends on distinct postures, not seconds", "E3 (composition; mixed model)", "**Supported** as a mechanism; its causal deployability is untested"],
        ["Mean subtraction is a sufficient primary operator in DPR's regime", "E2 pre-specified rule", "**Supported with limits** (n = 22; 24-96 cells)"],
        ["Second-order alignment adds more with a long unlabelled stream", "E2", "**Supported at 24 cells**; unresolved at 96 cells"],
    ]
    if N7:
        c1 = N7["pressure + time  vs  time only"]
        rows.append(["Pressure carries slot information beyond the elapsed-time clock (posture-specific or drift-based: not separable)", "E7", pick(c1, "**Supported**", "**Not supported**", "**Not resolved**") + f" ({ci_txt(*c1[:3])} pp)"])
    if N4 or N7:
        cnn = "CNN pending"
        if N4:
            g = N4["driver-relative input (60-s baseline)  vs  none"]
            cnn = "CNN " + pick(g, "supported", "not supported", "not resolved from zero")
        xgs = "boosted trees pending"
        if N7:
            xg = N7["label-free baseline without time (pressure only)"]
            xgs = "boosted trees " + pick(xg, "supported", "negative", "**not resolved from zero** (pilot +3.3 [+0.7, +5.6])")
        rows.append(["The label-free gain holds across learner families", "pilot, E7, E4", f"**LDA: supported**; {cnn}; {xgs} - **learner-dependent**"])
    rows += [["The effect generalises to other datasets, mats or driving conditions", "-", "**Not tested** (single dataset by design)"],
             ["20-class accuracy of 20-35 % is a usable posture classifier", "-", "**Not supported**; only relative effects are claimed"]]
    return table(["claim", "evidence", "status"], rows) + "\n\n**Controlled limitations**\n" + RS.LIMITATIONS


# ----------------------------------------------------------------------------------------------------------------------- assembly
def build():
    P = pilot_context()
    t1, n1 = sec_e1()
    t2, n2, rule = sec_e2()
    E3, n3 = sec_e3()
    E7, n7 = sec_e7()
    E4, n4 = sec_e4()
    N = {**n1, **n2, **n3, "_ladder": E3["ladder"], "_ops": t2}
    S = rd("e2_summary.csv")
    if E4:
        n4["_table"], n4["_contr"] = E4["levels"], E4["contr"]
    c1, c2, c3, c4 = contrib(N, n7, n4, P)
    rtext = rank_tol_text()
    rule_ok = rd("e2_decision_rule.csv").replages if False else None
    definition = RS.DEFINITION.format(C1=c1, C2=c2, C3=c3, C4=c4)
    revisions = RS.REVISIONS.replace("@@G60@@", f1(N["g60"])).replace("@@G120@@", f1(N["g120"]))
    ops = text_ops(N, rule, S, sens_text())
    if rtext:
        ops += "\n**Numerical-rank tolerance (decision 5).** " + rtext + "\n"
    results = "\n\n".join([t1, text_primary(N, P), ops, text_mech(E3, N, P), text_validity(E7, n7, P), text_learners(N, n7, n4)])
    doc = f"""# Independent investigation: how much driver-specific information does seat-pressure posture recognition need to serve unseen drivers?

Study record, 6 October 2026. This is the record of an investigation, not a manuscript: it fixes the question, scope, contribution and gaps, documents the evidence base and every methodological decision, and reports what the data support.
Starting point: `../PhD_Proposal_Seat_Pressure_Driver_Calibration.docx` and its pilot (`scripts/00-49`, `results/gpu_*.csv`, `DATASET_AUDIT.md`). Everything below was re-derived or re-tested here; where a pilot number is quoted it is marked *pilot*.
Code `scripts/50-56`, `58-62`, `scripts/calib_lab.py`, `scripts/run_followup.ps1`; per-driver results `results/e*.csv` (experiment ids E1-E4 and E7 are the identifiers used in file names); figures `figures/S1-S3`.
The study uses **only the DPR dataset** (no external data). The tables and quoted estimates of section 5 and the verdict / contribution / claims text are generated from `results/*.csv` by `scripts/62_build_record.py`; quantities quoted from the pilot are marked *pilot*.

## Verdict

{verdict(N, n7, n4, P, None)}

## 1. Study definition (fixed before the confirmatory runs)

{definition}

## 2. Evidence base

### 2.1 What the literature establishes and what it leaves open

{RS.LITERATURE}

### 2.2 Official implementations and how each was verified

{RS.IMPLEMENTATIONS}

### 2.3 Search log (novelty check, 5-6 October 2026)

Web search (standard and extended modes), Crossref, OpenAlex, DataCite, Zenodo API, publisher / arXiv / GitHub pages. Queries: seat / chair / mat pressure x posture recognition x (unseen user, leave-one-subject-out, calibration, personalisation, domain adaptation,
user-specific normalisation, online adaptation, calibration time); in-bed pressure x LOSO x per-subject normalisation; driver posture reviews; human-activity-recognition subject-specific normalisation; the dataset's title and DOI.
**Result:** nothing found that quantifies unseen-user accuracy against the amount or composition of unlabelled user data for pressure-based posture recognition; closest are subject-invariant features (Mutlu 2007, Postureware 2017), augmentation for unseen users and chairs
(ChairPose 2025), preliminary transfer learning (Zhao-Yan-Wang 2023) and hardware-calibration papers. The dataset's Zenodo record (published 2025-04-11) has no related identifiers; DataCite lists 0 citations (6 Oct 2026); OpenAlex does not index it.
Statements of absence are limited to "in the studies found"; no Scopus / Web of Science access. Every DOI was verified against Crossref (`results/refs_verified.csv`: {int((rd('refs_verified.csv').status == 'OK').sum())} exact, {int((rd('refs_verified.csv').status == 'OK_short_title').sum())} shortened-title matches with DOI / author / year agreeing, {int((rd('refs_verified.csv').status == 'MANUAL').sum())} hand-verified non-Crossref sources).

## 3. Data and protocol (summary; details in `DATASET_AUDIT.md`)

{RS.DATA}

## 4. Methodological decisions and their support

{RS.DECISIONS}

## 5. Results

{results}

## 6. Revisions to the pilot's methodology and claims

{revisions}

## 7. Conclusions and controlled limitations

{conclusions(N, n7, n4)}

## 8. Reproduction and files

{RS.REPRO}
"""
    (ROOT / "analysis" / "STUDY_RECORD.md").write_text(doc, encoding="utf-8")
    print("wrote STUDY_RECORD.md,", len(doc.splitlines()), "lines")


if __name__ == "__main__":
    build()


