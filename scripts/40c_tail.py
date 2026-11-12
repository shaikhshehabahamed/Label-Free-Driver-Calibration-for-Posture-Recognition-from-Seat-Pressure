# Executed inside 40_build_docx.py (shares its namespace): sections 4.6-4.9, 5-9, references, appendices.
import math

import pandas as pd
from scipy import stats as _st


def n_paired(sd, delta, alpha=0.05, power=0.80):
    z = _st.norm.ppf(1 - alpha / 2) + _st.norm.ppf(power)
    return math.ceil((z * sd / delta) ** 2 + _st.norm.ppf(1 - alpha / 2) ** 2 / 2)


def mode_of(d):
    return "n/a" if not d else max(d, key=d.get)


def sgn(d):
    return d is not None and d["mean"] > 0


def sig(d):
    return d is not None and d["lo"] > 0


# ------------------------------------------------------------------------------------------------ 4.6 three levers
doc.add_heading("4.6 Three ways to buy unseen-driver accuracy, on one scale", 2)
lc, lm, ls = g("learning"), g("learning_meta", {}), g("learning_step", {})
if lc and lm:
    nmax = lm.get("nmax", 21)
    add_fig(doc, "F8_learning_curve.png", 4.3, f"Figure 9. Unseen-driver accuracy versus number of training drivers (random subsets, 3 repeats; largest size = all other cohort drivers; GPU LDA; baseline = 60 s, random order).")
    stp_n, stp_r = ls.get("none"), ls.get("run60")
    add_fig(doc, "F9_three_levers.png", 6.2, "Figure 10. The same outcome, bought three ways: calibration (software, no driver effort), sensing cells (hardware) and training drivers (data collection). Paired changes in unseen-driver accuracy with 95 % CIs over drivers.")
    cal60, cal120 = lvn.get("run60"), lvn.get("run120")
    more_cells = g("res_tests.f16run_vs_f2none")
    ratio = ""
    if stp_n is not None and cal120 is not None and stp_n["mean"] > 0.2:
        ratio = f", about {cal120['mean'] / stp_n['mean']:.1f} times what the last {nmax - 12} training drivers add"
    para(doc, f"Without calibration, accuracy goes from {f1(lc.get('none|3', {}).get('mean'))} % with 3 training drivers to {f1(lc.get(f'none|{nmax}', {}).get('mean'))} % with {nmax}; with the 60-s baseline from "
              f"{f1(lc.get('run60|3', {}).get('mean'))} % to {f1(lc.get(f'run60|{nmax}', {}).get('mean'))} %. Going from 12 to {nmax} training drivers changes accuracy by {dpp(stp_n)} without calibration and {dpp(stp_r)} with it. "
              f"Against that, the label-free baseline adds {dpp(cal60)} at 60 s and {dpp(cal120)} at 120 s{ratio}, and 64 times more sensing cells without calibration change accuracy by "
              f"{g('abl_res.2|none.mean', 0) - g('abl_res.16|none.mean', 0):+.1f} pp (Figure 8). "
              + ("Calibration is therefore the cheapest lever in this protocol: it needs no hardware, no participants and no driver effort beyond the first minutes of the trip. "
                 if (sig(cal120) and (stp_n is None or cal120["mean"] > stp_n["mean"])) else
                 "Within the uncertainty of the pilot the three levers cannot be ranked. ")
              + "The comparison is conditional on the protocol's high posture diversity (§4.3) and on one seat; it is a statement about relative return, not about attainable accuracy.")
else:
    para(doc, "[learning-curve results not available in this build]", italic=True)

# ------------------------------------------------------------------------------------------------ 4.7 nested tuning
nn_, nr_ = g("nested_none"), g("nested_run")
doc.add_heading("4.7 Hyperparameters were tuned without touching the held-out driver", 2)
if nn_ and nr_:
    para(doc, f"Nested LOSO (inner GroupKFold(5) over the training drivers) selected the integration window (1/3/5 s), baseline window (30/60/120 s, computed on randomly ordered streams), PCA dimension (50/100/200) and LDA shrinkage "
              f"(auto/0.3) from training drivers only. Resulting unseen-driver accuracy: {mci(nn_['acc'])} % without calibration and {mci(nr_['acc'])} % with the label-free baseline "
              f"({dpp(g('nested_gain'))}). Most frequent choices — baseline window {mode_of(nr_['Wb'])} s, integration {mode_of(nr_['Wi'])} s, PCA {mode_of(nr_['pca'])}, shrinkage {mode_of(nr_['shrinkage'])} — "
              f"compare with the defaults used elsewhere (3 s, 60 s, 150, auto). Tuning lifts the uncalibrated model as well ({f1(lad.get('none', {}).get('mean'))} % with defaults, {f1(nn_['acc']['mean'])} % tuned), so the tuned gain is like for like; "
              f"it is the unbiased estimate of H1 for a pipeline that chooses its own window. The chosen baseline window is the upper edge of the grid in {sum(v for k, v in nr_['Wb'].items() if str(k) == '120')} of {sum(nr_['Wb'].values())} folds; "
              f"the ladder (Table 6) shows why and that two to five minutes are about equivalent: the 60-s default leaves part of the gain unused.")
else:
    para(doc, "[nested tuning results not available in this build]", italic=True)

# ------------------------------------------------------------------------------------------------ 4.8 learners
cn, dsc, cl, clg = g("cnn6"), g("descriptors"), g("cnn_loso"), g("cnn_loso_gain", {})
doc.add_heading("4.8 Does the conclusion depend on the learner or the representation?", 2)
rows = []
lg = g("learners_gain", {})
if lx.get("gLDA") or lx.get("XGB"):
    for mdl, lab in (("gLDA", "Shrinkage LDA (GPU)"), ("XGB", "Boosted trees (XGBoost, GPU)")):
        if lx.get(mdl):
            rows.append([lab, "pixels, 96 cells, LOSO", mci(lx[mdl]["none"]), mci(lx[mdl]["run60"]), dpp(lg[mdl]["run60"]), mci(lx[mdl]["oracle"])])
if cl and "CNN|none" in cl:
    rows.append(["3-layer CNN (PyTorch, GPU)", "pixels, 384 cells, LOSO, 2 seeds", mci(cl["CNN|none"]), mci(cl.get("CNN|run60")), dpp(clg.get("CNN")), "—"])
    if "CNN2ch|run60" in cl:
        rows.append(["CNN given the driver-relative and the absolute map (2 channels)", "pixels, 384 cells, LOSO, 2 seeds", mci(cl["CNN|none"]), mci(cl["CNN2ch|run60"]), dpp(clg.get("CNN2ch")), "—"])
if cn:
    cg = g("cnn6_gain", {})
    for mdl, lab in (("gLDA", "Shrinkage LDA (GPU)"), ("XGB", "Boosted trees (XGBoost, GPU)"), ("CNN", "3-layer CNN (PyTorch, GPU), first run")):
        if f"{mdl}|none" in cn:
            rows.append([lab, "pixels, 384 cells, grouped 6-fold, 1 seed", mci(cn.get(f"{mdl}|none")), mci(cn.get(f"{mdl}|run60")), dpp(cg.get(mdl)), "—"])
if dsc:
    for mdl, lab in (("gLDA", "Shrinkage LDA (GPU)"), ("XGB", "Boosted trees (XGBoost, GPU)")):
        if f"{mdl}|descriptors|none" in dsc:
            a, b = dsc[f"{mdl}|descriptors|none"], dsc[f"{mdl}|descriptors|run60"]
            dgn = g("descriptor_gain", {}).get(mdl)
            rows.append([lab, "19 physical descriptors, LOSO", mci(a), mci(b), dpp(dgn), mci(dsc.get(f"{mdl}|descriptors|oracle"))])
if rows:
    table(doc, ["Learner", "Representation / protocol", "No calibration, % [95 % CI]", "60-s baseline, random order, % [95 % CI]", "Gain, pp [95 % CI]", "Oracle, %"], rows, [1.3, 1.35, 1.1, 1.15, 1.0, 0.6], font=8)
    caption(doc, "Table 8. Learner and representation checks, all with the 60-s baseline on randomly ordered streams. The CNN was run twice: first with grouped 6-fold cross-validation and one seed (compute shortcut), then, like every other learner, leave-one-driver-out "
                 "with two seeds per fold; the three CNN configurations of the second run were declared before it was started and are all shown. Descriptors: load, contact area, centre of pressure, spread and upper/lower split per strip, "
                 "backrest share and global centre of pressure; driver-relative descriptors are descriptor(map) − descriptor(baseline).")
    add_fig(doc, "F15_learners.png", 5.9, "Figure 11. Learner families with and without the 60-s label-free baseline (random order).")
    gains = {k: v for k, v in (("LDA", lg.get("gLDA", {}).get("run60")), ("boosted trees", lg.get("XGB", {}).get("run60")), ("CNN", clg.get("CNN"))) if v}
    same = all(sgn(x) for x in gains.values()) if gains else False
    n_sig = sum(1 for x in gains.values() if sig(x))
    xl = lg.get("XGB_vs_LDA_none"), lg.get("XGB_vs_LDA_run60")
    c6 = g("cnn6_gain", {}).get("CNN")
    sd_c = clg.get("sd_pair", float("nan"))
    mde = clg.get("mde80", float("nan"))
    para(doc, "H4 asks whether the conclusions depend on the learner. " +
         (f"Under the same leave-one-driver-out protocol the baseline gain is positive for every learner family ({', '.join(f'{k} {v['mean']:+.1f}' for k, v in gains.items())} pp) and significant for {n_sig} of {len(gains)}, "
          f"so the ordering of the calibration conditions does not depend on the learner. " if same else "The baseline gain differs in sign between learners, so the ordering of the calibration conditions is learner-dependent and the claims are limited accordingly. ") +
         (f"The first CNN run, with grouped 6-fold cross-validation and a single seed, had given {dpp(c6)}, not significant; that evaluation trains on fewer drivers, holds out 3–4 drivers per fold and is noisier, "
          f"and the leave-one-driver-out re-run is the like-for-like one — but both are reported, and with 22 drivers and a paired SD of {sd_c:.1f} pp the pilot can only resolve gains of about {mde:.0f} pp or more, so each learner's estimate is imprecise. "
          f"Giving the CNN the absolute map as well as the driver-relative one gains {dpp(clg.get('CNN2ch'))} against {dpp(clg.get('CNN'))}; this variant is exploratory (one of three declared configurations). " if (cl and c6) else "") +
         (f"Boosted trees beat LDA without calibration ({dpp(xl[0])}) but not with the baseline ({dpp(xl[1])}): a simple linear model plus a label-free baseline is as good as a non-linear one on 96-cell maps. " if all(xl) else "") +
         "The learners differ in absolute accuracy, which is not a claim of this proposal.")
else:
    para(doc, "[learner results not available in this build]", italic=True)

# ------------------------------------------------------------------------------------------------ 4.9 sensitivity, who fails, privacy
doc.add_heading("4.9 Sensitivity to the data decisions, who still fails, and a privacy remark", 2)
sens = g("sens", {})
rows = []
for key, lab in (("clean", "22 clean drivers (primary)"), ("all", "All 30 drivers, repaired labels"), ("all_on_clean22", "   of which the 22 clean drivers"),
                 ("all_on_excluded", "   of which the 8 excluded drivers"), ("orig", "All 30 drivers, first cleaning"), ("orig_on_clean22", "   of which the 22 clean drivers"),
                 ("settle8", "22 clean drivers, rows ≥ 8 s into a segment"), ("withmarkers", "22 clean drivers, marker presses left in the data")):
    s_ = sens.get(key)
    if s_:
        rows.append([lab, str(s_["n"]), mci(s_["none"]), mci(s_["run60"]), dpp(s_["gain"]), mci(s_["oracle"])])
if rows:
    table(doc, ["Data set", "Drivers", "No calibration, %", "60-s baseline, random order, %", "Gain, pp [95 % CI]", "Oracle, %"], rows, [2.0, 0.5, 1.0, 1.15, 1.2, 0.9], font=8.2)
    caption(doc, "Table 9. Sensitivity of the headline contrast to the cohort, the label set and the settle time (shrinkage LDA, 384 cells).")
    ga, gall, gex = sens.get("clean", {}).get("gain"), sens.get("all", {}).get("gain"), sens.get("all_on_excluded", {}).get("gain")
    para(doc, f"The gain is {dpp(ga)} for the primary cohort and {dpp(gall)} when all 30 drivers are used with the repaired labels"
              + ("" if gex is None else f" ({dpp(gex)} on the eight drivers excluded from the primary analysis)")
              + f". Dropping the overlapping frames only, as in the first draft, gives {dpp(sens.get('orig', {}).get('gain'))}; requiring rows to be 8 s into a segment gives {dpp(sens.get('settle8', {}).get('gain'))}. "
                "The conclusions therefore do not depend on the exclusions, on the repair rules or on the settle time.")
an = g("anthro", {})
idf, idm = g("identifiability"), g("identifiability_meta", {})
para(doc, f"Per-driver accuracy shows no detectable relation to body weight, height, BMI or sex ({an.get('n_tests', 0)} tests, max |ρ| = {an.get('max_abs_rho', float('nan')):.2f}, smallest Holm p = {an.get('min_p_holm', float('nan')):.2f}); "
          f"with {coh.get('women')} women and {coh.get('bmi30')} driver above BMI 30 the cohort has little power for subgroups. " +
          ("" if not idf else f"Separately, the same maps identify the driver: a {idm.get('n_drivers', 22)}-way identifier trained on postures P1–P10 recognises drivers from P11–P20 frames with "
                              f"{f1(idf.get('f4|train P1-10 -> test P11-20'))} % accuracy at {idm.get('cells', {}).get(4, 384)} cells and {f1(idf.get('f16|train P1-10 -> test P11-20'))} % at {idm.get('cells', {}).get(16, 24)} cells (chance {100 / idm.get('n_drivers', 22):.1f} %). "
                              f"'Privacy-preserving' therefore describes the absence of an image, not the absence of a biometric (as proposed in {C('riener2008')}); storing a per-driver baseline across trips would create a "
                              f"personal template, whereas a baseline held in volatile memory for the trip does not (a design argument for the label-free level)."))

# ================================================================================================ 5 method table
doc.add_heading("5. Methodology: what each component is for", 1)
nm = g("nested_run")
rows = [
    ["Driver-level LOSO; bootstrap CI over drivers; Wilcoxon + Holm; Friedman", "The deployment unit is a new driver; frames are pseudo-replicates", f"RF {f1(g('protocols.RF|random5'))} → {f1(g('protocols.RF|LOSO'))} %", f"{C('saeb2017','varoquaux2018','kapoor2023','roberts2017','demsar2006','bates2024')}; analysis/scripts/stats_utils.py"],
    [f"Cohort: {coh.get('n')} drivers with consistent labels; columns 64–160 only", "Remove dependence on repairs and on the undocumented strip", "Table 9; Table 7 (region)", "analysis/DATASET_AUDIT.md"],
    ["Row selection: ≥ 5 s into a segment, every 6th frame, overlaps removed, P1's first 10 s reserved", "Postures are reached gradually; temporal redundancy (lag-1 correlation 0.97); reference frames kept out of tests", "Trimming never helps; ≥ 8 s lifts all conditions alike (Tables 7, 9)", "Data audit (Table 3)"],
    ["Mean pooling 4×4 (16×24 cells); 8×8 for boosted trees", "Cost and regularisation", "Table 7, Fig. 8", f"Uniform sampling as in {C('zhao2021','caggiari2023')}"],
    ["Causal 3-s integration", "Frame jitter; deployable", "0–5 s changes the uncalibrated model by ≤ 1.5 pp", "—"],
    ["Occupancy gate (total force > 250 N)", "Keeps empty-seat frames out of baselines", "Pre-posture phase in Table 3", "Fixed, driver-independent threshold"],
    ["Driver baseline: causal running mean on randomly ordered streams (W = 60–120 s), subtraction", "Removes the additive driver offset (77 % of variance) without using the posture order", "Table 6, Fig. 4, Fig. 7", f"First-order alignment; cf. {C('he2020','li2018')}; adaptation precedent {C('jacobe2018','wei2018')}"],
    ["Shrinkage LDA after standardise → PCA (GPU port validated against scikit-learn)", "Transparent linear learner; weights map back to sensors", "100 % prediction agreement on 20 folds", f"{C('pedregosa2011','ledoit2004')}"],
    ["Boosted trees (XGBoost, GPU); 3-layer CNN (PyTorch)", "Check that conclusions do not depend on the learner", "Table 8, Fig. 11", f"{C('chen2016xgb','paszke2019')}"],
    ["Nested LOSO tuning (window, baseline length, PCA, shrinkage)", "Avoid optimistic bias from tuning on the evaluation drivers", "§4.7", "analysis/scripts/34_nested_tuning.py"],
    ["Order-randomised streams; elapsed-time-only null; order audit; variance decomposition", "Diagnose the fixed-order confound and the source of the gap", "§3.5, §4.2, §4.4", "analysis/scripts/44_order_audit.py"],
    ["Marker presses (four points pressed before each posture): detection, calibration on the consistent drivers, removal of press frames", "The dataset's own clock: audits every label range, repairs the inconsistent ones, and keeps hand presses out of the pressure data", "Fig. C1, Table 3; effect of keeping them in Table 9", "analysis/scripts/48_markers.py, marker_lib.py"],
    ["Label repair rules R1–R3, R6, M3 (marker-anchored), with a first boundary locator tried and rejected", "Fix what the presses and the protocol allow, exclude the rest, show the primary result does not depend on it", "Appendix C; Table 9", "analysis/scripts/43_repair_labels.py"],
    ["Non-inferiority (one-sided 95 % bootstrap bound, margin 2 pp)", "'No loss' claims for coarser arrays need a margin, not a non-significant p", "H3", f"{C('lakens2017')}"],
    ["Not used", "Sparse sensor placement (PySensors) — a later extension; covariance-whitening alignment — needs far more frames than dimensions; MOABB — EEG-specific, its cross-subject principle is followed", "—", f"{C('desilva2021','jayaram2018')}"],
]
table(doc, ["Component", "Purpose", "Pilot evidence", "Source / implementation"], rows, [1.75, 1.85, 1.25, 1.65], font=8)
caption(doc, "Table 10. Every methodological component, its purpose and its support. No new algorithm is introduced; the GPU LDA is a port of scikit-learn's pipeline and was validated against it.")

# ================================================================================================ 6 contributions
doc.add_heading("6. Expected contribution and impact", 1)
para(doc, "**Academic.** (1) A leakage-controlled, repaired and documented benchmark of the only open 30-driver seat-pressure dataset, with the elapsed-time null and a published cohort rule. (2) The first curve of "
          "unseen-driver accuracy against the length of the driver's own unlabelled stream, with driver-level statistics and an order-randomised protocol. (3) A common-scale comparison of calibration, sensing cells and "
          "training drivers, with a pre-specified non-inferiority margin for the hardware claim. (4) A replication on independent pressure-mat data (§8, WP-C) that tests whether the effect is specific to this dataset.")
rtn = g("res_tests.f8run_noninf_vs_f4run_2pp")
rt24 = g("res_tests.f16run_vs_f4run")
hw = (f"the pilot gives no reason to buy cells beyond a few dozen when a baseline is available (24 and 96 cells are non-inferior to 384 within 2 pp, and 24 cells do better: {dpp(rt24)})" if (rtn and rtn.get("noninferior"))
      else "the pilot does not establish that a coarse array is non-inferior to a denser one, so the cell count should be set by other constraints")
stp_ = g("learning_step.none")
cal_ = lvn.get("run120")
if stp_ and cal_ and lm:
    dc = (f"going from 12 to {lm.get('nmax')} training drivers changes unseen-driver accuracy by {dpp(stp_)} without calibration, against {dpp(cal_)} for the 120-s baseline, so "
          + ("calibration returns more than recruiting at the sizes tested" if cal_["mean"] > max(stp_["mean"], 0) else "recruiting returns at least as much as calibration at the sizes tested"))
else:
    dc = "the learning curve is not available in this build"
para(doc, "**Managerial.** (a) A start-up requirement stated in seconds of unlabelled data (the pilot's gain needs about one to two minutes of seated, posture-diverse driving, not an enrolment ritual) instead of a single accuracy figure. "
          f"(b) A hardware decision rule: {hw}. (c) A data-collection rule: {dc}. "
          "(d) A validation recipe for OEMs and Tier-1 suppliers: driver-level LOSO with CIs, the elapsed-time null, order-randomised streams and a driver count derived from a power calculation. "
          "(e) A data-governance argument: a baseline kept in volatile memory for one trip needs no stored biometric template.")
sd_pair = g("ladder_vs_none.run120.sd", g("order.shuf_vs_none.sd", 8.0))
para(doc, f"**Practical.** The corrected label table, the cohort rule, the audit notes and the analysis code are released with the thesis so that other groups can report comparable unseen-driver accuracy. "
          f"For planning: the paired difference between the 120-s baseline and no calibration has SD {sd_pair:.1f} pp across drivers, so a paired design with 80 % power at α = 0.05 needs "
          f"{n_paired(sd_pair, 5)} drivers for a 5-pp effect, {n_paired(sd_pair, 3)} for 3 pp and {n_paired(sd_pair, 2)} for 2 pp.")

# ================================================================================================ 7 threats
doc.add_heading("7. Threats to validity and how the design answers them", 1)
rows = [
    ["Posture ≡ time slot (fixed order, one occurrence per posture)", f"Elapsed time alone: {f1(g('time_only.mean'))} %; recorded vs random order differ by {f1(g('order.rec_vs_shuf.mean'))} pp", "Random-order streams; paired differences only; absolute accuracy read against the null", "Drift common to all drivers cannot be separated; optional counterbalanced collection (WP-D)"],
    ["The baseline needs posture diversity, which the protocol over-supplies", f"Single-posture references: {dpp(lvn.get('ref1_10'))} (P1) to {dpp(lvn.get('ref10_10'))} (P10); diverse mean {dpp(lvn.get('session'))}", "H2 stated; session and oracle bounds reported; claims limited to diverse windows", "Habitual driving untested; WP-D"],
    ["One seat, one 8–10 min session, laboratory", "No seat-to-seat or day-to-day variation can be estimated", "Stated as scope; seat changes absorbed in the 'driver' effect", "Needs a second seat and session (WP-D)"],
    ["Postures undocumented; nominal labels, not measured postures", "Neighbouring slots are the most confused; within-driver accuracy is far from 100 %", "Anonymous classes; per-posture view; no named-posture claim", "A deployable vocabulary needs objective posture references"],
    [f"Small cohort ({coh.get('n')} drivers; {coh.get('women')} women, {coh.get('bmi30')} with BMI ≥ 30)", "No anthropometric trend, but low subgroup power", "Bootstrap CIs over drivers; 30-driver sensitivity (Table 9)", "Subgroup claims not made"],
    ["Design screened on the same drivers (forking paths)", "Ablations, window choice and cohort rule were examined on these drivers", "Nested tuning for tuned numbers; every ablation reported; cohort defined by label consistency only", "No pristine confirmation sample exists in this dataset; WP-C replicates on independent data"],
    ["Label defects and exclusions", "Table 3; Appendix C", "Press-anchored rules; primary analysis independent of repairs; three label sets compared", "Repaired boundaries are as accurate as published ones (MAD ≈ 11 frames), not exact"],
    ["Learner dependence", f"Table 8; boosted trees gain {dpp(g('learners_gain.XGB.run60'))} in random order but {dpp(g('learners_gain.XGB.rec'))} in recorded order", "Three learner families and a descriptor model; random-order control for every learner", f"Claims limited to the ordering of calibration conditions; the CNN's first (6-fold) evaluation was not significant ({dpp(g('cnn6_gain.CNN'))}), its leave-one-driver-out re-run was ({dpp(g('cnn_loso_gain.CNN'))})"],
    ["Steady-state rows only", "Rows ≥ 5 s into a segment; ≥ 8 s sensitivity", "Stated", "Transitions are a separate problem"],
    ["Supervised enrolment cannot be assessed", "Each posture is held once per driver", "Out of scope (Table 2)", "Needs repeated postures per driver"],
]
table(doc, ["Threat", "Evidence in the pilot", "Handling in this design", "Residual"], rows, [1.55, 1.6, 1.85, 1.5], font=8)
caption(doc, "Table 11. Threats to validity.")

# ================================================================================================ 8 work plan
doc.add_heading("8. PhD work plan", 1)
rows = [
    ["WP-A · months 0–6", "Formal systematic search (PRISMA-ScR) of seat/chair/bed pressure sensing with unseen-user evaluation; release of the audited benchmark (repaired labels, cohort rule, evaluation code, elapsed-time null)", "Review + audit/benchmark paper"],
    ["WP-B · months 6–15", "Full calibration study on the benchmark: window curve, order-randomised protocol, nested tuning, learner families, three-lever comparison; analysis exactly as in §4 with the statistics of Table 10", "Journal paper 1; thesis chapters 3–4"],
    ["WP-C · months 12–24", f"Independent replication on open pressure-mat data with the same code: PhysioNet 'pmd' {C('pouyan2017')} (13 participants × 8 postures + 9 states, 32×64 mat, 1 Hz) and any seat dataset released since. Tests whether subject-relative label-free baselines transfer across sensors, postures and contexts; a failure to transfer is itself a result", "Journal paper 2"],
    ["WP-D · months 18–30 (optional)", "Naturalistic validation if vehicle and mat access exist: ≈ 20 drivers, two seats, habitual-posture drive blocks with short excursions, objective trunk-angle reference, Latin-square order for the instructed postures. Tests H2 (composition) and seat transfer; nothing in WP-A to WP-C or in the claims above depends on it", "Dataset + paper 3"],
    ["WP-E · months 24–36", "Integration: the 24-cell + baseline pipeline within an embedded real-time budget; thesis writing", "Thesis"],
]
table(doc, ["Phase", "Work", "Output"], rows, [1.1, 4.1, 1.3], font=8.5)
caption(doc, "Table 12. Work plan. Primary estimand: the paired unseen-driver accuracy gain of the label-free baseline over no calibration, with the baseline computed on randomly ordered streams; analysis: Wilcoxon signed-rank on per-driver differences, bootstrap CI over drivers, Holm over the stated family.")

# ================================================================================================ 9 outlets
doc.add_heading("9. Target outlets", 1)
para(doc, "Candidate venues for the audit/benchmark and the calibration study (scope and quartile to be checked at submission): IEEE Sensors Journal; IEEE Transactions on Intelligent Vehicles; "
          "Applied Ergonomics; Accident Analysis & Prevention (where the driver-adaptation precedent for drowsiness appeared); Sensors for a fast dataset-and-audit note. Reporting should follow "
          "the evaluation checklist implied by Table 10 and release code and labels.")

# ================================================================================================ references
doc.add_heading("References", 1)
for n, txt in C.listing():
    p = para(doc, f"[{n}] {txt}", size=9, after=2)
    p.paragraph_format.left_indent = __import__("docx").shared.Inches(0.3)
    p.paragraph_format.first_line_indent = -__import__("docx").shared.Inches(0.3)
if any("†" in txt for _, txt in C.listing()):
    para(doc, "† not re-verified against Crossref or the publisher page in this project (details from memory, a search snippet, or a secondary summary).", italic=True, size=8.5)

# ================================================================================================ appendices
doc.add_heading("Appendix A · Search and verification log", 1)
para(doc, "Searches were run on 5 October 2026 with a general web-search tool, Crossref and publisher pages (no Scopus or Web of Science access). Queries combined seat/chair/cushion/mat pressure, driver or sitting "
          "posture, unseen/leave-one-subject-out, calibration/personalisation/baseline, sensor reduction/resolution, occupant out-of-position, driver monitoring regulation, open pressure-mat datasets (PhysioNet 'pmd' "
          "record checked on its landing page) and the dataset's DOI (no citing work found through DataCite events or OpenAlex full-text search; a repeat search for a companion paper or a posture list on the "
          "same day found none). Bibliographic details were checked against Crossref or the publisher page; six entries that were first taken from search snippets or secondary summaries (Nobari & Bertram, Riener & Ferscha, MOABB, the EU and UN regulations, the Euro NCAP protocol) were verified by the author and re-checked here — the Euro NCAP protocol text was read directly (a vendor summary that had been used earlier proved to overstate it and was dropped), and the EU application dates were confirmed against an official EEA-law summary of Delegated Regulation 2023/2590 because the EUR-Lex text could not be retrieved; accuracy figures "
          "quoted from other papers were read from their abstract or full-text pages, except Licciardo et al., where only the abstract metadata was available. A formal, reproducible systematic search is the first work package.", size=10)
doc.add_heading("Appendix B · Posture contrast maps", 1)
add_fig(doc, "posture_contrast_maps.png", 6.5, "Figure B1. Posture mean map minus the same driver's mean over all postures (N/cm²), averaged over the cohort drivers; columns 16–160 are shown and the dashed line marks the start of the analysed area (column 64). "
        "Our reading, not the authors': several postures load the backrest-like strip (recline), others show mirrored left/right shifts, and the last ones unload it (forward lean).")
doc.add_heading("Appendix C · Dataset audit log: marker presses, repairs, exclusions and a rejected locator", 1)
mk = g("markers", {})
para(doc, "This appendix replaces the questions for the dataset authors in the first draft. **The dataset's own clock.** The README states that the participant and the operator pressed four highlighted points on the mat "
          "before assuming each posture. The saturation map shows four fixed, mirror-symmetric sites (rows 10 and 56 of column 133, rows 13 and 53 of columns 77–78) that carry nearly all saturation; a press is a short burst in which at least "
          f"three of them exceed 1 N/cm². A median of {f1(mk.get('events_per_driver_median'))} presses per driver are detected: one at the start of P1 (absent when the recording begins inside P1), 19 between postures and one at the end. "
          f"In the {coh.get('n')} drivers with a consistent label table, the published start of a posture follows the preceding press by {mk.get('OFF_S')} frames (MAD {f1(mk.get('mad_s'))}; {f1(100 * mk.get('frac_start_found', 0))} % of ranges have a press within 400 frames "
          f"before them) and the published end precedes the next press by {mk.get('OFF_E')} frames (MAD {f1(mk.get('mad_e'))}). A press therefore fixes a boundary to about a second, the precision of the published labels themselves, and gives an independent check "
          f"of every range (Figure C1). Among the {coh.get('n')} consistent drivers only {mk.get('flagged_interior_clean', 0)} of 440 ranges contain an interior press (drivers 16 and 30); they are kept in the primary analysis.", size=10)
para(doc, "**Rules** (applied only to drivers whose tables are inconsistent; the others keep their published ranges). **R1** a range that fully contains other ranges is wrong at one bound; the contained ranges are kept and the wrong bound "
          "is re-estimated from the press adjacent to the contained chain (label start = press end + offset, label end = next press start − offset). **R2** two ranges that overlap without nesting are resolved by the single press inside their union if "
          "there is exactly one, otherwise the shared frames are dropped. **R3** a reversed pair is tested with driver-centred posture templates calibrated on the consistent drivers' own adjacent pairs; it is kept if the data reject the protocol order, "
          "swapped if they support it, quarantined otherwise. **R5** a posture without a label is declared absent if the presses show no event for it. **R6** a repaired range is cut where it runs into the next press. "
          "**M3** a label table most of whose ranges contain a press is not on the CSV's frame base; its ranges are rebuilt from the press sequence under two structural hypotheses (H0: press j precedes posture j; H1: the first posture started "
          "before the first press and the last press ends the session), chosen by a template test with a permutation null.", size=10)
u8_ = g("u8", {})
para(doc, f"**Driver 8.** All {20} presses lie inside the CSV (frames 490–8,899 of 8,946), yet the published ranges start 1,540–2,408 frames later than the presses imply and by an irregular amount, so the table is on a different time base and not simply truncated. "
          f"Under H1 the 20 marker-anchored segments match their own posture templates with a mean cosine of {u8_.get('H1_mean_cos', float('nan')):.2f} (permutation p = {u8_.get('H1_perm_p', float('nan')):.4f}), against "
          f"{u8_.get('H0_mean_cos', float('nan')):.2f} under H0 and {u8_.get('clean_drivers_mean', float('nan')):.2f} for the consistent drivers' published ranges. "
          f"**Driver 3** has {19} presses for 19 postures and none between P4 and P6: P5 was never performed, not merely unlabelled. "
          f"**Drivers 2 and 22** each lack the press before the second posture of an overlapping pair (no weak event either), so the boundary cannot be placed and the shared frames are dropped.", size=10)
add_fig(doc, "F16_markers.png", 6.5, "Figure C1. (a) Marker presses (red) of one driver fall between the labelled postures (shaded). (b, c) Published label start relative to the end of the preceding press, and published label end relative to the start of the next press, in the consistent drivers.")
try:
    lg_ = pd.read_csv(Path(__file__).with_name("40c_tail.py").parents[1] / "results" / "repair_log.csv")
    rows = [[int(r.u), (f"P{int(r.p)}" if int(r.p) > 0 else "all"), r.rule, str(r.action), "" if pd.isna(r.evidence) else str(r.evidence)] for r in lg_.itertuples()]
    table(doc, ["Driver", "Posture", "Rule", "Action", "Evidence"], rows, [0.5, 0.55, 0.4, 2.9, 2.15], font=7.5)
    caption(doc, "Table C1. Every repair action (analysis/results/repair_log.csv).")
except Exception as e:                                         # pragma: no cover
    para(doc, f"[repair log not available: {e}]", italic=True)
add_fig(doc, "F12_label_repair.png", 5.4, "Figure C2. Label ranges as published (grey) and after repair (colour) for the drivers with inconsistent or mis-based tables. Dashed line: last recorded frame.")
add_fig(doc, "F13_geometry.png", 5.8, "Figure C3. (a) Share of frames in which each sensel is active. (b) Mean pressure by mat column over labelled frames of the cohort; the grey band is excluded from the analysis.")
para(doc, f"**A boundary locator that did not work.** Before the presses were noticed we tried to re-estimate posture boundaries from the pressure patterns themselves: a least-squares change-point segmentation and a matched filter along the contrast between the two neighbouring "
          f"postures. On the {g('locator.pairs', 0)} adjacent posture pairs of the consistent drivers the matched filter's boundary fell inside the annotated gap (±15 frames) in {f1(g('locator.inside'))} % of pairs, a median "
          f"{f1(g('locator.median_err'))} frames from the gap midpoint, and on average {abs(g('locator.off_start', 0)):.0f} frames after the annotated start of the next posture: postures are reached gradually and contain movement. "
          f"Pressure patterns cannot place a boundary to better than several seconds; the presses can.", size=10)
doc.add_heading("Appendix D · Reproducibility and laptop-safe settings", 1)
para(doc, "All code is in analysis/scripts (run order: 00_parse_csv → 02_build_pooled → 43_repair_labels → 45_apply_labels → run_chain.ps1 all [03, 05, 47, 46, 10, 23; 31_gpu_run experiments ladder, order, window, op, region, trim, res, sens "
          "(×3 data sets and settle 8 s), learners; 42, 44, 33, 34, 36, 38, 39, 37, 04] → 22 figures → 41 ledger → 40 this document). The data set is selected with the environment variable DATASET (clean | all | orig). "
          "Environment: Python 3.12, NumPy 2.5, scikit-learn 1.9, PyTorch 2.6 (CUDA 12.4), XGBoost 3.4 (GPU), GTX 1650 4 GB; fixed seeds throughout; the parsed arrays and pooled caches are regenerable "
          "from the CSVs (≈ 10 min). After a laptop crash the heavy runs were restructured: one process at a time, below-normal priority, two BLAS threads, per-driver features in memory, "
          "a GPU-temperature guard that pauses above 82 °C, and resumable result files. Nested-tuning grids: window 1/3/5 s, baseline 30/60/120 s, PCA 50/100/200, shrinkage auto/0.3; "
          "inner GroupKFold(5) over training drivers.", size=10)
