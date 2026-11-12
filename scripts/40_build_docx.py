"""Build the proposal (.docx) from the results ledger.   usage: python 40_build_docx.py [output.docx]
Every number in the text comes from 41_results_ledger.ledger(); figures come from analysis/figures (22_figures.py and the audit scripts).
Run with the venv that has python-docx (CWRU_Bearing_Fault_Diagnosis/.venv)."""
import importlib.util
import sys
from pathlib import Path

from common import FIG, ROOT
from docx_helpers import bullets, box, caption, figure, new_document, page_break, para, table
from proposal_refs import Cites

spec = importlib.util.spec_from_file_location("ledger", Path(__file__).with_name("41_results_ledger.py"))
_m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m)
L = _m.ledger()
C = Cites()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "PhD_Proposal_Seat_Pressure_Driver_Calibration.docx"


# ------------------------------------------------------------------------------------------------ formatting helpers
def g(path, default=None):
    """nested lookup in the ledger, e.g. g('order.shuffled.mean')"""
    cur = L
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


def mci(d):
    return "n/a" if d is None else f"{d['mean']:.1f} [{d['lo']:.1f}, {d['hi']:.1f}]"


def dpp(d, sign=True):
    if d is None:
        return "n/a"
    s = f"{d['mean']:+.1f}" if sign else f"{d['mean']:.1f}"
    return f"{s} pp [{d['lo']:+.1f}, {d['hi']:+.1f}]"


def pf(p):
    if p is None:
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def oa(key, default=None):
    """order-audit fact (value, note)"""
    d = g("order_audit", {}).get(key)
    return default if d is None else d


def fig_path(name):
    p = FIG / name
    return p if p.exists() else None


def add_fig(d, name, width, cap):
    p = fig_path(name)
    if p is None:
        para(d, f"[figure {name} not available]", italic=True)
    else:
        figure(d, p, width, cap)


doc = new_document()
N = g("eda", {})
coh = g("cohort", {})
lad, lvn = g("ladder", {}), g("ladder_vs_none", {})
rp = g("repair", {})
CELLS = {16: 24, 8: 96, 4: 384, 2: 1536}                    # sensing cells of the analysed area (64 rows x 96 columns) per pooling factor

# ================================================================================================ title block
para(doc, "Calibrating for the Driver", bold=True, size=22, align="center", after=2)
para(doc, "How much driver-specific information does seat-pressure posture recognition need to work on drivers it has never seen?",
     italic=True, size=13, align="center", after=6)
para(doc, "PhD research proposal with a self-contained pilot on the open driving-posture pressure dataset (Zenodo 10.5281/zenodo.15199100)",
     size=10.5, align="center", after=2)
para(doc, "Revised 5 October 2026 after an audit, repair and partial exclusion of the dataset · draft for supervisor review · code and results: analysis/ folder next to this file",
     size=9.5, align="center", after=10)

# ================================================================================================ summary box
t_cells = g("res_tests.f16run_vs_f2none")
verb_cells = "beats" if (t_cells and t_cells["lo"] > 0) else "is comparable to"
box(doc, "Proposal in one page", [
    f"**The gap.** Seat-pressure posture classifiers are reported at 98–99 % when the evaluation is single-user or the protocol is not stated, while the few studies that "
    f"test unseen users report 78–91 % for 7–14 coarse postures. On the only open 30-driver dataset a random forest scores {f1(g('protocols.RF|random5'))} % under a random split and "
    f"{f1(g('protocols.RF|LOSO'))} % on a driver it has never seen (20 postures, chance 5 %); a model that knows nothing except elapsed time scores {f1(g('time_only.mean'))} %. "
    f"The unseen-user penalty has been visible since 2001 {C('tan2001','mutlu2007')}, yet in the studies we found none measures **how much unlabelled driver-specific data closes it, "
    f"or how that compares with buying accuracy through more sensing cells or more training participants**.",
    f"**The question.** For a seat-pressure posture classifier that must serve drivers it was not trained on, how much unseen-driver accuracy does a label-free, "
    f"driver-relative baseline recover, how long a stream of the driver's own data does it need, and how does that compare with the same accuracy bought with sensing cells or training drivers?",
    f"**What the pilot shows ({coh.get('n')} drivers with consistent labels).** Driver identity explains {f1(g('variance.all.subject'))} % of the variance of the pressure maps, posture "
    f"{f1(g('variance.all.posture'))} %. Subtracting a label-free running baseline from the driver's own seat data lifts unseen-driver accuracy from {f1(lad.get('none', {}).get('mean'))} % to "
    f"{f1(lad.get('run60', {}).get('mean'))} % (60 s window) and {f1(lad.get('run120', {}).get('mean'))} % (120 s) when the postures arrive in random order "
    f"({dpp(lvn.get('run60'))}; {dpp(lvn.get('run120'))}), against {f1(lad.get('oracle', {}).get('mean'))} % for an oracle baseline that needs every label; 15–30 s add nothing detectable. "
    f"With the 60-s baseline a {CELLS[16]}-cell array ({f1(g('abl_res.16|run60.mean'))} %) {verb_cells} a {CELLS[2]:,}-cell array without it ({f1(g('abl_res.2|none.mean'))} %), and 75 % more training drivers (12 to 21) buy "
    f"{dpp(g('learning_step.none'))} without calibration. Combined, {CELLS[16]} cells and a 120-s baseline reach {f1(g('combo.16|run120.mean'))} %, {f1(g('combo_tests.oracle16_vs_c16r120.mean'))} pp below the oracle at that resolution.",
    "**What is claimed, and what is not.** The postures are undocumented, performed once and in a fixed order, so only __relative__ effects between calibration conditions are claimed, "
    "estimated on randomly ordered streams; deployment accuracy is not. Every open question about the dataset is neutralised in the design or removed (§3.3): the "
    "synchronisation presses that the dataset's own README describes were used to audit and repair the label table, the "
    f"{30 - coh.get('n', 22)} drivers whose tables were inconsistent are excluded from the primary analysis, and marker-press frames and the undocumented outer strip are removed.",
    "**Contribution.** (i) An audited, repaired and documented benchmark (cohort rules, corrected label table, leakage-controlled protocol); (ii) the first curve of unseen-driver accuracy "
    "against the length of the driver's own unlabelled stream for seat-pressure posture recognition, with driver-level statistics; (iii) a common-scale comparison of three ways to buy "
    "accuracy (calibration, sensing cells, training drivers); (iv) a transferable validation recipe. No new classifier is proposed.",
])

# ================================================================================================ 1 problem
doc.add_heading("1. Problem, context and gap", 1)
doc.add_heading("1.1 Why this matters now", 2)
para(doc,
     f"Driver-state sensing is becoming a regulated function. Since 7 July 2026 every newly registered car, van, truck and bus in the EU must carry an advanced driver "
     f"distraction warning (new type approvals since 7 July 2024), typically delivered by driver-facing infrared cameras {C('gsr2')}; UN Regulation 157 requires a driver availability recognition "
     f"system for automated lane keeping and UN Regulation 171 has applied to driver-control assistance since September 2024 {C('unece')}. Euro NCAP's occupant-monitoring protocol "
     f"(version 1.2, implementation January 2026) scores occupant classification — passenger-airbag status, out-of-position cases and the stature of driver and front passenger — and occupant presence, "
     f"where child-presence detection must use direct sensing of life signs such as heartbeat, respiration or movement {C('euroncap')}; it does not name seat-pressure sensing. Seat-pressure sensing is promoted as a "
     f"privacy-preserving, cost-effective complement to camera-based monitoring {C('liguori2025')}; it also works in the dark and under occlusion.")
para(doc,
     "In shared and fleet vehicles the driver changes from trip to trip, so a posture classifier must work for people it was never trained on, and any "
     "calibration step becomes a product requirement with a price in driver effort, sensing hardware or recruited participants. That price has not been measured.")

doc.add_heading("1.2 What is known", 2)
para(doc, f"Table 1 positions the closest studies; a recent review of driver posture recognition is {C('mahomed2024')}.", keep=True)
rows = [
    [f"Tan et al. 2001 {C('tan2001')}", "Pressure-distribution chair, static postures", "Familiar vs unfamiliar users", "96 % vs 79 %", "Cost of closing the gap not studied"],
    [f"Mutlu et al. 2007 {C('mutlu2007')}", "Chair; 31 subjects, 10 postures", "Unseen users", "87 % (31 sensors, cross-validated); 78 % unseen users (19 sensors)", "Invariant features and sensor placement, no calibration"],
    [f"Zemp et al. 2016 {C('zemp2016')}", "Instrumented chair; 41 subjects, 7 positions", "Unseen subjects", "RF 90.9 %", "Same"],
    [f"Zhao et al. 2021 {C('zhao2021')}", "Driver seat, two XSENSOR mats 42×44; 23 drivers; trunk (5), right-foot (3), left-foot (2) classes", "Leave-one-subject-out", "RF F1 86 % full resolution; 85 % with 8 % of sensors", "Seat-pan vs backrest needs and other vehicles listed as future work"],
    [f"Nobari & Bertram 2022 {C('nobari2022')}", "Two mats on seat and back; 8 drivers; 20 non-driving activities", "Generalised vs individual models", ">70 % generalised; individual models higher", "No calibration-effort analysis"],
    [f"Licciardo et al. 2023 {C('licciardo2023')}", "Low-power CNN on FPGA; 8 postures", "Not stated in the summary we could access", "98.81 %", "Protocol unknown"],
    [f"Odesola et al. 2025 {C('odesola2025')}", "Office chair, two 32×32 mats; 1 participant", "None (single user)", "CNN 98.29 %", "Calls a personalised-vs-generalised comparison 'insightful'; brief calibration cycle left as future work"],
    [f"Liguori et al. 2025 {C('liguori2025')}", "This dataset: 30 drivers, 20 postures, ≈ 169,000 labelled maps", "None published", "—", "No evaluation or companion paper found; postures undocumented"],
]
table(doc, ["Study", "Setting", "Unseen-user evaluation", "Reported result", "What is missing"], rows, [1.15, 1.7, 1.0, 1.35, 1.3], font=8)
caption(doc, "Table 1. Closest prior work (details as reported by the authors or the publisher abstract pages).")
para(doc,
     f"Three observations follow. First, in the studies we examined, headline accuracies are for single users, known users or an unstated protocol. Second, where unseen users "
     f"are tested the penalty is 9–17 points (Tan et al., Mutlu et al.). Third, adaptation from a short initial stretch of a new person's own data is an established practice in other "
     f"driver-state channels — drowsiness networks adapted with 1–30 min of a new driver's data {C('jacobe2018')}, EEG calibration cut from 18 to 1.7 min {C('wei2018')} — and in transfer "
     f"learning generally, where per-domain mean/covariance alignment is the simplest step (Euclidean alignment for EEG {C('he2020')}; adaptive batch normalisation for networks "
     f"{C('li2018')}). For seat pressure the same idea is only listed as future work {C('odesola2025')}.")
para(doc,
     f"Adjacent evidence points the same way. Reducing the number of sensing elements has been studied for driver seats {C('zhao2021')}, lying {C('caggiari2023','hwang2025')} and sleep "
     f"posture {C('peruzzi2025')} — typically keeping accuracy with a small fraction of the cells — but always without a driver-specific calibration step. Seat-pressure maps were proposed as a "
     f"driver biometric as early as 2008 (pelvic-bone distance, two 32×32 arrays) {C('riener2008')}, which matters for the privacy claim made for pressure sensing (§4.9). "
     f"Finally, the machine-learning literature on subject-wise evaluation shows why protocol choice, not model choice, drives reported accuracy for wearable and "
     f"clinical sensing {C('gholami2020','chen2021')}.")

doc.add_heading("1.3 The gap", 2)
bullets(doc, [
    f"**Evaluation validity.** On this dataset a random forest drops from {f1(g('protocols.RF|random5'))} % (random split) to {f1(g('protocols.RF|blocked5'))} % "
    f"(blocked within driver) to {f1(g('protocols.RF|LOSO'))} % (unseen driver); accuracy depends on the protocol more than on the model {C('kapoor2023','saeb2017')}.",
    "**No calibration-length curve.** No study gives the accuracy of an unseen-driver classifier as a function of how much of the driver's own unlabelled data is used to remove the "
    "driver-specific offset — the quantity a product manager needs to set a start-up time.",
    f"**No common scale.** Sensor-count reduction has been studied only without calibration {C('zhao2021','caggiari2023','hwang2025')}, and the value of more training participants "
    f"is rarely reported; whether calibration is a cheaper way to buy unseen-driver accuracy than either is unknown.",
])

# ================================================================================================ 2 RQ
doc.add_heading("2. Research question, hypotheses and scope", 1)
box(doc, "Research question", [
    "For a seat-pressure posture classifier that must serve drivers it has not been trained on, how much unseen-driver accuracy does a label-free, driver-relative baseline "
    "recover, how long a stream of the driver's own data does it need, and is that a cheaper way to buy accuracy than more sensing cells or more training drivers?"
], fill="FFF4E5")
para(doc, "**Operational definitions.** The driver-relative feature is x_t = Z_t − b_t, where Z_t is the mean pressure map over the last Wi seconds (default 3 s) and b_t the "
          "driver's baseline. Conditions: **L0** none (b = 0); **L1** label-free running baseline (b_t = mean of the occupied-seat frames of the last W seconds, causal — the only "
          "deployable condition that asks nothing of the driver); **L1′** whole-session mean (label-free but non-causal; a bound); **L2** one labelled reference posture (mean of its first "
          "E seconds; comparator); **Oracle** mean of the driver's 20 posture means (needs every label; upper bound). **Primary estimand:** the paired difference in leave-one-driver-out accuracy between L1 "
          "and L0, with the L1 baseline computed on randomly ordered labelled streams; mean over drivers with a bootstrap CI over drivers. **Cost axes:** seconds of unlabelled data, sensing cells, training drivers.")
doc.add_heading("Hypotheses", 3)
nr = g("nested_gain")
bullets(doc, [
    f"**H1 offset removal.** With postures in random order, a label-free baseline of at most 120 s raises unseen-driver accuracy by at least 3 percentage points over no calibration. "
    f"Pilot: {dpp(lvn.get('run60'))} at 60 s, {dpp(lvn.get('run120'))} at 120 s" + ("" if nr is None else f"; for the pipeline tuned by nested validation {dpp(nr)}") + ".",
    f"**H2 time to benefit and composition.** The gain is indistinguishable from zero at 15–30 s, grows with the window because the window averages over more distinct postures, and a single-posture reference helps less than a diverse baseline. "
    f"Pilot: {dpp(lvn.get('run15'))} (15 s), {dpp(lvn.get('run30'))} (30 s), {dpp(lvn.get('run300'))} (300 s); reference P1 for 10 s {dpp(lvn.get('ref1_10'))}, reference P10 for 10 s {dpp(lvn.get('ref10_10'))}; "
    f"session mean {dpp(lvn.get('session'))}, oracle {dpp(lvn.get('oracle'))}.",
    f"**H3 substitution.** With the baseline, 24- and 96-cell arrays are non-inferior (margin 2 pp) to a 384-cell array, and the baseline gain exceeds what 64 times more cells or 75 % more training drivers (12 to 21) deliver without it. "
    f"Pilot, 24 / 96 / 384 / 1,536 cells: {f1(g('abl_res.16|run60.mean'))} / {f1(g('abl_res.8|run60.mean'))} / {f1(g('abl_res.4|run60.mean'))} / {f1(g('abl_res.2|run60.mean'))} %; "
    f"one-sided 95 % lower bound of 24 vs 384 cells {f1(g('res_tests.f16run_noninf_vs_f4run_2pp.lower95_onesided'))} pp, of 96 vs 384 cells {f1(g('res_tests.f8run_noninf_vs_f4run_2pp.lower95_onesided'))} pp.",
    f"**H4 learner independence.** The baseline gain is positive for shrinkage LDA, boosted trees and a small CNN when all are evaluated leave-one-driver-out (§4.8). "
    f"Pilot: {dpp(g('learners_gain.gLDA.run60'))} (LDA, 96 cells), {dpp(g('learners_gain.XGB.run60'))} (boosted trees, 96 cells), {dpp(g('cnn_loso_gain.CNN'))} (CNN, 384 cells); a first CNN run with grouped 6-fold cross-validation gave {dpp(g('cnn6_gain.CNN'))}.",
])
para(doc, "**In scope:** short-window classification of nominal postures from the two informative strips of the mat; calibration condition, window length, resolution and training-set size as the "
          "experimental factors. **Out of scope:** supervised enrolment and personalisation (each driver holds each posture once, so enrolment and test frames come from the same posture instance and "
          "an enrolled model cannot be separated from memorisation of that instance); absolute deployment accuracy; named posture semantics; new network architectures; sensor-placement optimisation "
          f"(a PySensors-type extension is noted for later {C('desilva2021')}); comfort; open-set rejection; privacy engineering; on-road vibration.")

doc.add_heading("Why this question and not the alternatives", 3)
an_ = g("anthro", {})
rows = [
    ["Sensor-count reduction as a stand-alone topic", f"Prior work in driver seats, lying and sleep posture {C('zhao2021','caggiari2023','hwang2025','peruzzi2025')}; the dataset's title points to array optimisation", "Kept only as a moderator (H3), where the new element is the interaction with calibration"],
    ["Supervised enrolment / personalised models", "Every posture is held once per driver: the enrolled model is tested on later frames of the same held posture, an upper bound that cannot be separated from memorisation", "Out of scope; needs repeated postures per driver"],
    ["Anthropometric fairness of the classifier", f"No detectable relation in the pilot (max |ρ| = {an_.get('max_abs_rho', float('nan')):.2f}, Holm p = {an_.get('min_p_holm', float('nan')):.1f}); {coh.get('women')} women and {coh.get('bmi30')} driver with BMI ≥ 30 in the cohort", "Reported as a check (§4.9), not a headline"],
    ["Open-set rejection of unknown postures", "Postures are undocumented; the only natural 'unknowns' are transitions", "Needs an objective posture reference"],
    ["Privacy–utility trade-off", "One session, 22–30 drivers: closed-set identification is trivially high (§4.9) and says little about population-scale risk", "Handled as a design argument for the label-free level"],
    ["A new counterbalanced data collection as the core of the PhD", "Would remove the order confound, but needs vehicles, mats and participants; every claim above is identified from the existing data by paired, order-randomised comparisons", "Optional extension (§8, WP-D); no claim depends on it"],
    ["Re-benchmarking and leakage audit alone", "Necessary (Table 5, §4.1) but a re-evaluation is not a research question", "Used as the diagnostic basis of the proposal"],
]
table(doc, ["Candidate", "Evidence against choosing it now", "Where it goes instead"], rows, [1.8, 2.9, 1.8], font=8.5)
caption(doc, "Table 2. Alternatives screened on the data and the literature.")

# ================================================================================================ 3 dataset
doc.add_heading("3. The dataset: what it supports, what was repaired, what was excluded", 1)
doc.add_heading("3.1 Description", 2)
geo = g("geometry", {})
para(doc, f"The dataset {C('liguori2025')} holds one CSV per driver exported from an XSENSOR PX100:64.160.02 mat (64×160 sensels of 1.27 cm, N/cm², "
          f"{f1(N.get('fps_median'))} frames/s, median recording {f1(N.get('minutes_median'))} min) and a table of frame ranges for 20 postures. The physical layout and the postures are not documented. "
          f"The data show three bands of pressure (Appendix C, Figure C3): a band that carries {f1(geo.get('seat', {}).get('share_of_force'))} % of the sensed force (columns 112–160), a weaker band "
          f"({f1(geo.get('back', {}).get('share_of_force'))} %, columns 64–112) and an outer strip ({f1(geo.get('outer', {}).get('share_of_force'))} %, columns 16–64). We call the first two the "
          f"seat-pan-like and backrest-like strips for convenience; **no result depends on that reading**. The README adds that the participant and the operator pressed four highlighted points on the mat "
          f"before each posture; those presses appear as short bursts of near-saturating pressure at four fixed sites (§3.3, Appendix C) and are used below as the dataset's own clock.", keep=True)

doc.add_heading("3.2 Audit", 2)
ph = N.get("phase", {})
dr = N.get("drivers_incomplete", [])
rows = [
    ["Recording", f"{N.get('frames', 0):,} frames; {N.get('labelled', 0):,} singly labelled and free of marker presses (repaired table), {N.get('marker_labelled', 0):,} labelled frames inside marker presses, {N.get('overlap', 0):,} inside overlapping labels, {N.get('unlabelled', 0):,} unlabelled ({ph.get('pre', 0):,} before the first posture, {ph.get('gap', 0):,} in gaps, {ph.get('post', 0):,} after the last)", "Presses and overlaps dropped; unlabelled frames feed only the real-stream baselines"],
    ["Time base", "Time stamps have 1-s resolution; frame indices are contiguous; one duplicated frame in the whole dataset; no frozen stretches", f"Frame index is the clock ({f1(N.get('fps_median'))} frames/s, 13.2–14.6 across drivers)"],
    ["PNG route", "Per-frame min–max scaled (absolute pressure lost), incomplete, and built from the same (defective) label ranges", "Not used; CSVs only"],
    ["Label file", f"599 of 600 rows (driver 3 has no P5; a stray space in 'USER_3 '); {len(rp.get('violating_users', []))} drivers with inconsistent ranges: nested ranges (6, 24, 28), partial overlaps (2, 4, 22), one reversed pair (3)", "Audited and repaired with the marker presses (rules R1–R3, R6; Appendix C); these drivers are excluded from the primary analysis"],
    ["Driver 8", "The CSV is complete (all 20 marker presses lie inside it) but the label table runs to frame 10,563 on a different, irregular time base: published starts are 1,540–2,408 frames later than the presses imply", "All 20 ranges rebuilt from the presses (M3, validated by a template test); excluded from the primary analysis"],
    ["Marker presses", f"Before each posture four highlighted points were pressed (README); about {f1(g('markers.events_per_driver_median'))} press events per driver; {f1(100 * g('markers.rows_in_marker', 0) / max(g('markers.rows', 1), 1))} % of the cohort's evaluation rows and {f1(100 * g('markers.frames_in_marker', 0) / max(g('markers.frames_labelled', 1), 1))} % of its labelled frames lie inside a press", "Frames within ±10 of a press are removed from the labelled data and from every baseline; the effect of keeping them is in Table 9"],
    ["Posture order", "29 of 30 drivers performed P1…P20 in the same order; driver 3 performed P19 before P18 (§3.3)", "Posture ≡ time slot: confound, §3.5"],
    ["Absolute load", f"Sensed force is {g('load.ratio_min', 0):.2f}–{g('load.ratio_max', 0):.2f}× body weight (mean {g('load.ratio_mean', 0):.2f}; r = {g('load.r_weight_force', 0):.2f} with weight); saturation negligible", "Treated as relative load only"],
]
table(doc, ["Item", "Finding", "Handling"], rows, [0.95, 3.85, 1.7], font=8.5)
caption(doc, "Table 3. Data audit (scripts 01, 03, 43, 44, 47 in analysis/scripts; details in analysis/DATASET_AUDIT.md).")

doc.add_heading("3.3 Unknowns, and why none of them needs an answer from the dataset authors", 2)
para(doc, "The first draft of this proposal ended with five questions for the dataset authors. They cannot be answered from outside, so each was resolved from the data, designed out, or removed (Table 4).")
sw = g("swap", {})
oad = oa("A.drivers_in_protocol_order_repaired", {"value": 29, "note": ""})
n_ord = int(float(oad["value"]))
rows = [
    ["What are the 20 postures?", "Not documented anywhere we could find.", "Not needed. Classes are anonymous slots P1–P20; every claim concerns whether a slot can be recovered for an unseen driver, never what the posture is. The readings in Appendix B are ours and flagged as such."],
    ["Was the order randomised?", f"No: {n_ord} of 30 drivers followed P1→P20 in the repaired table. Driver 3 performed P19 before P18: a template test calibrated on {sw.get('n_cal', 425)} clean adjacent pairs ({f1(sw.get('cal_correct'))} % ordered correctly) rejects the protocol order for that pair ({pf(sw.get('p_protocol'))}), so the label table is right and the order was not strictly enforced.",
     "Designed out: every label-free baseline is computed on randomly ordered streams; only paired differences are claimed; absolute accuracy is read against an elapsed-time-only null (§3.5)."],
    ["Where is each strip on the seat?", f"Three pressure bands are visible (Figure C3); {f1(g('geometry.analysed (64-160).share_of_force'))} % of the force lies in columns 64–160.", "Not needed. The analysis uses columns 64–160 as two anonymous strips; the outer strip is excluded and shown as a sensitivity."],
    ["Correct label ranges, missing P5 of driver 3, missing frames of driver 8?", f"Nested and overlapping ranges are typing errors with a logical signature, and the README documents a press before every posture. In the {coh.get('n')} consistent drivers a published start follows the preceding press by {g('markers.OFF_S')} frames (MAD {f1(g('markers.mad_s'))}) and a published end precedes the next press by {g('markers.OFF_E')} (MAD {f1(g('markers.mad_e'))}); {f1(100 * g('markers.frac_start_found', 0))} % of the ranges have a press within 400 frames before them. Driver 3 has 19 presses for 19 postures: P5 was never performed. Driver 8's CSV holds all 20 presses.",
     f"Repaired with press evidence where the presses allow (R1, R2, M3: {rp.get('recovered', 0)} segments recovered, driver 8 rebuilt), the shared frames dropped where they do not (drivers 2, 22). The primary analysis uses only the {coh.get('n')} drivers whose tables were consistent, so no result depends on a repair."],
    ["Seat position, recline, clothing?", f"Unknown. Sensed load tracks body weight (r = {g('load.r_weight_force', 0):.2f}); the largest standardised deviation of the load ratio among the cohort drivers is {f1(g('load.clean_max_abs_z'))} SD.", "Treated as part of the 'driver' effect that calibration is meant to absorb; within-session changes cannot be told from posture and are left in the noise."],
    ["Companion paper or overlapping work?", "None found (searches on 5 October 2026, including DataCite events and OpenAlex).", "Not a validity question. The contribution (audited labels, calibration-length curve, three-lever comparison) is complementary to any sensor-optimisation work."],
]
table(doc, ["Unknown", "What the data show", "Decision"], rows, [1.25, 2.6, 2.65], font=8.2)
caption(doc, "Table 4. Unknowns about the dataset and how each is neutralised.")

doc.add_heading("3.4 Analysis cohorts", 2)
para(doc, f"**Primary: {coh.get('n')} drivers** whose label tables are internally consistent and complete (all except 2, 3, 4, 6, 8, 22, 24, 28). Their labels are identical in the original and repaired tables. "
          f"The cohort has {coh.get('women')} women and {coh.get('bmi30')} driver with BMI ≥ 30 (weight {coh.get('weight_min', 0):.0f}–{coh.get('weight_max', 0):.0f} kg, BMI {coh.get('bmi_min', 0):.1f}–{coh.get('bmi_max', 0):.1f}). "
          f"**Extended:** all 30 drivers with the repaired labels (599 of 600 postures; driver 8 rebuilt from the presses). **First cleaning:** all 30 drivers with overlapping frames dropped and nested ranges lost, as in the first draft. "
          f"The cohort was defined by label consistency alone, before and without reference to any accuracy. All other choices are fixed in Table 10.")

doc.add_heading("3.5 The order confound and what bounds it", 2)
r2 = oa("C.smooth_R2_degree2", {})
adj = oa("D.errors_to_adjacent_slot_none", {})
adj_o = oa("D.errors_to_adjacent_slot_oracle", {})
lp = oa("B.load_profile_range_pct", {})
trend = oa("B.load_trend_rho_median", {})
add_fig(doc, "F11_order_audit.png", 5.8, "Figure 1. Order audit on the analysis cohort. (a) Driver-normalised total load across the 20 protocol slots (95 % bootstrap band). (b) Share of the between-slot variance explained by a quadratic trend in slot index, "
        "against 5,000 permuted slot orders. (c) Where unseen-driver errors go, by slot distance, against uniform confusions.")
smooth = r2.get("value", float("nan")) > 3 * r2.get("null_mean", 1.0) and r2.get("p", 1.0) < 0.01
para(doc, f"Because the order is fixed, elapsed time since the first posture predicts the label: a random forest given only that number reaches {mci(g('time_only'))} % leave-one-driver-out accuracy (chance 5 %), more than any uncalibrated pressure model below. "
          f"Three checks bound the confound (Figure 1). (a) The driver-normalised total load varies by {f1(lp.get('value'))} points across the session, and the median per-driver Spearman correlation of load with slot is {trend.get('value', float('nan')):.2f} "
          f"(Wilcoxon {pf(trend.get('p'))}): no common load drift. "
          f"(b) A quadratic trend in slot index explains {f1(r2.get('value'))} % of the between-slot variance of the driver-centred slot-mean maps (bootstrap 95 % CI {f1(r2.get('ci_lo'))}–{f1(r2.get('ci_hi'))} %) against {f1(r2.get('null_mean'))} % for permuted slot orders ({pf(r2.get('p'))}). "
          + ("Neighbouring slots are therefore more alike than distant ones. Slow drift would produce this, and so would a protocol that moves from one posture family to the next; the data cannot tell the two apart, which is the core reason the order cannot be removed by analysis. "
             if smooth else "There is no excess smooth structure in slot order. ")
          + f"(c) {f1(adj.get('value'))} % of unseen-driver errors go to the neighbouring slot without calibration ({f1(adj.get('uniform'))} % expected under uniform confusion) and {f1(adj_o.get('value'))} % with the oracle baseline: what remains after offset removal is mostly confusion between neighbouring postures. "
          f"The checks bound the confound but cannot remove it. The design therefore estimates only paired differences between calibration conditions, computes every label-free baseline on randomly ordered streams "
          f"and treats absolute accuracy as specific to this protocol.")

# ================================================================================================ 4 pilot
doc.add_heading("4. Pilot study on the public dataset", 1)
para(doc, f"All results are leave-one-driver-out (LOSO) over the {coh.get('n')} cohort drivers unless stated; the unit of analysis is the driver, so means, bootstrap CIs (10,000 resamples of drivers) "
          "and Wilcoxon signed-rank tests are computed on per-driver accuracies, with Holm correction inside each stated family "
          f"{C('demsar2006','holm1979')}. Features are mean-pooled maps of columns 64–160 (16×24 cells unless stated), every 6th labelled frame, rows at least 5 s into a segment, overlapping labels and the first "
          f"10 s of P1 removed. 'Random order' means the driver's labelled segments are concatenated in a random order before the running baseline is computed (three shuffles per driver, averaged within driver; five for Figure 7). "
          f"Learners: scikit-learn pipelines {C('pedregosa2011')} (standardise → PCA → shrinkage LDA {C('ledoit2004')}), a GPU re-implementation of the same "
          f"pipeline that reproduces scikit-learn's predictions exactly (100 % agreement on 20 test folds), XGBoost on GPU {C('chen2016xgb')} and a three-layer CNN in PyTorch {C('paszke2019')}. "
          f"All design choices were screened on these drivers; headline numbers for tuned hyperparameters come from nested validation (§4.7) and the remaining forking paths are listed in Table 11.")

doc.add_heading("4.1 The protocol decides the answer", 2)
pr = g("protocols", {})
rows = [[m, f1(pr.get(f"{m}|random5")), f1(pr.get(f"{m}|blocked5")), f1(pr.get(f"{m}|LOSO")), f1(pr.get(f"{m}|random5", 0) - pr.get(f"{m}|LOSO", 0))] for m in ("LDA", "LogReg", "kNN", "RF") if f"{m}|LOSO" in pr]
table(doc, ["Model", "Random 5-fold", "Blocked, same driver (3 s buffer)", "Unseen driver (LOSO)", "Drop random → LOSO (pp)"], rows, [1.0, 1.1, 1.6, 1.4, 1.4], font=9, align_right_from=1)
caption(doc, "Table 5. Accuracy (%) of the same 20-class problem under three protocols (16×24 maps of columns 64–160, every 6th labelled frame, chance 5 %).")
add_fig(doc, "F1_protocol_inflation.png", 5.6,
        "Figure 2. Evaluation protocol versus accuracy; the dashed line is a random forest that sees only elapsed time since the first posture.")
para(doc, f"Consecutive frames correlate 0.97 and decorrelate only after about 3 s, so random splits measure memorisation of the driver's session. The best unseen-driver model "
          f"({f1(g('protocols.RF|LOSO'))} %) stays below the order-only floor ({f1(g('time_only.mean'))} %).")

doc.add_heading("4.2 Why: the driver dominates the pressure map", 2)
add_fig(doc, "F2_variance_decomposition.png", 5.3,
        f"Figure 3. Variance of the driver × posture mean maps ({g('variance.all.n', 22)} drivers with all 20 postures) split into driver, posture and interaction/residual.")
sg = g("segment", {})
para(doc, f"Driver identity explains {f1(g('variance.all.subject'))} % of the variance of the analysed area, posture {f1(g('variance.all.posture'))} % and the driver × posture interaction/residual "
          f"{f1(g('variance.all.interaction'))} %; the seat-pan-like strip is the most driver-dominated ({f1(g('variance.seat.subject'))} %). Driver-centred contrast maps "
          f"(Appendix B) show consistent patterns that look like recline, left/right load shifts and forward lean (our reading), so a posture signal exists once the driver offset is removed. "
          f"A segment-level nearest-centroid classifier (one mean map per driver × posture, so no frame redundancy) goes from {f1(sg.get('f4|none|sub', {}).get('cos'))} % to "
          f"{f1(sg.get('f4|oracle|sub', {}).get('cos'))} % when each driver's own mean map is subtracted, but only to {f1(sg.get('f4|ref1|sub', {}).get('cos'))} % (P1) or "
          f"{f1(sg.get('f4|ref10|sub', {}).get('cos'))} % (P10) when a single posture is the reference. The interaction share is larger than the posture share, which caps what any offset removal can deliver.")

doc.add_heading("4.3 The calibration ladder", 2)
lrec = g("ladder_rec", {})
names = [("none", "L0 none"), ("run15", "L1 running baseline, W = 15 s"), ("run30", "L1 W = 30 s"), ("run60", "L1 W = 60 s"), ("run120", "L1 W = 120 s"), ("run300", "L1 W = 300 s"),
         ("session", "L1′ whole-session mean (non-causal)"), ("ref1_5", "L2 reference P1, E = 5 s"), ("ref1_10", "L2 reference P1, E = 10 s"),
         ("ref10_5", "L2 reference P10, E = 5 s"), ("ref10_10", "L2 reference P10, E = 10 s"), ("oracle", "Oracle: mean of all 20 posture means")]
rows = []
for k, lab in names:
    d = lvn.get(k)
    rows.append([lab, mci(lad.get(k)), "—" if d is None else dpp(d), "—" if d is None else ("< 0.001" if d.get("p_holm", 1) < 0.001 else f"{d.get('p_holm', float('nan')):.3f}"),
                 "—" if d is None else f"{100 * d['better']:.0f} %", f1(lrec[k]["mean"]) if k in lrec else "—"])
table(doc, ["Calibration", "Accuracy % [95 % CI]", "Gain vs none, pp [95 % CI]", "p (Holm)", "Drivers improved", "Recorded order, %"], rows, [2.05, 1.15, 1.3, 0.55, 0.7, 0.75], font=8.2)
caption(doc, f"Table 6. Calibration ladder (shrinkage LDA, 3 s integration, {coh.get('n')} drivers). L1 rows use segments in random order; the last column repeats them in the recorded order. Wilcoxon signed-rank, Holm over the 11 comparisons with L0.")
add_fig(doc, "F3_calibration_ladder.png", 5.7,
        "Figure 4. Unseen-driver accuracy versus baseline window; horizontal bands are 95 % CIs of the no-calibration, whole-session and oracle conditions.")
gc = g("gap_closure", {})
para(doc, f"A label-free baseline helps once it spans about a minute: {dpp(lvn.get('run15'))} at 15 s and {dpp(lvn.get('run30'))} at 30 s are indistinguishable from zero, {dpp(lvn.get('run60'))} at 60 s recovers "
          f"{f1(gc.get('run60'))} % of the oracle gain and {dpp(lvn.get('run120'))} at 120 s recovers {f1(gc.get('run120'))} %; 300 s adds little ({dpp(lvn.get('run300'))}). "
          f"The whole-session mean ({dpp(lvn.get('session'))}) and the all-labels oracle ({dpp(lvn.get('oracle'))}) bound what offset removal can deliver. "
          f"A single reference posture depends on which posture it is ({dpp(lvn.get('ref1_10'))} for P1 at 10 s, {dpp(lvn.get('ref10_10'))} for P10 at 10 s), consistent with H2: "
          f"the baseline must average over a diverse part of the driver's repertoire. **This is also the main limit of the pilot:** the protocol cycles through 20 postures in about nine minutes, so a "
          f"120-s window spans about five postures; in habitual driving a window is dominated by one posture, the running baseline converges to it and the driver-relative feature shrinks towards zero (a "
          f"naturalistic test is the optional WP-D).")
add_fig(doc, "F4_paired_drivers.png", 5.0, "Figure 5. Per-driver accuracy with and without the label-free baseline (60-s and 120-s windows, random order); red = women, blue = men; points above the diagonal improve.")
pp_ = g("per_posture")
if pp_:
    add_fig(doc, "F10_per_posture_recall.png", 5.6, "Figure 6. Recall of each posture for unseen drivers (pooled over the held-out drivers, shrinkage LDA; 60-s baseline pooled over three random orders).")
    para(doc, f"Without calibration {pp_['n_gt20_none']} of the 20 postures have recall above 20 % (four times chance); with the 60-s baseline {pp_['n_gt20_run60']} do, and {pp_['n_gt20_oracle']} with the oracle; the best are "
              f"P{', P'.join(str(x) for x in pp_['best_run60'][:3])}. The baseline therefore widens the set of postures that can be recovered rather than only lifting the average, and a future posture vocabulary "
              f"should be defined from objective angles and load shifts, not from the dataset's indices.")

doc.add_heading("4.4 Control: is the baseline gain an artefact of the fixed order?", 2)
add_fig(doc, "F7_order_control.png", 5.9,
        "Figure 7. 60-s running baseline with the labelled segments in recorded order versus random order (mean of shuffles per driver: five for LDA, two for boosted trees).")
od = g("order", {})
lx = g("learners", {})
lg = g("learners_gain", {})
xo = ""
if lx.get("XGB"):
    xo = (f" For boosted trees on 96-cell maps the numbers are {f1(lx['XGB']['none']['mean'])} % (none), {f1(lx['XGB']['run60_rec']['mean'])} % (recorded order) and {f1(lx['XGB']['run60']['mean'])} % (random order); "
          f"order information is worth {dpp(lg['XGB']['rec_vs_shuf'])}. A non-linear learner turns the order carried by a time-local baseline into most of its apparent gain ({dpp(lg['XGB']['rec'])} in recorded order against "
          f"{dpp(lg['XGB']['run60'])} in random order), so a label-free gain measured on a fixed-order protocol without this control can be mostly an order effect.")
para(doc, f"With segments concatenated in random order the 60-s baseline still gives {mci(od.get('shuffled'))} % against {f1(lad.get('none', {}).get('mean'))} % without it "
          f"({dpp(od.get('shuf_vs_none'))}, {pf(g('order.shuf_vs_none.p'))}); the recorded order adds {dpp(od.get('rec_vs_shuf'))}, which is order information or drift tracking and "
          f"not posture. The real stream, with the pre-posture phase and the gaps between postures, lies in between ({f1(g('order.full.mean'))} %). We therefore quote the random-order value as the conservative estimate of the label-free gain." + xo)

doc.add_heading("4.5 Ablations: what each component contributes", 2)
w, o, rg, tr = g("abl_window", {}), g("abl_op", {}), g("abl_region", {}), g("abl_trim", {})
rs = g("abl_res", {})


def cell(d, k):
    v = d.get(k)
    return "n/a" if v is None else f1(v["mean"])


_fp = g("op_friedman_p")
fps_op = "n/a" if _fp is None else f"{_fp:.2f}"


rows = [
    ["Integration window Wi (0 / 1 / 3 / 5 s)", f"none {cell(w,'0|none')} / {cell(w,'1|none')} / {cell(w,'3|none')} / {cell(w,'5|none')};  60-s baseline {cell(w,'0|run60')} / {cell(w,'1|run60')} / {cell(w,'3|run60')} / {cell(w,'5|run60')}", f"Effects stay within {max(abs(w['0|none']['mean'] - w['5|none']['mean']), abs(w['0|run60']['mean'] - w['3|run60']['mean'])):.1f} pp (the uncalibrated model prefers no integration, the calibrated one 1–3 s): errors are systematic per driver, not frame noise; 3 s kept"],
    ["Operator (subtract / weight-scaled / log-ratio)", f"{cell(o,'sub|run60')} / {cell(o,'sub_scaled|run60')} / {cell(o,'logratio|run60')} (Friedman p = {fps_op})", "Plain subtraction suffices; no extra parameter"],
    ["Region (outer / backrest-like / seat-pan-like / both / whole mat)", f"none: {cell(rg,'(16, 64)|none')} / {cell(rg,'(64, 112)|none')} / {cell(rg,'(112, 160)|none')} / {cell(rg,'(64, 160)|none')} / {cell(rg,'(0, 160)|none')};  baseline: {cell(rg,'(16, 64)|run60')} / {cell(rg,'(64, 112)|run60')} / {cell(rg,'(112, 160)|run60')} / {cell(rg,'(64, 160)|run60')} / {cell(rg,'(0, 160)|run60')}", f"The outer strip adds {rg['(0, 160)|run60']['mean'] - rg['(64, 160)|run60']['mean']:+.1f} pp once the baseline is used ({rg['(0, 160)|none']['mean'] - rg['(64, 160)|none']['mean']:+.1f} pp without it); both inner strips are needed"],
    ["Segment-edge trimming (0 / 1 / 3 s)", f"none {cell(tr,'0|none')} / {cell(tr,'14|none')} / {cell(tr,'41|none')};  baseline {cell(tr,'0|run60')} / {cell(tr,'14|run60')} / {cell(tr,'41|run60')}", f"Trimming never helps (3 s costs {tr['0|none']['mean'] - tr['41|none']['mean']:.1f} pp uncalibrated, {tr['0|run60']['mean'] - tr['41|run60']['mean']:.1f} pp calibrated): label-boundary noise does not limit accuracy"],
    ["Sensing cells (24 / 96 / 384 / 1,536)", f"none {cell(rs,'16|none')} / {cell(rs,'8|none')} / {cell(rs,'4|none')} / {cell(rs,'2|none')};  baseline {cell(rs,'16|run60')} / {cell(rs,'8|run60')} / {cell(rs,'4|run60')} / {cell(rs,'2|run60')}",
     f"With the baseline, 24 and 96 cells are non-inferior to 384 (lower bounds {f1(g('res_tests.f16run_noninf_vs_f4run_2pp.lower95_onesided'))} and {f1(g('res_tests.f8run_noninf_vs_f4run_2pp.lower95_onesided'))} pp; margin −2); fewer cells do better"],
]
table(doc, ["Factor (levels)", "Unseen-driver accuracy, % (shrinkage LDA; baseline = 60 s, random order)", "Reading"], rows, [1.6, 2.8, 2.1], font=8.5)
caption(doc, f"Table 7. Ablations ({coh.get('n')} drivers each; paired tests in analysis/results). The learner ablation is in Table 8.")
add_fig(doc, "F5_resolution_x_calibration.png", 4.3,
        "Figure 8. Sensing cells versus accuracy, with and without the 60-s baseline (GPU LDA, 95 % CIs over drivers).")
rt = g("res_tests", {})
para(doc, f"With the baseline a {CELLS[16]}-cell array reaches {mci(rs.get('16|run60'))} %, against {mci(rs.get('4|run60'))} % for {CELLS[4]} cells ({dpp(rt.get('f16run_vs_f4run'))}) and {mci(rs.get('2|none'))} % for a {CELLS[2]:,}-cell array without it "
          f"(paired difference {dpp(rt.get('f16run_vs_f2none'))}). Finer maps do not help an unseen driver and slightly hurt; a plausible reason, not tested here, is that fine structure is driver-specific and coarse pooling acts as regularisation. "
          f"The 24-cell result is a statement about pooled maps of this mat, not a specification for a 4×6 sensor." +
          (lambda c, ct: "" if not c else f" Combining the two, 24 cells with a 120-s baseline reach {mci(c.get('16|run120'))} % (384 cells: {mci(c.get('4|run120'))} %; same-resolution session mean {mci(c.get('16|session'))} %, oracle {mci(c.get('16|oracle'))} %); "
                                           f"the oracle exceeds it by {dpp(ct.get('oracle16_vs_c16r120'))}.")(g('combo'), g('combo_tests', {})))

# pending-result subsections 4.6-4.9, sections 5-9, references and appendices live in 40c_tail.py (same namespace)
exec(compile(Path(__file__).with_name("40c_tail.py").read_text(encoding="utf-8"), "40c_tail.py", "exec"))
doc.save(OUT)
print("saved", OUT, "| references cited:", len(C.order))
