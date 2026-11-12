"""Static blocks of STUDY_RECORD.md (evidence base, implementation verification, data summary, decisions).  Facts only: every reference was verified
(results/refs_verified.csv); quantities marked 'pilot' are the pilot's own results, re-derived or reproduced by E1; nothing here depends on the follow-up results.
"""

LITERATURE = """
| Theme | Source (verified) | What it establishes | What it leaves open |
|---|---|---|---|
| **A. Pressure-based posture recognition and the unseen-user penalty** | Tan, Slivovsky, Pentland 2001, IEEE/ASME Trans Mechatronics 6(3):261-268 | pressure-distribution chair; PCA classifier 96 % for familiar vs 79 % for unfamiliar users (pilot Table 1) | a 17-pp penalty is reported; what closes it is not studied (**G1**) |
| | Mutlu et al. 2007, UIST (doi:10.1145/1294211.1294237) | 31 subjects, 10 postures; subject-invariant features: 87 % cross-validated (31 sensors), 78 % unseen users (19 sensors) | invariance by hand-designed features and placement, not by adaptation (**G1**) |
| | Zemp et al. 2016, BioMed Res Int (doi:10.1155/2016/5978489) | 41 subjects, 7 positions; random forest 90.9 % on unseen subjects | no calibration analysis |
| | Zhao, Beurier, Wang, Wang 2021, Sensors 21(10):3346 | **23 drivers in a driving mock-up, two XSENSOR 42x44 mats (1.27 cm cells), leave-one-subject-out; RF F1 86 % (trunk 91, left foot 93, right foot 74); 85 % with 8 % of the sensors** (read in full for this study) | sensor reduction without any user calibration; one mock-up, laboratory only (**G1**) |
| | Nobari & Bertram 2022, AHFE (doi:10.54941/ahfe1002733) | 8 drivers, 20 non-driving activities; generalised model > 70 %, individual models higher | no calibration-effort analysis |
| | Zhao, Yan, Wang 2023, J Electr Eng 74(6) (doi:10.2478/jee-2023-0055) | subject-dependent / subject-independent / cross-subject settings; the cross-subject setting "greatly suffers from degraded accuracy"; transfer learning presented preliminarily (abstract) | which target-user data, how much, labelled or not: not characterised (**G1**) |
| | Odesola et al. 2025, Sensors 25(18):5610 | single participant, CNN 98.3 %; personalised vs generalised comparison called insightful; brief calibration cycle left as future work | the gap itself (**G1**) |
| | ChairPose 2025 (arXiv:2508.01850) | 8 users x 4 chairs; physics-based augmentation; 89.4 mm joint error for an unseen user and chair | generalisation by simulation, not by the user's own data |
| | Liang et al. 2017, IEEE PerCom ("Postureware" smart cushion) | user-invariant features plus an ensemble to improve cross-user generalisation | invariance by design |
| | Licciardo et al. 2023, SPIE 12571 / Liguori et al. 2025, Zenodo 10.5281/zenodo.15199100 (the DPR dataset) | the dataset group's CNN reports 98.81 % for 8 postures (protocol not stated in the accessible summary); the dataset: 30 drivers x 20 postures, 64x160 XSENSOR | no published evaluation, no companion paper, 0 citations (**G3**) |
| | Mahomed & Saha 2024, IEEE Access 12:176301 | review of driver posture recognition | - |
| **B. Evaluation validity** | Saeb et al. 2017; Little et al. 2017 (GigaScience); Kapoor & Narayanan 2023 (Patterns); Hammerla & Plotz 2015 (UbiComp); Dehghani et al. 2019 (Sensors); Varoquaux 2018 (NeuroImage); Roberts et al. 2017 (Ecography); Bates, Hastie, Tibshirani 2024 (JASA); Gholamiangonabadi et al. 2020; Chen et al. 2021 (ACM CSUR) | subject-wise evaluation is required for a new-user claim; random or record-wise splits and overlapping windows inflate accuracy; cross-validation error bars are wide for small samples; grouped data need blocked designs | no seat-pressure study reports an elapsed-time null or order-randomised baselines (**G3**) |
| **C. Label-free normalisation and alignment** | Atal 1974 (JASA 55); Viikki & Laurila 1998 (Speech Commun 25); Hermansky & Morgan 1994 (RASTA); Pujol, Macho, Nadeu 2006 (ICASSP, real-time mean-and-variance normalisation); Kaldi `apply-cmvn-sliding` | cepstral mean (and variance) normalisation removes a stationary speaker / channel offset from the user's own stream; causal sliding-window versions exist for real-time use | never applied to static pressure maps; the window / diversity requirement is unknown (**G2**) |
| | He & Wu 2020, IEEE TBME (Euclidean alignment); Zanini et al. 2018 and Rodrigues et al. 2019, IEEE TBME (Riemannian alignment, re-center-stretch-rotate) | per-subject covariance whitening / re-centring of multichannel EEG trials; 13 BCI paradigms | assumes zero-mean multichannel signals - not true for pressure maps (**G2**) |
| | Sun, Feng, Saenko 2016, AAAI (CORAL); Li et al. 2018, Pattern Recognit (AdaBN) | alignment of second-order statistics; re-estimation of batch-normalisation statistics on the target domain - both label-free, parameter-light | data requirement of the target statistics for pressure maps unknown (**G2**) |
| | Wang et al. 2021, ICLR (Tent); Liang et al. 2025, IJCV (test-time adaptation survey) | online entropy-minimisation adaptation of normalisation parameters on the test stream | needs online optimisation of the deployed model - not part of the label-free baseline claim |
| **D. Time-series adaptation benchmarks; simple baselines** | Ragab et al. 2023, ACM TKDD 17(8) (AdaTime); Wilson, Doppa, Cook 2020, KDD (CoDATS); Gulrajani & Lopez-Paz 2021, ICLR (DomainBed) | standardised benchmarks and official code for 12 UDA methods on sensor time series; with careful model selection empirical risk minimisation matches or beats 9 domain-generalisation algorithms on 7 benchmarks | UDA trains on the target stream and cannot select models without target labels; simple baselines stay the justified comparator |
| **E. Short adaptation from the new person's own data in other channels** | Jacobe de Naurois et al. 2018, Accid Anal Prev (drowsiness networks adapted per driver); Wei et al. 2018, NeuroImage (EEG subject transfer); Ferrari et al. 2020, IEEE Access (personalisation in HAR); Stisen et al. 2015, SenSys (device and user heterogeneity) | adaptation from a brief initial stretch of the new person's data is established practice elsewhere | for seat pressure it is only listed as future work (Odesola 2025) |
| **F. Seat-pressure dynamics** | Crawford et al. 2005, Arch Phys Med Rehabil 86(6):1221-1225 (12 non-wheelchair and 15 wheelchair users with MS, 8 min) | in non-wheelchair users average interface pressure fell during minutes 0-2, rose during 2-4 and did not change after 4 min | supports a drift component in the first minutes of a fixed-order 9-min protocol; it cannot be separated from the protocol's posture order (**G3**) |
| **G. Sensor reduction** | Zhao 2021; Caggiari et al. 2023 (Sensors 23:6872); Hwang et al. 2025 (Electron Lett); Peruzzi et al. 2025 (Sensors 25:458); Brunton et al. 2016 (SIAM J Appl Math); PySensors | accuracy kept with a small fraction of the cells in driver seats, lying, sleep posture | always without a user-specific calibration step |
| **H. Statistics** | Demsar 2006 (JMLR); Benavoli et al. 2017 (JMLR 18(77)); Nadeau & Bengio 2003; Lakens 2017; Field & Welsh 2007; Bates et al. 2015 (lme4); Holm 1979; Ledoit & Wolf 2004 | paired non-parametric tests with Holm correction over a stated family; equivalence / non-inferiority with a margin; cluster bootstrap and mixed models for repeated measures; shrinkage covariance | - |
"""

IMPLEMENTATIONS = """
| Method | Authoritative source (read 6 Oct 2026) | What was checked | Use in this study |
|---|---|---|---|
| Sliding mean / variance normalisation | Kaldi `apply-cmvn-sliding`, `SlidingWindowCmn` (src/feat/feature-functions.cc) | non-centred window = frames t - cmn_window ... t, expanding at the start of the stream; a minimum window forces limited look-ahead at the start (not used here: causality kept); variance normalisation optional | operators `sub` and `cmvn`; GPU window statistics agree with NumPy to 1e-7, the pandas-rolling and cumulative-sum implementations agree bit-for-bit (`results/operator_unit_tests.txt`, E1 unit test) |
| CORAL | official code `VisionLearningGroup/CORAL`, `Ex_CORAL.m` | per-domain z-score; `cov + eye(D)`; `A = Cs^(-1/2) Ct^(1/2)` | operator `wh`: shrinkage a toward a scaled identity; a = 0.5 on z-scored data equals (cov + I)/2 - agrees with the official formula to 8e-8 (3e-4 with MATLAB's unbiased covariance) |
| Euclidean alignment (EA) | He & Wu official MATLAB code `hehe91/EA` (README: official implementation); pyRiemann 0.12 `TLCenter(metric='euclid')`, `invsqrtm` (installed to a scratch folder, shared environment untouched) | mean trial covariance R, trials R^(-1/2) X | operator `ea` (uncentred second moment of the window, a = 0.1): aligned trial covariances agree with `TLCenter` to 2.6e-14; whitened frames have identity second moment to 4e-9 |
| AdaBN | Li et al. 2018, Algorithm 1 (no official PyTorch code found; the AdaTime repository does not implement it) | BN statistics re-estimated on target-domain inputs, weights untouched, no labels | `gpu_models.cnn_predict_adabn`: BN layers reset, the whole held-out driver forwarded in one batch (exact statistics) |
| AdaTime, Tent, DomainBed | `emadeldeen24/AdaTime` (MIT, 12 UDA methods), `DequanWang/tent` (MIT), Gulrajani & Lopez-Paz 2021 | read; **not run** | they adapt by training / entropy minimisation on the target stream (a per-driver optimisation, not available from the first frame), model selection needs target labels, and ERM with simple statistics is the justified comparator (section 4) |
| Shrinkage LDA | scikit-learn `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` | the torch port reproduces scikit-learn (pilot: 100 % prediction agreement on 20 folds); E1 re-runs the whole contrast with scikit-learn | primary learner; GPU port with an opt-in numerical-rank tolerance (1e-6) in the follow-up |
| XGBoost 3.4 GPU, PyTorch 2.6 | official packages | fixed small configurations declared in advance | robustness learners |
"""

REVISIONS = """
Statements of `PhD_Proposal_Seat_Pressure_Driver_Calibration.docx` (not rebuilt here) that this study changes, and why. Nothing in the pilot's data was altered; its headline survives.

| # | Proposal statement | Revision | Basis |
|---|---|---|---|
| 1 | "Oracle: mean of all 20 posture means" is the upper bound (Table 6, Figure 4, H2) | It is a **first-order reference**. At 24 cells a whole-session, label-free whitening beats it; at 96 cells it only matches it | section 5.3 |
| 2 | The requirement is "one to two minutes of unlabelled data" (H2, section 6a) | The requirement is **distinct postures**; seconds are a proxy that holds only for this protocol (postures of 13-27 s; a window spans 3.6 / 5.9 distinct postures at 60 / 120 s). How fast a real driver supplies distinct postures is unmeasured | section 5.4 |
| 3 | Headline gains +6.0 / +8.8 pp (3 shuffles) | +@@G60@@ / +@@G120@@ pp with 10 shuffles; shuffle noise (cohort mean SD 0.6-0.9 pp, per driver ~4 pp) is now reported and the compared conditions share their random orders | sections 5.1-5.2 |
| 4 | "Covariance-whitening alignment needs far more frames than dimensions" (Table 10, not used) | Tested at 24 and 96 cells: +1 to +2 pp over mean subtraction on causal windows, +3 to +4 pp with a whole session; still untested at 384 cells. Euclidean alignment as published does not beat mean subtraction | section 5.3 |
| 5 | "Habitual driving untested; the running baseline converges to the dominant posture" (section 4.3, Table 11) | Partly quantified: a baseline that may contain the current posture loses 0.4-0.8 pp at equal diversity (E3). A running window dominated by one posture would cancel the signal; the DPR protocol (each posture held once) cannot test this, so habitual driving remains the missing experiment | section 5.4 |
| 6 | WP-C: replicate on independent open pressure-mat data | Not part of this study, which is deliberately single-dataset; the single-dataset limitation stands | section 7 |
| 7 | H4 learner independence; Table 8 | **Narrowed:** re-runs on other random orders show the benefit is learner-dependent (shrinkage LDA robust; boosted trees small and not stable; the CNN in section 5.6, with AdaBN as the deep label-free comparator) | sections 5.5-5.6 |
| 8 | Random-order control (section 4.4) | Implementation warning: the permutation must be drawn **independently per driver**; a permutation shared by all drivers for a seed is a new fixed protocol order | section 5.1 |
| 9 | Contribution (iv) "a transferable validation recipe" | Extended with an independent clean-room re-implementation (feature-level unit test), a pre-specified operator decision rule and an incremental-over-time test | sections 5.1, 5.3, 5.5 |
"""

LIMITATIONS = """
* **One dataset, one seat, one 9-minute session; postures undocumented; slot = time.** Absolute accuracy (20-35 % of 20 classes, chance 5 %) is a property of this protocol, not a deployment figure; only relative effects between calibration conditions are claimed.
* **The benefit is tied to DPR's variance regime.** The driver explains 76.7 % of the variance of the pressure maps and the posture 8.6 % (pilot); the method removes a driver offset, so it can only pay where that offset dominates. Other regimes were not tested - the study is single-dataset by design.
* **A diverse baseline is not guaranteed in habitual driving.** E3 quantifies the dependence on diversity but cannot say how fast a real driver supplies distinct postures; a naturalistic collection is the missing experiment.
* **Small samples.** 22 drivers: CIs are wide, many secondary contrasts are positive in direction but unresolved after Holm correction, and subgroup claims (sex, BMI) are not possible.
* **Forking paths.** Cohort rule, region, window grid, operators and composition design were examined on the same drivers; mitigations are nested tuning of window / PCA / shrinkage (pilot), a pre-specified operator rule, an independent re-implementation and the 8-driver held-out sensitivity (pilot Table 9) - but no pristine confirmation sample exists in this dataset.
* **Operators were compared at 24 and 96 cells;** second-order alignment at 384 cells is untested. The CMVN variance floor and the whitening shrinkage were reported, not tuned.
* **Learners:** boosted trees and the CNN are fixed and untuned (direction-of-effect checks); AdaTime / Tent / DANN-type adaptation was read, not run (section 2.2).
* **The composition experiment is non-causal** and its "current posture excluded" variant uses the row's label (an upper bound); the label-free draw is the deployable analogue but is still not a causal system.
"""

REPRO = """
**Environment.** Python 3.12 (`CNC Turning (...)\\.venv`): numpy 2.5, scipy 1.18, pandas 3.0, scikit-learn 1.9, statsmodels 0.14, torch 2.6+cu124, xgboost 3.4 (GPU); GTX 1650 4 GB. pyRiemann 0.12 lives only in a scratch folder
(`pip install --no-deps --target <folder> pyriemann array_api_compat array_api_extra` with the system Python; the venv has no pip) and is used by the unit tests only. One heavy process at a time, 2 BLAS threads, below-normal priority,
GPU thermal pause at 82 C (`guard.py`); every experiment appends results per configuration and resumes after an interruption.

```
powershell -File analysis/scripts/run_followup.ps1 -steps e1,refs,e2,e3,e7,e4,ranktol,e2sens,sum     # needs the pilot's cache (run_chain.ps1)
python analysis/scripts/55_verify_operators.py <folder with pyriemann>                                  # operator unit tests -> results/operator_unit_tests.txt
python analysis/scripts/58_figures.py                                                                   # figures/S1-S3
python analysis/scripts/62_build_record.py                                                              # regenerates this file from results/*.csv
```

**Data:** only the DPR dataset and the pilot's cache; the study uses no external data.

| file | content |
|---|---|
| `results/e1_independent_replication.csv`, `e1_summary.csv` | clean-room replication (per driver) and its comparison with the pilot |
| `results/e2_operators.csv`, `e2_summary.csv`, `e2_decision_rule.csv`, `e2_rank_tol_check.csv`, `operator_unit_tests.txt` | operator comparison, decision rule, rank-tolerance diagnostic, unit tests |
| `results/e3_composition.csv`, `e3_summary.csv`, `e3_mixed_model.csv`, `e3_window_composition.csv` | seconds vs distinct postures, 10-shuffle ladder |
| `results/e7_time_adjusted.csv`, `e7_levels.csv`, `e7_summary.csv` | incremental value of pressure beyond elapsed time |
| `results/e4_cnn_adabn.csv`, `e4_summary.csv` | CNN and AdaBN |
| `results/refs_verified.csv` | Crossref verification of the evidence base |
| `figures/S1_replication_and_operators.png`, `S2_seconds_vs_postures.png`, `S3_validity_and_learners.png` | figures |
"""

DEFINITION = """
**Research question.** For a seat-pressure posture classifier that must serve drivers it was not trained on, how much unseen-driver accuracy does a label-free,
driver-relative baseline recover; what does that baseline need - seconds of data or distinct postures; is mean subtraction the justified operator or do the
normalisation / alignment methods of the speech, EEG and domain-adaptation literatures add enough to earn their complexity; and does the effect survive
validity checks against the dataset's fixed-order protocol?

| | Definition |
|---|---|
| **Unit of analysis** | the held-out driver (22 drivers with consistent label tables; leave-one-driver-out); frames are pseudo-replicates |
| **Primary estimand** | paired difference in unseen-driver accuracy: label-free running baseline (labelled segments in **random order**, 10 shuffles) minus no calibration; shrinkage LDA, 384 cells (4x4 pooled, columns 64-160), 3-s integration, rows >= 5 s into a posture |
| **Secondary estimands** | gain against the baseline's composition (k distinct postures x n seconds); gain of other operators over mean subtraction; incremental value of pressure beyond elapsed time; the same contrast for boosted trees and a CNN |
| **Calibration conditions** | L0 none; L1 causal running mean (W = 15-300 s); L1' whole-session mean (non-causal); oracle = mean of the driver's 20 posture means (needs every label) - a *first-order reference*, **not a ceiling** (section 5.3) |
| **In scope** | short-window classification of nominal postures from the two informative strips; calibration condition, window, composition and operator as factors |
| **Out of scope** | supervised enrolment (each posture is held once per driver, so enrolment and test frames share one posture instance); absolute deployment accuracy; named postures; new architectures; sensor-placement optimisation; on-road vibration; unsupervised domain adaptation that trains on the target driver (non-causal, one model per driver) |
| **Claims not made** | that 20-class accuracy of 20-30 % is a usable classifier; that gains transfer to habitual driving; that the 24-cell result specifies a sensor |

**Contribution.**

* **C1 - effect.** {C1}
* **C2 - mechanism.** {C2}
* **C3 - operator.** {C3}
* **C4 - validity recipe.** {C4}

**Fixed research gaps** (each narrow, supported by section 2):

* **G1 - no calibration curve.** In the seat / chair / mat pressure studies found (Tan 2001, Mutlu 2007, Zemp 2016, Zhao 2021, Nobari 2022, Zhao-Yan-Wang 2023, Odesola 2025, ChairPose 2025 and the DPR dataset itself) the unseen-user
  penalty is documented, but none reports how it depends on the amount *or composition* of the user's own unlabelled data.
* **G2 - operator transfer untested.** Sliding mean / variance normalisation (speech), Euclidean / Riemannian alignment (EEG), CORAL and AdaBN (domain adaptation) are established and have official
  implementations, but have not been benchmarked on static pressure maps, and EA's zero-mean assumption fails for them.
* **G3 - protocol validity.** Fixed-order, once-per-session protocols make posture a function of the time slot; the only open 30-driver seat dataset has no published evaluation, and no seat-pressure study reports an
  elapsed-time null or order-randomised baselines.
"""

DECISIONS = """
| # | Decision | Alternatives considered | Support | Outcome |
|---|---|---|---|---|
| 1 | The held-out **driver** is the unit; leave-one-driver-out; 95 % bootstrap CI over drivers, Wilcoxon + Holm inside a stated family; mixed models as a check | frame-level or grouped k-fold CV | pilot: the same random forest scores 93.3 % (random split), 53.0 % (blocked within driver), 18.7 % (unseen driver); references B | kept |
| 2 | The label-free baseline is computed on **randomly ordered** streams, with an **independent permutation per driver and seed**, 5-10 shuffles and common random numbers across compared conditions | recorded order; one permutation shared by all drivers | recorded order adds +2.7 pp (LDA) and +13.7 pp (boosted trees) of order information (pilot); E1: a shared permutation is a new fixed order; shuffle SD 0.6-0.9 pp (cohort mean), ~4 pp (per driver) | kept; 10 shuffles for the headline ladder |
| 3 | **Mean subtraction is the primary operator** unless an alternative clears a pre-specified rule: >= 1.0 pp over mean subtraction with Holm p < 0.05 at 60 s and 120 s at 24 and 96 cells | CMVN, CORAL-type whitening, Euclidean alignment as published | E2 (section 5.3); rule fixed after the first 24-cell runs had been seen, not changed afterwards | mean subtraction stays primary; CMVN / whitening reported as secondary operators |
| 4 | Hyperparameters of the alternatives are **reported, not tuned**: whitening shrinkage 0.1 / 0.25 / 0.5, CMVN variance floor 0.03 / 0.1 / 0.3 of the mean SD | tuning inside each fold | tuning would favour the more flexible operators; sensitivity run outside the decision family (section 5.3) | reported |
| 5 | LDA with an opt-in **numerical-rank tolerance** (1e-6) in every follow-up experiment | none (the pilot's setting) | a precaution after an 8-driver diagnostic showed fragility for per-cell scaling; the tolerance is the analogue of scikit-learn's `tol`; on the full cohort it turned out to be immaterial (`results/e2_rank_tol_check.csv`, section 5.3) | applied to all operators alike; pilot results untouched |
| 6 | **Shrinkage LDA is the primary learner**; boosted trees and a CNN are robustness checks, fixed and untuned | tuned deep models | LDA is transparent and its weights map to cells; the claim is the direction of the calibration effect, not learner ranking | kept |
| 7 | **No deep unsupervised domain adaptation** (DANN, Deep CORAL, Tent) in the comparison; AdaBN is the deep label-free comparator | AdaTime / Tent official code | they train or optimise on the target stream (per-driver model, not available from the first frame), model selection needs target labels, and ERM with simple statistics is hard to beat under realistic selection (Gulrajani & Lopez-Paz 2021; Ragab et al. 2023) | excluded, stated as a scope limit |
| 8 | Headline at **384 cells**; operator comparison at **24 and 96 cells** | tuning the pooling factor by nested CV | second-order statistics need many more effective frames than cells (a 60-s window holds ~20 non-overlapping 3-s samples); at 384 cells covariance alignment is ill-conditioned | second-order operators untested at 384 cells - a stated limit |
| 9 | Composition experiment draws the baseline **non-causally** from the driver's own data, with two variants: current posture excluded (uses the label - an upper bound that isolates diversity) and a label-free draw | only the causal running window | in a running window seconds and diversity are inseparable (it spans 1.6 / 2.3 / 3.6 / 5.9 / 10.0 distinct postures at 15 / 30 / 60 / 120 / 300 s) | both reported; the label-free draw is the deployable analogue |
| 10 | **Cohort** = 22 consistent drivers; repaired labels and all-30 as sensitivity (pilot) | all 30 repaired | independence from the repair rules (pilot Table 9: gain +5.3 to +7.6 pp across cohorts and settle times) | unchanged |
| 11 | **Elapsed-time null and an incremental test** (boosted trees on pressure + time vs time) | none | fixed protocol order makes posture a function of time | E7 |
"""

DATA = """
* **Cohort:** the 22 drivers whose label tables are internally consistent and complete (all but 2, 3, 4, 6, 8, 22, 24, 28); their labels are identical in the original and repaired tables, so nothing depends on the repair rules.
  The 8 excluded drivers (labels repaired with the dataset's own marker presses) are the pilot's sensitivity set.
* **Signal:** mat columns 64-160 (94 % of the sensed force) pooled 4x4 -> 16x24 = 384 cells (8x8 -> 96 and 16x16 -> 24 cells where stated); marker-press frames (+-10 frames) removed; 14,921 evaluation rows
  (every 6th frame, >= 5 s into a posture, first 10 s of P1 reserved); 20 classes (chance 5 %).
* **Features:** Z = causal 3-s mean; calibration x = op(Z; statistics of the driver's own unlabelled frames). Random-order streams: the driver's labelled segments are concatenated in an order drawn from
  `default_rng([seed, driver])` - **an independent permutation per driver and seed** (a permutation shared by all drivers is a new fixed protocol order and brings order information back; the E1 unit test exposed exactly this).
* **Learner:** standardise -> PCA(150) -> shrinkage LDA; numerical-rank tolerance 1e-6 in all follow-up experiments (section 4, decision 5). XGBoost (80 trees, depth 4) and a 3-layer CNN as robustness learners.
* **Statistics:** per-driver accuracy (mean over shuffles / seeds), 95 % percentile bootstrap over drivers (10,000), Wilcoxon signed-rank on paired differences with Holm correction inside each stated family, random-intercept mixed models as a
  robustness check, one-sided bootstrap bound for non-inferiority (pilot). Order confound (pilot): an elapsed-time-only random forest reaches 29.5 % [24.1, 34.9] on unseen drivers, above every uncalibrated pressure model.
"""
