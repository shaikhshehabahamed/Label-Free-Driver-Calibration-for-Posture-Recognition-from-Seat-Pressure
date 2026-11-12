"""Verify every reference of the evidence base against Crossref (DOI lookup; title search where no DOI is known).

Writes results/refs_verified.csv: key, status (OK | CHECK | NOT_FOUND), expected vs Crossref title, first author, year, venue, DOI, title similarity.
A reference is only cited in the study record when its status is OK (or it is listed as verified by hand from the publisher / arXiv page).
usage: python 51_verify_refs.py
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import RES

# key: (doi or None, expected title (fragment), first author family, year)
REFS = {
    # seat / chair / mat pressure posture recognition
    "tan2001": ("10.1109/3516.951364", "A sensing chair using pressure distribution sensors", "Tan", 2001),
    "mutlu2007": ("10.1145/1294211.1294237", "Robust, low-cost, non-intrusive sensing and recognition of seated postures", "Mutlu", 2007),
    "zemp2016": ("10.1155/2016/5978489", "Application of machine learning approaches for classifying sitting posture based on force and acceleration sensors", "Zemp", 2016),
    "zhao2021": ("10.3390/s21103346", "Exploration of driver posture monitoring using pressure sensors with lower resolution", "Zhao", 2021),
    "nobari2022": ("10.54941/ahfe1002733", "Generalized model for driver activity recognition in automated vehicles using pressure sensor array", "Nobari", 2022),
    "mahomed2024": ("10.1109/ACCESS.2024.3496578", "Driver posture recognition: a review", "Mahomed", 2024),
    "odesola2025": ("10.3390/s25185610", "A smart system for continuous sitting posture monitoring, assessment, and personalized feedback", "Odesola", 2025),
    "licciardo2023": ("10.1117/12.2665613", "Low-power CNN for real-time driver posture monitoring by image processing", "Licciardo", 2023),
    "riener2008": ("10.4108/ICST.ISVCS2008.3545", "Supporting implicit human-to-vehicle interaction: driver identification from sitting postures", "Riener", 2008),
    "caggiari2023": ("10.3390/s23156872", "Optimization of spatial and temporal configuration of a pressure sensing array to predict posture and mobility in lying", "Caggiari", 2023),
    "hwang2025": ("10.1049/ell2.70305", "Pressure sensor array resolution and sleep posture classification", "Hwang", 2025),
    "peruzzi2025": ("10.3390/s25020458", "Sleep posture detection via embedded machine learning on a reduced set of pressure sensors", "Peruzzi", 2025),
    "appsci2025_seat": ("10.3390/app152312744", "Deep Learning-Based Sitting Posture Recognition from Pressure Distribution Across Hard and Soft Seat Environment", "", 2025),
    "postureware": (None, "Smart cushion: A practical system for fine-grained sitting posture recognition", "", None),
    "zhao2023jee": ("10.2478/jee-2023-0055", "A comparative study on real-time sitting posture monitoring systems using pressure sensors", "Zhao", 2023),
    "zhenhai2009": ("10.4271/2009-01-1254", "Vehicle Occupant Posture Classification System using Seat Pressure Sensor for Intelligent Airbag", "", 2009),
    "crawford2005": ("10.1016/j.apmr.2004.08.010", "Impact of sitting time on seat-interface pressure and on pressure mapping with multiple sclerosis patients", "Crawford", 2005),
    "kyung2008": ("10.1016/j.ergon.2007.08.011", "Driver sitting comfort and discomfort (part II): Relationships with and prediction from interface pressure", "Kyung", 2008),
    # evaluation validity / protocol
    "saeb2017": ("10.1093/gigascience/gix019", "The need to approximate the use-case in clinical machine learning", "Saeb", 2017),
    "little2017": ("10.1093/gigascience/gix020", "Using and understanding cross-validation strategies", "Little", 2017),
    "kapoor2023": ("10.1016/j.patter.2023.100804", "Leakage and the reproducibility crisis in machine-learning-based science", "Kapoor", 2023),
    "hammerla2015": ("10.1145/2750858.2807551", "Let's (not) stick together: pairwise similarity biases cross-validation in activity recognition", "Hammerla", 2015),
    "dehghani2019": ("10.3390/s19225026", "A quantitative comparison of overlapping and non-overlapping sliding windows for human activity recognition using inertial sensors", "Dehghani", 2019),
    "varoquaux2018": ("10.1016/j.neuroimage.2017.06.061", "Cross-validation failure: small sample sizes lead to large error bars", "Varoquaux", 2018),
    "roberts2017": ("10.1111/ecog.02881", "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", "Roberts", 2017),
    "bates2024": ("10.1080/01621459.2023.2197686", "Cross-validation: what does it estimate and how well does it do it?", "Bates", 2024),
    "gholami2020": ("10.1109/ACCESS.2020.3010715", "Deep neural networks for human activity recognition with wearable sensors: leave-one-subject-out cross-validation for model selection", "Gholamiangonabadi", 2020),
    "chen2021csur": ("10.1145/3447744", "Deep learning for sensor-based human activity recognition: overview, challenges, and opportunities", "Chen", 2021),
    # label-free adaptation / normalisation operators and their evaluation
    "hewu2020": ("10.1109/TBME.2019.2913914", "Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach", "He", 2020),
    "zanini2018": ("10.1109/TBME.2017.2742541", "Transfer learning: a Riemannian geometry framework with applications to brain-computer interfaces", "Zanini", 2018),
    "rodrigues2019": ("10.1109/TBME.2018.2889705", "Riemannian Procrustes analysis: transfer learning for brain-computer interfaces", "Rodrigues", 2019),
    "adabn2018": ("10.1016/j.patcog.2018.03.005", "Adaptive batch normalization for practical domain adaptation", "Li", 2018),
    "coral2016": (None, "Return of frustratingly easy domain adaptation", "Sun", 2016),
    "adatime2023": ("10.1145/3587937", "ADATIME: A benchmarking suite for domain adaptation on time series data", "Ragab", 2023),   # 10.1145/3563781 (first recollection) does not exist
    "codats2020": ("10.1145/3394486.3403228", "Multi-source deep domain adaptation with weak supervision for time-series sensor data", "Wilson", 2020),
    "viikki1998": ("10.1016/S0167-6393(98)00033-8", "Cepstral domain segmental feature vector normalization for noise robust speech recognition", "Viikki", 1998),
    "atal1974": ("10.1121/1.1914702", "Effectiveness of linear prediction characteristics of the speech wave for automatic speaker identification and verification", "Atal", 1974),
    "rasta1994": ("10.1109/89.326616", "RASTA processing of speech", "Hermansky", 1994),
    "pujol2006": (None, "On real-time mean-and-variance normalization of speech recognition features", "Pujol", 2006),
    "gauvain1994": ("10.1109/89.279278", "Maximum a posteriori estimation for multivariate Gaussian mixture observations of Markov chains", "Gauvain", 1994),
    "liang2025tta": ("10.1007/s11263-024-02181-w", "A comprehensive survey on test-time adaptation under distribution shifts", "Liang", 2025),
    "jacobe2018": ("10.1016/j.aap.2018.08.017", "Adapting artificial neural networks to a specific driver enhances detection and prediction of drowsiness", "Jacob", 2018),
    "wei2018": ("10.1016/j.neuroimage.2018.03.032", "A subject-transfer framework for obviating inter- and intra-subject variability in EEG-based drowsiness detection", "Wei", 2018),
    "ferrari2020": ("10.1109/ACCESS.2020.2973425", "On the personalization of classification models for human activity recognition", "Ferrari", 2020),
    "stisen2015": ("10.1145/2809695.2809718", "Smart devices are different: assessing and mitigating mobile sensing heterogeneities for activity recognition", "Stisen", 2015),
    # statistics and learners
    "nadeau2003": ("10.1023/A:1024068626366", "Inference for the generalization error", "Nadeau", 2003),
    "lakens2017": ("10.1177/1948550617697177", "Equivalence tests: a practical primer for t tests, correlations, and meta-analyses", "Lakens", 2017),
    "fieldwelsh2007": ("10.1111/j.1467-9868.2007.00593.x", "Bootstrapping clustered data", "Field", 2007),
    "bates2015lme4": ("10.18637/jss.v067.i01", "Fitting linear mixed-effects models using lme4", "Bates", 2015),
    "ledoit2004": ("10.1016/S0047-259X(03)00096-4", "A well-conditioned estimator for large-dimensional covariance matrices", "Ledoit", 2004),
    "chen2016xgb": ("10.1145/2939672.2939785", "XGBoost: a scalable tree boosting system", "Chen", 2016),
    "brunton2016sspoc": ("10.1137/15M1036713", "Sparse sensor placement optimization for classification", "Brunton", 2016),
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "reference-verification-script/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def describe(m):
    title = (m.get("title") or [""])[0]
    au = (m.get("author") or [{}])[0].get("family", "")
    dp = (m.get("issued") or m.get("published") or {}).get("date-parts", [[None]])[0]
    return dict(cr_title=title, first_author=au, year=dp[0], venue=(m.get("container-title") or [""])[0], doi=m.get("DOI", ""))


rows = []
for key, (doi, title, au, yr) in REFS.items():
    rec = dict(key=key, expected_title=title, expected_doi=doi or "")
    try:
        if doi:
            m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))["message"]
        else:
            items = get("https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(title))["message"]["items"]
            m = max(items, key=lambda it: sim(title, (it.get("title") or [""])[0])) if items else None
        if m is None:
            rec["status"] = "NOT_FOUND"
        else:
            rec.update(describe(m))
            s = sim(title, rec["cr_title"])
            rec["title_sim"] = round(s, 3)
            au_ok = not au or au.lower() in rec["first_author"].lower() or not rec["first_author"]
            yr_ok = yr is None or abs((rec["year"] or yr) - yr) <= 1
            short = bool(doi) and (norm(rec["cr_title"]) in norm(title) or norm(title) in norm(rec["cr_title"]))   # one title contains the other (subtitle dropped / added); DOI, author and year must still match
            rec["status"] = "OK" if (s >= 0.80 and au_ok and yr_ok) else ("OK_short_title" if (short and au_ok and yr_ok) else "CHECK")
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"ERROR {type(e).__name__}"
    rows.append(rec)
    print(f"{rec['status']:9s} {key:16s} {rec.get('first_author', ''):14s} {rec.get('year', '')} {str(rec.get('venue', ''))[:45]:45s} sim {rec.get('title_sim', '')}", flush=True)
    time.sleep(0.15)

# items that are not in Crossref (arXiv / JMLR / workshop papers / code repositories): verified by hand from the publisher, arXiv or GitHub page on 2026-10-06
MANUAL = {
    "benavoli2017": "Benavoli A, Corani G, Demsar J, Zaffalon M. Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. JMLR 18(77):1-36, 2017 - jmlr.org/papers/v18/16-305.html",
    "riener2007": "Riener A, Ferscha A. Driver activity recognition from sitting postures. Mensch & Computer 2007 Workshopband (Automotive User Interfaces) - dl.gi.de/items/da4c804c-538f-45ab-9db2-40b01323ce86",
    "gulrajani2021": "Gulrajani I, Lopez-Paz D. In search of lost domain generalization. ICLR 2021 - arXiv:2007.01434 (ERM matches or beats 9 DG algorithms on 7 benchmarks under careful model selection)",
    "tent2021": "Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. Tent: fully test-time adaptation by entropy minimization. ICLR 2021 - arXiv:2006.10726; official code github.com/DequanWang/tent (not run: deployable-from-day-one requirement, see study record)",
    "chairpose2025": "ChairPose: pressure-based chair-morphology-grounded sitting pose estimation through simulation-assisted training - arXiv:2508.01850 (8 users x 4 chairs; physics-based augmentation for unseen users/chairs; no user calibration)",
    "papillon2025": "Papillon O et al. Sleep position classification using transfer learning for bed-based pressure sensors - arXiv:2505.08111 (pre-trained ViT; cross-subject; no per-user calibration curve)",
    "coral_code": "Sun B, Saenko K. CORAL official code github.com/VisionLearningGroup/CORAL, code/CORAL_Source_Code.zip, Ex_CORAL.m: z-score per domain, cov + eye, A = Cs^(-1/2) Ct^(1/2) (read 2026-10-06)",
    "ea_code": "He H, Wu D. official MATLAB code github.com/hehe91/EA (README: official implementation of the IEEE TBME paper; main_MI.m, CSPfeature.m)",
    "pyriemann": "pyRiemann 0.12 (github.com/pyRiemann/pyRiemann): transfer module TLCenter/TLScale/TLRotate (Riemannian Procrustes analysis), geometry.base.invsqrtm - used to unit-test the whitening / EA operators (55_verify_operators.py)",
    "kaldi_cmvn": "Kaldi apply-cmvn-sliding / SlidingWindowCmn (src/feat/feature-functions.cc): causal window t-cmn_window..t, expanding at the start, optional variance normalisation (read 2026-10-06)",
    "adatime_code": "Ragab E et al. AdaTime official code github.com/emadeldeen24/AdaTime (MIT; 12 UDA methods: DeepCORAL, MMDA, DANN, CDAN, DIRT-T, DSAN, HoMM, DDC, CoDATS, AdvSKM, SASA, CoTMix)",
    "dpr_zenodo": "Liguori R et al. DPR dataset, Zenodo doi:10.5281/zenodo.15199100, published 2025-04-11; no related identifiers; DataCite citationCount 0 on 2026-10-06",
}
rows += [dict(key=k, status="MANUAL", expected_title=v) for k, v in MANUAL.items()]
out = pd.DataFrame(rows)
out.to_csv(RES / "refs_verified.csv", index=False)
print(out.status.value_counts().to_string())
