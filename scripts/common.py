"""Shared loaders for the analysis scripts.

Three analysis datasets, selected with the environment variable DATASET (default 'clean'); result files of the non-default ones get a suffix:
  clean  (default, PRIMARY)  the drivers whose label table is internally consistent and complete (22), repaired-table labels (identical to the
                             original labels for these drivers).  File suffix ''.
  all    (sensitivity)       all 30 drivers with the repaired labels (43_repair_labels.py).  Suffix '_all'.
  orig   (sensitivity)       all 30 drivers with the first cleaning (overlapping frames dropped, nested ranges lost).  Suffix '_orig'.
The sensing area used by default is columns 64..160 of the 64 x 160 mat (two strips with posture-related load patterns); see DEFAULT_COLS.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "analysis" / "cache"
RES = ROOT / "analysis" / "results"
FIG = ROOT / "analysis" / "figures"
for d in (RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

# column ranges of the physical regions of the 64x160 mat (from the activity map, 03_eda.py); all are multiples of 16 so that every pooling factor aligns
REGIONS = {"outer": (16, 64), "back": (64, 112), "seat": (112, 160)}
DEFAULT_COLS = (64, 160)          # primary sensing area: backrest + seat strips; the outer strip (columns 0-64) is excluded (see 3.x of the proposal)


def dataset() -> str:
    d = os.environ.get("DATASET", "clean")
    if d not in ("clean", "all", "orig"):
        raise ValueError(d)
    return d


def keep_marker_frames() -> bool:
    """KEEP_MARKER_FRAMES=1 keeps the frames inside marker presses in the labelled data (sensitivity analysis only)"""
    return os.environ.get("KEEP_MARKER_FRAMES", "0") == "1"


def tag() -> str:
    return ("" if dataset() == "clean" else f"_{dataset()}") + ("_withmarkers" if keep_marker_frames() else "")


def clean_drivers():
    """drivers whose labels are internally consistent (no overlap / nesting / reversal / truncation) and that have all 20 postures with data"""
    r = pd.read_csv(RES / "labels_repaired.csv")
    ok = []
    for u, g in r.groupby("u"):
        if len(g) == 20 and g.use.all() and (g.status == "ok").all():
            ok.append(int(u))
    return ok


def cohort_users():
    return clean_drivers() if dataset() == "clean" else list(range(1, 31))


def load_users() -> pd.DataFrame:
    """Anthropometrics (decimal-comma BMI) with a numeric user id u=1..30."""
    df = pd.read_csv(ROOT / "Info_Users.csv", sep=";", decimal=",", index_col=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = ["id", "sex", "height_cm", "weight_kg", "bmi"]
    df["u"] = df["id"].str.extract(r"(\d+)").astype(int)
    return df


MARKER_PAD = 10          # frames added on both sides of a detected marker press


def marker_mask(fr: pd.DataFrame, pad: int = MARKER_PAD) -> np.ndarray:
    """True for frames inside (or within `pad` frames of) a marker press - the participant/operator pressing four highlighted points before each posture
    (README of the dataset; detected by 48_markers.py).  All False until results/marker_events.csv exists."""
    m = np.zeros(len(fr), bool)
    p = RES / "marker_events.csv"
    if not p.exists():
        return m
    ev = pd.read_csv(p)
    u_arr, f_arr = fr.u.values, fr.frame_id.values
    for u, e in ev.groupby("u"):
        pos = np.flatnonzero(u_arr == u)
        fid = f_arr[pos]
        mm = np.zeros(len(pos), bool)
        for s, t in zip(e.start.values, e.end.values):
            mm |= (fid >= s - pad) & (fid <= t + pad)
        m[pos] = mm
    return m


def load_frames(name: str = None) -> pd.DataFrame:
    """Frame table.  Label columns follow the repaired label set ('orig' dataset: the first-cleaning table, `frames_orig_labels.csv`).
    Labelled frames inside a marker press get label -2 (excluded from every analysis) unless KEEP_MARKER_FRAMES=1; column `marker` flags them."""
    fr = pd.read_csv(CACHE / (name or ("frames_orig_labels.csv" if dataset() == "orig" else "frames.csv")))
    fr["marker"] = marker_mask(fr)
    if not keep_marker_frames():
        fr.loc[fr.marker & (fr.label > 0), "label"] = -2
    return fr


def load_labels(version: str = "repaired") -> pd.DataFrame:
    """Label ranges (u, p, start, end, n_label).  'repaired' = usable ranges after 43_repair_labels.py; 'raw' = as parsed from the Zenodo file."""
    if version == "raw":
        return pd.read_csv(RES / "labels_clean.csv")
    r = pd.read_csv(RES / "labels_repaired.csv")
    r = r[r.use].copy()
    r["n_label"] = r.end - r.start + 1
    return r[["u", "p", "start", "end", "n_label"]].reset_index(drop=True)


def load_pooled(f: int) -> np.ndarray:
    return np.load(CACHE / f"pooled_f{f}.npy", mmap_mode="r")


def model_data(f: int = 4, stride: int = 6, trim: int = 0, cols=DEFAULT_COLS):
    """Single-labelled frames of the analysis cohort, every `stride`-th frame_id, dropping `trim` frames at both ends of each segment.

    Returns (X (n, 64/f, (cols[1]-cols[0])/f) float32, meta DataFrame with u, label, frame_id, seg, dist_edge, row).
    """
    fr = load_frames()
    m = (fr.label > 0) & (fr.frame_id % stride == 0) & (fr.dist_edge >= trim) & fr.u.isin(cohort_users())
    meta = fr.loc[m, ["u", "label", "frame_id", "dist_edge", "seg_start", "seg_end", "force_N"]].copy()
    meta["row"] = meta.index.values
    meta["seg"] = meta.u * 100 + meta.label
    X = np.asarray(load_pooled(f)[meta.row.values][:, :, cols[0] // f:cols[1] // f], dtype=np.float32)
    return X, meta.reset_index(drop=True)
