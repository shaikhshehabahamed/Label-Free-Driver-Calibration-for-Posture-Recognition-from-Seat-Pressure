"""Parse the XSENSOR CSV exports (quoted fields, decimal comma) into compact arrays.

Per user -> analysis/cache/User{u}_maps.npy   float32 (T, 64, 160), N/cm^2
            analysis/cache/User{u}_meta.npz   frame_id, time_s, avg_p, min_p, peak_p, contact_cm2, contact_pct

The raw files are ~14 GB of text; this is the only script that touches them. Each file is streamed in
batches of frames (low memory) and parsed with pandas' C parser (np.fromstring is quadratic on long strings).
Usage:  python 00_parse_csv.py 1 2 3        (specific users)
        python 00_parse_csv.py all --workers 3
"""
import argparse
import io
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Dataset_Users"
DST = ROOT / "analysis" / "cache"
H, W = 64, 160
BATCH_FRAMES = 250

# header-line prefix -> key (all values are the 2nd quoted field)
PREFIX = {
    b'"FRAME"': "frame_id",
    b'"Time"': "time",
    b'"Average Pressure': "avg_p",
    b'"Minimum Pressure': "min_p",
    b'"Peak Pressure': "peak_p",
    b'"Contact Area': "contact_cm2",
    b'"Contact %"': "contact_pct",
}


def _value(ln: bytes) -> bytes:
    return ln.split(b'","', 1)[1].strip(b'"\r\n')


def _flush(buf, out):
    df = pd.read_csv(io.BytesIO(b"".join(buf)), sep=",", quotechar='"', decimal=",",
                     header=None, dtype=np.float32, engine="c")
    out.append(df.to_numpy().reshape(-1, H, W))


def parse_user(u: int):
    t0 = time.time()
    hdr = {k: [] for k in PREFIX.values()}
    buf, out = [], []
    with open(SRC / f"User{u}.csv", "rb") as f:
        for ln in f:
            if ln[1:2].isdigit():
                buf.append(ln)
                if len(buf) == H * BATCH_FRAMES:
                    _flush(buf, out)
                    buf = []
                continue
            for pre, key in PREFIX.items():
                if ln.startswith(pre):
                    hdr[key].append(_value(ln))
                    break
    if buf:
        _flush(buf, out)
    maps = np.concatenate(out, axis=0)
    n = len(hdr["frame_id"])
    if maps.shape != (n, H, W):
        raise RuntimeError(f"User{u}: maps {maps.shape} vs {n} frame headers")
    hms = np.array([[int(x) for x in v.split(b":")] for v in hdr["time"]])
    meta = dict(frame_id=np.array(hdr["frame_id"], dtype=np.int64),
                time_s=hms[:, 0] * 3600 + hms[:, 1] * 60 + hms[:, 2])
    for k in ("avg_p", "min_p", "peak_p", "contact_cm2", "contact_pct"):
        meta[k] = np.array([float(v.replace(b",", b".")) for v in hdr[k]])
    np.save(DST / f"User{u}_maps.npy", maps)
    np.savez(DST / f"User{u}_meta.npz", **meta)
    return u, n, time.time() - t0, float(maps.max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("users", nargs="+")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    users = list(range(1, 31)) if a.users == ["all"] else [int(x) for x in a.users]
    DST.mkdir(parents=True, exist_ok=True)
    todo = [u for u in users if not ((DST / f"User{u}_maps.npy").exists() and (DST / f"User{u}_meta.npz").exists())]
    print(f"parsing {len(todo)} of {len(users)} users (skipping cached)", flush=True)
    with Pool(a.workers) as pool:
        for u, n, dt, mx in pool.imap_unordered(parse_user, todo):
            print(f"User{u}: {n} frames, max {mx:.2f} N/cm2, {dt:.1f}s", flush=True)
