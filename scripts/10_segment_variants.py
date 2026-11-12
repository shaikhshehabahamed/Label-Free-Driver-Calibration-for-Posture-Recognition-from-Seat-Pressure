"""Segment-level view (ONE mean map per driver x posture, so there is no frame redundancy): LOSO nearest-centroid accuracy for operator,
resolution and region.  Analysis cohort, columns 64-160 unless a region is named.   usage: [DATASET=clean] python 10_segment_variants.py
Centres: none | oracle (mean of the driver's own postures) | ref1, ref10 (the driver's own map of posture 1 or 10 as the single reference)."""
import numpy as np
import pandas as pd

from common import CACHE, DEFAULT_COLS, REGIONS, RES, cohort_users, tag

seg = np.load(CACHE / "segmean_fullres.npy")                    # (30, 21, 64, 160)
have = seg.sum(axis=(2, 3)) > 0
have[:, 0] = False
keep = np.zeros(30, bool)
keep[[u - 1 for u in cohort_users()]] = True
have[~keep] = False
SUB = [s for s in range(30) if keep[s]]


def pool(a, f, cols):
    a = a[..., cols[0]:cols[1]]
    return a.reshape(*a.shape[:-2], 64 // f, f, (cols[1] - cols[0]) // f, f).mean(axis=(-3, -1))


def make(F, mode, op, ref=None):
    """F (30,21,h,w).  baseline b per driver; returns centred array (nan where unavailable)."""
    out = np.full(F.shape, np.nan, np.float64)
    for s in SUB:
        ps = np.where(have[s])[0]
        if mode == "none":
            b = np.zeros(F.shape[2:])
        elif mode == "oracle":
            b = F[s, ps].mean(axis=0)
        else:
            if not have[s, ref]:
                continue
            b = F[s, ref]
        x = F[s, ps]
        if mode == "none":
            out[s, ps] = x
        elif op == "sub":
            out[s, ps] = x - b
        elif op == "sub_scaled":
            out[s, ps] = (x - b) / (b.sum() + 1e-9) * 1e3
        elif op == "logratio":
            e = 0.05
            out[s, ps] = np.log(x + e) - np.log(b + e)
    return out


def loso_nc(Fc, metric="cosine"):
    Fc = Fc.reshape(30, 21, -1)
    ok = n = 0
    for s in SUB:
        cent = np.full((21, Fc.shape[2]), np.nan)
        for p in range(1, 21):
            m = [t for t in SUB if t != s and have[t, p] and not np.isnan(Fc[t, p]).any()]
            if m:
                cent[p] = Fc[m, p].mean(axis=0)
        okc = ~np.isnan(cent[1:, 0])
        for p in np.where(have[s])[0]:
            x = Fc[s, p]
            if np.isnan(x).any():
                continue
            C = np.nan_to_num(cent[1:])
            if metric == "euclid":
                d = ((C - x) ** 2).sum(axis=1)
            else:
                d = 1 - (C @ x) / (np.linalg.norm(C, axis=1) * np.linalg.norm(x) + 1e-12)
            d[~okc] = np.inf
            ok += int(np.argmin(d)) + 1 == p
            n += 1
    return ok / n


rows = []
for f in (2, 4, 8, 16):
    F = pool(seg, f, DEFAULT_COLS)
    for mode, ref in (("none", None), ("oracle", None), ("ref", 1), ("ref", 10)):
        for op in (("sub",) if mode in ("none",) else ("sub", "sub_scaled", "logratio")):
            Fc = make(F, mode, op, ref)
            rows.append(dict(f=f, centre=f"{mode}{ref or ''}", op=op, acc_cos=loso_nc(Fc, "cosine"), acc_euc=loso_nc(Fc, "euclid")))
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
df.to_csv(RES / f"segment_level_variants{tag()}.csv", index=False)

print("\nregion ablation (f=4, oracle subtraction, cosine):")
reg_rows = []
for name, cols in {**REGIONS, "back+seat": (64, 160), "whole mat": (0, 160)}.items():
    Fr = pool(seg, 4, cols)
    r_ = dict(region=name, cols=str(cols), none=loso_nc(make(Fr, 'none', 'sub'), 'cosine'), oracle=loso_nc(make(Fr, 'oracle', 'sub'), 'cosine'))
    reg_rows.append(r_)
    print(f"  {name:>10}: none {r_['none']:.3f} | oracle {r_['oracle']:.3f}")
pd.DataFrame(reg_rows).to_csv(RES / f"segment_level_region{tag()}.csv", index=False)
