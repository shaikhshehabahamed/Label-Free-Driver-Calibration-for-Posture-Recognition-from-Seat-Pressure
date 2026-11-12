"""Building blocks for the follow-up experiments (E2 adaptation operators, E3 baseline composition, E7 time-adjusted validity).

Data access and the random-order stream mirror `50_independent_replication.py`, whose feature builders were shown to be bit-identical to the
pilot's (`ladder.py`).  Everything is per driver; statistics are causal (window = last W seconds, expanding at the start of the stream, i.e. Kaldi's
`apply-cmvn-sliding` in non-centred mode with min-window 0) unless the scope is 'session'.

Operators (x = op(Z_t; statistics of the driver's own unlabelled frames); Z_t = 3-s causal mean of the frames):
  sub    x = Z - m                                   mean subtraction (cepstral mean normalisation; the pilot's baseline)
  cmvn   x = (Z - m) / sqrt(v + (0.1 s)^2)           + per-cell scaling (cepstral mean-variance normalisation); s = cell-average SD (floor against unloaded cells)
  wh     x = C_a^{-1/2} (Z - m)                      + full whitening (CORAL, Sun et al. 2016: covariance + regularisation, here shrinkage towards a scaled identity)
  ea     x = R_a^{-1/2} Z,  R = C + m m^T            Euclidean alignment as published (He & Wu 2020): UNCENTRED second moment, no mean subtraction
with C_a = (1 - a) C + a tr(C)/d I (shrinkage a).  Statistics come from the frame-level data (normalise frames, then integrate: the operators are linear).
"""
import guard  # noqa: F401  (thread caps, priority) - first
import numpy as np
import pandas as pd
import torch

from common import CACHE, RES

FPS = 13.6
STRIDE, SETTLE_S, P1_ENROL_S, WI_S, OCC_N, MARK_PAD = 6, 5.0, 10.0, 3.0, 250.0, 10
COLS = (64, 160)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def roll_mean(P, w):
    """causal mean over the last w frames, expanding at the start (pandas; identical to the pilot's cumulative-sum code)"""
    return pd.DataFrame(P).rolling(int(w), min_periods=1).mean().to_numpy(dtype=np.float32)


class Driver:
    def __init__(self, u, fr, lab_all, mark_all, pm, f, cols=COLS):
        idx = np.flatnonzero(fr.u.to_numpy() == u)
        self.u, self.s, self.e = u, idx[0], idx[-1] + 1
        assert self.e - self.s == len(idx)
        self.P = np.asarray(pm[self.s:self.e][:, :, cols[0] // f:cols[1] // f], np.float32).reshape(len(idx), -1)
        self.d = self.P.shape[1]
        self.fid = fr.frame_id.to_numpy()[self.s:self.e]
        self.lab = lab_all[self.s:self.e]
        self.off = self.fid - fr.seg_start.to_numpy()[self.s:self.e]
        self.force = fr.force_N.to_numpy()[self.s:self.e]
        self.mk = mark_all[self.s:self.e]
        ok = (self.lab > 0) & (self.fid % STRIDE == 0) & (self.off >= int(round(SETTLE_S * FPS)))
        ok &= ~((self.lab == 1) & (self.off < int(round(P1_ENROL_S * FPS))))
        self.rows = np.flatnonzero(ok)                                       # local indices of the evaluation rows
        self.y = self.lab[self.rows]
        self.t_s = (self.fid[self.rows] - self.fid[self.lab > 0].min()) / FPS   # elapsed time since the first labelled frame (the order-only floor feature)
        self.Zfull = roll_mean(self.P, round(WI_S * FPS))                     # 3-s causal integration over the full recorded stream
        self.occ = (self.force > OCC_N) & ~self.mk

    def shuffled_stream(self, rng):
        """labelled segments (label order 1..20) in a random order drawn from rng: returns (frame order idx, stream position of every evaluation row)"""
        segs = [np.flatnonzero(self.lab == p) for p in np.unique(self.lab[self.lab > 0])]
        order = rng.permutation(len(segs))
        idx = np.concatenate([segs[i] for i in order])
        pos = np.empty(len(self.P), np.int64)
        pos[idx] = np.arange(len(idx))
        return idx, pos[self.rows]


def load_drivers(f=4, cols=COLS, users=None):
    """the 22-driver primary cohort (labels consistent), repaired labels, marker-press frames removed"""
    fr = pd.read_csv(CACHE / "frames.csv")
    ev = pd.read_csv(RES / "marker_events.csv")
    lab_all = fr.label.to_numpy(copy=True)
    mark_all = np.zeros(len(fr), bool)
    for u_, e_ in ev.groupby("u"):
        iu = np.flatnonzero(fr.u.to_numpy() == u_)
        fid = fr.frame_id.to_numpy()[iu]
        m = np.zeros(len(iu), bool)
        for s_, t_ in zip(e_.start.to_numpy(), e_.end.to_numpy()):
            m |= (fid >= s_ - MARK_PAD) & (fid <= t_ + MARK_PAD)
        mark_all[iu] = m
    lab_all[mark_all & (lab_all > 0)] = -2
    lr = pd.read_csv(RES / "labels_repaired.csv")
    cohort = [int(u) for u, g in lr.groupby("u") if len(g) == 20 and g.use.all() and (g.status == "ok").all()]
    pm = np.load(CACHE / f"pooled_f{f}.npy", mmap_mode="r")
    return [Driver(u, fr, lab_all, mark_all, pm, f, cols) for u in (users or cohort)]


# ------------------------------------------------------------------------------------------------ causal statistics on the GPU
def causal_mean_t(S, w):
    """causal mean of the last w frames of S (n,d), expanding at the start (float64 cumulative sums, as in the pilot)"""
    n = S.shape[0]
    cs = torch.cumsum(S.double(), 0)
    ar = torch.arange(n, device=S.device)
    lo = ar - w
    shifted = torch.where((lo >= 0)[:, None], cs[lo.clamp(min=0)], torch.zeros_like(cs))
    cnt = torch.minimum(ar + 1, torch.full_like(ar, w)).double()
    return ((cs - shifted) / cnt[:, None]).float()


def window_stats(S, prow, w, cov=True, batch=32):
    """mean (R,d) and covariance (R,d,d) of the last w frames of the stream S (n,d) at the stream positions prow (R,); window expands at the start."""
    ar = torch.arange(w, device=S.device)
    ms, cs = [], []
    for i in range(0, len(prow), batch):
        p = prow[i:i + batch]
        ix = p[:, None] - ar[None, :]
        valid = (ix >= 0).to(S.dtype)
        X = S[ix.clamp(min=0)]
        cnt = valid.sum(1)
        m = (X * valid[..., None]).sum(1) / cnt[:, None]
        ms.append(m)
        if cov:
            Xc = (X - m[:, None, :]) * valid[..., None]
            cs.append(torch.einsum("bwd,bwe->bde", Xc, Xc) / cnt[:, None, None])
    return torch.cat(ms), (torch.cat(cs) if cov else None)


def _apply_eig(lam, V, T, alpha):
    """M_a^{-1/2} T from the eigendecomposition M = V diag(lam) V^T, M_a = (1-alpha) M + alpha tr(M)/d I.  Batched (lam (R,d), V (R,d,d), T (R,d))
    or single (lam (d,), V (d,d), T (R,d)); float64 on the CPU (GTX 1650 float64 is ~32x slower and heats the laptop)."""
    mu = lam.sum(-1, keepdims=True) / lam.shape[-1]
    lam_a = np.maximum((1 - alpha) * np.maximum(lam, 0) + alpha * mu, 1e-12)
    if lam.ndim == 1:
        return ((T @ V) / np.sqrt(lam_a) @ V.T).astype(np.float32)
    y = np.einsum("rde,rd->re", V, T) / np.sqrt(lam_a)
    return np.einsum("rde,re->rd", V, y).astype(np.float32)


def _inv_sqrt_apply(M, T, alpha):
    """M (R,d,d) symmetric PSD, T (R,d): M_a^{-1/2} T (torch in / torch out; used by the unit tests)"""
    lam, V = np.linalg.eigh(np.asarray(M, np.float64))
    return torch.as_tensor(_apply_eig(lam, V, np.asarray(T, np.float64), alpha))


def apply_operator(op, Z, m, C, alpha=0.1):
    """Z (R,d) integrated frames; m (R,d) or (d,), C (R,d,d) or (d,d): statistics of the driver's unlabelled frames (torch tensors on one device)."""
    if op == "none":
        return Z
    if op == "sub":
        return Z - m
    if op == "cmvn":
        v = torch.diagonal(C, dim1=-2, dim2=-1).clamp(min=0)
        floor = (alpha if alpha > 0 else 0.1) * v.sqrt().mean(-1, keepdim=True)          # alpha = variance-floor fraction of the cell-average SD (default 0.1)
        return (Z - m) / torch.sqrt(v + floor ** 2)
    Zn, mn, Cn = (np.asarray(x.double().cpu()) for x in (Z, m, C))
    if op == "wh":
        M, T = Cn, Zn - mn
    elif op == "ea":
        M = Cn + (mn[..., :, None] * mn[..., None, :])
        T = Zn
    else:
        raise ValueError(op)
    lam, V = np.linalg.eigh(M)
    return torch.as_tensor(_apply_eig(lam, V, T, alpha))


def statistics(drv, scope, W, rng, need_cov):
    """(Z, m, C) of all evaluation rows of a driver on the CPU.  scope 'run': causal window of W s over the labelled segments in random order drawn
    from rng (Z = 3-s integrated stream); 'session': statistics of the occupied frames of the whole recording (non-causal; m, C single)."""
    if scope == "session":
        Po = torch.as_tensor(drv.P[drv.occ], device=DEV)
        m = Po.mean(0)
        C = ((Po - m).T @ (Po - m)) / Po.shape[0] if need_cov else None
        return torch.as_tensor(drv.Zfull[drv.rows]), m.cpu(), (C.cpu() if need_cov else None)
    idx, prow = drv.shuffled_stream(rng)
    S = torch.as_tensor(drv.P[idx], device=DEV)
    Zs = causal_mean_t(S, max(1, int(round(WI_S * FPS))))
    pr = torch.as_tensor(prow, device=DEV)
    m, C = window_stats(S, pr, max(1, int(round(W * FPS))), cov=need_cov)
    return Zs[pr].cpu(), m.cpu(), (C.cpu() if need_cov else None)


def features_multi(drv, ops, scope, W=60, rng=None):
    """features of all evaluation rows of a driver for several operators from ONE set of statistics.  ops: list of (op, alpha);
    scope 'full' (op none), 'oracle' (mean of the 20 posture means, op sub), 'session', 'run'.  Returns {(op, alpha): X float32}.
    The eigendecomposition is shared by all alphas of an operator."""
    if scope == "oracle":
        b = np.mean([drv.P[drv.lab == p].mean(0) for p in range(1, 21) if (drv.lab == p).any()], axis=0)
        return {("sub", 0.0): (drv.Zfull[drv.rows] - b).astype(np.float32)}
    if scope == "full":
        return {("none", 0.0): drv.Zfull[drv.rows].astype(np.float32)}
    need_cov = any(op not in ("none", "sub") for op, _ in ops)
    Z, m, C = statistics(drv, scope, W, rng, need_cov)
    out, eig = {}, {}
    for op, a in ops:
        if op in ("sub", "cmvn"):
            out[(op, a)] = apply_operator(op, Z, m, C, a).numpy()
            continue
        kind = "C" if op == "wh" else "R"
        if kind not in eig:
            Cn, mn = C.double().numpy(), m.double().numpy()
            eig[kind] = np.linalg.eigh(Cn if kind == "C" else Cn + (mn[..., :, None] * mn[..., None, :]))
        lam, V = eig[kind]
        T = (Z - m if op == "wh" else Z).double().numpy()
        out[(op, a)] = _apply_eig(lam, V, T, a)
    return out


def stack(drivers, fn):
    """features for all drivers: X (n,d), y (n,), u (n,)"""
    X = np.concatenate([fn(d) for d in drivers])
    y = np.concatenate([d.y for d in drivers])
    u = np.concatenate([np.full(len(d.rows), d.u) for d in drivers])
    return X, y, u


def append_rows(path, df):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
