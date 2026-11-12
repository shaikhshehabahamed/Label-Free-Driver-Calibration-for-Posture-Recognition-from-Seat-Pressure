"""Memory-light feature construction for the calibration-ladder study (one driver in RAM at a time).

Everything is causal and per driver:
    x_t = mean(maps over the last Wi seconds)  -  b_t          (b_t = driver baseline, see below)
    baseline specs
        none         b = 0
        run{W}       b_t = mean of occupied-seat frames in the last W seconds   (label-free, expanding at start-up)
        session      b = mean of occupied frames over the whole recording        (label-free, non-causal reference)
        oracle       b = mean of the driver's 20 posture-segment means           (needs all labels; upper reference)
        ref{r}_{E}   b = mean of the first E seconds of the driver's posture-r segment (one labelled reference posture)
Training and held-out drivers receive the identical treatment, and a driver's baseline never uses any other driver.
Features are returned ONLY for the requested rows (sorted global frame indices), so even the 32x80 resolution fits in memory.
"""
import numpy as np

from common import CACHE, DEFAULT_COLS, cohort_users, keep_marker_frames, load_frames

FPS = 13.6
OCC_N = 250.0            # occupied-seat gate: total sensed force above 250 N (fixed, driver independent)
E_MAX = int(round(10 * FPS))
STRIDE = 6


def causal_mean(P, w, m=None):
    """mean of P over the last w frames (t-w+1..t), restricted to frames with m True. returns (mean (T,d), count (T,))."""
    T = len(P)
    mm = np.ones(T) if m is None else m.astype(np.float64)
    cs = np.cumsum(P * mm[:, None], axis=0, dtype=np.float64)
    cn = np.cumsum(mm)
    lo = np.arange(T) - w
    S = cs - np.where((lo >= 0)[:, None], cs[np.maximum(lo, 0)], 0.0)
    N = cn - np.where(lo >= 0, cn[np.maximum(lo, 0)], 0.0)
    return (S / np.maximum(N, 1)[:, None]).astype(np.float32), N


class Stream:
    """Frame table + memory-mapped pooled maps at one pooling factor, restricted to a column range."""

    def __init__(self, f=4, cols=DEFAULT_COLS):
        self.fr = load_frames()
        self.fr["off"] = self.fr.frame_id - self.fr.seg_start
        self.pm = np.load(CACHE / f"pooled_f{f}.npy", mmap_mode="r")             # (N, 64/f, 160/f)
        self.sl = slice(cols[0] // f, cols[1] // f)
        self.f = f
        self.d = (64 // f) * (self.sl.stop - self.sl.start)
        self.occ = (self.fr.force_N.values > OCC_N) & ~(self.fr.marker.values & (not keep_marker_frames()))
        self.span = {u: (g.index[0], g.index[-1] + 1) for u, g in self.fr.groupby("u")}

    def raw(self, idx):
        """raw pooled frames (len(idx), d) float32 for arbitrary global frame indices."""
        return np.asarray(self.pm[np.asarray(idx)][:, :, self.sl], dtype=np.float32).reshape(len(idx), -1)

    def _user(self, u):
        s, e = self.span[u]
        return np.asarray(self.pm[s:e][:, :, self.sl], dtype=np.float32).reshape(e - s, -1)

    def _split_rows(self, rows):
        rows = np.asarray(rows)
        u_of = self.fr.u.values[rows]
        return {u: rows[u_of == u] for u in self.span}

    def features(self, rows, Wi=3.0, base="run60", op="sub", parts=False):
        """X (len(rows), d) float32 for the requested rows (NaN where the baseline is undefined).
        parts=True returns (Z, B): integrated maps and baselines separately (for descriptor features)."""
        rows = np.asarray(rows)
        fr = self.fr
        out_rows = {}
        wi = max(1, int(round(Wi * FPS)))
        for u, ru in self._split_rows(rows).items():
            if len(ru) == 0:
                continue
            s, e = self.span[u]
            loc = ru - s
            g = fr.iloc[s:e]
            Pu = self._user(u)
            Z = causal_mean(Pu, wi)[0] if Wi > 0 else Pu
            occ = self.occ[s:e]
            if base == "none":
                B = np.zeros_like(Z)
            elif base.startswith("run"):
                B, cnt = causal_mean(Pu, max(1, int(round(float(base[3:]) * FPS))), occ)
                B[cnt == 0] = np.nan
            elif base == "session":
                B = np.broadcast_to(Pu[occ].mean(axis=0), Z.shape)
            elif base == "oracle":
                segs = [Pu[(g.label == p).values].mean(axis=0) for p in range(1, 21) if (g.label == p).any()]
                B = np.broadcast_to(np.mean(segs, axis=0), Z.shape)
            elif base.startswith("ref"):
                r, E = base[3:].split("_")
                m = ((g.label == int(r)) & (g.off < int(round(float(E) * FPS)))).values
                B = np.broadcast_to(Pu[m].mean(axis=0), Z.shape) if m.any() else np.full_like(Z, np.nan)
            else:
                raise ValueError(base)
            Zr, Br = Z[loc], np.asarray(B[loc])
            if parts:
                out_rows[u] = (Zr, Br)
            elif op == "sub" or base == "none":
                out_rows[u] = Zr - Br
            elif op == "sub_scaled":
                out_rows[u] = (Zr - Br) / (np.abs(Br).sum(axis=1, keepdims=True) + 1e-6) * 1e3
            elif op == "logratio":
                out_rows[u] = np.log(Zr + 0.05) - np.log(Br + 0.05)
        us = sorted(out_rows)
        if parts:
            return (np.concatenate([out_rows[u][0] for u in us]), np.concatenate([out_rows[u][1] for u in us]))
        return np.concatenate([out_rows[u] for u in us]).astype(np.float32)

    def features_labelled_stream(self, rows, Wi=3.0, Wb=60.0, seed=None, op="sub", parts=False):
        """Order control.  Each driver's stream = its labelled single-covered frames only; if `seed` is given the driver's
        segments are concatenated in a RANDOM order, otherwise in the recorded order.  Baseline = running mean of the last
        Wb seconds of that stream (op: sub | sub_scaled | logratio, as in `features`).  Returns X for the requested rows."""
        rows = np.asarray(rows)
        fr = self.fr
        wi = max(1, int(round(Wi * FPS)))
        wb = max(1, int(round(Wb * FPS)))
        rng = np.random.default_rng(seed) if seed is not None else None
        out_rows = {}
        for u in self.span:
            s, e = self.span[u]
            if not ((rows >= s) & (rows < e)).any():
                continue
            g = fr.iloc[s:e]
            g = g[g.label > 0]
            segs = [d.index.values for _, d in g.groupby("label")]          # global indices
            if rng is not None:
                segs = [segs[i] for i in rng.permutation(len(segs))]
            else:
                segs = sorted(segs, key=lambda ix: ix[0])
            idx = np.concatenate(segs)
            Pu = self.raw(idx)
            Zs, Bs = causal_mean(Pu, wi)[0], causal_mean(Pu, wb)[0]
            if op == "sub":
                D = Zs - Bs
            elif op == "sub_scaled":
                D = (Zs - Bs) / (np.abs(Bs).sum(axis=1, keepdims=True) + 1e-6) * 1e3
            elif op == "logratio":
                D = np.log(Zs + 0.05) - np.log(Bs + 0.05)
            else:
                raise ValueError(op)
            pos = {int(i): k for k, i in enumerate(idx)}
            ru = rows[(rows >= s) & (rows < e)]
            if len(ru):
                sel = [pos[int(r)] for r in ru]
                out_rows[u] = (Zs[sel], Bs[sel]) if parts else D[sel]
        us = sorted(out_rows)
        if parts:
            return (np.concatenate([out_rows[u][0] for u in us]).astype(np.float32), np.concatenate([out_rows[u][1] for u in us]).astype(np.float32))
        return np.concatenate([out_rows[u] for u in us]).astype(np.float32)

    def rows(self, settle=5.0, trim=0, drop_p1_enrol=True, label_col="label"):
        """common evaluation / training rows: settled (>= `settle` s into the segment), strided, single-labelled."""
        fr = self.fr
        s = int(round(settle * FPS))
        m = (fr[label_col] > 0) & (fr.frame_id % STRIDE == 0) & (fr.off >= max(s, trim)) & (fr.dist_edge >= trim) & fr.u.isin(cohort_users())
        if drop_p1_enrol:
            m &= ~((fr.label == 1) & (fr.off < E_MAX))
        return np.where(m.values)[0]
