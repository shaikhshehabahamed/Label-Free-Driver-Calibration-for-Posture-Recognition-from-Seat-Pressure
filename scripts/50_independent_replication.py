"""E1 - independent (clean-room) replication of the pilot's headline contrast.

Re-implements, from the written specification only, the 22-driver leave-one-driver-out comparison of
    L0       no calibration                         x = Z
    L1       label-free running baseline            x = Z - b_W   (labelled segments in RANDOM order; W = 60 s and 120 s)
    L1'      whole-session mean (non-causal)        b = mean of the occupied-seat frames of the recording
    Oracle   mean of the driver's 20 posture means  (needs every label)
through a different code path from `ladder.py` / `gpu_models.py`: pandas rolling means instead of cumulative sums, scikit-learn's
StandardScaler -> PCA(150) -> LDA(lsqr, shrinkage='auto') instead of the torch port, an own marker mask / cohort rule, and FRESH shuffle seeds.
Only the cached pooled maps (f = 4, columns 64-160) and the frame table are shared with the pilot.

Step 1 (unit test): the new feature builder is compared with the pilot's on identical shuffle seeds (must agree to float32 precision).
Step 2: LOSO with fresh seeds; per-driver accuracies appended to results/e1_independent_replication.csv (resumable).
Step 3: comparison with the pilot's per-driver accuracies (results/gpu_ladder.csv) -> results/e1_summary.csv.
usage: python 50_independent_replication.py
"""
import guard  # noqa: F401  (thread caps, priority) - first
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stats_utils as su
from common import CACHE, RES

FPS = 13.6
F, C0, C1 = 4, 64, 160
STRIDE, SETTLE_S, P1_ENROL_S, WI_S, OCC_N, MARK_PAD = 6, 5.0, 10.0, 3.0, 250.0, 10
FRESH_SEEDS = (1001, 1002, 1003)
OUT = RES / "e1_independent_replication.csv"

# ------------------------------------------------------------------------------------------------ data (own loaders)
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
lab_all[mark_all & (lab_all > 0)] = -2                      # labelled frames inside a marker press are excluded

lr = pd.read_csv(RES / "labels_repaired.csv")
COHORT = [int(u) for u, g in lr.groupby("u") if len(g) == 20 and g.use.all() and (g.status == "ok").all()]
pm = np.load(CACHE / f"pooled_f{F}.npy", mmap_mode="r")        # (N, 64/F, 160/F)


def roll_mean(P, w):
    """causal mean over the last w frames (expanding at the start) - pandas rolling, independent of the pilot's cumulative-sum code"""
    return pd.DataFrame(P).rolling(int(w), min_periods=1).mean().to_numpy(dtype=np.float32)


class Driver:
    def __init__(self, u):
        idx = np.flatnonzero(fr.u.to_numpy() == u)
        self.u, self.s, self.e = u, idx[0], idx[-1] + 1
        assert self.e - self.s == len(idx)
        self.P = np.asarray(pm[self.s:self.e][:, :, C0 // F:C1 // F], np.float32).reshape(len(idx), -1)
        self.fid = fr.frame_id.to_numpy()[self.s:self.e]
        self.lab = lab_all[self.s:self.e]
        self.off = self.fid - fr.seg_start.to_numpy()[self.s:self.e]
        self.force = fr.force_N.to_numpy()[self.s:self.e]
        self.mk = mark_all[self.s:self.e]
        ok = (self.lab > 0) & (self.fid % STRIDE == 0) & (self.off >= int(round(SETTLE_S * FPS)))
        ok &= ~((self.lab == 1) & (self.off < int(round(P1_ENROL_S * FPS))))
        self.rows = np.flatnonzero(ok)                      # local indices of the evaluation rows
        self.Zfull = roll_mean(self.P, round(WI_S * FPS))   # 3-s causal integration over the full recorded stream

    def none(self):
        return self.Zfull[self.rows]

    def session(self):
        occ = (self.force > OCC_N) & ~self.mk
        return self.Zfull[self.rows] - self.P[occ].mean(0)

    def oracle(self):
        b = np.mean([self.P[self.lab == p].mean(0) for p in range(1, 21) if (self.lab == p).any()], axis=0)
        return self.Zfull[self.rows] - b

    def shuffled(self, rng, W_s):
        """labelled segments (label order 1..20) concatenated in a random order drawn from `rng`; Z, b = causal means of the last 3 s / W_s seconds.
        Every driver must get its OWN random order: a permutation shared by all drivers would be a new fixed protocol order and bring order
        information back (the control exists to remove it)."""
        segs = [np.flatnonzero(self.lab == p) for p in np.unique(self.lab[self.lab > 0])]
        order = rng.permutation(len(segs))
        idx = np.concatenate([segs[i] for i in order])
        S = self.P[idx]
        D = roll_mean(S, round(WI_S * FPS)) - roll_mean(S, round(W_s * FPS))
        pos = np.empty(len(self.P), int)
        pos[idx] = np.arange(len(idx))
        return D[pos[self.rows]]


def conditions():
    c = [("none", -1, 0), ("session", -1, 0), ("oracle", -1, 0)]
    c += [(f"run{W}", s, W) for W in (60, 120) for s in FRESH_SEEDS]
    return c


def features(d, cond, seed, W):
    if cond == "none":
        return d.none()
    if cond == "session":
        return d.session()
    if cond == "oracle":
        return d.oracle()
    return d.shuffled(np.random.default_rng([seed, d.u]), W)         # independent stream per (seed, driver)


# ------------------------------------------------------------------------------------------------ step 1: unit test against the pilot
def unit_test(tol=2e-3):
    from ladder import Stream
    st = Stream(f=F)
    rows = st.rows(SETTLE_S)
    mine_rows = np.concatenate([d.s + d.rows for d in DRV])
    assert np.array_equal(rows, np.sort(mine_rows)), "evaluation row sets differ"
    worst = {}
    for name, base, kw in (("none", "none", {}), ("session", "session", {}), ("oracle", "oracle", {}), ("run60 shuf seed0", "run60", dict(seed=0))):
        if kw:
            Xp = st.features_labelled_stream(rows, WI_S, 60.0, kw["seed"])
            rng = np.random.default_rng(kw["seed"])          # the pilot draws all drivers' permutations from ONE sequential generator (ascending driver id)
            Xm = np.concatenate([d.shuffled(rng, 60) for d in DRV])
        else:
            Xp = st.features(rows, WI_S, base)
            Xm = np.concatenate([getattr(d, name)() for d in DRV])
        worst[name] = float(np.abs(Xp - Xm).max())
        print(f"  unit test {name:18s} max |pilot - mine| = {worst[name]:.2e} over {Xp.shape}", flush=True)
    assert max(worst.values()) < tol, f"feature builders disagree: {worst}"
    return worst


def make_model():
    return make_pipeline(StandardScaler(), PCA(n_components=150, svd_solver="full"),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


if __name__ == "__main__":
    t0 = time.time()
    DRV = [Driver(u) for u in COHORT]
    print(f"cohort {len(COHORT)} drivers; rows per driver {[len(d.rows) for d in DRV][:6]}...; total rows {sum(len(d.rows) for d in DRV)}", flush=True)
    unit_test()
    done = set()
    if OUT.exists():
        o = pd.read_csv(OUT)
        done = set(zip(o.cond, o.seed, o.u))
    y = np.concatenate([d.lab[d.rows] for d in DRV])
    g = np.concatenate([np.full(len(d.rows), d.u) for d in DRV])
    for cond, seed, W in conditions():
        if all((cond, seed, d.u) in done for d in DRV):
            print("skip (done):", cond, seed, flush=True)
            continue
        X = np.concatenate([features(d, cond, seed, W) for d in DRV])
        for d in DRV:
            if (cond, seed, d.u) in done:
                continue
            te = g == d.u
            m = make_model().fit(X[~te], y[~te])
            acc = float((m.predict(X[te]) == y[te]).mean())
            pd.DataFrame([dict(cond=cond, seed=seed, u=d.u, acc=acc, n=int(te.sum()))]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  {cond:8s} seed {seed:5d}: done [{time.time() - t0:.0f}s]", flush=True)

    # ------------------------------------------------------------------------------------------ step 3: comparison with the pilot
    mine = pd.read_csv(OUT)
    mine_pd = mine.groupby(["cond", "u"]).acc.mean().unstack(0)                 # per-driver accuracy (mean over seeds), conditions as columns
    pil = pd.read_csv(RES / "gpu_ladder.csv")
    pil = pil[(pil.model == "gLDA") & (pil.f == 4)]

    def pilot(base, mode):
        p = pil[(pil.base == base) & (pil["mode"] == mode)]
        return p.groupby("u").acc.mean()

    ref = {"none": pilot("none", "full"), "session": pilot("session", "full"), "oracle": pilot("oracle", "full"),
           "run60": pilot("run60", "shuf"), "run120": pilot("run120", "shuf")}
    rows_out = []
    for c in ("none", "session", "oracle", "run60", "run120"):
        a, b = mine_pd[c].sort_index(), ref[c].sort_index()
        assert list(a.index) == list(b.index), (c, list(a.index), list(b.index))
        d = (a - b).to_numpy() * 100
        lo, hi = su.boot_ci(d)
        rows_out.append(dict(cond=c, pilot_mean=100 * b.mean(), mine_mean=100 * a.mean(), diff_pp=d.mean(), diff_ci_lo=lo, diff_ci_hi=hi,
                             r_per_driver=float(np.corrcoef(a, b)[0, 1])))
    # headline contrasts: gain over none, both implementations
    for c in ("run60", "run120", "session", "oracle"):
        gm = (mine_pd[c] - mine_pd["none"]).sort_index().to_numpy() * 100
        gp = (ref[c] - ref["none"]).sort_index().to_numpy() * 100
        lo, hi = su.boot_ci(gm)
        dlo, dhi = su.boot_ci(gm - gp)
        rows_out.append(dict(cond=f"GAIN {c} vs none", pilot_mean=gp.mean(), mine_mean=gm.mean(), diff_pp=(gm - gp).mean(), diff_ci_lo=dlo, diff_ci_hi=dhi,
                             r_per_driver=float(np.corrcoef(gm, gp)[0, 1]), mine_gain_ci_lo=lo, mine_gain_ci_hi=hi, mine_gain_p=su.wilcoxon(gm)))
    summ = pd.DataFrame(rows_out)
    summ.to_csv(RES / "e1_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(f"E1 finished in {time.time() - t0:.0f}s")
