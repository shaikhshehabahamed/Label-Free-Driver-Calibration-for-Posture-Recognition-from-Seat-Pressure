"""Figures of the independent follow-up study (S1 replication + operators, S2 seconds vs distinct postures, S3 validity check + deep-network comparator).
Style: documented reference palette (blue / orange / aqua = the first three, all-pairs-validated slots; ordinal blue ramp for ordered levels), thin marks,
solid hairline grid, one axis per panel, legend for every multi-series panel, text in neutral ink (colour lives in the marks).  Each figure is skipped
when its inputs are missing.   usage: python 58_figures.py
Note on uncertainty: whiskers / bands are 95 % bootstrap CIs over drivers of the quantity plotted (e.g. gain over no calibration).  Comparisons BETWEEN plotted
conditions are paired by driver with common random numbers and are far tighter than the whiskers suggest - they are tabulated in STUDY_RECORD.md.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import stats_utils as su
from common import FIG, RES

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"light": "#86b6ef", "mid": "#3987e5", "dark": "#1c5cab", "deep": "#0d366b"}      # ordinal blue ramp (steps 250 / 400 / 550 / 700)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "legend.fontsize": 8, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.solid_capstyle": "round",
})


def panel(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, color=INK, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)


def dot(ax, x, m, lo, hi, color, ms=6.5, z=3, **kw):
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=ms, mec=SURF, mew=1.2, elinewidth=1.1, capsize=0, zorder=z, **kw)


def line(ax, xs, ys, color, ms=5.5, z=3):
    ax.plot(xs, ys, "-o", color=color, lw=1.6, ms=ms, mec=SURF, mew=1.0, zorder=z)


def band(ax, xs, lo, hi, color):
    ax.fill_between(xs, lo, hi, color=color, alpha=0.14, lw=0, zorder=1)


def legend(ax, entries, loc="best", **kw):
    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor=SURF, ms=7, label=l) for l, c in entries]
    return ax.legend(handles=h, loc=loc, labelcolor=INK2, handletextpad=0.3, **kw)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def ci(v):
    lo, hi = su.boot_ci(np.asarray(v, float))
    return float(np.mean(v)), lo, hi


# ------------------------------------------------------------------------------------------------------------------ S1
def fig_s1():
    p1, p2 = RES / "e1_independent_replication.csv", RES / "e2_summary.csv"
    if not (p1.exists() and p2.exists()):
        return
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.28))
    e = pd.read_csv(p1).groupby(["cond", "u"]).acc.mean().unstack(0) * 100
    pil = pd.read_csv(RES / "gpu_ladder.csv")
    pil = pil[(pil.model == "gLDA") & (pil.f == 4)]

    def pv(base, mode):
        return pil[(pil.base == base) & (pil["mode"] == mode)].groupby("u").acc.mean() * 100

    ref = {"none": pv("none", "full"), "run60": pv("run60", "shuf"), "run120": pv("run120", "shuf"), "session": pv("session", "full"), "oracle": pv("oracle", "full")}
    names = [("none", "none"), ("run60", "60 s\nrunning"), ("run120", "120 s\nrunning"), ("session", "whole\nsession"), ("oracle", "oracle\n(labels)")]
    for i, (k, lab) in enumerate(names):
        for src, col, off in ((ref[k], BLUE, -0.13), (e[k], ORANGE, 0.13)):
            m, lo, hi = ci(src.dropna().values)
            dot(ax[0], i + off, m, lo, hi, col)
    ax[0].set_xticks(range(5), [n[1] for n in names])
    ax[0].set_xlim(-0.6, 4.6)
    panel(ax[0], "The headline reproduces", "unseen-driver accuracy (%), 20 classes")
    legend(ax[0], [("pilot (torch LDA)", BLUE), ("independent (scikit-learn LDA,\nnew random orders)", ORANGE)], loc="lower right")
    s = pd.read_csv(p2)
    ops = [("cmvn", "per-cell scaling (CMVN)", BLUE), ("wh(a=0.5)", "whitening (CORAL-type)", ORANGE), ("ea(a=0.1)", "Euclidean alignment as published", AQUA)]
    groups = [("run60", "60 s"), ("run120", "120 s"), ("session", "whole\nsession")]
    for a, f, ttl in ((ax[1], 16, "24 cells"), (ax[2], 8, "96 cells")):
        for i, (g, _) in enumerate(groups):
            for j, (op, _, col) in enumerate(ops):
                r = s[(s.f == f) & (s.group == g) & (s.op == op)].iloc[0]
                dot(a, i + (j - 1) * 0.2, r.diff_vs_sub, r.dvs_lo, r.dvs_hi, col)
        a.axhline(0, color=AXIS, lw=1.0, zorder=1)
        a.set_xticks(range(3), [g[1] for g in groups])
        a.set_xlim(-0.55, 2.55)
        a.set_ylim(-4.2, 8.6)
        panel(a, f"Beyond mean subtraction, {ttl}", "gain over mean subtraction (pp)" if f == 16 else None, "baseline window; 0 = mean subtraction" if f == 16 else "baseline window")
    legend(ax[1], [(o[1], o[2]) for o in ops], loc="upper left")
    save(fig, "S1_replication_and_operators")


# ------------------------------------------------------------------------------------------------------------------ S2
def fig_s2():
    p = RES / "e3_composition.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    pdv = d.groupby(["kind", "k", "n", "own", "W", "u"], dropna=False).acc.mean().reset_index()
    none = pdv[pdv.kind == "none"].set_index("u").acc

    def g(sel):
        return ((pdv[sel].set_index("u").acc - none).dropna() * 100).values

    def bag(k, n, own="excl"):
        return g((pdv.kind == "bag") & (pdv.own == own) & (pdv.k == k) & (pdv.n == n))

    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.6), gridspec_kw=dict(wspace=0.3))
    ks = [k for k in (1, 2, 3, 5, 8, 12, 19) if len(bag(k, 5))]
    ramp = {2: RAMP["light"], 5: RAMP["mid"], 10: RAMP["deep"]}
    for n in (2, 5, 10):
        xs, ms, los, his = [], [], [], []
        for k in ks:
            v = bag(k, n)
            if len(v):
                m, lo, hi = ci(v)
                xs.append(np.log2(k)); ms.append(m); los.append(lo); his.append(hi)
        line(ax[0], xs, ms, ramp[n])
        if n == 5:
            band(ax[0], xs, los, his, ramp[n])
    ax[0].set_xticks(np.log2(ks), [str(k) for k in ks])
    panel(ax[0], "More distinct postures help...", "gain over no calibration (pp)", "distinct postures in the baseline, k")
    legend(ax[0], [("2 s drawn per posture", ramp[2]), ("5 s (band: 95 % CI)", ramp[5]), ("10 s", ramp[10])], loc="lower right")
    for kk, col in ((1, RAMP["light"]), (3, RAMP["mid"]), (5, RAMP["dark"]), (12, RAMP["deep"])):
        xs, ms = [], []
        for n in (2, 5, 10):
            v = bag(kk, n)
            if len(v):
                xs.append(np.log2(n)); ms.append(v.mean())
        line(ax[1], xs, ms, col)
        if xs:
            ax[1].text(xs[-1] + 0.12, ms[-1], f"k = {kk}", color=INK2, va="center", fontsize=8)
    ax[1].set_xticks(np.log2([2, 5, 10]), ["2", "5", "10"])
    ax[1].set_xlim(0.8, 3.9)
    panel(ax[1], "...more seconds per posture barely do", None, "seconds drawn from each posture, n")
    # (c) running windows against the diversity they span
    wc = pd.read_csv(RES / "e3_window_composition.csv").set_index("W")
    xs, ms = [], []
    for W in (15, 30, 60, 120, 300):
        v = g((pdv.kind == "run") & (pdv.W == W))
        if len(v):
            m, lo, hi = ci(v)
            dot(ax[2], wc.loc[W, "distinct"], m, lo, hi, ORANGE, ms=6)
            ax[2].text(wc.loc[W, "distinct"] + 0.3, m - 1.0, f"{W} s", color=INK2, fontsize=7.5)
            xs.append(wc.loc[W, "distinct"]); ms.append(m)
    ax[2].plot(xs, ms, color=ORANGE, lw=1.6, zorder=2)
    kb = [k for k in ks]
    line(ax[2], kb, [bag(k, 5).mean() for k in kb], BLUE)
    ki = [k for k in (1, 3, 5, 12, 19) if len(bag(k, 5, "incl"))]
    line(ax[2], ki, [bag(k, 5, "incl").mean() for k in ki], AQUA)
    for kind, lab, x_, ha, dy in (("session", "whole-session mean", 19.6, "right", -0.95), ("oracle", "oracle (labels)", 0.7, "left", 0.3)):
        v = g(pdv.kind == kind)
        if len(v):
            ax[2].axhline(v.mean(), color=AXIS, lw=1.0, zorder=1)
            ax[2].text(x_, v.mean() + dy, lab, color=INK2, fontsize=7.5, ha=ha)
    ax[2].set_xlim(0.5, 19.8)
    ax[2].set_ylim(-1.5, 20.5)
    panel(ax[2], "A running window is a diversity curve", None, "distinct postures spanned (running) or drawn (built), k")
    legend(ax[2], [("running window (causal, 15-300 s)", ORANGE), ("k postures x 5 s, label-free draw", AQUA), ("k postures x 5 s, current posture excluded\n(uses the label: upper bound)", BLUE)],
           loc="upper left", fontsize=7.5)
    save(fig, "S2_seconds_vs_postures")


# ------------------------------------------------------------------------------------------------------------------ S3
def fig_s3():
    p7, p4 = RES / "e7_levels.csv", RES / "e4_cnn_adabn.csv"
    if not (p7.exists() or p4.exists()):
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw=dict(wspace=0.3))
    if p7.exists():                                                                          # (a) time-adjusted validity
        s = pd.read_csv(p7).set_index("config")
        order = [("time", "elapsed\ntime only", INK2), ("p_none", "pressure", BLUE), ("p_none+t", "pressure\n+ time", BLUE), ("p_run60", "pressure,\nbaseline", ORANGE), ("p_run60+t", "pressure,\nbaseline + time", ORANGE)]
        for i, (k, lab, col) in enumerate(order):
            r = s.loc[k]
            dot(ax[0], i, r["mean"], r.ci_lo, r.ci_hi, col)
        ax[0].set_xticks(range(5), [o[1] for o in order], fontsize=7.5)
        panel(ax[0], "Is there signal beyond the protocol clock?", "unseen-driver accuracy (%), boosted trees")
        legend(ax[0], [("no calibration", BLUE), ("label-free 60-s baseline", ORANGE)], loc="upper left")
    if p4.exists():                                                                          # (b) CNN
        raw = pd.read_csv(p4)
        c = raw.groupby("u")[["none", "none_adabn", "run60", "run60_adabn"]].mean() * 100
        order = [("none", "none", BLUE), ("none_adabn", "AdaBN", AQUA), ("run60", "driver-relative\ninput", ORANGE), ("run60_adabn", "driver-relative\n+ AdaBN", ORANGE)]
        for i, (k, lab, col) in enumerate(order):
            m, lo, hi = ci(c[k].values)
            dot(ax[1], i, m, lo, hi, col)
        ax[1].set_xticks(range(4), [o[1] for o in order], fontsize=7.5)
        panel(ax[1], "A deep network: input vs BN statistics", f"unseen-driver accuracy (%), CNN, {raw.seed.nunique()} seeds")
    save(fig, "S3_validity_and_learners")


if __name__ == "__main__":
    fig_s1()
    fig_s2()
    fig_s3()
