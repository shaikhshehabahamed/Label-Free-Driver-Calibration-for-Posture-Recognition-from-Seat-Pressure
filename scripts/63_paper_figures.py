"""Figures of the conference paper: P1 method schematic, P2 per-driver accuracy, P3 gain against the number of different postures.
IEEE one-column width (3.38 in), 8-pt Times New Roman labels, 300 dpi, white background, thin marks, hairline grid; colour is never the only code (marker shape and
line style differ).  Whiskers are 95 % bootstrap CIs over drivers.   usage: python 63_paper_figures.py
"""
import guard  # noqa: F401  (thread caps / low priority)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from common import FIG, RES

W_COL = 3.38
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#3a3935", "#6f6d66", "#e1e0d9", "#9a998f"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_FILL, GRAY_FILL = "#86b6ef", "#d4e5fa", "#ebeae4"
FAMILY = "Times New Roman" if "Times New Roman" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Serif"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.family": FAMILY, "font.size": 8,
    "mathtext.fontset": "stix", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.frameon": False, "legend.fontsize": 7.5, "axes.labelsize": 8,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png", "font:", FAMILY)


# ------------------------------------------------------------------------------------------------------------------------ P1
def fig_method():
    fig = plt.figure(figsize=(W_COL, 1.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")
    blocks = [("P12", 10), ("P3", 8), ("P18", 11), ("P7", 9), ("P15", 10), ("P9", 15)]
    x, y0, h, edges = 1.5, 19.0, 11.0, []
    for i, (lab, w) in enumerate(blocks):
        edges.append((x, x + w))
        ax.add_patch(Rectangle((x, y0), w, h, fc=(BLUE_FILL if i >= 2 else GRAY_FILL), ec=(BLUE if i >= 2 else AXIS), lw=0.7))
        cx = x + w / 2 if i < len(blocks) - 1 else x + (w - 5.0) / 2                  # the last label sits left of the 3-s window
        ax.text(cx, y0 + h / 2 - 0.4, lab, ha="center", va="center", fontsize=7.5, color=INK, zorder=5)
        x += w
    xt = edges[-1][1] - 1.5                                           # current time t, inside the last block
    xb = edges[2][0]                                                  # the baseline window starts at the third block
    # baseline bracket (above the stream)
    yb = 37.0
    ax.plot([xb, xb, xt, xt], [yb - 2.4, yb, yb, yb - 2.4], color=BLUE, lw=0.9, solid_capstyle="butt")
    ax.text((xb + xt) / 2, yb + 1.6, "Baseline window of $W$ seconds", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text((xb + xt) / 2, y0 + h + 3.0, "mean of its frames is $b_t$", ha="center", va="center", fontsize=7.5, color=BLUE)
    # 3-s window (below the stream)
    yz = 14.8
    zw = 5.0
    ax.add_patch(Rectangle((xt - zw, y0), zw, h, fc="#f8cdb9", ec=ORANGE, lw=0.8, zorder=3))
    ax.plot([xt - zw, xt - zw, xt, xt], [yz + 1.9, yz, yz, yz + 1.9], color=ORANGE, lw=0.9, solid_capstyle="butt")
    ax.text(xt - zw / 2 - 1.0, yz - 2.0, "last 3 s, mean $Z_t$", ha="center", va="top", fontsize=7.5, color=INK)
    # cursor, output formula and classifier
    ax.plot([xt, xt], [y0 - 1, yb], color=INK2, lw=0.7, ls=(0, (2.5, 1.8)), zorder=4)
    ax.text(xt, 45.2, "now, time $t$", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.add_patch(FancyArrowPatch((edges[-1][1] + 0.8, y0 + h / 2), (edges[-1][1] + 6.8, y0 + h / 2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK))
    ax.text(85.0, y0 + h / 2 + 2.6, "$x_t = Z_t - b_t$", ha="center", va="center", fontsize=8, color=INK)
    ax.text(85.0, y0 + h / 2 - 3.2, "to the classifier", ha="center", va="center", fontsize=7.2, color=INK2)
    # time axis
    ax.add_patch(FancyArrowPatch((1.5, 3.2), (edges[-1][1], 3.2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=MUTED))
    ax.text((1.5 + edges[-1][1]) / 2, 6.0, "time, with the postures in random order", ha="center", va="center", fontsize=7.5, color=INK2)
    save(fig, "P1_method_schematic")


# ------------------------------------------------------------------------------------------------------------------------ P2
def fig_drivers():
    d = pd.read_csv(RES / "e3_composition.csv")
    pdv = d.groupby(["kind", "W", "u"], dropna=False).acc.mean().reset_index()
    a0 = pdv[pdv.kind == "none"].set_index("u").acc * 100
    a1 = pdv[(pdv.kind == "run") & (pdv.W == 120)].set_index("u").acc * 100
    t = pd.DataFrame({"a0": a0, "a1": a1}).dropna()
    t["gain"] = t.a1 - t.a0
    t = t.sort_values("gain").reset_index()
    n_up, n_dn = int((t.gain > 0).sum()), int((t.gain <= 0).sum())
    print(f"P2: {len(t)} drivers; improved {n_up}, not improved {n_dn}; mean gain {t.gain.mean():.2f}; accuracy range {t.a0.min():.1f}-{t.a1.max():.1f}")
    fig, ax = plt.subplots(figsize=(W_COL, 2.7))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.99, bottom=0.185)
    ax.axvline(5.0, color=MUTED, lw=0.7, ls=(0, (1.5, 1.8)), zorder=1)
    for i, r in t.iterrows():
        col = BLUE if r.gain > 0 else ORANGE
        ax.plot([r.a0, r.a1], [i, i], color=col, lw=1.0, zorder=2, solid_capstyle="butt")
        ax.scatter(r.a0, i, s=11, facecolors="white", edgecolors=INK2, linewidths=0.7, zorder=3)
        ax.scatter(r.a1, i, s=11, color=col, edgecolors="white", linewidths=0.4, zorder=4, marker=("o" if r.gain > 0 else "s"))
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.set_xlim(0, 50)
    ax.set_xticks(range(0, 51, 10))
    ax.set_yticks([])
    ax.set_ylabel("Held-out drivers, sorted by gain")
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="y", visible=False)
    h = [Line2D([0], [0], marker="o", ls="none", mfc="white", mec=INK2, mew=0.7, ms=4.2, label="No calibration"),
         Line2D([0], [0], marker="o", ls="-", color=BLUE, ms=4.2, lw=1.0, mec="white", label=f"Rose ({n_up} drivers)"),
         Line2D([0], [0], marker="s", ls="-", color=ORANGE, ms=4.2, lw=1.0, mec="white", label=f"Fell ({n_dn} drivers)")]
    ax.legend(handles=h, loc="lower right", title="120-s baseline", title_fontsize=7.5, alignment="left", handletextpad=0.4, labelspacing=0.3, borderaxespad=0.1)
    save(fig, "P2_per_driver_accuracy")


# ------------------------------------------------------------------------------------------------------------------------ P3
def fig_distinct():
    s = pd.read_csv(RES / "e3_summary.csv")
    wc = pd.read_csv(RES / "e3_window_composition.csv").set_index("W")
    ex = {n: s[(s.block == "bag_excl") & (s.n == n)].sort_values("k") for n in (2, 5, 10)}
    ks = ex[5].k.values.astype(float)
    free = s[s.block == "bag_incl"].sort_values("k")
    run = s[s.block == "run"].sort_values("W")
    sess = float(s[s.block == "session"].gain.iloc[0])
    orac = float(s[s.block == "oracle"].gain.iloc[0])
    fig, ax = plt.subplots(figsize=(W_COL, 2.3))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.985, bottom=0.2)
    ax.set_xscale("log", base=2)
    lo, hi = np.minimum(ex[2].gain.values, ex[10].gain.values), np.maximum(ex[2].gain.values, ex[10].gain.values)
    ax.fill_between(ks, lo, hi, color=BLUE_LIGHT, alpha=0.45, lw=0, zorder=1)
    ax.plot(ks, ex[5].gain.values, "-o", color=BLUE, lw=1.2, ms=3.6, mec="white", mew=0.5, zorder=3)
    ax.plot(free.k.values, free.gain.values, "--D", color=AQUA, lw=1.1, ms=3.4, mec="white", mew=0.5, zorder=3)
    xr = np.array([wc.loc[int(w), "distinct"] for w in run.W])
    ax.errorbar(xr, run.gain.values, yerr=[run.gain.values - run.lo.values, run.hi.values - run.gain.values], fmt="^", color=ORANGE, ms=4.4, mec="white", mew=0.5,
                elinewidth=0.9, capsize=1.6, capthick=0.7, zorder=4)
    for xx, g, w in zip(xr, run.gain.values, run.W):
        ax.annotate(f"{int(w)} s", (xx, g), xytext=(3.5, -7.8), textcoords="offset points", fontsize=6.8, color=INK2, ha="left")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for yv, name, xx, ha in ((orac, "Oracle (uses labels)", 1.02, "left"), (sess, "Whole session", 1.02, "left")):
        ax.axhline(yv, color=INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.text(xx, yv + 0.25, name, fontsize=7, color=INK2, ha=ha, va="bottom")
    ax.set_xlim(0.9, 22)
    ax.set_ylim(-1.6, 19.6)
    ax.set_yticks(range(0, 15, 2))
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 8, 12, 19]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Different postures in the baseline")
    ax.set_ylabel("Gain over no calibration (points)")
    h = [Line2D([0], [0], color=BLUE, lw=1.2, marker="o", ms=3.6, mec="white", label="Draw without the current posture"),
         Line2D([0], [0], color=AQUA, lw=1.1, ls="--", marker="D", ms=3.4, mec="white", label="Label-free draw"),
         Line2D([0], [0], color=ORANGE, lw=0, marker="^", ms=4.4, mec="white", label="Running window")]
    ax.legend(handles=h, loc="upper left", handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    save(fig, "P3_distinct_postures")
    print("P3 running-window x:", np.round(xr, 2), "gains:", np.round(run.gain.values, 2))


if __name__ == "__main__":
    fig_method()
    fig_drivers()
    fig_distinct()
