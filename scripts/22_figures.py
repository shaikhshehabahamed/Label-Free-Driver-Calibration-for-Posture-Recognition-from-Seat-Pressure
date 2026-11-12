"""Figures for the proposal (v2: 22-driver primary cohort, columns 64-160, label-free baselines on randomly ordered streams).
Each figure is drawn only if its result files exist.   usage: python 22_figures.py
(F10 per-posture recall, F11 order audit, F12 label repair and F13 geometry are drawn by 42, 44, 46 and 47.)"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, RES
from stats_utils import boot_ci

spec = importlib.util.spec_from_file_location("ledger", Path(__file__).with_name("41_results_ledger.py"))
_m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m)
L = _m.ledger()

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 200})
C_NONE, C_RUN, C_ORA, C_REF, C_GREY, C_SHUF = "#7f7f7f", "#1f77b4", "#d62728", "#ff7f0e", "#bbbbbb", "#2ca02c"


def ci(v):
    v = np.asarray(v, float) * 100
    lo, hi = boot_ci(v)
    return v.mean(), lo, hi


def have(*names):
    return all((RES / n).exists() for n in names)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def err(d):
    return [[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]]


# ---- F1 protocol inflation + order floor ---------------------------------------------------------------------------------
if "protocols" in L:
    p = L["protocols"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    models = [m for m in ("LDA", "LogReg", "kNN", "RF") if f"{m}|LOSO" in p]
    w = 0.26
    for i, (prot, lab, col) in enumerate((("random5", "random frame split", "#9ecae1"), ("blocked5", "blocked, same driver", "#4292c6"), ("LOSO", "unseen driver (LOSO)", "#08306b"))):
        v = [p[f"{m}|{prot}"] for m in models]
        ax.bar(np.arange(len(models)) + (i - 1) * w, v, w, label=lab, color=col)
        for x, y in zip(np.arange(len(models)) + (i - 1) * w, v):
            ax.text(x, y + 1, f"{y:.0f}", ha="center", fontsize=7)
    ax.axhline(5, color="k", lw=0.8, ls=":", label="chance (5 %)")
    if "time_only" in L:
        ax.axhline(L["time_only"]["mean"], color=C_ORA, lw=1.1, ls="--", label=f"elapsed time alone, no pressure ({L['time_only']['mean']:.0f} %)")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_ylabel("accuracy, 20 postures (%)"); ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    savefig(fig, "F1_protocol_inflation.png")

# ---- F2 variance decomposition ---------------------------------------------------------------------------------------------
if "variance" in L:
    names = {"all": "analysed area (columns 64-160)", "back": "backrest-like strip", "seat": "seat-pan-like strip", "outer": "outer strip (excluded)"}
    v = pd.DataFrame([dict(region=names[k], **{kk: vv for kk, vv in val.items() if kk != "n"}) for k, val in L["variance"].items() if k in names])
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    left = np.zeros(len(v))
    for col, lab, c in (("subject", "driver", "#08306b"), ("posture", "posture", "#e6550d"), ("interaction", "driver x posture / residual", "#bdbdbd")):
        ax.barh(v.region, v[col], left=left, color=c, label=lab)
        for y, (l, x) in enumerate(zip(left, v[col])):
            if x > 6:
                ax.text(l + x / 2, y, f"{x:.0f}%", ha="center", va="center", color="w" if c != "#bdbdbd" else "k", fontsize=7)
        left += v[col].values
    ax.invert_yaxis(); ax.set_xlabel("share of variance in driver x posture mean maps (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    savefig(fig, "F2_variance_decomposition.png")

# ---- F3 calibration ladder + F4 paired scatter -------------------------------------------------------------------------------------------
if "ladder" in L:
    lad = L["ladder"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    runs = [(15, "run15"), (30, "run30"), (60, "run60"), (120, "run120"), (300, "run300")]
    xs = np.arange(len(runs))
    m = np.array([[lad[b]["mean"], lad[b]["lo"], lad[b]["hi"]] for _, b in runs])
    ax.errorbar(xs, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color=C_SHUF, marker="o", capsize=3, label="label-free running baseline over the last W s, segments in random order (primary)")
    if "ladder_rec" in L:
        mr = np.array([[L["ladder_rec"][b]["mean"], L["ladder_rec"][b]["lo"], L["ladder_rec"][b]["hi"]] for _, b in runs])
        ax.errorbar(xs + 0.12, mr[:, 0], yerr=[mr[:, 0] - mr[:, 1], mr[:, 2] - mr[:, 0]], color=C_RUN, marker="s", ms=4, capsize=2, ls="--", alpha=0.8, label="same, segments in recorded order (carries order information)")
    for lab, col, ls, txt in (("none", C_NONE, "-", "no calibration"), ("session", C_GREY, "--", "whole-session mean (label-free, non-causal)"), ("oracle", C_ORA, "-", "oracle (mean of all 20 labelled postures)")):
        ax.axhline(lad[lab]["mean"], color=col, ls=ls, lw=1.2, label=txt)
        ax.axhspan(lad[lab]["lo"], lad[lab]["hi"], color=col, alpha=0.10)
    for x_, (b, mk, lab) in zip((4.75, 4.95, 5.15), (("ref1_10", "s", "P1 reference, 10 s"), ("ref10_5", "D", "P10 reference, 5 s"), ("ref10_10", "^", "P10 reference, 10 s"))):
        ax.errorbar([x_], [lad[b]["mean"]], yerr=err(lad[b]), marker=mk, color=C_REF, ms=5, capsize=2, ls="", label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s} s" for s, _ in runs]); ax.set_xlabel("baseline window W"); ax.set_ylabel("unseen-driver accuracy (%)")
    ax.set_xlim(-0.4, 5.4)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    savefig(fig, "F3_calibration_ladder.png")

    d = pd.read_csv(RES / "gpu_ladder.csv")
    none = d[d.base == "none"].set_index("u").acc * 100
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    from common import load_users
    users = load_users().set_index("u")
    for ax, W in zip(axs, (60, 120)):
        run = d[(d.base == f"run{W}") & (d["mode"] == "shuf")].groupby("u").acc.mean() * 100
        x, y = none.loc[run.index], run
        ax.scatter(x, y, c=np.where(users.loc[run.index, "sex"] == "F", "#d62728", "#1f77b4"), s=22)
        lim = [0, max(x.max(), y.max()) + 4]
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("no calibration (%)")
        ax.set_title(f"{W}-s baseline: {(y > x).sum()} of {len(x)} drivers improve", fontsize=8)
    axs[0].set_ylabel("label-free baseline, random order (%)")
    savefig(fig, "F4_paired_drivers.png")

# ---- F5 resolution x calibration -----------------------------------------------------------------------------------------------
if "abl_res" in L:
    r = L["abl_res"]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    for base, col, lab in (("none", C_NONE, "no calibration"), ("run60", C_SHUF, "60 s baseline (random order)")):
        pts = np.array(sorted((6144 // (f * f), r[f"{f}|{base}"]["mean"], r[f"{f}|{base}"]["lo"], r[f"{f}|{base}"]["hi"]) for f in (16, 8, 4, 2)))
        ax.errorbar(pts[:, 0], pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], color=col, marker="o", capsize=3, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("sensing cells (mat columns 64-160, mean-pooled)"); ax.set_ylabel("unseen-driver accuracy (%)"); ax.legend(frameon=False, fontsize=7)
    savefig(fig, "F5_resolution_x_calibration.png")

# ---- F6 region ----------------------------------------------------------------------------------------------------------------------
if "abl_region" in L:
    r = L["abl_region"]
    names = {"(16, 64)": "outer strip", "(64, 112)": "backrest-like", "(112, 160)": "seat-pan-like", "(64, 160)": "both (analysed)", "(0, 160)": "whole mat"}
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    for i, (base, col, lab) in enumerate((("none", C_NONE, "no calibration"), ("run60", C_SHUF, "60 s baseline (random order)"))):
        v = [r[f"{k}|{base}"] for k in names]
        ax.bar(np.arange(len(names)) + (i - 0.5) * 0.38, [a["mean"] for a in v], 0.38, yerr=[[a["mean"] - a["lo"] for a in v], [a["hi"] - a["mean"] for a in v]], color=col, label=lab, capsize=2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names.values(), fontsize=7.5); ax.set_ylabel("unseen-driver accuracy (%)"); ax.legend(frameon=False, fontsize=7)
    savefig(fig, "F6_region.png")

# ---- F7 order control (linear + boosted trees) ------------------------------------------------------------------------------------------
if "order" in L:
    o = L["order"]
    groups = [("linear (LDA, 384 cells)", [("none", L["ladder"]["none"], C_NONE), ("baseline,\nrecorded order", o["recorded"], C_RUN), ("baseline,\nrandom order", o["shuffled"], C_SHUF)])]
    if "learners" in L and "XGB" in L["learners"]:
        x = L["learners"]["XGB"]
        groups.append(("boosted trees (XGBoost GPU, 96 cells)", [("none", x["none"], C_NONE), ("baseline,\nrecorded order", x["run60_rec"], C_RUN), ("baseline,\nrandom order", x["run60"], C_SHUF)]))
    fig, axs = plt.subplots(1, len(groups), figsize=(3.7 * len(groups), 3.0), sharey=True, squeeze=False)
    for ax, (title, items) in zip(axs[0], groups):
        for i, (lab, v, col) in enumerate(items):
            ax.bar(i, v["mean"], 0.6, yerr=err(v), color=col, capsize=3)
            ax.text(i, v["hi"] + 0.8, f"{v['mean']:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=7); ax.set_title(title, fontsize=8)
    axs[0][0].set_ylabel("unseen-driver accuracy (%)")
    savefig(fig, "F7_order_control.png")

# ---- F8 learning curve ----------------------------------------------------------------------------------------------------------------------
if "learning" in L:
    lc = L["learning"]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    for base, col, lab in (("none", C_NONE, "no calibration"), ("run60", C_SHUF, "60 s baseline (random order)")):
        ns = sorted(int(k.split("|")[1]) for k in lc if k.startswith(base + "|"))
        pts = np.array([(n, lc[f"{base}|{n}"]["mean"], lc[f"{base}|{n}"]["lo"], lc[f"{base}|{n}"]["hi"]) for n in ns])
        ax.errorbar(pts[:, 0], pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], color=col, marker="o", capsize=3, label=lab)
    ax.set_xlabel("number of training drivers"); ax.set_ylabel("unseen-driver accuracy (%)"); ax.legend(frameon=False, fontsize=7)
    savefig(fig, "F8_learning_curve.png")

# ---- F9 three ways to buy accuracy, one scale ---------------------------------------------------------------------------------------------------
items = []
if "ladder_vs_none" in L:
    lv = L["ladder_vs_none"]
    items.append(("label-free baseline, 60 s (software, zero driver effort)", lv["run60"], C_SHUF))
    items.append(("label-free baseline, 120 s", lv["run120"], C_SHUF))
    items.append(("oracle baseline (needs all labels; upper bound)", lv["oracle"], C_ORA))
if "abl_res" in L:
    r = L["abl_res"]
    a, b = r["2|none"], r["16|none"]
    items.append((f"64x more sensing cells, no calibration (24 -> 1,536)", dict(mean=a["mean"] - b["mean"], lo=a["lo"] - b["hi"], hi=a["hi"] - b["lo"]), C_GREY))
if "learning" in L and "none|12" in L["learning"] and f"none|{L['learning_meta']['nmax']}" in L["learning"]:
    lcd = pd.read_csv(RES / "gpu_learning_curve.csv").groupby(["base", "n_train", "u"]).acc.mean().reset_index()
    nmax = L["learning_meta"]["nmax"]
    a = lcd[(lcd.base == "none") & (lcd.n_train == nmax)].set_index("u").acc
    b = lcd[(lcd.base == "none") & (lcd.n_train == 12)].set_index("u").acc
    dd = (a - b).dropna() * 100
    lo, hi = boot_ci(dd.values)
    items.append((f"more training drivers, no calibration (12 -> {nmax})", dict(mean=float(dd.mean()), lo=float(lo), hi=float(hi)), C_GREY))
if items:
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for i, (lab, v, col) in enumerate(items):
        ax.barh(i, v["mean"], color=col, xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], capsize=3)
        ax.text(max(v["hi"], v["mean"]) + 0.4, i, f"{v['mean']:+.1f}", va="center", fontsize=7)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=7); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("paired change in unseen-driver accuracy (percentage points, 95 % CI over drivers)")
    savefig(fig, "F9_three_levers.png")

# ---- F14 sensitivity of the headline gain to cohort / label set / settle time -----------------------------------------------------------------------
if "sens" in L:
    s = L["sens"]
    rows = [("22 clean drivers (primary)", s.get("clean")), ("30 drivers, repaired labels", s.get("all")), ("   of which the 22 clean drivers", s.get("all_on_clean22")),
            ("   of which the 8 excluded drivers", s.get("all_on_excluded")), ("30 drivers, first cleaning", s.get("orig")), ("22 clean drivers, rows >= 8 s", s.get("settle8"))]
    rows = [(a, b) for a, b in rows if b]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, (lab, v) in enumerate(rows):
        g = v["gain"]
        ax.errorbar(g["mean"], i, xerr=[[g["mean"] - g["lo"]], [g["hi"] - g["mean"]]], marker="o", color=C_SHUF if i == 0 else C_RUN, capsize=3)
        ax.text(g["hi"] + 0.4, i, f"{g['mean']:+.1f} (n={v['n']})", va="center", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("gain of the 60-s random-order baseline over no calibration (pp, 95 % CI)")
    savefig(fig, "F14_sensitivity.png")

# ---- F15 learners ---------------------------------------------------------------------------------------------------------------------------------------
if "cnn6" in L or "learners" in L:
    items = []
    if "learners" in L:
        for m, lab in (("gLDA", "shrinkage LDA\n(LOSO, 96 cells)"), ("XGB", "boosted trees\n(LOSO, 96 cells)")):
            if m in L["learners"]:
                items.append((lab, L["learners"][m]["none"], L["learners"][m]["run60"]))
    if "cnn_loso" in L:
        cl_ = L["cnn_loso"]
        items.append(("3-layer CNN\n(LOSO, 384 cells)", cl_["CNN|none"], cl_["CNN|run60"]))
        if "CNN2ch|run60" in cl_:
            items.append(("CNN, both maps\n(LOSO, 384 cells)", cl_["CNN|none"], cl_["CNN2ch|run60"]))
    if "cnn6" in L:
        for m, lab in (("gLDA", "shrinkage LDA\n(6-fold, 384 cells)"), ("XGB", "boosted trees\n(6-fold, 384 cells)"), ("CNN", "3-layer CNN, first run\n(6-fold, 384 cells)")):
            if f"{m}|none" in L["cnn6"]:
                items.append((lab, L["cnn6"][f"{m}|none"], L["cnn6"][f"{m}|run60"]))
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    for i, (lab, a, b) in enumerate(items):
        ax.bar(i - 0.2, a["mean"], 0.38, yerr=err(a), color=C_NONE, capsize=2, label="no calibration" if i == 0 else None)
        ax.bar(i + 0.2, b["mean"], 0.38, yerr=err(b), color=C_SHUF, capsize=2, label="60 s baseline (random order)" if i == 0 else None)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=7); ax.set_ylabel("unseen-driver accuracy (%)"); ax.legend(frameon=False, fontsize=7)
    savefig(fig, "F15_learners.png")
print("figures written:", sorted(p.name for p in FIG.glob("F*.png")))
