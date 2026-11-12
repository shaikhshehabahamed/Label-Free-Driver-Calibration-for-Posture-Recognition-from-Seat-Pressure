"""Summaries of E7 (time-adjusted validity, boosted trees) and E4 (CNN: driver-relative input vs AdaBN).  Paired by driver, 95 % bootstrap CIs over drivers,
Wilcoxon signed-rank, Holm within each family."""
import numpy as np
import pandas as pd

import stats_utils as su
from common import RES

pd.set_option("display.width", 220)


def contrasts(wide, pairs, title):
    print(f"\n{title}")
    rows = []
    for a, b, label in pairs:
        dd = (wide[a] - wide[b]).dropna().values * 100
        lo, hi = su.boot_ci(dd)
        rows.append(dict(label=label, mean=dd.mean(), lo=lo, hi=hi, p=su.wilcoxon(dd), better=(dd > 0).mean()))
    t = pd.DataFrame(rows)
    t["p_holm"] = su.holm(t.p.values)
    print(t.round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------------------------------------------------------- E7
p = RES / "e7_time_adjusted.csv"
if p.exists():
    e = pd.read_csv(p)
    w = e.groupby(["cond", "u"]).acc.mean().unstack(0)
    s = su.summarize(w[["time", "p_none", "p_none+t", "p_run60", "p_run60+t"]])
    print("E7: unseen-driver accuracy of boosted trees (96 cells, LOSO, 22 drivers; mean over 3 model seeds / shuffles)")
    print(s.round(2).to_string(index=False))
    t7 = contrasts(w, [("p_none+t", "time", "pressure + time  vs  time only"),
                       ("p_run60+t", "time", "pressure (label-free baseline) + time  vs  time only"),
                       ("p_run60+t", "p_none+t", "label-free baseline given time (pressure + t)"),
                       ("p_run60", "p_none", "label-free baseline without time (pressure only)"),
                       ("p_none", "time", "pressure only (no calibration)  vs  time only"),
                       ("p_run60", "time", "pressure only (label-free baseline)  vs  time only")], "E7 paired contrasts (pp)")
    t7.to_csv(RES / "e7_summary.csv", index=False)
    s.to_csv(RES / "e7_levels.csv", index=False)

# ---------------------------------------------------------------------------------------------------------------- E4 (CNN / AdaBN)
p = RES / "e4_cnn_adabn.csv"
if p.exists():
    c = pd.read_csv(p)
    w = c.groupby("u")[["none", "none_adabn", "run60", "run60_adabn"]].mean()
    print("\nE4: 3-layer CNN, 384 cells, LOSO, 22 drivers, %d training seeds" % c.seed.nunique())
    print(su.summarize(w).round(2).to_string(index=False))
    t4 = contrasts(w, [("run60", "none", "driver-relative input (60-s baseline)  vs  none"),
                       ("none_adabn", "none", "AdaBN (target BN statistics)  vs  none"),
                       ("run60_adabn", "run60", "AdaBN on top of the driver-relative input"),
                       ("run60", "none_adabn", "driver-relative input  vs  AdaBN")], "E4 paired contrasts (pp)")
    t4.to_csv(RES / "e4_summary.csv", index=False)
