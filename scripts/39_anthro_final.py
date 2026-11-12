"""Does per-driver accuracy (or the benefit of calibration) depend on anthropometrics?  Spearman / Mann-Whitney, Holm over the family.
Analysis cohort; label-free baseline = 60-s running baseline on randomly ordered streams (mean of 3 shuffles per driver)."""
import numpy as np
import pandas as pd
from scipy import stats

from common import RES, load_users, tag
from stats_utils import holm

users = load_users().set_index("u")
lad = pd.read_csv(RES / f"gpu_ladder{tag()}.csv")
none = lad[lad.base == "none"].set_index("u").acc
run = lad[(lad.base == "run60") & (lad["mode"] == "shuf")].groupby("u").acc.mean()
tabs = {"LDA no calibration": none, "LDA 60 s baseline": run, "gain (baseline - none)": run - none}
x = RES / f"gpu_learners{tag()}.csv"
if x.exists():
    xg = pd.read_csv(x)
    xg = xg[xg.model == "XGB"]
    xn = xg[xg.base == "none"].set_index("u").acc
    xr = xg[(xg.base == "run60") & (xg["mode"] == "shuf")].groupby("u").acc.mean()
    tabs["XGB no calibration"] = xn
    tabs["XGB 60 s baseline"] = xr
    tabs["XGB gain"] = xr - xn
rows = []
n_f = int((users.loc[none.index, "sex"] == "F").sum())
for name, v in tabs.items():
    for col in ("weight_kg", "height_cm", "bmi"):
        r, p = stats.spearmanr(users.loc[v.index, col], v)
        rows.append(dict(outcome=name, predictor=col, rho=r, p=p))
    f, m = v[users.loc[v.index, "sex"] == "F"], v[users.loc[v.index, "sex"] == "M"]
    rows.append(dict(outcome=name, predictor=f"sex (F n={len(f)} vs M n={len(m)})", rho=np.nan, p=stats.mannwhitneyu(f, m).pvalue))
d = pd.DataFrame(rows)
d["p_holm"] = holm(d.p.values)
d.to_csv(RES / f"anthropometrics_final{tag()}.csv", index=False)
print(d.round(3).to_string(index=False))
print("\ndrivers:", len(none), "| women:", n_f, "| BMI>=30:", int((users.loc[none.index, "bmi"] >= 30).sum()))
print("min p_holm:", round(d.p_holm.min(), 3), "| max |rho|:", round(d.rho.abs().max(), 2))
