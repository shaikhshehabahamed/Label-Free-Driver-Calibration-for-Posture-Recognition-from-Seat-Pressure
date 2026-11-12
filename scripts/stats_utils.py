"""Driver-level paired statistics.  The unit of analysis is the held-out driver (n = 30), never the frame.

wilcoxon       two-sided Wilcoxon signed-rank on paired per-driver differences (Wilcoxon 1945; Demsar 2006 recommendation)
holm           Holm step-down adjustment (Holm 1979) over a stated family of comparisons
boot_ci        percentile bootstrap over drivers (10 000 resamples) for the mean / median paired difference
tost           equivalence: the 90 % bootstrap CI of the mean difference lies inside (-delta, +delta)  (Lakens 2017)
friedman       omnibus test over K configurations with drivers as blocks (Demsar 2006)
"""
import numpy as np
import pandas as pd
from scipy import stats


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(d, stat=np.mean, n=10000, alpha=0.05, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = stat(d[idx], axis=1)
    return np.quantile(bs, alpha / 2), np.quantile(bs, 1 - alpha / 2)


def wilcoxon(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)


def paired_vs_reference(wide, ref, metric_scale=100.0, seed=0):
    """wide: DataFrame drivers x configs (metric per driver).  Compare every config with `ref`.  Returns DataFrame."""
    rows = []
    for c in wide.columns:
        if c == ref:
            continue
        d = (wide[c] - wide[ref]).dropna().values * metric_scale
        lo, hi = boot_ci(d, np.mean, seed=seed)
        rows.append(dict(config=c, ref=ref, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, median_diff=np.median(d),
                         frac_drivers_better=(d > 0).mean(), n=len(d), p=wilcoxon(d)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p.values)
    return out


def tost(d, delta, n=10000, seed=0):
    """equivalence in mean paired difference (same units as d): 90 % CI inside +-delta."""
    lo, hi = boot_ci(np.asarray(d, float), np.mean, n=n, alpha=0.10, seed=seed)
    return dict(ci90_lo=lo, ci90_hi=hi, equivalent=bool(lo > -delta and hi < delta))


def noninferior(d, delta, n=10000, seed=0):
    """non-inferiority of a candidate (d = candidate - reference, same units as delta): the one-sided 95 % bootstrap lower bound of the
    mean difference must exceed -delta."""
    lo, _ = boot_ci(np.asarray(d, float), np.mean, n=n, alpha=0.10, seed=seed)
    return dict(lower95_onesided=lo, noninferior=bool(lo > -delta))


def friedman(wide):
    w = wide.dropna()
    stat, p = stats.friedmanchisquare(*[w[c].values for c in w.columns])
    return float(stat), float(p), len(w)


def summarize(wide, scale=100.0):
    """mean, 95% bootstrap CI over drivers, median, for each config (in percent)."""
    rows = []
    for c in wide.columns:
        v = wide[c].dropna().values * scale
        lo, hi = boot_ci(v, np.mean)
        rows.append(dict(config=c, mean=v.mean(), ci_lo=lo, ci_hi=hi, median=np.median(v), sd=v.std(ddof=1), n=len(v)))
    return pd.DataFrame(rows)
