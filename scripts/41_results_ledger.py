"""Single source of truth for every number quoted in the proposal.  `ledger()` returns a nested dict; running the file prints it.

All accuracies are in percent, means over the held-out drivers (unit of analysis), 95 % CIs are percentile bootstraps over drivers.
PRIMARY analysis = 22-driver clean cohort, columns 64-160 (result files without suffix); sensitivity files carry _all (30 drivers, repaired labels),
_orig (30 drivers, first cleaning) or _s8 (rows >= 8 s into a segment).  Label-free baselines are computed on randomly ordered labelled streams
(mode 'shuf', seeds averaged per driver) unless the key says 'rec' (recorded order) or 'full' (real stream).
Missing result files are reported as None so the document builder can flag them.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import RES, clean_drivers, load_frames, load_users
from stats_utils import boot_ci, holm, noninferior, wilcoxon


def _read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def ms(v):
    v = np.asarray(v, float) * 100
    lo, hi = boot_ci(v)
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), sd=float(v.std(ddof=1)), median=float(np.median(v)), n=int(len(v)))


def diff(a, b):
    """paired difference a-b in percentage points with bootstrap CI, Wilcoxon p, share of drivers improved"""
    a, b = a.align(b, join="inner")
    d = (np.asarray(a, float) - np.asarray(b, float)) * 100
    lo, hi = boot_ci(d)
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p=wilcoxon(d), better=float((d > 0).mean()), sd=float(d.std(ddof=1)), n=int(len(d)))


def drv(df, **flt):
    """per-driver mean accuracy (fraction) over the rows matching the filters (e.g. over shuffle seeds)"""
    m = np.ones(len(df), bool)
    for k, v in flt.items():
        m &= (df[k] == v).values
    return df[m].groupby("u").acc.mean()


def shuf(df, base, **flt):
    return drv(df, base=base, mode="shuf", **flt)


def ledger():
    L = {}
    users = load_users().set_index("u")
    clean = clean_drivers()
    L["cohort"] = dict(n=len(clean), drivers=clean, women=int((users.loc[clean, "sex"] == "F").sum()), bmi30=int((users.loc[clean, "bmi"] >= 30).sum()),
                       bmi_min=float(users.loc[clean, "bmi"].min()), bmi_max=float(users.loc[clean, "bmi"].max()),
                       weight_min=float(users.loc[clean, "weight_kg"].min()), weight_max=float(users.loc[clean, "weight_kg"].max()))

    # --- data audit and repair ----------------------------------------------------------------------------------------------------------
    e = _read("eda_users.csv")
    if e is not None:
        L["eda"] = dict(frames=int(e.frames.sum()), labelled=int(e.labelled.sum()), overlap=int(e.overlap.sum()), unlabelled=int(e.unlabelled.sum()),
                        fps_median=float(e.fps.median()), minutes_median=float(e.minutes.median()),
                        drivers_incomplete=e.loc[e.postures_with_data != 20, ["u", "postures_with_data"]].values.tolist())
        fr_ = load_frames()
        L["eda"]["phase"] = {k: int(v) for k, v in fr_.groupby("phase").size().items()}
        L["eda"]["marker_labelled"] = int((fr_.label == -2).sum())
        L["eda"]["median_frames_per_driver"] = float(e.frames.median())
    w = _read("eda_load_vs_weight.csv")
    if w is not None:
        L["load"] = dict(ratio_mean=float(w.load_ratio.mean()), ratio_min=float(w.load_ratio.min()), ratio_max=float(w.load_ratio.max()),
                         r_weight_force=float(stats.pearsonr(w.weight_kg, w.force_med)[0]))
        wc = w[w.u.isin(clean)]
        z = (wc.load_ratio - wc.load_ratio.mean()) / wc.load_ratio.std()
        L["load"]["clean_ratio_min"], L["load"]["clean_ratio_max"] = float(wc.load_ratio.min()), float(wc.load_ratio.max())
        L["load"]["clean_max_abs_z"] = float(z.abs().max())
    rp = _read("labels_repaired.csv")
    if rp is not None:
        st = rp.status.value_counts().to_dict()
        L["repair"] = dict(rows=int(len(rp)), status=st, n_clean_drivers=len(clean), recovered=int(st.get("recovered", 0)), repaired=int(st.get("repaired", 0)),
                           partial=int(st.get("partial_overlap", 0)), no_data=int(st.get("no_data", 0)), order_dev=int(st.get("order_deviation", 0)),
                           violating_users=sorted(int(u) for u in rp[rp.status.isin(["repaired", "recovered", "partial_overlap", "order_deviation"])].u.unique()))
    sw = _read("repair_swap_test.csv")
    if sw is not None and len(sw):
        L["swap"] = dict(delta=float(sw.delta.iloc[0]), p_protocol=float(sw.p_if_protocol_order.iloc[0]), n_cal=int(sw.n_cal.iloc[0]), cal_correct=float(sw.cal_correct_share.iloc[0] * 100))
    vm = _read("repair_validation_mf.csv")
    if vm is not None:
        L["locator"] = dict(pairs=int(len(vm)), inside=float(vm.inside.mean() * 100), median_err=float((vm.c - (vm.endA + vm.startB) / 2).abs().median()),
                            off_start=float(vm.off_start.median()))
    mcal = RES / "marker_calibration.json"
    if mcal.exists():
        mk = json.load(open(mcal))
        L["markers"] = mk
        mf = _read("marker_label_flags.csv")
        if mf is not None:
            L["markers"]["flagged_total"] = int(len(mf))
            L["markers"]["flagged_interior_clean"] = int((mf.clean_driver & mf.kinds.str.contains("interior")).sum())
        me = _read("marker_events.csv")
        if me is not None:
            L["markers"]["events_total"] = int(len(me))
            L["markers"]["events_per_driver_median"] = float(me.groupby("u").size().median())
    v8 = _read("repair_validation_u8.csv")
    if v8 is not None and len(v8):
        r8 = v8.iloc[0]
        L["u8"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) for k, v in r8.items()}
    ga = _read("geometry_audit.csv")
    if ga is not None:
        L["geometry"] = {r.band: dict(cols=r.cols, share_of_force=float(r.share_of_force * 100), mean_pressure=float(r.mean_pressure), active=float(r.active_share * 100)) for r in ga.itertuples()}
    oa = _read("order_audit.csv")
    if oa is not None:
        L["order_audit"] = {r["key"]: {k: v for k, v in r.items() if k != "key" and not (isinstance(v, float) and np.isnan(v))} for r in oa.to_dict("records")}

    # --- evaluation protocols (scikit-learn, 16x24 maps, stride 6, rows >= 0 s) ----------------------------------------------------------------
    p = _read("protocols_LDA_kNN_RF.csv")
    if p is None:
        p = _read("protocols_LDA_LogReg_kNN_RF.csv")
    if p is not None:
        L["protocols"] = {f"{r.model}|{r.protocol}": float(r.acc * 100) for r in p.itertuples()}
    t = _read("time_only_baseline.csv")
    if t is not None:
        L["time_only"] = ms(t.acc)
    vd = _read("variance_decomposition.csv")
    if vd is not None:
        L["variance"] = {r.region: dict(subject=float(r.subject), posture=float(r.posture), interaction=float(r.interaction), n=int(r.n_drivers)) for r in vd.itertuples()}

    # --- calibration ladder --------------------------------------------------------------------------------------------------------------------
    d = _read("gpu_ladder.csv")
    if d is not None:
        none = drv(d, base="none")
        W = {"none": none, "session": drv(d, base="session"), "oracle": drv(d, base="oracle")}
        for r in ("ref1_5", "ref1_10", "ref10_5", "ref10_10"):
            W[r] = drv(d, base=r)
        for s in (15, 30, 60, 120, 300):
            W[f"run{s}"] = shuf(d, f"run{s}")
        wide = pd.DataFrame(W)
        L["ladder"] = {b: ms(wide[b]) for b in wide.columns}
        L["ladder_rec"] = {f"run{s}": ms(drv(d, base=f"run{s}", mode="rec")) for s in (15, 30, 60, 120, 300)}
        cmp = [c for c in wide.columns if c != "none"]
        res = {c: diff(wide[c], wide["none"]) for c in cmp}
        for c, ph in zip(cmp, holm([res[c]["p"] for c in cmp])):
            res[c]["p_holm"] = float(ph)
        L["ladder_vs_none"] = res
        L["ladder_pairs"] = dict(run60_vs_ref10_10=diff(wide["run60"], wide["ref10_10"]), oracle_vs_run60=diff(wide["oracle"], wide["run60"]),
                                 session_vs_run60=diff(wide["session"], wide["run60"]), run60_vs_ref1_10=diff(wide["run60"], wide["ref1_10"]))
        L["ladder_rec_vs_none"] = {f"run{s}": diff(drv(d, base=f"run{s}", mode="rec"), none) for s in (15, 30, 60, 120, 300)}
        L["gap_closure"] = {b: float((wide[b].mean() - none.mean()) / (wide["oracle"].mean() - none.mean()) * 100) for b in ("run30", "run60", "run120", "session")}

    # --- order control ---------------------------------------------------------------------------------------------------------------------------
    o = _read("gpu_order.csv")
    if o is not None and d is not None:
        none = drv(d, base="none")
        seeds = sorted(o[o["mode"] == "shuf"].seed.unique())
        per_seed = pd.DataFrame({s: drv(o, mode="shuf", seed=s) for s in seeds})
        shm = per_seed.mean(axis=1)
        rec, full = drv(o, mode="rec"), drv(o, mode="full")
        L["order"] = dict(none=ms(none), full=ms(full), recorded=ms(rec), shuffled=ms(shm), n_shuffles=len(seeds),
                          shuffled_range=[float(per_seed.mean().min() * 100), float(per_seed.mean().max() * 100)],
                          shuf_vs_none=diff(shm, none), rec_vs_shuf=diff(rec, shm), rec_vs_none=diff(rec, none), full_vs_shuf=diff(full, shm))

    # --- ablations ------------------------------------------------------------------------------------------------------------------------------------
    for name, key in (("window", "Wi"), ("op", "op"), ("region", "cols"), ("trim", "trim"), ("res", "f")):
        dd = _read(f"gpu_{name}.csv")
        if dd is None:
            continue
        out = {}
        for lvl, g in dd.groupby(key, sort=False):
            for base in ("none", "oracle", "run60"):
                gb = g[g.base == base]
                if len(gb) == 0:
                    continue
                v = gb[gb["mode"] == "shuf"].groupby("u").acc.mean() if base == "run60" else gb.groupby("u").acc.mean()
                out[f"{lvl}|{base}"] = ms(v)
        L[f"abl_{name}"] = out
    r = _read("gpu_res.csv")
    if r is not None:
        wr = {f"{f}|none": drv(r, f=f, base="none") for f in (16, 8, 4, 2)}
        wr.update({f"{f}|run60": shuf(r, "run60", f=f) for f in (16, 8, 4, 2)})
        L["res_tests"] = dict(f16run_vs_f4run=diff(wr["16|run60"], wr["4|run60"]), f16run_vs_f2none=diff(wr["16|run60"], wr["2|none"]),
                              f16run_vs_f4none=diff(wr["16|run60"], wr["4|none"]), f16run_vs_f16none=diff(wr["16|run60"], wr["16|none"]),
                              f16run_noninf_vs_f4run_2pp=noninferior((wr["16|run60"] - wr["4|run60"]) * 100, 2.0),
                              f8run_noninf_vs_f4run_2pp=noninferior((wr["8|run60"] - wr["4|run60"]) * 100, 2.0),
                              f16run_noninf_vs_f2run_2pp=noninferior((wr["16|run60"] - wr["2|run60"]) * 100, 2.0),
                              f4run_vs_f4none=diff(wr["4|run60"], wr["4|none"]))
    cb = _read("gpu_combo.csv")
    if cb is not None and r is not None:
        c_ = {f"{f}|run120": shuf(cb, "run120", f=f) for f in (16, 8, 4)}
        c_["16|run300"] = shuf(cb, "run300", f=16)
        c_["16|session"] = drv(cb, base="session", f=16)
        c_["16|oracle"] = drv(cb, base="oracle", f=16)
        L["combo"] = {k: ms(v) for k, v in c_.items()}
        n16, n4 = drv(r, f=16, base="none"), drv(r, f=4, base="none")
        L["combo_tests"] = dict(c16r120_vs_c16none=diff(c_["16|run120"], n16), c16r120_vs_c4r120=diff(c_["16|run120"], c_["4|run120"]),
                                c16r120_vs_c4none=diff(c_["16|run120"], n4), oracle16_vs_c16r120=diff(c_["16|oracle"], c_["16|run120"]),
                                noninf_16_vs_4_run120=noninferior((c_["16|run120"] - c_["4|run120"]) * 100, 2.0))
    # operator friedman (random-order baseline)
    op = _read("gpu_op.csv")
    if op is not None:
        wo = pd.DataFrame({o_: shuf(op, "run60", op=o_) for o_ in ("sub", "sub_scaled", "logratio")})
        L["op_friedman_p"] = float(stats.friedmanchisquare(*[wo[c] for c in wo.columns]).pvalue)

    # --- sensitivity to cohort / label set / settle time ------------------------------------------------------------------------------------------------
    sens = {}
    for key, fn in (("clean", "gpu_sens.csv"), ("all", "gpu_sens_all.csv"), ("orig", "gpu_sens_orig.csv"), ("settle8", "gpu_sens_s8.csv"), ("withmarkers", "gpu_sens_withmarkers.csv")):
        dd = _read(fn)
        if dd is None:
            continue
        none, run, ora, ses = drv(dd, base="none"), shuf(dd, "run60"), drv(dd, base="oracle"), drv(dd, base="session")
        sens[key] = dict(none=ms(none), run60=ms(run), oracle=ms(ora), session=ms(ses), gain=diff(run, none), gain_oracle=diff(ora, none), n=int(len(none)))
        if key in ("all", "orig"):
            c = [u for u in clean if u in none.index]
            sens[key + "_on_clean22"] = dict(none=ms(none[c]), run60=ms(run[c]), oracle=ms(ora[c]), gain=diff(run[c], none[c]), n=len(c))
            ex = [u for u in none.index if u not in clean]
            sens[key + "_on_excluded"] = dict(none=ms(none[ex]), run60=ms(run[ex]), oracle=ms(ora[ex]), gain=diff(run[ex], none[ex]), n=len(ex))
    if sens:
        L["sens"] = sens

    # --- learners (f=8) ---------------------------------------------------------------------------------------------------------------------------------
    ln = _read("gpu_learners.csv")
    if ln is not None:
        L["learners"], L["learners_gain"] = {}, {}
        for m in ("gLDA", "XGB"):
            dm = ln[ln.model == m]
            if dm.empty:
                continue
            none, ora = drv(dm, base="none"), drv(dm, base="oracle")
            run, rec = shuf(dm, "run60"), drv(dm, base="run60", mode="rec")
            L["learners"][m] = dict(none=ms(none), oracle=ms(ora), run60=ms(run), run60_rec=ms(rec))
            L["learners_gain"][m] = dict(run60=diff(run, none), oracle=diff(ora, none), rec_vs_shuf=diff(rec, run), rec=diff(rec, none))
        if "gLDA" in L["learners"] and "XGB" in L["learners"]:
            dg, dx = ln[ln.model == "gLDA"], ln[ln.model == "XGB"]
            L["learners_gain"]["XGB_vs_LDA_none"] = diff(drv(dx, base="none"), drv(dg, base="none"))
            L["learners_gain"]["XGB_vs_LDA_run60"] = diff(shuf(dx, "run60"), shuf(dg, "run60"))

    # --- learning curve, tuning, descriptors, CNN, identifiability --------------------------------------------------------------------------------------
    lc = _read("gpu_learning_curve.csv")
    if lc is not None:
        g = lc.groupby(["base", "n_train", "u"]).acc.mean().reset_index()
        L["learning"] = {f"{b}|{n}": ms(gg.acc) for (b, n), gg in g.groupby(["base", "n_train"])}
        nmax = int(g.n_train.max())
        L["learning_meta"] = dict(nmax=nmax, sizes=sorted(int(x) for x in g.n_train.unique()))
        if (g.n_train == 12).any():
            L["learning_step"] = {}
            for base_ in ("none", "run60"):
                a = g[(g.base == base_) & (g.n_train == nmax)].set_index("u").acc
                b = g[(g.base == base_) & (g.n_train == 12)].set_index("u").acc
                L["learning_step"][base_] = diff(a, b)
        a = g[(g.base == "run60") & (g.n_train == nmax)].set_index("u").acc
        n_ = g[(g.base == "none") & (g.n_train == nmax)].set_index("u").acc
        L["learning_gain_at_max"] = diff(a, n_)
    for mode in ("none", "run"):
        nt = _read(f"gpu_nested_{mode}.csv")
        if nt is not None:
            L[f"nested_{mode}"] = dict(acc=ms(nt.acc), Wi=nt.Wi.value_counts().to_dict(), Wb=nt.Wb.value_counts().to_dict(), pca=nt.pca.value_counts().to_dict(),
                                       shrinkage=nt.shrinkage.value_counts().to_dict())
    if "nested_none" in L and "nested_run" in L:
        a, b = _read("gpu_nested_none.csv").set_index("u").acc, _read("gpu_nested_run.csv").set_index("u").acc
        L["nested_gain"] = diff(b, a)
    ds = _read("gpu_descriptors.csv")
    if ds is not None:
        L["descriptors"] = {f"{k[0]}|{k[1]}": ms(g.acc) for k, g in ds.groupby(["model", "config"], sort=False)}
        L["descriptor_gain"] = {}
        for m_ in ds.model.unique():
            dd_ = ds[(ds.model == m_) & (ds.config == "descriptors|run60")].set_index("u").acc
            dn_ = ds[(ds.model == m_) & (ds.config == "descriptors|none")].set_index("u").acc
            L["descriptor_gain"][m_] = diff(dd_, dn_)
    cn = _read("gpu_cnn_grouped6.csv")
    if cn is not None:
        L["cnn6"] = {f"{k[0]}|{k[1]}": ms(g.acc) for k, g in cn.groupby(["model", "base"], sort=False)}
        for m in cn.model.unique():
            a = cn[(cn.model == m) & (cn.base == "run60")].set_index("u").acc
            b = cn[(cn.model == m) & (cn.base == "none")].set_index("u").acc
            L.setdefault("cnn6_gain", {})[m] = diff(a, b)
    # linear + label-free baseline against non-linear learners without it (96-cell LOSO rows)
    if ln is not None and {"gLDA", "XGB"} <= set(ln.model):
        dg_, dx_ = ln[ln.model == "gLDA"], ln[ln.model == "XGB"]
        L.setdefault("learners_gain", {})["LDA_run60_vs_XGB_none"] = diff(shuf(dg_, "run60"), drv(dx_, base="none"))
    # CNN re-tested under the same protocol as the other learners (leave-one-driver-out, two seeds per fold)
    cl = _read("gpu_cnn_loso.csv")
    lad_ = _read("gpu_ladder.csv")
    if cl is not None and lad_ is not None and {"CNN|none", "CNN|run60"} <= set(cl.config):
        w_ = {c: g_.set_index("u").acc for c, g_ in cl.groupby("config")}
        L["cnn_loso"] = {c: ms(v) for c, v in w_.items()}
        gains = {"CNN": diff(w_["CNN|run60"], w_["CNN|none"])}
        if "CNN2ch|run60" in w_:
            gains["CNN2ch"] = diff(w_["CNN2ch|run60"], w_["CNN|none"])
            gains["2ch_vs_diff"] = diff(w_["CNN2ch|run60"], w_["CNN|run60"])
        lda_none = drv(lad_, base="none")
        lda_s0 = drv(lad_, base="run60", mode="shuf", seed=0)
        gains["LDA_same_shuffle"] = diff(lda_s0, lda_none)
        gains["LDA_run60_vs_CNN_none"] = diff(lda_s0, w_["CNN|none"])
        gains["CNN_none_vs_LDA_none"] = diff(w_["CNN|none"], lda_none)
        sd_ = float(((w_["CNN|run60"] - w_["CNN|none"]) * 100).std(ddof=1))
        gains["sd_pair"] = sd_
        gains["mde80"] = float(2.8 * sd_ / np.sqrt(len(w_["CNN|none"])))
        L["cnn_loso_gain"] = gains
    idf = _read("gpu_identifiability.csv")
    if idf is not None:
        L["identifiability"] = {f"f{int(r.f)}|{r.split}": float(r.id_acc * 100) for r in idf.itertuples()}
        L["identifiability_meta"] = dict(n_drivers=int(idf.n_drivers.iloc[0]), cells={int(r.f): int(r.cells) for r in idf.itertuples()})

    # --- per-posture view and confusion-based diagnostics -----------------------------------------------------------------------------------------------
    pp_ = _read("per_posture_recall.csv")
    if pp_ is not None:
        L["per_posture"] = dict(n_gt20_none=int((pp_.recall_none > 0.20).sum()), n_gt20_run60=int((pp_.recall_run60 > 0.20).sum()),
                                n_gt20_oracle=int((pp_.recall_oracle > 0.20).sum()),
                                best_run60=[int(x) for x in pp_.sort_values("recall_run60", ascending=False).posture.head(5)],
                                worst_run60=[int(x) for x in pp_.sort_values("recall_run60").posture.head(3)])
    an = _read("anthropometrics_final.csv")
    if an is not None:
        L["anthro"] = dict(max_abs_rho=float(an.rho.abs().max()), min_p_holm=float(an.p_holm.min()), n_tests=int(len(an)))
    sv = _read("segment_level_variants.csv")
    if sv is not None:
        L["segment"] = {f"f{int(r.f)}|{r.centre}|{r.op}": dict(cos=float(r.acc_cos * 100), euc=float(r.acc_euc * 100)) for r in sv.itertuples()}
    sr = _read("segment_level_region.csv")
    if sr is not None:
        L["segment_region"] = {r.region: dict(none=float(r.none * 100), oracle=float(r.oracle * 100)) for r in sr.itertuples()}
    return L


if __name__ == "__main__":
    print(json.dumps(ledger(), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())[: int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9])
