"""Repair the defects of Frame_posture_labels.csv from the dataset itself (no input from the dataset authors needed).

Two independent kinds of evidence are used.
 (1) The protocol: all drivers performed P1..P20 in this order, so end(Pk) < start(Pk+1) must hold.
 (2) The marker presses: the dataset README says the participant and the operator pressed four highlighted points on the mat before each posture; they appear as
     marker events (48_markers.py).  On the 22 consistent drivers the published start of a posture follows the end of its marker by OFF_S frames (MAD ~11) and the
     published end precedes the next marker by OFF_E frames (MAD ~10).  A marker-anchored boundary is therefore as good as a published one.

Rules (applied only to drivers whose tables are inconsistent; the other 22 keep their published ranges):
  R1  NESTING      a range that fully contains other ranges is wrong at ONE bound.  Contained ranges are kept; the wrong bound is re-estimated from the marker adjacent to the
                   contained chain (fallback: container cut back to the 10th-percentile duration of that posture).
  R2  OVERLAP      two ranges that overlap without nesting: resolved by the single marker event inside the union if there is exactly one; otherwise the shared frames are dropped.
  R3  REVERSAL     two ranges in reversed time order are tested with driver-centred posture templates (calibrated on the clean drivers' own adjacent pairs); kept if the data
                   reject the protocol order, swapped if they support it, quarantined otherwise.
  R5  MISSING      a posture without a label is declared absent if the markers show no event for it (one marker for two consecutive labelled postures).
  M3  FRAME BASE   a label table most of whose ranges contain a marker is not on the CSV's frame base (driver 8: all 20 markers lie inside the recording, the labels are 1,100-2,000
                   frames late and irregularly so); ranges are rebuilt from the marker order (event j precedes posture j) and validated by a template test with a permutation null.
  R6  MARKER TRIM  a repaired range is cut where it runs into the next marker press.
Change-point / matched-filter locators on the pressure patterns were tried first and rejected (see results/repair_validation_*.csv): they miss the annotated boundary by ~150 frames.

Outputs: results/labels_repaired.csv, results/repair_log.csv, results/repair_swap_test.csv, results/repair_validation_u8.csv, ../Frame_posture_labels_repaired.csv
"""
import json

import numpy as np
import pandas as pd

import marker_lib as ml
from common import CACHE, RES, ROOT, load_frames

FPS = 13.6
SETTLE = int(round(5 * FPS))
cal = json.load(open(RES / "marker_calibration.json"))
OFF_S, OFF_E = int(cal["OFF_S"]), int(cal["OFF_E"])
EVALL = pd.read_csv(RES / "marker_events.csv")
FLAGS = pd.read_csv(RES / "marker_label_flags.csv")
raw = pd.read_csv(RES / "labels_clean.csv")
raw["len"] = raw.end - raw.start + 1
fr = load_frames()
last_frame = fr.groupby("u").frame_id.max()
log = []


def note(u, p, rule, action, evidence=""):
    log.append(dict(u=u, p=p, rule=rule, action=action, evidence=evidence))


def events(u):
    return EVALL[EVALL.u == u].sort_values("start").reset_index(drop=True)


# ---- 1. protocol violations --------------------------------------------------------------------------------------------------------------------------
def violations(L):
    out = []
    L = L.reset_index(drop=True)
    for i in range(len(L) - 1):
        a, b = L.loc[i], L.loc[i + 1]
        if b.start <= a.end:
            out.append(("overlap_or_nest", i, i + 1))
        if b.end < a.start:
            out.append(("inversion", i, i + 1))
    return out


viol_users = sorted({u for u, g in raw.groupby("u") if violations(g.sort_values("p"))})
# frame-base mismatch: most ranges contain a marker
inter_share = {u: float((FLAGS[(FLAGS.u == u)].kinds.str.contains("interior")).sum() / 20) for u in range(1, 31)}
m3_users = sorted(u for u, s in inter_share.items() if s >= 0.5)
clean_users = [u for u in range(1, 31) if u not in viol_users and u not in m3_users and (raw[raw.u == u].end.max() <= last_frame[u])]
print("protocol violations:", viol_users, "| frame-base mismatch:", m3_users, "| clean drivers:", len(clean_users))
dur = raw[raw.u.isin(clean_users)].groupby("p")["len"]
p10 = dur.quantile(0.10).round().astype(int)
med = dur.median().round().astype(int)

rep = raw.copy()
rep["orig_start"], rep["orig_end"] = rep.start, rep.end
rep["use"] = True
rep["status"] = "ok"
rep["rule"] = ""


def row(u, p):
    return rep[(rep.u == u) & (rep.p == p)].index[0]


# ---- 2. R1 nesting ------------------------------------------------------------------------------------------------------------------------------------
for u in viol_users:
    ev = events(u)
    L = rep[rep.u == u].sort_values("p")
    idx = list(L.index)
    for i in idx:
        a = rep.loc[i]
        inside = [j for j in idx if j != i and rep.loc[j, "start"] >= a.start and rep.loc[j, "end"] <= a.end]
        if not inside:
            continue
        cp, inner_p = int(a.p), sorted(int(rep.loc[j, "p"]) for j in inside)
        j_first = min(inside, key=lambda j: rep.loc[j, "start"])
        j_last = max(inside, key=lambda j: rep.loc[j, "end"])
        if cp < min(inner_p):                                    # container earlier in the protocol -> END is wrong
            lim = int(rep.loc[j_first, "start"])
            cand = ev[(ev.end <= lim + 30) & (ev.start >= a.start + 60)]
            if len(cand):
                e = cand.iloc[-1]
                new_end = int(e.start - OFF_E)
                how = f"end {int(a.end)} -> {new_end}: marker at {int(e.start)}-{int(e.end)} precedes P{inner_p[0]}, label end = marker start - {OFF_E}"
            else:
                new_end = int(min(a.start + p10[cp] - 1, lim - 1))
                how = f"end {int(a.end)} -> {new_end}: no marker found; cut to the 10th-percentile duration {p10[cp]} fr"
            rep.loc[i, "end"] = new_end
        else:                                                    # container later -> START is wrong
            lim = int(rep.loc[j_last, "end"])
            cand = ev[(ev.start >= lim - 20) & (ev.end <= a.end - 40)]
            if len(cand):
                e = cand.iloc[0]
                new_start = int(e.end + OFF_S)
                how = f"start {int(a.start)} -> {new_start}: marker at {int(e.start)}-{int(e.end)} follows P{inner_p[-1]}, label start = marker end + {OFF_S}"
            else:
                new_start = int(max(a.end - p10[cp] + 1, lim + 1))
                how = f"start {int(a.start)} -> {new_start}: no marker found; end-anchored 10th-percentile duration {p10[cp]} fr"
            rep.loc[i, "start"] = new_start
        rep.loc[i, ["status", "rule"]] = ["repaired", "R1"]
        note(u, cp, "R1", how, f"fully contains P{','.join(map(str, inner_p))}, a consistent chain; duration {int(a.len)} fr vs median {med[cp]}")
        for j in inside:
            rep.loc[j, ["status", "rule"]] = ["recovered", "R1"]
            note(u, int(rep.loc[j, "p"]), "R1", "kept unchanged (it was lost to the overlap exclusion in the first cleaning)", "")

# ---- 3. R2 partial overlaps -----------------------------------------------------------------------------------------------------------------------------
for u in viol_users:
    ev = events(u)
    L = rep[(rep.u == u) & rep.use].sort_values("p").reset_index()
    for i in range(len(L) - 1):
        a, b = L.loc[i], L.loc[i + 1]
        if b.start <= a.end and b.end >= a.start and not (b.end < a.start):
            ia, ib = int(a["index"]), int(b["index"])
            cand = ev[(ev.start > a.start + 60) & (ev.end < b.end - 60)]
            ov = int(min(a.end, b.end) - max(a.start, b.start) + 1)
            if len(cand) == 1:
                e = cand.iloc[0]
                rep.loc[ia, "end"] = int(min(a.end, e.start - OFF_E))
                rep.loc[ib, "start"] = int(max(b.start, e.end + OFF_S)) if b.start < e.end else int(b.start)
                if b.start < e.end:
                    rep.loc[ib, "start"] = int(e.end + OFF_S)
                for k in (ia, ib):
                    rep.loc[k, ["status", "rule"]] = ["repaired", "R2"]
                note(u, int(a.p), "R2", f"overlap with P{int(b.p)} ({ov} fr) resolved: P{int(a.p)} ends at {int(rep.loc[ia, 'end'])}, P{int(b.p)} starts at {int(rep.loc[ib, 'start'])}",
                     f"exactly one marker inside the union ({int(e.start)}-{int(e.end)})")
            else:
                for k in (ia, ib):
                    if rep.loc[k, "status"] == "ok":
                        rep.loc[k, ["status", "rule"]] = ["partial_overlap", "R2"]
                note(u, int(a.p), "R2", f"overlap with P{int(b.p)} of {ov} fr: overlapped frames dropped from both", f"{len(cand)} marker events inside the union: cannot attribute the boundary (the marker before the second posture was not pressed)")

# ---- 4. R3 reversed pair (template test) ---------------------------------------------------------------------------------------------------------------------
pm = np.load(CACHE / "pooled_f8.npy", mmap_mode="r")
rows_of = {u: g.index.values for u, g in fr.groupby("u")}
fid_of = {u: g.frame_id.values for u, g in fr.groupby("u")}


def seg_feat(u, s, e, settle=SETTLE):
    m = (fid_of[u] >= s + settle) & (fid_of[u] <= e)
    if m.sum() < 20:
        return None
    X = np.asarray(pm[rows_of[u][m]][:, :, 8:], dtype=np.float32).reshape(int(m.sum()), -1)
    return X.mean(axis=0)


F, C = {}, {}
for u in clean_users:
    for r in raw[raw.u == u].itertuples():
        f = seg_feat(u, r.start, r.end)
        if f is not None:
            F[(u, r.p)] = f
for u in clean_users:
    ps = [p for p in range(1, 21) if (u, p) in F]
    mu = np.mean([F[(u, p)] for p in ps], axis=0)
    for p in ps:
        C[(u, p)] = F[(u, p)] - mu


def template(p, exclude=None):
    return np.mean([C[(u, p)] for u in clean_users if u != exclude and (u, p) in C], axis=0)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def delta(ca, cb, Ta, Tb):
    return (cos(ca, Ta) + cos(cb, Tb)) - (cos(ca, Tb) + cos(cb, Ta))


cal_pairs = []
for u in clean_users:
    for k in range(1, 20):
        if (u, k) in C and (u, k + 1) in C:
            cal_pairs.append(dict(u=u, k=k, delta=delta(C[(u, k)], C[(u, k + 1)], template(k, u), template(k + 1, u))))
cal_pairs = pd.DataFrame(cal_pairs)
print("\nswap-test calibration on clean adjacent pairs: share ordered correctly = %.1f%% (n=%d)" % (100 * (cal_pairs.delta > 0).mean(), len(cal_pairs)))
swap_rows = []
for u in viol_users:
    L = rep[rep.u == u].sort_values("p").reset_index()
    for i in range(len(L) - 1):
        a, b = L.loc[i], L.loc[i + 1]
        if not (b.end < a.start):
            continue
        ps, pl = int(a.p), int(b.p)
        i_first, i_second = int(a["index"]), int(b["index"])                    # file order: ps then pl; time order: pl then ps
        e, l = rep.loc[i_second], rep.loc[i_first]                             # earlier / later in time
        fa, fb = seg_feat(u, int(e.start), int(e.end)), seg_feat(u, int(l.start), int(l.end))
        decision = "quarantine"
        if fa is not None and fb is not None:
            fs = [seg_feat(u, int(r.start), int(r.end)) for r in rep[(rep.u == u) & ~rep.p.isin([ps, pl])].itertuples()]
            mu = np.mean([f for f in fs if f is not None] + [fa, fb], axis=0)
            ca, cb = fa - mu, fb - mu
            Ts, Tl = template(ps), template(pl)
            d = delta(ca, cb, Ts, Tl)
            print(f"driver {u}: cos(earlier,T{ps})={cos(ca, Ts):.3f} cos(earlier,T{pl})={cos(ca, Tl):.3f} | cos(later,T{ps})={cos(cb, Ts):.3f} cos(later,T{pl})={cos(cb, Tl):.3f}")
            p_prot = float((cal_pairs.delta <= d).mean())
            p_rev = float(((-cal_pairs.delta) >= d).mean())
            swap_rows.append(dict(u=u, earlier_label=pl, later_label=ps, delta=d, p_if_protocol_order=p_prot, p_if_reversed=p_rev, n_cal=len(cal_pairs), cal_correct_share=float((cal_pairs.delta > 0).mean())))
            decision = "keep_file_order" if p_prot <= 0.05 else ("swap" if (d > 0 and p_rev <= 0.05) else "quarantine")
        if decision == "keep_file_order":
            rep.loc[[i_first, i_second], ["status", "rule"]] = ["order_deviation", "R3"]
            note(u, pl, "R3", f"labels kept: this driver performed P{pl} before P{ps}", f"template test rejects the protocol order (delta={d:.3f}; p={p_prot:.3f} among {len(cal_pairs)} clean pairs); one marker precedes each of the two postures")
        elif decision == "swap":
            rep.loc[i_second, "p"], rep.loc[i_first, "p"] = ps, pl
            rep.loc[[i_first, i_second], ["status", "rule"]] = ["relabelled", "R3"]
            note(u, ps, "R3", f"labels of the reversed pair swapped", f"template score favours the protocol order (delta={d:.3f})")
        else:
            rep.loc[[i_first, i_second], ["use", "status", "rule"]] = [False, "quarantined", "R3"]
            note(u, ps, "R3", f"pair P{ps}/P{pl} quarantined", "template test not decisive")
pd.DataFrame(swap_rows).to_csv(RES / "repair_swap_test.csv", index=False)

# ---- 5. R5 missing posture ---------------------------------------------------------------------------------------------------------------------------------
for u, g in rep.groupby("u"):
    present = set(g.p)
    for p in sorted(set(range(1, 21)) - present):
        prev_ = g[g.p < p].sort_values("p")
        next_ = g[g.p > p].sort_values("p")
        ev = events(u)
        if len(prev_) and len(next_):
            a, b = prev_.iloc[-1], next_.iloc[0]
            inside = ev[(ev.start > a.start + 60) & (ev.start < b.start)]
            note(u, p, "R5", "posture has no label and no marker: never performed (19 markers for 19 postures)" if len(inside) <= 1 else "posture has no label; markers suggest it may be recoverable",
                 f"{len(inside)} marker event(s) between P{int(a.p)} and P{int(b.p)}; a performed posture is preceded by its own marker")
        else:
            note(u, p, "R5", "posture has no label", "")

# ---- 6. M3 frame-base mismatch (driver 8) ----------------------------------------------------------------------------------------------------------------
# In the consistent drivers the 21 marker events are: one at the start of P1 (absent when the recording starts inside P1), 19 boundaries, one after P20.  Two structural
# hypotheses are therefore possible for a driver with 20 events and a useless label table: H0 event j precedes posture j (the last posture runs to the end of the CSV);
# H1 the first posture started before the first event, event j precedes posture j+1 and the last event is the end-of-session press.  The template score decides.
val8 = []
for u in m3_users:
    ev = events(u)
    n_ev = len(ev)
    g_ = fr[fr.u == u]
    first = int(g_.frame_id.min())
    occ_on = g_[g_.force_N > 250]
    first_occ = int(occ_on.frame_id.min()) if len(occ_on) else first
    if n_ev < 20:
        note(u, 0, "M3", f"label table not on the CSV frame base but only {n_ev} marker events: left excluded", "")
        rep.loc[rep.u == u, ["use", "status", "rule"]] = [False, "no_data", "M3"]
        continue
    hyps = {}
    r0 = {p: (int(ev.end[p - 1] + OFF_S), int(ev.start[p] - OFF_E) if p < n_ev else int(last_frame[u])) for p in range(1, 21)}
    hyps["H0"] = r0
    r1 = {1: (first_occ, int(ev.start[0] - OFF_E))}
    r1.update({p: (int(ev.end[p - 2] + OFF_S), int(ev.start[p - 1] - OFF_E)) for p in range(2, 21)})
    hyps["H1"] = r1
    T = {p: template(p) for p in range(1, 21)}
    clean_stats = [np.mean([cos(C[(v, p)], template(p, v)) for p in range(1, 21) if (v, p) in C]) for v in clean_users]
    rng = np.random.default_rng(0)
    scores = {}
    for name, rg in hyps.items():
        feats = {p: seg_feat(u, *rg[p]) for p in range(1, 21)}
        ok_p = [p for p in range(1, 21) if feats[p] is not None]
        mu = np.mean([feats[p] for p in ok_p], axis=0)
        cc = {p: feats[p] - mu for p in ok_p}
        obs = float(np.mean([cos(cc[p], T[p]) for p in ok_p]))
        null = np.array([np.mean([cos(cc[p], T[q]) for p, q in zip(ok_p, rng.permutation(ok_p))]) for _ in range(5000)])
        scores[name] = dict(mean_cos=obs, null_mean=float(null.mean()), perm_p=float((1 + (null >= obs).sum()) / (1 + len(null))), n_seg=len(ok_p))
    best = max(scores, key=lambda k: scores[k]["mean_cos"])
    val8.append(dict(u=u, chosen=best, clean_drivers_mean=float(np.mean(clean_stats)), clean_drivers_p10=float(np.percentile(clean_stats, 10)),
                     **{f"{k}_{kk}": vv for k, v in scores.items() for kk, vv in v.items()}))
    print(f"driver {u}: template scores {scores}; clean drivers mean {np.mean(clean_stats):.3f} -> chosen {best}")
    rg = hyps[best]
    for p in range(1, 21):
        k = row(u, p)
        rep.loc[k, ["start", "end"]] = [rg[p][0], rg[p][1]]
        rep.loc[k, ["status", "rule"]] = ["marker_anchored", "M3"]
    L0 = raw[raw.u == u].sort_values("p")
    shift = L0.start.values - np.array([rg[p][0] for p in range(1, 21)])
    note(u, 0, "M3", f"all 20 ranges rebuilt from the {n_ev} marker events ({best}: " + ("first posture before the first press, event j precedes posture j+1, last event = end-of-session press)" if best == "H1" else "event j precedes posture j)"),
         f"published starts are {int(shift.min())} to {int(shift.max())} frames later than the markers imply (not constant) although all {n_ev} markers lie inside the recording (last frame {int(last_frame[u])})")
    note(u, 0, "M3", f"validation: mean template cosine {scores[best]['mean_cos']:.3f} for the chosen structure vs {scores['H0' if best == 'H1' else 'H1']['mean_cos']:.3f} for the other, {scores[best]['null_mean']:.3f} for random assignments (p = {scores[best]['perm_p']:.4f}); clean drivers {np.mean(clean_stats):.3f}", "")
pd.DataFrame(val8).to_csv(RES / "repair_validation_u8.csv", index=False)

# ---- 7. R6 marker trim for repaired ranges, truncation, flags ----------------------------------------------------------------------------------------------
for u in viol_users:
    ev = events(u)
    for k in rep[(rep.u == u) & (rep.status != "ok") & rep.use].index:
        r = rep.loc[k]
        nxt = ev[(ev.start > r.start + 30) & (ev.start < r.end + 5)]
        if len(nxt) and r.status in ("partial_overlap", "repaired", "recovered", "order_deviation"):
            new_end = int(nxt.iloc[0].start - OFF_E)
            if new_end < r.end and new_end > r.start + 10:
                note(u, int(r.p), "R6", f"end {int(r.end)} -> {new_end}: the range ran into the next marker press ({int(nxt.iloc[0].start)})", "")
                rep.loc[k, "end"] = new_end
for u, g in rep.groupby("u"):
    lf = int(last_frame[u])
    for k, r in g.iterrows():
        if r.use and r.start > lf:
            rep.loc[k, ["use", "status", "rule"]] = [False, "no_data", "R4"]
            note(u, int(r.p), "R4", "label beyond the last recorded frame; no data", f"last frame {lf}")
        elif r.use and r.end > lf:
            rep.loc[k, "end"] = lf
            note(u, int(r.p), "R4", f"range cut at the last recorded frame {lf}", "")

rep["n_label"] = rep.end - rep.start + 1
rep["short"] = rep.n_label < SETTLE + 20
fl = FLAGS.groupby(["u", "p"]).kinds.first()
rep["marker_flag"] = [fl.get((r.u, r.p), "") for r in rep.itertuples()]
rep = rep.sort_values(["u", "p"]).reset_index(drop=True)

rem = []
for u, g in rep[rep.use].groupby("u"):
    g = g.sort_values("start")
    ov = (g.start.values[1:] <= g.end.values[:-1])
    for i in np.where(ov)[0]:
        rem.append((u, int(g.p.values[i]), int(g.p.values[i + 1])))
print("\nremaining overlaps after repair (frames dropped by R2 when the frame table is built):", rem)
order_bad = [u for u, g in rep[rep.use].groupby("u") if list(g.sort_values("start").p) != sorted(g.p)]
print("drivers whose time order differs from the protocol order (genuine deviations kept):", order_bad)
print(rep.status.value_counts().to_string())
rep[["u", "p", "start", "end", "n_label", "orig_start", "orig_end", "use", "status", "rule", "short", "marker_flag"]].to_csv(RES / "labels_repaired.csv", index=False)
pd.DataFrame(log).to_csv(RES / "repair_log.csv", index=False)
out = rep[rep.use].sort_values(["u", "p"]).copy()
out["user"] = "USER_" + out.u.astype(str)
out["posture"] = "P" + out.p.astype(str)
out[["user", "posture", "start", "end"]].rename(columns={"start": "start_frame", "end": "end_frame"}).to_csv(ROOT / "Frame_posture_labels_repaired.csv", index=False)
print("\nrepair log:")
print(pd.DataFrame(log).to_string(index=False, max_colwidth=130))
