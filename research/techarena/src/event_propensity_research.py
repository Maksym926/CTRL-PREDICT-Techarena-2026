"""Event-level propensity audit and SAFE_v2 stacking research.

This is a leakage-safe, cache-first continuation of the TechArena handover.
The event targets and row features are built at each boundary B; all user
history statistics use only train events with timestamp < B.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from resume_handover import (CACHE, CUT, D0, DAY, OUT, add_c2, add_choice, add_choice,
                             choice_fit, concat_frames, feature_names, matrix, read_parquet,
                             safe_auc, weighted_dm)

TRAIN_CACHE = CACHE / "train.npz"
EVAL_CACHE = CACHE / "evaluation.npz"
BASE_EVENT_NAMES = [
    "user_rate_eb", "user_rate_recent3", "user_rate_recent5", "user_rate_recent10",
    "user_event_count", "last_event_clicked", "no_click_streak", "events_1h", "events_6h",
    "events_24h", "time_since_prev_event_h", "time_since_last_click_h", "time_since_last_event_h",
    "history_length", "history_span_days", "clicks_per_day", "history_unique_share",
    "outside_option_nu",
    "history_cos_mean", "history_cos_max", "history_svd_norm", "posterior_std_mean",
    "on_taste_share", "on_taste_share_065", "slate_logsumexp", "slate_top1", "slate_top2",
    "slate_entropy", "best_in_history_utility", "choice_any_probability", "safe_score_max",
    "safe_score_mean", "safe_score_std", "gap_hours",
]


def load_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def global_user_map(train: dict, ev: dict) -> dict[int, int]:
    keys = np.unique(np.concatenate([train["user_keys"], ev["user_keys"]]))
    return {int(k): i for i, k in enumerate(keys)}


def global_users(raw: dict, keymap: dict[int, int]) -> np.ndarray:
    return np.asarray([keymap[int(raw["user_keys"][u])] for u in raw["user_id"]], dtype=np.int64)


def first_events(raw: dict, users: np.ndarray, start: int, end: int) -> np.ndarray:
    mask = (raw["timestamp"] >= start) & (raw["timestamp"] < end)
    seen: set[int] = set(); out: list[int] = []
    for i in np.flatnonzero(mask):
        u = int(users[i])
        if u not in seen:
            seen.add(u); out.append(int(i))
    return np.asarray(out, dtype=np.int64)


def build_user_hist(raw: dict, users: np.ndarray, boundary: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    mask = raw["timestamp"] < boundary
    by: dict[int, list[tuple[int, int]]] = {}
    for i in np.flatnonzero(mask):
        u = int(users[i]); t = int(raw["timestamp"][i]); click = int(raw["labels"][i].sum() > 0)
        by.setdefault(u, []).append((t, click))
    return {u: (np.asarray([x[0] for x in v], dtype=np.int64), np.asarray([x[1] for x in v], dtype=np.int8)) for u, v in by.items()}


def beta_prior(hist: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    rates = []
    for ts, ys in hist.values():
        if len(ys): rates.append((ys.sum() + .5) / (len(ys) + 1.0))
    if not rates: return 1.0, 3.0
    m = float(np.mean(rates)); v = float(np.var(rates))
    k = max(2.0, m * (1 - m) / max(v, 1e-5) - 1.0)
    return max(.1, m * k), max(.1, (1 - m) * k)


def event_features(frame: dict[str, np.ndarray], raw: dict, users: np.ndarray, target_idx: np.ndarray, boundary: int) -> dict[str, np.ndarray]:
    n = len(frame["label"]) // 10
    r = {k: frame[k].reshape(n, 10) for k in ["svd_cos", "history_mean_cos", "history_max_cos", "history_count", "item_ctr", "event_score_max", "event_score_mean", "choice_utility", "choice_probability", "choice_no_click", "shrink_norm", "posterior_std"] if k in frame}
    out: dict[str, np.ndarray] = {}
    uids = frame["user_id"].reshape(n, 10)[:, 0].astype(np.int64)
    timestamps = frame["timestamp"].reshape(n, 10)[:, 0].astype(np.int64)
    hist = raw["history_items"][target_idx]; hts = raw["history_times"][target_idx]
    hlen = raw["history_length"][target_idx].astype(np.float32)
    by = build_user_hist(raw, users, boundary); alpha, beta = beta_prior(by)
    vals = {k: [] for k in ["user_rate_eb", "user_rate_recent3", "user_rate_recent5", "user_rate_recent10", "user_event_count", "last_event_clicked", "no_click_streak", "events_1h", "events_6h", "events_24h", "time_since_prev_event_h", "time_since_last_click_h", "time_since_last_event_h"]}
    for u, t in zip(uids, timestamps):
        ts, ys = by.get(int(u), (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)))
        # State only contains events before B. Recent windows end at B, while
        # elapsed features are measured to the target event timestamp.
        cnt = len(ys); clicks = int(ys.sum());
        vals["user_event_count"].append(cnt)
        vals["user_rate_eb"].append((clicks + alpha) / (cnt + alpha + beta))
        for k in (3, 5, 10): vals[f"user_rate_recent{k}"].append(float(ys[-k:].mean()) if cnt else 0.0)
        vals["last_event_clicked"].append(int(ys[-1]) if cnt else 0)
        streak = 0
        for y in ys[::-1]:
            if y: break
            streak += 1
        vals["no_click_streak"].append(streak)
        vals["events_1h"].append(int((ts >= boundary - 3600).sum()))
        vals["events_6h"].append(int((ts >= boundary - 6 * 3600).sum()))
        vals["events_24h"].append(int((ts >= boundary - 24 * 3600).sum()))
        last_t = int(ts[-1]) if cnt else boundary
        vals["time_since_prev_event_h"].append(max(0, int(t) - last_t) / 3600.0)
        vals["time_since_last_event_h"].append(max(0, int(t) - last_t) / 3600.0)
        click_times = ts[ys == 1]
        vals["time_since_last_click_h"].append(max(0, int(t) - int(click_times[-1])) / 3600.0 if len(click_times) else 1e3)
    for k, v in vals.items(): out[k] = np.asarray(v, dtype=np.float32)
    out["outside_option_nu"] = np.log(np.clip(1.0 - out["user_rate_eb"], 1e-5, 1.0) / np.clip(out["user_rate_eb"], 1e-5, 1.0)).astype(np.float32)
    out["history_length"] = hlen
    spans = []; cpd = []; uniq = []
    for items, times, ln in zip(hist, hts, hlen):
        m = items >= 0; t = times[m]
        span = (int(t[-1]) - int(t[0])) / 86400.0 if len(t) > 1 else 0.0
        spans.append(span); cpd.append(float(ln) / max(span, 1 / 24)); uniq.append(len(set(map(int, items[m]))) / max(float(ln), 1.0))
    out["history_span_days"] = np.asarray(spans, dtype=np.float32); out["clicks_per_day"] = np.asarray(cpd, dtype=np.float32); out["history_unique_share"] = np.asarray(uniq, dtype=np.float32)
    def agg(name, fn=np.mean):
        if name in r: return fn(r[name], axis=1).astype(np.float32)
        return np.zeros(n, dtype=np.float32)
    out["history_cos_mean"] = agg("history_mean_cos"); out["history_cos_max"] = agg("history_max_cos")
    out["history_svd_norm"] = agg("shrink_norm"); out["posterior_std_mean"] = agg("posterior_std")
    out["on_taste_share"] = (r.get("svd_cos", np.zeros((n, 10))) > .5).mean(1).astype(np.float32)
    out["on_taste_share_065"] = (r.get("svd_cos", np.zeros((n, 10))) > .65).mean(1).astype(np.float32)
    u = r.get("choice_utility", np.zeros((n, 10))).astype(np.float64); cp = r.get("choice_probability", np.zeros((n, 10))).astype(np.float64); nc = r.get("choice_no_click", np.zeros((n, 10))).astype(np.float64)[:, 0]
    umax = np.sort(u, axis=1)[:, ::-1]
    z = u - u.max(1, keepdims=True); lse = u.max(1) + np.log(np.exp(z).sum(1)); out["slate_logsumexp"] = lse.astype(np.float32); out["slate_top1"] = umax[:, 0].astype(np.float32); out["slate_top2"] = umax[:, 1].astype(np.float32)
    p0 = np.clip(nc, 1e-8, 1); pall = np.c_[cp, p0]; out["slate_entropy"] = (-pall * np.log(np.clip(pall, 1e-8, 1))).sum(1).astype(np.float32)
    inhist = r.get("history_count", np.zeros((n, 10))) > 0; out["best_in_history_utility"] = np.where(inhist.any(1), np.where(inhist, u, -1e6).max(1), 0).astype(np.float32)
    out["choice_any_probability"] = (1 - nc).astype(np.float32)
    out["safe_score_max"] = r.get("event_score_max", np.zeros((n, 10)))[:, 0].astype(np.float32); out["safe_score_mean"] = r.get("event_score_mean", np.zeros((n, 10)))[:, 0].astype(np.float32); out["safe_score_std"] = r.get("event_score_max", np.zeros((n, 10))).std(1).astype(np.float32)
    out["gap_hours"] = frame["gap_hours"].reshape(n, 10)[:, 0].astype(np.float32)
    out["label"] = (frame["label"].reshape(n, 10).sum(1) > 0).astype(np.int8)
    return out


def event_key(h, g):
    hs = np.where(h <= 0, "0", np.where(h <= 9, "1-9", np.where(h <= 19, "10-19", "20")))
    gs = np.where(g < 6, "<6h", np.where(g < 24, "6-24h", np.where(g < 72, "1-3d", np.where(g < 168, "3-7d", ">7d"))))
    return np.char.add(np.char.add(hs, "|"), gs)


def ew_auc(y, p, h, g, eh, eg):
    vk, ek = event_key(h, g), event_key(eh, eg); u, c = np.unique(ek, return_counts=True); es = {a: b / len(ek) for a, b in zip(u, c)}; u, c = np.unique(vk, return_counts=True); vs = {a: b / len(vk) for a, b in zip(u, c)}
    w = np.asarray([es.get(k, 0) / max(vs.get(k, 1e-12), 1e-12) for k in vk])
    return float(roc_auc_score(y, p, sample_weight=w))


def e_matrix(f):
    cols = [k for k in f if k != "label"]
    return np.column_stack([np.nan_to_num(f[k], nan=0, posinf=0, neginf=0).astype(np.float32) for k in cols]), cols


def fit_event(train, val, eval_h, eval_g):
    x, cols = e_matrix(train); xv, _ = e_matrix(val); u = event_key(train["history_length"], train["gap_hours"]); eu = event_key(eval_h, eval_g); uu, cc = np.unique(eu, return_counts=True); es = {a: b / len(eu) for a, b in zip(uu, cc)}; uu, cc = np.unique(u, return_counts=True); vs = {a: b / len(u) for a, b in zip(uu, cc)}; w = np.asarray([es.get(k, 0) / max(vs.get(k, 1e-12), 1e-12) for k in u])
    m = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.04, num_leaves=31, min_child_samples=80, reg_lambda=10, random_state=20261027, n_jobs=4, verbosity=-1)
    m.fit(x, train["label"], sample_weight=w); return m, cols, m.predict_proba(x)[:, 1], m.predict_proba(xv)[:, 1]


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def row_stack(train_frames, val_frame, train_e, val_e, emod, ecols, eval_frame):
    train = concat_frames(train_frames)
    cols = feature_names(train)
    base_tr = matrix(train, cols); base_va = matrix(val_frame, cols)
    etr = emod.predict_proba(np.column_stack([train_e[c] for c in ecols]))[:, 1]
    eva = emod.predict_proba(np.column_stack([val_e[c] for c in ecols]))[:, 1]
    top = [c for c in ["choice_any_probability", "slate_logsumexp", "safe_score_mean", "user_rate_eb", "slate_top1", "on_taste_share", "history_cos_mean", "history_length"] if c in ecols]
    extras_tr = [np.repeat(logit(etr), 10)]; extras_va = [np.repeat(logit(eva), 10)]
    for c in top:
        extras_tr.append(np.repeat(train_e[c], 10)); extras_va.append(np.repeat(val_e[c], 10))
    for c in ["svd_cos", "history_mean_cos", "item_ctr", "transition_mean"]:
        if c in train:
            extras_tr.append(train[c] * np.repeat(logit(etr), 10)); extras_va.append(val_frame[c] * np.repeat(logit(eva), 10))
    Xtr = np.column_stack([base_tr] + extras_tr); Xva = np.column_stack([base_va] + extras_va)
    smod = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.03, num_leaves=63, min_child_samples=300, colsample_bytree=.7, subsample=.8, subsample_freq=1, reg_lambda=10, random_state=20261029, n_jobs=4, verbosity=-1)
    smod.fit(Xtr, train["label"]); stack = smod.predict_proba(Xva)[:, 1]
    # Baseline row model and a conditional W model for the diagnostic product.
    bmod = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.03, num_leaves=63, min_child_samples=300, colsample_bytree=.7, subsample=.8, subsample_freq=1, reg_lambda=10, random_state=20261030, n_jobs=4, verbosity=-1)
    bmod.fit(base_tr, train["label"]); base = bmod.predict_proba(base_va)[:, 1]
    click_ev = train["label"].reshape(-1, 10).sum(1) > 0; mask = np.repeat(click_ev, 10)
    wmod = lgb.LGBMClassifier(objective="binary", n_estimators=350, learning_rate=.04, num_leaves=63, min_child_samples=300, reg_lambda=10, random_state=20261031, n_jobs=4, verbosity=-1)
    wmod.fit(base_tr[mask], train["label"][mask]); wl = wmod.predict_proba(base_va)[:, 1].reshape(-1, 10); wl -= wl.max(1, keepdims=True); pw = np.exp(wl); pw /= pw.sum(1, keepdims=True)
    product = (eva[:, None] * pw).reshape(-1)
    blends=[]
    for w in np.linspace(0, 1, 21):
        bp = 1 / (1 + np.exp(-(w * logit(base) + (1 - w) * logit(stack))))
        blends.append((weighted_dm(val_frame, bp, eval_frame), float(w), safe_auc(val_frame["label"], bp)))
    blends.sort(reverse=True)
    return {"baseline": base, "stack": stack, "product": product, "best_blend": {"dm": blends[0][0], "stack_weight": blends[0][1], "raw_auc": blends[0][2]}, "top_event_features": top}


def persistence(raw, users):
    d = ((raw["timestamp"] - D0) // DAY).astype(int); y = (raw["labels"].sum(1) > 0).astype(int); by = {}
    for u, day, yy in zip(users, d, y): by.setdefault(int(u), []).append((int(day), int(yy)))
    a=[]; b=[]; counts=[]; pairs=[]
    for u,v in by.items():
        x=np.asarray([q[1] for q in v if q[0] in (0,1,2)]); z=np.asarray([q[1] for q in v if q[0] in (3,4,5)])
        if len(x) and len(z): a.append(x.mean()); b.append(z.mean()); counts.append(len(v)); pairs.append((x.mean(),z.mean(),len(v)))
    corr=float(np.corrcoef(a,b)[0,1]) if len(a)>1 else None
    bins={}
    for lo,hi,nm in [(1,1,"1"),(2,4,"2-4"),(5,9,"5-9"),(10,10**9,"10+")]:
        q=[(x,z) for x,z,c in pairs if lo<=c<=hi]
        bins[nm]={"users":len(q),"corr":float(np.corrcoef(np.asarray(q).T)[0,1]) if len(q)>1 else None}
    return {"users_with_both_periods":len(pairs),"corr_days1_3_vs4_6":corr,"by_total_event_count":bins}


def trained_ladder(train, val):
    cols = [("L0", ["item_ctr"]), ("L1", ["history_count"]), ("L2", ["svd_cos", "history_mean_cos", "svd_dot", "history_length", "history_age_days"])]
    out={}
    for name, fs in cols:
        m=lgb.LGBMClassifier(objective="binary",n_estimators=180,learning_rate=.05,num_leaves=31,min_child_samples=200,random_state=20261028,n_jobs=4,verbosity=-1)
        m.fit(np.column_stack([train[f] for f in fs]), train["label"]); p=m.predict_proba(np.column_stack([val[f] for f in fs]))[:,1]; out[name]={"features":fs,"dm24_auc":safe_auc(val["label"],p)}
    return out


def main():
    raw=load_raw(TRAIN_CACHE); evraw=load_raw(EVAL_CACHE); keymap=global_user_map(raw,evraw); users=global_users(raw,keymap)
    eval_base=read_parquet(CACHE/"features_orig"/"evaluation.parquet"); eval_h=eval_base["history_length"][::10]; eval_g=eval_base["gap_hours"][::10]
    results={"persistence":persistence(raw,users),"folds":{},"single_features":{},"safe_event_reference":0.649}
    all_single={}
    for name,day in [("DM22",2),("DM23",3),("DM24",4)]:
        B=D0+day*DAY; bounds=[D0+int(i*DAY/2) for i in range(1,2*day)]; tr_events=[]; tr_frames=[]
        for b in bounds:
            idx=first_events(raw,users,b,CUT); f=add_c2(read_parquet(CACHE/"features_orig"/f"{b}.parquet"),CACHE/"features_c"/f"c2_{b}.parquet"); cm=choice_fit(f); f=add_choice(f,cm); tr_frames.append(f); tr_events.append(event_features(f,raw,users,idx,b))
        train_e={k:np.concatenate([x[k] for x in tr_events]) for k in tr_events[0]}; idxv=first_events(raw,users,B,CUT); vf=add_c2(read_parquet(CACHE/"features_orig"/f"{B}.parquet"),CACHE/"features_c"/f"c2_{B}.parquet"); cm=choice_fit(concat_frames(tr_frames)); vf=add_choice(vf,cm); val_e=event_features(vf,raw,users,idxv,B)
        m, ecols, ptr, pev=fit_event(train_e,val_e,eval_h,eval_g); edma=ew_auc(val_e["label"],pev,val_e["history_length"],val_e["gap_hours"],eval_h,eval_g)
        singles={}
        for c in BASE_EVENT_NAMES:
            singles[c]=ew_auc(val_e["label"],val_e[c],val_e["history_length"],val_e["gap_hours"],eval_h,eval_g) if np.unique(val_e[c]).size>1 else None; all_single.setdefault(c,[]).append(singles[c])
        eval_rows=eval_base
        # Row-level stack. Its event predictions are generated by E trained on
        # the source windows; train-row predictions are in-sample diagnostics,
        # while validation predictions remain boundary-safe.
        stack_preds=row_stack(tr_frames,vf,train_e,val_e,m,ecols,eval_base)
        row_metrics={}
        for nm in ["baseline","stack","product"]:
            pp=stack_preds[nm]; row_metrics[nm]={"raw_auc":safe_auc(vf["label"],pp),"DM":weighted_dm(vf,pp,eval_base)}
            # Event AUC of the row scores, using sum over candidates.
            row_metrics[nm]["event_auc_raw"]=safe_auc((vf["label"].reshape(-1,10).sum(1)>0).astype(np.int8),pp.reshape(-1,10).sum(1))
        row_metrics["best_blend"]=stack_preds["best_blend"]
        results["folds"][name]={"event_model_auc_reweighted":edma,"event_model_auc_raw":safe_auc(val_e["label"],pev),"single_features":singles,"event_model_features":ecols,"row_models":row_metrics,"stack_top_event_features":stack_preds["top_event_features"]}
        if name=="DM24": results["ladder"]=trained_ladder(concat_frames([add_c2(read_parquet(CACHE/"features_orig"/f"{b}.parquet"),CACHE/"features_c"/f"c2_{b}.parquet") for b in bounds]),vf)
        print(name,"E",edma,flush=True)
    for c,vals in all_single.items(): results["single_features"][c]={"DM24":vals[2],"DM_mean3":float(np.mean(vals))}
    results["event_model_DM_mean3"]=float(np.mean([results["folds"][k]["event_model_auc_reweighted"] for k in ("DM22","DM23","DM24")]))
    results["notes"]="E-model uses boundary-safe user/session/history/slate features. SAFE_v2 stacking is not promoted unless the row-level gate passes; this run records event-model evidence first."
    (OUT/"event_propensity_results.json").write_text(json.dumps(results,indent=2,default=float))
    write_reports(results)


def write_reports(r):
    p=r["persistence"]; lines=["# Event propensity research", "", f"User event-rate persistence: correlation days 1–3 vs 4–6 = **{p['corr_days1_3_vs4_6']}** across {p['users_with_both_periods']} users with events in both periods.", "", "| Idea | Event AUC DM24 | Event AUC mean | DM_mean3 gain |", "|---|---:|---:|---:|"]
    for c,v in sorted(r["single_features"].items(), key=lambda kv: -(kv[1]["DM_mean3"] or 0)):
        lines.append(f"| {c} | {v['DM24']:.6f} | {v['DM_mean3']:.6f} | — |")
    lines += [f"| Event model v2 | {r['folds']['DM24']['event_model_auc_reweighted']:.6f} | {r['event_model_DM_mean3']:.6f} | — |", "", "## Persistence by user event count", "", "| Total events | Users | Correlation |", "|---|---:|---:|"]
    for k,v in p["by_total_event_count"].items(): lines.append(f"| {k} | {v['users']} | {v['corr']} |")
    stack_vals = {nm: [r['folds'][k]['row_models'][nm]['DM'] for k in ('DM22','DM23','DM24')] for nm in ('baseline','stack','product')}
    lines += ["", "## SAFE_v2 feedback", "", "| Variant | DM22 | DM23 | DM24 | DM_mean3 |", "|---|---:|---:|---:|---:|"]
    for nm, vals in stack_vals.items(): lines.append(f"| {nm} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {np.mean(vals):.6f} |")
    lines.append(f"| best global logit blend | — | — | — | {np.mean([r['folds'][k]['row_models']['best_blend']['dm'] for k in ('DM22','DM23','DM24')]):.6f} |")
    (OUT/"event_propensity_report.md").write_text("\n".join(lines)+"\n")
    (OUT/"event_model_debug.md").write_text("# Event model debug\n\nThe prior product's row AUC was low because multiplying a calibrated event probability by a conditional row softmax changes the global row ordering. The corrected E model uses event labels `(sum(clicked over the 10 candidates) > 0)`, target rows selected by the same first-event-per-user indices, and event-level history/gap reweighting matched to evaluation cells. Its reweighted event AUCs are:\n\n"+"\n".join(f"- {k}: {v['event_model_auc_reweighted']:.6f} (raw {v['event_model_auc_raw']:.6f})" for k,v in r['folds'].items())+f"\n\nDM_mean3: **{r['event_model_DM_mean3']:.6f}**; SAFE_v2 reference: **0.649 event AUC**.\n\nThe stacking feedback is included in `event_propensity_report.md`; the experiment gate is 0.793256 and the full gate is 0.799824.\n")
    (OUT/"ladder_report.md").write_text("# SlateNet / trained ladder audit\n\nStatic audit: `src/slatenet.py` correctly constructs `[no_click, c1..c10]`, uses target index `clicked_slot + 1`, and reads prediction columns `1:`, so the suspected off-by-one is not present in the checked source. A neural one-epoch argmax audit could not run because Torch is absent from the local runtime. The trained cached-feature ladder below is the CPU fallback.\n\n| Level | Features | DM24 AUC |\n|---|---|---:|\n"+"\n".join(f"| {k} | {', '.join(v['features'])} | {v['dm24_auc']:.6f} |" for k,v in r.get('ladder',{}).items())+"\n")


if __name__ == "__main__": main()
