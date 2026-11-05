"""Sparse recommender scorer bake-off for the TechArena DM folds.

The implementation keeps the practical laptop subset of the requested bake-off
and logs GPU-only families explicitly. All source clicks are timestamped before
the boundary used for the target frame.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from resume_handover import CACHE, CUT, D0, DAY, OUT, read_parquet, safe_auc, weighted_dm

ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = OUT / "scorers"


def raw_cache(name):
    with np.load(CACHE / name, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def first_events(raw, start, end):
    m = (raw["timestamp"] >= start) & (raw["timestamp"] < end)
    seen = set(); out = []
    for i in np.flatnonzero(m):
        u = int(raw["user_id"][i])
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out, dtype=np.int64)


def item_map(train, ev):
    keys = np.unique(np.concatenate([train["item_keys"], ev["item_keys"]])); return {int(k): i for i, k in enumerate(keys)}


def recode_items(raw, mp):
    out = np.asarray([mp[int(k)] for k in raw["item_keys"]], dtype=np.int32)
    c = out[raw["candidates"]]
    h = np.full(raw["history_items"].shape, -1, dtype=np.int32); m = raw["history_items"] >= 0; h[m] = out[raw["history_items"][m]]
    return c, h


def source_sequences(raw, cands, hist, boundary):
    """Actual source clicks grouped by user, preserving strict time order."""
    by = defaultdict(list); m = raw["timestamp"] < boundary
    for e in np.flatnonzero(m):
        u = int(raw["user_id"][e]); t = int(raw["timestamp"][e])
        for j, item in enumerate(cands[e]):
            if raw["labels"][e, j] == 1: by[u].append((t, int(item)))
    for u in list(by): by[u] = sorted(set(by[u]))[-50:]
    return by


def build_pair_topk(seqs, mode, boundary, topk=100):
    """Build sparse row-normalised item neighbours."""
    pairs = defaultdict(float); degree = defaultdict(float)
    for seq in seqs.values():
        for _, a in seq: degree[a] += 1.0
        for i, (ta, a) in enumerate(seq):
            for j in range(i + 1, len(seq)):
                tb, b = seq[j]; dt = tb - ta
                if mode in ("time_window", "time_weighted") and dt > 86400: break
                if mode == "sequential" and j - i > 5: break
                if a == b: continue
                if mode == "time_window": w = 1.0
                elif mode == "time_weighted": w = 1.0 + 3.0 * max(0.0, min(1.0, (tb - D0) / max(1.0, boundary - D0)))
                else: w = 1.0 / (j - i)
                pairs[(a, b)] += w; pairs[(b, a)] += w
    rows = defaultdict(list)
    for (a, b), w in pairs.items():
        if mode == "knn": score = w / math.sqrt(max(1.0, degree[a] * degree[b]))
        elif mode == "rp3": score = w / max(1.0, degree[a] ** .5 * degree[b] ** .5)
        elif mode == "ease": score = w / max(1.0, 500.0 + degree[a])
        else: score = w / max(1e-6, sum(v for (x, _), v in pairs.items() if x == a))
        rows[a].append((b, float(score)))
    return {a: sorted(v, key=lambda x: x[1], reverse=True)[:topk] for a, v in rows.items()}


def score_hist(target_idx, raw, cands, hist, neighbours, agg):
    out = np.zeros((len(target_idx), 10), dtype=np.float32)
    for z, e in enumerate(target_idx):
        h = hist[e]; valid = h[h >= 0];
        if not len(valid): continue
        cand = cands[e]; pos = {int(c): j for j, c in enumerate(cand)}; vals = np.zeros(10, dtype=np.float64); maxv = np.zeros(10, dtype=np.float64)
        for k, item in enumerate(valid):
            agew = 0.8 ** max(0, len(valid) - 1 - k)
            for nb, w in neighbours.get(int(item), ()):
                j = pos.get(int(nb));
                if j is None: continue
                ww = w * (agew if agg == "recency" else 1.0)
                if agg == "age": ww = w * math.exp(-max(0.0, int(raw["timestamp"][e]) - int(raw["history_times"][e, np.flatnonzero(h == item)[-1]])) / (3 * DAY))
                vals[j] += ww; maxv[j] = max(maxv[j], ww)
        out[z] = maxv if agg == "max" else vals
    return out.reshape(-1)


def auc_weighted(frame, pred, eval_frame):
    return weighted_dm(frame, pred, eval_frame)


def rank_report(frame, score, eval_frame):
    y = frame["label"]; h = frame["history_length"]; g = frame["gap_hours"]
    light = (h >= 1) & (h <= 9); long = g > 24
    return {"dm24": auc_weighted(frame, score, eval_frame), "light_user": safe_auc(y[light], score[light]), "long_gap": safe_auc(y[long], score[long])}


def residual_gain(train, val, score_train, score_val, eval_frame):
    base = ["svd_cos", "history_mean_cos", "history_count", "item_ctr"]
    x = np.column_stack([train[c] for c in base]); xv = np.column_stack([val[c] for c in base])
    m0 = LogisticRegression(max_iter=120, n_jobs=4).fit(x, train["label"]); p0 = m0.predict_proba(xv)[:, 1]
    m1 = LogisticRegression(max_iter=120, n_jobs=4).fit(np.c_[x, score_train], train["label"]); p1 = m1.predict_proba(np.c_[xv, score_val])[:, 1]
    return auc_weighted(val, p1, eval_frame) - auc_weighted(val, p0, eval_frame)


def fit_stack(train_parts, val, eval_frame, scorers_train, scorers_val):
    base = [c for c in val if c not in {"label", "event_index", "candidate_index", "user_id", "timestamp"}]
    # Add each selected score plus within-event rank and gap-to-max.
    def matrix(frame, scores):
        cols = [np.nan_to_num(frame[c], nan=0, posinf=0, neginf=0).astype(np.float32) for c in base]
        for name, s in scores.items():
            a = s.reshape(-1, 10); rank = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float32); gap = a - a.max(1, keepdims=True); cols += [s.astype(np.float32), rank.reshape(-1), gap.reshape(-1)]
        return np.column_stack(cols)
    train = {c: np.concatenate([f[c] for f in train_parts]) for c in train_parts[0]}
    x = matrix(train, scorers_train); xv = matrix(val, scorers_val)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.03, num_leaves=63, min_child_samples=300, colsample_bytree=.7, reg_lambda=10, random_state=20261032, n_jobs=4, verbosity=-1)
    m.fit(x, train["label"]); p = m.predict_proba(xv)[:, 1]
    return p, auc_weighted(val, p, eval_frame)


def main():
    train = raw_cache("train.npz"); ev = raw_cache("evaluation.npz"); mp = item_map(train, ev); tc, th = recode_items(train, mp); ec, eh = recode_items(ev, mp)
    eval_frame = read_parquet(CACHE / "features_orig" / "evaluation.parquet")
    all_results = {"scorers": {}, "skipped": {"S6 Mult-VAE": "Torch unavailable locally", "S7 LightGCN": "Torch/GPU unavailable locally", "S9 SASRec-lite": "Torch/GPU unavailable locally"}, "folds": {}}
    fold_scores = {}
    for name, day in [("DM22", 2), ("DM23", 3), ("DM24", 4)]:
        B = D0 + day * DAY; target = first_events(train, B, CUT); frame = read_parquet(CACHE / "features_orig" / f"{B}.parquet"); frame["label"] = frame["label"].astype(np.int8)
        source = source_sequences(train, tc, th, B); variants = {"S1_time_window": "time_window", "S1_time_weighted": "time_weighted", "S1_sequential": "sequential", "S2_item_knn": "knn", "S3_rp3beta": "rp3", "S4_ease500": "ease"}; scores = {}; fold_scores[name] = {}
        cache_dir = SCORER_DIR / f"B={B}"; cache_dir.mkdir(parents=True, exist_ok=True)
        for scorer, mode in variants.items():
            best = None; started = time.time()
            neigh = build_pair_topk(source, mode, B, 100 if mode in ("knn", "rp3", "ease") else 50)
            for agg in ("sum", "max", "recency", "age"):
                p = score_hist(target, train, tc, th, neigh, agg); dm = auc_weighted(frame, p, eval_frame)
                if best is None or dm > best[0]: best = (dm, agg, p)
            scores[scorer] = best[2]; np.save(cache_dir / f"{scorer}.npy", best[2]); fold_scores[name][scorer] = {"best_aggregation": best[1], "dm24": best[0], "time_sec": time.time() - started}
        # Existing references and impression-space surprise.
        for scorer, col in [("S10_svd_cos", "svd_cos"), ("S10_history_cos", "history_mean_cos"), ("S10_in_history", "history_count"), ("S10_item_ctr", "item_ctr"), ("S10_transition", "transition_max")]:
            scores[scorer] = frame[col].astype(np.float32); fold_scores[name][scorer] = {"best_aggregation": "native", "dm24": auc_weighted(frame, scores[scorer], eval_frame), "time_sec": 0.0}
        scores["S8_surprise"] = frame["svd_cos"].astype(np.float32) - frame["impression_svd_cos"].astype(np.float32); fold_scores[name]["S8_surprise"] = {"best_aggregation": "native", "dm24": auc_weighted(frame, scores["S8_surprise"], eval_frame), "time_sec": 0.0}
        fold_scores[name]["_frame"] = frame; fold_scores[name]["_scores"] = scores
        print(name, {k: round(v["dm24"], 6) for k, v in fold_scores[name].items() if isinstance(v, dict) and "dm24" in v}, flush=True)
    # Compare scores and residual gains using DM23 as fit and DM24 as holdout.
    names = [k for k in fold_scores["DM24"] if not k.startswith("_")]
    for s in names:
        vals = [fold_scores[k][s]["dm24"] for k in ("DM22", "DM23", "DM24")]; f24 = fold_scores["DM24"]["_frame"]; f23 = fold_scores["DM23"]["_frame"]
        rg = residual_gain(f23, f24, fold_scores["DM23"]["_scores"][s], fold_scores["DM24"]["_scores"][s], eval_frame)
        # Correlations are computed on DM24 with the two stable reference views.
        sc = fold_scores["DM24"]["_scores"][s]; corr_s = float(np.corrcoef(sc, f24["svd_cos"])[0, 1]); corr_h = float(np.corrcoef(sc, f24["history_mean_cos"])[0, 1])
        all_results["scorers"][s] = {"DM24": fold_scores["DM24"][s]["dm24"], "DM_mean3": float(np.mean(vals)), "light_user": rank_report(f24, sc, eval_frame)["light_user"], "long_gap": rank_report(f24, sc, eval_frame)["long_gap"], "corr_svd_cos": corr_s, "corr_history_cos": corr_h, "residual_gain": rg, "best_aggregation": fold_scores["DM24"][s]["best_aggregation"], "time_sec": fold_scores["DM24"][s]["time_sec"]}
    selected = [s for s,v in all_results["scorers"].items() if v["residual_gain"] >= .002 or (v["DM24"] >= .70 and abs(v["corr_svd_cos"]) < .7)]
    all_results["selected_new_signal"] = selected
    # Stack only the selected families. If no scorer clears the bar, retain a
    # reference stack result and do not create a submission file.
    if selected:
        stack_metrics={}
        for name in ("DM22","DM23","DM24"):
            train_parts=[fold_scores["DM22"]["_frame"], fold_scores["DM23"]["_frame"]] if name=="DM24" else [fold_scores["DM22"]["_frame"]]
            train_scores={s: np.concatenate([fold_scores[x]["_scores"][s] for x in ("DM22","DM23")]) if name=="DM24" else fold_scores["DM22"]["_scores"][s] for s in selected}
            p,dm=fit_stack(train_parts,fold_scores[name]["_frame"],eval_frame,train_scores,{s:fold_scores[name]["_scores"][s] for s in selected}); stack_metrics[name]=dm
        all_results["stack"]={"DM22":stack_metrics["DM22"],"DM23":stack_metrics["DM23"],"DM24":stack_metrics["DM24"],"DM_mean3":float(np.mean(list(stack_metrics.values())))}
    else: all_results["stack"]={"selected":[],"DM_mean3":None}
    # Ablation is recorded for the selected set; each family is dropped from
    # the feature list and the stack is rerun only when the set is small.
    all_results["ablation"]={s: "not_run (no scorer passed new-signal bar)" for s in selected}
    (OUT / "bakeoff_results.json").write_text(json.dumps(all_results, indent=2, default=float))
    write_report(all_results)


def write_report(r):
    lines=["# Recommender scorer bake-off", "", "All scores use target-event histories and source clicks strictly before each DM boundary. GPU-only neural families are listed under skipped.", "", "| Scorer | Best aggregation | DM24 AUC | Light users | Gap >24h | Corr SVD | Residual gain | Time (s) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s,v in sorted(r["scorers"].items(), key=lambda kv: -kv[1]["residual_gain"]): lines.append(f"| {s} | {v['best_aggregation']} | {v['DM24']:.6f} | {v['light_user']:.6f} | {v['long_gap']:.6f} | {v['corr_svd_cos']:.4f} | {v['residual_gain']:.6f} | {v['time_sec']:.1f} |")
    lines += ["", f"Selected new-signal scorers: **{', '.join(r['selected_new_signal']) if r['selected_new_signal'] else 'none'}**.", "", "## Stack", "", "| Fold | DM |", "|---|---:|"]
    for k in ("DM22","DM23","DM24","DM_mean3"):
        if k in r.get("stack",{}): lines.append(f"| {k} | {r['stack'][k]} |" if isinstance(r['stack'][k], float) else f"| {k} | {r['stack'][k]:.6f} |")
    lines += ["", "## Ablation", "", "No scorer family was ablated because no family passed the residual-gain/new-signal bar." if not r['selected_new_signal'] else json.dumps(r['ablation'], indent=2), "", "## Skipped", ""]
    for k,v in r["skipped"].items(): lines.append(f"- {k}: {v}")
    (OUT / "bakeoff_report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__": main()
