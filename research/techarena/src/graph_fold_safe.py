"""Fold-safe graph scorers and a bounded SAFE_LOGMF10 graph audit.

All graph state is built from ``click_log_before(data, B)``.  The script is
deliberately self contained so that the graph family can be reproduced after
the original graph80 generator was lost.  It reports the per-fold single
feature metrics first, then runs a small within-slate graph-feature stack on
DM24.  A production file is written only when the requested local gate is
met.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import click_log_before, join_global
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, train_frames
from .run_phase1 import first_events, subset
from .submit import validate_submission, write_submission


FOLDS = (("DM22", 2), ("DM23", 3), ("DM24", 4))
NAMES = (
    "item_knn",
    "user_knn",
    "rp3_a05_b03", "rp3_a05_b05", "rp3_a05_b07",
    "rp3_a10_b03", "rp3_a10_b05", "rp3_a10_b07",
    "p3_a05", "p3_a10", "item_pagerank",
)


def _target(data, boundary):
    idx = first_events(data.train, boundary, CUT)
    return subset(data.train, idx), np.asarray(idx, dtype=np.int64)


def _sequences(log):
    """Chronological, unique item sequences for each user."""
    users = defaultdict(list)
    for u, i, ts in log:
        users[int(u)].append((int(ts), int(i)))
    out = {}
    for u, vals in users.items():
        vals.sort()
        seen = set()
        seq = []
        for _, i in vals:
            if i not in seen:
                seen.add(i); seq.append(i)
        out[u] = np.asarray(seq, dtype=np.int32)
    return out


def _pair_table(seqs, n_items):
    """Return directed co-click pairs and item click degree.

    Pairs are represented as one int64 code and aggregated with numpy rather
    than a Python dictionary.  This keeps the 50k-item graph bounded on the
    laptop while retaining the complete boundary-safe click log.
    """
    chunks = []
    degree = np.zeros(n_items, dtype=np.float64)
    for a in seqs.values():
        if len(a) == 0:
            continue
        degree[a] += 1.0
        if len(a) < 2:
            continue
        # Every ordered pair is used by the directed random-walk scorers.
        x = np.repeat(a, len(a))
        y = np.tile(a, len(a))
        keep = x != y
        chunks.append((x[keep].astype(np.int64) * n_items + y[keep]).astype(np.int64))
    if not chunks:
        return np.empty(0, np.int64), np.empty(0, np.int32), degree
    codes = np.concatenate(chunks)
    codes.sort()
    uniq, counts = np.unique(codes, return_counts=True)
    return uniq.astype(np.int64), counts.astype(np.int32), degree


def _ranges(codes, n_items):
    src = codes // n_items
    starts = np.searchsorted(src, np.arange(n_items), side="left")
    ends = np.searchsorted(src, np.arange(n_items), side="right")
    return starts, ends


def _knn_rows(codes, counts, degree, n_items, topk=100, shrink=10.0):
    starts, ends = _ranges(codes, n_items)
    rows = {}
    for s in np.flatnonzero(ends > starts):
        a, b = int(starts[s]), int(ends[s])
        dst = codes[a:b] % n_items
        w = counts[a:b].astype(np.float64) / (np.sqrt(max(degree[s], 1.0) * np.maximum(degree[dst], 1.0)) + shrink)
        if len(w) > topk:
            keep = np.argpartition(w, -topk)[-topk:]
            keep = keep[np.argsort(w[keep])[::-1]]
        else:
            keep = np.argsort(w)[::-1]
        rows[int(s)] = (dst[keep].astype(np.int32), w[keep].astype(np.float32))
    return rows


def _score_history(target, rows, n_items, mode="sum"):
    out = np.zeros((target.n_events, 10), dtype=np.float32)
    for e in range(target.n_events):
        h = target.history_items[e]
        h = h[h >= 0]
        if len(h) == 0:
            continue
        pos = {int(c): j for j, c in enumerate(target.candidates[e])}
        vals = np.zeros(10, dtype=np.float64)
        mx = np.zeros(10, dtype=np.float64)
        for k, src in enumerate(h):
            neigh = rows.get(int(src))
            if neigh is None:
                continue
            # Recent history receives a mild preference for the recency-aware
            # graph family; plain graph scorers use one.
            rw = 1.0
            for dst, w in zip(*neigh):
                j = pos.get(int(dst))
                if j is not None:
                    vals[j] += float(w) * rw
                    mx[j] = max(mx[j], float(w) * rw)
        out[e] = mx if mode == "max" else vals
    return out.reshape(-1)


def _weighted_rows(codes, counts, degree, n_items, alpha, beta, topk=100):
    starts, ends = _ranges(codes, n_items)
    rows = {}
    for s in np.flatnonzero(ends > starts):
        a, b = int(starts[s]), int(ends[s])
        dst = codes[a:b] % n_items
        den = np.maximum(degree[s], 1.0) ** alpha * np.maximum(degree[dst], 1.0) ** beta
        w = counts[a:b].astype(np.float64) / den
        if len(w) > topk:
            keep = np.argpartition(w, -topk)[-topk:]
            keep = keep[np.argsort(w[keep])[::-1]]
        else:
            keep = np.argsort(w)[::-1]
        rows[int(s)] = (dst[keep].astype(np.int32), w[keep].astype(np.float32))
    return rows


def _user_knn_score(data, target, seqs, topk=50):
    """Top-overlap user kNN, with an inverted item-to-user index."""
    inv = defaultdict(list)
    for u, a in seqs.items():
        for i in a:
            inv[int(i)].append(int(u))
    seqsets = {u: set(a.tolist()) for u, a in seqs.items()}
    out = np.zeros((target.n_events, 10), np.float32)
    for e in range(target.n_events):
        u = int(target.user_id[e]); h = target.history_items[e]; h = h[h >= 0]
        if len(h) == 0:
            continue
        overlap = defaultdict(int)
        for i in h:
            for v in inv.get(int(i), ()):
                if v != u: overlap[v] += 1
        if not overlap:
            continue
        cand_users = sorted(overlap.items(), key=lambda q: (q[1] / math.sqrt(max(len(seqsets.get(q[0], ())), 1) * max(len(h), 1)), q[1]), reverse=True)[:topk]
        weights = []
        for v, ov in cand_users:
            sim = ov / math.sqrt(max(len(seqs.get(v, ())), 1) * len(h))
            weights.append((v, sim))
        den = sum(w for _, w in weights) + 1e-8
        for j, c in enumerate(target.candidates[e]):
            num = sum(w for v, w in weights if int(c) in seqsets.get(v, set()))
            out[e, j] = num / den
    return out.reshape(-1)


def _pagerank(codes, counts, degree, n_items, topk=100):
    starts, ends = _ranges(codes, n_items)
    rows, cols, vals = [], [], []
    for s in np.flatnonzero(ends > starts):
        a, b = int(starts[s]), int(ends[s]); dst = codes[a:b] % n_items
        w = counts[a:b].astype(np.float64); z = w.sum()
        if len(w) > topk:
            keep = np.argpartition(w, -topk)[-topk:]
            dst, w = dst[keep], w[keep]
            z = w.sum()
        if z <= 0: continue
        rows.extend(dst.tolist()); cols.extend([int(s)] * len(dst)); vals.extend((w / z).tolist())
    if not vals:
        return np.ones(n_items, np.float32) / n_items
    A = sparse.csc_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float64)
    p = np.ones(n_items, np.float64) / n_items
    for _ in range(40):
        q = 0.15 / n_items + 0.85 * (A @ p)
        if np.max(np.abs(q - p)) < 1e-10: break
        p = q
    return p.astype(np.float32)


def _score_fold(data, boundary, include_user=True):
    t0 = time.time()
    target, _ = _target(data, boundary)
    frame = pd.read_parquet(CACHE_DIR / "features_orig" / f"{int(boundary)}.parquet")
    evalf = pd.read_parquet(CACHE_DIR / "features_orig" / "evaluation.parquet")
    log = click_log_before(data, boundary)
    seqs = _sequences(log)
    codes, counts, degree = _pair_table(seqs, data.n_items)
    knn = _knn_rows(codes, counts, degree, data.n_items)
    scores = {"item_knn": _score_history(target, knn, data.n_items)}
    if include_user:
        scores["user_knn"] = _user_knn_score(data, target, seqs)
    for a in (0.5, 1.0):
        for b in (0.3, 0.5, 0.7):
            scores[f"rp3_a{int(a*10):02d}_b{int(b*10):02d}"] = _score_history(target, _weighted_rows(codes, counts, degree, data.n_items, a, b), data.n_items)
    for a in (0.5, 1.0):
        scores[f"p3_a{int(a*10):02d}"] = _score_history(target, _weighted_rows(codes, counts, degree, data.n_items, a, 0.0), data.n_items)
    pr = _pagerank(codes, counts, degree, data.n_items)
    scores["item_pagerank"] = pr[target.candidates.reshape(-1)].astype(np.float32)
    return {"frame": frame, "evalf": evalf, "scores": scores, "log_rows": int(len(log)), "elapsed_sec": time.time() - t0}


def _metrics(frame, score, evalf, lmf=None):
    y = frame["label"].to_numpy()
    light = frame["history_length"].to_numpy().astype(int)
    m = (light >= 1) & (light <= 9)
    w = weighted_auc(frame, score, evalf)
    out = {
        "dm": float(w["joint_auc_supported"]),
        "raw_auc": float(roc_auc_score(y, score)),
        "light_user_auc": float(roc_auc_score(y[m], score[m])) if np.unique(y[m]).size > 1 else None,
        "corr_svd_cos": float(spearmanr(score, frame["svd_cos"].to_numpy()).statistic),
    }
    if lmf is not None:
        out["corr_lmf"] = float(spearmanr(score, lmf).statistic)
    return out


def _add_graph_features(frame, scores, prefix="g"):
    out = frame.reset_index(drop=True).copy()
    for name, s in scores.items():
        z = np.asarray(s, np.float32).reshape(-1, 10)
        out[f"{prefix}_{name}_score"] = z.reshape(-1)
        out[f"{prefix}_{name}_rank"] = np.argsort(np.argsort(-z, axis=1), axis=1).astype(np.float32).reshape(-1)
        out[f"{prefix}_{name}_gap"] = (z - z.max(axis=1, keepdims=True)).astype(np.float32).reshape(-1)
    return out


def _base_cols(frame):
    return [c for c in frame.columns if c not in {"label", "event_index", "candidate_index", "user_id", "timestamp", "window_id", "sampling_weight"}]


def _fit_stack(dm24, data):
    """DM24 residual test using the same SAFE_LOGMF10 LMF feature path."""
    from .mf_variant_experiment import add_base, sample_events
    train_bounds = bounds_for(4, half=True)
    train, val, _ = train_frames(data, D0 + 4 * DAY, train_bounds, "a3_c1_c2", choice_maxiter=20)
    parts = []; off = 0
    for wi, b in enumerate(train_bounds):
        n = int((train["window_id"].to_numpy() == wi).sum())
        p = OUTPUT_DIR / f"logmf_cosine_prod{wi}_{int(b)}.npy"
        part = train.iloc[off:off+n].copy(); part = add_base(part, np.load(p)); parts.append(part); off += n
    train = pd.concat(parts, ignore_index=True)
    val = add_base(val, np.load(OUTPUT_DIR / f"logmf_cosine_val_dm24_{D0+4*DAY}.npy"))
    # Use the graph scores from the corresponding half-day windows.  Rebuilding
    # these is intentionally explicit and boundary-safe; cache them below.
    gtrain = []
    for wi, b in enumerate(train_bounds):
        cache = OUTPUT_DIR / f"graph_scores_{int(b)}.npz"
        if not cache.exists():
            # User-kNN is the only scorer that is expensive enough to make a
            # seven-window residual probe impractical.  The requested user-
            # kNN is fully evaluated on DM22/23/24 above; the stack uses the
            # item-graph family that can be rebuilt in a bounded time.
            fold = _score_fold(data, b, include_user=False)
            np.savez_compressed(cache, **fold["scores"])
        z = np.load(cache)
        s = {k: z[k] for k in z.files if k != "user_knn"}
        m = train["window_id"].to_numpy() == wi
        gtrain.append(_add_graph_features(train.loc[m].reset_index(drop=True), s))
    train_g = pd.concat(gtrain, ignore_index=True)
    val_g = _add_graph_features(val, {k: v for k, v in dm24["scores"].items() if k != "user_knn"})
    # Compare SAFE_LOGMF10 and the within-slate graph extension.
    evalf = dm24["evalf"]
    sample = sample_events(train)
    weights = importance_weights(sample, evalf)
    results = {}
    for tag, tr, va in (("base", train, val), ("graph", train_g, val_g)):
        cols = _base_cols(tr)
        model = lgb.LGBMClassifier(objective="binary", n_estimators=250, learning_rate=.05,
            num_leaves=63, max_bin=127, min_child_samples=300, colsample_bytree=.7,
            subsample=.8, subsample_freq=1, reg_lambda=10., random_state=20261025, n_jobs=8, verbosity=-1)
        xx = sample[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0) if tag=="base" else sample_events(train_g)[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        yy = sample["label"].astype(int) if tag=="base" else sample_events(train_g)["label"].astype(int)
        vv = va[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        model.fit(xx, yy, sample_weight=weights if tag=="base" else importance_weights(sample_events(train_g),evalf), eval_set=[(vv, va["label"].astype(int))], eval_metric="auc", callbacks=[lgb.early_stopping(60, verbose=False)])
        pred=model.predict_proba(vv,num_iteration=model.best_iteration_)[:,1]
        results[tag]={"dm24":float(weighted_auc(va,pred,evalf)["joint_auc_supported"]),"best_iteration":int(model.best_iteration_)}
    results["gain"] = results["graph"]["dm24"] - results["base"]["dm24"]
    return results


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    report = {"protocol": "click_log_before per boundary; first event per user; 50,000 item codes", "folds": {}, "scorers": {}, "stack": {}}
    for name, day in FOLDS:
        b = D0 + day * DAY
        print("building", name, b, flush=True)
        score_cache = OUTPUT_DIR / f"graph_scores_{int(b)}.npz"
        if score_cache.exists():
            z = np.load(score_cache)
            fold = {"scores": {k: z[k] for k in z.files}, "frame": pd.read_parquet(CACHE_DIR / "features_orig" / f"{int(b)}.parquet"), "evalf": pd.read_parquet(CACHE_DIR / "features_orig" / "evaluation.parquet"), "log_rows": -1, "elapsed_sec": 0.0}
        else:
            fold = _score_fold(data, b)
        report["folds"][name] = {"boundary": int(b), "log_rows": fold["log_rows"], "elapsed_sec": fold["elapsed_sec"]}
        lmf = None
        if name == "DM24":
            p = OUTPUT_DIR / f"logmf_cosine_val_dm24_{int(b)}.npy"
            if p.exists(): lmf = np.load(p)
        for s, p in fold["scores"].items():
            m = _metrics(fold["frame"], p, fold["evalf"], lmf=lmf)
            report["scorers"].setdefault(s, {})[name] = m
        np.savez_compressed(OUTPUT_DIR / f"graph_scores_{int(b)}.npz", **fold["scores"])
        print(name, {s: round(report["scorers"][s][name]["dm"], 6) for s in fold["scores"]}, flush=True)
    for s in NAMES:
        vals = [report["scorers"][s][name]["dm"] for name, _ in FOLDS]
        report["scorers"][s]["dm_mean3"] = float(np.mean(vals))
    print("running DM24 graph-feature residual", flush=True)
    z = np.load(OUTPUT_DIR / f"graph_scores_{int(D0+4*DAY)}.npz")
    dm24 = {"scores": {k: z[k] for k in z.files}, "frame": pd.read_parquet(CACHE_DIR / "features_orig" / f"{int(D0+4*DAY)}.parquet"), "evalf": pd.read_parquet(CACHE_DIR / "features_orig" / "evaluation.parquet")}
    try:
        report["stack"] = _fit_stack(dm24, data)
    except Exception as exc:
        report["stack"] = {"status": "failed", "error": repr(exc)}
        print("stack failed", repr(exc), flush=True)
    # Full production is gated on the experiment residual and is intentionally
    # not attempted by default; no user-visible file is created on a fail.
    (OUTPUT_DIR / "graph_fold_safe_results.json").write_text(json.dumps(report, indent=2, default=float))
    lines = ["# Fold-safe graph family", "", "All graph state uses `click_log_before(data, B)`.", "", "| scorer | DM22 | DM23 | DM24 | mean | corr SVD (DM24) | corr LMF (DM24) | light AUC (DM24) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in NAMES:
        r = report["scorers"][s]; d = r["DM24"]
        lines.append(f"| {s} | {r['DM22']['dm']:.6f} | {r['DM23']['dm']:.6f} | {d['dm']:.6f} | {r['dm_mean3']:.6f} | {d.get('corr_svd_cos', float('nan')):.4f} | {d.get('corr_lmf', float('nan')):.4f} | {d.get('light_user_auc', float('nan')):.6f} |")
    lines += ["", "## Stack", json.dumps(report["stack"], indent=2, default=float)]
    (OUTPUT_DIR / "graph_fold_safe_report.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
