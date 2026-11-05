"""Evaluate SAFE_v2 features augmented with the validated logistic-MF cosine.

This is deliberately a bounded DM24 gate.  The logistic-MF feature is fit at
each training/validation boundary and scored on that boundary's first-event
targets, so the comparison is aligned with the SAFE_v2 feature path.  A
production file is only built after the DM24 residual gate is positive.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import join_global
from .choice import fit_choice
from .logistic_mf_final import fit_model
from .phase_a import weighted_auc
from .phase_c_real import cached_frame
from .phase_c_revision import (
    FULL_SEEDS,
    add_choice_pair,
    add_family,
    bounds_for,
    train_frames,
    target_for,
)
from .phase_c_real import shrink_extra
from .run_phase1 import first_events, subset
from .submit import validate_submission, write_submission
from .train import feature_columns


LAMBDA = 10.0
FEATURE = "logmf_cosine"
PROD_TREES = 350
PROD_CHOICE_MAXITER = 20


def score_events(data, events, model) -> np.ndarray:
    """Score arbitrary CachedEvents in global user/item code space."""
    U = model["U"]
    V = model["V"]
    out = np.empty(events.n_events * 10, dtype=np.float32)
    for e in range(events.n_events):
        u = U[int(events.user_id[e])]
        c = events.candidates[e].astype(np.int64, copy=False)
        v = V[c]
        out[e * 10 : (e + 1) * 10] = (v @ u) / (
            (np.linalg.norm(v, axis=1) + 1e-8) * (np.linalg.norm(u) + 1e-8)
        )
    return out


def add_score(frame, score):
    if len(frame) != len(score):
        raise ValueError(f"row mismatch for {FEATURE}: frame={len(frame)} score={len(score)}")
    out = frame.reset_index(drop=True).copy()
    out[FEATURE] = np.asarray(score, dtype=np.float32)
    return out


def add_choice_pair_bounded(train, val):
    """Same SAFE choice features, with a bounded fit for this cheap probe."""
    choice = fit_choice(train, maxiter=PROD_CHOICE_MAXITER)
    tu, tp, tn = choice.predict(train)
    train = train.copy()
    train["choice_utility"] = tu
    train["choice_probability"] = tp
    train["choice_no_click"] = np.repeat(tn, 10)
    vu, vp, vn = choice.predict(val)
    val = val.copy()
    val["choice_utility"] = vu
    val["choice_probability"] = vp
    val["choice_no_click"] = np.repeat(vn, 10)
    return train, val


def fit_logmf_score(data, boundary, label, events=None):
    path = OUTPUT_DIR / f"logmf_cosine_{label}_{int(boundary)}.npy"
    if path.exists():
        return np.load(path), {"cached": True, "fit_sec": 0.0}
    t0 = time.time()
    model = fit_model(data, int(boundary), LAMBDA, alternations=3)
    target = target_for(data, int(boundary)) if events is None else events
    score = score_events(data, target, model)
    np.save(path, score)
    return score, {"cached": False, "fit_sec": time.time() - t0, "log_rows": model["log_rows"]}


def run_dm24(data):
    boundary = D0 + 4 * DAY
    train_bounds = [D0 + i * DAY for i in range(1, 4)]
    # This is the same A3+C1+C2+choice family used by SAFE_v2 validation.
    train, val, _ = train_frames(data, boundary, train_bounds, "a3_c1_c2")
    train_parts = []
    fit_meta = {}
    offset = 0
    for wi, b in enumerate(train_bounds):
        n = int((train["window_id"].to_numpy() == wi).sum())
        score, meta = fit_logmf_score(data, b, f"train{wi}")
        if n != len(score):
            raise ValueError(f"window {wi} frame/score mismatch: {n}/{len(score)}")
        part = train.iloc[offset : offset + n].copy()
        part[FEATURE] = score
        train_parts.append(part)
        offset += n
        fit_meta[f"train_{int(b)}"] = meta
    train = __import__("pandas").concat(train_parts, ignore_index=True)
    val_score, meta = fit_logmf_score(data, boundary, "val_dm24")
    val = add_score(val, val_score)
    fit_meta["val"] = meta

    # Use the existing fast, early-stopped SAFE-family comparison for both
    # models.  This measures incremental value of the MF feature, not a new
    # tree configuration.
    base_cols = feature_columns(train.drop(columns=[FEATURE]))
    model0 = lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.1,
        num_leaves=63, max_bin=127, min_child_samples=300,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_lambda=10.0, random_state=20261024, n_jobs=4, verbosity=-1,
    )
    model0.fit(train[base_cols], train["label"].astype(int),
               eval_set=[(val[base_cols], val["label"].astype(int))],
               eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    p0 = model0.predict_proba(val[base_cols], num_iteration=model0.best_iteration_)[:, 1]
    dm0 = weighted_auc(val, p0, cached_frame(data, CUT, "eval"))

    cols = feature_columns(train)
    model1 = lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.1,
        num_leaves=63, max_bin=127, min_child_samples=300,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_lambda=10.0, random_state=20261024, n_jobs=4, verbosity=-1,
    )
    model1.fit(train[cols], train["label"].astype(int),
               eval_set=[(val[cols], val["label"].astype(int))],
               eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    p1 = model1.predict_proba(val[cols], num_iteration=model1.best_iteration_)[:, 1]
    dm1 = weighted_auc(val, p1, cached_frame(data, CUT, "eval"))
    result = {
        "fold": "DM24",
        "boundary": int(boundary),
        "lambda": LAMBDA,
        "feature": FEATURE,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "base_dm24": float(dm0["joint_auc_supported"]),
        "with_logmf_dm24": float(dm1["joint_auc_supported"]),
        "gain": float(dm1["joint_auc_supported"] - dm0["joint_auc_supported"]),
        "base_raw_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(val["label"], p0)),
        "with_raw_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(val["label"], p1)),
        "base_best_iteration": int(model0.best_iteration_),
        "with_best_iteration": int(model1.best_iteration_),
        "fit_meta": fit_meta,
    }
    np.savez_compressed(OUTPUT_DIR / "logmf_dm24_stack_predictions.npz", base=p0, with_logmf=p1)
    (OUTPUT_DIR / "logmf_stack_dm24.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def build_production(data):
    """Fit the full SAFE family with the logistic-MF feature and export it."""
    train_bounds = bounds_for(6, half=True)
    train_parts = []
    fit_meta = {}
    for wi, b in enumerate(train_bounds):
        frame = add_family(data, cached_frame(data, b).copy(), b, "a3_c1_c2")
        score, meta = fit_logmf_score(data, b, f"prod{wi}")
        frame = add_score(frame, score)
        frame["window_id"] = wi
        train_parts.append(frame)
        fit_meta[f"train_{int(b)}"] = meta
        print("production window", int(b), len(frame), flush=True)

    import pandas as pd

    train = pd.concat(train_parts, ignore_index=True)
    eval_frame = cached_frame(data, CUT, "eval").copy()
    eval_extra = shrink_extra(data, data.evaluation, CUT, k=10.0, uncertainty=True)
    eval_frame = pd.concat([eval_frame.reset_index(drop=True), eval_extra.reset_index(drop=True)], axis=1)
    eval_score, meta = fit_logmf_score(data, CUT, "prod_eval", events=data.evaluation)
    eval_frame = add_score(eval_frame, eval_score)
    train, eval_frame = add_choice_pair_bounded(train, eval_frame)
    cols = feature_columns(train)
    scores = np.zeros(len(eval_frame), dtype=np.float64)
    best_iterations = []
    for seed in FULL_SEEDS:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=PROD_TREES, learning_rate=0.05,
            num_leaves=63, max_bin=127, min_child_samples=300,
            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            reg_lambda=10.0, random_state=seed, n_jobs=4, verbosity=-1,
        )
        model.fit(train[cols], train["label"].astype(int))
        scores += model.predict_proba(eval_frame[cols])[:, 1]
        best_iterations.append(PROD_TREES)
        print("production seed complete", seed, flush=True)
    scores = np.clip(scores / len(FULL_SEEDS), 1e-7, 1 - 1e-7)
    rowids = data.evaluation.rowids.reshape(-1)
    evaluation = pd.DataFrame({"rowid": rowids})
    example = pd.read_csv(EXAMPLE_PATH)
    out = OUTPUT_DIR / "SAFE_LOGMF10.csv"
    write_submission(rowids, scores, evaluation, out, example)
    written = pd.read_csv(out)
    validate_submission(written, evaluation, example)
    checks = {
        "path": str(out),
        "rows": int(len(written)),
        "unique_rowids": int(written.rowid.nunique()),
        "missing_rowids": int(len(set(example.rowid) - set(written.rowid))),
        "duplicate_rowids": int(written.rowid.duplicated().sum()),
        "min": float(written.predicted_probability.min()),
        "max": float(written.predicted_probability.max()),
        "mean": float(written.predicted_probability.mean()),
        "seeds": list(FULL_SEEDS),
        "trees": PROD_TREES,
        "lambda": LAMBDA,
        "feature": FEATURE,
        "fit_meta": {**fit_meta, "eval": meta},
    }
    (OUTPUT_DIR / "SAFE_LOGMF10_checks.json").write_text(json.dumps(checks, indent=2, default=float))
    return checks


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    result = {} if "--production-only" in __import__("sys").argv else run_dm24(data)
    if "--production" in __import__("sys").argv or "--production-only" in __import__("sys").argv:
        result["production"] = build_production(data)
        (OUTPUT_DIR / "logmf_stack_production.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
