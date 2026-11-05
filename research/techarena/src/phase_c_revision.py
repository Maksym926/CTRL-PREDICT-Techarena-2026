"""Corrected gate audit, experiment combinations, and final-mode check.

This module keeps the original feature path and the cached boundary frames,
but separates C1 point-estimate columns from C2 uncertainty columns.  It
never writes a submission unless the requested full-data gate is met.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import join_global
from .features import build_features, build_stats
from .phase_a import bucket_auc, weighted_auc
from .run_phase1 import first_events, subset
from .svd_model import fit_impression_svd, fit_svd
from .submit import write_submission
from .train import feature_columns, fit_binary
from .phase_c_real import cached_frame, downsample_events, importance_weights, shrink_extra


EXPERIMENT_BASELINE = 0.7881166670123592
FULL_SEEDS = (20261024, 20261025, 20261026)
C1_COLS = ["shrink_dot", "shrink_cos", "shrink_norm"]
C2_COLS = C1_COLS + ["posterior_std", "posterior_utility_std"]
CHOICE_COLS = ["choice_probability", "choice_utility", "choice_no_click"]


def target_for(data, boundary):
    return subset(data.train, first_events(data.train, boundary, CUT))


def bounds_for(day: int, half: bool = True):
    if half:
        return [D0 + int(i * DAY / 2) for i in range(1, 2 * day)]
    return [D0 + i * DAY for i in range(1, day)]


def cached_c2_extra(data, boundary):
    path = CACHE_DIR / "features_c" / f"c2_{int(boundary)}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_parquet(path)
    extra = shrink_extra(data, target_for(data, boundary), boundary, k=10.0, uncertainty=True)
    extra.to_parquet(path, index=False)
    print("cached", path, flush=True)
    return extra


def add_family(data, frame, boundary, family):
    if family == "a3":
        return frame
    extra = cached_c2_extra(data, boundary)
    cols = C1_COLS if family == "a3_c1" else C2_COLS
    return pd.concat([frame.reset_index(drop=True), extra[cols].reset_index(drop=True)], axis=1)


def add_choice_pair(train, val):
    choice = fit_choice(train, maxiter=80)
    tu, tp, tn = choice.predict(train)
    vu, vp, vn = choice.predict(val)
    train = train.copy()
    val = val.copy()
    train["choice_utility"] = tu
    train["choice_probability"] = tp
    train["choice_no_click"] = np.repeat(tn, 10)
    val["choice_utility"] = vu
    val["choice_probability"] = vp
    val["choice_no_click"] = np.repeat(vn, 10)
    return train, val, choice


def train_frames(data, boundary, train_bounds, family, choice_maxiter=80):
    train_parts = []
    for wi, b in enumerate(train_bounds):
        f = add_family(data, cached_frame(data, b).copy(), b, family)
        f["window_id"] = wi
        train_parts.append(f)
    train = pd.concat(train_parts, ignore_index=True)
    val = add_family(data, cached_frame(data, boundary).copy(), boundary, family)
    if family != "a3":
        # The default preserves the original SAFE audit.  Bounded probes can
        # lower the optimizer budget while keeping the same feature family.
        if choice_maxiter == 80:
            train, val, choice = add_choice_pair(train, val)
        else:
            choice = fit_choice(train, maxiter=choice_maxiter)
            train = train.copy(); val = val.copy()
            tu, tp, tn = choice.predict(train); vu, vp, vn = choice.predict(val)
            train["choice_utility"] = tu; train["choice_probability"] = tp; train["choice_no_click"] = np.repeat(tn, 10)
            val["choice_utility"] = vu; val["choice_probability"] = vp; val["choice_no_click"] = np.repeat(vn, 10)
    else:
        choice = None
    return train, val, choice


def fit_experiment_fold(data, boundary, train_bounds, eval_frame, family):
    train, val, choice = train_frames(data, boundary, train_bounds, family)
    sample = downsample_events(train)
    weights = importance_weights(sample, eval_frame)
    model, cols, pred, auc = _fit_fast(sample, val, weights, seed=20261024)
    out = {
        "auc": auc,
        "reweight": weighted_auc(val, pred, eval_frame),
        "buckets": bucket_auc(val, pred),
        "model": model,
        "columns": cols,
        "train": sample,
        "val": val,
        "prediction": pred,
        "choice": choice,
    }
    return out


def _fit_fast(train, val, weights, seed):
    cols = feature_columns(train)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=0.1,
        num_leaves=63, max_bin=127, min_child_samples=300,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_lambda=10.0, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(
        train[cols], train["label"].astype(int), sample_weight=weights,
        eval_set=[(val[cols], val["label"].astype(int))], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = model.predict_proba(val[cols], num_iteration=model.best_iteration_)[:, 1]
    return model, cols, pred, float(roc_auc_score(val["label"], pred))


def run_experiment(data, eval_frame):
    result = {}
    for family in ["a3", "a3_c1", "a3_c1_c2"]:
        result[family] = {}
        for name, day in [("DM23", 3), ("DM24", 4)]:
            b = D0 + day * DAY
            fold = fit_experiment_fold(data, b, bounds_for(day, half=True), eval_frame, family)
            result[family][name] = {k: v for k, v in fold.items() if k not in {"model", "columns", "train", "val", "prediction", "choice"}}
            if family == "a3_c1_c2":
                # Retain only the corrected C2 folds for the feature audit.
                result[family][name]["_objects"] = fold
            print("experiment", family, name, "auc=%.6f dm=%.6f" % (fold["auc"], fold["reweight"]["joint_auc_supported"]), flush=True)
        x = result[family]
        x["DM_mean"] = float((x["DM23"]["reweight"]["joint_auc_supported"] + x["DM24"]["reweight"]["joint_auc_supported"]) / 2.0)
    return result


def c2_audit(folds):
    audit = {"columns": C2_COLS, "folds": {}}
    for name in ["DM23", "DM24"]:
        fold = folds["a3_c1_c2"][name]["_objects"]
        train, val, model, cols = fold["train"], fold["val"], fold["model"], fold["columns"]
        variance = train[C2_COLS].var(ddof=0).to_dict()
        single = {}
        for col in C2_COLS:
            values = val[col].to_numpy()
            single[col] = float(roc_auc_score(val["label"], values)) if np.unique(values).size > 1 else None
        gains = model.booster_.feature_importance(importance_type="gain")
        gain_map = {c: float(gains[i]) for i, c in enumerate(cols)}
        audit["folds"][name] = {
            "variance": {k: float(v) for k, v in variance.items()},
            "single_feature_auc": single,
            "lightgbm_gain": {c: gain_map.get(c, 0.0) for c in C2_COLS},
            "training_columns": [c for c in cols if c in C2_COLS],
        }
    return audit


def full_production_fold(data, boundary, train_bounds, eval_frame):
    train, val, choice = train_frames(data, boundary, train_bounds, "a3")
    # Original production includes the frozen global no-click choice features.
    if choice is None:
        train, val, choice = add_choice_pair(train, val)
    model, cols, pred, auc, imp = fit_binary(train, val, seed=20261024)
    return {
        "auc": auc,
        "reweight": weighted_auc(val, pred, eval_frame),
        "buckets": bucket_auc(val, pred),
        "best_iteration": int(model.best_iteration_),
        "columns": cols,
        "model": model,
        "train": train,
        "val": val,
        "pred": pred,
        "choice": choice,
        "importance": imp,
    }


def run_full_production(data, eval_frame):
    result = {}
    for name, day in [("DM23", 3), ("DM24", 4)]:
        b = D0 + day * DAY
        result[name] = full_production_fold(data, b, bounds_for(day, half=False), eval_frame)
        print("full production", name, result[name]["reweight"], flush=True)
    result["DM_mean"] = float((result["DM23"]["reweight"]["joint_auc_supported"] + result["DM24"]["reweight"]["joint_auc_supported"]) / 2.0)
    return result


def final_seed_model(train, val, seed):
    cols = feature_columns(train)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=2500, learning_rate=0.03,
        num_leaves=63, max_bin=127, min_child_samples=300,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_lambda=10.0, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(train[cols], train["label"].astype(int), eval_set=[(val[cols], val["label"].astype(int))], eval_metric="auc", callbacks=[lgb.early_stopping(80, verbose=False)])
    return model, cols


def final_fold(data, boundary, train_bounds, eval_frame, family):
    train, val, choice = train_frames(data, boundary, train_bounds, family)
    # Final mode is full data: no event downsampling and no importance weights.
    preds=[]; best=[]
    for seed in FULL_SEEDS:
        model, cols = final_seed_model(train, val, seed)
        preds.append(model.predict_proba(val[cols], num_iteration=model.best_iteration_)[:,1])
        best.append(int(model.best_iteration_))
    pred=np.mean(preds,axis=0)
    return {"auc":float(roc_auc_score(val["label"],pred)),"reweight":weighted_auc(val,pred,eval_frame),"buckets":bucket_auc(val,pred),"best_iterations":best,"train":train,"val":val,"choice":choice,"columns":cols}


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    eval_frame = cached_frame(data, CUT, "eval")
    experiment = run_experiment(data, eval_frame)
    audit = c2_audit(experiment)
    production = run_full_production(data, eval_frame)
    # The best experiment candidate must clear +0.004 versus the corrected
    # experiment baseline before it is eligible for the expensive final mode.
    candidates = {k: experiment[k]["DM_mean"] for k in ["a3", "a3_c1", "a3_c1_c2"]}
    best_family = max(candidates, key=candidates.get)
    eligible = candidates[best_family] - EXPERIMENT_BASELINE >= 0.004
    final = {"eligible": bool(eligible), "family": best_family, "folds": {}}
    if eligible:
        for name, day in [("DM23", 3), ("DM24", 4)]:
            b = D0 + day * DAY
            final["folds"][name] = final_fold(data, b, bounds_for(day, half=True), eval_frame, best_family)
            print("final", best_family, name, final["folds"][name]["reweight"], flush=True)
        final["DM_mean"] = float((final["folds"]["DM23"]["reweight"]["joint_auc_supported"] + final["folds"]["DM24"]["reweight"]["joint_auc_supported"]) / 2.0)
        final["beats_production_gate"] = bool(final["DM_mean"] - production["DM_mean"] >= 0.003)
    else:
        final["DM_mean"] = None
        final["beats_production_gate"] = False
    # Remove non-serializable training objects while retaining metrics.
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k,v in x.items() if k not in {"_objects","model","train","val","pred","prediction","choice","columns"}}
        if isinstance(x, list): return [clean(v) for v in x]
        if isinstance(x, (np.floating,np.integer)): return x.item()
        return x
    out={"experiment_baseline_DM_mean":EXPERIMENT_BASELINE,"experiment":clean(experiment),"c2_audit":clean(audit),"production_full":clean(production),"final":clean(final)}
    (OUTPUT_DIR/"phase_c_revision_results.json").write_text(json.dumps(out,indent=2,default=float))
    print(json.dumps(out,indent=2,default=float))


if __name__ == "__main__":
    main()
