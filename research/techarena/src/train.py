"""Leakage-safe LightGBM training utilities."""

from __future__ import annotations

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DROP_COLUMNS = {"label", "event_index", "candidate_index", "user_id", "timestamp", "window_id", "sampling_weight"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in DROP_COLUMNS]


def fit_binary(train: pd.DataFrame, validation: pd.DataFrame, seed: int = 20261024, train_weight=None, validation_weight=None):
    columns = feature_columns(train)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=300,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        train[columns],
        train["label"].astype(int),
        sample_weight=train_weight,
        eval_set=[(validation[columns], validation["label"].astype(int))],
        eval_sample_weight=[validation_weight] if validation_weight is not None else None,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = model.predict_proba(validation[columns], num_iteration=model.best_iteration_)[:, 1]
    score = float(roc_auc_score(validation["label"], pred))
    importance = sorted(
        ((name, float(value)) for name, value in zip(columns, model.feature_importances_)),
        key=lambda x: x[1], reverse=True,
    )
    return model, columns, pred, score, importance


def bucket_report(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    work = frame[["history_length", "gap_hours", "label"]].copy()
    work["prediction"] = prediction
    reports = {}
    history_bucket = pd.cut(work["history_length"], [-1, 0, 9, 19, 20], labels=["0", "1-9", "10-19", "20"])
    gap_bucket = pd.cut(work["gap_hours"], [-np.inf, 12, 24, 48, 96, np.inf], labels=["<=12h", "12-24h", "24-48h", "48-96h", ">96h"])
    for name, groups in (("history_length", history_bucket), ("gap_hours", gap_bucket)):
        reports[name] = {}
        for key in groups.cat.categories:
            mask = groups == key
            if mask.sum() and work.loc[mask, "label"].nunique() > 1:
                reports[name][str(key)] = {"rows": int(mask.sum()), "auc": float(roc_auc_score(work.loc[mask, "label"], work.loc[mask, "prediction"]))}
            else:
                reports[name][str(key)] = {"rows": int(mask.sum()), "auc": None}
    return reports


def save_report(path: str | Path, score: float, model, importance, buckets: dict, extra: dict | None = None) -> Path:
    result = {"auc": score, "best_iteration": int(model.best_iteration_), "top_features": importance[:20], "buckets": buckets}
    if extra:
        result.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path
