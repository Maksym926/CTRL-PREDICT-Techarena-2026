"""Build the Phase-3 choice-feature submission."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import join_global
from .features import build_features, build_stats
from .run_phase1 import first_events, subset
from .svd_model import fit_impression_svd, fit_svd
from .submit import write_submission
from .train import feature_columns, fit_binary


def add_choice(frame: pd.DataFrame, choice, train: bool = False) -> pd.DataFrame:
    utility, probability, no_click = choice.predict(frame)
    result = frame.copy()
    result["choice_utility"] = utility
    result["choice_probability"] = probability
    result["choice_no_click"] = np.repeat(no_click, 10)
    return result


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    train_parts = []
    for boundary in [D0 + i * DAY for i in range(1, 6)]:
        target = subset(data.train, first_events(data.train, boundary, CUT))
        stats = build_stats(data, boundary)
        svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=11)
        svd32 = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=32)
        imp = fit_impression_svd(data.train, boundary, data.n_users, data.n_items, rank=10)
        train_parts.append(build_features(data, target, boundary, svd, stats, svd_alt=svd32, impression_svd=imp))
        print(f"training window {boundary}", flush=True)
    train_frame = pd.concat(train_parts, ignore_index=True)

    val_boundary = D0 + 6 * DAY
    val_target = subset(data.train, first_events(data.train, val_boundary, CUT))
    val_stats = build_stats(data, val_boundary)
    val_svd = fit_svd(val_stats["click_log"], data.n_users, data.n_items, rank=11)
    val_svd32 = fit_svd(val_stats["click_log"], data.n_users, data.n_items, rank=32)
    val_imp = fit_impression_svd(data.train, val_boundary, data.n_users, data.n_items, rank=10)
    val_frame = build_features(data, val_target, val_boundary, val_svd, val_stats, svd_alt=val_svd32, impression_svd=val_imp)

    choice = fit_choice(train_frame, maxiter=120)
    train_aug = add_choice(train_frame, choice, train=True)
    val_aug = add_choice(val_frame, choice)
    model, columns, val_pred, val_auc, importance = fit_binary(train_aug, val_aug)
    n_trees = max(50, round(model.best_iteration_ * 1.15))
    final_model = lgb.LGBMClassifier(
        objective="binary", n_estimators=n_trees, learning_rate=0.05,
        num_leaves=63, min_child_samples=300, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, reg_lambda=10.0,
        random_state=20261024, n_jobs=4, verbosity=-1,
    )
    final_model.fit(train_aug[columns], train_aug["label"].astype(int))

    eval_stats = build_stats(data, CUT)
    eval_svd = fit_svd(eval_stats["click_log"], data.n_users, data.n_items, rank=11)
    eval_svd32 = fit_svd(eval_stats["click_log"], data.n_users, data.n_items, rank=32)
    eval_imp = fit_impression_svd(data.train, CUT, data.n_users, data.n_items, rank=10)
    eval_frame = build_features(data, data.evaluation, CUT, eval_svd, eval_stats, svd_alt=eval_svd32, impression_svd=eval_imp)
    eval_aug = add_choice(eval_frame, choice)
    scores = final_model.predict_proba(eval_aug[columns])[:, 1]
    scores = np.clip(scores.astype(float), 1e-7, 1 - 1e-7)
    rowids = data.evaluation.rowids.reshape(-1)
    evaluation = pd.DataFrame({"rowid": rowids})
    example = pd.read_csv(EXAMPLE_PATH)
    output = OUTPUT_DIR / "submission_choice_features.csv"
    write_submission(rowids, scores, evaluation, output, example)
    report = {
        "output": str(output), "rows": int(len(scores)), "validation_auc": float(val_auc),
        "best_iteration": int(model.best_iteration_), "final_trees": int(n_trees),
        "mean_prediction": float(scores.mean()), "min_prediction": float(scores.min()), "max_prediction": float(scores.max()),
        "history_mean_prediction": float(scores[eval_frame["history_count"].to_numpy() > 0].mean()),
        "other_mean_prediction": float(scores[eval_frame["history_count"].to_numpy() == 0].mean()),
        "top_features": importance[:20],
    }
    (OUTPUT_DIR / "choice_submission.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
