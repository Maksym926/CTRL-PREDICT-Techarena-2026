"""Fit the Phase-1 model on all training windows and write a checked CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .build_cache import build_cache
from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, EVAL_PATH, EXAMPLE_PATH, OUTPUT_DIR, TRAIN_PATH
from .dataset import join_global
from .features import build_features, build_stats
from .run_phase1 import ensure_caches, first_events, subset
from .svd_model import fit_svd
from .svd_model import fit_impression_svd
from .submit import write_submission
from .train import feature_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--alt-rank", type=int, default=32)
    parser.add_argument("--trees", type=int, default=0, help="override tree count; 0 uses phase1 best_iteration x 1.15")
    args = parser.parse_args()
    train_cache, eval_cache = ensure_caches()
    data = join_global(load_cache(train_cache), load_cache(eval_cache))
    train_parts = []
    for boundary in [D0 + i * DAY for i in range(1, 6)]:
        indices = first_events(data.train, boundary, CUT)
        target = subset(data.train, indices)
        stats = build_stats(data, boundary)
        svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=args.rank)
        svd_alt = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=args.alt_rank)
        impression_svd = fit_impression_svd(data.train, boundary, data.n_users, data.n_items, rank=10)
        train_parts.append(build_features(data, target, boundary, svd, stats, svd_alt=svd_alt, impression_svd=impression_svd))
        print(f"training window {boundary}: {len(indices):,} events", flush=True)
    train_frame = pd.concat(train_parts, ignore_index=True)
    phase_report = OUTPUT_DIR / "phase1_validation.json"
    best_iteration = 300
    if phase_report.exists():
        best_iteration = int(json.loads(phase_report.read_text()).get("best_iteration", best_iteration))
    n_trees = int(args.trees or max(50, round(best_iteration * 1.15)))
    columns = feature_columns(train_frame)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=n_trees, learning_rate=0.05,
        num_leaves=63, min_child_samples=300, colsample_bytree=0.7,
        subsample=0.8, subsample_freq=1, reg_lambda=10.0,
        random_state=20261024, n_jobs=4, verbosity=-1,
    )
    model.fit(train_frame[columns], train_frame["label"].astype(int))

    eval_stats = build_stats(data, CUT)
    eval_svd = fit_svd(eval_stats["click_log"], data.n_users, data.n_items, rank=args.rank)
    eval_svd_alt = fit_svd(eval_stats["click_log"], data.n_users, data.n_items, rank=args.alt_rank)
    eval_impression_svd = fit_impression_svd(data.train, CUT, data.n_users, data.n_items, rank=10)
    eval_frame = build_features(data, data.evaluation, CUT, eval_svd, eval_stats, svd_alt=eval_svd_alt, impression_svd=eval_impression_svd)
    scores = model.predict_proba(eval_frame[columns])[:, 1]
    scores = np.clip(scores.astype(float), 1e-7, 1 - 1e-7)
    rowids = data.evaluation.rowids.reshape(-1)
    evaluation = pd.DataFrame({"rowid": rowids})
    example = pd.read_csv(EXAMPLE_PATH)
    output = OUTPUT_DIR / "submission_phase1.csv"
    write_submission(rowids, scores, evaluation, output, example)
    report = {
        "output": str(output), "rows": int(len(scores)), "trees": n_trees,
        "mean_prediction": float(scores.mean()), "min_prediction": float(scores.min()), "max_prediction": float(scores.max()),
        "history_rows": int((eval_frame["history_count"].to_numpy() > 0).sum()),
        "history_mean_prediction": float(scores[eval_frame["history_count"].to_numpy() > 0].mean()),
        "other_mean_prediction": float(scores[eval_frame["history_count"].to_numpy() == 0].mean()),
    }
    (OUTPUT_DIR / "phase1_submission.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
