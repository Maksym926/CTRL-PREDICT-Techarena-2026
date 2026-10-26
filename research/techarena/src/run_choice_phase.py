"""Evaluate the frozen-SVD no-click choice model and its LightGBM blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .features import build_features, build_stats
from .run_phase1 import first_events, subset
from .svd_model import fit_svd
from .svd_model import fit_impression_svd
from .train import fit_binary


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    train_parts = []
    for boundary in [D0 + i * DAY for i in range(1, 6)]:
        target = subset(data.train, first_events(data.train, boundary, CUT))
        stats = build_stats(data, boundary)
        svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=11)
        svd32 = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=32)
        impression_svd = fit_impression_svd(data.train, boundary, data.n_users, data.n_items, rank=10)
        train_parts.append(build_features(data, target, boundary, svd, stats, svd_alt=svd32, impression_svd=impression_svd))
        print(f"window {boundary}", flush=True)
    train_frame = __import__("pandas").concat(train_parts, ignore_index=True)
    val_boundary = D0 + 6 * DAY
    val_target = subset(data.train, first_events(data.train, val_boundary, CUT))
    val_stats = build_stats(data, val_boundary)
    val_svd = fit_svd(val_stats["click_log"], data.n_users, data.n_items, rank=11)
    val_svd32 = fit_svd(val_stats["click_log"], data.n_users, data.n_items, rank=32)
    val_impression_svd = fit_impression_svd(data.train, val_boundary, data.n_users, data.n_items, rank=10)
    val_frame = build_features(data, val_target, val_boundary, val_svd, val_stats, svd_alt=val_svd32, impression_svd=val_impression_svd)

    choice = fit_choice(train_frame, maxiter=120)
    utility, choice_prob, no_click = choice.predict(val_frame)
    labels = val_frame["label"].to_numpy()
    result = {
        "choice_utility_auc": float(roc_auc_score(labels, utility)),
        "choice_probability_auc": float(roc_auc_score(labels, choice_prob)),
        "no_click_mean": float(no_click.mean()),
    }
    model, columns, gbdt_pred, gbdt_auc, importance = fit_binary(train_frame, val_frame)
    result["gbdt_auc"] = gbdt_auc
    train_utility, train_choice_prob, train_no_click = choice.predict(train_frame)
    train_aug = train_frame.copy()
    val_aug = val_frame.copy()
    train_aug["choice_utility"] = train_utility
    train_aug["choice_probability"] = train_choice_prob
    train_aug["choice_no_click"] = np.repeat(train_no_click, 10)
    val_aug["choice_utility"] = utility
    val_aug["choice_probability"] = choice_prob
    val_aug["choice_no_click"] = np.repeat(no_click, 10)
    _, _, aug_pred, aug_auc, _ = fit_binary(train_aug, val_aug)
    result["gbdt_with_choice_features_auc"] = aug_auc
    for weight in [0.05, 0.1, 0.2, 0.3, 0.5]:
        blend = (1 - weight) * gbdt_pred + weight * choice_prob
        result[f"blend_{weight}"] = float(roc_auc_score(labels, blend))
    path = OUTPUT_DIR / "phase3_choice_validation.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
