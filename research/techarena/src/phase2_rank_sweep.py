"""Cheap Phase-2 sweep: compare SVD ranks on clean first-event validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .cache import load_cache
from .dataset import join_global
from .features import build_stats, _history_features
from .run_phase1 import first_events, subset
from .svd_model import fit_svd


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    boundary = D0 + 6 * DAY
    target = subset(data.train, first_events(data.train, boundary, CUT))
    labels = target.labels.reshape(-1)
    result = {}
    stats = build_stats(data, boundary)
    for rank in range(9, 14):
        svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=rank)
        scores = {"svd_dot": [], "svd_cos": [], "history_mean_cos": [], "history_max_cos": [], "history_last_cos": [], "history_weighted_cos": []}
        for e in range(target.n_events):
            c = target.candidates[e]
            values = _history_features(target, e, c, svd)
            for key, value in zip(scores, values[3:]):
                scores[key].extend(np.asarray(value, dtype=float).tolist())
        result[str(rank)] = {
            key: float(roc_auc_score(labels, value)) for key, value in scores.items()
        }
        print(rank, result[str(rank)], flush=True)
    path = OUTPUT_DIR / "phase2_rank_sweep.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)


if __name__ == "__main__":
    main()
