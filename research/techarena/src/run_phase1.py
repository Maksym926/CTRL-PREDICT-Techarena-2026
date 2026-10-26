"""Run the first clean SVD + LightGBM validation and create a prediction file."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .build_cache import build_cache
from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, EVAL_PATH, EXAMPLE_PATH, OUTPUT_DIR, TRAIN_PATH
from .dataset import join_global
from .features import build_features
from .features import build_stats
from .svd_model import fit_svd
from .svd_model import fit_impression_svd
from .train import fit_binary, bucket_report, save_report


def first_events(events, start: int, end: int):
    mask = (events.timestamp >= start) & (events.timestamp < end)
    seen = set()
    selected = []
    for e in np.flatnonzero(mask):
        user = int(events.user_id[e])
        if user not in seen:
            seen.add(user)
            selected.append(int(e))
    return selected


def subset(events, indices):
    idx = np.asarray(indices, dtype=np.int64)
    return type(events)(
        candidates=events.candidates[idx], labels=events.labels[idx], user_id=events.user_id[idx],
        timestamp=events.timestamp[idx], history_items=events.history_items[idx], history_times=events.history_times[idx],
        history_length=events.history_length[idx], rowids=events.rowids[idx] if events.rowids is not None else None,
        item_keys=events.item_keys, user_keys=events.user_keys,
    )


def ensure_caches():
    train_cache = CACHE_DIR / "train.npz"
    eval_cache = CACHE_DIR / "evaluation.npz"
    if not train_cache.exists():
        build_cache(TRAIN_PATH, train_cache, labelled=True)
    if not eval_cache.exists():
        build_cache(EVAL_PATH, eval_cache, labelled=False)
    return train_cache, eval_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--alt-rank", type=int, default=32)
    parser.add_argument("--quick", action="store_true", help="use one training window for a smoke run")
    args = parser.parse_args()
    train_cache, eval_cache = ensure_caches()
    data = join_global(load_cache(train_cache), load_cache(eval_cache))
    train_parts = []
    boundaries = [D0 + i * DAY for i in range(1, 6)]
    if args.quick:
        boundaries = boundaries[-1:]
    for boundary in boundaries:
        indices = first_events(data.train, boundary, CUT)
        target = subset(data.train, indices)
        stats = build_stats(data, boundary)
        svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=args.rank)
        svd_alt = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=args.alt_rank)
        impression_svd = fit_impression_svd(data.train, boundary, data.n_users, data.n_items, rank=10)
        part = build_features(data, target, boundary, svd, stats, svd_alt=svd_alt, impression_svd=impression_svd)
        train_parts.append(part)
        print(f"window {boundary}: {len(indices):,} events, {len(part):,} rows", flush=True)
    train_frame = pd.concat(train_parts, ignore_index=True)

    val_boundary = D0 + 6 * DAY
    val_indices = first_events(data.train, val_boundary, CUT)
    val_target = subset(data.train, val_indices)
    val_stats = build_stats(data, val_boundary)
    val_svd = fit_svd(val_stats["click_log"], data.n_users, data.n_items, rank=args.rank)
    val_svd_alt = fit_svd(val_stats["click_log"], data.n_users, data.n_items, rank=args.alt_rank)
    val_impression_svd = fit_impression_svd(data.train, val_boundary, data.n_users, data.n_items, rank=10)
    val_frame = build_features(data, val_target, val_boundary, val_svd, val_stats, svd_alt=val_svd_alt, impression_svd=val_impression_svd)
    model, columns, prediction, auc, importance = fit_binary(train_frame, val_frame)
    report = save_report(OUTPUT_DIR / "phase1_validation.json", auc, model, importance, bucket_report(val_frame, prediction), {
        "rank": args.rank, "train_rows": int(len(train_frame)), "validation_rows": int(len(val_frame)),
        "validation_events": int(len(val_indices)),
    })
    print(json.dumps({"auc": auc, "report": str(report), "top_features": importance[:10]}, indent=2))


if __name__ == "__main__":
    main()
