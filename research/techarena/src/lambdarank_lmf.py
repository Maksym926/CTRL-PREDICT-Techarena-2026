"""DM24 LambdaRank probe on the validated SAFE_LOGMF10 feature path."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .mf_variant_experiment import add_base, sample_events
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, train_frames


DROP = {"label", "event_index", "candidate_index", "user_id", "timestamp", "window_id", "sampling_weight"}


def feature_cols(df):
    return [c for c in df.columns if c not in DROP]


def with_lmf(data, train, val, bounds, boundary):
    parts = []
    offset = 0
    for wi, b in enumerate(bounds):
        n = int((train["window_id"].to_numpy() == wi).sum())
        p = OUTPUT_DIR / f"logmf_cosine_prod{wi}_{int(b)}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(add_base(train.iloc[offset:offset+n].copy(), np.load(p)))
        offset += n
    train = pd.concat(parts, ignore_index=True)
    val = add_base(val, np.load(OUTPUT_DIR / f"logmf_cosine_val_dm24_{int(boundary)}.npy"))
    return train, val


def groups(df):
    # Each cached feature frame writes ten consecutive rows per event.
    if len(df) % 10:
        raise ValueError(f"rows are not complete slates: {len(df)}")
    return np.full(len(df) // 10, 10, dtype=np.int32)


def clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def main():
    t0 = time.time()
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    boundary = D0 + 4 * DAY
    bounds = bounds_for(4, half=True)
    train, val, _ = train_frames(data, boundary, bounds, "a3_c1_c2", choice_maxiter=20)
    train, val = with_lmf(data, train, val, bounds, boundary)
    evalf = cached_frame(data, CUT, "eval")
    # Keep complete events and use the established 35% all-negative event draw.
    sample = sample_events(train)
    cols = feature_cols(train)
    X = clean(sample, cols); y = sample["label"].astype(np.int8).to_numpy()
    Xv = clean(val, cols); yv = val["label"].astype(np.int8).to_numpy()
    w = importance_weights(sample, evalf)
    params = dict(objective="lambdarank", n_estimators=400, learning_rate=.05,
                  num_leaves=154, min_child_samples=201, max_bin=63,
                  colsample_bytree=.6728134504603546, subsample=.9165758721143165,
                  subsample_freq=1, reg_lambda=43.70944332031455,
                  reg_alpha=6.406134606768282, random_state=20261025,
                  n_jobs=8, verbosity=-1, label_gain=[0, 1],
                  lambdarank_truncation_level=10, lambdarank_norm=True)
    model = lgb.LGBMRanker(**params)
    model.fit(X, y, group=groups(sample), sample_weight=w,
              eval_set=[(Xv, yv)], eval_group=[groups(val)],
              eval_at=[1, 3, 5, 10], callbacks=[lgb.early_stopping(60, verbose=False)])
    rank_score = model.predict(Xv, num_iteration=model.best_iteration_)
    dm_rank = weighted_auc(val, rank_score, evalf)
    # The pointwise SAFE_LOGMF10 model is stored from the earlier DM24 run.
    point_path = OUTPUT_DIR / "mf_variant_base_lmf10_DM24.npy"
    if point_path.exists():
        point_score = np.load(point_path)
    else:
        point_score = np.full(len(val), np.nan, dtype=np.float32)
    result = {
        "boundary": int(boundary), "train_rows": int(len(sample)), "train_events": int(len(sample)//10),
        "val_rows": int(len(val)), "best_iteration": int(model.best_iteration_),
        "rank_dm24": float(dm_rank["joint_auc_supported"]),
        "rank_raw_auc": float(roc_auc_score(yv, rank_score)),
        "rank_history_only": float(dm_rank["history_only_auc"]),
        "pointwise_reference_dm24": 0.7989857073691065,
        "rank_gain": float(dm_rank["joint_auc_supported"] - 0.7989857073691065),
        "params": params,
        "elapsed_sec": time.time() - t0,
    }
    if np.isfinite(point_score).all() and len(point_score) == len(rank_score):
        # Within-event rank score is intentionally centered; pointwise score
        # retains global event calibration.
        rr = rank_score.reshape(-1, 10)
        pp = point_score.reshape(-1, 10)
        centered = rr - rr.mean(axis=1, keepdims=True)
        for alpha in (0.05, 0.10, 0.20, 0.30, 0.50):
            blend = (pp + alpha * centered).reshape(-1)
            dm = weighted_auc(val, blend, evalf)["joint_auc_supported"]
            result[f"blend_alpha_{alpha:.2f}"] = float(dm)
    np.save(OUTPUT_DIR / "lambdarank_lmf_dm24.npy", rank_score)
    (OUTPUT_DIR / "lambdarank_lmf_dm24.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
