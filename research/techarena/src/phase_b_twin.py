"""Phase B: structure-preserving synthetic twin and estimator ceiling."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import CachedEvents, load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import GlobalDataset, join_global
from .features import build_features, build_stats
from .run_phase1 import first_events, subset
from .svd_model import fit_svd
from .train import fit_binary


def _utility_arrays(events: CachedEvents, svd, item_clicks: dict[int, int], repeat_boost: float, no_click_bias: float, scale: float, pop_weight: float):
    user_vec = svd.user_vectors[events.user_id]
    item_vec = svd.item_vectors[events.candidates]
    utilities = scale * np.einsum("erd,ed->er", item_vec, user_vec)
    click_lookup = np.zeros(len(svd.item_vectors), dtype=np.float64)
    if item_clicks:
        keys = np.fromiter(item_clicks.keys(), dtype=np.int64)
        keys = keys[(keys >= 0) & (keys < len(click_lookup))]
        click_lookup[keys] = [item_clicks[int(x)] for x in keys]
    utilities += pop_weight * np.log1p(click_lookup[events.candidates])
    repeat = np.any(events.history_items[:, :, None] == events.candidates[:, None, :], axis=1)
    utilities += repeat_boost * repeat
    logits = np.concatenate([np.full((events.n_events, 1), no_click_bias), utilities], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return utilities, repeat, probs


def _calibrate_no_click(events, svd, item_clicks, repeat_boost, scale, pop_weight, target_no_click=0.37):
    lo, hi = -10.0, 10.0
    for _ in range(24):
        mid = (lo + hi) / 2
        _, _, probs = _utility_arrays(events, svd, item_clicks, repeat_boost, mid, scale, pop_weight)
        if probs[:, 0].mean() > target_no_click:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def fit_generator(data: GlobalDataset, boundary: int):
    stats = build_stats(data, boundary)
    true_svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=11)
    # Scale is tuned around the observed SVD norm; popularity is deliberately
    # small because the taste vectors carry the latent signal.
    target = data.train
    calibration_idx = np.arange(0, target.n_events, max(1, target.n_events // 10_000))
    calibration_events = CachedEvents(
        candidates=target.candidates[calibration_idx], labels=target.labels[calibration_idx], user_id=target.user_id[calibration_idx],
        timestamp=target.timestamp[calibration_idx], history_items=target.history_items[calibration_idx], history_times=target.history_times[calibration_idx],
        history_length=target.history_length[calibration_idx], rowids=None, item_keys=target.item_keys, user_keys=target.user_keys,
    )
    best = None
    for repeat_boost in np.linspace(0.0, 4.0, 5):
        no_click_bias = _calibrate_no_click(calibration_events, true_svd, stats["item_clicks"], float(repeat_boost), 4.0, 0.08)
        util, repeat, probs = _utility_arrays(calibration_events, true_svd, stats["item_clicks"], float(repeat_boost), no_click_bias, 4.0, 0.08)
        row_prob = probs[:, 1:].reshape(-1)
        repeat_rows = repeat.reshape(-1)
        ctr_repeat = float(row_prob[repeat_rows].mean()) if repeat_rows.any() else 0.0
        ctr_other = float(row_prob[~repeat_rows].mean()) if (~repeat_rows).any() else 0.0
        error = abs(ctr_repeat - 0.293) + abs(ctr_other - 0.0568)
        candidate = (error, float(repeat_boost), float(no_click_bias), ctr_repeat, ctr_other, float(probs[:, 0].mean()))
        if best is None or candidate < best:
            best = candidate
    _, repeat_boost, no_click_bias, ctr_repeat, ctr_other, no_click_rate = best
    return true_svd, stats, {"repeat_boost": repeat_boost, "no_click_bias": no_click_bias, "scale": 4.0, "pop_weight": 0.08, "ctr_repeat": ctr_repeat, "ctr_other": ctr_other, "no_click_rate": no_click_rate}


def simulate_labels(data: GlobalDataset, true_svd, stats, params, seed=20261024):
    rng = np.random.default_rng(seed)
    train = data.train
    histories = {}
    history_times = {}
    sim_h = np.full_like(train.history_items, -1)
    sim_ht = np.full_like(train.history_times, -1)
    sim_labels = np.full_like(train.labels, -1)
    order = np.argsort(train.timestamp, kind="stable")
    for e in order:
        user = int(train.user_id[e])
        if user not in histories:
            valid = train.history_items[e] >= 0
            histories[user] = list(train.history_items[e, valid].astype(int))[-20:]
            history_times[user] = list(train.history_times[e, valid].astype(int))[-20:]
        h = histories[user][-20:]
        ht = history_times[user][-20:]
        if h:
            sim_h[e, -len(h):] = h
            sim_ht[e, -len(ht):] = ht
        user_vec = true_svd.user_vector(user, np.asarray(h, dtype=np.int64))
        candidates = train.candidates[e]
        repeat = np.any(np.asarray(h)[:, None] == candidates[None, :], axis=0) if h else np.zeros(10, dtype=bool)
        utility = params["scale"] * (true_svd.item_vectors[candidates] @ user_vec)
        utility += params["pop_weight"] * np.log1p(np.asarray([stats["item_clicks"].get(int(x), 0) for x in candidates]))
        utility += params["repeat_boost"] * repeat
        logits = np.r_[params["no_click_bias"], utility]
        logits -= logits.max()
        prob = np.exp(logits)
        prob /= prob.sum()
        selected = int(rng.choice(11, p=prob))
        sim_labels[e] = 0
        if selected > 0:
            sim_labels[e, selected - 1] = 1
            histories[user].append(int(candidates[selected - 1]))
            history_times[user].append(int(train.timestamp[e]))
    simulated_train = CachedEvents(
        candidates=train.candidates.copy(), labels=sim_labels, user_id=train.user_id.copy(), timestamp=train.timestamp.copy(),
        history_items=sim_h, history_times=sim_ht, history_length=(sim_h >= 0).sum(axis=1).astype(np.int16),
        rowids=train.rowids.copy() if train.rowids is not None else None, item_keys=train.item_keys, user_keys=train.user_keys,
    )
    return GlobalDataset(train=simulated_train, evaluation=data.evaluation, n_users=data.n_users, n_items=data.n_items)


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    true_svd, real_stats, params = fit_generator(data, CUT)
    twin = simulate_labels(data, true_svd, real_stats, params)
    boundary = D0 + 6 * DAY
    val_target = subset(twin.train, first_events(twin.train, boundary, CUT))
    # Oracle uses the generator's true utility on the held-out candidates.
    oracle_util, _, oracle_probs = _utility_arrays(val_target, true_svd, real_stats["item_clicks"], params["repeat_boost"], params["no_click_bias"], params["scale"], params["pop_weight"])
    oracle_utility_auc = float(roc_auc_score(val_target.labels.reshape(-1), oracle_util.reshape(-1)))
    oracle_auc = float(roc_auc_score(val_target.labels.reshape(-1), oracle_probs[:, 1:].reshape(-1)))
    val_stats = build_stats(twin, boundary)
    estimated_svd = fit_svd(val_stats["click_log"], twin.n_users, twin.n_items, rank=11)
    val_features = build_features(twin, val_target, boundary, estimated_svd, val_stats)
    svd_auc = float(roc_auc_score(val_features["label"], val_features["svd_cos"]))
    # Full pipeline uses the same five-window construction as the submitted path,
    # with the choice features included after the twin estimator is fitted.
    train_parts = []
    for b in [D0 + i * DAY for i in range(1, 6)]:
        target = subset(twin.train, first_events(twin.train, b, CUT))
        st = build_stats(twin, b)
        s11 = fit_svd(st["click_log"], twin.n_users, twin.n_items, rank=11)
        train_parts.append(build_features(twin, target, b, s11, st))
    train_frame = pd.concat(train_parts, ignore_index=True)
    choice = fit_choice(train_frame, maxiter=80)
    train_aug = train_frame.copy()
    u, p, n = choice.predict(train_frame)
    train_aug["choice_utility"] = u; train_aug["choice_probability"] = p; train_aug["choice_no_click"] = np.repeat(n, 10)
    val_aug = val_features.copy()
    u, p, n = choice.predict(val_features)
    val_aug["choice_utility"] = u; val_aug["choice_probability"] = p; val_aug["choice_no_click"] = np.repeat(n, 10)
    _, _, pred, full_auc, _ = fit_binary(train_aug, val_aug)
    light_users = val_aug["history_length"].to_numpy() <= 5
    light_auc = float(roc_auc_score(val_aug.loc[light_users, "label"], pred[light_users])) if light_users.sum() else None
    result = {
        "generator_parameters": params,
        "simulated_no_click_rate": float((twin.train.labels.sum(axis=1) == 0).mean()),
        "simulated_history_mean": float(twin.train.history_length.mean()),
        "oracle_auc": oracle_auc,
        "oracle_utility_auc": oracle_utility_auc,
        "svd_feature_auc": svd_auc,
        "full_pipeline_auc": float(full_auc),
        "full_pipeline_light_user_auc": light_auc,
        "oracle_minus_svd": oracle_auc - svd_auc,
        "oracle_minus_pipeline": oracle_auc - full_auc,
    }
    (OUTPUT_DIR / "phase_b_twin_results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
