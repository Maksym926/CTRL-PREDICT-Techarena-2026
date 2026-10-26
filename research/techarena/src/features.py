"""Boundary-safe feature construction for candidate rows."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from .cache import CachedEvents
from .config import DAY
from .dataset import GlobalDataset, click_log_before
from .svd_model import SVDModel


def _counter(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for value in values:
        out[int(value)] = out.get(int(value), 0) + 1
    return out


def _prior_ctr(clicks: int, impressions: int, global_ctr: float) -> float:
    return float((clicks + 20.0 * global_ctr) / (impressions + 20.0))


def build_stats(data: GlobalDataset, boundary: int) -> dict:
    """Statistics whose source impressions and labels are strictly before B."""
    imp_item: dict[int, int] = defaultdict(int)
    click_item: dict[int, int] = defaultdict(int)
    imp_user: dict[int, int] = defaultdict(int)
    click_user: dict[int, int] = defaultdict(int)
    shown_pairs: set[tuple[int, int]] = set()
    clicked_pairs: set[tuple[int, int]] = set()
    event_mask = data.train.timestamp < int(boundary)
    total_imp = 0
    total_click = 0
    for e in np.flatnonzero(event_mask):
        user = int(data.train.user_id[e])
        imp_user[user] += 10
        total_imp += 10
        for j, item in enumerate(data.train.candidates[e]):
            item = int(item)
            imp_item[item] += 1
            shown_pairs.add((user, item))
            if int(data.train.labels[e, j]) == 1:
                click_item[item] += 1
                clicked_pairs.add((user, item))
                click_user[user] += 1
                total_click += 1
    click_log = click_log_before(data, int(boundary))
    item_click_1d: dict[int, int] = defaultdict(int)
    item_click_3d: dict[int, int] = defaultdict(int)
    item_click_7d: dict[int, int] = defaultdict(int)
    if len(click_log):
        for user, item, ts in click_log:
            age = int(boundary) - int(ts)
            if age < DAY:
                item_click_1d[int(item)] += 1
            if age < 3 * DAY:
                item_click_3d[int(item)] += 1
            if age < 7 * DAY:
                item_click_7d[int(item)] += 1
    transition_count: dict[tuple[int, int], int] = defaultdict(int)
    transition_out: dict[int, int] = defaultdict(int)
    if len(click_log):
        order = np.lexsort((click_log[:, 2], click_log[:, 0]))
        ordered = click_log[order]
        for start in range(len(ordered)):
            user = int(ordered[start, 0])
            source = int(ordered[start, 1])
            for lag in (1, 2, 3):
                end = start + lag
                if end >= len(ordered) or int(ordered[end, 0]) != user:
                    break
                destination = int(ordered[end, 1])
                weight = 1.0 / lag
                transition_count[(source, destination)] += weight
                transition_out[source] += weight
    return {
        "boundary": int(boundary),
        "item_impressions": dict(imp_item),
        "item_clicks": dict(click_item),
        "user_impressions": dict(imp_user),
        "user_clicks": dict(click_user),
        "global_ctr": float(total_click / total_imp) if total_imp else 0.0,
        "click_log": click_log,
        "shown_pairs": shown_pairs,
        "clicked_pairs": clicked_pairs,
        "item_click_1d": dict(item_click_1d),
        "item_click_3d": dict(item_click_3d),
        "item_click_7d": dict(item_click_7d),
        "transition_count": dict(transition_count),
        "transition_out": dict(transition_out),
    }


def _history_features(event: CachedEvents, e: int, candidates: np.ndarray, svd: SVDModel):
    history = event.history_items[e]
    times = event.history_times[e]
    valid = history >= 0
    h = history[valid]
    ht = times[valid]
    n = len(candidates)
    count = np.zeros(n, dtype=np.float32)
    last_pos = np.full(n, 99, dtype=np.float32)
    age = np.full(n, 1e12, dtype=np.float32)
    if len(h):
        for j, item in enumerate(candidates):
            where = np.flatnonzero(h == item)
            if len(where):
                count[j] = len(where)
                last_pos[j] = len(h) - 1 - where[-1]
                age[j] = max(0, int(event.timestamp[e]) - int(ht[where[-1]]))
    user_vec = svd.user_vector(int(event.user_id[e]), h)
    item_vec = svd.item_vectors[candidates]
    norm_u = np.linalg.norm(user_vec) + 1e-8
    norm_i = np.linalg.norm(item_vec, axis=1) + 1e-8
    svd_dot = item_vec @ user_vec
    svd_cos = svd_dot / (norm_i * norm_u)
    # The leading normalized-SVD axis is dominated by global popularity.
    # Removing it exposes the stable taste subspace without using labels at B.
    user_res = user_vec.copy()
    item_res = item_vec.copy()
    if user_res.shape[0] > 1:
        user_res[0] = 0.0
        item_res[:, 0] = 0.0
    norm_ur = np.linalg.norm(user_res) + 1e-8
    norm_ir = np.linalg.norm(item_res, axis=1) + 1e-8
    svd_resid_dot = item_res @ user_res
    svd_resid_cos = svd_resid_dot / (norm_ir * norm_ur)
    if len(h):
        hvec = svd.item_vectors[h]
        hnorm = np.linalg.norm(hvec, axis=1) + 1e-8
        pair = item_vec @ hvec.T / (norm_i[:, None] * hnorm[None, :])
        hres = hvec.copy()
        hres[:, 0] = 0.0
        hrnorm = np.linalg.norm(hres, axis=1) + 1e-8
        pair_res = item_res @ hres.T / (norm_ir[:, None] * hrnorm[None, :])
        mean_cos = pair.mean(axis=1)
        max_cos = pair.max(axis=1)
        last_cos = pair[:, -1]
        weights = np.power(0.8, np.arange(len(h) - 1, -1, -1, dtype=np.float32))
        weighted_cos = (pair @ weights) / weights.sum()
        resid_mean_cos = pair_res.mean(axis=1)
        resid_max_cos = pair_res.max(axis=1)
        resid_last_cos = pair_res[:, -1]
        resid_weighted_cos = (pair_res @ weights) / weights.sum()
    else:
        mean_cos = max_cos = last_cos = weighted_cos = np.zeros(n, dtype=np.float32)
        resid_mean_cos = resid_max_cos = resid_last_cos = resid_weighted_cos = np.zeros(n, dtype=np.float32)
    return (
        count, last_pos, age, svd_dot, svd_cos, mean_cos, max_cos, last_cos, weighted_cos,
        svd_resid_dot, svd_resid_cos, resid_mean_cos, resid_max_cos, resid_last_cos, resid_weighted_cos,
    )


def build_features(data: GlobalDataset, target: CachedEvents, boundary: int, svd: SVDModel, stats: dict, svd_alt: SVDModel | None = None, impression_svd: SVDModel | None = None) -> pd.DataFrame:
    rows = []
    global_ctr = stats["global_ctr"]
    for e in range(target.n_events):
        candidates = target.candidates[e]
        (
            hist_count, last_pos, age, svd_dot, svd_cos,
            mean_cos, max_cos, last_cos, weighted_cos, svd_resid_dot, svd_resid_cos,
            resid_mean_cos, resid_max_cos, resid_last_cos, resid_weighted_cos,
        ) = _history_features(target, e, candidates, svd)
        if svd_alt is not None:
            alt_values = _history_features(target, e, candidates, svd_alt)
            _, _, _, alt_dot, alt_cos, alt_mean, alt_max, alt_last, alt_weighted, alt_resid_dot, alt_resid_cos, alt_resid_mean, alt_resid_max, alt_resid_last, alt_resid_weighted = alt_values
        else:
            alt_dot = alt_cos = alt_mean = alt_max = alt_last = alt_weighted = np.zeros(10, dtype=np.float32)
            alt_resid_dot = alt_resid_cos = alt_resid_mean = alt_resid_max = alt_resid_last = alt_resid_weighted = np.zeros(10, dtype=np.float32)
        if impression_svd is not None:
            imp_user = impression_svd.user_vector(int(target.user_id[e]), target.history_items[e])
            imp_item = impression_svd.item_vectors[candidates]
            imp_dot = imp_item @ imp_user
            imp_cos = imp_dot / ((np.linalg.norm(imp_item, axis=1) + 1e-8) * (np.linalg.norm(imp_user) + 1e-8))
        else:
            imp_dot = imp_cos = np.zeros(10, dtype=np.float32)
        raw_item_ctr = np.empty(10, dtype=np.float32)
        raw_user_ctr = np.empty(10, dtype=np.float32)
        item_imp = np.empty(10, dtype=np.float32)
        item_click = np.empty(10, dtype=np.float32)
        shown_before = np.empty(10, dtype=np.float32)
        clicked_before = np.empty(10, dtype=np.float32)
        click_1d = np.empty(10, dtype=np.float32)
        click_3d = np.empty(10, dtype=np.float32)
        click_7d = np.empty(10, dtype=np.float32)
        transition_last = np.zeros(10, dtype=np.float32)
        transition_mean = np.zeros(10, dtype=np.float32)
        transition_max = np.zeros(10, dtype=np.float32)
        user = int(target.user_id[e])
        ui = stats["user_impressions"].get(user, 0)
        uc = stats["user_clicks"].get(user, 0)
        for j, item in enumerate(candidates):
            item = int(item)
            ii = stats["item_impressions"].get(item, 0)
            ic = stats["item_clicks"].get(item, 0)
            item_imp[j] = ii
            item_click[j] = ic
            raw_item_ctr[j] = _prior_ctr(ic, ii, global_ctr)
            raw_user_ctr[j] = _prior_ctr(uc, ui, global_ctr)
            shown_before[j] = float((user, item) in stats["shown_pairs"])
            clicked_before[j] = float((user, item) in stats["clicked_pairs"])
            click_1d[j] = stats["item_click_1d"].get(item, 0)
            click_3d[j] = stats["item_click_3d"].get(item, 0)
            click_7d[j] = stats["item_click_7d"].get(item, 0)
            valid_h = target.history_items[e][target.history_items[e] >= 0]
            if len(valid_h):
                transition_values = np.asarray([
                    stats["transition_count"].get((int(source), item), 0.0) /
                    max(1.0, stats["transition_out"].get(int(source), 0.0))
                    for source in valid_h
                ], dtype=np.float32)
                transition_last[j] = transition_values[-1]
                transition_mean[j] = transition_values.mean()
                transition_max[j] = transition_values.max()
        event_max = float(svd_cos.max())
        event_mean = float(svd_cos.mean())
        rank = (10 - np.argsort(np.argsort(svd_cos))).astype(np.float32)
        item_ctr_max = float(raw_item_ctr.max())
        item_ctr_mean = float(raw_item_ctr.mean())
        item_ctr_std = float(raw_item_ctr.std())
        item_ctr_rank = (10 - np.argsort(np.argsort(raw_item_ctr))).astype(np.float32)
        item_imp_max = float(item_imp.max())
        item_imp_mean = float(item_imp.mean())
        for j in range(10):
            rows.append({
                "event_index": e,
                "candidate_index": j,
                "user_id": user,
                "timestamp": int(target.timestamp[e]),
                "gap_hours": (int(target.timestamp[e]) - int(boundary)) / 3600.0,
                "history_length": int(target.history_length[e]),
                "history_count": hist_count[j],
                "history_last_pos": last_pos[j],
                "history_age_days": min(age[j] / 86400.0, 1e6),
                "svd_dot": svd_dot[j],
                "svd_cos": svd_cos[j],
                "svd_resid_dot": svd_resid_dot[j],
                "svd_resid_cos": svd_resid_cos[j],
                "history_resid_mean_cos": resid_mean_cos[j],
                "history_resid_max_cos": resid_max_cos[j],
                "history_resid_last_cos": resid_last_cos[j],
                "history_resid_weighted_cos": resid_weighted_cos[j],
                "svd32_dot": alt_dot[j],
                "svd32_cos": alt_cos[j],
                "svd32_resid_dot": alt_resid_dot[j],
                "svd32_resid_cos": alt_resid_cos[j],
                "history32_resid_mean_cos": alt_resid_mean[j],
                "history32_resid_max_cos": alt_resid_max[j],
                "history32_resid_last_cos": alt_resid_last[j],
                "history32_resid_weighted_cos": alt_resid_weighted[j],
                "history32_mean_cos": alt_mean[j],
                "history32_max_cos": alt_max[j],
                "history32_last_cos": alt_last[j],
                "history32_weighted_cos": alt_weighted[j],
                "impression_svd_dot": imp_dot[j],
                "impression_svd_cos": imp_cos[j],
                "history_mean_cos": mean_cos[j],
                "history_max_cos": max_cos[j],
                "history_last_cos": last_cos[j],
                "history_weighted_cos": weighted_cos[j],
                "item_impressions": item_imp[j],
                "item_clicks": item_click[j],
                "item_ctr": raw_item_ctr[j],
                "item_ctr_rank": item_ctr_rank[j],
                "item_ctr_gap": raw_item_ctr[j] - item_ctr_max,
                "event_item_ctr_max": item_ctr_max,
                "event_item_ctr_mean": item_ctr_mean,
                "event_item_ctr_std": item_ctr_std,
                "item_impressions_gap": item_imp[j] - item_imp_max,
                "event_item_impressions_max": item_imp_max,
                "event_item_impressions_mean": item_imp_mean,
                "user_ctr": raw_user_ctr[j],
                "shown_before": shown_before[j],
                "clicked_before": clicked_before[j],
                "shown_nonclick_before": max(0.0, shown_before[j] - clicked_before[j]),
                "item_click_1d": click_1d[j],
                "item_click_3d": click_3d[j],
                "item_click_7d": click_7d[j],
                "item_trend": click_1d[j] / max(0.1, click_7d[j] / 7.0),
                "transition_last": transition_last[j],
                "transition_mean": transition_mean[j],
                "transition_max": transition_max[j],
                "event_score_rank": rank[j],
                "event_score_gap": svd_cos[j] - event_max,
                "event_score_max": event_max,
                "event_score_mean": event_mean,
                "label": int(target.labels[e, j]),
            })
    return pd.DataFrame(rows)


def training_windows(data: GlobalDataset, boundaries: list[int], cutoff: int):
    """Yield (boundary, first-event-per-user target) windows."""
    for boundary in boundaries:
        mask = (data.train.timestamp >= boundary) & (data.train.timestamp < cutoff)
        indices = np.flatnonzero(mask)
        seen: set[int] = set()
        selected: list[int] = []
        for e in indices:
            user = int(data.train.user_id[e])
            if user not in seen:
                seen.add(user)
                selected.append(int(e))
        yield boundary, selected
