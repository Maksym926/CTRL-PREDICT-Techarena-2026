"""Phase-A2 fixed-window features.

This module wraps the original feature builder so the old baseline remains
reproducible. Counts and rates exposed to the model are recomputed from the
three days before B, with one-day trend and scale-free share/staleness fields.
"""

from __future__ import annotations

from collections import defaultdict, deque
import numpy as np

from .config import DAY
from .features import build_features as _build_features
from .features import build_stats as _build_stats


def build_stats(data, boundary: int, window_days: int = 3):
    # Keep the all-history click log and transitions from the established path.
    stats = _build_stats(data, boundary)
    lo = int(boundary) - window_days * DAY
    lo1 = int(boundary) - DAY
    mask = (data.train.timestamp < int(boundary)) & (data.train.timestamp >= lo)
    mask1 = (data.train.timestamp < int(boundary)) & (data.train.timestamp >= lo1)
    item_imp = defaultdict(int); item_click = defaultdict(int)
    item_imp1 = defaultdict(int); item_click1 = defaultdict(int)
    user_imp = defaultdict(int); user_click = defaultdict(int)
    user_recent = defaultdict(lambda: deque(maxlen=50))
    for e in np.flatnonzero(mask):
        u = int(data.train.user_id[e])
        for j, item in enumerate(data.train.candidates[e]):
            item = int(item); y = int(data.train.labels[e, j] == 1)
            item_imp[item] += 1; item_click[item] += y
            user_imp[u] += 1; user_click[u] += y
            user_recent[u].append(y)
            if mask1[e]:
                item_imp1[item] += 1; item_click1[item] += y
    total_imp = sum(item_imp.values()); total_click = sum(item_click.values())
    days = float(window_days)
    stats["item_impressions"] = {k: v / days for k, v in item_imp.items()}
    stats["item_clicks"] = {k: v / days for k, v in item_click.items()}
    stats["user_impressions"] = {k: v / days for k, v in user_imp.items()}
    stats["user_clicks"] = {k: v / days for k, v in user_click.items()}
    stats["global_ctr"] = float(total_click / max(total_imp, 1))
    stats["item_click_1d"] = {k: v for k, v in item_click1.items()}
    stats["item_click_3d"] = {k: v / days for k, v in item_click.items()}
    stats["item_click_7d"] = {k: v / days for k, v in item_click.items()}
    stats["user_ctr50"] = {u: float(sum(v) / max(len(v), 1)) for u, v in user_recent.items()}
    stats["windowed"] = True
    stats["window_days"] = window_days
    stats["item_impressions_1d"] = dict(item_imp1)
    stats["item_click_share"] = {k: float(v / max(total_click, 1)) for k, v in item_click.items()}
    values = np.zeros(data.n_items, dtype=np.float64)
    for k, v in stats["item_click_share"].items():
        values[int(k)] = v
    order = np.argsort(values, kind="stable")
    pct = np.empty(data.n_items, dtype=np.float64)
    pct[order] = np.arange(data.n_items, dtype=np.float64) / max(data.n_items - 1, 1)
    stats["item_click_share_pct"] = {int(k): float(pct[int(k)]) for k in item_click}
    stats["item_last_click"] = {}
    for _, item, ts in stats["click_log"]:
        stats["item_last_click"][int(item)] = max(stats["item_last_click"].get(int(item), -1), int(ts))
    return stats


def build_features(data, target, boundary: int, svd, stats, svd_alt=None, impression_svd=None):
    frame = _build_features(data, target, boundary, svd, stats, svd_alt=svd_alt, impression_svd=impression_svd)
    n = len(frame)
    share = np.zeros(n, dtype=np.float32)
    share_pct = np.zeros(n, dtype=np.float32)
    user_imp_rate = np.zeros(n, dtype=np.float32)
    user_click_rate = np.zeros(n, dtype=np.float32)
    user_ctr50 = np.zeros(n, dtype=np.float32)
    stale = np.zeros(n, dtype=np.float32)
    gap_trend = np.zeros(n, dtype=np.float32)
    for e in range(target.n_events):
        u = int(target.user_id[e])
        gap = (int(target.timestamp[e]) - int(boundary)) / 3600.0
        for j, item0 in enumerate(target.candidates[e]):
            item = int(item0); r = e * 10 + j
            share[r] = stats["item_click_share"].get(item, 0.0)
            share_pct[r] = stats["item_click_share_pct"].get(item, 0.0)
            user_imp_rate[r] = stats["user_impressions"].get(u, 0.0)
            user_click_rate[r] = stats["user_clicks"].get(u, 0.0)
            user_ctr50[r] = stats["user_ctr50"].get(u, 0.0)
            last = stats["item_last_click"].get(item, int(boundary))
            stale[r] = min(max((int(boundary) - last) / DAY, 0.0), 1e6)
            trend = float(frame.iloc[r]["item_trend"])
            gap_trend[r] = gap * trend
    frame["item_click_share"] = share
    frame["item_click_share_pct"] = share_pct
    frame["user_impressions_rate"] = user_imp_rate
    frame["user_clicks_rate"] = user_click_rate
    frame["user_ctr50"] = user_ctr50
    frame["item_staleness_days"] = stale
    frame["gap_item_trend"] = gap_trend
    # The impression SVD dot is explicitly cosine in A2.
    if "impression_svd_cos" in frame:
        frame["impression_svd_dot"] = frame["impression_svd_cos"]
    return frame
