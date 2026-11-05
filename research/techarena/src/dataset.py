"""Join train/evaluation caches onto one global user/item code space."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .cache import CachedEvents


@dataclass
class GlobalDataset:
    train: CachedEvents
    evaluation: CachedEvents
    n_users: int
    n_items: int


def _recode(values: np.ndarray, lookup: dict[int, int], fill: int = -1) -> np.ndarray:
    result = np.full(values.shape, fill, dtype=np.int32)
    valid = values >= 0
    if valid.any():
        result[valid] = np.asarray([lookup[int(x)] for x in values[valid]], dtype=np.int32)
    return result


def _recode_events(events: CachedEvents, item_map: dict[int, int], user_map: dict[int, int]) -> CachedEvents:
    if events.item_keys is None or events.user_keys is None:
        raise ValueError("cache is missing local key arrays")
    candidate_keys = np.full(events.candidates.shape, -1, dtype=np.int64)
    valid_candidates = events.candidates >= 0
    candidate_keys[valid_candidates] = events.item_keys[events.candidates[valid_candidates]]
    history_keys = np.full(events.history_items.shape, -1, dtype=np.int64)
    valid_history = events.history_items >= 0
    history_keys[valid_history] = events.item_keys[events.history_items[valid_history]]
    user_keys = events.user_keys[events.user_id]
    return CachedEvents(
        candidates=_recode(candidate_keys, item_map),
        labels=events.labels.copy(),
        user_id=_recode(user_keys, user_map),
        timestamp=events.timestamp.copy(),
        history_items=_recode(history_keys, item_map),
        history_times=events.history_times.copy(),
        history_length=events.history_length.copy(),
        rowids=events.rowids.copy() if events.rowids is not None else None,
        item_keys=np.arange(len(item_map), dtype=np.int64),
        user_keys=np.arange(len(user_map), dtype=np.int64),
    )


def join_global(train: CachedEvents, evaluation: CachedEvents) -> GlobalDataset:
    if train.item_keys is None or evaluation.item_keys is None or train.user_keys is None or evaluation.user_keys is None:
        raise ValueError("caches need item_keys and user_keys; rebuild them with src.build_cache")
    item_keys = np.unique(np.concatenate([train.item_keys, evaluation.item_keys]))
    user_keys = np.unique(np.concatenate([train.user_keys, evaluation.user_keys]))
    item_map = {int(x): int(i) for i, x in enumerate(item_keys)}
    user_map = {int(x): int(i) for i, x in enumerate(user_keys)}
    return GlobalDataset(
        train=_recode_events(train, item_map, user_map),
        evaluation=_recode_events(evaluation, item_map, user_map),
        n_users=len(user_keys),
        n_items=len(item_keys),
    )


def click_log_before(data: GlobalDataset, boundary: int) -> np.ndarray:
    """Build the deduplicated click log allowed at a boundary."""
    rows: set[tuple[int, int, int]] = set()
    for events in (data.train, data.evaluation):
        for e in range(events.n_events):
            for j, item in enumerate(events.candidates[e]):
                if events.timestamp[e] < boundary and events.labels[e, j] == 1:
                    rows.add((int(events.user_id[e]), int(item), int(events.timestamp[e])))
            valid = events.history_items[e] >= 0
            for item, ts in zip(events.history_items[e, valid], events.history_times[e, valid]):
                if int(ts) < boundary:
                    rows.add((int(events.user_id[e]), int(item), int(ts)))
    if not rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(sorted(rows), dtype=np.int64)
