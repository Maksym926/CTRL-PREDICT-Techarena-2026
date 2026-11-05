"""Strictly-before-boundary statistics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .arrays import EventTable


@dataclass
class BoundaryStats:
    boundary: int
    item_impressions: dict[int, int]
    item_clicks: dict[int, int]
    user_impressions: dict[int, int]
    user_clicks: dict[int, int]
    click_log: np.ndarray


def _add(mapping: dict[int, int], key: int, value: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + value


def build_stats(events: EventTable, boundary: int) -> BoundaryStats:
    """Build statistics from event impressions strictly before ``boundary``.

    A target event's supplied history remains available to feature code, but it
    is never inserted into these aggregate counters.
    """
    item_imp: dict[int, int] = {}
    item_click: dict[int, int] = {}
    user_imp: dict[int, int] = {}
    user_click: dict[int, int] = {}
    click_rows: list[tuple[int, int, int]] = []
    mask = events.timestamp < int(boundary)
    for e in np.flatnonzero(mask):
        user = int(events.user_id[e])
        for j, item in enumerate(events.candidates[e]):
            item = int(item)
            _add(item_imp, item)
            _add(user_imp, user)
            if int(events.labels[e, j]) == 1:
                _add(item_click, item)
                _add(user_click, user)
                click_rows.append((user, item, int(events.timestamp[e])))
    return BoundaryStats(
        boundary=int(boundary),
        item_impressions=item_imp,
        item_clicks=item_click,
        user_impressions=user_imp,
        user_clicks=user_click,
        click_log=np.asarray(click_rows, dtype=np.int64).reshape((-1, 3)) if click_rows else np.empty((0, 3), dtype=np.int64),
    )
