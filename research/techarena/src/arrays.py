"""Event-level arrays shared by feature and model code."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import EVENT_SIZE, HISTORY_SIZE


ITEM_HEX = re.compile(r"(?:^|_)([0-9a-f]{8,})$")


def parse_json_list(value: object) -> list:
    """Parse a CSV JSON-list cell without accepting malformed values silently."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value)
    if not text or text == "[]":
        return []
    result = json.loads(text)
    if not isinstance(result, list):
        raise ValueError(f"history cell is not a list: {text[:80]}")
    return result


def compact_id(value: object) -> int:
    """Turn an opaque prefixed hex identifier into a stable integer."""
    text = str(value)
    match = ITEM_HEX.search(text)
    if match:
        # Use the low 60 bits.  Real IDs are random, while fake IDs often have
        # leading zero padding; taking the high digits would collapse them.
        return int(match.group(1)[-15:], 16)
    # Fake data uses the same prefixes, but this keeps the helper general.
    return hash(text) & ((1 << 63) - 1)


@dataclass
class EventTable:
    event_id: np.ndarray
    user_id: np.ndarray
    timestamp: np.ndarray
    candidates: np.ndarray
    labels: np.ndarray
    history_items: np.ndarray
    history_times: np.ndarray
    history_length: np.ndarray
    rowids: np.ndarray | None = None
    time_bucket: np.ndarray | None = None
    day_type: np.ndarray | None = None

    @property
    def n_events(self) -> int:
        return int(self.candidates.shape[0])


def _build_code_map(values: Iterable[object]) -> dict[object, int]:
    unique = pd.unique(pd.Series(list(values), dtype="object"))
    return {value: int(i) for i, value in enumerate(unique)}


def frame_to_events(frame: pd.DataFrame, labelled: bool = True) -> EventTable:
    """Convert row-level data to strict 10-candidate event arrays."""
    required = {"event_id", "user_id", "item_id", "timestamp", "history_item_ids", "history_timestamps"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame.reset_index(drop=True)
    counts = frame.groupby("event_id", sort=False).size()
    bad = counts[counts != EVENT_SIZE]
    if not bad.empty:
        raise ValueError(f"events without exactly {EVENT_SIZE} rows: {bad.to_dict()}")

    groups = list(frame.groupby("event_id", sort=False))
    user_map = _build_code_map(frame["user_id"].tolist())
    item_values = []
    for _, group in groups:
        item_values.extend(group["item_id"].tolist())
        item_values.extend(parse_json_list(group.iloc[0]["history_item_ids"]))
    item_map = _build_code_map(item_values)

    n = len(groups)
    candidates = np.full((n, EVENT_SIZE), -1, dtype=np.int64)
    labels = np.full((n, EVENT_SIZE), -1, dtype=np.int8)
    history_items = np.full((n, HISTORY_SIZE), -1, dtype=np.int64)
    history_times = np.full((n, HISTORY_SIZE), -1, dtype=np.int64)
    event_ids = np.empty(n, dtype=object)
    user_ids = np.empty(n, dtype=np.int64)
    timestamp = np.empty(n, dtype=np.int64)
    rowids = np.empty((n, EVENT_SIZE), dtype=object) if "rowid" in frame else None
    time_bucket = np.empty(n, dtype=np.int16) if "time_bucket" in frame else None
    day_type = np.empty(n, dtype=np.int8) if "day_type" in frame else None

    for e, (event_id, group) in enumerate(groups):
        first = group.iloc[0]
        event_ids[e] = event_id
        user_ids[e] = user_map[first["user_id"]]
        timestamp[e] = int(first["timestamp"])
        if time_bucket is not None:
            time_bucket[e] = int(first["time_bucket"])
        if day_type is not None:
            day_type[e] = int(first["day_type"])
        for j, (_, row) in enumerate(group.iterrows()):
            candidates[e, j] = item_map[row["item_id"]]
            labels[e, j] = int(row["clicked"]) if labelled and "clicked" in row else -1
            if rowids is not None:
                rowids[e, j] = row["rowid"]
        h_items = parse_json_list(first["history_item_ids"])[-HISTORY_SIZE:]
        h_times = parse_json_list(first["history_timestamps"])[-HISTORY_SIZE:]
        if len(h_items) != len(h_times):
            raise ValueError(f"history item/time length mismatch for event {event_id}")
        start = HISTORY_SIZE - len(h_items)
        history_items[e, start:] = [item_map[x] for x in h_items]
        history_times[e, start:] = [int(x) for x in h_times]
        if np.any(history_times[e] >= timestamp[e]):
            raise ValueError(f"history contains a timestamp at/after event {event_id}")

    return EventTable(
        event_id=event_ids,
        user_id=user_ids,
        timestamp=timestamp,
        candidates=candidates,
        labels=labels,
        history_items=history_items,
        history_times=history_times,
        history_length=(history_items >= 0).sum(axis=1).astype(np.int16),
        rowids=rowids,
        time_bucket=time_bucket,
        day_type=day_type,
    )
