"""Read and write compact event-level caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class CachedEvents:
    candidates: np.ndarray
    labels: np.ndarray
    user_id: np.ndarray
    timestamp: np.ndarray
    history_items: np.ndarray
    history_times: np.ndarray
    history_length: np.ndarray
    rowids: np.ndarray | None = None
    item_keys: np.ndarray | None = None
    user_keys: np.ndarray | None = None

    @property
    def n_events(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.candidates.size)


def save_cache(path: str | Path, events: CachedEvents) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "candidates": events.candidates.astype(np.int32, copy=False),
        "labels": events.labels.astype(np.int8, copy=False),
        "user_id": events.user_id.astype(np.int32, copy=False),
        "timestamp": events.timestamp.astype(np.int64, copy=False),
        "history_items": events.history_items.astype(np.int32, copy=False),
        "history_times": events.history_times.astype(np.int64, copy=False),
        "history_length": events.history_length.astype(np.int16, copy=False),
    }
    if events.rowids is not None:
        kwargs["rowids"] = events.rowids
    if events.item_keys is not None:
        kwargs["item_keys"] = events.item_keys.astype(np.int64, copy=False)
    if events.user_keys is not None:
        kwargs["user_keys"] = events.user_keys.astype(np.int64, copy=False)
    np.savez(path, **kwargs)
    return path


def load_cache(path: str | Path) -> CachedEvents:
    with np.load(path, allow_pickle=True) as data:
        return CachedEvents(
            candidates=data["candidates"],
            labels=data["labels"],
            user_id=data["user_id"],
            timestamp=data["timestamp"],
            history_items=data["history_items"],
            history_times=data["history_times"],
            history_length=data["history_length"],
            rowids=data["rowids"] if "rowids" in data else None,
            item_keys=data["item_keys"] if "item_keys" in data else None,
            user_keys=data["user_keys"] if "user_keys" in data else None,
        )
