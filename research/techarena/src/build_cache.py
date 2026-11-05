"""Streaming raw CSV to compact event-level NPZ caches."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .arrays import compact_id, parse_json_list
from .cache import CachedEvents, save_cache


def _read_chunks(path: str | Path, labelled: bool, chunksize: int):
    usecols = [
        "rowid", "event_id", "user_id", "item_id", "timestamp",
        "history_item_ids", "history_timestamps",
    ]
    if labelled:
        usecols.append("clicked")
    dtypes = {
        "rowid": "string", "event_id": "string", "user_id": "string", "item_id": "string",
        "timestamp": "int64", "history_item_ids": "string", "history_timestamps": "string",
    }
    if labelled:
        dtypes["clicked"] = "int8"
    yield from pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize, keep_default_na=False)


def _append_event(group: pd.DataFrame, labelled: bool, out: dict[str, list]) -> None:
    if len(group) != 10:
        raise ValueError(f"event {group.iloc[0]['event_id']} has {len(group)} rows")
    first = group.iloc[0]
    h_items = parse_json_list(first["history_item_ids"])[-20:]
    h_times = [int(x) for x in parse_json_list(first["history_timestamps"])[-20:]]
    if len(h_items) != len(h_times):
        raise ValueError(f"history item/time mismatch for {first['event_id']}")
    timestamp = int(first["timestamp"])
    if any(x >= timestamp for x in h_times):
        raise ValueError(f"history is not strictly before event {first['event_id']}")
    candidates = [compact_id(x) for x in group["item_id"].tolist()]
    out["candidates"].append(candidates)
    out["labels"].append(group["clicked"].to_numpy(np.int8).tolist() if labelled else [-1] * 10)
    out["user_id"].append(compact_id(first["user_id"]))
    out["timestamp"].append(timestamp)
    out["history_items"].append([compact_id(x) for x in h_items])
    out["history_times"].append(h_times)
    out["rowids"].append(group["rowid"].tolist())


def build_cache(input_path: str | Path, output_path: str | Path, labelled: bool, chunksize: int = 500_000) -> Path:
    out = defaultdict(list)
    carry = None
    processed = 0
    for chunk_no, chunk in enumerate(_read_chunks(input_path, labelled, chunksize), start=1):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_event = chunk.iloc[-1]["event_id"]
        complete = chunk[chunk["event_id"] != last_event]
        if not complete.empty:
            for _, group in complete.groupby("event_id", sort=False):
                _append_event(group, labelled, out)
                processed += 1
        carry = chunk[chunk["event_id"] == last_event].copy()
        if chunk_no % 5 == 0:
            print(f"processed {processed:,} events", flush=True)
    if carry is not None and not carry.empty:
        for _, group in carry.groupby("event_id", sort=False):
            _append_event(group, labelled, out)
            processed += 1

    raw_candidates = np.asarray(out["candidates"], dtype=np.int64)
    raw_hist = np.full((processed, 20), -1, dtype=np.int64)
    raw_times = np.full((processed, 20), -1, dtype=np.int64)
    for e, (items, times) in enumerate(zip(out["history_items"], out["history_times"])):
        raw_hist[e, 20 - len(items):] = items
        raw_times[e, 20 - len(times):] = times
    all_items = np.unique(np.concatenate([raw_candidates.ravel(), raw_hist[raw_hist >= 0]]))
    item_codes = {int(value): int(i) for i, value in enumerate(all_items)}
    candidates = np.vectorize(item_codes.__getitem__, otypes=[np.int32])(raw_candidates)
    history_items = np.full(raw_hist.shape, -1, dtype=np.int32)
    valid = raw_hist >= 0
    history_items[valid] = np.vectorize(item_codes.__getitem__, otypes=[np.int32])(raw_hist[valid])
    users = np.asarray(out["user_id"], dtype=np.int64)
    unique_users, user_codes = np.unique(users, return_inverse=True)
    rowids = np.asarray(out["rowids"], dtype=object)
    events = CachedEvents(
        candidates=candidates,
        labels=np.asarray(out["labels"], dtype=np.int8),
        user_id=user_codes.astype(np.int32),
        timestamp=np.asarray(out["timestamp"], dtype=np.int64),
        history_items=history_items,
        history_times=raw_times,
        history_length=(history_items >= 0).sum(axis=1).astype(np.int16),
        rowids=rowids,
        item_keys=all_items,
        user_keys=unique_users,
    )
    save_cache(output_path, events)
    meta = Path(str(output_path) + ".meta.json")
    meta.write_text(json.dumps({"events": processed, "rows": int(raw_candidates.size), "items": int(len(all_items)), "users": int(len(unique_users))}, indent=2))
    print(json.dumps({"events": processed, "rows": int(raw_candidates.size), "items": int(len(all_items)), "users": int(len(unique_users)), "output": str(output_path)}, indent=2))
    return Path(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    build_cache(args.input, args.output, labelled=not args.evaluation, chunksize=args.chunksize)
