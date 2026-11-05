"""Chunked ingestion helpers.

The first implementation keeps the raw columns long enough to validate the
schema, then writes compact event-level parquet.  History JSON is parsed once
per event by :func:`frame_to_events`.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .arrays import frame_to_events


RAW_DTYPES = {
    "rowid": "string",
    "event_id": "string",
    "user_id": "string",
    "item_id": "string",
    "timestamp": "int64",
    "time_bucket": "int16",
    "day_type": "int8",
    "history_item_ids": "string",
    "history_timestamps": "string",
    "clicked": "int8",
}


def read_csv_chunks(path: str | Path, chunksize: int = 500_000, labelled: bool = True):
    """Yield validated raw chunks; the caller owns the cache strategy."""
    usecols = list(RAW_DTYPES)
    if not labelled:
        usecols.remove("clicked")
    dtypes = {key: value for key, value in RAW_DTYPES.items() if key in usecols}
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=dtypes, keep_default_na=False):
        if chunk.empty:
            continue
        yield chunk


def read_small(path: str | Path, labelled: bool = True) -> pd.DataFrame:
    chunks = list(read_csv_chunks(path, chunksize=500_000, labelled=labelled))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def validate_event_schema(frame: pd.DataFrame, labelled: bool = True) -> None:
    frame_to_events(frame, labelled=labelled)
