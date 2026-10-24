"""One-off conversion of the competition CSVs into normalised Parquet tables.

Usage: python src/convert_to_parquet.py

Each CSV is split into:
  - <split>_candidates.parquet: one row per candidate (rowid, event_id, item_id[, clicked])
  - <split>_events.parquet:     one row per event, with the history JSON parsed into list columns
"""
import time
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "parquet"
SOURCES = {"train": DATA_DIR / "train.csv", "eval": DATA_DIR / "evaluation.csv"}

EVENT_COLS = ["event_id", "user_id", "timestamp", "time_bucket", "day_type",
              "history_item_ids", "history_timestamps"]


def scan(path: Path) -> pl.LazyFrame:
    # Every column as String: IDs must never be parsed as numbers
    return pl.scan_csv(path, infer_schema=False)


def write_candidates(src: Path, out: Path, has_label: bool) -> None:
    cols = [pl.col("rowid"), pl.col("event_id"), pl.col("item_id")]
    if has_label:
        cols.append(pl.col("clicked").cast(pl.Int8))
    scan(src).select(cols).sink_parquet(out, compression="zstd")


def write_events(src: Path, out: Path) -> None:
    (
        scan(src)
        .select(EVENT_COLS)
        # Deduplicate first, so the history JSON is parsed once per event, not once per candidate
        .unique(subset="event_id", keep="any")
        .with_columns(
            pl.col("timestamp").cast(pl.Int64),
            pl.col("time_bucket").cast(pl.Int8),
            pl.col("day_type").cast(pl.Int8),
            pl.col("history_item_ids").str.json_decode(pl.List(pl.String)),
            pl.col("history_timestamps").str.json_decode(pl.List(pl.Int64)),
        )
        # unique() does not preserve order; sort so the output is deterministic
        .sort("timestamp", "event_id")
        .sink_parquet(out, compression="zstd")
    )


def timed(label: str, fn, out: Path) -> float:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    print(f"{label:<28} {elapsed:7.1f} s   {out.stat().st_size / 1e6:8.1f} MB   {out}", flush=True)
    return elapsed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for split, src in SOURCES.items():
        has_label = split == "train"
        cand_out = OUT_DIR / f"{split}_candidates.parquet"
        events_out = OUT_DIR / f"{split}_events.parquet"
        total += timed(f"{split}_candidates.parquet", lambda: write_candidates(src, cand_out, has_label), cand_out)
        total += timed(f"{split}_events.parquet", lambda: write_events(src, events_out), events_out)
    print(f"Total conversion time: {total:.1f} s")


if __name__ == "__main__":
    main()
