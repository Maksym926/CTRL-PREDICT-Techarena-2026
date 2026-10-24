"""Checks that data/parquet/ is a faithful, well-formed conversion of the competition CSVs.

Usage: python src/verify_parquet.py   (run from the repo root, after convert_to_parquet.py)
"""
import json
import random
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_candidates, load_events, load_joined  # noqa: E402

TRAIN_CSV = Path("data/train.csv")
EXPECTED_ROWS = {
    ("train", "candidates"): 6_868_240,
    ("train", "events"): 686_824,
    ("eval", "candidates"): 529_390,
    ("eval", "events"): 52_939,
}
ID_PREFIXES = {"rowid": "row_", "event_id": "evt_", "user_id": "usr_", "item_id": "itm_"}
MAX_HISTORY = 20
SPOT_CHECKS = 3

failures = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def check_split(split: str) -> None:
    cands, events = load_candidates(split), load_events(split)

    # 1. Row counts
    for kind, df in (("candidates", cands), ("events", events)):
        expected = EXPECTED_ROWS[(split, kind)]
        report(f"1. {split} {kind} row count", df.height == expected, f"{df.height:,} (expected {expected:,})")

    # 2. Exactly 10 candidates per event
    sizes = cands.group_by("event_id").len()["len"]
    report(f"2. {split} 10 candidates per event", (sizes == 10).all(),
           f"group sizes min={sizes.min()} max={sizes.max()}, {(sizes != 10).sum():,} events not of size 10")

    # 3. Candidate and event IDs match both ways
    cand_ids, event_ids = cands["event_id"].unique(), events["event_id"]
    orphan_cands = (~cand_ids.is_in(event_ids.implode())).sum()
    orphan_events = (~event_ids.is_in(cand_ids.implode())).sum()
    dup_events = event_ids.is_duplicated().sum()
    report(f"3. {split} event_id sets match", orphan_cands == 0 and orphan_events == 0 and dup_events == 0,
           f"{orphan_cands:,} candidate events missing from events, {orphan_events:,} events without candidates, "
           f"{dup_events:,} duplicated event rows")

    # 4. At most one click per event (train only)
    if split == "train":
        per_event = cands.group_by("event_id").agg(pl.col("clicked").sum().alias("clicks"))
        report("4. train at most one click per event", per_event["clicks"].max() <= 1,
               f"max clicks in an event = {per_event['clicks'].max()}, "
               f"click rate = {cands['clicked'].mean():.4f} (expected ~0.0629), "
               f"events with a click = {(per_event['clicks'] > 0).mean():.4f}")

    # 5. History lists aligned and capped
    lens = events.select(
        pl.col("history_item_ids").list.len().alias("n_items"),
        pl.col("history_timestamps").list.len().alias("n_ts"),
    )
    mismatched = (lens["n_items"] != lens["n_ts"]).sum()
    n = lens["n_items"]
    report(f"5. {split} history lengths", mismatched == 0 and n.max() <= MAX_HISTORY,
           f"{mismatched:,} mismatched; length min={n.min()} median={n.median():.0f} max={n.max()}, "
           f"{(n == 0).sum():,} empty histories")

    # 6. History timestamps sorted and strictly before the event
    hist = events.select(
        (pl.col("history_timestamps") == pl.col("history_timestamps").list.sort()).alias("sorted"),
        (pl.col("history_timestamps").list.max() < pl.col("timestamp")).fill_null(True).alias("before"),
    )
    unsorted, not_before = (~hist["sorted"]).sum(), (~hist["before"]).sum()
    report(f"6. {split} history timestamps ordered and in the past", unsorted == 0 and not_before == 0,
           f"{unsorted:,} events with unsorted history, {not_before:,} with history at/after event time")

    # 7. One event per user (eval only)
    if split == "eval":
        dup_users = events["user_id"].is_duplicated().sum()
        report("7. eval one event per user", dup_users == 0,
               f"{events['user_id'].n_unique():,} unique users, {dup_users:,} rows with a repeated user_id")

    # 8. ID columns are strings with the expected prefix
    for kind, df in (("candidates", cands), ("events", events)):
        for col, prefix in ID_PREFIXES.items():
            if col not in df.columns:
                continue
            is_str = df.schema[col] == pl.String
            bad = (~df[col].str.starts_with(prefix)).sum() if is_str else df.height
            report(f"8. {split} {kind}.{col} format", is_str and bad == 0,
                   f"dtype={df.schema[col]}, {bad:,} values not starting with '{prefix}'")


def check_spot_rows() -> None:
    # 9. A few random rows match the original CSV after parsing
    cands = load_candidates("train")
    ids = random.Random(0).sample(cands["rowid"].to_list(), SPOT_CHECKS)
    del cands
    parquet_rows = {r["rowid"]: r for r in load_joined("train").filter(pl.col("rowid").is_in(ids)).to_dicts()}
    csv_rows = pl.scan_csv(TRAIN_CSV, infer_schema=False).filter(pl.col("rowid").is_in(ids)).collect().to_dicts()

    mismatches = []
    for raw in csv_rows:
        pq = parquet_rows.get(raw["rowid"])
        expected = {
            **{c: raw[c] for c in ("rowid", "event_id", "user_id", "item_id")},
            **{c: int(raw[c]) for c in ("clicked", "timestamp", "time_bucket", "day_type")},
            "history_item_ids": json.loads(raw["history_item_ids"]),
            "history_timestamps": json.loads(raw["history_timestamps"]),
        }
        if pq is None or any(pq[k] != v for k, v in expected.items()):
            mismatches.append(raw["rowid"])
    ok = len(csv_rows) == SPOT_CHECKS and not mismatches
    report("9. spot check vs CSV", ok,
           f"{len(csv_rows)}/{SPOT_CHECKS} rows found in CSV, mismatching: {mismatches or 'none'} ({', '.join(ids)})")


def check_load_time() -> None:
    # 10. Cost of loading the joined training table
    start = time.perf_counter()
    df = load_joined("train")
    elapsed = time.perf_counter() - start
    report("10. load_joined('train')", True,
           f"{elapsed:.1f} s, {df.height:,} rows, estimated in-memory size {df.estimated_size('gb'):.2f} GB")


def main() -> None:
    for split in ("train", "eval"):
        check_split(split)
    check_spot_rows()
    check_load_time()
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
