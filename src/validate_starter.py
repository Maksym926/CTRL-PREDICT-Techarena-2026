"""Score the unmodified starter solution on the local validation split.

Usage: python src/validate_starter.py [--cutoff-day 4] [--keep-csv]

The starter only reads CSV, so this writes the fit period and the validation period of train.csv to temporary CSVs
(in the original format), runs starter_solution.py on them, and scores its predictions with src.validation.score.
If the local AUC lands close to the starter's leaderboard AUC (0.619958), the split can be trusted.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.validation import DEFAULT_CUTOFF_DAY, make_split, score  # noqa: E402

TRAIN_CSV = Path("data/train.csv")
TMP_DIR = Path("data/validation")
STARTER = Path("starter_solution.py")


def write_period_csv(out: Path, keep, drop_label: bool) -> None:
    lf = pl.scan_csv(TRAIN_CSV, infer_schema=False).filter(keep(pl.col("timestamp").cast(pl.Int64)))
    if drop_label:
        lf = lf.drop("clicked")
    lf.sink_csv(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-day", type=int, default=DEFAULT_CUTOFF_DAY)
    ap.add_argument("--keep-csv", action="store_true", help="keep the temporary CSVs in data/validation/")
    a = ap.parse_args()

    split = make_split(a.cutoff_day)
    print(f"cutoff day {a.cutoff_day}: fit {split.fit_events.height:,} events, "
          f"val {split.val_events.height:,} events (first per user)", flush=True)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fit_csv = TMP_DIR / f"fit_c{a.cutoff_day}.csv"
    val_csv = TMP_DIR / f"val_c{a.cutoff_day}.csv"
    preds_csv = TMP_DIR / f"starter_preds_c{a.cutoff_day}.csv"

    start = time.perf_counter()
    write_period_csv(fit_csv, lambda ts: ts < split.cutoff_ts, drop_label=False)
    # Every event after the cutoff, so we can also report the naive "all events" AUC; the starter scores each
    # row independently, so predicting extra rows does not change the first-per-user predictions.
    write_period_csv(val_csv, lambda ts: ts >= split.cutoff_ts, drop_label=True)
    print(f"wrote temporary CSVs in {time.perf_counter() - start:.0f} s", flush=True)

    start = time.perf_counter()
    subprocess.run([sys.executable, str(STARTER), "--train", str(fit_csv), "--evaluation", str(val_csv),
                    "--out", str(preds_csv)], check=True)
    runtime = time.perf_counter() - start

    preds = pl.read_csv(preds_csv, schema_overrides={"rowid": pl.String, "predicted_probability": pl.Float64})
    all_val = pl.read_parquet("data/parquet/train_candidates.parquet").join(
        pl.read_parquet("data/parquet/train_events.parquet", columns=["event_id", "timestamp"])
        .filter(pl.col("timestamp") >= split.cutoff_ts), on="event_id", how="semi")

    print(f"\nstarter runtime: {runtime:.0f} s")
    print(f"local AUC, first event per user (the validation metric): {score(split.val_candidates, preds):.6f}")
    print(f"local AUC, all events after cutoff (naive, for comparison): {score(all_val, preds):.6f}")
    print("leaderboard AUC of the same starter:                       0.619958")

    if not a.keep_csv:
        for p in (fit_csv, val_csv, preds_csv):
            p.unlink()


if __name__ == "__main__":
    main()
