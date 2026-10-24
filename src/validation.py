"""Time-based validation split that mimics how evaluation.csv was built.

evaluation.csv holds each user's FIRST event after the training week, and its histories only contain clicks from
before the training week ended. We reproduce that inside train: fit on the days before a cutoff, and validate on each
user's first event at or after the cutoff. Their histories then only reach back into the fit period, as in eval.

    from src.validation import make_split, score
    split = make_split()                 # fit days 1-4, validate days 5-7
    ...build features from split.fit_* only, predict split.val_candidates...
    print(score(split.val_candidates, predictions))
"""
from dataclasses import dataclass

import polars as pl

from src.data import load_candidates, load_events
from src.evaluate import auc

DAY = 86_400
# Fit on days 1-4 (Mon-Thu); validation then falls ~73% on day 5 (Fri), like eval's ~67% on its first day (Mon).
DEFAULT_CUTOFF_DAY = 4


@dataclass
class Split:
    fit_events: pl.DataFrame
    fit_candidates: pl.DataFrame
    val_events: pl.DataFrame
    val_candidates: pl.DataFrame  # includes `clicked`: only for scoring, never for features
    cutoff_ts: int


def make_split(cutoff_day: int = DEFAULT_CUTOFF_DAY) -> Split:
    """Fit = events before day `cutoff_day + 1`; val = each user's first event from then on."""
    events = load_events("train")
    candidates = load_candidates("train")
    cutoff_ts = events["timestamp"].min() + cutoff_day * DAY

    fit_events = events.filter(pl.col("timestamp") < cutoff_ts)
    val_events = (
        events.filter(pl.col("timestamp") >= cutoff_ts)
        .sort("timestamp", "event_id")
        .group_by("user_id", maintain_order=True)
        .first()
        .select(events.columns)
    )
    return Split(
        fit_events=fit_events,
        fit_candidates=candidates.join(fit_events.select("event_id"), on="event_id", how="semi"),
        val_events=val_events,
        val_candidates=candidates.join(val_events.select("event_id"), on="event_id", how="semi"),
        cutoff_ts=cutoff_ts,
    )


def score(val_candidates: pl.DataFrame, predictions: pl.DataFrame) -> float:
    """AUC of `predictions` (submission format: rowid, predicted_probability) on the validation rows."""
    joined = val_candidates.select("rowid", "clicked").join(
        predictions.select("rowid", "predicted_probability"), on="rowid", how="left"
    )
    missing = joined["predicted_probability"].null_count()
    if missing:
        raise ValueError(f"{missing:,} validation rows have no prediction")
    return auc(joined["clicked"].to_numpy(), joined["predicted_probability"].to_numpy())
