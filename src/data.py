from pathlib import Path

import polars as pl

PARQUET_DIR = Path("data/parquet")


def load_events(split: str) -> pl.DataFrame:
    """split: 'train' or 'eval'"""
    return pl.read_parquet(PARQUET_DIR / f"{split}_events.parquet")


def load_candidates(split: str) -> pl.DataFrame:
    """split: 'train' or 'eval'"""
    return pl.read_parquet(PARQUET_DIR / f"{split}_candidates.parquet")


def load_joined(split: str) -> pl.DataFrame:
    """Candidates with their event columns attached (join on event_id)."""
    return load_candidates(split).join(load_events(split), on="event_id", how="left")
