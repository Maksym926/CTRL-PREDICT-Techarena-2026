"""Configuration for the from-scratch TechArena CTR pipeline.

The defaults point at the original competition files in Downloads.  All paths
can be overridden with environment variables so the fake-data test and a
portable checkout use the same code.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TECARENA_DATA_DIR", "/Users/monishobaid/Downloads"))
TRAIN_PATH = Path(os.environ.get("TECARENA_TRAIN", DATA_DIR / "train.csv"))
EVAL_PATH = Path(os.environ.get("TECARENA_EVAL", DATA_DIR / "evaluation.csv"))
EXAMPLE_PATH = Path(os.environ.get("TECARENA_EXAMPLE", DATA_DIR / "example_submission.csv"))
OUTPUT_DIR = Path(os.environ.get("TECARENA_OUTPUT", ROOT / "outputs"))
CACHE_DIR = Path(os.environ.get("TECARENA_CACHE", ROOT / "cache"))

D0 = 1_737_331_200  # 2025-01-20 00:00 UTC
CUT = 1_737_936_000  # 2025-01-27 00:00 UTC
DAY = 86_400
EVENT_SIZE = 10
HISTORY_SIZE = 20
RANDOM_SEED = 20261024

for _path in (OUTPUT_DIR, CACHE_DIR):
    _path.mkdir(parents=True, exist_ok=True)
