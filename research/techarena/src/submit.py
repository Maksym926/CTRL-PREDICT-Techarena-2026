"""Submission validation and writing."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def validate_submission(prediction: pd.DataFrame, evaluation: pd.DataFrame, example: pd.DataFrame | None = None) -> None:
    expected = ["rowid", "predicted_probability"]
    if list(prediction.columns) != expected:
        raise ValueError(f"header must be {expected}, got {list(prediction.columns)}")
    if len(prediction) != len(evaluation):
        raise ValueError(f"row count {len(prediction)} != evaluation {len(evaluation)}")
    if prediction["rowid"].duplicated().any():
        raise ValueError("duplicate rowids")
    expected_ids = list(example["rowid"]) if example is not None else list(evaluation["rowid"])
    if set(prediction["rowid"]) != set(expected_ids):
        raise ValueError("submission rowid set differs from evaluation/example")
    if example is not None and not prediction["rowid"].equals(example["rowid"]):
        raise ValueError("submission rowids are not in example order")
    values = prediction["predicted_probability"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite prediction")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("prediction outside [0,1]")


def write_submission(rowids, scores, evaluation: pd.DataFrame, path: str | Path, example: pd.DataFrame | None = None) -> Path:
    result = pd.DataFrame({"rowid": list(rowids), "predicted_probability": np.asarray(scores, dtype=float)})
    validate_submission(result, evaluation, example)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return path
