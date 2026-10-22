import sys

import numpy as np
import pandas as pd

EXPECTED_HEADER = ["rowid", "predicted_probability"]


def check(sub_path: str, reference_path: str) -> None:
    """Validate a submission against the official rules before uploading.

    reference_path: data/example_submission.csv or data/evaluation.csv (only `rowid` is read).
    """
    sub = pd.read_csv(sub_path, dtype={"rowid": str})
    ref_ids = pd.read_csv(reference_path, usecols=["rowid"], dtype={"rowid": str})["rowid"]

    assert list(sub.columns) == EXPECTED_HEADER, f"Header must be {EXPECTED_HEADER}, got {list(sub.columns)}"
    assert not sub["rowid"].duplicated().any(), f"{sub['rowid'].duplicated().sum()} duplicate rowids"

    missing = set(ref_ids) - set(sub["rowid"])
    unknown = set(sub["rowid"]) - set(ref_ids)
    assert not missing, f"{len(missing)} evaluation rowids missing, e.g. {list(missing)[:3]}"
    assert not unknown, f"{len(unknown)} unknown rowids, e.g. {list(unknown)[:3]}"

    p = pd.to_numeric(sub["predicted_probability"], errors="coerce")
    assert p.notna().all(), f"{p.isna().sum()} empty or non-numeric predictions"
    assert np.isfinite(p).all(), "Non-finite predictions (inf)"
    assert ((p >= 0) & (p <= 1)).all(), f"Predictions outside [0, 1]: min={p.min()}, max={p.max()}"

    print(f"Submission valid: {len(sub):,} rows, predictions in [{p.min():.6g}, {p.max():.6g}], "
          f"{p.nunique():,} distinct values")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/check_submission.py <submission.csv> <reference.csv>")
        sys.exit(1)
    check(sys.argv[1], sys.argv[2])
