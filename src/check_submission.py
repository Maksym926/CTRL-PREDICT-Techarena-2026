import sys

import numpy as np
import pandas as pd


def check(sub_path: str, sample_path: str) -> None:
    """Validate a submission against the organisers' sample submission before uploading."""
    sub, sample = pd.read_csv(sub_path), pd.read_csv(sample_path)
    assert list(sub.columns) == list(sample.columns), f"Columns differ: {list(sub.columns)} vs {list(sample.columns)}"
    assert len(sub) == len(sample), f"Row count {len(sub)} != {len(sample)}"
    id_col, pred_col = sample.columns[0], sample.columns[-1]
    assert sub[id_col].equals(sample[id_col]), "IDs or row order differ from sample"
    assert sub[pred_col].notna().all(), "Missing predictions"
    assert np.isfinite(sub[pred_col]).all(), "Non-finite predictions"
    print("Submission looks valid")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/check_submission.py <submission.csv> <sample_submission.csv>")
        sys.exit(1)
    check(sys.argv[1], sys.argv[2])
