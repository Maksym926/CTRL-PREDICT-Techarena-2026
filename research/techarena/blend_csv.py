"""50/50 rank blend of two submission CSVs (rowid, predicted_probability), e.g. the binary and softmax-objective models.

Usage (from the repo root):
    python research/techarena/blend_csv.py submissions/safe_v2_step5.csv submissions/safe_v2_step5_softmax.csv \
        submissions/safe_v2_step5_blend.csv
Both files must cover exactly the same rows; the output is the mean of the two normalised ranks (only order matters
for AUC), in (0, 1].
"""
import sys

import numpy as np
import pandas as pd

a_path, b_path, out_path = sys.argv[1:4]
a, b = pd.read_csv(a_path), pd.read_csv(b_path)
assert len(a) == len(b) and a["rowid"].is_unique and set(a["rowid"]) == set(b["rowid"]), "row sets differ"
m = a.merge(b, on="rowid", suffixes=("_a", "_b"))
assert m[["predicted_probability_a", "predicted_probability_b"]].notna().all().all(), "missing predictions"
r = 0.5 * (m["predicted_probability_a"].rank() + m["predicted_probability_b"].rank()) / len(m)
pd.DataFrame({"rowid": m["rowid"], "predicted_probability": r}).to_csv(out_path, index=False)
rho = np.corrcoef(m["predicted_probability_a"].rank(), m["predicted_probability_b"].rank())[0, 1]
print(f"wrote {out_path}: {len(m):,} rows, rank correlation of the inputs {rho:.4f}")
