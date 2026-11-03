"""Validation AUC of a 50/50 rank blend of two validation prediction files (rowid, predicted_probability; written by
the steps to data/validation/*_val_c4.parquet), to check a blend before building its submission.

Usage (from the repo root):
    python research/techarena/blend_val.py data/validation/safe_v2_step7_val_c4.parquet \
        data/validation/softmax_step7_val_c4.parquet
Same blend as blend_csv.py (mean of the two normalised ranks).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation import make_split  # noqa: E402

a_path, b_path = sys.argv[1:3]
lab = make_split(4).val_candidates.select("rowid", "clicked").to_pandas()
d = lab.assign(rowid=lab["rowid"].astype(str))
for name, path in (("a", a_path), ("b", b_path)):
    p = pd.read_parquet(path)
    d = d.merge(p.assign(rowid=p["rowid"].astype(str)).rename(columns={"predicted_probability": name}), on="rowid")
assert len(d) == len(lab), "files do not cover the validation rows"
y = d["clicked"].to_numpy()
blend = 0.5 * (d["a"].rank() + d["b"].rank()) / len(d)
print(f"{Path(a_path).name}: {roc_auc_score(y, d['a']):.6f}")
print(f"{Path(b_path).name}: {roc_auc_score(y, d['b']):.6f}")
print(f"50/50 rank blend: {roc_auc_score(y, blend):.6f}")
