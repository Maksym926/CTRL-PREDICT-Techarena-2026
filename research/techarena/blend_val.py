"""Validation AUC of a 50/50 rank blend of two validation prediction files (rowid, predicted_probability; written by
the steps to data/validation/*_val_c4.parquet), to check a blend before building its submission.

Usage (from the repo root):
    python research/techarena/blend_val.py data/validation/safe_v2_step7_val_c4.parquet \
        data/validation/softmax_step7_val_c4.parquet [--factorised 0.1 0.7]
Same blend as blend_csv.py (mean of the two normalised ranks). --factorised W_WHETHER W_WHICH also scores the
factorised blend (see blend_csv.py): each model's event click mass and within-event shares are mixed separately.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation import make_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend_csv import factorised  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("a_path")
ap.add_argument("b_path")
ap.add_argument("--factorised", type=float, nargs=2, metavar=("W_WHETHER", "W_WHICH"))
args = ap.parse_args()
lab = make_split(4).val_candidates.select("rowid", "event_id", "clicked").to_pandas()
d = lab.assign(rowid=lab["rowid"].astype(str))
for name, path in (("a", args.a_path), ("b", args.b_path)):
    p = pd.read_parquet(path)
    d = d.merge(p.assign(rowid=p["rowid"].astype(str)).rename(columns={"predicted_probability": name}), on="rowid")
assert len(d) == len(lab), "files do not cover the validation rows"
y = d["clicked"].to_numpy()
blend = 0.5 * (d["a"].rank() + d["b"].rank()) / len(d)
print(f"{Path(args.a_path).name}: {roc_auc_score(y, d['a']):.6f}")
print(f"{Path(args.b_path).name}: {roc_auc_score(y, d['b']):.6f}")
print(f"50/50 rank blend: {roc_auc_score(y, blend):.6f}")
if args.factorised:
    f = factorised(d["event_id"].to_numpy(), d["a"].to_numpy(), d["b"].to_numpy(), *args.factorised)
    print(f"factorised blend {args.factorised[0]:g}/{args.factorised[1]:g}: {roc_auc_score(y, f):.6f}")
    print(f"  (rank correlation with the 50/50 rank blend {np.corrcoef(blend, pd.Series(f).rank())[0, 1]:.6f})")
