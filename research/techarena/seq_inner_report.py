"""Compare sequence-model variants on the evaluation fit's inner holdout (each user's first event in the 12 h before the
training cutoff, scored by models fitted on events before that), from the inner_full_eval.parquet files the Kaggle
notebooks write (--save-inner). Validation is not touched, so the winner can then be validated honestly.

Usage (from the repo root):
    python research/techarena/seq_inner_report.py seq_aug seq_pair_b/more seq_pair_b/cos
Each argument is a folder under research/techarena/cache/seq. Also scores the equal-weight average of every pair of
folders and of all of them, since an ensemble of variants is a candidate too.
Decision rule, fixed before the results: the best row goes on to step 7 (safe_v2_step7.py --seq ..., validation) only
if it beats the reference folder (the first one named "more", else the first argument) by at least +0.001 here.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEQ = Path(__file__).resolve().parent / "cache" / "seq"
MIN_GAIN = 0.001


def decompose(y: np.ndarray, s: np.ndarray) -> dict:
    Y, S = y.reshape(-1, 10), s.reshape(-1, 10)
    c = Y.max(1) == 1
    cs = (S[c] * Y[c]).sum(1)
    return {"global": roc_auc_score(y, s), "which": ((S[c] < cs[:, None]).sum(1) / 9).mean(),
            "whether": roc_auc_score(Y.max(1), S.sum(1))}


names = sys.argv[1:]
frames = {n: pd.read_parquet(SEQ / n / "inner_full_eval.parquet") for n in names}
first = frames[names[0]]
for n, f in frames.items():
    assert len(f) == len(first) and (f["rowid"].to_numpy() == first["rowid"].to_numpy()).all(), f"{n}: rows differ"
y = first["clicked"].to_numpy()
print(f"inner holdout: {len(y) // 10:,} events, click share {y.mean():.4f}")
rows = {n: f["seq_prob"].to_numpy() for n, f in frames.items()}
for k in range(2, len(names) + 1):
    for combo in itertools.combinations(names, k):
        rows[" + ".join(combo)] = np.mean([frames[n]["seq_prob"].to_numpy() for n in combo], 0)
ref = next((n for n in names if n.endswith("more")), names[0])
res = pd.DataFrame({n: decompose(y, s) for n, s in rows.items()}).T
res["vs_ref"] = res["global"] - res.loc[ref, "global"]
print(res.sort_values("global", ascending=False).round(6).to_string())
best = res["global"].idxmax()
gain = res.loc[best, "vs_ref"]
print(f"\nreference: {ref}; best: {best} ({gain:+.6f}) -> "
      + ("GO: run safe_v2_step7.py with it" if gain >= MIN_GAIN else f"STOP: below +{MIN_GAIN}"))
