"""Stage 2: does the sequence model add anything on top of the LightGBM stack? (validation, after seq_model.py --mode val)

    python research/techarena/seq_blend_check.py

Decision rule, fixed before seeing the sequence model's results: rank blend 0.75 * stack + 0.25 * sequence model;
stage 3 (sequence features per window) only if it beats the stack alone by at least +0.001 local AUC. Other weights
are printed for information only, since choosing a weight on validation would leak.
"""
import sys
from pathlib import Path

import pandas as pd
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation import make_split, score  # noqa: E402

W_SEQ, MIN_GAIN = 0.25, 0.001
SEQ = "research/techarena/cache/seq/seq_val.parquet"
STACK = "data/validation/optuna_step4_val_c4.parquet"
OTHERS = {"choice model v2 alone": "data/validation/choice_mf_val_c4.parquet",
          "step 2 (submitted)": "data/validation/safe_v2_step2_val_c4.parquet"}


def auc(df: pd.DataFrame, col: str) -> float:
    return score(split.val_candidates, pl.DataFrame({"rowid": df["rowid"].tolist(),
                                                     "predicted_probability": df[col].to_numpy()}))


split = make_split(4)
d = pd.read_parquet(STACK).rename(columns={"predicted_probability": "stack"}).merge(
    pd.read_parquet(SEQ)[["rowid", "seq_prob"]], on="rowid")
assert len(d) == split.val_candidates.height, "row mismatch"
for name, f in OTHERS.items():
    d = d.merge(pd.read_parquet(f).rename(columns={"predicted_probability": name}), on="rowid")

base = auc(d, "stack")
print(f"{'model':32s} {'local AUC':>10s} {'spearman vs stack':>18s}")
for c in ["stack", "seq_prob", *OTHERS]:
    print(f"{c:32s} {auc(d, c):10.6f} {spearmanr(d[c], d["stack"]).statistic:18.4f}")
print(f"seq vs choice model v2 spearman: {spearmanr(d["seq_prob"], d['choice model v2 alone']).statistic:.4f}")

r_stack, r_seq = d["stack"].rank(pct=True), d["seq_prob"].rank(pct=True)
print("\nrank blends (stack weight, seq weight):")
for w in (0.1, 0.25, 0.4, 0.5):
    d["blend"] = (1 - w) * r_stack + w * r_seq
    a = auc(d, "blend")
    tag = "  <- pre-declared" if w == W_SEQ else "  (info only)"
    print(f"  {1 - w:.2f} / {w:.2f}: {a:.6f} ({a - base:+.6f}){tag}")
    if w == W_SEQ:
        decided = a - base
print(f"\nDECISION: blend gain {decided:+.6f} vs required +{MIN_GAIN}: "
      + ("GO to stage 3 (sequence features per window)" if decided >= MIN_GAIN else "STOP: no meaningful gain"))
