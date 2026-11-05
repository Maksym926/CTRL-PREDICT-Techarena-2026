"""Step 3b: the step 2 model trained on windows every 6 hours instead of every 12 (leak-free).

Usage (from the repo root, after safe_v2_step2.py validation):
    python research/techarena/safe_v2_step3_6h.py            # validation (step 2 = 0.804154)

Twice as many training windows (boundaries at days 0.25, 0.5, ..., 3.75 for validation). Half-day windows reuse every
cache; the new quarter-day windows build SAFE_v2, step 1 and step 2 features with the same code. Also reports a
fixed 50/50 rank blend with the step 2 predictions.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_step2 import build  # noqa: E402
from src.data import load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / "safe_v2_step3_6h.json"
STEP2_VAL = Path("data/validation/safe_v2_step2_val_c4.parquet")
BASELINE = 0.804154


def bounds_6h(day: int) -> list[int]:
    return [lf.D0 + i * lf.DAY // 4 for i in range(1, 4 * day)]


def rank(x: np.ndarray) -> np.ndarray:
    return x.argsort().argsort() / (len(x) - 1)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = bounds_6h(lf.VAL_DAY)
    data = lf.load_data(b)  # needed for the SAFE_v2 frames of the new quarter-day boundaries
    train_all = lf.training_windows(data, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    del data, train_all
    train_base, val = build("val", bounds, frames, lf.boundary_frame(None, b), b)
    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(bounds)} windows, {len(tr):,} training rows, {len(cols)} features")

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 3b: 6-hour windows", va, pred, split, eval_frame)
    lf.log(f"change vs step 2 ({BASELINE}): {res['local_auc'] - BASELINE:+.6f}")

    p2 = pl.DataFrame({"rowid": va["rowid"].astype(str).to_numpy()}).join(pl.read_parquet(STEP2_VAL), on="rowid",
                                                                          how="left")["predicted_probability"].to_numpy()
    corr = float(np.corrcoef(rank(pred), rank(p2))[0, 1])
    blend = lf.evaluate("BLEND 50/50 with step 2", va, 0.5 * rank(pred) + 0.5 * rank(p2), split, eval_frame)
    lf.log(f"rank correlation with step 2: {corr:.4f}")

    results = {"step3_6h": {**res, "trees": n_trees, "windows": len(bounds), "train_rows": len(tr),
                            "delta_vs_step2": res["local_auc"] - BASELINE},
               "blend_with_step2": {**blend, "rank_corr": corr},
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        "data/validation/safe_v2_step3_6h_val_c4.parquet", index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    validate()
