"""Option 3: LightGBM training windows every 6 h instead of 12 h, so many more distinct events become training targets.

    python research/techarena/safe_v2_dense.py            # validation (baseline: step 4 + Optuna = 0.804643), ~2 h
    python research/techarena/safe_v2_dense.py --quick    # smoke test: one new boundary, 10% of users, small models

Same features and parameters as optuna_step4.py (step 4 + Optuna). No sequence features: none exist for the new
boundaries. The new boundaries (days 0.25, 0.75, ..., 3.75) are added to the usual 0.5, 1.0, ..., 3.5. Day 0.5 stays
FIRST because safe_v2_step2 takes its taste-vector reference basis from bounds[0] (cached windows must keep the same
basis); the latest boundary (day 3.75) stays LAST, so it is the inner holdout. Every other cache is keyed by boundary
and holds the same rows as before, so the 12-hourly windows are reused unchanged and only the new ones are built.
Learning curve beforehand (learning_curve.py): +0.0008 / +0.0006 per doubling of users, so expect a few 1e-4.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

QUICK = "--quick" in sys.argv
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_step4 import build  # noqa: E402
from src.data import load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / ("safe_v2_dense_quick.json" if QUICK else "safe_v2_dense.json")
BASELINE = 0.804643  # step 4 + Optuna, 12-hourly windows
lf.LGB_PARAMS.update(json.loads((HERE / "outputs" / "optuna_step4.json").read_text())["best_params"])


def main() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    half = lf.bounds_for(lf.VAL_DAY, half=True)
    quarter = [lf.D0 + k * lf.DAY // 4 for k in range(1, 4 * lf.VAL_DAY, 2)]  # days 0.25, 0.75, ..., 3.75
    if QUICK:
        quarter = quarter[-1:]
    bounds = [half[0]] + sorted(half[1:] + quarter)
    assert bounds[0] == half[0] and bounds[-1] == max(bounds)
    lf.log(f"{len(bounds)} training windows (days {', '.join(f'{(x - lf.D0) / lf.DAY:g}' for x in bounds)})")

    data = lf.load_data(b)
    train_all = lf.training_windows(data, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, val = build("val", bounds, frames, lf.boundary_frame(data, b), b)
    del data, train_all, fixed, frames
    if QUICK:  # after build, so the caches hold every user's rows
        train_base = train_base[train_base["user_id"] % 10 == 0].reset_index(drop=True)
        lf.MAX_TREES = 150
    tr, va = (train_base, val) if QUICK else lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(tr):,} training rows ({tr['rowid'].nunique():,} distinct candidate rows), {len(cols)} features")

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 4 + Optuna, 6-hourly windows (leak-free)", va, pred, split, eval_frame)
    gain = res["local_auc"] - BASELINE
    lf.log(f"change vs 12-hourly windows ({BASELINE}): {gain:+.6f}")
    results = {**res, "trees": n_trees, "windows": len(bounds), "training_rows": len(tr),
               "distinct_rows": int(tr["rowid"].nunique()), "delta_vs_12h": gain,
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    if not QUICK:
        pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
            "data/validation/safe_v2_dense_val_c4.parquet", index=False)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
