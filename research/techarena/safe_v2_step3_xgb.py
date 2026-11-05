"""Step 3a: XGBoost on the step 2 features, blended 50/50 (rank average) with the step 2 LightGBM (leak-free).

Usage (from the repo root, after safe_v2_step2.py validation):
    python research/techarena/safe_v2_step3_xgb.py            # validation (LightGBM step 2 = 0.804154)
    python research/techarena/safe_v2_step3_xgb.py --submit   # needs safe_v2_step2.py --submit first

The tree count comes from the same inner time holdout as LightGBM (latest window vs. targets before it); the blend
uses fixed equal weights, so nothing is tuned on the validation events.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_step2 import build  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / "safe_v2_step3_xgb.json"
LGB_VAL = Path("data/validation/safe_v2_step2_val_c4.parquet")
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", learning_rate=0.05, max_depth=8,
                  min_child_weight=50, subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0, max_bin=127, nthread=12)
MAX_TREES, PATIENCE = 3000, 80
SEEDS = (0, 1)


def xgb_trees(tr: pd.DataFrame, bounds: list[int], cols: list[str]) -> int:
    """Tree count on the inner holdout: latest window vs. targets dated before it (as lf.inner_holdout_trees)."""
    last = len(bounds) - 1
    inner_tr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    inner_va = tr[tr["window_id"] == last]
    dtr = xgb.DMatrix(inner_tr[cols], label=inner_tr["label"])
    dva = xgb.DMatrix(inner_va[cols], label=inner_va["label"])
    booster = xgb.train({**XGB_PARAMS, "seed": SEEDS[0]}, dtr, MAX_TREES, evals=[(dva, "holdout")],
                        early_stopping_rounds=PATIENCE, verbose_eval=False)
    lf.log(f"  XGBoost inner holdout chose {booster.best_iteration + 1} trees (holdout AUC {booster.best_score:.5f})")
    return booster.best_iteration + 1


def xgb_fit_predict(tr: pd.DataFrame, target: pd.DataFrame, cols: list[str], n_trees: int) -> np.ndarray:
    dtr = xgb.DMatrix(tr[cols], label=tr["label"])
    dte = xgb.DMatrix(target[cols])
    preds = []
    for seed in SEEDS:
        preds.append(xgb.train({**XGB_PARAMS, "seed": seed}, dtr, n_trees).predict(dte))
        lf.log(f"  XGBoost seed {seed} done")
    return np.mean(preds, axis=0)


def rank(x: np.ndarray) -> np.ndarray:
    return x.argsort().argsort() / (len(x) - 1)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, val = build("val", bounds, frames, lf.boundary_frame(None, b), b)
    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    n_trees = xgb_trees(tr, bounds, cols)
    p_xgb = xgb_fit_predict(tr, va, cols, n_trees)
    lgb_val = pl.read_parquet(LGB_VAL)
    p_lgb = pl.DataFrame({"rowid": va["rowid"].astype(str).to_numpy()}).join(lgb_val, on="rowid", how="left")[
        "predicted_probability"].to_numpy()

    results = {"lightgbm_step2": lf.evaluate("LightGBM step 2 (saved)", va, p_lgb, split, eval_frame),
               "xgboost": {**lf.evaluate("XGBoost", va, p_xgb, split, eval_frame), "trees": n_trees},
               "blend_50_50": lf.evaluate("BLEND 50/50 rank average", va, 0.5 * rank(p_lgb) + 0.5 * rank(p_xgb),
                                          split, eval_frame),
               "rank_corr": float(np.corrcoef(rank(p_lgb), rank(p_xgb))[0, 1]),
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    lf.log(f"rank correlation LightGBM vs XGBoost: {results['rank_corr']:.4f}")
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": p_xgb}).to_parquet(
        "data/validation/safe_v2_step3_xgb_val_c4.parquet", index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    """XGBoost on all of train, blended 50/50 with submissions/safe_v2_step2.csv."""
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    train_all = lf.training_windows(None, bounds)
    frames = [train_all[train_all["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, test = build("full", bounds, frames, lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT,
                             target_events=load_events("eval"), target_cands=load_candidates("eval"))
    tr, te = lf.crossfit_choice(train_base, test)
    cols = lf.model_columns(tr)
    n_trees = xgb_trees(tr, bounds, cols)
    p_xgb = xgb_fit_predict(tr, te, cols, n_trees)
    lgb_sub = pd.read_csv(lf.REPO / "submissions" / "safe_v2_step2.csv", dtype={"rowid": str})
    rowids = te["rowid"].astype(str).to_numpy()
    p_lgb = pd.DataFrame({"rowid": rowids}).merge(lgb_sub, on="rowid", how="left")["predicted_probability"].to_numpy()
    assert not np.isnan(p_lgb).any(), "step 2 submission is missing rows"
    out = lf.REPO / "submissions" / "safe_v2_step3_blend.csv"
    pd.DataFrame({"rowid": rowids, "predicted_probability": 0.5 * rank(p_lgb) + 0.5 * rank(p_xgb)}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} XGBoost trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv[1:] else validate()
