"""Optuna tuning of LightGBM on the step 4 features, leak-free.

Usage (from the repo root, after safe_v2_step4.py has filled its caches):
    python research/techarena/optuna_step4.py [--trials 40] [--minutes 90]

Each trial fits on the training windows dated before the latest window and scores that window (the inner holdout, as in
lf.inner_holdout_trees), at learning rate 0.05 with early stopping. Validation is never seen during the search. The best
parameters then run once through the usual protocol (learning rate 0.03, inner-holdout tree count, 3 seeds) and are
scored on validation. Best parameters go to outputs/optuna_step4.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_step4 import BASELINE as STEP2, build  # noqa: E402
from src.data import load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / "optuna_step4.json"
STEP4 = 0.804574
SEARCH_LR = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=90)
    args = ap.parse_args()
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

    last = len(bounds) - 1
    inner_tr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    inner_va = tr[tr["window_id"] == last]
    fixed_params = dict(objective="binary", metric="auc", max_bin=lf.LGB_PARAMS["max_bin"], subsample_freq=1,
                        n_jobs=-1, verbosity=-1, seed=lf.FULL_SEEDS[0])
    dtr = lgb.Dataset(inner_tr[cols], inner_tr["label"], params={"max_bin": fixed_params["max_bin"]},
                      free_raw_data=False)
    dva = lgb.Dataset(inner_va[cols], inner_va["label"], reference=dtr, free_raw_data=False)
    lf.log(f"search: {len(inner_tr):,} train / {len(inner_va):,} inner-holdout rows, {len(cols)} features")

    def fit(params: dict) -> tuple[float, int]:
        m = lgb.train({**fixed_params, "learning_rate": SEARCH_LR, **params}, dtr, num_boost_round=lf.MAX_TREES,
                      valid_sets=[dva], callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
        return m.best_score["valid_0"]["auc"], m.best_iteration

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 3000, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        )
        auc, trees = fit(params)
        trial.set_user_attr("trees", trees)
        lf.log(f"trial {trial.number:3d}: inner AUC {auc:.6f} ({trees} trees) {params}")
        return auc

    current = {k: lf.LGB_PARAMS[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample",
                                             "reg_lambda")} | {"reg_alpha": 1e-3, "min_split_gain": 0.0}  # 1e-3 = bottom of the log range (current: 0)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(current)  # trial 0 = the current parameters, the reference point
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, timeout=args.minutes * 60)

    ref = study.trials[0].value
    best = study.best_params
    lf.log(f"search done: {len(study.trials)} trials, current params {ref:.6f} -> best {study.best_value:.6f} "
           f"({study.best_value - ref:+.6f}) {best}")

    # final: usual protocol with the best parameters, scored once on validation
    lf.LGB_PARAMS.update(best)
    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 4 + Optuna (leak-free)", va, pred, split, eval_frame)
    lf.log(f"change vs step 4 ({STEP4}): {res['local_auc'] - STEP4:+.6f} | vs step 2 ({STEP2}): "
           f"{res['local_auc'] - STEP2:+.6f}")

    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        "data/validation/optuna_step4_val_c4.parquet", index=False)
    results = {**res, "trees": n_trees, "best_params": best, "inner_auc_current": ref,
               "inner_auc_best": study.best_value, "n_trials": len(study.trials),
               "delta_vs_step4": res["local_auc"] - STEP4,
               "trials": [{"n": t.number, "auc": t.value, "trees": t.user_attrs.get("trees"), **t.params}
                          for t in study.trials if t.value is not None],
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
