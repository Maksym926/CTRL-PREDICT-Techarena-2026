"""Optuna re-tuning of the binary LightGBM on step 7's frames (step 5 + 50-click sequence features), leak-free.

The current parameters (outputs/optuna_step4.json) were tuned on step 4's 109 features; step 7 has 140, led by the
sequence-model columns. Same protocol as optuna_step4.py: each trial fits on the training windows dated before the
latest window and scores that window (the inner holdout) at learning rate 0.05 with early stopping; validation is never
seen during the search. Trial 0 is the current parameters. The binned training set is built once and reused
(feature_pre_filter off, so min_child_samples can change between trials).

Usage (from the repo root):
    python research/techarena/optuna_step7.py --seq stage3_h50_s30 [--trials 40] [--minutes 100]
Then score the best parameters once on validation with the usual protocol (learning rate 0.03, inner-holdout trees,
3 seeds): python research/techarena/safe_v2_step7.py --seq stage3_h50_s30 --params research/techarena/outputs/optuna_step7.json --tag _s30_opt7
Output: outputs/optuna_step7.json (best_params, every trial).
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

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="stage3_h50_s30", help="folder under cache/seq with the 50-click features")
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--minutes", type=float, default=100)
A = ap.parse_args()
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root)

OUT = HERE / "outputs" / "optuna_step7.json"
SEARCH_LR = 0.05
LAB = HERE / "cache" / "lab"
NEW = ["seq50_prob", "seq50_noclick", "seq50_rank"]


def load_train() -> pd.DataFrame:
    """step6_train without step 6's columns, plus seq50_* per training window (as safe_v2_step7.py)."""
    f = pd.read_parquet(LAB / "step6_train.parquet")
    f = f.drop(columns=[c for c in f.columns if c.endswith(("_best_other", "_margin"))
                        or c in ("choice_first_votes", "choice_mean_rank")])
    parts = []
    for wi, b in enumerate(lf.bounds_for(lf.VAL_DAY, half=True)):
        w = f[f["window_id"] == wi].reset_index(drop=True)
        s = pd.read_parquet(HERE / "cache" / "seq" / A.seq / f"seq_b{b}.parquet")
        s = s.assign(rowid=s["rowid"].astype(str)).rename(columns=dict(zip(["seq_prob", "seq_noclick", "seq_rank"], NEW)))
        w = w.assign(rowid=w["rowid"].astype(str)).merge(s, on="rowid", how="left")
        assert not w[NEW].isna().any().any(), f"window {wi}: rows without sequence features"
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    t0 = time.perf_counter()
    current = json.loads((HERE / "outputs" / "optuna_step4.json").read_text())["best_params"]
    tr = load_train()
    cols = lf.model_columns(tr)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    last = len(bounds) - 1
    inner = (tr["window_id"] < last) & (tr["timestamp"] < bounds[last])
    hold = tr["window_id"] == last
    lf.log(f"{len(cols)} features; inner holdout {inner.sum():,} train / {hold.sum():,} holdout rows")
    dtrain = lgb.Dataset(tr.loc[inner, cols].to_numpy(np.float32), tr.loc[inner, "label"].to_numpy(),
                         params={"max_bin": lf.LGB_PARAMS["max_bin"], "feature_pre_filter": False}, free_raw_data=False)
    dvalid = lgb.Dataset(tr.loc[hold, cols].to_numpy(np.float32), tr.loc[hold, "label"].to_numpy(), reference=dtrain)
    del tr
    base = {"objective": "binary", "metric": "auc", "learning_rate": SEARCH_LR, "subsample_freq": 1,
            "max_bin": lf.LGB_PARAMS["max_bin"], "n_jobs": -1, "verbosity": -1, "seed": lf.FULL_SEEDS[0],
            "feature_pre_filter": False}

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 3000, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        )
        b = lgb.train({**base, **p}, dtrain, num_boost_round=lf.MAX_TREES, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
        auc = float(b.best_score["valid_0"]["auc"])
        trial.set_user_attr("trees", int(b.best_iteration))
        lf.log(f"  trial {trial.number}: inner AUC {auc:.6f}, {b.best_iteration} trees, {p}")
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(current)  # trial 0 = the current parameters, the reference point
    study.optimize(objective, n_trials=A.trials, timeout=A.minutes * 60)
    ref = study.trials[0].value
    lf.log(f"search done: {len(study.trials)} trials, current params {ref:.6f} -> best {study.best_value:.6f} "
           f"({study.best_value - ref:+.6f}) {study.best_params}")
    OUT.write_text(json.dumps({
        "seq": A.seq, "inner_auc_current": ref, "inner_auc_best": study.best_value, "best_params": study.best_params,
        "n_trials": len(study.trials), "minutes": round((time.perf_counter() - t0) / 60, 1),
        "trials": [{"n": t.number, "auc": t.value, "trees": t.user_attrs.get("trees"), **t.params}
                   for t in study.trials if t.value is not None]}, indent=2))
    lf.log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
