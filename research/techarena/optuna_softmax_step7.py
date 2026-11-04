"""Optuna tuning of the softmax-objective partner (softmax_step7.py) on step 7's frames, leak-free.

The partner ran on the SAFE_v2 default parameters until 4 Nov; step 4's tuned binary parameters alone lifted it from
0.807347 to 0.807807 on validation, so its own search may find more. Protocol as optuna_step7.py: each trial fits the
event-softmax objective (softmax_objective.py) on the training windows dated before the latest window, early-stopped
on that window's AUC (the inner holdout) at learning rate 0.05; validation is never seen. Trial 0 = step 4's tuned
parameters (the current partner), trial 1 = the old defaults.

Usage (from the repo root):
    python research/techarena/optuna_softmax_step7.py --seq stage3_h50_s30 [--trials 15] [--minutes 80]
Then: python research/techarena/softmax_step7.py --seq stage3_h50_s30 --params research/techarena/outputs/optuna_softmax_step7.json --tag _s30_optsm
Output: outputs/optuna_softmax_step7.json (best_params, every trial).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="stage3_h50_s30", help="folder under cache/seq with the 50-click features")
ap.add_argument("--trials", type=int, default=15)
ap.add_argument("--minutes", type=float, default=80)
ap.add_argument("--threads", type=int, default=-1)
A = ap.parse_args()
sys.argv = [sys.argv[0], "--threads", str(A.threads)]  # softmax_objective parses the command line on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import softmax_objective as so  # noqa: E402

lf = so.lf
OUT = HERE / "outputs" / "optuna_softmax_step7.json"
SEARCH_LR = 0.05
LAB = HERE / "cache" / "lab"
NEW = ["seq50_prob", "seq50_noclick", "seq50_rank"]
DEFAULTS = {k: lf.LGB_PARAMS[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample", "reg_lambda")}


def load_train() -> pd.DataFrame:
    """step6_train without step 6's columns, plus seq50_*, sorted into blocks of 10 rows (as softmax_step7.py)."""
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
    f = pd.concat(parts, ignore_index=True)
    f = f.sort_values(["window_id", "event_index", "candidate_index"], kind="stable").reset_index(drop=True)
    blk = f["event_index"].to_numpy().reshape(-1, 10)
    assert (blk == blk[:, :1]).all(), "events are not blocks of 10 rows"
    return f


def main() -> None:
    t0 = time.perf_counter()
    tuned = json.loads((HERE / "outputs" / "optuna_step4.json").read_text())["best_params"]
    tr = load_train()
    cols = lf.model_columns(tr)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    last = len(bounds) - 1
    inner = (tr["window_id"] < last) & (tr["timestamp"] < bounds[last])
    hold = tr["window_id"] == last
    Xi, yi = tr.loc[inner, cols].to_numpy(np.float32), tr.loc[inner, "label"].to_numpy(np.float64)
    Xh, yh = tr.loc[hold, cols].to_numpy(np.float32), tr.loc[hold, "label"].to_numpy(np.float64)
    del tr
    lf.log(f"{len(cols)} features; inner holdout {len(yi):,} train / {len(yh):,} holdout rows")
    base_params = dict(lf.LGB_PARAMS)

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
        lf.LGB_PARAMS.clear()
        lf.LGB_PARAMS.update({**base_params, **p, "learning_rate": SEARCH_LR})
        b = so.fit("softmax", Xi, yi, lf.MAX_TREES, valid=(Xh, yh))
        auc = float(b.best_score["valid_0"]["auc"])
        trial.set_user_attr("trees", int(b.best_iteration))
        lf.log(f"  trial {trial.number}: inner AUC {auc:.6f}, {b.best_iteration} trees, {p}")
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(tuned)  # trial 0 = the current partner (step 4's parameters)
    study.enqueue_trial({**DEFAULTS, "reg_alpha": 1e-3, "min_split_gain": 0.0})  # trial 1 = the old defaults
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
