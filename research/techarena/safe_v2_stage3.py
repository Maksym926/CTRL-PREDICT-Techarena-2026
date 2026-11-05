"""Stage 3: step 4 + Optuna parameters + sequence-model features (seq_prob / seq_noclick / seq_rank) per window.

Usage (from the repo root, after the Kaggle run: unzip seq_stage3.zip into research/techarena/cache/seq/<name>/):
    python research/techarena/safe_v2_stage3.py [--seq stage3]            # validation (baseline: step 4 + Optuna)
    python research/techarena/safe_v2_stage3.py [--seq stage3] --submit   # submissions/safe_v2_<name>.csv
--seq names the folder under cache/seq and the outputs: stage3 = 1 seed (local 0.805031), stage3_s5 = 5-seed average.

Each window's sequence features come from a model trained only on events before that window's boundary
(seq_model.py --mode windows), the same leak-free rule as the choice-model features. Decision rule, fixed before
seeing the result: submit only if validation gains at least +0.0005 over step 4 + Optuna.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_step4 import build  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="stage3")
ap.add_argument("--submit", action="store_true")
ARGS = ap.parse_args()
NAME = f"safe_v2_{ARGS.seq}"
SEQ_DIR = HERE / "cache" / "seq" / ARGS.seq
OUT = HERE / "outputs" / f"{NAME}.json"
BASELINE = 0.804643  # step 4 + Optuna
STAGE3_1SEED = 0.805031
MIN_GAIN = 0.0005
SEQ_COLS = ["seq_prob", "seq_noclick", "seq_rank"]
lf.LGB_PARAMS.update(json.loads((HERE / "outputs" / "optuna_step4.json").read_text())["best_params"])


def add_seq(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    s = pd.read_parquet(SEQ_DIR / f"seq_{name}.parquet").assign(rowid=lambda x: x["rowid"].astype(str))
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(s, on="rowid", how="left")
    if out[SEQ_COLS].isna().any().any():
        raise SystemExit(f"seq_{name}: {out['seq_prob'].isna().sum():,} rows without sequence features")
    return out


def with_seq(train_base: pd.DataFrame, bounds: list[int], target: pd.DataFrame, target_name: str):
    parts = [add_seq(train_base[train_base["window_id"] == wi].reset_index(drop=True), f"b{b}")
             for wi, b in enumerate(bounds)]
    return pd.concat(parts, ignore_index=True), add_seq(target, target_name)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, val = build("val", bounds, frames, lf.boundary_frame(None, b), b)
    train_base, val = with_seq(train_base, bounds, val, f"b{b}")

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    seq_alone = lf.evaluate("sequence model alone (window fit)", val, val["seq_prob"].to_numpy(), split, eval_frame)

    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    assert set(SEQ_COLS) <= set(cols), "sequence features not picked up"
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STAGE 3 (leak-free)", va, pred, split, eval_frame)
    gain = res["local_auc"] - BASELINE
    lf.log(f"change vs step 4 + Optuna ({BASELINE}): {gain:+.6f} -> "
           + ("SUBMIT (gain >= +0.0005)" if gain >= MIN_GAIN else "DON'T SUBMIT (gain < +0.0005)"))
    lf.log(f"change vs stage 3 with 1 seed ({STAGE3_1SEED}): {res['local_auc'] - STAGE3_1SEED:+.6f}")

    imp = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -imp[c])
    results = {"seq_alone": seq_alone, "stage3": {**res, "trees": n_trees, "delta_vs_step4_optuna": gain,
                                                  "seq_feature_ranks": {c: order.index(c) + 1 for c in SEQ_COLS},
                                                  "top_features_by_gain": [(c, round(float(imp[c]))) for c in order[:15]]},
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        f"data/validation/{NAME}_val_c4.parquet", index=False)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    train_all = lf.training_windows(None, bounds)
    frames = [train_all[train_all["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, test = build("full", bounds, frames, lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT,
                             target_events=load_events("eval"), target_cands=load_candidates("eval"))
    train_base, test = with_seq(train_base, bounds, test, "full_eval")
    tr, te = lf.crossfit_choice(train_base, test)
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)
    out = lf.REPO / "submissions" / f"{NAME}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if ARGS.submit else validate()
