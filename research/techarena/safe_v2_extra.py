"""Stage 3 (5-seed sequence features) + the small wins from the detector tests: slate, item-trend and transition
features (signal_detectors.py options 2-4) and EASE scores (ease_detector.py).

Usage (from the repo root):
    python research/techarena/safe_v2_extra.py            # validation (baseline: stage 3 with 5 seeds = 0.805634)
    python research/techarena/safe_v2_extra.py --submit   # submissions/safe_v2_extra.csv
    python research/techarena/safe_v2_extra.py --quick    # smoke test: 10% of users, small models

Every extra feature for a window (and for the validation or evaluation targets) uses only data from before that
boundary, or the target event's own slate. EASE uses lambda 800, chosen on training window 5 in ease_detector.py.
Decision rule, fixed before seeing the result: build the submission only if validation gains at least +0.0005.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--submit", action="store_true")
ap.add_argument("--quick", action="store_true")
ARGS = ap.parse_args()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import signal_detectors as sd  # noqa: E402  (reads --quick from the command line, then clears it)
import ease_detector as ed  # noqa: E402
import safe_v2_stage3 as s3  # noqa: E402  (Optuna parameters are already loaded into lf.LGB_PARAMS)
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

lf = sd.lf
s3.SEQ_DIR = HERE / "cache" / "seq" / "stage3_s5"
NAME = "safe_v2_extra_quick" if ARGS.quick else "safe_v2_extra"
OUT = HERE / "outputs" / f"{NAME}.json"
BASELINE = 0.805634  # stage 3 with 5-seed sequence features
MIN_GAIN = 0.0005
LAMBDA = 800.0
EXTRA = sd.OPT2 + sd.OPT3 + sd.OPT4 + ["ease_all", "ease_last20", "ease_all_rank", "ease_all_gap"]


def extra_features(frame: pd.DataFrame, b: int, keys: pl.DataFrame, cands, clicks, n_users, n_items) -> pd.DataFrame:
    """The extra columns for one window (or the target frame), from data before b; rows keyed by rowid."""
    rid = pl.DataFrame({"rowid": frame["rowid"].astype(str).to_numpy()})
    target = keys.join(rid, on="rowid")
    assert target.height == rid.height and target.height % 10 == 0, f"boundary {b}: incomplete events"
    f = sd.window_features(target, b, cands, clicks, n_users, n_items)
    e = ed.ease_features(target, b, clicks, n_users, n_items, (LAMBDA,))
    f = f.merge(e.drop(columns="event_id"), on="rowid")
    g = f.groupby("event_id")["ease_all"]
    f["ease_all_rank"] = g.rank(ascending=False, method="average")
    f["ease_all_gap"] = f["ease_all"] - g.transform("max")
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(f[["rowid"] + EXTRA], on="rowid", how="left")
    assert len(out) == len(frame) and out["slate_cos_click"].notna().all(), f"boundary {b}: extra features missing"
    return out


def with_extra(train_base: pd.DataFrame, bounds: list[int], target: pd.DataFrame, target_b: int, keys, target_keys,
               cands, clicks, n_users, n_items):
    parts = []
    for wi, b in enumerate(bounds):
        parts.append(extra_features(train_base[train_base["window_id"] == wi].reset_index(drop=True), b, keys, cands,
                                    clicks, n_users, n_items))
        lf.log(f"extra features for window {wi}")
    return pd.concat(parts, ignore_index=True), extra_features(target, target_b, target_keys, cands, clicks, n_users,
                                                               n_items)


def subsample(train_base: pd.DataFrame, target: pd.DataFrame):
    """--quick: 10% of users."""
    return train_base[train_base["user_id"] % 10 == 0].reset_index(drop=True), target


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, val = s3.build("val", bounds, frames, lf.boundary_frame(None, b), b)
    train_base, val = s3.with_seq(train_base, bounds, val, f"b{b}")
    if ARGS.quick:
        train_base, val = subsample(train_base, val)
    cands, clicks, n_users, n_items = sd.load_logs()
    keys = cands.select("rowid", "event_id", "u", "i")
    train_base, val = with_extra(train_base, bounds, val, b, keys, keys, cands, clicks, n_users, n_items)

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    tr, va = (train_base, val) if ARGS.quick else lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    assert set(EXTRA) <= set(cols), "extra features not picked up"
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    if ARGS.quick:
        lf.MAX_TREES = 150
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STAGE 3 + small wins (leak-free)", va, pred, split, eval_frame)
    gain = res["local_auc"] - BASELINE
    decision = "SUBMIT" if gain >= MIN_GAIN else "DON'T SUBMIT"
    lf.log(f"change vs stage 3 with 5 seeds ({BASELINE}): {gain:+.6f} -> {decision} (needs +{MIN_GAIN})")

    imp = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -imp[c])
    results = {**res, "trees": n_trees, "delta_vs_stage3_s5": gain, "decision": decision,
               "extra_feature_ranks": {c: order.index(c) + 1 for c in EXTRA},
               "top_features_by_gain": [(c, round(float(imp[c]))) for c in order[:15]],
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    if not ARGS.quick:
        pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
            f"data/validation/{NAME}_val_c4.parquet", index=False)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    train_all = lf.training_windows(None, bounds)
    frames = [train_all[train_all["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, test = s3.build("full", bounds, frames, lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT,
                                target_events=load_events("eval"), target_cands=load_candidates("eval"))
    train_base, test = s3.with_seq(train_base, bounds, test, "full_eval")
    cands, clicks, n_users, n_items, eval_keys = sd.load_logs(with_eval=True)
    keys = cands.select("rowid", "event_id", "u", "i")
    train_base, test = with_extra(train_base, bounds, test, lf.CUT, keys, eval_keys, cands, clicks, n_users, n_items)
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
