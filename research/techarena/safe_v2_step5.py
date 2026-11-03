"""Step 5: stage 3 (5-seed sequence features) + the small wins (safe_v2_extra.py) + a choice model whose auxiliary
task contrasts each past click with items drawn like the slates' fillers (src/choice_mf.py --aux-pop 1).

Usage (from the repo root):
    python research/techarena/safe_v2_step5.py            # validation (baseline: stage 3 with 5 seeds = 0.805634)
    python research/techarena/safe_v2_step5.py --submit   # submissions/safe_v2_step5.csv
    python research/techarena/safe_v2_step5.py --quick    # smoke test: 10% of users, small models

Why: the non-clicked candidates of clicked and no-click events are statistically identical, so every slate looks like
popularity-matched filler (+ the real next item in clicked events). The v2 choice model's auxiliary task used UNIFORM
random items as contrast, which teaches "popular = clicked", a signal the fillers cancel. Drawn like the fillers
(weight 1, 9 contrast items, 4 per event: best inner holdout of 5 variants, logs/cmf_test_*.log), the choice model
alone improved from 0.788006 to 0.789461 on validation. Its features (cmfp_prob / cmfp_noclick / cmfp_rank) are trained per window on events before
each boundary, the same leak-free protocol as the other choice-model features.
Decision rule, fixed before seeing the result: build the submission only if validation gains at least +0.0005.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--submit", action="store_true")
ap.add_argument("--quick", action="store_true")
ARGS = ap.parse_args()
sys.argv = sys.argv[:1] + (["--quick"] if ARGS.quick else [])  # safe_v2_extra parses the command line on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safe_v2_extra as ex  # noqa: E402  (stage 3 + small wins; loads the Optuna parameters)
from safe_v2_choicemf import cmf_features  # noqa: E402
from safe_v2_step1 import CACHE, CMF2_ARGS  # noqa: E402
from src.choice_mf import code_maps  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

lf, sd, s3 = ex.lf, ex.sd, ex.s3
NAME = "safe_v2_step5_quick" if ARGS.quick else "safe_v2_step5"
OUT = HERE / "outputs" / f"{NAME}.json"
BASELINE = 0.805634  # stage 3 with 5-seed sequence features
MIN_GAIN = 0.0005
POP_ARGS = SimpleNamespace(**{**vars(CMF2_ARGS), "aux_pop": 1, "aux_weight": 1.0, "aux_neg": 9, "aux_ratio": 4})
POP_EPOCHS = 3  # best inner-holdout epoch of src/choice_mf.py with these settings
POP_COLS = ["cmfp_prob", "cmfp_noclick", "cmfp_rank"]


def pop_cmf(frame: pd.DataFrame, b: int, events, cands, umap, imap, scope: str, **target) -> pd.DataFrame:
    f = cmf_features(frame[["rowid"]], b, events, cands, umap, imap, CACHE / "choicemf_pop" / scope, args=POP_ARGS,
                     epochs=POP_EPOCHS, **target)
    f = f.rename(columns={c: c.replace("cmf_", "cmfp_") for c in ("cmf_prob", "cmf_noclick", "cmf_rank")})
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(f, on="rowid", how="left")
    assert len(out) == len(frame) and out["cmfp_prob"].notna().all(), f"boundary {b}: pop choice features missing"
    return out


def with_pop(train_base, bounds, target, target_b, scope, **target_tables):
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()
    parts = [pop_cmf(train_base[train_base["window_id"] == wi].reset_index(drop=True), b, events, cands, umap, imap,
                     scope) for wi, b in enumerate(bounds)]
    return pd.concat(parts, ignore_index=True), pop_cmf(target, target_b, events, cands, umap, imap, scope,
                                                        **target_tables)


def dump(frame: pd.DataFrame, name: str) -> None:
    """Save a final frame to cache/lab/; float64 columns become float32, one column at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    lab = CACHE / "lab"
    lab.mkdir(parents=True, exist_ok=True)
    arrays = [pa.array(frame[c].to_numpy(np.float32) if frame[c].dtype == np.float64 else frame[c].to_numpy())
              for c in frame.columns]
    pq.write_table(pa.Table.from_arrays(arrays, names=list(frame.columns)), lab / f"{name}.parquet")
    lf.log(f"dumped {name} ({len(frame):,} rows) to {lab}")


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
    train_base, val = with_pop(train_base, bounds, val, b, "val")  # before subsampling, so the cache holds every user
    if ARGS.quick:
        train_base, val = ex.subsample(train_base, val)
    cands, clicks, n_users, n_items = sd.load_logs()
    keys = cands.select("rowid", "event_id", "u", "i")
    train_base, val = ex.with_extra(train_base, bounds, val, b, keys, keys, cands, clicks, n_users, n_items)

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    tr, va = (train_base, val) if ARGS.quick else lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    assert set(ex.EXTRA + POP_COLS) <= set(cols), "new features not picked up"
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    if ARGS.quick:
        lf.MAX_TREES = 150
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 5: stage 3 + small wins + filler-matched choice model (leak-free)", va, pred, split,
                      eval_frame)
    gain = res["local_auc"] - BASELINE
    decision = "SUBMIT" if gain >= MIN_GAIN else "DON'T SUBMIT"
    lf.log(f"change vs stage 3 with 5 seeds ({BASELINE}): {gain:+.6f} -> {decision} (needs +{MIN_GAIN})")
    alone = lf.evaluate("filler-matched choice model alone", val, val["cmfp_prob"].to_numpy(), split, eval_frame)

    imp = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -imp[c])
    results = {**res, "trees": n_trees, "delta_vs_stage3_s5": gain, "decision": decision, "cmfp_alone": alone,
               "new_feature_ranks": {c: order.index(c) + 1 for c in ex.EXTRA + POP_COLS},
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
    target_tables = {"target_events": load_events("eval"), "target_cands": load_candidates("eval")}
    train_base, test = s3.build("full", bounds, frames, lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT,
                                **target_tables)
    train_base, test = s3.with_seq(train_base, bounds, test, "full_eval")
    train_base, test = with_pop(train_base, bounds, test, lf.CUT, "full", **target_tables)
    cands, clicks, n_users, n_items, eval_keys = sd.load_logs(with_eval=True)
    keys = cands.select("rowid", "event_id", "u", "i")
    train_base, test = ex.with_extra(train_base, bounds, test, lf.CUT, keys, eval_keys, cands, clicks, n_users,
                                     n_items)
    tr, te = lf.crossfit_choice(train_base, test)
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    try:  # frames for lab variants (lab_submit.py --base step5); never at the submission's expense
        dump(tr, "step5_full_train")
        dump(te, "step5_full_test")
    except Exception as err:  # noqa: BLE001
        lf.log(f"frame dump skipped ({type(err).__name__}: {err})")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)
    out = lf.REPO / "submissions" / f"{NAME}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if ARGS.submit else validate()
