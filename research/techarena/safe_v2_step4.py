"""Step 4: step 2 + two wider choice models (12 and 16 dims) as extra features, minus the useless hist_n_cos_gt* counts.

Usage (from the repo root, after safe_v2_step2.py):
    python research/techarena/safe_v2_step4.py            # validation (baseline: step 2 = 0.804154)
    python research/techarena/safe_v2_step4.py --submit   # submissions/safe_v2_step4.csv

Why: a 16-dim choice model (correlation 0.90 with the 11-dim one) carries taste information the step 2 stack lacks.
Each extra model is the v2 configuration with a different width, trained per window only on events before that
window's boundary (same leak-free protocol as cmf_features), and adds prob / no-click / rank columns.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_choicemf import cmf_features  # noqa: E402
from safe_v2_step1 import CACHE, CMF2_ARGS, CMF2_EPOCHS  # noqa: E402
from safe_v2_step2 import build as build_step2  # noqa: E402
from src.choice_mf import code_maps  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / "safe_v2_step4.json"
BASELINE = 0.804154
WIDTHS = (12, 16)
DROP = [f"hist_n_cos_gt{t}" for t in (5, 7, 9)]


def wide_cmf(frame: pd.DataFrame, boundary: int, events, cands, umap, imap, scope: str, **target) -> pd.DataFrame:
    """Attach cmf{w}_prob / cmf{w}_noclick / cmf{w}_rank for each extra width, trained only before `boundary`."""
    for w in WIDTHS:
        args = SimpleNamespace(**{**vars(CMF2_ARGS), "dim": w, "svd_rank": w})
        f = cmf_features(frame[["rowid"]], boundary, events, cands, umap, imap, CACHE / f"choicemf_d{w}" / scope,
                         args=args, epochs=CMF2_EPOCHS, **target)
        f = f.rename(columns={c: c.replace("cmf_", f"cmf{w}_") for c in ("cmf_prob", "cmf_noclick", "cmf_rank")})
        frame = frame.assign(rowid=frame["rowid"].astype(str)).merge(f, on="rowid", how="left")
    return frame


def build(scope: str, bounds: list[int], frames: list[pd.DataFrame], target_frame: pd.DataFrame, target_boundary: int,
          **target):
    train_base, tgt = build_step2(scope, bounds, frames, target_frame, target_boundary, **target)
    torch.set_num_threads(12)
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()
    parts = [wide_cmf(train_base[train_base["window_id"] == wi].reset_index(drop=True), b, events, cands, umap, imap,
                      scope) for wi, b in enumerate(bounds)]
    tgt = wide_cmf(tgt, target_boundary, events, cands, umap, imap, scope, **target)
    return pd.concat(parts, ignore_index=True).drop(columns=DROP), tgt.drop(columns=DROP)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, val = build("val", bounds, frames, lf.boundary_frame(None, b), b)

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    results = {f"cmf{w}_alone": lf.evaluate(f"choice MF {w} dims alone", val, val[f"cmf{w}_prob"].to_numpy(), split,
                                            eval_frame) for w in WIDTHS}

    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 4 (leak-free)", va, pred, split, eval_frame)
    lf.log(f"change vs step 2 ({BASELINE}): {res['local_auc'] - BASELINE:+.6f}")

    gain = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -gain[c])
    new = [c for c in cols if c.startswith(tuple(f"cmf{w}_" for w in WIDTHS))]
    results["step4"] = {**res, "trees": n_trees, "delta_vs_step2": res["local_auc"] - BASELINE,
                        "top_features_by_gain": [(c, round(float(gain[c]))) for c in order[:15]],
                        "new_feature_ranks": {c: order.index(c) + 1 for c in new}}
    results["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        "data/validation/safe_v2_step4_val_c4.parquet", index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    train_all = lf.training_windows(None, bounds)
    frames = [train_all[train_all["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, test = build("full", bounds, frames, lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT,
                             target_events=load_events("eval"), target_cands=load_candidates("eval"))
    tr, te = lf.crossfit_choice(train_base, test)
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)
    out = lf.REPO / "submissions" / "safe_v2_step4.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv[1:] else validate()
