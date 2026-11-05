"""Leak-free SAFE_v2 + choice-MF features, scored on the team validation split.

Usage (from the repo root, after leakfree_safe_v2.py, and for --submit leakfree_safe_v2.py --submit, cached features):
    python research/techarena/safe_v2_choicemf.py            # validation score (~20 min)
    python research/techarena/safe_v2_choicemf.py --submit   # submissions/safe_v2_cmf.csv (~60 min)

For every training window (and for validation) a choice-MF model (src/choice_mf.py, settings picked on its inner
holdout) is trained ONLY on events before that window's boundary and scores the window's target events. Its outputs
become three extra LightGBM features; everything else is the fixed protocol of leakfree_safe_v2.py, whose baseline
local AUC on the same rows is 0.791945.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from src.choice_mf import click_pairs, click_svd, code_maps, encode, predict, strip_time, train  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

CMF_ARGS = SimpleNamespace(dim=32, svd_rank=11, scale=4.0, l2=0.1, lr=3e-3, batch=1024, seed=0)
CMF_EPOCHS = 2  # best epoch on the choice-MF inner holdout for every l2 tried
CMF_CACHE = HERE / "cache" / "choicemf"
OUT = HERE / "outputs" / "safe_v2_choicemf.json"
BASELINE = 0.791945


def cmf_features(frame: pd.DataFrame, boundary: int, events: pl.DataFrame, cands: pl.DataFrame,
                 umap: dict, imap: dict, cache: Path = CMF_CACHE, target_events: pl.DataFrame | None = None,
                 target_cands: pl.DataFrame | None = None, args: SimpleNamespace = CMF_ARGS,
                 epochs: int = CMF_EPOCHS) -> pd.DataFrame:
    """Attach choice-MF features to `frame`, using a model trained only on train events before `boundary`.
    The frame's rows are looked up in target_events/target_cands (default: the train tables)."""
    target_events = events if target_events is None else target_events
    target_cands = cands if target_cands is None else target_cands
    path = cache / f"{boundary}.parquet"
    if path.exists():
        feats = pd.read_parquet(path)
    else:
        start = time.perf_counter()
        rowids = frame["rowid"].astype(str).unique().tolist()  # plain list: polars rejects pandas' Arrow string array
        target_ids = pl.DataFrame({"rowid": rowids}).join(target_cands, on="rowid").select("event_id")
        target = encode(target_events.join(target_ids.unique(), on="event_id", how="semi"), target_cands, umap, imap)
        fit = encode(events.filter(pl.col("timestamp") < boundary), cands, umap, imap)
        if getattr(args, "no_time", 0):
            fit, target = strip_time(fit), strip_time(target)
        clicks = click_pairs(boundary, umap, imap) if getattr(args, "aux_weight", 0) > 0 else None
        model, _ = train(fit, len(umap), len(imap), args, epochs, click_svd(boundary, umap, imap, args.svd_rank),
                         clicks=clicks)
        p = predict(model, target).reshape(-1, 10)
        feats = pd.DataFrame({"rowid": target["rowid"].reshape(-1), "cmf_prob": p.reshape(-1),
                              "cmf_noclick": np.repeat(1.0 - p.sum(1), 10),
                              "cmf_rank": ((-p).argsort(1).argsort(1) + 1).reshape(-1).astype(np.float32)})
        cache.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path, index=False)
        lf.log(f"choice MF for boundary day {(boundary - lf.D0) / lf.DAY:.1f}: trained on {len(fit['user']):,} events, "
               f"scored {len(target['user']):,} in {time.perf_counter() - start:.0f} s")
    # Left merge keeps row order (events stay contiguous); cast both keys so string dtypes match
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(
        feats.assign(rowid=feats["rowid"].astype(str)), on="rowid", how="left")
    assert out["cmf_prob"].notna().all(), "choice-MF features missing for some rows"
    return out


def validate() -> None:
    t0 = time.perf_counter()
    torch.set_num_threads(12)
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()

    # Cached SAFE_v2 frames (built by leakfree_safe_v2.py), fixed protocol: targets end at the validation boundary
    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    parts = [cmf_features(fixed[fixed["window_id"] == wi].reset_index(drop=True), bound, events, cands, umap, imap)
             for wi, bound in enumerate(bounds)]
    train_base = pd.concat(parts, ignore_index=True)
    val = cmf_features(lf.boundary_frame(None, b), b, events, cands, umap, imap)

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    results = {"cmf_alone": lf.evaluate("choice MF alone", val, val["cmf_prob"].to_numpy(), split, eval_frame)}

    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(tr):,} training rows, {len(cols)} features (SAFE_v2 + {[c for c in cols if c.startswith('cmf_')]})")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("SAFE_v2 + choice MF (leak-free)", va, pred, split, eval_frame)
    lf.log(f"change vs SAFE_v2 leak-free baseline ({BASELINE}): {res['local_auc'] - BASELINE:+.6f}")

    gain = sorted(zip(cols, m.booster_.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    results["safe_v2_plus_cmf"] = {**res, "trees": n_trees, "delta_vs_baseline": res["local_auc"] - BASELINE,
                                   "top_features_by_gain": [(c, round(float(g))) for c, g in gain[:15]]}
    results["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        "data/validation/safe_v2_cmf_val_c4.parquet", index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    """Same model on all of train (13 half-day windows), predicting evaluation.csv."""
    t0 = time.perf_counter()
    torch.set_num_threads(12)
    bounds = lf.bounds_for(7, half=True)  # days 0.5 ... 6.5; targets run to the end of train
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()
    cache = CMF_CACHE / "full"  # the validation cache only covers targets before day 4

    train_all = lf.training_windows(None, bounds)  # cached by leakfree_safe_v2.py --submit
    parts = [cmf_features(train_all[train_all["window_id"] == wi].reset_index(drop=True), bound, events, cands,
                          umap, imap, cache) for wi, bound in enumerate(bounds)]
    train_base = pd.concat(parts, ignore_index=True)
    test = cmf_features(lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT, events, cands, umap, imap, cache,
                        target_events=load_events("eval"), target_cands=load_candidates("eval"))

    tr, te = lf.crossfit_choice(train_base, test)
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)

    out = lf.REPO / "submissions" / "safe_v2_cmf.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv[1:] else validate()
