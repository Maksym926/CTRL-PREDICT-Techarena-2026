"""Step 1: SAFE_v2 + improved choice model + top-k history similarity + event-level features (leak-free).

Usage (from the repo root, after leakfree_safe_v2.py has cached its SAFE_v2 frames):
    python research/techarena/safe_v2_step1.py            # validation (baseline: 0.796472 = SAFE_v2 + choice MF v1)
    python research/techarena/safe_v2_step1.py --submit   # submissions/safe_v2_step1.csv

Changes against safe_v2_choicemf.py:
  - choice MF v2: 11 dims, no time inputs, auxiliary task on all past clicks (0.7882 alone vs 0.7858)
  - top-k similarity: mean of the 3 highest rank-11 SVD cosines between the candidate and its history items, and
    counts of history items above 0.5 / 0.7 / 0.9
  - event level: hours since the last click, the user's past events and clicked events (smoothed click rate),
    distinct items clicked
Every feature uses only data from before its window's boundary.
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
from safe_v2_choicemf import cmf_features  # noqa: E402
from src.choice_mf import HISTORY, click_pairs, click_svd, code_maps, encode  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

CMF2_ARGS = SimpleNamespace(dim=11, svd_rank=11, scale=4.0, l2=0.01, lr=3e-3, batch=1024, seed=0, user_emb=1,
                            no_time=1, aux_weight=0.3, aux_ratio=2, aux_neg=20)
CMF2_EPOCHS = 2  # best epoch on the choice-MF inner holdout for this configuration
CACHE = HERE / "cache"
OUT = HERE / "outputs" / "safe_v2_step1.json"
BASELINE = 0.796472
THRESHOLDS = (0.5, 0.7, 0.9)


def extra_features(frame: pd.DataFrame, boundary: int, events: pl.DataFrame, cands: pl.DataFrame, umap: dict,
                   imap: dict, cache: Path, target_events: pl.DataFrame | None = None,
                   target_cands: pl.DataFrame | None = None) -> pd.DataFrame:
    """Top-k history similarity and event-level features for the frame's rows, from data before `boundary`."""
    target_events = events if target_events is None else target_events
    target_cands = cands if target_cands is None else target_cands
    path = cache / f"{boundary}.parquet"
    if path.exists():
        feats = pd.read_parquet(path)
    else:
        start = time.perf_counter()
        rowids = frame["rowid"].astype(str).unique().tolist()
        target_ids = pl.DataFrame({"rowid": rowids}).join(target_cands, on="rowid").select("event_id")
        d = encode(target_events.join(target_ids.unique(), on="event_id", how="semi"), target_cands, umap, imap)
        n, pad = len(d["user"]), len(imap)

        # Top-k similarity between each candidate and the individual history items
        _, items = click_svd(boundary, umap, imap, 11)
        vec = np.vstack([items, np.zeros((1, items.shape[1]), np.float32)])
        sim = np.einsum("ncr,nkr->nck", vec[d["cand"]], vec[d["hist"]])
        valid = (d["hist"] != pad)[:, None, :]
        ranked = -np.sort(-np.where(valid, sim, -np.inf), axis=2)[:, :, :3]
        top = np.where(np.isfinite(ranked), ranked, np.nan)
        cols = {"hist_top3_cos": np.nanmean(top, axis=2) if top.size else top}
        for t in THRESHOLDS:
            cols[f"hist_n_cos_gt{int(t * 10)}"] = ((sim > t) & valid).sum(2).astype(np.float32)

        # Event level: hours since the last click, and the user's past activity before the boundary
        hist_len = np.rint(d["noclick"][:, 0] * HISTORY)
        hours_since = np.where(hist_len > 0, np.expm1(d["noclick"][:, 1]), np.nan)
        past = events.filter(pl.col("timestamp") < boundary).select("event_id", "user_id")
        clicked_events = cands.join(past, on="event_id", how="semi").group_by("event_id").agg(pl.col("clicked").max())
        past = past.join(clicked_events, on="event_id", how="left")
        pu = past["user_id"].replace_strict(umap, return_dtype=pl.Int64).to_numpy()
        n_events = np.bincount(pu, minlength=len(umap))
        n_clicked = np.bincount(pu, weights=past["clicked"].fill_null(0).to_numpy(), minlength=len(umap))
        rate = (n_clicked + 5 * 0.63) / (n_events + 5)  # shrunk towards the global event click rate
        cu, _ = click_pairs(boundary, umap, imap)
        n_items = np.bincount(cu, minlength=len(umap))
        u = d["user"]
        event_cols = {"hours_since_last_click": hours_since, "user_past_events": n_events[u],
                      "user_event_click_rate": rate[u], "user_distinct_clicked": n_items[u]}

        feats = pd.DataFrame({"rowid": d["rowid"].reshape(-1),
                              **{k: v.reshape(-1).astype(np.float32) for k, v in cols.items()},
                              **{k: np.repeat(v, 10).astype(np.float32) for k, v in event_cols.items()}})
        cache.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path, index=False)
        lf.log(f"extra features for boundary day {(boundary - lf.D0) / lf.DAY:.1f}: {n:,} events "
               f"in {time.perf_counter() - start:.0f} s")
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(
        feats.assign(rowid=feats["rowid"].astype(str)), on="rowid", how="left")
    assert out["user_past_events"].notna().all(), "extra features missing for some rows"
    return out


def with_features(frame, boundary, events, cands, umap, imap, scope, **target):
    frame = cmf_features(frame, boundary, events, cands, umap, imap, CACHE / "choicemf_v2" / scope,
                         args=CMF2_ARGS, epochs=CMF2_EPOCHS, **target)
    return extra_features(frame, boundary, events, cands, umap, imap, CACHE / "extra" / scope, **target)


def validate() -> None:
    t0 = time.perf_counter()
    torch.set_num_threads(12)
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()

    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    parts = [with_features(fixed[fixed["window_id"] == wi].reset_index(drop=True), bound, events, cands, umap, imap,
                           "val") for wi, bound in enumerate(bounds)]
    train_base = pd.concat(parts, ignore_index=True)
    val = with_features(lf.boundary_frame(None, b), b, events, cands, umap, imap, "val")

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    results = {"cmf_v2_alone": lf.evaluate("choice MF v2 alone", val, val["cmf_prob"].to_numpy(), split, eval_frame),
               "top3_alone": lf.evaluate("hist_top3_cos alone", val, val["hist_top3_cos"].fillna(-1).to_numpy(),
                                         split, eval_frame)}

    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 1 (leak-free)", va, pred, split, eval_frame)
    lf.log(f"change vs SAFE_v2 + choice MF v1 ({BASELINE}): {res['local_auc'] - BASELINE:+.6f}")

    gain = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    new = [c for c in cols if c.startswith(("hist_top3", "hist_n_cos", "hours_since", "user_past", "user_event",
                                            "user_distinct", "cmf_"))]
    order = sorted(cols, key=lambda c: -gain[c])
    results["step1"] = {**res, "trees": n_trees, "delta_vs_baseline": res["local_auc"] - BASELINE,
                        "top_features_by_gain": [(c, round(float(gain[c]))) for c in order[:15]],
                        "new_feature_rank": {c: order.index(c) + 1 for c in new}}
    results["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        "data/validation/safe_v2_step1_val_c4.parquet", index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    """Same model on all of train (13 half-day windows), predicting evaluation.csv."""
    t0 = time.perf_counter()
    torch.set_num_threads(12)
    bounds = lf.bounds_for(7, half=True)
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()

    train_all = lf.training_windows(None, bounds)
    parts = [with_features(train_all[train_all["window_id"] == wi].reset_index(drop=True), bound, events, cands,
                           umap, imap, "full") for wi, bound in enumerate(bounds)]
    train_base = pd.concat(parts, ignore_index=True)
    test = with_features(lf.boundary_frame(None, lf.CUT, evaluation=True), lf.CUT, events, cands, umap, imap, "full",
                         target_events=load_events("eval"), target_cands=load_candidates("eval"))

    tr, te = lf.crossfit_choice(train_base, test)
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)

    out = lf.REPO / "submissions" / "safe_v2_step1.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv[1:] else validate()
