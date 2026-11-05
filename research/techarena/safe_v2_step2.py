"""Step 2: step 1 + the choice model's taste vectors as LightGBM features (leak-free).

Usage (from the repo root, after safe_v2_step1.py):
    python research/techarena/safe_v2_step2.py            # validation (baseline: step 1 = 0.798967)
    python research/techarena/safe_v2_step2.py --submit   # submissions/safe_v2_step2.csv

Adds 33 features per candidate: the user's 11 taste values, the candidate's 11 item values and their 11 products
(which sum to the taste-match score). Each window trains its own choice model from its own SVD, whose axes can come out
flipped or rotated, so every window's vectors are rotated (orthogonal Procrustes on the item vectors) into one shared
basis: the SVD at the earliest training boundary, built only from data before every window's targets. Rotations keep
every dot product unchanged.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root and loads Monish's package)
from safe_v2_step1 import CACHE, CMF2_ARGS, CMF2_EPOCHS, with_features  # noqa: E402
from src.choice_mf import click_pairs, click_svd, code_maps, encode, strip_time, taste_vectors, train  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / "safe_v2_step2.json"
BASELINE = 0.798967
DIM = CMF2_ARGS.dim


def vector_features(frame: pd.DataFrame, boundary: int, events: pl.DataFrame, cands: pl.DataFrame, umap: dict,
                    imap: dict, cache: Path, ref_items: np.ndarray, target_events: pl.DataFrame | None = None,
                    target_cands: pl.DataFrame | None = None) -> pd.DataFrame:
    """Taste vectors of a choice model trained only on events before `boundary`, rotated into the reference basis."""
    target_events = events if target_events is None else target_events
    target_cands = cands if target_cands is None else target_cands
    path = cache / f"{boundary}.parquet"
    if path.exists():
        feats = pd.read_parquet(path)
    else:
        start = time.perf_counter()
        rowids = frame["rowid"].astype(str).unique().tolist()
        target_ids = pl.DataFrame({"rowid": rowids}).join(target_cands, on="rowid").select("event_id")
        target = strip_time(encode(target_events.join(target_ids.unique(), on="event_id", how="semi"),
                                   target_cands, umap, imap))
        fit = strip_time(encode(events.filter(pl.col("timestamp") < boundary), cands, umap, imap))
        model, _ = train(fit, len(umap), len(imap), CMF2_ARGS, CMF2_EPOCHS,
                         click_svd(boundary, umap, imap, CMF2_ARGS.svd_rank), clicks=click_pairs(boundary, umap, imap))
        users, items = taste_vectors(model, target)

        # Orthogonal Procrustes: the rotation that best maps this model's item vectors onto the reference basis
        q = model.item.weight[:len(imap)].detach().numpy()
        a, _, bt = np.linalg.svd(q.T @ ref_items)
        rot = a @ bt
        users, items = users @ rot, items @ rot

        n = len(users)
        cols = {f"taste_u{k}": np.repeat(users[:, k], 10) for k in range(DIM)}
        cols |= {f"taste_i{k}": items[:, :, k].reshape(-1) for k in range(DIM)}
        cols |= {f"taste_p{k}": (users[:, None, k] * items[:, :, k]).reshape(-1) for k in range(DIM)}
        feats = pd.DataFrame({"rowid": target["rowid"].reshape(-1),
                              **{k: v.astype(np.float32) for k, v in cols.items()}})
        cache.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path, index=False)
        lf.log(f"taste vectors for boundary day {(boundary - lf.D0) / lf.DAY:.1f}: {n:,} events "
               f"in {time.perf_counter() - start:.0f} s")
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(
        feats.assign(rowid=feats["rowid"].astype(str)), on="rowid", how="left")
    assert out["taste_u0"].notna().all(), "taste vectors missing for some rows"
    return out


def build(scope: str, bounds: list[int], frames: list[pd.DataFrame], target_frame: pd.DataFrame, target_boundary: int,
          **target):
    torch.set_num_threads(12)
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()
    _, ref_items = click_svd(bounds[0], umap, imap, DIM)  # reference basis: earliest boundary, no target data
    vec_cache = CACHE / "taste_vectors" / scope
    parts = []
    for bound, f in zip(bounds, frames):
        f = with_features(f, bound, events, cands, umap, imap, scope)
        parts.append(vector_features(f, bound, events, cands, umap, imap, vec_cache, ref_items))
    tgt = with_features(target_frame, target_boundary, events, cands, umap, imap, scope, **target)
    tgt = vector_features(tgt, target_boundary, events, cands, umap, imap, vec_cache, ref_items, **target)
    return pd.concat(parts, ignore_index=True), tgt


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
    dot = val[[f"taste_p{k}" for k in range(DIM)]].sum(1).to_numpy()
    results = {"taste_dot_alone": lf.evaluate("rotated taste dot product alone", val, dot, split, eval_frame)}

    tr, va = lf.crossfit_choice(train_base, val)
    cols = lf.model_columns(tr)
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 2 (leak-free)", va, pred, split, eval_frame)
    lf.log(f"change vs step 1 ({BASELINE}): {res['local_auc'] - BASELINE:+.6f}")

    gain = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -gain[c])
    taste = [c for c in order if c.startswith("taste_")]
    results["step2"] = {**res, "trees": n_trees, "delta_vs_step1": res["local_auc"] - BASELINE,
                        "top_features_by_gain": [(c, round(float(gain[c]))) for c in order[:15]],
                        "best_taste_feature_ranks": {c: order.index(c) + 1 for c in taste[:10]},
                        "taste_share_of_gain": float(sum(gain[c] for c in taste) / sum(gain.values()))}
    results["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
        "data/validation/safe_v2_step2_val_c4.parquet", index=False)
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
    out = lf.REPO / "submissions" / "safe_v2_step2.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv[1:] else validate()
