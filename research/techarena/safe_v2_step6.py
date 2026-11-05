"""Step 6: step 5 + "versus the other nine candidates" features (+ optional wider filler-matched choice models).

Usage (from the repo root):
    python research/techarena/safe_v2_step6.py            # validation (baseline: step 5 = 0.806442)
    python research/techarena/safe_v2_step6.py --submit   # submissions/safe_v2_step6.csv
    python research/techarena/safe_v2_step6.py --quick    # smoke test: 10% of users, small models
    add --dump to save the final validation frames to cache/lab/ for fast feature experiments

Why: LightGBM scores one candidate row at a time, so it cannot see that ANOTHER candidate in the same event is a
repeat click or the sequence model's clear favourite. Only the SVD cosine and item CTR had such event-relative columns.
Within-event discrimination drives the global AUC (headroom test: +0.01 "which item" AUC moves the global AUC by
about +0.0095, against +0.0008 for the same gain in "whether the user clicks"), so for each strong per-candidate score
this adds the best value among the other nine candidates and this candidate's margin over it, plus how many choice
models rank the candidate first and its mean rank across them. All are functions of the event's own leak-free features.
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
ap.add_argument("--dump", action="store_true")
ap.add_argument("--wide", default="", help="comma-separated widths of extra filler-matched choice models, e.g. 12,16")
ARGS = ap.parse_args()
sys.argv = sys.argv[:1] + (["--quick"] if ARGS.quick else [])  # step 5 / safe_v2_extra parse the command line on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safe_v2_step5 as s5  # noqa: E402  (step 5 = stage 3 + small wins + filler-matched choice model)
from safe_v2_choicemf import cmf_features  # noqa: E402
from safe_v2_step1 import CACHE  # noqa: E402
from src.choice_mf import code_maps  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

ex, lf, sd, s3 = s5.ex, s5.lf, s5.sd, s5.s3
NAME = "safe_v2_step6_quick" if ARGS.quick else "safe_v2_step6"
OUT = HERE / "outputs" / f"{NAME}.json"
BASELINE = 0.806442  # step 5
MIN_GAIN = 0.0005
WIDE = [int(w) for w in ARGS.wide.split(",") if w]
CHOICE = ["cmfp_prob", "cmf12_prob", "cmf16_prob", "cmf_prob", "choice_probability", "seq_prob"] + \
         [f"cmfp{w}_prob" for w in WIDE]
CROSS = CHOICE + ["svd_resid_dot", "svd32_resid_dot", "ease_all", "history_count", "clicked_before"]


def wide_pop(frame: pd.DataFrame, b: int, events, cands, umap, imap, scope: str, **target) -> pd.DataFrame:
    for w in WIDE:
        args = SimpleNamespace(**{**vars(s5.POP_ARGS), "dim": w, "svd_rank": w})
        f = cmf_features(frame[["rowid"]], b, events, cands, umap, imap, CACHE / f"choicemf_pop_d{w}" / scope,
                         args=args, epochs=s5.POP_EPOCHS, **target)
        f = f.rename(columns={c: c.replace("cmf_", f"cmfp{w}_") for c in ("cmf_prob", "cmf_noclick", "cmf_rank")})
        frame = frame.assign(rowid=frame["rowid"].astype(str)).merge(f, on="rowid", how="left")
        assert frame[f"cmfp{w}_prob"].notna().all(), f"boundary {b}: width-{w} choice features missing"
    return frame


def with_wide(train_base, bounds, target, target_b, scope, **target_tables):
    if not WIDE:
        return train_base, target
    events, cands = load_events("train"), load_candidates("train")
    umap, imap = code_maps()
    parts = [wide_pop(train_base[train_base["window_id"] == wi].reset_index(drop=True), b, events, cands, umap, imap,
                      scope) for wi, b in enumerate(bounds)]
    return pd.concat(parts, ignore_index=True), wide_pop(target, target_b, events, cands, umap, imap, scope,
                                                         **target_tables)


def cross_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Event-relative columns: best of the other nine candidates, margin over it, first-place votes, mean rank."""
    keys = [k for k in ("window_id", "event_index") if k in frame]
    order = np.lexsort([frame["candidate_index"].to_numpy()] + [frame[k].to_numpy() for k in reversed(keys)])
    n = len(frame)
    assert n % 10 == 0
    for k in keys:
        blk = frame[k].to_numpy()[order].reshape(-1, 10)
        assert (blk == blk[:, :1]).all(), "events are not blocks of 10 rows"
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    new, votes, ranks = {}, np.zeros((n // 10, 10)), []
    for c in [c for c in CROSS if c in frame]:
        s = frame[c].to_numpy(np.float64)[order].reshape(-1, 10)
        srt = np.sort(s, axis=1)
        top1, top2 = srt[:, -1:], srt[:, -2:-1]
        other = np.where(s >= top1, top2, top1)  # ties for first: the other one is as good, margin 0
        new[f"{c}_best_other"], new[f"{c}_margin"] = other, s - other
        if c in CHOICE:
            votes += s >= top1
            ranks.append((-s).argsort(1).argsort(1))
    new["choice_first_votes"] = votes
    new["choice_mean_rank"] = np.mean(ranks, axis=0)
    added = pd.DataFrame({name: m.ravel()[inv].astype(np.float32) for name, m in new.items()}, index=frame.index)
    return pd.concat([frame, added], axis=1)


def dump(frame: pd.DataFrame, name: str) -> None:
    """Save a final frame to cache/lab/ for fast experiments; float64 columns become float32, one column at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    lab = CACHE / "lab"
    lab.mkdir(parents=True, exist_ok=True)
    arrays = [pa.array(frame[c].to_numpy(np.float32) if frame[c].dtype == np.float64 else frame[c].to_numpy())
              for c in frame.columns]
    pq.write_table(pa.Table.from_arrays(arrays, names=list(frame.columns)), lab / f"{name}.parquet")
    lf.log(f"dumped {name} ({len(frame):,} rows) to {lab}")


def new_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.endswith(("_best_other", "_margin")) or c.startswith("choice_first_votes")
            or c == "choice_mean_rank" or c.startswith("cmfp") and c[4:6].isdigit()]


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
    train_base, val = s5.with_pop(train_base, bounds, val, b, "val")
    train_base, val = with_wide(train_base, bounds, val, b, "val")
    if ARGS.quick:
        train_base, val = ex.subsample(train_base, val)
    cands, clicks, n_users, n_items = sd.load_logs()
    keys = cands.select("rowid", "event_id", "u", "i")
    train_base, val = ex.with_extra(train_base, bounds, val, b, keys, keys, cands, clicks, n_users, n_items)

    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    tr, va = (train_base, val) if ARGS.quick else lf.crossfit_choice(train_base, val)
    tr, va = cross_features(tr), cross_features(va)
    cols = lf.model_columns(tr)
    added = new_columns(tr)
    assert added and set(added) <= set(cols), "new features not picked up"
    lf.log(f"{len(tr):,} training rows, {len(cols)} features ({len(added)} new)")
    if ARGS.dump:
        dump(tr, "step6_train")
        dump(va, "step6_val")
    if ARGS.quick:
        lf.MAX_TREES = 150
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    res = lf.evaluate("STEP 6: step 5 + versus-the-other-candidates features (leak-free)", va, pred, split, eval_frame)
    gain = res["local_auc"] - BASELINE
    decision = "SUBMIT" if gain >= MIN_GAIN else "DON'T SUBMIT"
    lf.log(f"change vs step 5 ({BASELINE}): {gain:+.6f} -> {decision} (needs +{MIN_GAIN})")

    imp = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -imp[c])
    results = {**res, "trees": n_trees, "delta_vs_step5": gain, "decision": decision, "wide": WIDE,
               "new_feature_ranks": {c: order.index(c) + 1 for c in added},
               "top_features_by_gain": [(c, round(float(imp[c]))) for c in order[:20]],
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
    train_base, test = s5.with_pop(train_base, bounds, test, lf.CUT, "full", **target_tables)
    train_base, test = with_wide(train_base, bounds, test, lf.CUT, "full", **target_tables)
    cands, clicks, n_users, n_items, eval_keys = sd.load_logs(with_eval=True)
    keys = cands.select("rowid", "event_id", "u", "i")
    train_base, test = ex.with_extra(train_base, bounds, test, lf.CUT, keys, eval_keys, cands, clicks, n_users,
                                     n_items)
    tr, te = lf.crossfit_choice(train_base, test)
    tr, te = cross_features(tr), cross_features(te)
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    try:  # always: lets a lab variant be refit without rebuilding the features; never at the submission's expense
        dump(tr, "step6_full_train")
        dump(te, "step6_full_test")
    except Exception as err:  # noqa: BLE001
        lf.log(f"frame dump skipped ({type(err).__name__}: {err})")
    n_trees = lf.inner_holdout_trees(tr, bounds, cols)
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)
    out = lf.REPO / "submissions" / f"{NAME}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if ARGS.submit else validate()
