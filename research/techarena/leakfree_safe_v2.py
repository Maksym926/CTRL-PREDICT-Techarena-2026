"""Leak-free re-run of Monish's SAFE_v2 model on the team's Parquet data and validation split.

Usage (from the repo root):
    python research/techarena/leakfree_safe_v2.py            # validation: original vs fixed protocol (~60 min)
    python research/techarena/leakfree_safe_v2.py --submit   # fixed protocol on all of train -> submission CSV

Monish's feature code is reused unchanged (features.py, svd_model.py, choice.py, phase_c_real.shrink_extra). Only the
validation protocol is fixed:
  1. Training targets stop at the validation boundary. His folds took each user's first event up to the END of train,
     so ~41% of validation events were also training rows, labels included.
  2. Early stopping uses an inner time holdout (the latest training window), never the validation set.
  3. Choice-model features for training rows are cross-fitted by user instead of predicted in-sample.
His original protocol is run on the same features for comparison. Data comes from data/parquet via src.data, and
scoring uses src.validation (fit days 1-4 = his DM24 fold, validate on each user's first event from day 5 on).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from src.data import load_candidates, load_events  # noqa: E402
from src.validation import make_split, score  # noqa: E402


def _import_monish_package() -> None:
    """Load research/techarena/src as package `monish` (both code bases name their package `src`)."""
    pkg = HERE / "src"
    spec = importlib.util.spec_from_file_location("monish", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["monish"] = module
    spec.loader.exec_module(module)


_import_monish_package()
from monish import features as m_features  # noqa: E402
from monish.cache import CachedEvents  # noqa: E402
from monish.choice import fit_choice  # noqa: E402
from monish.config import CUT, D0, DAY  # noqa: E402
from monish.dataset import GlobalDataset, click_log_before  # noqa: E402
from monish.features import build_features, build_stats  # noqa: E402
from monish.phase_a import weighted_auc  # noqa: E402
from monish.phase_c_real import shrink_extra  # noqa: E402
from monish.phase_c_revision import C2_COLS, FULL_SEEDS, add_choice_pair, bounds_for  # noqa: E402
from monish.run_phase1 import first_events, subset  # noqa: E402
from monish.svd_model import fit_impression_svd, fit_svd  # noqa: E402
from monish.train import feature_columns  # noqa: E402

CACHE = HERE / "cache" / "leakfree"
OUT = HERE / "outputs" / "leakfree_safe_v2.json"
SUBMISSION = REPO / "submissions" / "safe_v2_leakfree.csv"
VAL_DAY = 4  # his DM24 fold == src.validation cutoff day 4
# His final_seed_model settings (SAFE_v2), except n_jobs which only affects speed
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, max_bin=127, min_child_samples=300,
                  colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=10.0, n_jobs=-1, verbosity=-1)
MAX_TREES, PATIENCE = 2500, 80


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data: Parquet -> Monish's GlobalDataset
def _to_cached(events: pl.DataFrame, cands: pl.DataFrame, user_code: dict, item_code: dict, labelled: bool) -> CachedEvents:
    # File order is chronological; (timestamp, event_id) also breaks ties exactly like src.validation.make_split
    events = events.sort("timestamp", "event_id").with_row_index("e")
    cols = ["item_id", "rowid"] + (["clicked"] if labelled else [])
    grouped = cands.group_by("event_id", maintain_order=True).agg(cols)
    joined = events.join(grouped, on="event_id", how="left").sort("e")
    n = joined.height

    flat = joined.select(cols).explode(cols)
    candidates = flat["item_id"].replace_strict(item_code, return_dtype=pl.Int32).to_numpy().reshape(n, 10)
    rowids = flat["rowid"].to_numpy().astype(object).reshape(n, 10)
    labels = (flat["clicked"].to_numpy().astype(np.int8).reshape(n, 10) if labelled
              else np.full((n, 10), -1, dtype=np.int8))

    # Histories right-aligned in a (n, 20) array padded with -1, as in his build_cache
    hist = (joined.select("e", "history_item_ids", "history_timestamps")
            .filter(pl.col("history_item_ids").list.len() > 0)
            .with_columns(pl.col("history_item_ids").list.len().alias("n"))
            .explode(["history_item_ids", "history_timestamps"])
            .with_columns(pl.int_range(pl.len()).over("e").alias("k")))
    e_idx = hist["e"].to_numpy()
    pos = 20 - hist["n"].to_numpy() + hist["k"].to_numpy()
    history_items = np.full((n, 20), -1, dtype=np.int32)
    history_times = np.full((n, 20), -1, dtype=np.int64)
    history_items[e_idx, pos] = hist["history_item_ids"].replace_strict(item_code, return_dtype=pl.Int32).to_numpy()
    history_times[e_idx, pos] = hist["history_timestamps"].to_numpy()

    return CachedEvents(
        candidates=candidates, labels=labels,
        user_id=joined["user_id"].replace_strict(user_code, return_dtype=pl.Int32).to_numpy(),
        timestamp=joined["timestamp"].to_numpy().astype(np.int64),
        history_items=history_items, history_times=history_times,
        history_length=(history_items >= 0).sum(axis=1).astype(np.int16), rowids=rowids,
        item_keys=np.arange(len(item_code), dtype=np.int64), user_keys=np.arange(len(user_code), dtype=np.int64),
    )


def load_global() -> GlobalDataset:
    te, tc, ee, ec = load_events("train"), load_candidates("train"), load_events("eval"), load_candidates("eval")
    users = pl.concat([te["user_id"], ee["user_id"]]).unique().sort()
    items = pl.concat([tc["item_id"], ec["item_id"], te["history_item_ids"].explode().drop_nulls(),
                       ee["history_item_ids"].explode().drop_nulls()]).unique().sort()
    user_code = {u: i for i, u in enumerate(users)}
    item_code = {it: i for i, it in enumerate(items)}
    return GlobalDataset(train=_to_cached(te, tc, user_code, item_code, labelled=True),
                         evaluation=_to_cached(ee, ec, user_code, item_code, labelled=False),
                         n_users=len(user_code), n_items=len(item_code))


# ---------------------------------------------------------------- speed-up: vectorised click_log_before
def install_fast_click_log(data: GlobalDataset, check_boundary: int) -> None:
    """Replace his pure-Python click_log_before (22M loop steps per call) with an identical vectorised version."""
    parts = []
    e, j = np.nonzero(data.train.labels == 1)
    parts.append(np.c_[data.train.user_id[e], data.train.candidates[e, j], data.train.timestamp[e]])
    for ev in (data.train, data.evaluation):
        e, k = np.nonzero(ev.history_items >= 0)
        parts.append(np.c_[ev.user_id[e], ev.history_items[e, k], ev.history_times[e, k]])
    all_clicks = np.unique(np.concatenate(parts).astype(np.int64), axis=0)  # sorted, like sorted(set(rows))

    def fast(_data, boundary):
        return all_clicks[all_clicks[:, 2] < int(boundary)]

    start = time.perf_counter()
    same = np.array_equal(click_log_before(data, check_boundary), fast(data, check_boundary))
    log(f"vectorised click_log_before identical to original: {same} (original took {time.perf_counter() - start:.0f} s)")
    if not same:
        raise SystemExit("fast click log differs from the original; aborting")
    m_features.click_log_before = fast  # build_stats (and shrink_extra through it) look the name up here


# ---------------------------------------------------------------- features: his code path, one frame per boundary
def boundary_frame(data: GlobalDataset, boundary: int, evaluation: bool = False) -> pd.DataFrame:
    """His cached_frame + C2 extras: targets = each user's first event in [boundary, CUT), or all of evaluation.csv
    (scored at boundary CUT, as in his build_safe_v2)."""
    path = CACHE / ("evaluation.parquet" if evaluation else f"{boundary}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    start = time.perf_counter()
    target = data.evaluation if evaluation else subset(data.train, first_events(data.train, boundary, CUT))
    stats = build_stats(data, boundary)
    s11 = fit_svd(stats["click_log"], data.n_users, data.n_items, 11)
    s32 = fit_svd(stats["click_log"], data.n_users, data.n_items, 32)
    imp = fit_impression_svd(data.train, boundary, data.n_users, data.n_items, 10)
    frame = build_features(data, target, boundary, s11, stats, svd_alt=s32, impression_svd=imp)
    extra = shrink_extra(data, target, boundary, k=10.0, uncertainty=True)
    frame = pd.concat([frame, extra[C2_COLS]], axis=1)
    frame["rowid"] = target.rowids.reshape(-1)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log(f"built {'evaluation' if evaluation else 'boundary'} day {(boundary - D0) / DAY:.1f}: "
        f"{target.n_events:,} events in {time.perf_counter() - start:.0f} s")
    return frame


def model_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(frame) if c != "rowid"]


def crossfit_choice(train: pd.DataFrame, val: pd.DataFrame):
    """Choice features for training rows from a model that never saw those users; val uses a model on all train."""
    train, val = train.copy(), val.copy()
    fold = train["user_id"].to_numpy() % 2
    out = {c: np.empty(len(train)) for c in ("choice_utility", "choice_probability", "choice_no_click")}
    for k in (0, 1):
        m = fold == k
        u, p, n = fit_choice(train[~m], maxiter=80).predict(train[m])
        out["choice_utility"][m], out["choice_probability"][m], out["choice_no_click"][m] = u, p, np.repeat(n, 10)
    for c, v in out.items():
        train[c] = v
    u, p, n = fit_choice(train, maxiter=80).predict(val)
    val["choice_utility"], val["choice_probability"], val["choice_no_click"] = u, p, np.repeat(n, 10)
    return train, val


# ---------------------------------------------------------------- evaluation
def evaluate(name: str, val: pd.DataFrame, pred: np.ndarray, split, eval_frame: pd.DataFrame) -> dict:
    preds = pl.DataFrame({"rowid": val["rowid"].astype(str).to_numpy(), "predicted_probability": pred})
    res = {"local_auc": score(split.val_candidates, preds),
           "his_weighted_auc": weighted_auc(val, pred, eval_frame)["joint_auc_supported"]}
    log(f"{name}: local AUC {res['local_auc']:.6f} | his eval-reweighted AUC {res['his_weighted_auc']:.6f}")
    return res


# ---------------------------------------------------------------- shared steps of both modes
def load_data(check_boundary: int) -> GlobalDataset:
    log("loading Parquet data into his GlobalDataset format")
    data = load_global()
    log(f"{data.train.n_events:,} train events, {data.evaluation.n_events:,} eval events, "
        f"{data.n_users:,} users, {data.n_items:,} items")
    install_fast_click_log(data, check_boundary)
    return data


def training_windows(data: GlobalDataset, bounds: list[int]) -> pd.DataFrame:
    frames = []
    for wi, bound in enumerate(bounds):
        f = boundary_frame(data, bound)
        f["window_id"] = wi
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def inner_holdout_trees(tr: pd.DataFrame, bounds: list[int], cols: list[str]) -> int:
    """Tree count chosen on the latest training window, fitting only on targets dated before it."""
    last = len(bounds) - 1
    inner_tr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    inner_va = tr[tr["window_id"] == last]
    m = lgb.LGBMClassifier(n_estimators=MAX_TREES, random_state=FULL_SEEDS[0], **LGB_PARAMS)
    m.fit(inner_tr[cols], inner_tr["label"], eval_set=[(inner_va[cols], inner_va["label"])], eval_metric="auc",
          callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    log(f"  inner holdout ({len(inner_tr):,} train / {len(inner_va):,} holdout rows) chose {m.best_iteration_} trees")
    return int(m.best_iteration_)


def fit_seeds(tr: pd.DataFrame, target: pd.DataFrame, cols: list[str], n_trees: int):
    """Average of the three SAFE_v2 seeds; returns the predictions and the last model."""
    preds = []
    for seed in FULL_SEEDS:
        m = lgb.LGBMClassifier(n_estimators=n_trees, random_state=seed, **LGB_PARAMS)
        m.fit(tr[cols], tr["label"])
        preds.append(m.predict_proba(target[cols])[:, 1])
        log(f"  seed {seed} done")
    return np.mean(preds, axis=0), m


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(VAL_DAY)
    b = split.cutoff_ts
    assert b == D0 + VAL_DAY * DAY
    bounds = bounds_for(VAL_DAY, half=True)  # his DM24 training windows: days 0.5, 1.0, ..., 3.5

    data = load_data(b)
    train_all = training_windows(data, bounds)
    val = boundary_frame(data, b)
    assert len(val) == split.val_candidates.height and set(val["rowid"]) == set(split.val_candidates["rowid"]), \
        "his validation targets differ from src.validation"
    log(f"validation rows match src.validation exactly: {len(val):,}")

    ev = data.evaluation
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev.history_length, 10).astype(float),
                               "gap_hours": np.repeat((ev.timestamp - CUT) / 3600.0, 10)})
    results = {"svd_cos_alone": evaluate("svd_cos alone (no training)", val, val["svd_cos"].to_numpy(), split, eval_frame)}

    # ---- 1. his original protocol: targets to end of train, in-sample choice, early stopping on validation
    tr, va, _ = add_choice_pair(train_all, val)
    cols = model_columns(tr)
    overlap = va["rowid"].isin(set(tr["rowid"])).mean()  # hash lookup; np.isin on strings is quadratic
    log(f"original protocol: {len(tr):,} training rows; validation rows also in training: {overlap:.1%}")
    preds, best = [], []
    for seed in FULL_SEEDS:
        m = lgb.LGBMClassifier(n_estimators=MAX_TREES, random_state=seed, **LGB_PARAMS)
        m.fit(tr[cols], tr["label"], eval_set=[(va[cols], va["label"])], eval_metric="auc",
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        preds.append(m.predict_proba(va[cols], num_iteration=m.best_iteration_)[:, 1])
        best.append(int(m.best_iteration_))
        log(f"  original protocol seed {seed}: {best[-1]} trees")
    results["original_protocol"] = {**evaluate("ORIGINAL protocol (leaky)", va, np.mean(preds, axis=0), split, eval_frame),
                                    "train_rows": len(tr), "val_rows_in_training": float(overlap), "trees": best}
    del tr, va

    # ---- 2. fixed protocol: targets end at the validation boundary
    fixed = train_all[train_all["timestamp"] < b].reset_index(drop=True)
    tr, va = crossfit_choice(fixed, val)
    overlap = va["rowid"].isin(set(tr["rowid"])).mean()  # hash lookup; np.isin on strings is quadratic
    log(f"fixed protocol: {len(tr):,} training rows; validation rows also in training: {overlap:.1%}")

    n_trees = inner_holdout_trees(tr, bounds, cols)  # latest window (targets in [day 3.5, day 4)) as holdout
    pred, m = fit_seeds(tr, va, cols, n_trees)
    gain = sorted(zip(cols, m.booster_.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    results["fixed_protocol"] = {**evaluate("FIXED protocol (leak-free)", va, pred, split, eval_frame),
                                 "train_rows": len(tr), "val_rows_in_training": float(overlap), "trees": n_trees,
                                 "top_features_by_gain": [(c, round(float(g))) for c, g in gain[:15]]}

    results["runtime_min"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    """Fixed protocol on all of train (windows every half day up to day 6.5), predicting evaluation.csv."""
    t0 = time.perf_counter()
    bounds = bounds_for(7, half=True)  # days 0.5, 1.0, ..., 6.5; targets run to the end of train (eval is later)
    data = load_data(CUT)
    train_all = training_windows(data, bounds)
    test = boundary_frame(data, CUT, evaluation=True)

    tr, te = crossfit_choice(train_all, test)
    cols = model_columns(tr)
    log(f"submission: {len(tr):,} training rows from {len(bounds)} windows, {len(te):,} evaluation rows")
    n_trees = inner_holdout_trees(tr, bounds, cols)  # latest window (targets in [day 6.5, day 7)) as holdout
    pred, _ = fit_seeds(tr, te, cols, n_trees)

    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(SUBMISSION, index=False)
    log(f"wrote {SUBMISSION} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="train on all of train and write the submission CSV")
    submit() if ap.parse_args().submit else validate()
