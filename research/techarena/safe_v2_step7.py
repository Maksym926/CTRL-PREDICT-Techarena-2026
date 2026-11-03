"""Step 7: step 5 + sequence-model features from histories of up to 50 known clicks (kaggle/seq_h50.ipynb), added next
to the 20-click ones. Built on the frames step 5 already computed, so only the new columns are merged in:
validation from cache/lab/step6_*.parquet (step 5's frames + step 6's event-relative columns, which are dropped),
submission from cache/lab/step5_full_*.parquet.

Usage (from the repo root, after unzipping seq_h50.zip into research/techarena/cache/seq/stage3_h50/):
    python research/techarena/safe_v2_step7.py --baseline   # step 5 retrained from the frames: 0.806467 (original run 0.806442)
    python research/techarena/safe_v2_step7.py              # validation (baseline: step 5 from the frames = 0.806467)
    python research/techarena/safe_v2_step7.py --submit     # submissions/safe_v2_step7.csv
    python research/techarena/safe_v2_step7.py --seq seq_aug --tag _aug   # other sequence features: outputs named
                                                            # safe_v2_step7_aug.*, so step 7's own files stay intact

Why: within-event accuracy keeps rising with the user's known clicks, but the sequence model saw only the last 20;
35% of validation targets (37% of evaluation users) have more. Same leak-free protocol: each extra click is strictly
older than the oldest of the event's given 20.
Decision rule, fixed before seeing the result: build the submission only if validation gains at least +0.0005.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", action="store_true", help="no new columns: checks the frames reproduce step 5")
ap.add_argument("--submit", action="store_true")
ap.add_argument("--seq", default="stage3_h50", help="folder under cache/seq with the 50-click features")
ap.add_argument("--tag", default="", help="suffix for the output names (needed with another --seq)")
ap.add_argument("--extra", default="", help="another folder under cache/seq whose per-boundary files (rowid + "
                                            "columns, e.g. fullmf_model.py's fmf_*) are merged in as well")
ap.add_argument("--swap", nargs="*", default=[], help="prefix=folder: the folder under cache/ (cmf_seeds.py --merge, "
                                                      "<boundary>.parquet) replaces the <prefix>_prob/_noclick/_rank columns")
ap.add_argument("--lgb", nargs="*", default=[], help="key=value LightGBM overrides on top of the Optuna parameters, "
                                                     "e.g. learning_rate=0.015 num_leaves=31")
ap.add_argument("--max-trees", type=int, default=0, help="cap of the inner-holdout tree search (default 2500)")
ap.add_argument("--tree-mult", type=float, default=1.0, help="scale the inner-holdout tree count for the full fit "
                                                             "(which has one more window of data)")
ap.add_argument("--share-feats", action="store_true", help="add the sequence models' within-event log-shares and their "
                                                            "50- minus 20-click difference as columns")
ap.add_argument("--n-seeds", type=int, default=0, help="average this many LightGBM seeds instead of the usual 3")
ARGS = ap.parse_args()
if (ARGS.seq != "stage3_h50" or ARGS.extra or ARGS.swap or ARGS.lgb or ARGS.max_trees or ARGS.tree_mult != 1.0
        or ARGS.share_feats or ARGS.n_seeds) and not ARGS.tag:
    raise SystemExit("another --seq, --extra, --swap, --lgb, --max-trees, --tree-mult, --share-feats or --n-seeds needs "
                     "a --tag, or step 7's own outputs would be overwritten")
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402
from src.data import load_events  # noqa: E402
from src.validation import make_split  # noqa: E402

lf.LGB_PARAMS.update(json.loads((HERE / "outputs" / "optuna_step4.json").read_text())["best_params"])  # as step 5
for kv in ARGS.lgb:
    key, value = kv.split("=", 1)
    lf.LGB_PARAMS[key] = json.loads(value)
if ARGS.max_trees:
    lf.MAX_TREES = ARGS.max_trees
if ARGS.n_seeds:  # the usual seeds first, then new ones derived from them
    lf.FULL_SEEDS = (list(lf.FULL_SEEDS) + [s + 7919 * k for k in range(1, 10) for s in lf.FULL_SEEDS])[:ARGS.n_seeds]
LAB = HERE / "cache" / "lab"
SEQ_DIR = HERE / "cache" / "seq" / ARGS.seq
NAME = "safe_v2_step5_from_frames" if ARGS.baseline else "safe_v2_step7" + ARGS.tag
OUT = HERE / "outputs" / f"{NAME}.json"
BASELINE = 0.806467  # step 5 retrained from these same frames (--baseline; the original step 5 run gave 0.806442)
STEP7 = 0.806901  # step 7 itself (stage3_h50), the reference for other sequence features
MIN_GAIN = 0.0005
NEW = ["seq50_prob", "seq50_noclick", "seq50_rank"]


def load(name: str) -> pd.DataFrame:
    f = pd.read_parquet(LAB / f"{name}.parquet")
    step6 = [c for c in f.columns if c.endswith(("_best_other", "_margin")) or c in ("choice_first_votes",
                                                                                    "choice_mean_rank")]
    return f.drop(columns=step6)


def add_seq50(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    s = pd.read_parquet(SEQ_DIR / f"seq_{name}.parquet")
    s = s.assign(rowid=s["rowid"].astype(str)).rename(columns=dict(zip(["seq_prob", "seq_noclick", "seq_rank"], NEW)))
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(s, on="rowid", how="left")
    if len(out) != len(frame) or out[NEW].isna().any().any():
        raise SystemExit(f"seq_{name}: rows without 50-click sequence features")
    if ARGS.extra:
        x = pd.read_parquet(HERE / "cache" / "seq" / ARGS.extra / f"seq_{name}.parquet")
        cols = [c for c in x.columns if c != "rowid"]
        out = out.merge(x.assign(rowid=x["rowid"].astype(str)), on="rowid", how="left")
        if len(out) != len(frame) or out[cols].isna().any().any():
            raise SystemExit(f"{ARGS.extra}/seq_{name}: rows without the extra features")
    for spec in ARGS.swap:
        prefix, folder = spec.split("=")
        cols = [f"{prefix}_prob", f"{prefix}_noclick", f"{prefix}_rank"]
        b = lf.CUT if name == "full_eval" else int(name[1:])
        x = pd.read_parquet(HERE / "cache" / folder / f"{b}.parquet")
        x = x.assign(rowid=x["rowid"].astype(str)).rename(columns=dict(zip(["cmf_prob", "cmf_noclick", "cmf_rank"], cols)))
        out = out.drop(columns=cols).merge(x, on="rowid", how="left")
        if len(out) != len(frame) or out[cols].isna().any().any():
            raise SystemExit(f"{folder}/{b}: rows without the replacement {prefix} features")
    return add_shares(out) if ARGS.share_feats else out


def add_shares(f: pd.DataFrame) -> pd.DataFrame:
    """--share-feats (one window or target frame at a time): each sequence model's log-share within the event and the
    50- minus 20-click difference, which only carries information for users with more than 20 known clicks."""
    new = {}
    for c in ("seq50_prob", "seq_prob"):
        v = f[c].clip(lower=1e-7)
        new[f"{c}_lshare"] = np.log(v / v.groupby(f["event_index"], sort=False).transform("sum"))
    new["seq50_minus_seq20"] = new["seq50_prob_lshare"] - new["seq_prob_lshare"]
    return f.assign(**new)


def with_seq50(train: pd.DataFrame, bounds: list[int], target: pd.DataFrame, target_name: str):
    if ARGS.baseline:
        return train, target
    parts = [add_seq50(train[train["window_id"] == wi].reset_index(drop=True), f"b{b}") for wi, b in enumerate(bounds)]
    return pd.concat(parts, ignore_index=True), add_seq50(target, target_name)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    tr, va = with_seq50(load("step6_train"), bounds, load("step6_val"), f"b{split.cutoff_ts}")
    cols = lf.model_columns(tr)
    assert ARGS.baseline or set(NEW) <= set(cols)
    lf.log(f"{len(tr):,} training rows, {len(cols)} features")
    ev = load_events("eval")
    eval_frame = pd.DataFrame({"history_length": np.repeat(ev["history_item_ids"].list.len().to_numpy(), 10).astype(float),
                               "gap_hours": np.repeat((ev["timestamp"].to_numpy() - lf.CUT) / 3600.0, 10)})
    n_trees = int(round(lf.inner_holdout_trees(tr, bounds, cols) * ARGS.tree_mult))
    pred, m = lf.fit_seeds(tr, va, cols, n_trees)
    label = "step 5 retrained from the saved frames" if ARGS.baseline else "STEP 7: step 5 + 50-click sequence features"
    res = lf.evaluate(label, va, pred, split, eval_frame)
    gain = res["local_auc"] - BASELINE
    decision = "SUBMIT" if gain >= MIN_GAIN and not ARGS.baseline else "DON'T SUBMIT"
    lf.log(f"change vs step 5 ({BASELINE}): {gain:+.6f} -> {decision} (needs +{MIN_GAIN})")
    if ARGS.tag:
        lf.log(f"change vs step 7 ({STEP7}, sequence features stage3_h50): {res['local_auc'] - STEP7:+.6f}")
    imp = dict(zip(cols, m.booster_.feature_importance(importance_type="gain")))
    order = sorted(cols, key=lambda c: -imp[c])
    results = {**res, "trees": n_trees, "delta_vs_step5": gain, "decision": decision,
               "lgb_overrides": ARGS.lgb, "max_trees": lf.MAX_TREES, "share_feats": ARGS.share_feats,
               "lgb_seeds": list(lf.FULL_SEEDS),
               "new_feature_ranks": {c: order.index(c) + 1 for c in NEW if c in cols},
               "top_features_by_gain": [(c, round(float(imp[c]))) for c in order[:15]],
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    if not ARGS.baseline:
        results["seq50_alone"] = lf.evaluate("50-click sequence model alone", va, va["seq50_prob"].to_numpy(), split,
                                             eval_frame)
        pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": pred}).to_parquet(
            f"data/validation/{NAME}_val_c4.parquet", index=False)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    tr, te = with_seq50(load("step5_full_train"), bounds, load("step5_full_test"), "full_eval")
    cols = lf.model_columns(tr)
    lf.log(f"submission: {len(tr):,} training rows, {len(te):,} evaluation rows, {len(cols)} features")
    n_trees = int(round(lf.inner_holdout_trees(tr, bounds, cols) * ARGS.tree_mult))
    pred, _ = lf.fit_seeds(tr, te, cols, n_trees)
    out = lf.REPO / "submissions" / f"{NAME}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": pred}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if ARGS.submit else validate()
