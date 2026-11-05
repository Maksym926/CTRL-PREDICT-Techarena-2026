"""Lambdarank LightGBM on step 7's frames: a third stack model that only ranks the candidates inside each event
("which"), as an extra input of final_blend.py's conditional-logit which part (--val-c / --sub-c).

Same frames and columns as safe_v2_step7.py and softmax_step7.py: step 5's saved frames + seq50_* from
cache/seq/<--seq>, optional --swap and --share-feats. Trained on clicked events only (lambdarank has no gradient on an
event without a click). Trees: early stopping on NDCG@10 (with one clicked item, 1 / log2(1 + its rank)) on the latest
training window, then a refit on all windows; the submission averages the three SAFE_v2 seeds. Outputs are raw scores:
only their order inside an event matters (final_blend.py turns them into within-event log-softmax shares).

Usage (from the repo root):
    python research/techarena/rank_step7.py --seq stage3_h50_s30 --tag _s30            # validation
    python research/techarena/rank_step7.py --seq stage3_h50_s30 --tag _s30 --submit   # submissions/safe_v2_step7_rank_s30.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="stage3_h50_s30", help="folder under cache/seq with the 50-click features")
ap.add_argument("--submit", action="store_true")
ap.add_argument("--tag", default="", help="suffix for the output names")
ap.add_argument("--swap", nargs="*", default=[], help="prefix=folder: as safe_v2_step7.py --swap")
ap.add_argument("--share-feats", action="store_true", help="as safe_v2_step7.py --share-feats")
A = ap.parse_args()
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402
from src.validation import make_split  # noqa: E402

lf.LGB_PARAMS.update(json.loads((HERE / "outputs" / "optuna_step4.json").read_text())["best_params"])  # as step 5
PARAMS = {**lf.LGB_PARAMS, "objective": "lambdarank", "metric": "ndcg", "eval_at": [10]}
LAB = HERE / "cache" / "lab"
SEQ_DIR = HERE / "cache" / "seq" / A.seq
NEW = ["seq50_prob", "seq50_noclick", "seq50_rank"]


def add_seq50(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    s = pd.read_parquet(SEQ_DIR / f"seq_{name}.parquet")
    s = s.assign(rowid=s["rowid"].astype(str)).rename(columns=dict(zip(["seq_prob", "seq_noclick", "seq_rank"], NEW)))
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(s, on="rowid", how="left")
    if len(out) != len(frame) or out[NEW].isna().any().any():
        raise SystemExit(f"seq_{name}: rows without 50-click sequence features")
    for spec in A.swap:
        prefix, folder = spec.split("=")
        cols = [f"{prefix}_prob", f"{prefix}_noclick", f"{prefix}_rank"]
        b = lf.CUT if name == "full_eval" else int(name[1:])
        x = pd.read_parquet(HERE / "cache" / folder / f"{b}.parquet")
        x = x.assign(rowid=x["rowid"].astype(str)).rename(columns=dict(zip(["cmf_prob", "cmf_noclick", "cmf_rank"], cols)))
        out = out.drop(columns=cols).merge(x, on="rowid", how="left")
        if len(out) != len(frame) or out[cols].isna().any().any():
            raise SystemExit(f"{folder}/{b}: rows without the replacement {prefix} features")
    if A.share_feats:  # same columns as safe_v2_step7.add_shares
        new = {}
        for c in ("seq50_prob", "seq_prob"):
            v = out[c].clip(lower=1e-7)
            new[f"{c}_lshare"] = np.log(v / v.groupby(out["event_index"], sort=False).transform("sum"))
        new["seq50_minus_seq20"] = new["seq50_prob_lshare"] - new["seq_prob_lshare"]
        out = out.assign(**new)
    return out


def load(name: str, bounds: list[int], target: str | None = None) -> pd.DataFrame:
    """A saved frame without step 6's columns, plus seq50_*, sorted into blocks of 10 rows per event."""
    f = pd.read_parquet(LAB / f"{name}.parquet")
    f = f.drop(columns=[c for c in f.columns if c.endswith(("_best_other", "_margin"))
                        or c in ("choice_first_votes", "choice_mean_rank")])
    if target is None:
        f = pd.concat([add_seq50(f[f["window_id"] == wi].reset_index(drop=True), f"b{b}")
                       for wi, b in enumerate(bounds)], ignore_index=True)
    else:
        f = add_seq50(f, target)
    keys = [k for k in ("window_id", "event_index", "candidate_index") if k in f]
    f = f.sort_values(keys, kind="stable").reset_index(drop=True)
    blk = f["event_index"].to_numpy().reshape(-1, 10)
    assert (blk == blk[:, :1]).all(), f"{name}: events are not blocks of 10 rows"
    return f


def clicked_rows(f: pd.DataFrame) -> np.ndarray:
    return np.repeat(f["label"].to_numpy().reshape(-1, 10).max(1) == 1, 10)


def dataset(f: pd.DataFrame, cols: list[str], ref=None) -> lgb.Dataset:
    return lgb.Dataset(f[cols].to_numpy(np.float32), f["label"].to_numpy(np.float64), group=np.full(len(f) // 10, 10),
                       reference=ref, free_raw_data=True)


def inner_trees(tr: pd.DataFrame, bounds: list[int], cols: list[str]) -> int:
    last = len(bounds) - 1
    inner = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    hold = tr[tr["window_id"] == last]
    d_in = dataset(inner, cols)
    m = lgb.train({**PARAMS, "seed": lf.FULL_SEEDS[0]}, d_in, num_boost_round=lf.MAX_TREES,
                  valid_sets=[dataset(hold, cols, ref=d_in)],
                  callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
    lf.log(f"  inner holdout ({len(inner):,} / {len(hold):,} rows of clicked events) chose {m.best_iteration} trees")
    return int(m.best_iteration or lf.MAX_TREES)


def fit_predict(tr: pd.DataFrame, target: pd.DataFrame, cols: list[str], n_trees: int, seeds) -> np.ndarray:
    d = dataset(tr, cols)
    Xt = target[cols].to_numpy(np.float32)
    preds = []
    for seed in seeds:
        m = lgb.train({**PARAMS, "seed": seed}, d, num_boost_round=n_trees)
        preds.append(m.predict(Xt))
        lf.log(f"  seed {seed} done")
    return np.mean(preds, axis=0)


def which_auc(score: np.ndarray, y: np.ndarray) -> float:
    s, t = score.reshape(-1, 10), y.reshape(-1, 10)
    k = t.max(1) == 1
    s, t = s[k], t[k]
    pos = s[np.arange(len(s)), t.argmax(1)]
    return float(((s < pos[:, None]).sum(1) + 0.5 * ((s == pos[:, None]).sum(1) - 1)).mean() / 9)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    tr, va = load("step6_train", bounds), load("step6_val", bounds, f"b{split.cutoff_ts}")
    cols = lf.model_columns(tr)
    assert set(NEW) <= set(cols)
    tr = tr[clicked_rows(tr)].reset_index(drop=True)
    lf.log(f"lambdarank: {len(tr):,} training rows of clicked events, {len(cols)} features")
    n_trees = inner_trees(tr, bounds, cols)
    p = fit_predict(tr, va, cols, n_trees, lf.FULL_SEEDS[:1])  # one seed for validation, as softmax_step7.py
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": p}).to_parquet(
        f"data/validation/rank_step7{A.tag}_val_c4.parquet", index=False)
    res = {"seq": A.seq, "swap": A.swap, "share_feats": A.share_feats, "trees": n_trees,
           "which_auc": which_auc(p, va["label"].to_numpy()), "minutes": round((time.perf_counter() - t0) / 60, 1)}
    lf.log(f"RANK STEP 7: {res}")
    (HERE / "outputs" / f"rank_step7{A.tag}.json").write_text(json.dumps(res, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    tr, te = load("step5_full_train", bounds), load("step5_full_test", bounds, "full_eval")
    cols = lf.model_columns(tr)
    assert set(NEW) <= set(cols)
    tr = tr[clicked_rows(tr)].reset_index(drop=True)
    n_trees = inner_trees(tr, bounds, cols)
    p = fit_predict(tr, te, cols, n_trees, lf.FULL_SEEDS)
    out = lf.REPO / "submissions" / f"safe_v2_step7_rank{A.tag}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": p}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if A.submit else validate()
