"""Softmax-objective LightGBM on step 7's features (step 5 + 50-click sequence columns): the blend partner of step 7.

The current partner (submissions/safe_v2_step5_softmax.csv) was trained on step 5's features, without the 50-click
columns that gave the binary model +0.00043. This is the same objective (softmax_objective.py, default LightGBM
parameters as before) on exactly the frames safe_v2_step7.py uses: step 5's saved frames + seq50_* from
cache/seq/<--seq>.

Usage (from the repo root):
    python research/techarena/softmax_step7.py                     # validation: alone and rank-blended with step 7
    python research/techarena/softmax_step7.py --submit            # submissions/safe_v2_step7_softmax.csv (3 seeds)
    python research/techarena/softmax_step7.py --seq stage3_h50_x --binary data/validation/<binary run>_val_c4.parquet

Trees: early stopping on the latest training window (inner holdout), then a refit on all windows. Validation uses one
seed, as softmax_blend_val.py did for the current partner; the submission averages three.
Reference: blend(step 7, current partner) = 0.807134 on validation (experiment 005, LB 0.821624).
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
ap.add_argument("--seq", default="stage3_h50", help="folder under cache/seq with the 50-click features")
ap.add_argument("--binary", default="data/validation/safe_v2_step7_val_c4.parquet",
                help="validation predictions of the binary model to blend with")
ap.add_argument("--submit", action="store_true")
ap.add_argument("--threads", type=int, default=-1)
ap.add_argument("--tag", default="", help="suffix for the output names")
A = ap.parse_args()
sys.argv = [sys.argv[0], "--threads", str(A.threads)]  # softmax_objective parses the command line on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import softmax_objective as so  # noqa: E402
from src.validation import make_split  # noqa: E402

lf = so.lf
LAB = HERE / "cache" / "lab"
SEQ_DIR = HERE / "cache" / "seq" / A.seq
NEW = ["seq50_prob", "seq50_noclick", "seq50_rank"]
OLD_PARTNER = "data/validation/softmax_step6_val_c4.parquet"  # validation predictions of the current partner
REFERENCE = 0.807134


def add_seq50(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    s = pd.read_parquet(SEQ_DIR / f"seq_{name}.parquet")
    s = s.assign(rowid=s["rowid"].astype(str)).rename(columns=dict(zip(["seq_prob", "seq_noclick", "seq_rank"], NEW)))
    out = frame.assign(rowid=frame["rowid"].astype(str)).merge(s, on="rowid", how="left")
    if len(out) != len(frame) or out[NEW].isna().any().any():
        raise SystemExit(f"seq_{name}: rows without 50-click sequence features")
    return out


def load(name: str, bounds: list[int], target: str | None = None) -> pd.DataFrame:
    """A saved frame without step 6's columns, plus seq50_*: per training window, or from `target` for a target set.
    Sorted into blocks of 10 rows per event, as the softmax objective needs."""
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


def inner_trees(tr: pd.DataFrame, bounds: list[int], cols: list[str]) -> int:
    last = len(bounds) - 1
    inner = (tr["window_id"] < last) & (tr["timestamp"] < bounds[last])
    hold = tr["window_id"] == last
    b = so.fit("softmax", tr.loc[inner, cols].to_numpy(np.float32), tr.loc[inner, "label"].to_numpy(np.float64),
               lf.MAX_TREES, valid=(tr.loc[hold, cols].to_numpy(np.float32), tr.loc[hold, "label"].to_numpy(np.float64)))
    return b.best_iteration or lf.MAX_TREES


def rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy() / len(x)


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    tr, va = load("step6_train", bounds), load("step6_val", bounds, f"b{split.cutoff_ts}")
    cols = lf.model_columns(tr)
    assert set(NEW) <= set(cols)
    n_trees = inner_trees(tr, bounds, cols)
    lf.log(f"softmax: inner holdout chose {n_trees} trees ({len(tr):,} training rows, {len(cols)} features)")
    b = so.fit("softmax", tr[cols].to_numpy(np.float32), tr["label"].to_numpy(np.float64), n_trees)
    p = so.predict("softmax", b, va[cols].to_numpy(np.float32))
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": p}).to_parquet(
        f"data/validation/softmax_step7{A.tag}_val_c4.parquet", index=False)

    lab = split.val_candidates.select("rowid", "clicked").to_pandas()
    d = va[["rowid"]].astype(str).assign(soft7=p).merge(lab, on="rowid")
    for name, path in (("binary", A.binary), ("soft_old", OLD_PARTNER)):
        q = pd.read_parquet(path)
        d = d.merge(q.assign(rowid=q["rowid"].astype(str)).rename(columns={"predicted_probability": name}), on="rowid")
    assert len(d) == len(va), "validation rows lost in the merges"
    y = d["clicked"].to_numpy(np.float64)
    res = {"seq": A.seq, "binary": A.binary, "trees": int(n_trees),
           "softmax7_alone": so.decompose(y, d["soft7"].to_numpy()),
           "binary_alone": so.decompose(y, d["binary"].to_numpy()),
           "blend_binary_old_partner": so.decompose(y, 0.5 * (rank(d["binary"].to_numpy()) + rank(d["soft_old"].to_numpy()))),
           "blend_binary_softmax7": so.decompose(y, 0.5 * (rank(d["binary"].to_numpy()) + rank(d["soft7"].to_numpy())))}
    res["delta_vs_old_partner"] = res["blend_binary_softmax7"]["global"] - res["blend_binary_old_partner"]["global"]
    res["delta_vs_005"] = res["blend_binary_softmax7"]["global"] - REFERENCE
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    for k, v in res.items():
        lf.log(f"  {k}: {v}")
    (HERE / "outputs" / f"softmax_step7{A.tag}.json").write_text(json.dumps(res, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    tr, te = load("step5_full_train", bounds), load("step5_full_test", bounds, "full_eval")
    cols = lf.model_columns(tr)
    assert set(NEW) <= set(cols)
    n_trees = inner_trees(tr, bounds, cols)
    lf.log(f"softmax submission: inner holdout chose {n_trees} trees; {len(tr):,} training rows, {len(cols)} features")
    X, y, Xt = tr[cols].to_numpy(np.float32), tr["label"].to_numpy(np.float64), te[cols].to_numpy(np.float32)
    del tr
    preds = []
    for seed in lf.FULL_SEEDS:
        params = {**lf.LGB_PARAMS, "seed": seed, "objective": so.softmax_objective, "n_jobs": A.threads}
        m = lgb.train(params, lgb.Dataset(X, y, init_score=np.full(len(y), so.INIT)), num_boost_round=n_trees)
        preds.append(so.group_softmax(m.predict(Xt, raw_score=True) + so.INIT))
        lf.log(f"  seed {seed} done")
    out = lf.REPO / "submissions" / f"safe_v2_step7_softmax{A.tag}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": np.mean(preds, axis=0)}).to_csv(out,
                                                                                                          index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if A.submit else validate()
