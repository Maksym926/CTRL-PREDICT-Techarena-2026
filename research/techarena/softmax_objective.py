"""Does LightGBM do better with the event's own likelihood? Binary log-loss vs a softmax over the 10 candidates plus a
no-click option (utility 0), on the step 6 frames (cache/lab/step6_*.parquet, written by safe_v2_step6.py --dump).

Usage (from the repo root):
    python research/techarena/softmax_objective.py           # inner holdout (latest window), then validation
    python research/techarena/softmax_objective.py --quick   # 10% of events, 300 trees max

Why: a clicked event is the real next item + 9 fillers and a no-click event is 10 fillers, so P(click j) =
exp(f_j) / (1 + sum_k exp(f_k)). The binary objective scores each row alone and can only learn the competition between
candidates through features; this objective builds it in. Gradient p_j - y_j, diagonal Hessian p_j (1 - p_j).
Both arms use the same features, parameters and data; trees are chosen by early stopping on the inner holdout
(windows before the latest one -> the latest window), then each arm is refit on all windows and scored on validation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--submit", action="store_true", help="softmax objective on the submission frames (step6_full_*)")
ap.add_argument("--prefix", default="step6")
ap.add_argument("--threads", type=int, default=-1)
ap.add_argument("--frac", type=float, default=1.0, help="share of users kept in the training windows (both arms)")
ARGS = ap.parse_args()
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402
from src.validation import make_split  # noqa: E402

LAB = HERE / "cache" / "lab"
OUT = HERE / "outputs" / ("softmax_objective_quick.json" if ARGS.quick else "softmax_objective.json")
INIT = float(np.log(0.63 / 0.37 / 10))  # start where 10 e^c / (1 + 10 e^c) = the click-event rate


def user_subsample(frame: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Keep a fixed share of users (whole events stay together); the same users for both arms."""
    if frac >= 1.0:
        return frame
    users = np.unique(frame["user_id"].to_numpy())
    keep = users[np.random.default_rng(20261103).random(len(users)) < frac]
    return frame[frame["user_id"].isin(keep)].reset_index(drop=True)


def group_softmax(raw: np.ndarray) -> np.ndarray:
    f = raw.reshape(-1, 10)
    m = np.maximum(f.max(1, keepdims=True), 0.0)
    e = np.exp(f - m)
    return (e / (np.exp(-m) + e.sum(1, keepdims=True))).ravel()


def softmax_objective(preds, data):
    p = group_softmax(preds)
    return p - data.get_label(), np.maximum(p * (1.0 - p), 1e-7)


def softmax_auc(preds, data):
    return "auc", roc_auc_score(data.get_label(), group_softmax(preds)), True


def decompose(y: np.ndarray, s: np.ndarray) -> dict:
    Y, S = y.reshape(-1, 10), s.reshape(-1, 10)
    c = Y.max(1) == 1
    cs = (S[c] * Y[c]).sum(1)
    return {"global": float(roc_auc_score(y, s)), "which": float(((S[c] < cs[:, None]).sum(1) / 9).mean()),
            "whether": float(roc_auc_score(Y.max(1), S.sum(1)))}


def load(name: str) -> pd.DataFrame:
    f = pd.read_parquet(LAB / f"{ARGS.prefix}_{name}.parquet")
    keys = [k for k in ("window_id", "event_index", "candidate_index") if k in f]
    f = f.sort_values(keys, kind="stable").reset_index(drop=True)
    blk = f["event_index"].to_numpy().reshape(-1, 10)
    assert (blk == blk[:, :1]).all(), f"{name}: events are not blocks of 10 rows"
    if ARGS.quick:
        ev = np.arange(len(f) // 10)
        f = f[np.repeat(ev % 10 == 0, 10)].reset_index(drop=True)
    return f


def fit(arm: str, X, y, n_trees: int, valid=None):
    params = {**lf.LGB_PARAMS, "seed": lf.FULL_SEEDS[0], "n_jobs": ARGS.threads}
    if arm == "softmax":
        params["objective"] = softmax_objective
        train = lgb.Dataset(X, y, init_score=np.full(len(y), INIT), free_raw_data=True)
    else:
        train = lgb.Dataset(X, y, free_raw_data=True)
    kw = {}
    if valid is not None:
        Xv, yv = valid
        vs = lgb.Dataset(Xv, yv, reference=train,
                         init_score=np.full(len(yv), INIT) if arm == "softmax" else None)
        params["metric"] = "None" if arm == "softmax" else "auc"
        kw = dict(valid_sets=[vs], feval=softmax_auc if arm == "softmax" else None,
                  callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
    return lgb.train(params, train, num_boost_round=n_trees, **kw)


def predict(arm: str, booster, X) -> np.ndarray:
    if arm == "softmax":
        return group_softmax(booster.predict(X, raw_score=True) + INIT)
    return booster.predict(X)


def main() -> None:
    t0 = time.perf_counter()
    tr, va = load("train"), load("val")
    tr = user_subsample(tr, ARGS.frac)
    cols = lf.model_columns(tr)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    last = len(bounds) - 1
    inner_tr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    inner_va = tr[tr["window_id"] == last]
    max_trees = 300 if ARGS.quick else lf.MAX_TREES
    lf.log(f"{len(tr):,} training rows ({len(inner_tr):,} inner fit / {len(inner_va):,} inner holdout), "
           f"{len(va):,} validation rows, {len(cols)} features")
    Xi, yi = inner_tr[cols].to_numpy(np.float32), inner_tr["label"].to_numpy(np.float64)
    Xh, yh = inner_va[cols].to_numpy(np.float32), inner_va["label"].to_numpy(np.float64)
    res, inner_pred = {}, {}
    for arm in ("binary", "softmax"):
        b = fit(arm, Xi, yi, max_trees, valid=(Xh, yh))
        n = b.best_iteration or max_trees
        inner_pred[arm] = predict(arm, b, Xh)
        res[arm] = {"trees": int(n), "inner": decompose(yh, inner_pred[arm])}
        lf.log(f"  {arm}: inner holdout {res[arm]['inner']} with {n} trees")
    blend = 0.5 * (pd.Series(inner_pred["binary"]).rank().to_numpy() + pd.Series(inner_pred["softmax"]).rank().to_numpy())
    res["rank_blend_inner"] = decompose(yh, blend)
    lf.log(f"  rank blend: inner holdout {res['rank_blend_inner']}")
    del Xi, yi

    split = make_split(lf.VAL_DAY)
    X, y = tr[cols].to_numpy(np.float32), tr["label"].to_numpy(np.float64)
    Xv = va[cols].to_numpy(np.float32)
    lab = split.val_candidates.select("rowid", "clicked").to_pandas()
    yv = va[["rowid"]].astype(str).merge(lab, on="rowid", how="left")["clicked"].to_numpy()
    assert not np.isnan(yv.astype(float)).any() or ARGS.quick
    keep = ~pd.isna(yv)
    val_pred = {}
    for arm in ("binary", "softmax"):
        b = fit(arm, X, y, res[arm]["trees"])
        val_pred[arm] = predict(arm, b, Xv)
        res[arm]["validation"] = decompose(yv[keep].astype(float), val_pred[arm][keep])
        lf.log(f"  {arm}: validation {res[arm]['validation']}")
    vb = 0.5 * (pd.Series(val_pred["binary"]).rank().to_numpy() + pd.Series(val_pred["softmax"]).rank().to_numpy())
    res["rank_blend_validation"] = decompose(yv[keep].astype(float), vb[keep])
    lf.log(f"  rank blend: validation {res['rank_blend_validation']}")
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def submit() -> None:
    """Softmax objective on the submission frames: trees from the latest training window, three seeds averaged."""
    t0 = time.perf_counter()
    tr, te = load("full_train"), load("full_test")
    cols = lf.model_columns(tr)
    bounds = lf.bounds_for(7, half=True)
    last = len(bounds) - 1
    inner_tr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    inner_va = tr[tr["window_id"] == last]
    b = fit("softmax", inner_tr[cols].to_numpy(np.float32), inner_tr["label"].to_numpy(np.float64), lf.MAX_TREES,
            valid=(inner_va[cols].to_numpy(np.float32), inner_va["label"].to_numpy(np.float64)))
    n_trees = b.best_iteration or lf.MAX_TREES
    lf.log(f"softmax submission: inner holdout chose {n_trees} trees; {len(tr):,} training rows, {len(cols)} features")
    del inner_tr, inner_va, b
    X, y, Xt = tr[cols].to_numpy(np.float32), tr["label"].to_numpy(np.float64), te[cols].to_numpy(np.float32)
    preds = []
    for seed in lf.FULL_SEEDS:
        params = {**lf.LGB_PARAMS, "seed": seed, "objective": softmax_objective}
        m = lgb.train(params, lgb.Dataset(X, y, init_score=np.full(len(y), INIT)), num_boost_round=n_trees)
        preds.append(group_softmax(m.predict(Xt, raw_score=True) + INIT))
        lf.log(f"  seed {seed} done")
    out = lf.REPO / "submissions" / f"safe_v2_{ARGS.prefix}_softmax.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": np.mean(preds, axis=0)}).to_csv(out,
                                                                                                          index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if ARGS.submit else main()
