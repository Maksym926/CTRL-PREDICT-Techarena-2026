"""Feature lab on the step 6 frames (cache/lab/step6_*.parquet, from safe_v2_step6.py --dump): does a pre-averaged
choice-model ensemble help the stack more than the separate model columns it already has?

Usage (from the repo root, after the 12- and 16-dim filler-matched caches exist for the val scope):
    python research/techarena/lab_ensemble.py [--threads 6]

Why: averaging filler-matched choice models of different widths beats each one alone by a wide margin (validation:
11 dims 0.789461, 12 dims 0.788599, 16 dims 0.781110; mean of 11+12 0.791187, of 11+12+16 0.792005), because widths
start from different SVD subspaces and their taste errors partly cancel. The stack already has four choice models as
separate columns, so the question is what it gains from two more and from pre-averaged ensemble columns.
Arms, same data and LightGBM parameters: A = step 6 features; B = A + the 12- and 16-dim filler-matched models
(prob / no-click / rank) + the mean and spread over the six choice models of each candidate's logit, and the
event-renormalised mean probability with its rank. Trees per arm by early stopping on the inner holdout (latest
window), then one refit per arm on every window, scored on validation.
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
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--frac", type=float, default=0.5, help="share of users kept in the training windows (both arms)")
ARGS = ap.parse_args()
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402
from safe_v2_step1 import CACHE  # noqa: E402
from src.validation import make_split  # noqa: E402

OUT = HERE / "outputs" / "lab_ensemble.json"
WIDTHS = (12, 16)  # extra filler-matched choice models, from cache/choicemf_pop_d{w}/<scope>/
MODELS = ["cmf_prob", "cmf12_prob", "cmf16_prob", "cmfp_prob"] + [f"cmfp{w}_prob" for w in WIDTHS]


def load(name: str) -> pd.DataFrame:
    f = pd.read_parquet(CACHE / "lab" / f"step6_{name}.parquet")
    keys = [k for k in ("window_id", "event_index", "candidate_index") if k in f]
    return f.sort_values(keys, kind="stable").reset_index(drop=True)


def user_subsample(frame: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Keep a fixed share of users (whole events stay together); the same users for every arm."""
    if frac >= 1.0:
        return frame
    users = np.unique(frame["user_id"].to_numpy())
    keep = users[np.random.default_rng(20261103).random(len(users)) < frac]
    return frame[frame["user_id"].isin(keep)].reset_index(drop=True)


def attach_wide(frame: pd.DataFrame, boundary_of_row: np.ndarray, scope: str = "val") -> pd.DataFrame:
    for w in WIDTHS:
        parts = []
        for b in np.unique(boundary_of_row):
            sub = frame[boundary_of_row == b]
            c = pd.read_parquet(CACHE / f"choicemf_pop_d{w}" / scope / f"{b}.parquet")
            c = c.rename(columns={k: k.replace("cmf_", f"cmfp{w}_") for k in ("cmf_prob", "cmf_noclick", "cmf_rank")})
            m = sub[["rowid"]].astype(str).merge(c.assign(rowid=c["rowid"].astype(str)), on="rowid", how="left")
            assert len(m) == len(sub) and m[f"cmfp{w}_prob"].notna().all(), f"boundary {b}: width {w} missing"
            parts.append(m.drop(columns="rowid").set_index(sub.index))
        frame = frame.join(pd.concat(parts))
    return frame


def ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    p = np.clip(frame[MODELS].to_numpy(np.float64), 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))
    mean_p = p.mean(1).reshape(-1, 10)
    share = mean_p / mean_p.sum(1, keepdims=True)
    return frame.assign(ens_logit_mean=lo.mean(1).astype(np.float32), ens_logit_std=lo.std(1).astype(np.float32),
                        ens_share=share.ravel().astype(np.float32),
                        ens_share_rank=(-share).argsort(1).argsort(1).ravel().astype(np.float32))


def decompose(y: np.ndarray, s: np.ndarray) -> dict:
    Y, S = y.reshape(-1, 10), s.reshape(-1, 10)
    c = Y.max(1) == 1
    cs = (S[c] * Y[c]).sum(1)
    return {"global": float(roc_auc_score(y, s)), "which": float(((S[c] < cs[:, None]).sum(1) / 9).mean()),
            "whether": float(roc_auc_score(Y.max(1), S.sum(1)))}


def main() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    tr, va = load("train"), load("val")
    tr = user_subsample(tr, ARGS.frac)
    tr = attach_wide(tr, np.asarray(bounds)[tr["window_id"].to_numpy()])
    va = attach_wide(va, np.full(len(va), split.cutoff_ts))
    tr, va = ensemble(tr), ensemble(va)
    base = [c for c in lf.model_columns(tr) if not c.startswith(tuple(f"cmfp{w}_" for w in WIDTHS) + ("ens_",))]
    arms = {"A_step6": base, "B_plus_wide_and_ensemble": lf.model_columns(tr)}
    lab = split.val_candidates.select("rowid", "clicked").to_pandas()
    yv = va[["rowid"]].astype(str).merge(lab, on="rowid", how="left")["clicked"].to_numpy(np.float64)
    single = {c: decompose(yv, va[c].to_numpy()) for c in MODELS + ["ens_share"]}
    for c, r in single.items():
        lf.log(f"  alone on validation: {c} {r}")
    last = len(bounds) - 1
    inner = (tr["window_id"] < last) & (tr["timestamp"] < bounds[last])
    hold = tr["window_id"] == last
    params = {**lf.LGB_PARAMS, "n_jobs": ARGS.threads}
    res = {"alone": single}
    for name, cols in arms.items():
        m = lgb.LGBMClassifier(n_estimators=lf.MAX_TREES, random_state=lf.FULL_SEEDS[0], **params)
        m.fit(tr.loc[inner, cols], tr.loc[inner, "label"], eval_set=[(tr.loc[hold, cols], tr.loc[hold, "label"])],
              eval_metric="auc", callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
        n = int(m.best_iteration_)
        ih = decompose(tr.loc[hold, "label"].to_numpy(np.float64), m.predict_proba(tr.loc[hold, cols])[:, 1])
        m = lgb.LGBMClassifier(n_estimators=n, random_state=lf.FULL_SEEDS[0], **params).fit(tr[cols], tr["label"])
        v = decompose(yv, m.predict_proba(va[cols])[:, 1])
        res[name] = {"features": len(cols), "trees": n, "inner": ih, "validation": v}
        lf.log(f"  {name}: {len(cols)} features, {n} trees | inner {ih} | validation {v}")
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
