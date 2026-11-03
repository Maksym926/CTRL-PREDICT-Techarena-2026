"""Submission from the step 6 submission frames (cache/lab/step6_full_*.parquet, written by safe_v2_step6.py --submit)
with a lab variant that won on validation: --ensemble (12- and 16-dim filler-matched choice models + choice-model ensemble
columns, lab_ensemble.py) and/or --softmax (event softmax objective, softmax_objective.py).

Usage (from the repo root):
    python research/techarena/lab_submit.py --ensemble            # submissions/safe_v2_step6_ens.csv
    python research/techarena/lab_submit.py --ensemble --softmax  # submissions/safe_v2_step6_ens_softmax.csv
Trees: early stopping on the latest training window (inner holdout), then three seeds averaged, as in the other steps.
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ensemble", action="store_true")
ap.add_argument("--softmax", action="store_true")
ap.add_argument("--base", choices=["step6", "step5"], default="step6",
                help="step5: start from safe_v2_step5.py --submit's frames and add step 6's event-relative columns")
ARGS = ap.parse_args()
assert ARGS.ensemble or ARGS.softmax, "choose --ensemble and/or --softmax"
sys.argv = sys.argv[:1]  # the lab modules parse the command line on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab_ensemble as le  # noqa: E402
import softmax_objective as so  # noqa: E402

lf = le.lf


def load(name: str) -> pd.DataFrame:
    f = pd.read_parquet(le.CACHE / "lab" / f"{ARGS.base}_{name}.parquet")
    if ARGS.base == "step5":
        from safe_v2_step6 import cross_features
        f = cross_features(f)
    keys = [k for k in ("window_id", "event_index", "candidate_index") if k in f]
    return f.sort_values(keys, kind="stable").reset_index(drop=True)


def main() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    tr, te = load("full_train"), load("full_test")
    if ARGS.ensemble:
        tr = le.attach_wide(tr, np.asarray(bounds)[tr["window_id"].to_numpy()], scope="full")
        te = le.attach_wide(te, np.full(len(te), lf.CUT), scope="full")
        tr, te = le.ensemble(tr), le.ensemble(te)
    cols = lf.model_columns(tr)
    arm = "softmax" if ARGS.softmax else "binary"
    last = len(bounds) - 1
    inner = (tr["window_id"] < last) & (tr["timestamp"] < bounds[last])
    hold = tr["window_id"] == last
    b = so.fit(arm, tr.loc[inner, cols].to_numpy(np.float32), tr.loc[inner, "label"].to_numpy(np.float64),
               lf.MAX_TREES, valid=(tr.loc[hold, cols].to_numpy(np.float32), tr.loc[hold, "label"].to_numpy(np.float64)))
    n_trees = b.best_iteration or lf.MAX_TREES
    lf.log(f"{arm} objective, {len(cols)} features: inner holdout chose {n_trees} trees; {len(tr):,} training rows")
    del b
    X, y, Xt = tr[cols].to_numpy(np.float32), tr["label"].to_numpy(np.float64), te[cols].to_numpy(np.float32)
    preds = []
    for seed in lf.FULL_SEEDS:
        params = {**lf.LGB_PARAMS, "seed": seed}
        if ARGS.softmax:
            params["objective"] = so.softmax_objective
            m = lgb.train(params, lgb.Dataset(X, y, init_score=np.full(len(y), so.INIT)), num_boost_round=n_trees)
            preds.append(so.group_softmax(m.predict(Xt, raw_score=True) + so.INIT))
        else:
            m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_trees)
            preds.append(m.predict(Xt))
        lf.log(f"  seed {seed} done")
    tag = "_".join(t for t, on in (("ens", ARGS.ensemble), ("softmax", ARGS.softmax)) if on)
    out = lf.REPO / "submissions" / f"safe_v2_{ARGS.base}_{tag}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": np.mean(preds, axis=0)}).to_csv(out,
                                                                                                          index=False)
    lf.log(f"wrote {out} ({n_trees} trees, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
