"""Full-data check of the binary + softmax blend before building it: a softmax-objective LightGBM on the step 6 frames
(all users), rank-blended 50/50 with the actual 3-seed binary validation predictions of step 5 (the CSV it would be
blended with) and of step 6.

Usage (from the repo root, after safe_v2_step6.py --dump):
    python research/techarena/softmax_blend_val.py [--threads 6]

Why: on half the users (softmax_objective.py), the softmax objective alone was slightly worse than binary log-loss
(validation 0.804896 vs 0.805008) but their 50/50 rank blend was better on both the inner holdout (+0.00045) and
validation (+0.00050). Part of that may be plain model averaging, which the 3-seed binary already has.
Decision rule, fixed before seeing the result: build the blend submission only if blend(step 5, softmax) reaches
step 5 + 0.0005 = 0.806942 on validation. One softmax seed here; the submission would average three.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=6)
A = ap.parse_args()
sys.argv = [sys.argv[0], "--threads", str(A.threads)]  # softmax_objective parses the command line on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import softmax_objective as so  # noqa: E402
from src.validation import make_split  # noqa: E402

lf = so.lf
OUT = HERE / "outputs" / "softmax_blend_val.json"
STEP5 = 0.806442
MIN_GAIN = 0.0005


def rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy() / len(x)


def main() -> None:
    t0 = time.perf_counter()
    tr, va = so.load("train"), so.load("val")
    cols = lf.model_columns(tr)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    last = len(bounds) - 1
    inner = (tr["window_id"] < last) & (tr["timestamp"] < bounds[last])
    hold = tr["window_id"] == last
    b = so.fit("softmax", tr.loc[inner, cols].to_numpy(np.float32), tr.loc[inner, "label"].to_numpy(np.float64),
               lf.MAX_TREES, valid=(tr.loc[hold, cols].to_numpy(np.float32), tr.loc[hold, "label"].to_numpy(np.float64)))
    n_trees = b.best_iteration or lf.MAX_TREES
    lf.log(f"softmax: inner holdout chose {n_trees} trees ({len(tr):,} training rows, {len(cols)} features)")
    del b
    b = so.fit("softmax", tr[cols].to_numpy(np.float32), tr["label"].to_numpy(np.float64), n_trees)
    p_soft = so.predict("softmax", b, va[cols].to_numpy(np.float32))
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": p_soft}).to_parquet(
        "data/validation/softmax_step6_val_c4.parquet", index=False)

    lab = make_split(lf.VAL_DAY).val_candidates.select("rowid", "clicked").to_pandas()
    d = va[["rowid"]].astype(str).assign(soft=p_soft).merge(lab, on="rowid")
    for name in ("step5", "step6"):
        p = pd.read_parquet(f"data/validation/safe_v2_{name}_val_c4.parquet")
        d = d.merge(p.assign(rowid=p["rowid"].astype(str)).rename(columns={"predicted_probability": name}), on="rowid")
    y = d["clicked"].to_numpy(np.float64)
    res = {"trees": int(n_trees), "softmax_alone": so.decompose(y, d["soft"].to_numpy())}
    for name in ("step5", "step6"):
        res[f"{name}_alone"] = so.decompose(y, d[name].to_numpy())
        res[f"blend_{name}_softmax"] = so.decompose(y, 0.5 * (rank(d[name].to_numpy()) + rank(d["soft"].to_numpy())))
    gain = res["blend_step5_softmax"]["global"] - STEP5
    res["delta_blend_step5_vs_step5"] = gain
    res["decision"] = "SUBMIT" if gain >= MIN_GAIN else "DON'T SUBMIT"
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    for k, v in res.items():
        lf.log(f"  {k}: {v}")
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
