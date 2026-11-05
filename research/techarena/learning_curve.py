"""Does LightGBM gain from more distinct training events? Learning curve on the inner holdout.

    python research/techarena/learning_curve.py     # ~25 min on a free CPU

Same setup as signal_detectors.py (stage 3 features, fit on windows 0-5, score window 6; validation untouched). Fits
on 25% / 50% / 100% of users (every row of a chosen user, so whole events go in or out), always scoring all of window 6
at the same LightGBM settings. Training on every event of each half-day window (instead of each user's first event
after the boundary) would give about 4x the distinct events; if halving them costs little, that build is not worth it.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_detectors as sd  # noqa: E402  (sets cwd to the repo root, loads the stage 3 setup)

lf = sd.lf
FRACTIONS = (0.25, 0.5, 1.0)
OUT = sd.HERE / "outputs" / "learning_curve.json"


def main() -> None:
    tr, cols, bounds = sd.stage3_frame()
    last = len(bounds) - 1
    itr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    iva = tr[tr["window_id"] == last]
    y = iva["label"].to_numpy()
    draw = np.random.default_rng(0).random(int(tr["user_id"].max()) + 1)  # one fixed draw per user
    params = {**lf.LGB_PARAMS, "learning_rate": 0.05}
    res = {}
    for frac in FRACTIONS:
        fit = itr[draw[itr["user_id"].to_numpy()] < frac]
        m = lgb.LGBMClassifier(n_estimators=lf.MAX_TREES, random_state=lf.FULL_SEEDS[0], **params)
        m.fit(fit[cols], fit["label"], eval_set=[(iva[cols], y)], eval_metric="auc",
              callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
        auc = float(roc_auc_score(y, m.predict_proba(iva[cols])[:, 1]))
        res[str(frac)] = {"auc": auc, "trees": int(m.best_iteration_), "rows": len(fit),
                          "target_events": int(len(fit) // 10)}  # event-window pairs
        lf.log(f"{frac:.0%} of users: {len(fit):,} rows -> inner AUC {auc:.6f} ({m.best_iteration_} trees)")
    a25, a50, a100 = (res[str(f)]["auc"] for f in FRACTIONS)
    res["per_doubling"] = {"25->50": a50 - a25, "50->100": a100 - a50}
    res["extrapolated_4x"] = 2 * (a100 - a50)  # two more doublings at the last step's slope (optimistic)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nper doubling of distinct events: 25->50% {a50 - a25:+.6f} | 50->100% {a100 - a50:+.6f}")
    print(f"extrapolated gain from 4x the distinct events (two more doublings): {2 * (a100 - a50):+.6f}")


if __name__ == "__main__":
    main()
