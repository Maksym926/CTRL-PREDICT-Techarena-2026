"""Detector test for full-rank item-to-item models (EASE and item-kNN), on the inner holdout only.

    python research/techarena/ease_detector.py            # ~50 min on a free CPU, ~8 GB of RAM
    python research/techarena/ease_detector.py --quick    # smoke test: 10% of users, 3k items, small models

Same setup as signal_detectors.py (stage 3 features, fit on windows 0-5, score window 6; validation untouched). For each
window boundary b, on the TOP items with the most clicking users before b:
  G = X'X from the binary user x item click matrix X before b
  EASE (Steck 2019): B = -P / diag(P) with zero diagonal, P = (G + lambda I)^-1;  score(u, j) = x_u . B[:, j]
  item-kNN: cosine S = G_ij / sqrt(G_ii G_jj) with zero diagonal;               score(u, j) = x_u . S[:, j]
with x_u = all of u's clicks before b (_all) or only the last 20 (_last20). Candidates outside TOP get no score (NaN).
Every taste model in the stack is low-rank (SVD 11/32, choice models 11-16, sequence model 64 dims); these keep every
item-item weight. lambda is chosen on window 5 (training rows, so window 6 stays clean) by within-event AUC.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from scipy.linalg import inv
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_detectors as sd  # noqa: E402  (reads --quick, sets cwd to the repo root, loads the stage 3 setup)

lf = sd.lf
TOP = 3000 if sd.QUICK else 20000
LAMBDAS = (50.0, 200.0, 800.0)
OUT = sd.HERE / "outputs" / ("ease_detector_quick.json" if sd.QUICK else "ease_detector.json")
EASE = ["ease_all", "ease_last20", "ease_all_rank", "ease_all_gap"]
KNN = ["knn_all", "knn_last20", "knn_all_rank"]
REFERENCE = ["cmf12_prob", "cmf_prob", "choice_probability", "seq_prob", "svd_cos", "transition_mean"]


def score(xu: sp.csr_matrix, mat: np.ndarray, ccol: np.ndarray) -> np.ndarray:
    """x_u . mat[:, candidate] for every (event, candidate); NaN where the candidate is outside TOP."""
    out = np.full(ccol.shape, np.nan, np.float32)
    for s in range(0, len(ccol), 4096):
        e = min(s + 4096, len(ccol))
        blk = np.asarray(xu[s:e] @ mat)
        cc = ccol[s:e]
        ok = cc >= 0
        out[s:e][ok] = blk[np.nonzero(ok)[0], cc[ok]]
    return out.ravel()


def ease_features(target: pl.DataFrame, b: int, clicks: pl.DataFrame, n_users: int, n_items: int,
                  lambdas: tuple[float, ...]) -> pd.DataFrame:
    """target: rowid, event_id, u, i for all 10 candidates of each target event; models use clicks before b only."""
    t0 = time.perf_counter()
    target = target.sort("event_id", "i")
    n_ev = target.height // 10
    ev_u = target["u"].to_numpy()[::10]
    cand = target["i"].to_numpy().reshape(n_ev, 10)
    pc = clicks.filter(pl.col("ts") < b)
    pairs = pc.unique(["u", "i"])
    deg = np.bincount(pairs["i"].to_numpy(), minlength=n_items)
    keep = np.argsort(-deg, kind="stable")[:TOP]
    keep = keep[deg[keep] > 0]
    col = np.full(n_items, -1, np.int64)
    col[keep] = np.arange(len(keep))
    ccol = col[cand]

    def matrix(u: np.ndarray, i: np.ndarray) -> sp.csr_matrix:
        c = col[i]
        ok = c >= 0
        x = sp.csr_matrix((np.ones(ok.sum(), np.float32), (u[ok], c[ok])), shape=(n_users, len(keep)))
        x.sum_duplicates()
        x.data[:] = 1.0
        return x

    x = matrix(pairs["u"].to_numpy(), pairs["i"].to_numpy())
    last = pc.sort("u", "ts").group_by("u", maintain_order=True).tail(20)
    xs = {"all": x[ev_u], "last20": matrix(last["u"].to_numpy(), last["i"].to_numpy())[ev_u]}
    g = (x.T @ x).toarray().astype(np.float32, copy=False)
    f = {}

    s = g.copy()  # item-kNN: cosine of co-click counts
    root = np.sqrt(np.diag(s)).astype(np.float32) + 1e-6
    s /= root[:, None]
    s /= root[None, :]
    np.fill_diagonal(s, 0.0)
    for name, xu in xs.items():
        f[f"knn_{name}"] = score(xu, s, ccol)
    del s

    for lam in lambdas:  # EASE
        p = g.copy()
        p[np.diag_indices_from(p)] += lam
        p = inv(p, overwrite_a=True, check_finite=False)
        p /= -np.diag(p).copy()[None, :]
        np.fill_diagonal(p, 0.0)
        tag = "" if len(lambdas) == 1 else f"_l{int(lam)}"
        for name, xu in xs.items():
            f[f"ease_{name}{tag}"] = score(xu, p, ccol)
        del p
    lf.log(f"  boundary {b}: {len(keep):,} items modelled ({100 * (ccol >= 0).mean():.1f}% of candidates), "
           f"{len(lambdas)} lambda(s), {time.perf_counter() - t0:.0f} s")
    return pd.DataFrame({"rowid": target["rowid"].to_numpy(), "event_id": target["event_id"].to_numpy(), **f})


def add_ranks(tr: pd.DataFrame) -> pd.DataFrame:
    """Within-event rank (1 = best) and gap to the event's best, which LightGBM cannot derive per row by itself."""
    g = tr.groupby(["window_id", "event_id"])
    tr["ease_all_rank"] = g["ease_all"].rank(ascending=False, method="average")
    tr["ease_all_gap"] = tr["ease_all"] - g["ease_all"].transform("max")
    tr["knn_all_rank"] = g["knn_all"].rank(ascending=False, method="average")
    return tr


def standalone(df: pd.DataFrame, c: str) -> dict:
    y, ev = df["label"].to_numpy(), df["event_id"].to_numpy()
    x = df[c].to_numpy(float)
    x = np.where(np.isnan(x), np.nanmin(x) - 1, x)  # unscored candidates rank last
    return {"global_auc": float(roc_auc_score(y, x)), "within_event_auc": sd.within_auc(ev, y, x),
            "coverage": float(df[c].notna().mean())}


def main() -> None:
    t0 = time.perf_counter()
    tr, cols, bounds = sd.stage3_frame()
    cands, clicks, n_users, n_items = sd.load_logs()
    keys = cands.select("rowid", "event_id", "u", "i")

    # lambda on window 5 (training rows)
    w5 = len(bounds) - 2
    lab = tr.loc[tr["window_id"] == w5, ["rowid", "label"]]
    target = keys.join(pl.DataFrame({"rowid": lab["rowid"].to_numpy()}), on="rowid")
    f5 = ease_features(target, bounds[w5], clicks, n_users, n_items, LAMBDAS).merge(lab, on="rowid")
    choice = {lam: standalone(f5, f"ease_all_l{int(lam)}")["within_event_auc"] for lam in LAMBDAS}
    lam = max(choice, key=choice.get)
    lf.log(f"lambda on window 5 (within-event AUC): {choice} -> {lam}")

    tr = sd.attach(tr, bounds, keys, lambda t, bw: ease_features(t, bw, clicks, n_users, n_items, (lam,)))
    tr = add_ranks(tr)
    iva, arm = sd.inner_arms(tr, cols, bounds)
    res = {"top_items": TOP, "lambda": lam, "lambda_choice_window5": {str(k): v for k, v in choice.items()},
           "standalone_window6": {c: standalone(iva, c) for c in ["ease_all", "ease_last20", "knn_all", "knn_last20"]
                                  + [c for c in REFERENCE if c in iva]}}
    for c, r in res["standalone_window6"].items():
        lf.log(f"  {c:20s} global {r['global_auc']:.4f} | within-event {r['within_event_auc']:.4f} | "
               f"coverage {r['coverage']:.2f}")

    res["arms"] = {}
    base, trees, _, _ = arm([])
    res["arms"]["baseline"] = {"auc": base, "trees": trees}
    lf.log(f"baseline: inner AUC {base:.6f} ({trees} trees)")
    for name, extra in (("ease", EASE), ("ease_knn", EASE + KNN)):
        a, t, _, ranks = arm(extra)
        res["arms"][name] = {"auc": a, "delta": a - base, "trees": t, "new_feature_ranks": ranks}
        lf.log(f"+{name}: inner AUC {a:.6f} ({a - base:+.6f}, {t} trees) ranks {ranks}")
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
