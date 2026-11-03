"""Ceiling test: how much AUC does a choice model lose purely to estimation noise, with this much data?

    python research/techarena/ceiling_test.py     # ~25 min on CPU

A history-free world, so nothing real leaks into the simulated one:
  1. Generator ("truth"): choice model v2 settings (11 dims, SVD start, auxiliary clicks) fitted to ALL real train
     events, then used without history inputs: utility = user vector . item vector + item bias, no-click = user bias.
  2. Simulated labels for every real train event (same users, slates and times), sampled from its softmax; also with
     sharper utilities (x TEMPS) in case the fitted generator is smoother than the real world.
  3. Estimator: the same model trained on the simulated labels before day 4 only (epochs from the inner holdout, SVD
     start and auxiliary clicks from simulated event clicks only), scored on each user's first event from day 4 on.
  4. Reference: the same estimator trained on the REAL labels and scored on the real validation targets. The world
     whose estimator lands nearest this reference is the realistic one.
ceiling = AUC of the true probabilities on the simulated targets; estimation loss = ceiling - estimator AUC.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import src.choice_mf as cmf  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402

DAY = 86_400
D0, CUT = 1_737_331_200, 1_737_936_000
B = D0 + 4 * DAY  # validation boundary
ARGS = SimpleNamespace(dim=11, svd_rank=11, scale=4.0, l2=0.01, lr=3e-3, batch=1024, seed=0, user_emb=1, no_time=1,
                       aux_weight=0.3, aux_ratio=2, aux_neg=20)  # choice model v2 (safe_v2_step1.CMF2_ARGS)
GEN_EPOCHS, MAX_EPOCHS = 2, 4
TEMPS = (1.0, 1.5)
OUT = HERE / "outputs" / "ceiling_test.json"


def subset(d: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in d.items()}


def no_history(d: dict, pad: int) -> dict:
    d = dict(d)
    d["hist"] = np.full_like(d["hist"], pad)
    d["rep"], d["noclick"] = np.zeros_like(d["rep"]), np.zeros_like(d["noclick"])
    return d


def first_from(d: dict, start: int, end: int | None = None) -> np.ndarray:
    """Each user's first event in [start, end); rows are already in (timestamp, event_id) order."""
    idx = np.flatnonzero((d["ts"] >= start) & (d["ts"] < (end if end is not None else np.iinfo(np.int64).max)))
    _, first = np.unique(d["user"][idx], return_index=True)
    return np.sort(idx[first])


def event_clicks(d: dict, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (user, item) pairs clicked in events before b (labels of d only, no histories)."""
    m = (d["ts"] < b) & (d["label"] > 0)
    pairs = np.unique(np.stack([d["user"][m], d["cand"][m, d["label"][m] - 1]], 1), axis=0)
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def event_svd(d: dict, b: int, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """cmf.click_svd's normalised SVD, from event clicks before b only."""
    r, c = event_clicks(d, b)
    du, di = np.bincount(r, minlength=n_users) + 1.0, np.bincount(c, minlength=n_items) + 1.0
    m = sp.coo_matrix((1 / np.sqrt(du[r] * di[c]), (r, c)), shape=(n_users, n_items)).tocsr()
    u, s, vt = svds(m, k=ARGS.svd_rank, random_state=20261024)

    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return unit(u * s).astype(np.float32), unit(vt.T).astype(np.float32)


def logits_of(model: cmf.ChoiceMF, d: dict, batch: int = 8192) -> np.ndarray:
    model.eval()
    x = cmf.tensors(d)
    out = []
    with torch.no_grad():
        for s in range(0, len(d["user"]), batch):
            out.append(model(*(t[s:s + batch] for t in x))[0].numpy())
    return np.concatenate(out)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def decompose(y: np.ndarray, p: np.ndarray) -> dict:
    """Global AUC, which-item (pairwise share in clicked events), whether-click (event AUC of summed probability)."""
    Y, P = y.reshape(-1, 10), p.reshape(-1, 10)
    c = Y.max(1) == 1
    cp = (P[c] * Y[c]).sum(1)
    return {"global": float(roc_auc_score(y, p)), "which": float(((P[c] < cp[:, None]).sum(1) / 9).mean()),
            "whether": float(roc_auc_score(Y.max(1), P.sum(1)))}


def estimate(d: dict, n_users: int, n_items: int, name: str) -> tuple[dict, np.ndarray]:
    """Fit on events before B (epochs from the inner holdout), score each user's first event from B on."""
    t0 = time.perf_counter()
    inner_b = B - DAY // 2
    hold = subset(d, first_from(d, inner_b, B))
    _, curve = cmf.train(subset(d, np.flatnonzero(d["ts"] < inner_b)), n_users, n_items, ARGS, MAX_EPOCHS,
                         event_svd(d, inner_b, n_users, n_items), holdout=hold, clicks=event_clicks(d, inner_b))
    epochs = int(np.argmax(curve)) + 1
    model, _ = cmf.train(subset(d, np.flatnonzero(d["ts"] < B)), n_users, n_items, ARGS, epochs,
                         event_svd(d, B, n_users, n_items), clicks=event_clicks(d, B))
    val = first_from(d, B)
    p = cmf.predict(model, subset(d, val))
    y = (np.arange(1, 11) == d["label"][val][:, None]).reshape(-1).astype(int)
    res = {"epochs": epochs, "inner_curve": curve, **decompose(y, p), "minutes": round((time.perf_counter() - t0) / 60, 1)}
    cmf.log(f"{name}: estimator AUC {res['global']:.4f} (which {res['which']:.4f}, whether {res['whether']:.4f}), "
            f"{epochs} epoch(s)")
    return res, val


def main() -> None:
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    umap, imap = cmf.code_maps()
    n_users, n_items = len(umap), len(imap)
    te = load_events("train").sort("timestamp", "event_id")
    d = no_history(cmf.encode(te, load_candidates("train"), umap, imap), pad=n_items)
    cmf.log(f"{len(d['user']):,} train events, {n_users:,} users, {n_items:,} items")

    # 1. generator on all real data (real SVD start and real auxiliary clicks, histories included)
    gen, _ = cmf.train(d, n_users, n_items, ARGS, GEN_EPOCHS, cmf.click_svd(CUT, umap, imap, ARGS.svd_rank),
                       clicks=cmf.click_pairs(CUT, umap, imap))
    true_logits = logits_of(gen, d)
    cmf.log("generator fitted")

    # 4. reference: the estimator on the real labels
    res = {"reference_real": estimate(d, n_users, n_items, "REAL labels (reference)")[0], "worlds": {}}

    rng = np.random.default_rng(20261102)
    for temp in TEMPS:
        p_true = softmax(true_logits * temp)
        sim = dict(d)
        sim["label"] = np.minimum((p_true.cumsum(1) < rng.random(len(p_true))[:, None]).sum(1), 10)  # float slack
        est, val = estimate(sim, n_users, n_items, f"simulated world x{temp}")
        y = (np.arange(1, 11) == sim["label"][val][:, None]).reshape(-1).astype(int)
        ceiling = decompose(y, p_true[val, 1:].reshape(-1))
        res["worlds"][str(temp)] = {"click_rate": float((sim["label"] > 0).mean()), "ceiling": ceiling,
                                    "estimator": est, "estimation_loss": ceiling["global"] - est["global"]}
        cmf.log(f"world x{temp}: click rate {(sim['label'] > 0).mean():.3f} | ceiling {ceiling['global']:.4f} "
                f"(which {ceiling['which']:.4f}, whether {ceiling['whether']:.4f}) | estimator {est['global']:.4f} "
                f"| estimation loss {ceiling['global'] - est['global']:+.4f}")

    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.write_text(json.dumps(res, indent=2))
    ref = res["reference_real"]["global"]
    print(f"\n==== CEILING TEST (validation-style targets, history-free choice model) ====")
    print(f"real labels, estimator: {ref:.4f}   (real click rate {(d['label'] > 0).mean():.3f})")
    for t, w in res["worlds"].items():
        print(f"world x{t}: estimator {w['estimator']['global']:.4f} (vs real {w['estimator']['global'] - ref:+.4f}) | "
              f"ceiling {w['ceiling']['global']:.4f} | estimation loss {w['estimation_loss']:+.4f}")
    print(f"{res['minutes']} min, wrote {OUT}")


if __name__ == "__main__":
    main()
