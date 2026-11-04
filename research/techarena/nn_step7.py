"""Neural stack on step 7's frames: an MLP over the same 140 columns as the LightGBM models, trained with the event
softmax (10 candidates + no-click), as a differently-biased partner for final_blend.py (--val-c / --sub-c).

Same frames and protocol as safe_v2_step7.py / softmax_step7.py: step 5's saved frames + seq50_* from cache/seq/<--seq>
(optional --swap); epochs chosen on the inner holdout (training windows dated before the latest window -> that window),
then a refit on all windows. Columns: rank-Gauss transform fitted on the training rows (NaN -> the median), so the
heavy-tailed counts and probabilities all look standard normal.
    candidate utility u_i = MLP(x_i)        no-click utility u_0 = MLP0([mean_i h_i, max_i h_i])
    P(click i) = softmax(u_0, u_1..u_10)_i
--context: u_i = MLP([h_i, h_i - mean_j h_j, max_j h_j]) instead (the candidate sees the rest of its event).
Validation averages --seeds fits (default 3); the submission the same.

Usage (from the repo root):
    python research/techarena/nn_step7.py --seq stage3_h50_s30 --tag _s30            # validation
    python research/techarena/nn_step7.py --seq stage3_h50_s30 --tag _s30 --submit   # submissions/safe_v2_step7_nn_s30.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="stage3_h50_s30", help="folder under cache/seq with the 50-click features")
ap.add_argument("--submit", action="store_true")
ap.add_argument("--tag", default="", help="suffix for the output names")
ap.add_argument("--swap", nargs="*", default=[], help="prefix=folder: as safe_v2_step7.py --swap")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed-offset", type=int, default=0, help="refit seeds k + offset (the inner holdout fit is unchanged)")
ap.add_argument("--max-epochs", type=int, default=12)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=1024, help="events per batch")
ap.add_argument("--wd", type=float, default=1e-5, help="AdamW weight decay")
ap.add_argument("--context", action="store_true",
                help="candidate head also sees h_i - mean_j h_j and max_j h_j (the rest of its event)")
ap.add_argument("--threads", type=int, default=8)
A = ap.parse_args()
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (chdir to the repo root)
from src.validation import make_split  # noqa: E402

torch.set_num_threads(A.threads)
LAB = HERE / "cache" / "lab"
SEQ_DIR = HERE / "cache" / "seq" / A.seq
NEW = ["seq50_prob", "seq50_noclick", "seq50_rank"]
QS = np.linspace(0, 1, 1001)


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


class RankGauss:
    """Per column: training quantiles -> uniform -> standard normal; NaN -> 0 (the median). A value shared by many rows
    (e.g. 0 for 60% of them) maps to the middle of its run of quantiles."""

    def fit(self, X: np.ndarray) -> "RankGauss":
        self.maps = []
        for j in range(X.shape[1]):
            q = np.nanquantile(X[:, j], QS)
            if np.isnan(q).all():
                self.maps.append(None)
                continue
            uq, inv = np.unique(q, return_inverse=True)
            self.maps.append((uq, np.bincount(inv, weights=QS) / np.bincount(inv)))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape, dtype=np.float32)
        for j, m in enumerate(self.maps):
            if m is None or len(m[0]) == 1:
                continue
            x = X[:, j]
            z = ndtri(np.clip(np.interp(x, m[0], m[1]), 0.0005, 0.9995))
            z[np.isnan(x)] = 0.0
            out[:, j] = z
        return out


class EventMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float, context: bool = False):
        super().__init__()
        self.context = context
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout))
        self.cand = (nn.Sequential(nn.Linear(3 * (hidden // 2), hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
                     if context else nn.Linear(hidden // 2, 1))
        self.noclick = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: events x 10 x features -> logits events x 11
        h = self.body(x)
        if self.context:
            u = self.cand(torch.cat([h, h - h.mean(1, keepdim=True), h.amax(1, keepdim=True).expand_as(h)], -1))
        else:
            u = self.cand(h)
        u = u.squeeze(-1)
        u0 = self.noclick(torch.cat([h.mean(1), h.amax(1)], -1))
        return torch.cat([u0, u], 1)


def labels(f: pd.DataFrame) -> np.ndarray:
    y = f["label"].to_numpy().reshape(-1, 10)
    return np.where(y.max(1) > 0, y.argmax(1) + 1, 0)


def predict(model: EventMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + 8192])), 1)[:, 1:].numpy())
    return np.concatenate(out).ravel()


def train(X: np.ndarray, y: np.ndarray, epochs: int, seed: int, hold=None):
    """X: events x 10 x features. Returns the model and the per-epoch holdout AUC curve (if hold)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EventMLP(X.shape[2], A.hidden, A.dropout, A.context)
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    curve = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        for i in range(0, len(order), A.batch):
            idx = torch.from_numpy(order[i:i + A.batch])
            loss = nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if hold is not None:
            p = predict(model, hold[0])
            curve.append(float(roc_auc_score(hold[1], p)))
            lf.log(f"    epoch {ep + 1}: inner-holdout AUC {curve[-1]:.6f}")
    return model, curve


def run(tr: pd.DataFrame, target: pd.DataFrame, bounds: list[int]) -> tuple[np.ndarray, dict]:
    cols = lf.model_columns(tr)
    assert set(NEW) <= set(cols)
    last = len(bounds) - 1
    inner = ((tr["window_id"] < last) & (tr["timestamp"] < bounds[last])).to_numpy()
    hold = (tr["window_id"] == last).to_numpy()
    raw = tr[cols].to_numpy(np.float32)
    y_all = labels(tr)
    ev_inner, ev_hold = inner.reshape(-1, 10)[:, 0], hold.reshape(-1, 10)[:, 0]
    rg = RankGauss().fit(raw[inner])
    Xi = rg.transform(raw[inner]).reshape(-1, 10, len(cols))
    Xh = rg.transform(raw[hold]).reshape(-1, 10, len(cols))
    yh = tr["label"].to_numpy()[hold]
    lf.log(f"NN: {len(cols)} features, inner {ev_inner.sum():,} / holdout {ev_hold.sum():,} events")
    _, curve = train(Xi, y_all[ev_inner], A.max_epochs, lf.FULL_SEEDS[0], hold=(Xh, yh))
    epochs = int(np.argmax(curve)) + 1
    lf.log(f"NN: inner holdout chose {epochs} epochs (AUC {max(curve):.6f})")
    del Xi, Xh
    rg = RankGauss().fit(raw)
    X = rg.transform(raw).reshape(-1, 10, len(cols))
    Xt = rg.transform(target[cols].to_numpy(np.float32)).reshape(-1, 10, len(cols))
    del raw
    preds = []
    for k in range(A.seeds):
        model, _ = train(X, y_all, epochs, lf.FULL_SEEDS[0] + 101 * (k + A.seed_offset))
        preds.append(predict(model, Xt))
        lf.log(f"  NN seed {k} done")
    return np.mean(preds, axis=0), {"epochs": epochs, "inner_curve": curve, "features": len(cols)}


def validate() -> None:
    t0 = time.perf_counter()
    split = make_split(lf.VAL_DAY)
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    tr, va = load("step6_train", bounds), load("step6_val", bounds, f"b{split.cutoff_ts}")
    p, info = run(tr, va, bounds)
    pd.DataFrame({"rowid": va["rowid"].astype(str), "predicted_probability": p}).to_parquet(
        f"data/validation/nn_step7{A.tag}_val_c4.parquet", index=False)
    y = va["label"].to_numpy()
    res = {**info, "seq": A.seq, "swap": A.swap, "local_auc": float(roc_auc_score(y, p)),
           "minutes": round((time.perf_counter() - t0) / 60, 1), "args": vars(A)}
    lf.log(f"NN STEP 7: local AUC {res['local_auc']:.6f} ({info['epochs']} epochs)")
    (HERE / "outputs" / f"nn_step7{A.tag}.json").write_text(json.dumps(res, indent=2))


def submit() -> None:
    t0 = time.perf_counter()
    bounds = lf.bounds_for(7, half=True)
    tr, te = load("step5_full_train", bounds), load("step5_full_test", bounds, "full_eval")
    p, info = run(tr, te, bounds)
    out = lf.REPO / "submissions" / f"safe_v2_step7_nn{A.tag}.csv"
    pd.DataFrame({"rowid": te["rowid"].astype(str), "predicted_probability": p}).to_csv(out, index=False)
    lf.log(f"wrote {out} ({info['epochs']} epochs, {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    submit() if A.submit else validate()
