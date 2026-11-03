"""Probe: is a taste model trained against the whole catalogue better than ours? Every model here learns "this item vs
the ~10 shown (or 9-20 sampled)" items; recommender research finds full-catalogue cross-entropy much stronger.

The simplest such model: free user and item vectors plus an item bias, trained on every known (user, item) click
before the boundary with a softmax over all 50,000 items. A candidate's score is log p(item | user) - log q(item), q
= the filler distribution (the real item is drawn by taste, fillers by q). Epochs come from the inner holdout (each
user's first event in the 12 h before the boundary, model fitted on clicks before that); validation is scored once.

Usage (from the repo root):
    python research/techarena/fullmf_probe.py [--dim 32] [--threads 6]
Reports within-event "which item" AUC on validation (all clicked events, and those whose clicked item is new to the
user) next to the choice model, the 50-click sequence model and step 7 on the same events.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--max-epochs", type=int, default=12)
ap.add_argument("--threads", type=int, default=6)
A = ap.parse_args()
torch.set_num_threads(A.threads)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seq_model as sm  # noqa: E402

T0 = time.perf_counter()
d, ev, n_users, n_items = sm.load(HERE.parents[1] / "data" / "parquet")
D0 = int(d.ts.min())
B = D0 + 4 * sm.DAY


def pairs_before(b: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (user, item) clicks known before b: history entries timestamped before b (any event's history; a
    history holds the last 20 clicks, so these are all visible by b) plus labelled clicks of train events before b."""
    us, its = [], []
    for e in (d, ev):
        m = (e.hist > 0) & (e.hist_ts < b)
        us.append(np.repeat(e.user, e.hist.shape[1]).reshape(e.hist.shape)[m])
        its.append(e.hist[m])
    m = (d.click > 0) & (d.ts < b)[:, None]
    us.append(np.repeat(d.user, sm.N_CAND).reshape(m.shape)[m])
    its.append(d.cand[m])
    p = np.unique(np.stack([np.concatenate(us), np.concatenate(its)], 1), axis=0)
    return p[:, 0], p[:, 1]


def log_q(b: int) -> np.ndarray:
    before = d.ts < b
    q = np.bincount(d.cand[before][d.click[before] == 0], minlength=n_items + 1).astype(float) + 1.0
    return np.log(q / q.sum())


class FullMF(nn.Module):
    def __init__(self, init: np.ndarray):
        super().__init__()
        self.user = nn.Embedding(n_users, A.dim)
        nn.init.zeros_(self.user.weight)
        self.item = nn.Embedding(n_items + 1, A.dim)
        nn.init.normal_(self.item.weight, std=0.01)
        with torch.no_grad():
            self.item.weight[:, :init.shape[1]] = torch.as_tensor(init, dtype=torch.float32)
        self.bias = nn.Parameter(torch.zeros(n_items + 1))

    def logits(self, u: torch.Tensor) -> torch.Tensor:  # over item codes 1..n_items (0 = pad excluded)
        return self.user(u) @ self.item.weight[1:].T + self.bias[1:]

    def score(self, u: np.ndarray, cand: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            uu = self.user(torch.as_tensor(u))
            c = self.item(torch.as_tensor(cand))
            return ((c * uu[:, None]).sum(-1) + self.bias[torch.as_tensor(cand)]).numpy()


def which(scores: np.ndarray, click: np.ndarray, keep: np.ndarray | None = None) -> float:
    c = click.sum(1) == 1
    if keep is not None:
        c &= keep
    s, y = scores[c], click[c] > 0
    cs = s[y][:, None]
    return float((((s < cs) + 0.5 * (s == cs)) * ~y).sum(1).mean() / 9)


def fit(b: int, epochs: int, hold=None) -> tuple[FullMF, list[float]]:
    torch.manual_seed(0)
    u, i = pairs_before(b)
    model = FullMF(sm.svd_init(d, b, n_users, n_items))
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd)
    ut, it = torch.as_tensor(u), torch.as_tensor(i - 1)  # targets indexed over items 1..n_items
    curve = []
    sm.log(f"  {len(u):,} click pairs before the boundary")
    for ep in range(epochs):
        perm = torch.randperm(len(u))
        tot, t = 0.0, time.perf_counter()
        for s in range(0, len(u), A.batch):
            bi = perm[s:s + A.batch]
            loss = nn.functional.cross_entropy(model.logits(ut[bi]), it[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        msg = f"  epoch {ep + 1}: loss {tot / len(u):.4f} ({time.perf_counter() - t:.0f} s)"
        if hold is not None:
            hu, hc, hy, lq = hold
            curve.append(which(model.score(hu, hc) - lq[hc], hy))
            msg += f", inner which-AUC {curve[-1]:.5f}"
            if len(curve) >= 3 and curve[-1] < max(curve) and curve[-2] < max(curve):
                sm.log(msg)
                break
        sm.log(msg)
    return model, curve


bi = B - sm.INNER_GAP
h_idx = sm.first_after(d, bi, until=B)
lq_in = log_q(bi)
sm.log(f"inner fit: clicks before B - 12 h; holdout {len(h_idx):,} events")
_, curve = fit(bi, A.max_epochs, hold=(d.user[h_idx], d.cand[h_idx], d.click[h_idx], lq_in))
epochs = int(np.argmax(curve)) + 1
sm.log(f"best inner epoch {epochs} (which-AUC {max(curve):.5f}); refit on clicks before day 4")
model, _ = fit(B, epochs)

v_idx = sm.first_after(d, B)
cand, click, rowid = d.cand[v_idx], d.click[v_idx], d.rowid[v_idx]
lq = log_q(B)
s_raw = model.score(d.user[v_idx], cand)
hist = d.hist[v_idx].copy()
hist[d.hist_ts[v_idx] >= B] = 0
in_hist = ((cand[:, :, None] == hist[:, None, :]) & (hist[:, None, :] > 0)).any(2)
new_click = ~(in_hist & (click > 0)).any(1)
rows = {"full-catalogue MF (log p - log q)": s_raw - lq[cand], "full-catalogue MF (log p)": s_raw}
others = {"choice model (cheap config)": ("data/validation/choice_mf_val_c4_h20_test.parquet", "predicted_probability"),
          "50-click sequence model, 5 seeds": ("research/techarena/cache/seq/stage3_h50/seq_b1737676800.parquet", "seq_prob"),
          "step 7 stack": ("data/validation/safe_v2_step7_val_c4.parquet", "predicted_probability")}
key = pd.DataFrame({"rowid": rowid.ravel().astype(str)})
for name, (path, col) in others.items():
    p = pd.read_parquet(HERE.parents[1] / path)
    m = key.merge(p.assign(rowid=p["rowid"].astype(str))[["rowid", col]], on="rowid", how="left")
    assert m[col].notna().all(), name
    rows[name] = m[col].to_numpy().reshape(-1, sm.N_CAND)
print(f"\nvalidation, {int((click.sum(1) == 1).sum()):,} clicked events ({int(((click.sum(1) == 1) & new_click).sum()):,} "
      f"with a new item); minutes {(time.perf_counter() - T0) / 60:.1f}")
print(f"{'model':38s} {'which, all':>11s} {'which, new item':>16s}")
for name, s in rows.items():
    print(f"{name:38s} {which(s, click):11.5f} {which(s, click, new_click):16.5f}")
