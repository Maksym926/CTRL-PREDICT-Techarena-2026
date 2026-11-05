"""Full-catalogue taste features per leak-free window (fullmf_probe.py's model): free user and item vectors plus an
item bias, trained on every known (user, item) click before the boundary with a softmax over all 50,000 items. For
each target event's candidates it writes fmf_score = log p(item | user) - log q(item) (q = the filler distribution;
the real item is drawn by taste, fillers by q), fmf_logp = log p(item | user) and fmf_rank (1 = best fmf_score in the
event), averaged over --seeds seeds, as seq_<name>.parquet + .json (the layout of seq_model.py), so
safe_v2_step7.py --extra <folder> can merge them into the frames.

    python research/techarena/fullmf_model.py --data data/parquet --out research/techarena/cache/seq/fullmf --mode windows
    python research/techarena/fullmf_model.py --data data/parquet --out <dir> --mode val     # day 4 only

Leak-free like seq_model.py: a boundary's model sees only clicks timestamped before it (history entries of any event:
a history holds the last 20 clicks, so every such click is already visible in a history by the boundary; plus
labelled clicks of train events before it). Epochs are chosen once, on the evaluation fit's inner holdout (each user's
first event in the 12 h before the cutoff, model fitted on clicks before that), and reused for every boundary.
Runs latest first and skips boundaries already written (restart-safe). Self-contained apart from seq_model.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import seq_model as sm

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mode", choices=["val", "windows"], default="windows")
ap.add_argument("--dim", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--max-epochs", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scale", type=float, default=4.0, help="starting utility = scale * cosine (as the choice model)")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--only", type=int, default=0, help="smoke test: first N jobs only")
ap.add_argument("--fit-frac", type=float, default=1.0, help="smoke test: train on this share of the click pairs")
A = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if A.threads:
    torch.set_num_threads(A.threads)


class FullMF(nn.Module):
    """Starts at the click-SVD solution: item vectors = sqrt(scale) * unit SVD vectors, user vectors = the mean of
    their clicked items' vectors, item bias = log popularity; training then moves it by full-catalogue softmax."""

    def __init__(self, n_users: int, n_items: int, init: np.ndarray, u: np.ndarray, i: np.ndarray):
        super().__init__()
        r = min(A.dim, init.shape[1])
        v = init[:, :r] * np.sqrt(A.scale)
        sums = np.zeros((n_users, r))
        np.add.at(sums, u, v[i])
        users = sums / np.maximum(np.bincount(u, minlength=n_users), 1)[:, None]
        pop = np.log1p(np.bincount(i, minlength=n_items + 1).astype(float))
        self.user = nn.Embedding(n_users, A.dim)
        nn.init.normal_(self.user.weight, std=0.01)
        self.item = nn.Embedding(n_items + 1, A.dim)
        nn.init.normal_(self.item.weight, std=0.01)
        with torch.no_grad():
            self.item.weight[:, :r] = torch.as_tensor(v, dtype=torch.float32)
            self.user.weight[:, :r] = torch.as_tensor(users, dtype=torch.float32)
        self.bias = nn.Parameter(torch.as_tensor(pop - pop[1:].mean(), dtype=torch.float32))

    def logits(self, u: torch.Tensor) -> torch.Tensor:  # over item codes 1..n_items (code 0 = pad excluded)
        return self.user(u) @ self.item.weight[1:].T + self.bias[1:]

    @torch.no_grad()
    def score(self, u: np.ndarray, cand: np.ndarray, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Candidate utilities and log p(candidate | user) (log-softmax over the whole catalogue)."""
        util, logp = [], []
        for s in range(0, len(u), batch):
            uu = torch.as_tensor(u[s:s + batch], device=DEV)
            cc = torch.as_tensor(cand[s:s + batch], device=DEV)
            x = (self.item(cc) * self.user(uu)[:, None]).sum(-1) + self.bias[cc]
            util.append(x.cpu().numpy())
            logp.append((x - torch.logsumexp(self.logits(uu), 1, keepdim=True)).cpu().numpy())
        return np.concatenate(util), np.concatenate(logp)


def pairs_before(d, ev, b: int) -> tuple[np.ndarray, np.ndarray]:
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


def log_q(d, b: int, n_items: int) -> np.ndarray:
    before = d.ts < b
    q = np.bincount(d.cand[before][d.click[before] == 0], minlength=n_items + 1).astype(float) + 1.0
    return np.log(q / q.sum())


def which(scores: np.ndarray, click: np.ndarray) -> float:
    c = click.sum(1) == 1
    s, y = scores[c], click[c] > 0
    cs = s[y][:, None]
    return float((((s < cs) + 0.5 * (s == cs)) * ~y).sum(1).mean() / 9)


def fit(d, ev, n_users, n_items, b: int, epochs: int, seed: int, hold=None):
    torch.manual_seed(seed)
    u, i = pairs_before(d, ev, b)
    if A.fit_frac < 1:
        keep = np.random.default_rng(seed).random(len(u)) < A.fit_frac
        u, i = u[keep], i[keep]
    model = FullMF(n_users, n_items, sm.svd_init(d, b, n_users, n_items), u, i).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd)
    ut, it = torch.as_tensor(u, device=DEV), torch.as_tensor(i - 1, device=DEV)
    curve = []
    if hold is not None:  # the starting point (click SVD), for reference
        hu, hc, hy, lq = hold
        sm.log(f"  start (SVD init): inner which-AUC {which(model.score(hu, hc)[0] - lq[hc], hy):.5f}")
    for ep in range(epochs):
        perm = torch.randperm(len(u), device=DEV)
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
            curve.append(which(model.score(hu, hc)[0] - lq[hc], hy))
            msg += f", inner which-AUC {curve[-1]:.5f}"
        sm.log(msg)
        if hold is not None and len(curve) >= 3 and curve[-1] < max(curve) and curve[-2] < max(curve):
            break
    return model, curve, len(u)


def main() -> None:
    sm.log(f"device {DEV}")
    A.out.mkdir(parents=True, exist_ok=True)
    d, ev, n_users, n_items = sm.load(A.data)
    d0 = int(d.ts.min())
    if A.mode == "val":
        jobs = [(f"b{d0 + 4 * sm.DAY}", d0 + 4 * sm.DAY, d)]
    else:
        jobs = [("full_eval", d0 + 7 * sm.DAY, ev)] + [(f"b{d0 + k * sm.DAY // 2}", d0 + k * sm.DAY // 2, d)
                                                    for k in range(13, 0, -1)]
    if A.only:
        jobs = jobs[:A.only]
    epochs, curve = None, []
    for name, b, target in jobs:
        done = A.out / f"seq_{name}.json"
        if done.exists():
            epochs = json.loads(done.read_text())["epochs"]
            sm.log(f"{name}: already done ({epochs} epochs), skipping")
            continue
        if epochs is None:  # epoch search once, on the first (latest) boundary's inner holdout
            bi = b - sm.INNER_GAP
            h = sm.first_after(d, bi, until=b)
            sm.log(f"{name}: inner holdout {len(h):,} events (clicks before b - 12 h)")
            _, curve, _ = fit(d, ev, n_users, n_items, bi, A.max_epochs, A.seed,
                              hold=(d.user[h], d.cand[h], d.click[h], log_q(d, bi, n_items)))
            epochs = int(np.argmax(curve)) + 1
            sm.log(f"{name}: best inner epoch {epochs} (which-AUC {max(curve):.5f})")
        idx = np.arange(len(target)) if target is ev else sm.first_after(d, b)
        u, cand = target.user[idx], target.cand[idx]
        lq = log_q(d, b, n_items)
        utils, logps = [], []
        for s in range(A.seeds):
            model, _, n_pairs = fit(d, ev, n_users, n_items, b, epochs, A.seed + s)
            ut, lp = model.score(u, cand)
            utils.append(ut)
            logps.append(lp)
        score = np.mean(utils, 0) - lq[cand]
        out = pd.DataFrame({"rowid": target.rowid[idx].ravel(), "fmf_score": score.ravel().astype(np.float32),
                            "fmf_logp": np.mean(logps, 0).ravel().astype(np.float32),
                            "fmf_rank": ((-score).argsort(1).argsort(1) + 1).ravel().astype(np.float32)})
        info = {"boundary": b, "epochs": epochs, "inner_curve": curve, "seeds": A.seeds, "pairs": int(n_pairs),
                "targets": int(len(idx)), "args": {k: str(v) for k, v in vars(A).items()}}
        if target is d:
            info["which_auc"] = which(score, d.click[idx])
            sm.log(f"{name}: targets' which-AUC {info['which_auc']:.5f}")
        out.to_parquet(A.out / f"seq_{name}.parquet", index=False)
        done.write_text(json.dumps(info, indent=2))
        sm.log(f"wrote {A.out / f'seq_{name}.parquet'} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
