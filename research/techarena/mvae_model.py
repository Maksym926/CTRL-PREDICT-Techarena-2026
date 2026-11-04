"""Mult-VAE taste model (4 Nov): a variational autoencoder over each user's set of known clicks, with a multinomial
(softmax over all 50,000 items) likelihood and a linear decoder: p(item | user) = softmax(V z + b), z ~ q(z | clicks).
This is the generator's own form as measured (static taste of ~10 dims + popularity, each click a draw from a
softmax over the catalogue); the encoder gives a Bayesian (shrunk) taste estimate for users with few clicks, and the
full-catalogue loss trains every item vector on every click (the slate-trained models see 9 random fillers only).

For each target event's candidates it writes seq_prob = the within-event softmax of
    mvae_score = log p(item | user's clicks before the boundary) - log q(item)   (q = filler distribution)
plus mvae_score and mvae_logp, as cache/seq/<out>/seq_<name>.parquet + .json (the layout of seq_model.py), so
final_blend.py --seq-extra <out> can use it as an extra within-event component.

    python research/techarena/mvae_model.py --data data/parquet --out research/techarena/cache/seq/mvae --mode val
    python research/techarena/mvae_model.py --data data/parquet --out research/techarena/cache/seq/mvae --mode eval

Leak-free: a boundary's model sees only clicks timestamped before it (history entries of any event, labelled clicks
of train events before it), as fullmf_model.py. Epochs are chosen on that boundary's inner holdout (each user's first
event in the 12 h before it, model fitted on clicks before b - 12 h), checked every --check epochs.
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
ap.add_argument("--mode", choices=["val", "eval", "both"], default="val")
ap.add_argument("--dim", type=int, default=32, help="latent taste dims")
ap.add_argument("--hidden", type=int, default=600)
ap.add_argument("--beta", type=float, default=0.2, help="KL weight after annealing")
ap.add_argument("--anneal", type=float, default=0.3, help="share of the epochs over which beta rises from 0")
ap.add_argument("--drop", type=float, default=0.5, help="input dropout (share of a user's clicks hidden)")
ap.add_argument("--target", choices=["all", "held"], default="all",
                help="all = reconstruct every click (standard Mult-VAE); held = predict only the hidden clicks")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--max-epochs", type=int, default=150)
ap.add_argument("--check", type=int, default=5, help="inner-holdout check every N epochs")
ap.add_argument("--patience", type=int, default=3, help="checks without improvement before stopping")
ap.add_argument("--epochs", type=int, default=0, help="skip the search and use this many epochs")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--samples", type=int, default=0, help="score with the mean of S posterior samples (0 = posterior mean)")
ap.add_argument("--svd-init", type=int, default=1, help="start the item vectors at the click SVD (rank 11)")
ap.add_argument("--scale", type=float, default=4.0, help="SVD start: item vectors = sqrt(scale) * unit SVD vectors")
ap.add_argument("--threads", type=int, default=0)
A = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if A.threads:
    torch.set_num_threads(A.threads)


def pairs_before(d, ev, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (user, item) clicks known before b (same as fullmf_model.pairs_before)."""
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


class Bags:
    """Per user, the item codes (1..n_items) clicked before the boundary, as a padded matrix (0 = pad)."""

    def __init__(self, u: np.ndarray, i: np.ndarray, n_users: int):
        order = np.lexsort((i, u))
        u, i = u[order], i[order]
        self.n = np.bincount(u, minlength=n_users)
        width = max(int(self.n.max()), 1)
        self.items = np.zeros((n_users, width), np.int64)
        start = np.cumsum(self.n) - self.n
        pos = np.arange(len(u)) - np.repeat(start, self.n)
        self.items[u, pos] = i
        self.users = np.flatnonzero(self.n > 0)


class MultVAE(nn.Module):
    def __init__(self, n_items: int, pop: np.ndarray, init: np.ndarray | None = None):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_items + 1, A.hidden, mode="sum", padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.enc_b = nn.Parameter(torch.zeros(A.hidden))
        self.cnt = nn.Linear(1, A.hidden)
        self.mu = nn.Linear(A.hidden, A.dim)
        self.lv = nn.Linear(A.hidden, A.dim)
        self.dec = nn.Parameter(torch.randn(n_items, A.dim) * 0.02)  # item vectors, codes 1..n_items -> rows 0..
        if init is not None:  # click-SVD start (rows = item codes incl. pad 0), as fullmf_model.py
            r = min(A.dim, init.shape[1], A.hidden)
            with torch.no_grad():
                self.dec[:, :r] = torch.as_tensor(init[1:, :r] * np.sqrt(A.scale), dtype=torch.float32)
                self.emb.weight[1:, :r] = torch.as_tensor(init[1:, :r], dtype=torch.float32)
                # encoder output starts as the identity on the SVD dims: z ~ tanh(sum of clicked unit vectors / sqrt n)
                self.mu.weight.mul_(0.01)
                self.mu.weight[:r, :r] += torch.eye(r) * np.sqrt(A.scale)
                self.mu.bias.zero_()
                self.lv.weight.mul_(0.01)
                self.lv.bias.fill_(-4.0)
                self.cnt.weight.zero_()
                self.cnt.bias.zero_()
        self.bias = nn.Parameter(torch.as_tensor(pop, dtype=torch.float32))

    def encode(self, items: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """items: B x W codes (0 = pad); keep: B x W float mask of the clicks the encoder sees."""
        n = keep.sum(1, keepdim=True)
        w = keep / torch.sqrt(n.clamp(min=1.0))
        h = self.emb(items, per_sample_weights=w) + self.enc_b + self.cnt(torch.log1p(n))
        h = torch.tanh(h)
        return self.mu(h), self.lv(h).clamp(-8, 8)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.dec.T + self.bias


def batches(bags: Bags, users: np.ndarray, size: int):
    for s in range(0, len(users), size):
        uu = users[s:s + size]
        w = max(int(bags.n[uu].max()), 1)
        yield uu, torch.as_tensor(bags.items[uu, :w], device=DEV)


def fit(bags: Bags, n_items: int, pop: np.ndarray, init, epochs: int, seed: int, hold=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MultVAE(n_items, pop, init).to(DEV)
    if hold is not None:
        hu, hc, hy, lq = hold
        sm.log(f"  start: inner which-AUC {which(score(model, bags, hu, hc)[0] - lq[hc], hy):.5f}")
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd)
    total_steps = epochs * int(np.ceil(len(bags.users) / A.batch))
    anneal_steps = max(1, int(A.anneal * total_steps))
    step, curve, best_state, since = 0, [], None, 0
    for ep in range(epochs):
        model.train()
        t, tot, nb = time.perf_counter(), 0.0, 0
        for uu, items in batches(bags, rng.permutation(bags.users), A.batch):
            valid = (items > 0).float()
            keep = valid * (torch.rand_like(valid) >= A.drop).float()
            empty = keep.sum(1) == 0
            keep[empty] = valid[empty]  # never hide every click of a user
            mu, lv = model.encode(items, keep)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            logp = torch.log_softmax(model.logits(z), 1)
            tgt = valid if A.target == "all" else valid * (1 - keep)
            tgt = torch.where(empty[:, None], valid, tgt) if A.target == "held" else tgt
            ll = (torch.gather(logp, 1, (items - 1).clamp(min=0)) * tgt).sum(1)
            kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(1)
            beta = A.beta * min(1.0, step / anneal_steps)
            loss = (-ll + beta * kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            tot += loss.item()
            nb += 1
        msg = f"  epoch {ep + 1}: loss {tot / nb:.3f} ({time.perf_counter() - t:.1f} s)"
        if hold is not None and ((ep + 1) % A.check == 0 or ep + 1 == epochs):
            hu, hc, hy, lq = hold
            sc = score(model, bags, hu, hc)[0] - lq[hc]
            curve.append((ep + 1, which(sc, hy)))
            msg += f", inner which-AUC {curve[-1][1]:.5f}"
            if curve[-1][1] >= max(c for _, c in curve):
                since = 0
            else:
                since += 1
            sm.log(msg)
            if since >= A.patience:
                break
        elif (ep + 1) % A.check == 0:
            sm.log(msg)
    return model, curve


@torch.no_grad()
def score(model: MultVAE, bags: Bags, u: np.ndarray, cand: np.ndarray, size: int = 2048):
    """For events of users u with candidates cand (n x 10 codes): logit and log p(candidate | clicks)."""
    model.eval()
    util, logp = [], []
    for s in range(0, len(u), size):
        uu = u[s:s + size]
        w = max(int(bags.n[uu].max()), 1)
        items = torch.as_tensor(bags.items[uu, :w], device=DEV)
        keep = (items > 0).float()
        mu, lv = model.encode(items, keep)
        cc = torch.as_tensor(cand[s:s + size] - 1, device=DEV).clamp(min=0)
        if A.samples:
            acc_u = acc_l = 0.0
            for _ in range(A.samples):
                lg = model.logits(mu + torch.randn_like(mu) * torch.exp(0.5 * lv))
                acc_u = acc_u + torch.gather(lg, 1, cc)
                acc_l = acc_l + torch.gather(torch.log_softmax(lg, 1), 1, cc)
            util.append((acc_u / A.samples).cpu().numpy())
            logp.append((acc_l / A.samples).cpu().numpy())
        else:
            lg = model.logits(mu)
            util.append(torch.gather(lg, 1, cc).cpu().numpy())
            logp.append(torch.gather(torch.log_softmax(lg, 1), 1, cc).cpu().numpy())
    return np.concatenate(util), np.concatenate(logp)


def pop_init(i: np.ndarray, n_items: int) -> np.ndarray:
    pop = np.log1p(np.bincount(i - 1, minlength=n_items).astype(float))
    return pop - pop.mean()


def main() -> None:
    sm.log(f"device {DEV}; args {vars(A)}")
    A.out.mkdir(parents=True, exist_ok=True)
    d, ev, n_users, n_items = sm.load(A.data)
    d0 = int(d.ts.min())
    jobs = []
    if A.mode in ("val", "both"):
        jobs.append((f"b{d0 + 4 * sm.DAY}", d0 + 4 * sm.DAY, d))
    if A.mode in ("eval", "both"):
        jobs.append(("full_eval", d0 + 7 * sm.DAY, ev))
    for name, b, target in jobs:
        epochs, curve = A.epochs, []
        if not epochs:
            bi = b - sm.INNER_GAP
            h = sm.first_after(d, bi, until=b)
            u_i, i_i = pairs_before(d, ev, bi)
            bags_i = Bags(u_i, i_i, n_users)
            sm.log(f"{name}: inner holdout {len(h):,} events; {len(u_i):,} clicks of {len(bags_i.users):,} users "
                   f"before b - 12 h")
            init_i = sm.svd_init(d, bi, n_users, n_items) if A.svd_init else None
            _, curve = fit(bags_i, n_items, pop_init(i_i, n_items), init_i, A.max_epochs, A.seed,
                           hold=(d.user[h], d.cand[h], d.click[h], log_q(d, bi, n_items)))
            epochs = max(curve, key=lambda c: c[1])[0]
            sm.log(f"{name}: best inner epoch {epochs} (which-AUC {max(c for _, c in curve):.5f})")
        idx = np.arange(len(target)) if target is ev else sm.first_after(d, b)
        u, cand = target.user[idx], target.cand[idx]
        uu, ii = pairs_before(d, ev, b)
        bags = Bags(uu, ii, n_users)
        lq = log_q(d, b, n_items)
        init = sm.svd_init(d, b, n_users, n_items) if A.svd_init else None
        utils, logps = [], []
        for s in range(A.seeds):
            t = time.perf_counter()
            model, _ = fit(bags, n_items, pop_init(ii, n_items), init, epochs, A.seed + 100 + s)
            ut, lp = score(model, bags, u, cand)
            utils.append(ut)
            logps.append(lp)
            msg = f"{name}: seed {s} fitted ({epochs} epochs, {time.perf_counter() - t:.0f} s)"
            if target is d:
                msg += f", targets' which-AUC {which(ut - lq[cand], d.click[idx]):.5f}"
            sm.log(msg)
        sc = np.mean(utils, 0) - lq[cand]
        share = np.exp(sc - sc.max(1, keepdims=True))
        share /= share.sum(1, keepdims=True)
        out = pd.DataFrame({"rowid": target.rowid[idx].ravel(), "seq_prob": np.clip(share, 1e-6, 1).ravel().astype(np.float32),
                            "mvae_score": sc.ravel().astype(np.float32),
                            "mvae_logp": np.mean(logps, 0).ravel().astype(np.float32)})
        info = {"boundary": b, "epochs": epochs, "inner_curve": curve, "seeds": A.seeds, "pairs": int(len(uu)),
                "targets": int(len(idx)), "args": {k: str(v) for k, v in vars(A).items()}}
        if target is d:
            info["which_auc"] = which(sc, d.click[idx])
            sm.log(f"{name}: targets' which-AUC, {A.seeds} seeds averaged {info['which_auc']:.5f}")
        out.to_parquet(A.out / f"seq_{name}.parquet", index=False)
        (A.out / f"seq_{name}.json").write_text(json.dumps(info, indent=2))
        sm.log(f"wrote {A.out / f'seq_{name}.parquet'} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
