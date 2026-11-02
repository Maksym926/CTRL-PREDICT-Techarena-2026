"""Sequence model: candidates attend to the user's click history (transformer), softmax over 10 candidates + no click.

Self-contained (numpy, pandas, pyarrow, scipy, scikit-learn, torch), so it runs unchanged on a Kaggle GPU or locally.

    python seq_model.py --data data/parquet --out research/techarena/cache/seq --mode val
    python seq_model.py --data data/parquet --out research/techarena/cache/seq --mode full
    python seq_model.py --data data/parquet --out research/techarena/cache/seq --mode boundary --boundary 1737374400 ...
    python seq_model.py --data /kaggle/input/... --out seq_stage3 --mode windows   # stage 3, all 14 fits

Leak-free, like the rest of the pipeline. For a boundary B the model trains only on train events before B, and target
histories are cut to clicks before B. The epoch count comes from an inner holdout (each user's first event in the 12 h
before B, scored by a model fitted on events before that), then the model is refitted on everything before B for that
many epochs. Validation is never used for any choice.

Modes:
  val       B = day 4 (src.validation's cutoff); scores each user's first event from B on and reports its AUC.
  full      B = end of train; scores evaluation.
  boundary  each given B; scores each user's first train event from B on (for LightGBM features per window).
  windows   stage 3: full + boundary at every LightGBM window (day 0.5, 1.0, ..., 6.5). The validation windows are the
            first 7 of these and validation's own boundary (day 4) is one of them, so one run serves both validation
            and submission. Runs latest first and skips boundaries already written (safe to restart). A boundary with
            fewer than MIN_INNER events before its inner holdout reuses the epoch count of the boundary after it.
Each writes rowid, seq_prob, seq_noclick, seq_rank (+ a JSON with the settings and inner-holdout curve).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
from torch import nn

DAY, HOUR = 86_400, 3_600
HIST, N_CAND, N_AGE = 20, 10, 24
INNER_GAP = 12 * HOUR
MIN_INNER = 20_000  # fewer inner-fit events than this: no epoch search (only the day 0.5 window)
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- data
class Events:
    """One split as arrays: history (oldest first, 0 = pad) and 10 candidates per event. Item codes start at 1."""

    def __init__(self, events, cands, user_index: pd.Index, item_index: pd.Index, labelled: bool):
        n = events.num_rows
        self.event_id = events.column("event_id").to_numpy()
        self.user = user_index.get_indexer(events.column("user_id").to_numpy()).astype(np.int64)
        self.ts = events.column("timestamp").to_numpy().astype(np.int64)
        self.hist = np.zeros((n, HIST), np.int64)
        self.hist_ts = np.zeros((n, HIST), np.int64)
        for col, out, codes in (("history_item_ids", self.hist, True), ("history_timestamps", self.hist_ts, False)):
            arr = events.column(col).combine_chunks()
            off = arr.offsets.to_numpy()
            lens = np.diff(off)
            vals = arr.flatten().to_numpy(zero_copy_only=False)
            if len(lens) and lens.max() > HIST:  # keep the most recent HIST
                raise ValueError(f"history longer than {HIST}")
            rows = np.repeat(np.arange(n), lens)
            pos = np.arange(len(vals)) - np.repeat(off[:-1], lens)
            out[rows, pos] = item_index.get_indexer(vals) + 1 if codes else vals

        ev_pos = pd.Index(self.event_id).get_indexer(cands.column("event_id").to_numpy())
        order = np.argsort(ev_pos, kind="stable")
        if not (np.bincount(ev_pos, minlength=n) == N_CAND).all():
            raise ValueError("every event must have exactly 10 candidates")
        self.cand = (item_index.get_indexer(cands.column("item_id").to_numpy()[order]) + 1).reshape(n, N_CAND)
        self.rowid = cands.column("rowid").to_numpy()[order].reshape(n, N_CAND)
        self.click = (cands.column("clicked").to_numpy()[order].reshape(n, N_CAND).astype(np.float32) if labelled
                      else np.zeros((n, N_CAND), np.float32))

    def __len__(self) -> int:
        return len(self.ts)


def load(data: Path):
    log("loading parquet")
    te, tc = pq.read_table(data / "train_events.parquet"), pq.read_table(data / "train_candidates.parquet")
    ee, ec = pq.read_table(data / "eval_events.parquet"), pq.read_table(data / "eval_candidates.parquet")
    users = pd.Index(pd.unique(np.concatenate([te.column("user_id").to_numpy(), ee.column("user_id").to_numpy()])))
    item_parts = [tc.column("item_id").to_numpy(), ec.column("item_id").to_numpy(),
                  te.column("history_item_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                  ee.column("history_item_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False)]
    items = pd.Index(pd.unique(np.concatenate(item_parts)))
    train, ev = Events(te, tc, users, items, True), Events(ee, ec, users, items, False)
    log(f"{len(train):,} train events, {len(ev):,} eval events, {len(users):,} users, {len(items):,} items")
    return train, ev, len(users), len(items)


def first_after(d: Events, b: int, until: int | None = None) -> np.ndarray:
    """Index of each user's first event at or after b (ties broken by event_id, as src.validation does)."""
    idx = np.flatnonzero((d.ts >= b) & (d.ts < until if until is not None else True))
    idx = idx[np.lexsort((d.event_id[idx], d.ts[idx]))]
    _, first = np.unique(d.user[idx], return_index=True)
    return np.sort(idx[first])


def batch_arrays(d: Events, idx: np.ndarray, cut: int | None = None) -> dict[str, np.ndarray]:
    """Model inputs for events idx; with cut, history clicks at or after cut are dropped (leak-free targets)."""
    hist, hts = d.hist[idx].copy(), d.hist_ts[idx]
    if cut is not None:
        hist[hts >= cut] = 0
    valid = hist > 0
    ts = d.ts[idx][:, None]
    age_h = np.where(valid, (ts - hts) / HOUR, 0.0)
    age = np.clip((np.log2(1 + np.maximum(age_h, 0)) * 2).astype(np.int64), 0, N_AGE - 1)
    last = np.where(valid, hts, -1).max(1)
    gap = np.where(last > 0, np.log1p(np.maximum(d.ts[idx] - last, 0) / HOUR), np.log1p(1000.0))
    rep = np.minimum(((d.cand[idx][:, :, None] == hist[:, None, :]) & valid[:, None, :]).sum(2), 3)
    return {"user": d.user[idx], "hist": hist, "age": age, "cand": d.cand[idx], "rep": rep,
            "dense": np.stack([gap, valid.sum(1) / HIST, (rep > 0).sum(1) / N_CAND], 1).astype(np.float32),
            "click": d.click[idx]}


def svd_init(d: Events, b: int, n_users: int, n_items: int, rank: int = 11) -> np.ndarray:
    """Item vectors (rows = item codes incl. pad 0) from the normalised click SVD on clicks before b."""
    before = d.ts < b
    hu = np.repeat(d.user[before], HIST).reshape(-1, HIST)
    hi, hts = d.hist[before], d.hist_ts[before]
    keep = (hi > 0) & (hts < b)
    cu = np.repeat(d.user[before], N_CAND).reshape(-1, N_CAND)
    ck = d.click[before] > 0
    u = np.concatenate([hu[keep], cu[ck]])
    i = np.concatenate([hi[keep], d.cand[before][ck]])
    x = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(n_users, n_items + 1))
    x.sum_duplicates()
    x.data[:] = 1.0  # binary: each user-item click pair once
    du, di = np.asarray(x.sum(1)).ravel() + 1.0, np.asarray(x.sum(0)).ravel() + 1.0
    xs = sp.diags(1 / np.sqrt(du)) @ x @ sp.diags(1 / np.sqrt(di))
    _, _, vt = svds(xs.astype(np.float64), k=rank, random_state=20261024)
    v = vt.T  # same as src.choice_mf.click_svd: rank-k item vectors, L2-normalised, so the MF path starts at a cosine
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------- model
class SeqModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int, heads: int, dropout: float,
                 init: np.ndarray | None):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.normal_(self.item.weight, std=0.01)
        if init is not None:
            with torch.no_grad():
                self.item.weight[:, :init.shape[1]] = torch.as_tensor(init, dtype=torch.float32)
                self.item.weight[0] = 0
        self.item_bias = nn.Embedding(n_items + 1, 1)
        nn.init.zeros_(self.item_bias.weight)
        self.user = nn.Embedding(n_users, dim)
        nn.init.zeros_(self.user.weight)
        self.age = nn.Embedding(N_AGE, dim)
        self.pos = nn.Embedding(HIST + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.rep = nn.Embedding(4, 1)
        nn.init.zeros_(self.rep.weight)
        self.mf_scale = nn.Parameter(torch.tensor(4.0))  # sharpness of the MF path (choice-MF default scale)
        self.score = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, 1))
        self.noclick = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.register_buffer("item0", self.item.weight.detach().clone())  # anchor for --emb-l2

    def penalty(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """Squared drift of the batch's item vectors from their start, plus squared user vectors (start at 0)."""
        ids = torch.cat([b["hist"], b["cand"]], 1)
        m = (ids > 0).float()
        drift = ((self.item(ids) - self.item0[ids]) ** 2).sum(-1)
        return (drift * m).sum() / m.sum() + (self.user(b["user"]) ** 2).sum(-1).mean()

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits (batch, 11): column 0 = no click, 1..10 = candidates."""
        hist = b["hist"]
        n = hist.shape[0]
        e, u = self.item(hist), self.user(b["user"])
        # direct MF path: raw history average + user vector, so training starts from the SVD taste match
        q = e.sum(1) / (hist > 0).sum(1, keepdim=True).clamp(min=1) + u
        tok = e + self.age(b["age"])
        seq = torch.cat([u[:, None], tok], 1) + self.pos.weight[None]
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=hist.device), hist == 0], 1)
        h = self.encoder(seq, src_key_padding_mask=pad)
        w = (~pad).float()[:, :, None]
        p = (h * w).sum(1) / w.sum(1)
        c = self.item(b["cand"])
        a, _ = self.cross(c, h, h, key_padding_mask=pad, need_weights=False)
        pe = p[:, None].expand_as(c)
        s = self.score(torch.cat([c, a, pe, c * a, c * pe], -1)).squeeze(-1)
        s = s + self.mf_scale * (c * q[:, None]).sum(-1) + self.item_bias(b["cand"]).squeeze(-1) \
            + self.rep(b["rep"]).squeeze(-1)
        s0 = self.noclick(torch.cat([p, b["dense"]], -1))
        return torch.cat([s0, s], 1)


def to_device(arrs: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in arrs.items()}


def loss_fn(logits: torch.Tensor, click: torch.Tensor) -> torch.Tensor:
    """Softmax choice loss; several clicks share the target mass, no click puts it on column 0."""
    k = click.sum(1, keepdim=True)
    target = torch.cat([(k == 0).float(), click / k.clamp(min=1)], 1)
    return -(target * torch.log_softmax(logits, 1)).sum(1).mean()


@torch.no_grad()
def predict(model: SeqModel, data: dict[str, torch.Tensor], batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(data["hist"]), batch):
        out.append(torch.softmax(model({k: v[s:s + batch] for k, v in data.items()}), 1).cpu().numpy())
    return np.concatenate(out)


def fit(args, d: Events, n_users: int, n_items: int, b: int, epochs: int, device,
        holdout: tuple[dict, np.ndarray] | None = None) -> tuple[SeqModel, list[float]]:
    """Train on events before b. With a holdout, log its AUC after each epoch (and stop after `patience` misses)."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    idx = np.flatnonzero(d.ts < b)
    if args.fit_frac < 1:
        idx = np.sort(np.random.choice(idx, int(len(idx) * args.fit_frac), replace=False))
    train = to_device(batch_arrays(d, idx), device)
    model = SeqModel(n_users, n_items, args.dim, args.layers, args.heads, args.dropout,
                     svd_init(d, b, n_users, n_items) if args.svd_init else None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    curve, best, misses = [], -1.0, 0
    if holdout is not None:
        p0 = predict(model, holdout[0])[:, 1:]
        log(f"  epoch 0 (untrained, SVD start): inner-holdout AUC {roc_auc_score(holdout[1].ravel(), p0.ravel()):.6f}")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(idx), device=device)
        total, t = 0.0, time.perf_counter()
        for s in range(0, len(idx), args.batch):
            bi = perm[s:s + args.batch]
            batch = {k: v[bi] for k, v in train.items()}
            loss = loss_fn(model(batch), batch["click"])
            if args.emb_l2 > 0:
                loss = loss + args.emb_l2 * model.penalty(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(bi)
            step = s // args.batch + 1
            if args.log_every and step % args.log_every == 0:
                log(f"    step {step}: loss {total / min(s + args.batch, len(idx)):.4f}")
        msg = f"  epoch {ep}: loss {total / len(idx):.4f} ({time.perf_counter() - t:.0f} s)"
        if holdout is not None:
            p = predict(model, holdout[0])[:, 1:]
            curve.append(float(roc_auc_score(holdout[1].ravel(), p.ravel())))
            msg += f", inner-holdout AUC {curve[-1]:.6f}"
            best, misses = (curve[-1], 0) if curve[-1] > best else (best, misses + 1)
        log(msg)
        if holdout is not None and misses >= args.patience:
            break
    return model, curve


def run_boundary(args, d: Events, n_users: int, n_items: int, b: int, target_idx: np.ndarray, target: Events,
                 device, fallback_epochs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Pick epochs on the inner holdout before b, refit on everything before b, score the targets."""
    bi = b - INNER_GAP
    n_inner = int(np.sum(d.ts < bi))
    if n_inner < MIN_INNER and fallback_epochs is not None:
        curve, epochs = [], fallback_epochs
        log(f"boundary {b}: only {n_inner:,} inner-fit events, using {epochs} epoch(s) from the next boundary")
    else:
        hold_idx = first_after(d, bi, until=b)
        hold = batch_arrays(d, hold_idx, cut=bi)
        log(f"boundary {b}: inner fit {n_inner:,} events, inner holdout {len(hold_idx):,} events")
        _, curve = fit(args, d, n_users, n_items, bi, args.max_epochs, device,
                       holdout=(to_device(hold, device), hold["click"]))
        epochs = int(np.argmax(curve)) + 1
        log(f"boundary {b}: best inner epoch {epochs} (AUC {max(curve):.6f})")
    if args.inner_only:
        return None, {"boundary": b, "inner_curve": curve, "epochs": epochs, "best_inner_auc": max(curve, default=None)}
    log(f"refitting on {np.sum(d.ts < b):,} events")
    model, _ = fit(args, d, n_users, n_items, b, epochs, device)
    probs = predict(model, to_device(batch_arrays(target, target_idx, cut=b), device))
    p = probs[:, 1:]
    out = pd.DataFrame({"rowid": target.rowid[target_idx].ravel(), "seq_prob": p.ravel(),
                        "seq_noclick": np.repeat(probs[:, 0], N_CAND),
                        "seq_rank": ((-p).argsort(1).argsort(1) + 1).ravel().astype(np.float32)})
    return out, {"boundary": b, "inner_curve": curve, "epochs": epochs, "targets": int(len(target_idx))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["val", "full", "boundary", "windows"], default="val")
    ap.add_argument("--boundary", type=int, nargs="*", default=[])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--emb-l2", type=float, default=0.0, help="anchor item vectors to their start, users to 0")
    ap.add_argument("--inner-only", action="store_true", help="only the inner-holdout curve (for choosing settings)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--svd-init", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-frac", type=float, default=1.0, help="<1 only for quick smoke tests")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0, help="print the running loss every N steps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    d, ev, n_users, n_items = load(args.data)
    d0 = int(d.ts.min())

    if args.mode == "val":
        jobs = [("val", d0 + 4 * DAY, d)]
    elif args.mode == "full":
        jobs = [("full_eval", d0 + 7 * DAY, ev)]
    elif args.mode == "windows":
        jobs = [("full_eval", d0 + 7 * DAY, ev)] + [(f"b{d0 + i * DAY // 2}", d0 + i * DAY // 2, d)
                                                    for i in range(13, 0, -1)]
    else:
        jobs = [(f"b{b}", b, d) for b in args.boundary]

    last_epochs = None
    for name, b, target in jobs:
        idx = np.arange(len(ev)) if target is ev else first_after(d, b)
        name += args.tag
        done = args.out / f"seq_{name}.json"
        if args.mode == "windows" and done.exists():
            last_epochs = json.loads(done.read_text())["epochs"]
            log(f"{name}: already done ({last_epochs} epoch(s)), skipping")
            continue
        out, info = run_boundary(args, d, n_users, n_items, b, idx, target, device, fallback_epochs=last_epochs)
        last_epochs = info["epochs"]
        info |= {"args": {k: str(v) for k, v in vars(args).items()}, "minutes": round((time.perf_counter() - T0) / 60, 1)}
        if out is None:  # --inner-only
            (args.out / f"seq_{name}_inner.json").write_text(json.dumps(info, indent=2))
            continue
        if args.mode == "val":
            info["val_auc"] = float(roc_auc_score(d.click[idx].ravel(), out["seq_prob"].to_numpy()))
            log(f"VALIDATION AUC {info['val_auc']:.6f} (choice model v2 alone: 0.7882)")
        out.to_parquet(args.out / f"seq_{name}.parquet", index=False)
        (args.out / f"seq_{name}.json").write_text(json.dumps(info, indent=2))
        log(f"wrote {args.out / f'seq_{name}.parquet'} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
