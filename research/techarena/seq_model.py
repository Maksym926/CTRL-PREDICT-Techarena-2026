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

Options beyond the original model:
  --skips N   the items shown but not clicked in the user's last N events join the sequence as extra tokens (with
              their age and a "skipped" type vector), plus a per-candidate "skipped before" count.
  --seeds N   train N seeds and average their probabilities (epochs chosen by the first seed's inner curve;
              with --inner-only, also reports the N-seed ensemble's inner AUC).
  --augment   also train on every pre-period click (clicks before day 0, known only from histories) as a next-item
              event: the clicked item + 9 fillers drawn from the filler distribution. The data looks built that way
              (the non-clicked candidates of clicked and no-click events are statistically identical), and these are
              2.4x the event clicks before day 4. Their loss covers the 10 candidates only (no no-click option).
              That version lost 0.001; --augment-fill 0.019 fixes two ways its events differed from real ones (no
              history items among the fillers, so an in-history candidate was always the real click; histories
              right-aligned while the real ones are left-aligned) and allows --hist.
  --hist N    histories of up to N clicks instead of the data's last 20: each event keeps its given history and gets
              the user's known clicks that are strictly older than the oldest of them prepended (from every history
              list and every labelled train click), so nothing at or after the event can enter. "Which item" accuracy
              keeps rising with known clicks, and evaluation users have 7 days of them.
  --search-once  (windows) the epoch count comes from the first, largest fit's inner holdout and is reused for the
              other boundaries instead of searching at each (with 20-click histories 12 of 13 searches gave 1 epoch).
  --sched cos  linear warmup (--warmup share of the steps), then cosine decay of the learning rate to 0 over exactly
              --sched-epochs epochs; no epoch search (the inner holdout is still scored after each epoch). With a
              constant rate the inner AUC falls from epoch 2 on, so the single epoch ends at full learning rate.
  --right-align  position embeddings count back from the latest click (the newest is always the last position),
              instead of forward from the oldest, so a position means the same recency for every history length.
  --save-inner  where the epoch search runs, also fit the other seeds on the inner fit and write the seed-averaged
              inner-holdout predictions (inner_<name>.parquet), so variants and their ensembles can be compared
              without touching validation.
  --no-user   no free user vector (the user token stays as a zero vector + position): a user is only their history.
              The inner AUC falls fast after epoch 1 (0.820 -> 0.816 -> 0.797), a sign of memorised per-user vectors.
  --hist-drop P  while training, hide each history click with probability P (as padding), so the model cannot lean
              on any single click and sees each event with a different history on every pass.
  --item-lr-mult M  learning rate of the item vectors and biases = M x --lr (0 freezes them at the SVD start).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

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
        self.skip = self.skip_ts = None  # set by attach_skips

    def __len__(self) -> int:
        return len(self.ts)


def attach_skips(train: Events, ev: Events, k: int) -> None:
    """For every event, the items shown but not clicked in the user's k previous train events (most recent event
    first, 10 slots each, 0 = pad or clicked). Only events strictly earlier in time count, so every skip is known when
    the event happens; batch_arrays additionally drops skips at or after the cut for targets."""
    order = np.lexsort((train.event_id, train.ts, train.user))
    u, t = train.user[order], train.ts[order]
    skipped = np.where(train.click[order] > 0, 0, train.cand[order])

    def gather(pos_end: np.ndarray, user: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        items, times = np.zeros((len(user), k * N_CAND), np.int64), np.zeros((len(user), k * N_CAND), np.int64)
        for lag in range(1, k + 1):
            prev = pos_end - lag
            p = np.clip(prev, 0, None)
            ok = (prev >= 0) & (u[p] == user) & (t[p] < ts)
            cols = slice((lag - 1) * N_CAND, lag * N_CAND)
            items[ok, cols] = skipped[p[ok]]
            times[ok, cols] = t[p[ok]][:, None]
        return items, times

    items, times = gather(np.arange(len(order)), u, t)  # train: the k events before each one
    train.skip, train.skip_ts = np.empty_like(items), np.empty_like(times)
    train.skip[order], train.skip_ts[order] = items, times
    # evaluation: the user's last k train events
    ev.skip, ev.skip_ts = gather(np.searchsorted(u, ev.user, side="right"), ev.user, ev.ts)


def extend_histories(train: Events, ev: Events, n_hist: int) -> None:
    """--hist: widen every history to n_hist, prepending the user's known clicks strictly older than the oldest given
    one (most recent first fills the free slots, stored oldest first). Known clicks = every (user, time, item) in any
    train or eval history plus every labelled train click."""
    us, ts, its = [], [], []
    for d in (train, ev):
        m = d.hist > 0
        us.append(np.repeat(d.user, d.hist.shape[1]).reshape(d.hist.shape)[m])
        ts.append(d.hist_ts[m])
        its.append(d.hist[m])
    m = train.click > 0
    us.append(np.repeat(train.user, N_CAND).reshape(m.shape)[m])
    ts.append(np.repeat(train.ts, N_CAND).reshape(m.shape)[m])
    its.append(train.cand[m])
    clicks = np.unique(np.stack([np.concatenate(us), np.concatenate(ts), np.concatenate(its)], 1), axis=0)
    key = clicks[:, 0] * 10 ** 10 + clicks[:, 1]  # sorted by user, then time (timestamps < 1e10)
    for d in (train, ev):
        n, width = len(d), d.hist.shape[1]
        given = (d.hist > 0).sum(1)
        start = np.searchsorted(key, d.user * 10 ** 10, side="left")
        oldest = np.where(given > 0, d.hist_ts[:, 0], 0)  # oldest first, so column 0 is the oldest given click
        end = np.where(given > 0, np.searchsorted(key, d.user * 10 ** 10 + oldest, side="left"), start)
        k = np.minimum(end - start, n_hist - given)  # older clicks that fit
        hist, hist_ts = np.zeros((n, n_hist), np.int64), np.zeros((n, n_hist), np.int64)
        rows = np.repeat(np.arange(n), k)
        pos = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        src = np.repeat(end - k, k) + pos
        hist[rows, pos], hist_ts[rows, pos] = clicks[src, 2], clicks[src, 1]
        rows = np.repeat(np.arange(n), given)
        pos = np.arange(given.sum()) - np.repeat(np.cumsum(given) - given, given)
        hist[rows, np.repeat(k, given) + pos] = d.hist[rows, pos]
        hist_ts[rows, np.repeat(k, given) + pos] = d.hist_ts[rows, pos]
        assert width <= n_hist and (hist > 0).sum(1).max() <= n_hist
        d.hist, d.hist_ts = hist, hist_ts
        log(f"histories widened to {n_hist}: {(k > 0).mean():.1%} of {n:,} events gained older clicks "
            f"(mean length {given.mean():.1f} -> {(hist > 0).sum(1).mean():.1f})")


def load(data: Path, skips: int = 0):
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
    if skips:
        attach_skips(train, ev, skips)
        log(f"skipped items from up to {skips} previous events: {(train.skip > 0).sum(1).mean():.1f} per train event, "
            f"{(ev.skip > 0).sum(1).mean():.1f} per eval event")
    return train, ev, len(users), len(items)


def first_after(d: Events, b: int, until: int | None = None) -> np.ndarray:
    """Index of each user's first event at or after b (ties broken by event_id, as src.validation does)."""
    idx = np.flatnonzero((d.ts >= b) & (d.ts < until if until is not None else True))
    idx = idx[np.lexsort((d.event_id[idx], d.ts[idx]))]
    _, first = np.unique(d.user[idx], return_index=True)
    return np.sort(idx[first])


def age_bucket(ts: np.ndarray, at: np.ndarray, valid: np.ndarray) -> np.ndarray:
    age_h = np.where(valid, (ts - at) / HOUR, 0.0)
    return np.clip((np.log2(1 + np.maximum(age_h, 0)) * 2).astype(np.int64), 0, N_AGE - 1)


def repeats(cand: np.ndarray, items: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per candidate, how often it appears in items (capped at 3)."""
    return np.minimum(((cand[:, :, None] == items[:, None, :]) & valid[:, None, :]).sum(2), 3)


def batch_arrays(d: Events, idx: np.ndarray, cut: int | None = None, skips: int = 0) -> dict[str, np.ndarray]:
    """Model inputs for events idx; with cut, history clicks (and skips) at or after cut are dropped (leak-free
    targets). skips > 0 adds the items skipped in the user's `skips` previous events."""
    hist, hts = d.hist[idx].copy(), d.hist_ts[idx]
    if cut is not None:
        hist[hts >= cut] = 0
    valid = hist > 0
    ts = d.ts[idx][:, None]
    age = age_bucket(ts, hts, valid)
    last = np.where(valid, hts, -1).max(1)
    gap = np.where(last > 0, np.log1p(np.maximum(d.ts[idx] - last, 0) / HOUR), np.log1p(1000.0))
    cand = d.cand[idx]
    rep = repeats(cand, hist, valid)
    dense = [gap, valid.sum(1) / hist.shape[1], (rep > 0).sum(1) / N_CAND]
    out = {"user": d.user[idx], "hist": hist, "age": age, "cand": cand, "rep": rep, "click": d.click[idx]}
    if skips:
        sk, sts = d.skip[idx, :skips * N_CAND].copy(), d.skip_ts[idx, :skips * N_CAND]
        if cut is not None:
            sk[sts >= cut] = 0
        sv = sk > 0
        out |= {"skip": sk, "skip_age": age_bucket(ts, sts, sv), "skip_rep": repeats(cand, sk, sv)}
        dense.append(sv.sum(1) / sk.shape[1])
    out["dense"] = np.stack(dense, 1).astype(np.float32)
    return out


def pseudo_events(d: Events, b: int, n_items: int, seed: int, frac: float = 1.0, hist_fill: float = 0.0,
                  decay: float = 0.85) -> SimpleNamespace:
    """--augment: every pre-period click (before the first train event; known only from histories) as an extra
    next-item event, the way a clicked event looks: the clicked item plus 9 fillers drawn from the filler distribution
    (non-clicked candidates of events before b), the user's earlier pre-period clicks as history. Pre-period clicks
    precede every boundary, so this is leak-free for any b.
    hist_fill > 0 (--augment-fill) builds them like the real events instead: histories left-aligned (oldest first,
    pads last, as Events stores them; the original right-aligned ones gave positions another meaning) and as wide as
    the real ones (so it combines with --hist), and each filler slot is, with probability hist_fill, one of the user's
    own earlier clicks (recency weights decay ** position, newest = 0), as the real filler sampler does in ~2% of
    slots. Without that, an in-history candidate of a pseudo event is always the real click (~30% in real events)."""
    d0 = int(d.ts.min())
    width = d.hist.shape[1] if hist_fill > 0 else HIST
    m = (d.hist > 0) & (d.hist_ts < d0)
    raw = np.stack([np.repeat(d.user, d.hist.shape[1]).reshape(d.hist.shape)[m], d.hist_ts[m], d.hist[m]], 1)
    clicks = np.unique(raw, axis=0)  # sorted by user, time, item
    rng = np.random.default_rng(seed)
    if frac < 1:
        clicks = clicks[np.sort(rng.choice(len(clicks), int(len(clicks) * frac), replace=False))]
    u, t, it = clicks[:, 0], clicks[:, 1], clicks[:, 2]
    n = len(u)
    j = np.arange(n)
    start = np.searchsorted(u, u, side="left")  # first row of each user's clicks
    if hist_fill > 0:
        k = np.minimum(j - start, width)  # earlier clicks that fit, stored from slot 0
        src = (j - k)[:, None] + np.arange(width)[None, :]
        ok = np.arange(width)[None, :] < k[:, None]
    else:
        src = j[:, None] - np.arange(HIST, 0, -1)[None, :]  # the 20 rows before, oldest first, right-aligned
        ok = src >= start[:, None]
    hist = np.where(ok, it[np.clip(src, 0, n - 1)], 0)
    hist_ts = np.where(ok, t[np.clip(src, 0, n - 1)], 0)
    before = d.ts < b
    q = np.bincount(d.cand[before][d.click[before] == 0], minlength=n_items + 1).astype(float)
    q[0] = 0.0
    q /= q.sum()
    cand = rng.choice(n_items + 1, size=(n, N_CAND), p=q)
    pos = rng.integers(0, N_CAND, n)
    if hist_fill > 0:  # history fillers: newest-first position r with weight decay ** r, truncated to the history
        fill = (rng.random((n, N_CAND)) < hist_fill) & (k[:, None] > 0)
        rows, slot = np.nonzero(fill)
        kk = k[rows]
        r = np.floor(np.log1p(-rng.random(len(rows)) * (1 - decay ** kk)) / np.log(decay)).astype(np.int64)
        cand[rows, slot] = hist[rows, kk - 1 - np.minimum(r, kk - 1)]
    cand[j, pos] = it
    clash = cand == it[:, None]
    clash[j, pos] = False
    while clash.any():  # a filler drawn equal to the clicked item
        cand[clash] = rng.choice(n_items + 1, size=int(clash.sum()), p=q)
        clash = cand == it[:, None]
        clash[j, pos] = False
    click = np.zeros((n, N_CAND), np.float32)
    click[j, pos] = 1.0
    return SimpleNamespace(user=u, ts=t, hist=hist, hist_ts=hist_ts, cand=cand, click=click, skip=None, skip_ts=None)


def svd_init(d: Events, b: int, n_users: int, n_items: int, rank: int = 11) -> np.ndarray:
    """Item vectors (rows = item codes incl. pad 0) from the normalised click SVD on clicks before b."""
    before = d.ts < b
    width = d.hist.shape[1]
    hu = np.repeat(d.user[before], width).reshape(-1, width)
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
                 init: np.ndarray | None, skips: bool = False, n_hist: int = HIST, right_align: bool = False,
                 no_user: bool = False):
        super().__init__()
        self.right_align = right_align
        self.no_user = no_user
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
        self.pos = nn.Embedding(n_hist + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.rep = nn.Embedding(4, 1)
        nn.init.zeros_(self.rep.weight)
        self.mf_scale = nn.Parameter(torch.tensor(4.0))  # sharpness of the MF path (choice-MF default scale)
        self.score = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, 1))
        self.noclick = nn.Sequential(nn.Linear(dim + 3 + skips, dim), nn.GELU(), nn.Linear(dim, 1))
        if skips:  # created last, so the model without skips initialises exactly as before
            self.skip_type = nn.Parameter(torch.randn(dim) * 0.02)  # marks a token as shown-not-clicked
            self.skip_rep = nn.Embedding(4, 1)
            nn.init.zeros_(self.skip_rep.weight)
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
        if self.no_user:
            u = torch.zeros_like(u)
        # direct MF path: raw history average + user vector, so training starts from the SVD taste match
        q = e.sum(1) / (hist > 0).sum(1, keepdim=True).clamp(min=1) + u
        tok = e + self.age(b["age"])
        if self.right_align:  # histories are left-aligned (oldest first, pads last): the newest click gets position width
            width = hist.shape[1]
            n_valid = (hist > 0).sum(1, keepdim=True)
            pid = (width - n_valid + 1 + torch.arange(width, device=hist.device)[None]).clamp(max=width)
            pos = self.pos(torch.cat([torch.zeros_like(n_valid), pid], 1))
        else:
            pos = self.pos.weight[None]
        seq = torch.cat([u[:, None], tok], 1) + pos
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=hist.device), hist == 0], 1)
        if "skip" in b:  # skipped items join the sequence as extra tokens (no position, a type vector instead)
            stok = self.item(b["skip"]) + self.age(b["skip_age"]) + self.skip_type
            seq, pad = torch.cat([seq, stok], 1), torch.cat([pad, b["skip"] == 0], 1)
        h = self.encoder(seq, src_key_padding_mask=pad)
        w = (~pad).float()[:, :, None]
        p = (h * w).sum(1) / w.sum(1)
        c = self.item(b["cand"])
        a, _ = self.cross(c, h, h, key_padding_mask=pad, need_weights=False)
        pe = p[:, None].expand_as(c)
        s = self.score(torch.cat([c, a, pe, c * a, c * pe], -1)).squeeze(-1)
        s = s + self.mf_scale * (c * q[:, None]).sum(-1) + self.item_bias(b["cand"]).squeeze(-1) \
            + self.rep(b["rep"]).squeeze(-1)
        if "skip" in b:
            s = s + self.skip_rep(b["skip_rep"]).squeeze(-1)
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


def fit(args, d: Events, n_users: int, n_items: int, b: int, epochs: int, device, seed: int,
        holdout: tuple[dict, np.ndarray] | None = None) -> tuple[SeqModel, list[float], np.ndarray | None]:
    """Train on events before b. With a holdout, log its AUC after each epoch (and stop after `patience` misses);
    also returns the holdout predictions of the best epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.flatnonzero(d.ts < b)
    if args.fit_frac < 1:
        idx = np.sort(np.random.choice(idx, int(len(idx) * args.fit_frac), replace=False))
    arrays = batch_arrays(d, idx, skips=args.skips)
    if args.augment:
        pe = pseudo_events(d, b, n_items, seed, frac=args.fit_frac, hist_fill=args.augment_fill)
        extra = batch_arrays(pe, np.arange(len(pe.user)))
        arrays = {k: np.concatenate([arrays[k], extra[k]]) for k in arrays}
        arrays["pseudo"] = np.r_[np.zeros(len(idx), bool), np.ones(len(pe.user), bool)]
        log(f"  + {len(pe.user):,} pre-period next-item events")
    train = to_device(arrays, device)
    n_rows = len(arrays["user"])
    model = SeqModel(n_users, n_items, args.dim, args.layers, args.heads, args.dropout,
                     svd_init(d, b, n_users, n_items) if args.svd_init else None, skips=args.skips > 0,
                     n_hist=d.hist.shape[1], right_align=args.right_align, no_user=args.no_user).to(device)
    if args.item_lr_mult != 1.0:
        emb = [model.item.weight, model.item_bias.weight]
        rest = [q for q in model.parameters() if all(q is not e for e in emb)]
        opt = torch.optim.AdamW([{"params": rest}, {"params": emb, "lr": args.lr * args.item_lr_mult}],
                                lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = None
    if args.sched == "cos":  # warmup, then cosine to 0 at the end of the last epoch
        total = epochs * -(-n_rows // args.batch)
        warm = max(1, int(args.warmup * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, total - warm))))
    curve, best, misses, best_pred = [], -1.0, 0, None
    if holdout is not None:
        p0 = predict(model, holdout[0])[:, 1:]
        log(f"  epoch 0 (untrained, SVD start): inner-holdout AUC {roc_auc_score(holdout[1].ravel(), p0.ravel()):.6f}")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_rows, device=device)
        total, t = 0.0, time.perf_counter()
        for s in range(0, n_rows, args.batch):
            bi = perm[s:s + args.batch]
            batch = {k: v[bi] for k, v in train.items()}
            if args.hist_drop > 0:  # training only: random history clicks become padding
                h = batch["hist"]
                batch["hist"] = h.masked_fill(torch.rand(h.shape, device=h.device) < args.hist_drop, 0)
            logits = model(batch)
            if "pseudo" in batch:  # extra events have no no-click option: their loss covers the 10 candidates only
                logits = torch.cat([logits[:, :1].masked_fill(batch["pseudo"][:, None], -1e9), logits[:, 1:]], 1)
            loss = loss_fn(logits, batch["click"])
            if args.emb_l2 > 0:
                loss = loss + args.emb_l2 * model.penalty(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            total += loss.item() * len(bi)
            step = s // args.batch + 1
            if args.log_every and step % args.log_every == 0:
                log(f"    step {step}: loss {total / min(s + args.batch, n_rows):.4f}")
        msg = f"  epoch {ep}: loss {total / n_rows:.4f} ({time.perf_counter() - t:.0f} s)"
        if holdout is not None:
            probs = predict(model, holdout[0])
            curve.append(float(roc_auc_score(holdout[1].ravel(), probs[:, 1:].ravel())))
            msg += f", inner-holdout AUC {curve[-1]:.6f}"
            if sched is not None:  # fixed schedule: the model is the last epoch's
                best_pred = probs
            elif curve[-1] > best:
                best, misses, best_pred = curve[-1], 0, probs
            else:
                misses += 1
        log(msg)
        if holdout is not None and misses >= args.patience and sched is None:
            break
    return model, curve, best_pred


def run_boundary(args, d: Events, n_users: int, n_items: int, b: int, target_idx: np.ndarray, target: Events,
                 device, fallback_epochs: int | None = None, name: str = "") -> tuple[pd.DataFrame, dict]:
    """Pick epochs on the inner holdout before b, refit on everything before b, score the targets."""
    bi = b - INNER_GAP
    n_inner = int(np.sum(d.ts < bi))
    ens_auc = None
    if args.search_once and fallback_epochs is not None:
        curve, epochs = [], fallback_epochs
        log(f"boundary {b}: --search-once, using {epochs} epoch(s) from the first boundary")
    elif n_inner < MIN_INNER and fallback_epochs is not None:
        curve, epochs = [], fallback_epochs
        log(f"boundary {b}: only {n_inner:,} inner-fit events, using {epochs} epoch(s) from the next boundary")
    else:
        hold_idx = first_after(d, bi, until=b)
        hold = batch_arrays(d, hold_idx, cut=bi, skips=args.skips)
        hold_dev = to_device(hold, device)
        log(f"boundary {b}: inner fit {n_inner:,} events, inner holdout {len(hold_idx):,} events")
        n_ep = args.max_epochs if args.sched == "const" else args.sched_epochs
        _, curve, best_pred = fit(args, d, n_users, n_items, bi, n_ep, device, args.seed,
                                  holdout=(hold_dev, hold["click"]))
        epochs = int(np.argmax(curve)) + 1 if args.sched == "const" else n_ep
        log(f"boundary {b}: inner epoch {epochs} (AUC {curve[epochs - 1]:.6f})")
        if (args.inner_only and args.seeds > 1) or args.save_inner:
            # the ensemble's inner score: the first seed at the chosen epoch + the others trained for that many epochs
            preds = [best_pred] + [predict(fit(args, d, n_users, n_items, bi, epochs, device, args.seed + s)[0],
                                           hold_dev) for s in range(1, args.seeds)]
            ens = np.mean(preds, 0)
            ens_auc = float(roc_auc_score(hold["click"].ravel(), ens[:, 1:].ravel()))
            log(f"boundary {b}: {args.seeds}-seed ensemble inner AUC {ens_auc:.6f}")
            if args.save_inner:
                pd.DataFrame({"rowid": d.rowid[hold_idx].ravel(), "seq_prob": ens[:, 1:].ravel(),
                              "seq_noclick": np.repeat(ens[:, 0], N_CAND), "clicked": hold["click"].ravel()}
                             ).to_parquet(args.out / f"inner_{name}.parquet", index=False)
    if args.inner_only:
        info = {"boundary": b, "inner_curve": curve, "epochs": epochs, "best_inner_auc": max(curve, default=None)}
        if ens_auc is not None:
            info["ensemble_inner_auc"] = ens_auc
        return None, info
    log(f"refitting {args.seeds} seed(s) on {np.sum(d.ts < b):,} events")
    target_dev = to_device(batch_arrays(target, target_idx, cut=b, skips=args.skips), device)
    probs = np.mean([predict(fit(args, d, n_users, n_items, b, epochs, device, args.seed + s)[0], target_dev)
                     for s in range(args.seeds)], 0)
    p = probs[:, 1:]
    out = pd.DataFrame({"rowid": target.rowid[target_idx].ravel(), "seq_prob": p.ravel(),
                        "seq_noclick": np.repeat(probs[:, 0], N_CAND),
                        "seq_rank": ((-p).argsort(1).argsort(1) + 1).ravel().astype(np.float32)})
    info = {"boundary": b, "inner_curve": curve, "epochs": epochs, "targets": int(len(target_idx))}
    if ens_auc is not None:
        info["ensemble_inner_auc"] = ens_auc
    return out, info


def main(argv: list[str] | None = None, loaded: tuple | None = None) -> None:
    """argv as on the command line; `loaded` = load(...) output to reuse across runs (e.g. variants in a notebook)."""
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
    ap.add_argument("--seeds", type=int, default=1, help="average this many seeds (seed, seed+1, ...)")
    ap.add_argument("--skips", type=int, default=0,
                    help="add the items shown but not clicked in the user's last N events (0 = off)")
    ap.add_argument("--augment", action="store_true", help="add pre-period clicks as extra next-item events")
    ap.add_argument("--augment-fill", type=float, default=0.0,
                    help="--augment built like real events: left-aligned histories, and this share of filler slots "
                         "drawn from the user's own earlier clicks (real slates: ~0.019)")
    ap.add_argument("--hist", type=int, default=HIST, help="history length: the given 20 + older known clicks")
    ap.add_argument("--search-once", action="store_true",
                    help="windows: choose epochs on the first (latest) boundary's inner holdout, reuse them for the rest")
    ap.add_argument("--sched", choices=["const", "cos"], default="const",
                    help="cos: warmup, then cosine decay to 0 over exactly --sched-epochs epochs (no epoch search)")
    ap.add_argument("--warmup", type=float, default=0.03, help="--sched cos: share of the steps spent warming up")
    ap.add_argument("--sched-epochs", type=int, default=1, help="--sched cos: number of epochs the decay spans")
    ap.add_argument("--right-align", action="store_true", help="position embeddings count back from the latest click")
    ap.add_argument("--save-inner", action="store_true",
                    help="write the seed-averaged inner-holdout predictions where the epoch search runs")
    ap.add_argument("--no-user", action="store_true", help="no free user vector: a user is only their history")
    ap.add_argument("--hist-drop", type=float, default=0.0, help="training: hide each history click with this probability")
    ap.add_argument("--item-lr-mult", type=float, default=1.0, help="item vectors' learning rate = this x --lr (0 = frozen)")
    ap.add_argument("--fit-frac", type=float, default=1.0, help="<1 only for quick smoke tests")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0, help="print the running loss every N steps")
    args = ap.parse_args(argv)
    if args.augment and args.skips:
        raise SystemExit("--augment with --skips is not supported (pre-period events have no skipped items)")
    if args.augment and args.hist != HIST and args.augment_fill <= 0:
        raise SystemExit(f"--augment builds {HIST}-click histories; with --hist, use --augment-fill")
    if args.augment_fill > 0 and not args.augment:
        raise SystemExit("--augment-fill needs --augment")
    if args.hist < HIST:
        raise SystemExit(f"--hist must be at least {HIST} (the data's own history length)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    d, ev, n_users, n_items = loaded or load(args.data, args.skips)
    if args.skips and (d.skip is None or d.skip.shape[1] < args.skips * N_CAND):
        raise SystemExit(f"--skips {args.skips}: data was loaded with fewer previous events")
    if d.hist.shape[1] != args.hist:
        if loaded is not None:
            raise SystemExit(f"--hist {args.hist}: the preloaded data has {d.hist.shape[1]}-click histories")
        extend_histories(d, ev, args.hist)
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
        out, info = run_boundary(args, d, n_users, n_items, b, idx, target, device, fallback_epochs=last_epochs,
                                 name=name)
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
