"""Choice matrix factorisation: learn user/item taste vectors directly from the 10-candidate + no-click choice task.

Usage: python src/choice_mf.py [--dim 64] [--l2 1e-4] [--cutoff-day 4]

Unlike click-matrix SVD, every impression is used: each event is an 11-way choice (no click, or one of 10 candidates).
    candidate utility = item bias + (user vector + weighted mean of history item vectors) . item vector + repeat features
    no-click utility  = user no-click bias + history length / time-since-last-click terms
The softmax probability of each candidate is the prediction. The epoch count is chosen on an inner holdout (each user's
first event in the last half day before the cutoff, fitting on earlier events), then the model is refitted on all fit
events and scored once with src.validation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_candidates, load_events  # noqa: E402
from src.evaluate import auc  # noqa: E402
from src.validation import DAY, make_split, score  # noqa: E402

HISTORY = 20
OUT_DIR = Path("data/validation")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- encoding
def code_maps() -> tuple[dict, dict]:
    te, ee = load_events("train"), load_events("eval")
    tc, ec = load_candidates("train"), load_candidates("eval")
    users = pl.concat([te["user_id"], ee["user_id"]]).unique().sort()
    items = pl.concat([tc["item_id"], ec["item_id"], te["history_item_ids"].explode().drop_nulls(),
                       ee["history_item_ids"].explode().drop_nulls()]).unique().sort()
    return {u: i for i, u in enumerate(users)}, {it: i for i, it in enumerate(items)}


def encode(events: pl.DataFrame, cands: pl.DataFrame, umap: dict, imap: dict) -> dict[str, np.ndarray]:
    """Arrays for a set of events (in the order given); history right-aligned so slot 19 is the latest click."""
    events = events.with_row_index("e")
    labelled = "clicked" in cands.columns
    cols = ["item_id", "rowid"] + (["clicked"] if labelled else [])
    grouped = cands.join(events.select("event_id"), on="event_id", how="semi") \
                   .group_by("event_id", maintain_order=True).agg(cols)
    j = events.join(grouped, on="event_id", how="left").sort("e")
    n, pad = j.height, len(imap)

    flat = j.select(cols).explode(cols)
    cand = flat["item_id"].replace_strict(imap, return_dtype=pl.Int64).to_numpy().reshape(n, 10)
    rowid = flat["rowid"].to_numpy().astype(str).reshape(n, 10)
    if labelled:
        clicked = flat["clicked"].to_numpy().reshape(n, 10)
        label = np.where(clicked.any(1), clicked.argmax(1) + 1, 0)  # 0 = no click
    else:
        label = np.full(n, -1)

    hist_len = j["history_item_ids"].list.len().to_numpy()
    hist = np.full((n, HISTORY), pad, dtype=np.int64)
    hist_t = np.zeros((n, HISTORY), dtype=np.int64)
    h = (j.select("e", "history_item_ids", "history_timestamps").filter(pl.col("history_item_ids").list.len() > 0)
         .with_columns(pl.col("history_item_ids").list.len().alias("n"))
         .explode(["history_item_ids", "history_timestamps"])
         .with_columns(pl.int_range(pl.len()).over("e").alias("k")))
    e_idx, slot = h["e"].to_numpy(), HISTORY - h["n"].to_numpy() + h["k"].to_numpy()
    hist[e_idx, slot] = h["history_item_ids"].replace_strict(imap, return_dtype=pl.Int64).to_numpy()
    hist_t[e_idx, slot] = h["history_timestamps"].to_numpy()

    ts = j["timestamp"].to_numpy()
    # Repeat features: is the candidate in the history, how often, how long ago it was last clicked
    eq = cand[:, :, None] == hist[:, None, :]
    count = eq.sum(-1)
    last_slot = np.where(eq, np.arange(HISTORY), -1).max(-1)
    present = last_slot >= 0
    last_t = np.take_along_axis(hist_t, np.maximum(last_slot, 0), axis=1)
    age_h = np.where(present, (ts[:, None] - last_t) / 3600.0, 0.0)
    rep = np.stack([present, np.log1p(count), present * np.exp(-age_h / 24.0), present * np.log1p(age_h)], -1)

    gap_h = np.where(hist_len > 0, (ts - hist_t[:, -1]) / 3600.0, 0.0)
    noclick = np.stack([hist_len / HISTORY, np.log1p(gap_h), hist_len == 0], -1)

    return {"user": j["user_id"].replace_strict(umap, return_dtype=pl.Int64).to_numpy(), "cand": cand, "hist": hist,
            "rep": rep.astype(np.float32), "noclick": noclick.astype(np.float32), "label": label, "rowid": rowid,
            "ts": ts}


def first_per_user(events: pl.DataFrame) -> pl.DataFrame:
    return events.sort("timestamp", "event_id").group_by("user_id", maintain_order=True).first().select(events.columns)


def click_pairs(boundary: int, umap: dict, imap: dict) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (user, item) clicks allowed at a boundary: labelled clicks before it plus every history click
    (train and eval histories) timestamped before it."""
    te = load_events("train")
    lab = (load_candidates("train").filter(pl.col("clicked") == 1)
           .join(te.filter(pl.col("timestamp") < boundary).select("event_id", "user_id"), on="event_id")
           .select("user_id", "item_id"))
    hist = (pl.concat([te, load_events("eval")]).select("user_id", "history_item_ids", "history_timestamps")
            .explode(["history_item_ids", "history_timestamps"]).filter(pl.col("history_timestamps") < boundary)
            .select("user_id", pl.col("history_item_ids").alias("item_id")))
    pairs = pl.concat([lab, hist]).unique()
    return (pairs["user_id"].replace_strict(umap, return_dtype=pl.Int64).to_numpy(),
            pairs["item_id"].replace_strict(imap, return_dtype=pl.Int64).to_numpy())


def click_svd(boundary: int, umap: dict, imap: dict, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Monish's normalised click-matrix SVD at a boundary (see click_pairs). Returns L2-normalised user and item
    vectors."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    r, c = click_pairs(boundary, umap, imap)
    du = np.bincount(r, minlength=len(umap)) + 1.0
    di = np.bincount(c, minlength=len(imap)) + 1.0
    m = coo_matrix((1 / np.sqrt(du[r] * di[c]), (r, c)), shape=(len(umap), len(imap))).tocsr()
    u, s, vt = svds(m, k=rank, random_state=20261024)
    users, items = u * s, vt.T

    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return unit(users).astype(np.float32), unit(items).astype(np.float32)


# ---------------------------------------------------------------- model
class ChoiceMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, user_emb: bool = True):
        super().__init__()
        pad = n_items
        # Without a free per-user vector, a user's taste is built only from their history items, so a user with
        # few events cannot memorise their own choices.
        self.user_emb = user_emb
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=pad)      # candidate side
        self.context = nn.Embedding(n_items + 1, dim, padding_idx=pad)   # history side
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=pad)
        self.user_noclick = nn.Embedding(n_users, 1)
        self.slot_weight = nn.Parameter(torch.zeros(HISTORY))            # learned recency weighting of history slots
        self.rep = nn.Linear(4, 1)
        self.noclick = nn.Linear(3, 1)
        self.pad = pad
        for emb in (self.user, self.item, self.context):
            nn.init.normal_(emb.weight, std=0.01)
        for emb in (self.item_bias, self.user_noclick):
            nn.init.zeros_(emb.weight)
        self.register_buffer("user0", self.user.weight.detach().clone())
        self.register_buffer("item0", self.item.weight.detach().clone())
        self.register_buffer("context0", torch.zeros_like(self.context.weight))

    def init_from_svd(self, users: np.ndarray, items: np.ndarray, scale: float) -> None:
        """Start taste vectors at the SVD solution (utility = scale * SVD cosine) and regularise towards it.
        Without user vectors, history items start at the SVD item vectors (user taste = mean of history)."""
        r = users.shape[1]
        with torch.no_grad():
            self.user.weight[:, :r] = torch.from_numpy(users) * scale ** 0.5
            self.item.weight[:items.shape[0], :r] = torch.from_numpy(items) * scale ** 0.5
            if not self.user_emb:
                self.context.weight[:items.shape[0], :r] = torch.from_numpy(items) * scale ** 0.5
                self.context0.copy_(self.context.weight.detach())
        self.user0.copy_(self.user.weight.detach())
        self.item0.copy_(self.item.weight.detach())

    def forward(self, user, cand, hist, rep, noclick):
        mask = hist != self.pad
        w = torch.softmax(self.slot_weight.expand_as(hist).masked_fill(~mask, -1e9), dim=1) * mask
        user_vec = (w.unsqueeze(-1) * self.context(hist)).sum(1)
        if self.user_emb:
            user_vec = user_vec + self.user(user)
        item_vec = self.item(cand)
        util = (item_vec * user_vec.unsqueeze(1)).sum(-1) + self.item_bias(cand).squeeze(-1) + self.rep(rep).squeeze(-1)
        no_click = self.noclick(noclick).squeeze(-1) + self.user_noclick(user).squeeze(-1)
        logits = torch.cat([no_click.unsqueeze(1), util], dim=1)
        # Penalise drift from the starting point (SVD init, or zero for parts without one)
        l2 = ((item_vec - self.item0[cand]).pow(2).sum(-1).mean()
              + (self.context(hist) - self.context0[hist]).pow(2).sum(-1).mean())
        if self.user_emb:
            l2 = l2 + (self.user(user) - self.user0[user]).pow(2).sum(-1).mean()
        else:
            l2 = l2 + self.user_noclick(user).pow(2).sum(-1).mean()  # the only per-user parameter left
        return logits, l2

    def click_loss(self, user, pos, n_neg: int):
        """Auxiliary task on every past click: rank the clicked item above n_neg random items (taste only)."""
        neg = torch.randint(0, self.pad, (len(user), n_neg))
        items = torch.cat([pos.unsqueeze(1), neg], dim=1)
        scores = (self.item(items) * self.user(user).unsqueeze(1)).sum(-1)
        return nn.functional.cross_entropy(scores, torch.zeros(len(user), dtype=torch.long))


def tensors(d: dict) -> list[torch.Tensor]:
    return [torch.from_numpy(np.array(d[k])) for k in ("user", "cand", "hist", "rep", "noclick")]  # writable copies


def predict(model: ChoiceMF, d: dict, batch: int = 8192) -> np.ndarray:
    model.eval()
    x = tensors(d)
    out = []
    with torch.no_grad():
        for s in range(0, len(d["user"]), batch):
            logits, _ = model(*(t[s:s + batch] for t in x))
            out.append(torch.softmax(logits, 1)[:, 1:].numpy())
    return np.concatenate(out).reshape(-1)


def holdout_auc(model: ChoiceMF, holdout: dict) -> float:
    labels = (np.arange(1, 11) == holdout["label"][:, None]).reshape(-1)  # class j+1 = candidate j clicked
    return auc(labels, predict(model, holdout))


def cells(d: dict) -> np.ndarray:
    """History-length x time-since-last-click cell of each event (the buckets Monish used for reweighting)."""
    hist_len = np.rint(d["noclick"][:, 0] * HISTORY)
    gap_h = np.expm1(d["noclick"][:, 1])
    return np.digitize(hist_len, [1, 10, 20]) * 5 + np.digitize(gap_h, [6, 24, 72, 168])


def match_weights(d: dict, target: dict, cap: float = 50.0) -> np.ndarray:
    """Per-event weights that make d's cell mix match target's (target = unlabelled evaluation events)."""
    c, t = cells(d), cells(target)
    share_d = np.bincount(c, minlength=20) / len(c)
    share_t = np.bincount(t, minlength=20) / len(t)
    w = np.minimum(share_t[c] / np.maximum(share_d[c], 1e-12), cap)
    return (w / w.mean()).astype(np.float32)


def strip_time(d: dict) -> dict:
    """Drop time-dependent inputs: repeat age features and time since last click."""
    d = dict(d)
    d["rep"] = d["rep"].copy()
    d["rep"][:, :, 2:] = 0.0
    d["noclick"] = d["noclick"].copy()
    d["noclick"][:, 1] = 0.0
    return d


def train(d: dict, n_users: int, n_items: int, args, epochs: int, svd, holdout: dict | None = None, clicks=None):
    """Returns the model and, with a holdout, the holdout AUC after each epoch. `clicks` = (users, items) of all
    past clicks for the auxiliary click task (used when args.aux_weight > 0)."""
    torch.manual_seed(args.seed)
    model = ChoiceMF(n_users, n_items, args.dim, user_emb=bool(getattr(args, "user_emb", 1)))
    if args.svd_rank:
        model.init_from_svd(*svd, scale=args.scale)
    if getattr(args, "no_time", 0):
        model.slot_weight.requires_grad_(False)  # equal weight for every history slot
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    x, y = tensors(d), torch.from_numpy(d["label"])
    weight = torch.from_numpy(d["weight"]) if "weight" in d else None
    use_aux = clicks is not None and getattr(args, "aux_weight", 0) > 0
    aux = (torch.from_numpy(clicks[0]), torch.from_numpy(clicks[1])) if use_aux else None
    curve = []
    if holdout is not None:
        log(f"  before training: holdout AUC {holdout_auc(model, holdout):.5f}")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y))
        total = 0.0
        for s in range(0, len(y), args.batch):
            idx = perm[s:s + args.batch]
            logits, l2 = model(*(t[idx] for t in x))
            ce = nn.functional.cross_entropy(logits, y[idx], reduction="none")
            ce = (ce * weight[idx]).mean() if weight is not None else ce.mean()
            loss = ce + args.l2 * l2
            if aux is not None:
                j = torch.randint(0, len(aux[0]), (len(idx) * args.aux_ratio,))
                loss = loss + args.aux_weight * model.click_loss(aux[0][j], aux[1][j], args.aux_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        msg = f"  epoch {epoch + 1}: train loss {total / len(y):.4f}"
        if holdout is not None:
            curve.append(holdout_auc(model, holdout))
            msg += f", holdout AUC {curve[-1]:.5f}"
        log(msg)
    return model, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-day", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--svd-rank", type=int, default=11, help="initialise taste vectors from click SVD (0 = random)")
    ap.add_argument("--scale", type=float, default=4.0, help="initial utility = scale * SVD cosine")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--user-emb", type=int, default=1, help="0 = no free per-user taste vector (history only)")
    ap.add_argument("--reweight", type=int, default=0, help="1 = weight training events to eval's history/gap mix")
    ap.add_argument("--no-time", type=int, default=0, help="1 = no time-dependent inputs or recency weighting")
    ap.add_argument("--aux-weight", type=float, default=0.0, help="weight of the auxiliary all-past-clicks task")
    ap.add_argument("--aux-ratio", type=int, default=2, help="click pairs per choice event in each batch")
    ap.add_argument("--aux-neg", type=int, default=20, help="random negative items per click pair")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(12)
    t0 = time.perf_counter()

    split = make_split(args.cutoff_day)
    b, inner_b = split.cutoff_ts, split.cutoff_ts - DAY // 2
    umap, imap = code_maps()
    n_users, n_items = len(umap), len(imap)

    fit_all = encode(split.fit_events, split.fit_candidates, umap, imap)
    inner_fit = encode(split.fit_events.filter(pl.col("timestamp") < inner_b), split.fit_candidates, umap, imap)
    inner_hold = encode(first_per_user(split.fit_events.filter(pl.col("timestamp") >= inner_b)),
                        split.fit_candidates, umap, imap)
    val = encode(split.val_events, split.val_candidates, umap, imap)
    log(f"fit {len(fit_all['user']):,} events | inner fit {len(inner_fit['user']):,} / holdout "
        f"{len(inner_hold['user']):,} | val {len(val['user']):,} | {n_users:,} users, {n_items:,} items")

    if args.reweight:  # target mix: the unlabelled evaluation events
        target = encode(load_events("eval"), load_candidates("eval"), umap, imap)
        for d in (fit_all, inner_fit):
            d["weight"] = match_weights(d, target)
            log(f"  reweighted {len(d['weight']):,} events: max weight {d['weight'].max():.1f}, effective sample size "
                f"{d['weight'].sum() ** 2 / (d['weight'] ** 2).sum():,.0f}")
    if args.no_time:
        fit_all, inner_fit, inner_hold, val = (strip_time(d) for d in (fit_all, inner_fit, inner_hold, val))

    # Each phase gets its own SVD, built only from clicks before that phase's boundary
    svd_inner = click_svd(inner_b, umap, imap, args.svd_rank) if args.svd_rank else None
    svd_full = click_svd(b, umap, imap, args.svd_rank) if args.svd_rank else None

    # Auxiliary click pairs, again only from before each phase's boundary
    clicks_inner = click_pairs(inner_b, umap, imap) if args.aux_weight > 0 else None
    clicks_full = click_pairs(b, umap, imap) if args.aux_weight > 0 else None
    if clicks_full is not None:
        log(f"auxiliary click task: {len(clicks_inner[0]):,} / {len(clicks_full[0]):,} past clicks (inner / full)")

    log("choosing epochs on the inner holdout")
    _, curve = train(inner_fit, n_users, n_items, args, args.max_epochs, svd_inner, holdout=inner_hold,
                     clicks=clicks_inner)
    epochs = int(np.argmax(curve)) + 1
    log(f"best epoch {epochs} (holdout AUC {max(curve):.5f}); refitting on all fit events")

    model, _ = train(fit_all, n_users, n_items, args, epochs, svd_full, clicks=clicks_full)
    p = predict(model, val)
    preds = pl.DataFrame({"rowid": val["rowid"].reshape(-1), "predicted_probability": p})
    val_auc = score(split.val_candidates, preds)
    log(f"VALIDATION local AUC: {val_auc:.6f}   [starter 0.6039 | SVD alone 0.7701 | SAFE_v2 leak-free 0.7919]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.write_parquet(OUT_DIR / f"choice_mf_val_c{args.cutoff_day}.parquet")
    result = {"args": vars(args), "holdout_curve": curve, "epochs": epochs, "val_auc": val_auc,
              "minutes": round((time.perf_counter() - t0) / 60, 1)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
