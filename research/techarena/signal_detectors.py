"""Detector tests for four new signal families, on the inner holdout only (validation is never scored).

    python research/techarena/signal_detectors.py            # ~45 min on a free CPU
    python research/techarena/signal_detectors.py --quick    # smoke test: 10% of users, small models

Setup = the stage 3 inner holdout: the stage 3 features (step 4 + Optuna + 1-seed sequence features) on the 7 training
windows of the validation run; fit on windows 0-5 (targets before day 3.5), score window 6 (each user's first event in
[day 3.5, day 4)). Every new feature for a window uses only data from before that window's boundary, or the target
event's own slate (known when the event happens).

  1. second stage: a small LightGBM on the baseline's out-of-sample window-6 scores + slate context (2-fold by user)
  2. slate as a hint: similarity of a candidate to the other 9 in its slate and to the user's past impressions
  3. item trends: item CTR in the last 6 / 24 h vs all time, exposure share, item age, hours since last click
  4. item-to-item transitions: recency-weighted and lift-normalised P(candidate | last 5 clicks), two-step transitions
The stack already has cruder versions of 2-4 (impression_svd_cos, shown_nonclick_before, item_click_1d/3d/7d,
item_trend, transition_last/mean/max). Arms 2-4 and all three together are compared with the baseline at the same
LightGBM settings (learning rate 0.05, early stopping on window 6, as in the Optuna search); a gain under about +0.0005
is noise.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

QUICK = "--quick" in sys.argv
sys.argv = sys.argv[:1]  # safe_v2_stage3 parses the command line when imported
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root)
from safe_v2_stage3 import build, with_seq  # noqa: E402  (also loads the Optuna parameters into lf.LGB_PARAMS)
from src.validation import make_split  # noqa: E402

DATA = Path("data/parquet")
OUT = HERE / "outputs" / ("signal_detectors_quick.json" if QUICK else "signal_detectors.json")
HOUR = 3600
RANK = 8 if QUICK else 32
OPT2 = ["slate_cos_click", "slate_spread_click", "slate_cos_imp", "slate_spread_imp", "user_imp_cos"]
OPT3 = ["item_ctr6", "item_ctr24", "item_trend6", "item_trend24", "item_expo24", "item_age_h", "item_since_click_h"]
OPT4 = ["trans_w", "trans_lift_w", "trans2"]
RECENCY = np.array([1.0, 0.7, 0.5, 0.35, 0.25])  # weights of the last 5 clicks, most recent first


# ---------------------------------------------------------------- data
def load_logs(with_eval: bool = False):
    """Integer-coded candidates (rowid, event_id, u, i, ts, clicked) and the click log (u, i, ts): event clicks plus
    every history click of train and evaluation events, as in Monish's click_log_before. with_eval also returns the
    evaluation candidates (rowid, event_id, u, i) in the same codes."""
    te = pl.read_parquet(DATA / "train_events.parquet")
    tc = pl.read_parquet(DATA / "train_candidates.parquet")
    ee = pl.read_parquet(DATA / "eval_events.parquet")
    items = pl.concat([tc["item_id"], te["history_item_ids"].explode(), ee["history_item_ids"].explode()]).drop_nulls().unique()
    users = pl.concat([te["user_id"], ee["user_id"]]).unique()
    imap = pl.DataFrame({"item_id": items, "i": np.arange(len(items), dtype=np.int64)})
    umap = pl.DataFrame({"user_id": users, "u": np.arange(len(users), dtype=np.int64)})
    events = te.select("event_id", "user_id", "timestamp").join(umap, on="user_id")
    cands = (tc.join(events, on="event_id").join(imap, on="item_id")
             .select("rowid", "event_id", "u", "i", pl.col("timestamp").alias("ts"), "clicked"))
    hist = (pl.concat([te.select("user_id", "history_item_ids", "history_timestamps"),
                       ee.select("user_id", "history_item_ids", "history_timestamps")])
            .explode(["history_item_ids", "history_timestamps"]).drop_nulls()
            .join(umap, on="user_id").join(imap, left_on="history_item_ids", right_on="item_id")
            .select("u", "i", pl.col("history_timestamps").alias("ts")))
    clicks = pl.concat([cands.filter(pl.col("clicked") == 1).select("u", "i", "ts"), hist]).unique()
    if not with_eval:
        return cands, clicks, len(users), len(items)
    eval_keys = (pl.read_parquet(DATA / "eval_candidates.parquet")
                 .join(ee.select("event_id", "user_id").join(umap, on="user_id"), on="event_id")
                 .join(imap, on="item_id").select("rowid", "event_id", "u", "i"))
    assert eval_keys.height == 10 * ee.height, "evaluation candidates did not all map to known users and items"
    return cands, clicks, len(users), len(items), eval_keys


def svd_items(u: np.ndarray, i: np.ndarray, n_users: int, n_items: int):
    """Unit item vectors from the degree-normalised binary user x item matrix (as src.choice_mf.click_svd), and the
    binary matrix itself."""
    x = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(n_users, n_items))
    x.sum_duplicates()
    x.data[:] = 1.0
    du, di = np.asarray(x.sum(1)).ravel() + 1.0, np.asarray(x.sum(0)).ravel() + 1.0
    xs = sp.diags(1 / np.sqrt(du)) @ x @ sp.diags(1 / np.sqrt(di))
    _, _, vt = svds(xs.astype(np.float64), k=RANK, random_state=20261024)
    v = vt.T
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8), x


def pairs(m: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(m[rows, cols]).ravel() if len(rows) else np.zeros(0)


# ---------------------------------------------------------------- new features for one window
def window_features(target: pl.DataFrame, b: int, cands: pl.DataFrame, clicks: pl.DataFrame, n_users: int,
                    n_items: int) -> pd.DataFrame:
    """target: rowid, event_id, u, i for all 10 candidates of each target event; everything else only before b.
    (The same event can be a target in several windows, with different features, so rows are keyed by window too.)"""
    target = target.sort("event_id", "i")
    n_ev = target.height // 10
    ev_u = target["u"].to_numpy()[::10]
    cand = target["i"].to_numpy().reshape(n_ev, 10)
    flat = cand.ravel()
    past = cands.filter(pl.col("ts") < b)
    pc = clicks.filter(pl.col("ts") < b)
    f = {}

    # 2. slate as a hint
    v_click, _ = svd_items(pc["u"].to_numpy(), pc["i"].to_numpy(), n_users, n_items)
    v_imp, x_imp = svd_items(past["u"].to_numpy(), past["i"].to_numpy(), n_users, n_items)
    for name, v in (("click", v_click), ("imp", v_imp)):
        vc = v[cand]
        tot = vc.sum(1, keepdims=True)
        others = (tot - vc) / 9
        f[f"slate_cos_{name}"] = ((vc * others).sum(-1) / (np.linalg.norm(others, axis=-1) + 1e-8)).ravel()
        spread = ((tot[:, 0] ** 2).sum(-1) - (vc ** 2).sum((1, 2))) / 90  # mean pairwise cosine in the slate
        f[f"slate_spread_{name}"] = np.repeat(spread, 10)
    centre = x_imp @ v_imp  # sum of the vectors of every item the user was shown
    centre /= np.linalg.norm(centre, axis=1, keepdims=True) + 1e-8
    f["user_imp_cos"] = (v_imp[cand] * centre[ev_u][:, None, :]).sum(-1).ravel()

    # 3. item trends
    pi, pts, pck = past["i"].to_numpy(), past["ts"].to_numpy(), past["clicked"].to_numpy().astype(float)
    g = pck.mean()

    def ctr(mask: np.ndarray, a: float = 20.0):
        imp = np.bincount(pi[mask], minlength=n_items)
        return (np.bincount(pi[mask], weights=pck[mask], minlength=n_items) + a * g) / (imp + a), imp

    ctr_all, imp_all = ctr(np.ones(len(pi), bool))
    ctr6, _ = ctr(pts >= b - 6 * HOUR)
    ctr24, imp24 = ctr(pts >= b - 24 * HOUR)
    first = np.full(n_items, np.iinfo(np.int64).max)
    np.minimum.at(first, pi, pts)
    last_click = np.full(n_items, -1, np.int64)
    np.maximum.at(last_click, pi[pck > 0], pts[pck > 0])
    item = {"item_ctr6": ctr6, "item_ctr24": ctr24, "item_trend6": ctr6 - ctr_all, "item_trend24": ctr24 - ctr_all,
            "item_expo24": imp24 / (imp_all + 1.0),
            "item_age_h": np.where(first < b, (b - first) / HOUR, np.nan),
            "item_since_click_h": np.where(last_click >= 0, (b - last_click) / HOUR, np.nan)}
    for k, a in item.items():
        f[k] = a[flat]

    # 4. transitions between consecutive clicks of a user
    pcs = pc.sort("u", "ts")
    cu, ci = pcs["u"].to_numpy(), pcs["i"].to_numpy()
    same = cu[1:] == cu[:-1]
    t = sp.csr_matrix((np.ones(same.sum()), (ci[:-1][same], ci[1:][same])), shape=(n_items, n_items))
    t.sum_duplicates()
    p = (sp.diags(1 / (np.asarray(t.sum(1)).ravel() + 5.0)) @ t).tocsr()  # P(next | prev), shrunk for rare items
    pop = np.bincount(ci, minlength=n_items) / len(ci)
    lastk = pcs.group_by("u", maintain_order=True).agg(pl.col("i").tail(5).reverse().alias("last"))
    m = pl.DataFrame({"u": ev_u}).with_row_index("e").join(lastk, on="u", how="left").sort("e")
    last = np.full((n_ev, 5), -1, np.int64)
    for e, lst in enumerate(m["last"].to_list()):
        if lst:
            last[e, :len(lst)] = lst
    tw, lw, wsum = np.zeros(len(flat)), np.zeros(len(flat)), np.zeros(len(flat))
    for k in range(5):
        h = np.repeat(last[:, k], 10)
        ok = h >= 0
        val = np.zeros(len(flat))
        val[ok] = pairs(p, h[ok], flat[ok])
        tw += RECENCY[k] * val
        lw += RECENCY[k] * np.log1p(val / (pop[flat] + 1e-6))
        wsum += RECENCY[k] * ok
    f["trans_w"] = np.where(wsum > 0, tw / np.maximum(wsum, 1e-9), np.nan)
    f["trans_lift_w"] = np.where(wsum > 0, lw / np.maximum(wsum, 1e-9), np.nan)
    ok0 = last[:, 0] >= 0
    two = np.full(len(flat), np.nan)
    if ok0.any():
        mp = (p[last[ok0, 0]] @ p).tocsr()  # two steps from the last click
        rows = np.repeat(np.arange(ok0.sum()), 10)
        two[np.repeat(ok0, 10)] = pairs(mp, rows, cand[ok0].ravel())
    f["trans2"] = two
    return pd.DataFrame({"rowid": target["rowid"].to_numpy(), "event_id": target["event_id"].to_numpy(), **f})


# ---------------------------------------------------------------- scoring helpers
def decompose(ev: np.ndarray, y: np.ndarray, s: np.ndarray) -> dict:
    """Which item (pairwise share in clicked events) and whether any click (event AUC of the summed score)."""
    o = np.argsort(ev, kind="stable")
    Y, S = y[o].reshape(-1, 10), s[o].reshape(-1, 10)
    c = Y.max(1) == 1
    cs = (S[c] * Y[c]).sum(1)
    return {"global": float(roc_auc_score(y, s)), "which": float(((S[c] < cs[:, None]).sum(1) / 9).mean()),
            "whether": float(roc_auc_score(Y.max(1), S.sum(1)))}


def within_auc(ev: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    o = np.argsort(ev, kind="stable")
    Y, X = y[o].reshape(-1, 10), x[o].reshape(-1, 10)
    c = Y.max(1) == 1
    cx = (X[c] * Y[c]).sum(1)
    return float((((X[c] < cx[:, None]).sum(1) + 0.5 * ((X[c] == cx[:, None]).sum(1) - 1)) / 9).mean())


def stage2(iva: pd.DataFrame, p: np.ndarray) -> dict:
    """Option 1: re-score window 6 from the baseline's out-of-sample scores + slate context, 2-fold by user. Boosted
    from the baseline's own logit (init_score), so it only learns corrections to it."""
    d = pd.DataFrame({"event_id": iva["event_id"].to_numpy(), "user_id": iva["user_id"].to_numpy(),
                      "label": iva["label"].to_numpy(), "p": p})
    for c in ("cmf_noclick", "choice_no_click", "seq_noclick", "user_ctr", "history_length", "gap_hours"):
        if c in iva:
            d[c] = iva[c].to_numpy()
    d["s"] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    g = d.groupby("event_id")
    d["ev_sum_p"], d["ev_max_p"] = g["p"].transform("sum"), g["p"].transform("max")
    d["rank"] = g["s"].rank(ascending=False, method="first")
    top2 = g["s"].transform(lambda x: x.nlargest(2).iloc[-1])
    ev_max = g["s"].transform("max")
    d["s_vs_best_other"] = d["s"] - np.where(d["s"] >= ev_max, top2, ev_max)
    d["s_vs_mean_other"] = d["s"] - (g["s"].transform("sum") - d["s"]) / 9
    e = np.exp(d["s"] - ev_max)
    d["share"] = e / e.groupby(d["event_id"]).transform("sum")
    d["ev_gap12"] = ev_max - top2
    feats = [c for c in d.columns if c not in ("event_id", "user_id", "label")]
    uid = d["user_id"].to_numpy()
    fold = np.searchsorted(np.unique(uid), uid) % 2  # alternate users (user_id % 2 fails on --quick's sample)
    out = np.empty(len(d))
    per_fold = []
    for k in (0, 1):
        tr, te = fold != k, fold == k
        m = lgb.LGBMClassifier(n_estimators=100 if QUICK else 300, learning_rate=0.02, num_leaves=7,
                               min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=10.0, verbosity=-1, random_state=0)
        m.fit(d.loc[tr, feats], d.loc[tr, "label"], init_score=d.loc[tr, "s"].to_numpy())
        out[te] = 1 / (1 + np.exp(-(d.loc[te, "s"].to_numpy() + m.predict(d.loc[te, feats], raw_score=True))))
        per_fold.append(float(roc_auc_score(d.loc[te, "label"], out[te]) - roc_auc_score(d.loc[te, "label"], p[te])))
    ev, y = d["event_id"].to_numpy(), d["label"].to_numpy()
    return {"base": decompose(ev, y, p), "stage2": decompose(ev, y, out), "gain_per_fold": per_fold}


# ---------------------------------------------------------------- shared setup (also used by ease_detector.py)
def stage3_frame() -> tuple[pd.DataFrame, list[str], list[int]]:
    """The stage 3 training frame of the validation run (choice features cross-fitted), its feature columns and the
    window boundaries."""
    split = make_split(lf.VAL_DAY)
    b = split.cutoff_ts
    bounds = lf.bounds_for(lf.VAL_DAY, half=True)
    train_all = lf.training_windows(None, bounds)
    fixed = train_all[train_all["timestamp"] < b]
    frames = [fixed[fixed["window_id"] == wi].reset_index(drop=True) for wi in range(len(bounds))]
    train_base, val = build("val", bounds, frames, lf.boundary_frame(None, b), b)
    train_base, val = with_seq(train_base, bounds, val, f"b{b}")
    tr = train_base if QUICK else lf.crossfit_choice(train_base, val)[0]  # quick: skip the choice cross-fit
    cols = lf.model_columns(tr)
    if QUICK:
        tr = tr[tr["user_id"] % 10 == 0]
    tr["rowid"] = tr["rowid"].astype(str)
    lf.log(f"stage 3 features: {len(tr):,} rows, {len(cols)} features")
    return tr, cols, bounds


def attach(tr: pd.DataFrame, bounds: list[int], keys: pl.DataFrame, features) -> pd.DataFrame:
    """Add features(target, boundary) for every window; keys = rowid, event_id, u, i of all train candidates."""
    parts = []
    for wi, bw in enumerate(bounds):
        rid = pl.DataFrame({"rowid": tr.loc[tr["window_id"] == wi, "rowid"].to_numpy()})
        target = keys.join(rid, on="rowid")
        assert target.height == rid.height and target.height % 10 == 0, f"window {wi}: incomplete events"
        parts.append(features(target, bw).assign(window_id=wi))
        lf.log(f"new features for window {wi} ({target.height // 10:,} events)")
    n = len(tr)
    tr = tr.merge(pd.concat(parts, ignore_index=True), on=["rowid", "window_id"], how="left")
    assert len(tr) == n and tr["event_id"].notna().all(), "new features did not line up with the stage 3 rows"
    return tr


def inner_arms(tr: pd.DataFrame, cols: list[str], bounds: list[int]):
    """Window 6 (the inner holdout) and a function fitting LightGBM on windows 0-5 with cols + extra columns."""
    last = len(bounds) - 1
    itr = tr[(tr["window_id"] < last) & (tr["timestamp"] < bounds[last])]
    iva = tr[tr["window_id"] == last].reset_index(drop=True)
    y = iva["label"].to_numpy()
    params = {**lf.LGB_PARAMS, "learning_rate": 0.05}
    lf.log(f"inner holdout: {len(itr):,} fit rows / {len(iva):,} window-6 rows")

    def arm(extra: list[str]) -> tuple[float, int, np.ndarray, dict]:
        use = cols + extra
        m = lgb.LGBMClassifier(n_estimators=150 if QUICK else lf.MAX_TREES, random_state=lf.FULL_SEEDS[0], **params)
        m.fit(itr[use], itr["label"], eval_set=[(iva[use], y)], eval_metric="auc",
              callbacks=[lgb.early_stopping(lf.PATIENCE, verbose=False)])
        p = m.predict_proba(iva[use])[:, 1]
        gain = dict(zip(use, m.booster_.feature_importance(importance_type="gain")))
        order = sorted(use, key=lambda c: -gain[c])
        return float(roc_auc_score(y, p)), int(m.best_iteration_), p, {c: order.index(c) + 1 for c in extra}

    return iva, arm


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = time.perf_counter()
    tr, cols, bounds = stage3_frame()
    cands, clicks, n_users, n_items = load_logs()
    tr = attach(tr, bounds, cands.select("rowid", "event_id", "u", "i"),
                lambda target, bw: window_features(target, bw, cands, clicks, n_users, n_items))
    iva, arm = inner_arms(tr, cols, bounds)
    y, ev = iva["label"].to_numpy(), iva["event_id"].to_numpy()

    res = {"arms": {}}
    base_auc, trees, p_base, _ = arm([])
    res["arms"]["baseline"] = {"auc": base_auc, "trees": trees}
    lf.log(f"baseline: inner AUC {base_auc:.6f} ({trees} trees)")

    resid = y - p_base
    res["single_features"] = {}
    for c in OPT2 + OPT3 + OPT4:
        x = iva[c].to_numpy(float)
        x = np.where(np.isnan(x), np.nanmedian(x), x)
        a, w = roc_auc_score(y, x), within_auc(ev, y, x)
        res["single_features"][c] = {"global_auc": float(max(a, 1 - a)), "within_event_auc": float(max(w, 1 - w)),
                                     "spearman_with_baseline_residual": float(spearmanr(x, resid).statistic)}

    res["option1_stage2"] = stage2(iva, p_base)
    s2 = res["option1_stage2"]
    lf.log(f"option 1 (second stage): {s2['base']['global']:.6f} -> {s2['stage2']['global']:.6f} "
           f"(per fold {s2['gain_per_fold'][0]:+.6f} / {s2['gain_per_fold'][1]:+.6f})")

    for name, extra in (("option2_slate", OPT2), ("option3_trends", OPT3), ("option4_transitions", OPT4),
                        ("options2to4", OPT2 + OPT3 + OPT4)):
        a, t, _, ranks = arm(extra)
        res["arms"][name] = {"auc": a, "delta": a - base_auc, "trees": t, "new_feature_ranks": ranks}
        lf.log(f"{name}: inner AUC {a:.6f} ({a - base_auc:+.6f}, {t} trees) ranks {ranks}")

    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.write_text(json.dumps(res, indent=2))

    print("\n================ SUMMARY (inner holdout = day-3.5 window; validation untouched) ================")
    print(f"{'test':24s} {'inner AUC':>10s} {'change':>10s}")
    print(f"{'baseline (stage 3)':24s} {base_auc:10.6f}")
    print(f"{'1 second stage':24s} {s2['stage2']['global']:10.6f} {s2['stage2']['global'] - s2['base']['global']:+10.6f}"
          f"   which {s2['base']['which']:.4f}->{s2['stage2']['which']:.4f}, "
          f"whether {s2['base']['whether']:.4f}->{s2['stage2']['whether']:.4f}")
    for name in ("option2_slate", "option3_trends", "option4_transitions", "options2to4"):
        r = res["arms"][name]
        print(f"{name:24s} {r['auc']:10.6f} {r['delta']:+10.6f}")
    print("\nsingle features (window 6): global AUC / within-event AUC / spearman with baseline residual")
    for c, r in res["single_features"].items():
        print(f"  {c:20s} {r['global_auc']:.4f} / {r['within_event_auc']:.4f} / {r['spearman_with_baseline_residual']:+.4f}")
    print(f"\n{res['minutes']} min, wrote {OUT}")


if __name__ == "__main__":
    main()
