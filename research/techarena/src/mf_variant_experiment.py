"""DM24-first evaluation of logistic-MF variants on SAFE_LOGMF10 features."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .logistic_mf_final import fit_model
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, target_for, train_frames


BASE_FEATURE = "logmf_cosine"
RNG_SEED = 20261025


def score_events(data, events, model):
    U, V = model["U"], model["V"]
    cu, bi = model["cu"], model["bi"]
    cos = np.empty(events.n_events * 10, np.float32)
    dot = np.empty(events.n_events * 10, np.float32)
    for e in range(events.n_events):
        u = U[int(events.user_id[e])]
        c = events.candidates[e].astype(np.int64, copy=False)
        v = V[c]
        d = v @ u + bi[c] + cu[int(events.user_id[e])]
        co = (v @ u) / ((np.linalg.norm(v, axis=1) + 1e-8) * (np.linalg.norm(u) + 1e-8))
        dot[e * 10 : (e + 1) * 10] = d
        cos[e * 10 : (e + 1) * 10] = co
    return cos, dot


def score_cache(data, boundary, label, rank, lam, recent_days=None):
    suffix = f"r{rank}_l{int(lam)}" + (f"_recent{recent_days}d" if recent_days else "")
    path = OUTPUT_DIR / f"mf_{label}_{int(boundary)}_{suffix}.npz"
    if path.exists():
        z = np.load(path)
        return z["cosine"], z["dot"], {"cached": True, "fit_sec": 0.0}
    t0 = time.time()
    model = fit_model(data, int(boundary), float(lam), alternations=3, rank=rank, recent_days=recent_days)
    target = target_for(data, int(boundary))
    cos, dot = score_events(data, target, model)
    np.savez_compressed(path, cosine=cos, dot=dot)
    return cos, dot, {"cached": False, "fit_sec": time.time() - t0, "fit_rows": model["log_rows"]}


def add_variant(frame, cosine, dot, prefix):
    if len(frame) != len(cosine) or len(frame) != len(dot):
        raise ValueError(f"length mismatch for {prefix}")
    out = frame.reset_index(drop=True).copy()
    c = np.asarray(cosine, np.float32).reshape(-1, 10)
    d = np.asarray(dot, np.float32).reshape(-1, 10)
    rank = np.argsort(np.argsort(c, axis=1), axis=1).astype(np.float32).reshape(-1)
    gap = (c - c.max(axis=1, keepdims=True)).astype(np.float32).reshape(-1)
    out[f"{prefix}_cosine"] = c.reshape(-1)
    out[f"{prefix}_dot"] = d.reshape(-1)
    out[f"{prefix}_rank"] = rank
    out[f"{prefix}_gap"] = gap
    return out


def add_base(frame, cosine):
    if len(frame) != len(cosine):
        raise ValueError("length mismatch for base logistic-MF feature")
    out = frame.reset_index(drop=True).copy()
    out[BASE_FEATURE] = np.asarray(cosine, np.float32)
    return out


def sample_events(frame, rate=0.35):
    keys = frame["window_id"].to_numpy(np.int64) * 10_000_000 + frame["event_index"].to_numpy(np.int64)
    grouped = frame.assign(_key=keys).groupby("_key", sort=True)["label"].max()
    rng = np.random.default_rng(RNG_SEED)
    keep = grouped.to_numpy() == 1
    neg = ~keep
    keep[neg] = rng.random(int(neg.sum())) < rate
    selected = grouped.index.to_numpy()[keep]
    return frame.loc[np.isin(keys, selected)].reset_index(drop=True)


def fit_compare(train, val, evalf, tag):
    sample = sample_events(train)
    cols = [c for c in train.columns if c not in {"label", "event_index", "candidate_index", "user_id", "timestamp", "window_id", "sampling_weight"}]
    X = sample[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    XV = val[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = sample["label"].astype(int)
    weights = importance_weights(sample, evalf)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=250, learning_rate=0.05,
        num_leaves=63, max_bin=127, min_child_samples=300,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_lambda=10.0, random_state=20261025, n_jobs=8, verbosity=-1,
    )
    model.fit(X, y, sample_weight=weights, eval_set=[(XV, val["label"].astype(int))],
              eval_metric="auc", callbacks=[lgb.early_stopping(60, verbose=False)])
    pred = model.predict_proba(XV, num_iteration=model.best_iteration_)[:, 1]
    dm = weighted_auc(val, pred, evalf)
    result = {
        "tag": tag,
        "dm24": float(dm["joint_auc_supported"]),
        "history_dm24": float(dm["history_only_auc"]),
        "raw_auc": float(roc_auc_score(val["label"], pred)),
        "train_rows": int(len(sample)),
        "train_events": int(sample["event_index"].nunique()),
        "best_iteration": int(model.best_iteration_),
    }
    np.save(OUTPUT_DIR / f"mf_variant_{tag}_DM24.npy", pred)
    return result


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    B = D0 + 4 * DAY
    train_bounds = bounds_for(4, half=True)
    train, val, _ = train_frames(data, B, train_bounds, "a3_c1_c2", choice_maxiter=20)
    evalf = cached_frame(data, CUT, "eval")
    train_parts = []
    offset = 0
    meta = {}
    for wi, b in enumerate(train_bounds):
        n = int((train["window_id"].to_numpy() == wi).sum())
        # The production artifact's lambda=10 feature is the base feature.
        base_path = OUTPUT_DIR / f"logmf_cosine_prod{wi}_{int(b)}.npy"
        if not base_path.exists():
            raise FileNotFoundError(base_path)
        part = train.iloc[offset : offset + n].copy()
        part = add_base(part, np.load(base_path))
        train_parts.append(part)
        offset += n
    train = pd.concat(train_parts, ignore_index=True)
    val = add_base(val, np.load(OUTPUT_DIR / f"logmf_cosine_val_dm24_{int(B)}.npy"))

    baseline = fit_compare(train, val, evalf, "base_lmf10")
    results = {"boundary": int(B), "baseline": baseline, "variants": {}, "fit_meta": meta}
    variants = {
        "lmf100_r11": (11, 100.0, None),
        "lmf10_r16": (16, 10.0, None),
        "lmf10_recent3": (11, 10.0, 3),
    }
    for name, (rank, lam, recent) in variants.items():
        t0 = time.time()
        tr_parts = []
        offset = 0
        for wi, b in enumerate(train_bounds):
            n = int((train["window_id"].to_numpy() == wi).sum())
            cos, dot, m = score_cache(data, b, f"train{wi}_{name}", rank, lam, recent)
            part = train.iloc[offset : offset + n].copy()
            part = add_variant(part, cos, dot, name)
            tr_parts.append(part)
            offset += n
            meta[f"{name}_train_{int(b)}"] = m
        tr = pd.concat(tr_parts, ignore_index=True)
        cos, dot, m = score_cache(data, B, f"val_{name}", rank, lam, recent)
        va = add_variant(val, cos, dot, name)
        meta[f"{name}_val"] = m
        result = fit_compare(tr, va, evalf, name)
        result["gain_vs_base"] = float(result["dm24"] - baseline["dm24"])
        result["elapsed_sec"] = time.time() - t0
        results["variants"][name] = result
        print(result, flush=True)
    (OUTPUT_DIR / "mf_variant_dm24_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
