"""Resume the TechArena handover from the cached feature frames.

This script deliberately uses PyArrow/Numpy/LightGBM directly so it can run in
the project's Python 3.12 runtime even when pandas is unavailable.  It rebuilds
the SAFE_v2 feature path for the three DM folds, computes the event/within-event
decomposition, and evaluates a leakage-safe two-stage event/choice model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
D0 = 1_737_331_200
DAY = 86_400
CUT = 1_737_936_000

BASE_DROP = {"event_index", "candidate_index", "user_id", "timestamp", "label"}
CHOICE_NAMES = [
    "svd_dot", "svd_cos", "svd32_dot", "svd32_cos", "history_mean_cos",
    "history32_mean_cos", "item_ctr", "history_count", "history_age_days", "gap_hours",
]
C2_NAMES = ["shrink_dot", "shrink_cos", "shrink_norm", "posterior_std", "posterior_utility_std"]


def read_parquet(path: Path) -> dict[str, np.ndarray]:
    tab = pq.read_table(path)
    out: dict[str, np.ndarray] = {}
    for name in tab.column_names:
        arr = tab[name].to_numpy(zero_copy_only=False)
        if arr.dtype.kind in "iu":
            out[name] = arr.astype(np.int64, copy=False)
        else:
            out[name] = arr.astype(np.float32, copy=False)
    return out


def concat_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = frames[0].keys()
    return {n: np.concatenate([f[n] for f in frames]) for n in names}


def choice_fit(frame: dict[str, np.ndarray], maxiter: int = 80) -> dict[str, np.ndarray | float]:
    x = np.stack([frame[n].astype(np.float64) for n in CHOICE_NAMES], axis=1)
    n_events = len(x) // 10
    x = x.reshape(n_events, 10, -1)
    y = frame["label"].reshape(n_events, 10)
    cls = np.zeros(n_events, dtype=np.int64)
    for e in range(n_events):
        hit = np.flatnonzero(y[e] == 1)
        if len(hit):
            cls[e] = int(hit[0] + 1)
    mean = x.reshape(-1, x.shape[-1]).mean(0)
    scale = x.reshape(-1, x.shape[-1]).std(0)
    scale[scale < 1e-6] = 1.0
    x = (x - mean) / scale
    hist = np.minimum(frame["history_length"].reshape(n_events, 10)[:, 0], 20.0) / 20.0
    nf = x.shape[-1]

    def fun(p, grad=True):
        w, bias, hw = p[:nf], p[nf], p[nf + 1]
        u = np.einsum("ecf,f->ec", x, w)
        logits = np.concatenate([bias + hw * hist[:, None], u], axis=1)
        logits -= logits.max(1, keepdims=True)
        ex = np.exp(logits)
        prob = ex / ex.sum(1, keepdims=True)
        loss = -np.log(np.maximum(prob[np.arange(n_events), cls], 1e-12)).mean() + 0.005 * np.dot(w, w)
        if not grad:
            return float(loss)
        r = prob.copy(); r[np.arange(n_events), cls] -= 1.0
        gw = np.einsum("ec,ecf->f", r[:, 1:], x) / n_events + .01 * w
        return float(loss), np.r_[gw, r[:, 0].mean(), np.mean(r[:, 0] * hist)]

    p0 = np.zeros(nf + 2); p0[nf] = math.log(.37 / .63)
    res = minimize(lambda p: fun(p, True), p0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-8})
    return {"mean": mean, "scale": scale, "weights": res.x[:nf], "bias": float(res.x[nf]), "hist_weight": float(res.x[nf + 1])}


def add_choice(frame: dict[str, np.ndarray], model: dict) -> dict[str, np.ndarray]:
    x = np.stack([frame[n].astype(np.float64) for n in CHOICE_NAMES], axis=1)
    x = (x - model["mean"]) / model["scale"]
    u = (x @ model["weights"]).reshape(-1, 10)
    h = np.minimum(frame["history_length"].reshape(-1, 10)[:, 0], 20.0) / 20.0
    logits = np.concatenate([(model["bias"] + model["hist_weight"] * h)[:, None], u], axis=1)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    out = dict(frame)
    out["choice_utility"] = u.reshape(-1).astype(np.float32)
    out["choice_probability"] = p[:, 1:].reshape(-1).astype(np.float32)
    out["choice_no_click"] = np.repeat(p[:, 0], 10).astype(np.float32)
    return out


def add_c2(frame: dict[str, np.ndarray], path: Path) -> dict[str, np.ndarray]:
    extra = read_parquet(path)
    out = dict(frame)
    for n in C2_NAMES:
        out[n] = extra[n].astype(np.float32, copy=False)
    return out


def feature_names(frame: dict[str, np.ndarray]) -> list[str]:
    return [n for n in frame if n not in BASE_DROP]


def matrix(frame: dict[str, np.ndarray], cols: list[str]) -> np.ndarray:
    x = np.column_stack([np.nan_to_num(frame[n].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) for n in cols])
    return x


def fit_binary(train: dict[str, np.ndarray], val: dict[str, np.ndarray], trees: int = 650, lr: float = .03, seed: int = 20261024):
    cols = feature_names(train)
    model = lgb.LGBMClassifier(objective="binary", n_estimators=trees, learning_rate=lr, num_leaves=63,
                                min_child_samples=300, colsample_bytree=.7, subsample=.8,
                                subsample_freq=1, reg_lambda=10.0, random_state=seed, n_jobs=4, verbosity=-1)
    model.fit(matrix(train, cols), train["label"].astype(np.int8))
    return model, cols, model.predict_proba(matrix(val, cols))[:, 1]


def group_events(frame: dict[str, np.ndarray], pred: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n = len(frame["label"]) // 10
    out = {"history_length": frame["history_length"].reshape(n, 10)[:, 0],
           "gap_hours": frame["gap_hours"].reshape(n, 10)[:, 0],
           "label": (frame["label"].reshape(n, 10).sum(1) > 0).astype(np.int8)}
    for name in ["svd_cos", "history_mean_cos", "choice_utility", "item_ctr", "transition_max", "event_score_max", "event_score_mean", "user_ctr"]:
        if name in frame:
            a = frame[name].reshape(n, 10)
            out[name + "_max"] = a.max(1); out[name + "_mean"] = a.mean(1); out[name + "_std"] = a.std(1)
            out[name + "_top2gap"] = np.sort(a, axis=1)[:, -1] - np.sort(a, axis=1)[:, -2]
    out["in_history"] = (frame["history_count"].reshape(n, 10) > 0).sum(1)
    out["history_length"] = out["history_length"].astype(np.float32)
    if pred is not None:
        out["pred_sum"] = pred.reshape(n, 10).sum(1)
        out["pred_max"] = pred.reshape(n, 10).max(1)
    return out


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None


def bucket_metrics(frame: dict[str, np.ndarray], pred: np.ndarray) -> dict:
    h = frame["history_length"]
    g = frame["gap_hours"]
    out = {"history": {}, "gap": {}}
    for name, vals, edges, labels in [("history", h, [-1, 0, 9, 19, 20], ["0", "1-9", "10-19", "20"]),
                                      ("gap", g, [-np.inf, 6, 24, 72, 168, np.inf], ["<6h", "6-24h", "1-3d", "3-7d", ">7d"])]:
        for i, lab in enumerate(labels):
            m = (vals > edges[i]) & (vals <= edges[i + 1])
            out[name][lab] = {"rows": int(m.sum()), "auc": safe_auc(frame["label"][m], pred[m])}
    return out


def weighted_dm(frame: dict[str, np.ndarray], pred: np.ndarray, eval_frame: dict[str, np.ndarray]) -> float:
    """Global ROC AUC after matching evaluation history/gap cell mass."""
    def key(f):
        h = f["history_length"]; g = f["gap_hours"]
        hs = np.where(h <= 0, "0", np.where(h <= 9, "1-9", np.where(h <= 19, "10-19", "20")))
        gs = np.where(g < 6, "<6h", np.where(g < 24, "6-24h", np.where(g < 72, "1-3d", np.where(g < 168, "3-7d", ">7d"))))
        return np.char.add(np.char.add(hs, "|"), gs)
    vk, ek = key(frame), key(eval_frame)
    uniq, counts = np.unique(ek, return_counts=True); es = {u: c / len(ek) for u, c in zip(uniq, counts)}
    vkuniq, vcounts = np.unique(vk, return_counts=True); vs = {u: c / len(vk) for u, c in zip(vkuniq, vcounts)}
    w = np.asarray([es.get(k, 0.0) / max(vs.get(k, 1e-12), 1e-12) for k in vk])
    return float(roc_auc_score(frame["label"], pred, sample_weight=w))


def decomposition(frame: dict[str, np.ndarray], pred: np.ndarray) -> dict:
    y = frame["label"].astype(np.int8); ev = group_events(frame, pred)
    ey, es, em = ev["label"], ev["pred_sum"], ev["pred_max"]
    click_ev = y.reshape(-1, 10).sum(1) > 0
    yy = y.reshape(-1, 10); pp = pred.reshape(-1, 10)
    top1 = []; mrr = []; ea = []
    for i in np.flatnonzero(click_ev):
        order = np.argsort(-pp[i]); hit = int(np.flatnonzero(yy[i, order] == 1)[0]) + 1
        top1.append(hit == 1); mrr.append(1.0 / hit); ea.append(safe_auc(yy[i], pp[i]))
    # Perfect event propensity: retain within-event ordering and remove all
    # rows from no-click events.
    perfect_event = pred.copy(); perfect_event.reshape(-1, 10)[~click_ev, :] = 0.0
    # Perfect within-event ordering while retaining the event propensity.  Use
    # the percentile rank of the event score as the coarse component and a
    # small within-event tie breaker.  This keeps event ordering dominant while
    # giving the clicked row the best possible position inside its event.
    order = np.argsort(np.argsort(es, kind="stable"), kind="stable")
    event_percentile = order / max(len(es) - 1, 1)
    within = np.tile(np.arange(10, 0, -1, dtype=np.float64), len(es))
    perfect_within = np.repeat(event_percentile, 10) + (within + yy.reshape(-1) * 10.0) * 1e-7
    return {"global_auc": safe_auc(y, pred), "event_auc_sum": safe_auc(ey, es), "event_auc_max": safe_auc(ey, em),
            "event_click_rate": float(ey.mean()), "top1_hit_rate": float(np.mean(top1)), "mrr": float(np.mean(mrr)),
            "mean_per_event_auc": float(np.nanmean([x for x in ea if x is not None])),
            "oracle_perfect_event_auc": safe_auc(y, perfect_event),
            "oracle_perfect_within_auc": safe_auc(y, perfect_within),
            "history_bucket": bucket_metrics(frame, pred),
            "event_history_bucket": {}}


def two_stage(train: dict[str, np.ndarray], val: dict[str, np.ndarray], baseline: np.ndarray) -> tuple[np.ndarray, dict]:
    tr_e = group_events(train); va_e = group_events(val)
    ecols = [n for n in tr_e if n not in {"label"}]
    emod = lgb.LGBMClassifier(objective="binary", n_estimators=350, learning_rate=.04, num_leaves=31,
                              min_child_samples=80, reg_lambda=10.0, random_state=20261024, n_jobs=4, verbosity=-1)
    emod.fit(np.column_stack([tr_e[c] for c in ecols]), tr_e["label"])
    pe = emod.predict_proba(np.column_stack([va_e[c] for c in ecols]))[:, 1]
    cols = feature_names(train)
    clicked = train["label"].reshape(-1, 10).sum(1) > 0
    row_mask = np.repeat(clicked, 10)
    wmod = lgb.LGBMClassifier(objective="binary", n_estimators=450, learning_rate=.04, num_leaves=63,
                              min_child_samples=300, colsample_bytree=.8, reg_lambda=10.0,
                              random_state=20261025, n_jobs=4, verbosity=-1)
    wmod.fit(matrix(train, cols)[row_mask], train["label"][row_mask].astype(np.int8))
    wl = wmod.predict_proba(matrix(val, cols))[:, 1].reshape(-1, 10)
    wl -= wl.max(1, keepdims=True); pw = np.exp(wl); pw /= pw.sum(1, keepdims=True)
    p = (pe[:, None] * pw).reshape(-1)
    return p, {"event_auc": safe_auc(va_e["label"], pe), "within_choice_auc": safe_auc(val["label"], pw.reshape(-1)),
               "event_features": ecols, "row_features": cols}


def main():
    # Use all available half-day cached windows and the matching C2 extras.
    folds = {}
    for name, day in [("DM22", 2), ("DM23", 3), ("DM24", 4)]:
        boundary = D0 + day * DAY
        bounds = [D0 + int(i * DAY / 2) for i in range(1, 2 * day)]
        trains = []
        for b in bounds:
            f = add_c2(read_parquet(CACHE / "features_orig" / f"{b}.parquet"), CACHE / "features_c" / f"c2_{b}.parquet")
            trains.append(f)
        train = concat_frames(trains)
        val = add_c2(read_parquet(CACHE / "features_orig" / f"{boundary}.parquet"), CACHE / "features_c" / f"c2_{boundary}.parquet")
        cm = choice_fit(train)
        train = add_choice(train, cm); val = add_choice(val, cm)
        model, cols, pred = fit_binary(train, val)
        eval_frame = read_parquet(CACHE / "features_orig" / "evaluation.parquet")
        dec = decomposition(val, pred)
        dec["DM_reweighted"] = weighted_dm(val, pred, eval_frame)
        tsp, tsmeta = two_stage(train, val, pred)
        dec_ts = decomposition(val, tsp)
        dec_ts["DM_reweighted"] = weighted_dm(val, tsp, eval_frame)
        # A global logit blend is tested on the same validation fold.  The
        # weight is reported per fold and the final recommendation is the
        # mean-performing common weight across the three folds.
        eps = 1e-6
        def logit(a):
            return np.log(np.clip(a, eps, 1 - eps) / np.clip(1 - a, eps, 1 - eps))
        best = {"weight": 0.0, "dm": -1.0, "auc": -1.0}
        for w in np.linspace(0, 1, 21):
            bp = 1 / (1 + np.exp(-(w * logit(pred) + (1 - w) * logit(tsp))))
            dm = weighted_dm(val, bp, eval_frame)
            if dm > best["dm"]: best = {"weight": float(w), "dm": dm, "auc": safe_auc(val["label"], bp)}
        dec_ts["logit_blend_best"] = best
        folds[name] = {"rows": int(len(val["label"])), "events": int(len(val["label"]) // 10),
                       "baseline": dec, "two_stage": {"metrics": tsmeta, "decomposition": dec_ts}}
        print(name, "baseline", dec["global_auc"], "event", dec["event_auc_sum"], "two-stage", dec_ts["global_auc"], flush=True)
    # Tune a single global blend weight on the diagnostic DM mean.  Predictions
    # are retained only in memory; no file is eligible without the prescribed gate.
    out = {"folds": folds, "mode": "full diagnostic reconstruction", "trees": 650,
           "safe_v2_reference": {"DM22": .7859872505557265, "DM23": .7959620121431473, "DM24": .8055218787124601,
                                  "DM_mean3": .795823879742},
           "note": "Baseline is a cached SAFE_v2 feature-path reconstruction; portal scores are not inferred."}
    (OUT / "handover_results.json").write_text(json.dumps(out, indent=2, default=float))
    write_reports(out)


def write_reports(out: dict):
    lines = ["# SAFE_v2 decomposition diagnosis", "", "This is a local cached-feature reconstruction using the same DM first-event folds. It does not claim a portal score.", "", "| Fold | baseline AUC | event AUC (sum) | event AUC (max) | top-1 | MRR | mean per-event AUC | oracle event | oracle within |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ["DM22", "DM23", "DM24"]:
        d = out["folds"][k]["baseline"]
        lines.append(f"| {k} | {d['global_auc']:.6f} | {d['event_auc_sum']:.6f} | {d['event_auc_max']:.6f} | {d['top1_hit_rate']:.4f} | {d['mrr']:.4f} | {d['mean_per_event_auc']:.6f} | {d['oracle_perfect_event_auc']:.6f} | {d['oracle_perfect_within_auc']:.6f} |")
    lines += ["", "The oracle columns quantify the two ceilings: perfect event propensity while preserving row ordering, and perfect within-event ordering while preserving event propensity. The larger gap identifies the bottleneck.", "", "## Two-stage evaluation", "", "| Fold | baseline AUC | baseline DM | two-stage AUC | two-stage DM | event AUC | W row AUC | best blend weight/DM |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ["DM22", "DM23", "DM24"]:
        b = out["folds"][k]["baseline"]; t = out["folds"][k]["two_stage"]
        blend = t['decomposition']['logit_blend_best']
        lines.append(f"| {k} | {b['global_auc']:.6f} | {b['DM_reweighted']:.6f} | {t['decomposition']['global_auc']:.6f} | {t['decomposition']['DM_reweighted']:.6f} | {t['metrics']['event_auc']:.6f} | {t['metrics']['within_choice_auc']:.6f} | {blend['weight']:.2f}/{blend['dm']:.6f} |")
    lines += ["", "The two-stage model is diagnostic only. It becomes a candidate submission only if its reweighted DM_mean3 clears the handover gate; this run writes no submission file until that condition is met."]
    (OUT / "decomposition_report.md").write_text("\n".join(lines) + "\n")
    (OUT / "two_stage_report.md").write_text("\n".join(lines[lines.index("## Two-stage evaluation"):]) + "\n")


if __name__ == "__main__":
    main()
