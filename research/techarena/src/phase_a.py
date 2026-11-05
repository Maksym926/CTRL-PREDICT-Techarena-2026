"""Phase A: diagnose the local-to-portal gap without adding CTR models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .features import build_features, build_stats
from .run_phase1 import first_events, subset
from .svd_model import fit_impression_svd, fit_svd
from .train import fit_binary


def build_frame(data, boundaries: list[int], target_boundary: int, target_end: int):
    parts = []
    for boundary in boundaries:
        target = subset(data.train, first_events(data.train, boundary, target_end))
        stats = build_stats(data, boundary)
        svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=11)
        svd32 = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=32)
        imp = fit_impression_svd(data.train, boundary, data.n_users, data.n_items, rank=10)
        parts.append(build_features(data, target, boundary, svd, stats, svd_alt=svd32, impression_svd=imp))
        print(f"built train window {boundary} with {len(target.timestamp):,} events", flush=True)
    train_frame = pd.concat(parts, ignore_index=True)
    target = subset(data.train, first_events(data.train, target_boundary, target_end))
    stats = build_stats(data, target_boundary)
    svd = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=11)
    svd32 = fit_svd(stats["click_log"], data.n_users, data.n_items, rank=32)
    imp = fit_impression_svd(data.train, target_boundary, data.n_users, data.n_items, rank=10)
    target_frame = build_features(data, target, target_boundary, svd, stats, svd_alt=svd32, impression_svd=imp)
    return train_frame, target_frame


def add_choice(frame, choice):
    utility, probability, no_click = choice.predict(frame)
    result = frame.copy()
    result["choice_utility"] = utility
    result["choice_probability"] = probability
    result["choice_no_click"] = np.repeat(no_click, 10)
    return result


def bucket_labels(frame: pd.DataFrame, gap_col: str = "gap_hours"):
    history = pd.cut(frame["history_length"], [-1, 0, 9, 19, 20], labels=["0", "1-9", "10-19", "20"])
    gap = pd.cut(frame[gap_col], [-np.inf, 6, 24, 72, 168, np.inf], labels=["<6h", "6-24h", "1-3d", "3-7d", ">7d"])
    return history.astype(str), gap.astype(str)


def bucket_auc(frame: pd.DataFrame, prediction: np.ndarray, label_col: str = "label"):
    history, gap = bucket_labels(frame)
    work = frame[[label_col]].copy()
    work["prediction"] = prediction
    result = {"history_length": {}, "gap": {}}
    for name, groups in (("history_length", history), ("gap", gap)):
        for bucket in sorted(groups.unique(), key=str):
            mask = groups.to_numpy() == bucket
            y = work.loc[mask, label_col].to_numpy()
            result[name][bucket] = {
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "auc": float(roc_auc_score(y, work.loc[mask, "prediction"])) if len(np.unique(y)) > 1 else None,
            }
    return result


def weighted_auc(frame: pd.DataFrame, prediction: np.ndarray, eval_frame: pd.DataFrame):
    """Reweight by history/gap cells and return support-aware diagnostics."""
    vh, vg = bucket_labels(frame)
    eh, eg = bucket_labels(eval_frame)
    vkey = np.asarray(vh + "|" + vg)
    ekey = np.asarray(eh + "|" + eg)
    v_counts = pd.Series(vkey).value_counts()
    e_counts = pd.Series(ekey).value_counts(normalize=True)
    weights = np.zeros(len(frame), dtype=np.float64)
    supported_eval_mass = 0.0
    for key, eval_share in e_counts.items():
        mask = vkey == key
        if mask.any():
            weights[mask] = float(eval_share / mask.mean())
            supported_eval_mass += float(eval_share)
    valid = weights > 0
    auc = float(roc_auc_score(frame.loc[valid, "label"], prediction[valid], sample_weight=weights[valid])) if valid.any() else None
    # History-only weighting has complete support and is useful even when the
    # standard B=26 validation has no 1-3d or 3-7d rows.
    h_counts = pd.Series(np.asarray(eh)).value_counts(normalize=True)
    v_history = np.asarray(vh)
    v_h_share = pd.Series(v_history).value_counts(normalize=True)
    hw = np.asarray([h_counts.get(x, 0.0) / max(v_h_share.get(x, 0.0), 1e-12) for x in v_history])
    history_auc = float(roc_auc_score(frame["label"], prediction, sample_weight=hw))
    return {
        "joint_auc_supported": auc,
        "joint_eval_mass_supported": supported_eval_mass,
        "joint_validation_row_share_supported": float(valid.mean()),
        "history_only_auc": history_auc,
        "eval_history_share": {str(k): float(v) for k, v in h_counts.items()},
        "eval_joint_share": {str(k): float(v) for k, v in e_counts.items()},
    }


def adversarial_report(validation: pd.DataFrame, evaluation: pd.DataFrame):
    drop = {"label", "event_index", "candidate_index", "user_id", "timestamp"}
    columns = [c for c in validation.columns if c not in drop and c in evaluation.columns]
    x = pd.concat([validation[columns], evaluation[columns]], ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.r_[np.zeros(len(validation), dtype=np.int8), np.ones(len(evaluation), dtype=np.int8)]
    rng = np.random.default_rng(20261024)
    sample = rng.choice(len(x), size=min(120_000, len(x)), replace=False)
    model = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31, learning_rate=0.08, random_state=20261024)
    model.fit(x.iloc[sample], y[sample])
    pred = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, pred))
    perm_sample = rng.choice(len(x), size=min(30_000, len(x)), replace=False)
    perm = permutation_importance(model, x.iloc[perm_sample], y[perm_sample], n_repeats=2, random_state=20261024, n_jobs=4, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[::-1][:10]
    top = [{"feature": columns[i], "importance": float(perm.importances_mean[i])} for i in order]
    return {"auc": auc, "top_shifted_features": top}


def main():
    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    # Submitted-model validation: the same five windows and feature path.
    train_frame, validation = build_frame(data, [D0 + i * DAY for i in range(1, 6)], D0 + 6 * DAY, CUT)
    choice = fit_choice(train_frame, maxiter=120)
    train_aug = add_choice(train_frame, choice)
    validation_aug = add_choice(validation, choice)
    model, columns, val_prediction, standard_auc, importance = fit_binary(train_aug, validation_aug)

    eval_stats = build_stats(data, CUT)
    eval_svd = fit_svd(eval_stats["click_log"], data.n_users, data.n_items, rank=11)
    eval_svd32 = fit_svd(eval_stats["click_log"], data.n_users, data.n_items, rank=32)
    eval_imp = fit_impression_svd(data.train, CUT, data.n_users, data.n_items, rank=10)
    evaluation = build_features(data, data.evaluation, CUT, eval_svd, eval_stats, svd_alt=eval_svd32, impression_svd=eval_imp)
    evaluation_aug = add_choice(evaluation, choice)
    report = {
        "standard_validation_auc": float(standard_auc),
        "validation_rows": int(len(validation_aug)),
        "evaluation_rows": int(len(evaluation_aug)),
        "validation_buckets": bucket_auc(validation_aug, val_prediction),
        "evaluation_buckets": {
            "history_length": bucket_labels(evaluation_aug)[0].value_counts(normalize=True).sort_index().to_dict(),
            "gap": bucket_labels(evaluation_aug)[1].value_counts(normalize=True).sort_index().to_dict(),
        },
        "reweighting": weighted_auc(validation_aug, val_prediction, evaluation_aug),
        "adversarial": adversarial_report(validation_aug, evaluation_aug),
        "top_model_features": importance[:20],
    }

    # Long-gap validation at B=Jan24, with only earlier rolling windows.
    long_boundary = D0 + 4 * DAY
    long_train, long_validation = build_frame(data, [D0 + i * DAY for i in range(1, 4)], long_boundary, CUT)
    long_choice = fit_choice(long_train, maxiter=120)
    long_train = add_choice(long_train, long_choice)
    long_validation = add_choice(long_validation, long_choice)
    long_model, _, long_prediction, long_auc, _ = fit_binary(long_train, long_validation)
    report["long_gap_validation"] = {
        "boundary": long_boundary,
        "auc": float(long_auc),
        "rows": int(len(long_validation)),
        "buckets": bucket_auc(long_validation, long_prediction),
        "reweighting": weighted_auc(long_validation, long_prediction, evaluation_aug),
    }

    (OUTPUT_DIR / "phase_a_gap_diagnosis.json").write_text(json.dumps(report, indent=2, default=float))
    np.savez(OUTPUT_DIR / "phase_a_predictions.npz", validation_prediction=val_prediction, validation_label=validation_aug["label"].to_numpy(), validation_history=validation_aug["history_length"].to_numpy(), validation_gap=validation_aug["gap_hours"].to_numpy())
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
