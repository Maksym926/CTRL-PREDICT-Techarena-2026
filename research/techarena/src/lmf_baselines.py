"""Compute SAFE_LOGMF10 experiment/full fold baselines."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .mf_variant_experiment import add_base, fit_compare, sample_events
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, target_for, train_frames


def score_path(boundary):
    for p in sorted(OUTPUT_DIR.glob(f"logmf_cosine_prod*_{int(boundary)}.npy")):
        return p
    if int(boundary) == int(D0 + 4 * DAY):
        p = OUTPUT_DIR / f"logmf_cosine_val_dm24_{int(boundary)}.npy"
        if p.exists():
            return p
    raise FileNotFoundError(f"no cached lambda=10 score for {boundary}")


def add_base_windows(data, train, bounds, val, boundary):
    parts = []
    offset = 0
    for wi, b in enumerate(bounds):
        n = int((train["window_id"].to_numpy() == wi).sum())
        part = train.iloc[offset : offset + n].copy()
        part = add_base(part, np.load(score_path(b)))
        parts.append(part)
        offset += n
    train = pd.concat(parts, ignore_index=True)
    val = add_base(val, np.load(score_path(boundary)))
    return train, val


def fit_full(train, val, evalf, tag):
    cols = [c for c in train.columns if c not in {"label", "event_index", "candidate_index", "user_id", "timestamp", "window_id", "sampling_weight"}]
    X = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    XV = val[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train["label"].astype(int)
    preds = []
    for seed in (20261024, 20261025, 20261026):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=350, learning_rate=0.05,
            num_leaves=63, max_bin=127, min_child_samples=300,
            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            reg_lambda=10.0, random_state=seed, n_jobs=8, verbosity=-1,
        )
        model.fit(X, y)
        preds.append(model.predict_proba(XV)[:, 1])
        print(tag, "seed", seed, flush=True)
    pred=np.mean(preds,axis=0)
    dm=weighted_auc(val,pred,evalf)
    return {"tag":tag,"dm":float(dm["joint_auc_supported"]),"history_dm":float(dm["history_only_auc"]),"raw_auc":float(__import__("sklearn.metrics",fromlist=["roc_auc_score"]).roc_auc_score(val["label"],pred)),"train_rows":int(len(train)),"val_rows":int(len(val))}


def run_fold(data, name, day):
    boundary = D0 + day * DAY
    bounds = bounds_for(day, half=True)
    t0=time.time()
    train, val, _ = train_frames(data, boundary, bounds, "a3_c1_c2", choice_maxiter=20)
    train, val = add_base_windows(data, train, bounds, val, boundary)
    evalf = cached_frame(data, CUT, "eval")
    exp = fit_compare(train, val, evalf, f"baseline_{name}")
    full = fit_full(train, val, evalf, f"full_{name}")
    return {"experiment":exp,"full":full,"elapsed_sec":time.time()-t0,"boundary":boundary,"train_bounds":bounds}


def main():
    data=join_global(load_cache(CACHE_DIR/"train.npz"),load_cache(CACHE_DIR/"evaluation.npz"))
    out={"model":"SAFE_LOGMF10","mode_details":{"negative_rate":0.35,"choice_maxiter":20,"trees":250,"full_trees":350},"folds":{}}
    for name,day in (("DM22",2),("DM23",3),("DM24",4)):
        out["folds"][name]=run_fold(data,name,day)
        print(name,out["folds"][name],flush=True)
    for mode in ("experiment","full"):
        out[mode+"_DM_mean3"]=float(np.mean([out["folds"][n][mode]["dm" if mode=="full" else "dm24"] for n in ("DM22","DM23","DM24")]))
    (OUTPUT_DIR/"safe_logmf10_baselines.json").write_text(json.dumps(out,indent=2,default=float))
    print(json.dumps(out,indent=2,default=float))

if __name__=="__main__": main()
