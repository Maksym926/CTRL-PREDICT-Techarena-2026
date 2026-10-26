"""Leakage-safe v3 feature families.

The functions here operate on a target event frame at a boundary B. All
statistics and click logs come from timestamps strictly before B.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from .features import build_stats
from .svd_model import SVDModel, fit_svd
from .config import CACHE_DIR, DAY


def _weighted_svd(click_log, n_users, n_items, rank, weights):
    pairs = click_log[:, :2].astype(np.int64, copy=False)
    rows, cols = pairs[:, 0], pairs[:, 1]
    vals = np.asarray(weights, dtype=np.float64)
    ud = np.bincount(rows, weights=vals, minlength=n_users) + 1.0
    idg = np.bincount(cols, weights=vals, minlength=n_items) + 1.0
    vals = vals / np.sqrt(ud[rows] * idg[cols])
    mat = coo_matrix((vals, (rows, cols)), shape=(n_users, n_items)).tocsr()
    k = max(1, min(rank, min(mat.shape) - 1))
    u, s, vt = svds(mat, k=k, which="LM", return_singular_vectors=True, random_state=20261024)
    order = np.argsort(s)[::-1]
    u, s, vt = u[:, order], s[order], vt[order]
    return SVDModel((u * s[None, :]).astype(np.float32), vt.T.astype(np.float32), idg.astype(np.float32), ud.astype(np.float32))


def time_decay_svd(click_log, n_users, n_items, boundary, half_life_days, rank=11):
    age = np.maximum(0.0, (int(boundary) - click_log[:, 2].astype(np.float64)) / DAY)
    weights = np.power(0.5, age / float(half_life_days))
    return _weighted_svd(click_log, n_users, n_items, rank, weights)


def bootstrap_click_log(click_log, seed):
    """Resample each user's click rows with replacement, preserving user mass."""
    rng = np.random.default_rng(seed)
    order = np.argsort(click_log[:, 0], kind="mergesort")
    ordered = click_log[order]
    out = []
    start = 0
    while start < len(ordered):
        end = start + 1
        user = ordered[start, 0]
        while end < len(ordered) and ordered[end, 0] == user:
            end += 1
        idx = rng.integers(start, end, size=end - start)
        out.append(ordered[idx])
        start = end
    return np.concatenate(out, axis=0) if out else click_log.copy()


def _scores_for_model(target, model):
    dot_all=[]; cos_all=[]; hist_all=[]; prob_all=[]
    for e in range(target.n_events):
        h = target.history_items[e]
        h = h[h >= 0]
        u0 = model.user_vector(int(target.user_id[e]), h)
        iv = model.item_vectors[target.candidates[e]]
        un = np.linalg.norm(u0) + 1e-8
        inn = np.linalg.norm(iv, axis=1) + 1e-8
        dot = iv @ u0
        cos = dot / (inn * un)
        if len(h):
            hv = model.item_vectors[h].mean(axis=0)
            hn = np.linalg.norm(hv) + 1e-8
            hist = (iv @ hv) / (inn * hn)
        else:
            hist = np.zeros(10, dtype=np.float32)
        z = np.r_[0.0, dot.astype(np.float64)]
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        dot_all.extend(dot); cos_all.extend(cos); hist_all.extend(hist); prob_all.extend(p[1:])
    return [np.asarray(x, dtype=np.float32) for x in (dot_all, cos_all, hist_all, prob_all)]


def bagged_extra(data, target, boundary, n_boot=15):
    st = build_stats(data, boundary)
    views=[]
    for b in range(n_boot):
        boot = bootstrap_click_log(st["click_log"], 20261024 + b)
        model = fit_svd(boot, data.n_users, data.n_items, rank=11)
        views.append(_scores_for_model(target, model))
    # (bootstrap, feature, row) -> (row, feature, bootstrap)
    arr=np.stack(views, axis=0).transpose(2,1,0)
    names=["dot","cos","hist_cos","choice_prob"]
    out={}
    for j,name in enumerate(names):
        out[f"bag_{name}_mean"]=arr[:,j,:].mean(axis=1)
        out[f"bag_{name}_std"]=arr[:,j,:].std(axis=1)
    return pd.DataFrame(out)


def multi_view_extra(data, target, boundary):
    st=build_stats(data,boundary)
    models={
        "r10":fit_svd(st["click_log"],data.n_users,data.n_items,rank=10),
        "r12":fit_svd(st["click_log"],data.n_users,data.n_items,rank=12),
        "decay3":time_decay_svd(st["click_log"],data.n_users,data.n_items,boundary,3.0,11),
        "decay10":time_decay_svd(st["click_log"],data.n_users,data.n_items,boundary,10.0,11),
    }
    result={}
    for name,m in models.items():
        _,cos,hist,_=_scores_for_model(target,m)
        result[f"view_{name}_cos"]=cos
        result[f"view_{name}_hist_cos"]=hist
    return pd.DataFrame(result)


def gap_extra(data, target, boundary):
    st=build_stats(data,boundary)
    log=st["click_log"]
    last={}
    for u,i,t in log:
        item=int(i)
        last[item]=max(last.get(item,-1),int(t))
    total=max(1,len(log))
    rows=[]
    for e in range(target.n_events):
        gap=max(0.0,(int(target.timestamp[e])-int(boundary))/DAY)
        for item in target.candidates[e]:
            item=int(item)
            c1=float(st["item_click_1d"].get(item,0))
            c3=float(st["item_click_3d"].get(item,0))
            c7=float(st["item_click_7d"].get(item,0))
            rate1=c1; rate3=c3/3.0; rate7=c7/7.0
            slope=(rate1-rate3)/(rate3+1.0)
            projected=rate7*(1.0+slope*gap)
            change=rate1-rate3
            share=float(st["item_clicks"].get(item,0))/total
            age=(int(target.timestamp[e])-last.get(item,int(boundary)))/DAY if item in last else 999.0
            rows.append((projected,age,change,gap*share,share))
    a=np.asarray(rows,dtype=np.float32)
    return pd.DataFrame({"gap_trend_projected":a[:,0],"item_last_click_age_days":a[:,1],"item_click_rate_change_1d_3d":a[:,2],"gap_popularity_interaction":a[:,3],"item_click_share":a[:,4]})


def cached_v3(data, target, boundary, family):
    root=CACHE_DIR/"features_v3"; root.mkdir(parents=True,exist_ok=True)
    path=root/f"{family}_{int(boundary)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if family=="bag":
        extra=bagged_extra(data,target,boundary,15)
    elif family=="multi":
        extra=multi_view_extra(data,target,boundary)
    elif family=="gap":
        extra=gap_extra(data,target,boundary)
    elif family=="all":
        extra=pd.concat([bagged_extra(data,target,boundary,15),multi_view_extra(data,target,boundary),gap_extra(data,target,boundary)],axis=1)
    else:
        raise ValueError(family)
    extra.to_parquet(path,index=False)
    print("cached v3",family,boundary,path,flush=True)
    return extra
