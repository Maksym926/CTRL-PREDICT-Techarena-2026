"""Per-scorer LightGBM residual gains for the completed graph audit."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, train_frames
from .mf_variant_experiment import add_base, sample_events


def cols(f):
    return [c for c in f.columns if c not in {"label","event_index","candidate_index","user_id","timestamp","window_id","sampling_weight"}]


def fit_one(train, val, evalf, extra=None):
    use = cols(train)
    x = train[use].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    xv = val[use].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sm = sample_events(train)
    xs = sm[use].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    w = importance_weights(sm, evalf)
    model = lgb.LGBMClassifier(objective='binary',n_estimators=250,learning_rate=.05,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.,random_state=20261025,n_jobs=8,verbosity=-1)
    model.fit(xs,sm['label'].astype(int),sample_weight=w,eval_set=[(xv,val['label'].astype(int))],eval_metric='auc',callbacks=[lgb.early_stopping(60,verbose=False)])
    p=model.predict_proba(xv,num_iteration=model.best_iteration_)[:,1]
    return float(weighted_auc(val,p,evalf)['joint_auc_supported']), int(model.best_iteration_)


def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    B=D0+4*DAY; bounds=bounds_for(4,half=True)
    train,val,_=train_frames(data,B,bounds,'a3_c1_c2',choice_maxiter=20)
    parts=[]; off=0
    for wi,b in enumerate(bounds):
        n=int((train['window_id'].to_numpy()==wi).sum())
        p=OUTPUT_DIR/f'logmf_cosine_prod{wi}_{int(b)}.npy'
        parts.append(add_base(train.iloc[off:off+n].copy(),np.load(p))); off+=n
    train=pd.concat(parts,ignore_index=True)
    val=add_base(val,np.load(OUTPUT_DIR/f'logmf_cosine_val_dm24_{int(B)}.npy'))
    evalf=cached_frame(data,CUT,'eval')
    base,biter=fit_one(train,val,evalf)
    names=[]
    for z in np.load(OUTPUT_DIR/f'graph_scores_{int(B)}.npz').files:
        if z!='user_knn': names.append(z)
    result={'base_dm24':base,'base_iteration':biter,'scorers':{}}
    for name in names:
        tr_parts=[]; off=0
        for wi,b in enumerate(bounds):
            n=int((train['window_id'].to_numpy()==wi).sum())
            z=np.load(OUTPUT_DIR/f'graph_scores_{int(b)}.npz')[name]
            part=train.iloc[off:off+n].copy(); q=z.reshape(-1,10)
            part[f'g_{name}_score']=z
            part[f'g_{name}_rank']=np.argsort(np.argsort(-q,axis=1),axis=1).astype(np.float32).reshape(-1)
            part[f'g_{name}_gap']=(q-q.max(axis=1,keepdims=True)).astype(np.float32).reshape(-1)
            tr_parts.append(part); off+=n
        tr=pd.concat(tr_parts,ignore_index=True)
        z=np.load(OUTPUT_DIR/f'graph_scores_{int(B)}.npz')[name]; q=z.reshape(-1,10)
        va=val.copy(); va[f'g_{name}_score']=z; va[f'g_{name}_rank']=np.argsort(np.argsort(-q,axis=1),axis=1).astype(np.float32).reshape(-1); va[f'g_{name}_gap']=(q-q.max(axis=1,keepdims=True)).astype(np.float32).reshape(-1)
        dm,it=fit_one(tr,va,evalf)
        result['scorers'][name]={'dm24':dm,'gain':dm-base,'iteration':it}
        print(name,dm,dm-base,flush=True)
    (OUTPUT_DIR/'graph_residual_per_scorer.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))


if __name__=='__main__': main()
