"""Candidate-conditioned SVD attention features and a DM24 residual test."""
from __future__ import annotations
import json, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .features import build_stats
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, target_for, train_frames
from .run_phase1 import first_events
from .svd_model import fit_svd
from .mf_variant_experiment import add_base, sample_events


def attention_scores(data, boundary, temperatures=(.10,.20,.40), decay=7.0):
    target=target_for(data,boundary); st=build_stats(data,boundary); svd=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11); V=svd.item_vectors.astype(np.float32)
    out={f'din_attn_t{int(t*100):02d}':[] for t in temperatures}; out['din_attn_last']=[]; out['din_attn_max']=[]; out['din_attn_entropy']=[]
    for e in range(target.n_events):
        h=target.history_items[e]; ht=target.history_times[e]; m=h>=0; h=h[m].astype(np.int64); ht=ht[m].astype(np.int64); c=target.candidates[e].astype(np.int64); cv=V[c]
        if len(h)==0:
            for k in out: out[k].extend([0.0]*10)
            continue
        hv=V[h]; pair=(cv@hv.T)/((np.linalg.norm(cv,axis=1)[:,None]+1e-8)*(np.linalg.norm(hv,axis=1)[None,:]+1e-8)); age=np.maximum(0,int(target.timestamp[e])-ht)/86400.0; rec=-age/decay
        for t in temperatures:
            a=(pair/t+rec[None,:]); a-=a.max(1,keepdims=True); w=np.exp(a); w/=w.sum(1,keepdims=True)+1e-8; out[f'din_attn_t{int(t*100):02d}'].extend((w*pair).sum(1).astype(np.float32).tolist())
        # Temperature-free recency attention and entropy at t=.2.
        a=pair/.2+rec[None,:]; a-=a.max(1,keepdims=True); w=np.exp(a); w/=w.sum(1,keepdims=True)+1e-8
        out['din_attn_last'].extend(pair[:,-1].astype(np.float32).tolist()); out['din_attn_max'].extend(pair.max(1).astype(np.float32).tolist()); out['din_attn_entropy'].extend((-(w*np.log(np.maximum(w,1e-8))).sum(1)).astype(np.float32).tolist())
    return {k:np.asarray(v,np.float32) for k,v in out.items()}


def add_features(df, scores):
    out=df.reset_index(drop=True).copy()
    for k,v in scores.items(): out[k]=v
    return out


def main():
    t0=time.time(); data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); B=D0+4*DAY; bounds=bounds_for(4,half=True)
    train,val,_=train_frames(data,B,bounds,'a3_c1_c2',choice_maxiter=20)
    parts=[]; off=0
    for wi,b in enumerate(bounds):
        n=int((train['window_id'].to_numpy()==wi).sum()); p=OUTPUT_DIR/f'logmf_cosine_prod{wi}_{int(b)}.npy'; s=attention_scores(data,b); np.savez_compressed(OUTPUT_DIR/f'din_svd_{int(b)}.npz',**s); parts.append(add_features(add_base(train.iloc[off:off+n].copy(),np.load(p)),s)); off+=n; print('DIN train',b,n,flush=True)
    train=pd.concat(parts,ignore_index=True); val=add_features(add_base(val,np.load(OUTPUT_DIR/f'logmf_cosine_val_dm24_{int(B)}.npy')),attention_scores(data,B)); evalf=cached_frame(data,CUT,'eval')
    res={'boundary':int(B),'base_dm24':0.7989857073691065,'single':{},'models':{},'elapsed_sec':None}
    for k in [c for c in val.columns if c.startswith('din_')]: res['single'][k]={'dm24':float(weighted_auc(val,val[k].to_numpy(),evalf)['joint_auc_supported']),'raw_auc':float(roc_auc_score(val['label'],val[k])),'corr_svd':float(pd.Series(val[k]).corr(pd.Series(val['svd_cos']),method='spearman'))}
    sm=sample_events(train); cols=[c for c in train.columns if c not in {'label','event_index','candidate_index','user_id','timestamp','window_id','sampling_weight'}]; X=sm[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); XV=val[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); w=importance_weights(sm,evalf); model=lgb.LGBMClassifier(objective='binary',n_estimators=250,learning_rate=.05,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.,random_state=20261025,n_jobs=8,verbosity=-1); model.fit(X,sm['label'].astype(int),sample_weight=w,eval_set=[(XV,val['label'].astype(int))],eval_metric='auc',callbacks=[lgb.early_stopping(60,verbose=False)]); p=model.predict_proba(XV,num_iteration=model.best_iteration_)[:,1]; dm=weighted_auc(val,p,evalf)['joint_auc_supported']; res['models']['all_din_features']={'dm24':float(dm),'gain':float(dm-res['base_dm24']),'best_iteration':int(model.best_iteration_)}; res['elapsed_sec']=time.time()-t0; (OUTPUT_DIR/'din_svd_dm24.json').write_text(json.dumps(res,indent=2,default=float)); print(json.dumps(res,indent=2,default=float))
if __name__=='__main__': main()
