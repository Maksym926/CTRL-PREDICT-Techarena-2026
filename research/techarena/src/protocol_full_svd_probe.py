"""Diagnostic: replace cross-fold SVD columns in all-event protocol data.

This is an upper-bound diagnostic only.  The full pre-B SVD is valid for the
evaluation cutoff but can see later pre-B labels for early training rows, so a
positive result must be followed by temporal cross-fitting before promotion.
"""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .features import build_stats
from .phase_a import weighted_auc
from .phase_c_real import cached_frame,importance_weights
from .protocol_train import sample_mask,user_weights
from .svd_model import fit_svd

DROP={'label','event_index','candidate_index','user_id','timestamp','window_id','sampling_weight'}

def replace_scores(data,df,svd,svd32):
    df=df.copy(); n=len(df)//10; out={k:[] for k in ['svd_dot','svd_cos','svd32_dot','svd32_cos','history_mean_cos','history_max_cos','history_last_cos','history_weighted_cos']}
    for e in df.event_index.to_numpy()[::10]:
        e=int(e); u=int(data.train.user_id[e]); h=data.train.history_items[e]; h=h[h>=0]; c=data.train.candidates[e].astype(np.int64)
        def calc(s):
            uv=s.user_vector(u,h); iv=s.item_vectors[c]; dot=iv@uv; co=dot/((np.linalg.norm(iv,axis=1)+1e-8)*(np.linalg.norm(uv)+1e-8))
            if len(h):
                hv=s.item_vectors[h]; pair=iv@hv.T/((np.linalg.norm(iv,axis=1)[:,None]+1e-8)*(np.linalg.norm(hv,axis=1)[None,:]+1e-8)); w=np.power(.8,np.arange(len(h)-1,-1,-1,dtype=np.float32)); return dot,co,pair.mean(1),pair.max(1),pair[:,-1],(pair@w)/w.sum()
            z=np.zeros(10,np.float32); return dot,co,z,z,z,z
        a=calc(svd); b=calc(svd32)
        vals={'svd_dot':a[0],'svd_cos':a[1],'history_mean_cos':a[2],'history_max_cos':a[3],'history_last_cos':a[4],'history_weighted_cos':a[5],'svd32_dot':b[0],'svd32_cos':b[1]}
        for k in out: out[k].extend(vals[k].tolist())
    for k,v in out.items(): df[k]=np.asarray(v,np.float32)
    return df

def main():
    t=time.time(); data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); B=D0+4*DAY; train=pd.read_parquet(CACHE_DIR/'protocol'/f'train_all_{B}.parquet'); val=pd.read_parquet(CACHE_DIR/'protocol'/f'val_{B}.parquet'); evalf=cached_frame(data,CUT,'eval'); st=build_stats(data,B); svd=fit_svd(st['click_log'],data.n_users,data.n_items,11); svd32=fit_svd(st['click_log'],data.n_users,data.n_items,32); print('SVD ready',flush=True); train=replace_scores(data,train,svd,svd32); val=replace_scores(data,val,svd,svd32)
    # Replace only rows where event_index is original data index; val is first
    # events and shares that indexing.
    cols=[c for c in train.columns if c not in DROP]; events=np.sort(train.event_index.unique()); rng=np.random.default_rng(20261024); randoms={int(e):float(x) for e,x in zip(events,rng.random(len(events)))}; mask=sample_mask(train,randoms); sm=train.loc[mask]; X=sm[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); XV=val[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); w=user_weights(train,mask,evalf); model=lgb.LGBMClassifier(objective='binary',n_estimators=1600,learning_rate=.03,num_leaves=63,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.,random_state=20261025,n_jobs=8,verbosity=-1); model.fit(X,sm['label'].astype(int),sample_weight=w,eval_set=[(XV,val['label'].astype(int))],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)]); p=model.predict_proba(XV,num_iteration=model.best_iteration_)[:,1]; dm=weighted_auc(val,p,evalf); out={'boundary':int(B),'rows':int(len(sm)),'events':int(sm.event_index.nunique()),'dm24':float(dm['joint_auc_supported']),'raw_auc':float(roc_auc_score(val['label'],p)),'best_iteration':int(model.best_iteration_),'baseline':0.7989857073691065,'gain':float(dm['joint_auc_supported']-0.7989857073691065),'diagnostic':'full pre-B SVD sees later pre-B labels for early training rows; not promotion-safe','elapsed_sec':time.time()-t}; (OUTPUT_DIR/'protocol_full_svd_probe.json').write_text(json.dumps(out,indent=2,default=float)); np.save(OUTPUT_DIR/'protocol_full_svd_probe_DM24.npy',p); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
