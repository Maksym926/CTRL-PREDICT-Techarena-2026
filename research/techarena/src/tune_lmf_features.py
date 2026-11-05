"""Bounded random search for SAFE_LOGMF10 experiment-mode parameters."""
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
from .lmf_baselines import add_base, score_path
from .phase_a import weighted_auc
from .phase_c_real import cached_frame, importance_weights
from .phase_c_revision import bounds_for, train_frames
from .mf_variant_experiment import sample_events


SEEDS=(20261024,20261025,20261026)


def build_fold(data, day):
    B=D0+day*DAY; bounds=bounds_for(day,half=True)
    train,val,_=train_frames(data,B,bounds,'a3_c1_c2',choice_maxiter=20)
    parts=[]; off=0
    for wi,b in enumerate(bounds):
        n=int((train['window_id'].to_numpy()==wi).sum()); p=train.iloc[off:off+n].copy(); p=add_base(p,np.load(score_path(b))); parts.append(p); off+=n
    train=pd.concat(parts,ignore_index=True); val=add_base(val,np.load(score_path(B))); evalf=cached_frame(data,CUT,'eval')
    sample=sample_events(train)
    cols=[c for c in train.columns if c not in {'label','event_index','candidate_index','user_id','timestamp','window_id','sampling_weight'}]
    X=sample[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); XV=val[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); y=sample['label'].astype(int); w=importance_weights(sample,evalf)
    return B,train,val,evalf,cols,X,XV,y,w


def trial(params, folds, trial_id):
    vals=[]
    for name,(B,train,val,evalf,cols,X,XV,y,w) in folds.items():
        model=lgb.LGBMClassifier(objective='binary',n_estimators=250,learning_rate=0.05,
            num_leaves=params['num_leaves'],max_bin=params['max_bin'],min_child_samples=params['min_data_in_leaf'],
            colsample_bytree=params['feature_fraction'],subsample=params['bagging_fraction'],subsample_freq=1,
            reg_alpha=params['lambda_l1'],reg_lambda=params['lambda_l2'],random_state=20261024+trial_id,n_jobs=8,verbosity=-1)
        model.fit(X,y,sample_weight=w,eval_set=[(XV,val['label'].astype(int))],eval_metric='auc',callbacks=[lgb.early_stopping(40,verbose=False)])
        p=model.predict_proba(XV,num_iteration=model.best_iteration_)[:,1]; dm=weighted_auc(val,p,evalf)['joint_auc_supported']; vals.append(float(dm)); print('trial',trial_id,name,dm,flush=True)
    return vals


def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); folds={}
    for name,day in (('DM22',2),('DM23',3),('DM24',4)):
        t=time.time(); folds[name]=build_fold(data,day); print('built',name,'sec',time.time()-t,flush=True)
    rng=np.random.default_rng(20261025)
    trials=[{'num_leaves':63,'min_data_in_leaf':300,'feature_fraction':0.7,'bagging_fraction':0.8,'lambda_l1':0.0,'lambda_l2':10.0,'max_bin':127}]
    for _ in range(4):
        trials.append({'num_leaves':int(rng.integers(31,256)),'min_data_in_leaf':int(rng.integers(100,2001)),'feature_fraction':float(rng.uniform(.4,.9)),'bagging_fraction':float(rng.uniform(.6,1.0)),'lambda_l1':float(rng.uniform(0,10)),'lambda_l2':float(rng.uniform(0,50)),'max_bin':int(rng.choice([63,127,255]))})
    results=[]
    for i,p in enumerate(trials):
        vals=trial(p,folds,i); results.append({'trial':i,'params':p,'DM22':vals[0],'DM23':vals[1],'DM24':vals[2],'DM_mean3':float(np.mean(vals))})
    best=max(results,key=lambda x:x['DM_mean3']); out={'max_trials':30,'trials_run':len(results),'results':results,'best':best,'baseline':results[0]}
    (OUTPUT_DIR/'lmf_lgbm_tuning.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))

if __name__=='__main__': main()
