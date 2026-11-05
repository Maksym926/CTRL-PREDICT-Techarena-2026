"""Seven-seed full-mode confirmation for the tuned SAFE_LOGMF feature set."""
from __future__ import annotations
import json, time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import join_global
from .lmf_baselines import add_base, score_path
from .phase_a import weighted_auc
from .phase_c_real import cached_frame
from .phase_c_revision import bounds_for, train_frames
from .submit import validate_submission, write_submission


SEEDS=(20261024,20261025,20261026,20261027,20261028,20261029,20261030)
PARAMS={"num_leaves":154,"min_data_in_leaf":201,"feature_fraction":0.6728134504603546,"bagging_fraction":0.9165758721143165,"lambda_l1":6.406134606768282,"lambda_l2":43.70944332031455,"max_bin":63}
TREES=625


def build_fold(data, day):
    B=D0+day*DAY; bounds=bounds_for(day,half=True)
    train,val,_=train_frames(data,B,bounds,'a3_c1_c2',choice_maxiter=20)
    parts=[]; off=0
    for wi,b in enumerate(bounds):
        n=int((train['window_id'].to_numpy()==wi).sum()); p=train.iloc[off:off+n].copy(); p=add_base(p,np.load(score_path(b))); parts.append(p); off+=n
    return pd.concat(parts,ignore_index=True),add_base(val,np.load(score_path(B))),cached_frame(data,CUT,'eval')


def full_fold(train,val,evalf,name):
    cols=[c for c in train.columns if c not in {'label','event_index','candidate_index','user_id','timestamp','window_id','sampling_weight'}]
    X=train[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); XV=val[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); y=train['label'].astype(int)
    preds=[]
    for seed in SEEDS:
        model=lgb.LGBMClassifier(objective='binary',n_estimators=TREES,learning_rate=0.02,
            num_leaves=PARAMS['num_leaves'],max_bin=PARAMS['max_bin'],min_child_samples=PARAMS['min_data_in_leaf'],
            colsample_bytree=PARAMS['feature_fraction'],subsample=PARAMS['bagging_fraction'],subsample_freq=1,
            reg_alpha=PARAMS['lambda_l1'],reg_lambda=PARAMS['lambda_l2'],random_state=seed,n_jobs=8,verbosity=-1)
        model.fit(X,y); preds.append(model.predict_proba(XV)[:,1]); print(name,'seed',seed,flush=True)
    p=np.mean(preds,axis=0); dm=weighted_auc(val,p,evalf)
    return {'name':name,'dm':float(dm['joint_auc_supported']),'history_dm':float(dm['history_only_auc']),'raw_auc':float(roc_auc_score(val['label'],p)),'train_rows':int(len(train)),'val_rows':int(len(val))}


def build_submission(data):
    bounds=bounds_for(6,half=True); parts=[]
    for wi,b in enumerate(bounds):
        # SAFE production windows are first-event windows through Jan 25.5.
        train_part,_val,_=build_fold(data,6) if False else (None,None,None)
        # Build directly to avoid reusing a validation fold.
        from .phase_c_revision import add_family
        frame=add_family(data,cached_frame(data,b).copy(),b,'a3_c1_c2'); frame=add_base(frame,np.load(score_path(b))); frame['window_id']=wi; parts.append(frame)
    train=pd.concat(parts,ignore_index=True)
    evalf=cached_frame(data,CUT,'eval').copy()
    from .phase_c_real import shrink_extra
    evalf=pd.concat([evalf.reset_index(drop=True),shrink_extra(data,data.evaluation,CUT,k=10.0,uncertainty=True).reset_index(drop=True)],axis=1)
    evalf=add_base(evalf,np.load(score_path(CUT)))
    from .choice import fit_choice
    choice=fit_choice(train,maxiter=20); tu,tp,tn=choice.predict(train); eu,ep,en=choice.predict(evalf)
    train=train.copy(); train['choice_utility']=tu; train['choice_probability']=tp; train['choice_no_click']=np.repeat(tn,10)
    evalf=evalf.copy(); evalf['choice_utility']=eu; evalf['choice_probability']=ep; evalf['choice_no_click']=np.repeat(en,10)
    cols=[c for c in train.columns if c not in {'label','event_index','candidate_index','user_id','timestamp','window_id','sampling_weight'}]
    X=train[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); XE=evalf[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); y=train['label'].astype(int); scores=[]
    for seed in SEEDS:
        model=lgb.LGBMClassifier(objective='binary',n_estimators=TREES,learning_rate=0.02,num_leaves=PARAMS['num_leaves'],max_bin=PARAMS['max_bin'],min_child_samples=PARAMS['min_data_in_leaf'],colsample_bytree=PARAMS['feature_fraction'],subsample=PARAMS['bagging_fraction'],subsample_freq=1,reg_alpha=PARAMS['lambda_l1'],reg_lambda=PARAMS['lambda_l2'],random_state=seed,n_jobs=8,verbosity=-1)
        model.fit(X,y); scores.append(model.predict_proba(XE)[:,1]); print('production seed',seed,flush=True)
    score=np.clip(np.mean(scores,axis=0),1e-7,1-1e-7); rowids=data.evaluation.rowids.reshape(-1); ev=pd.DataFrame({'rowid':rowids}); ex=pd.read_csv(EXAMPLE_PATH); path=OUTPUT_DIR/'SAFE_LOGMF_v2.csv'; write_submission(rowids,score,ev,path,ex); written=pd.read_csv(path); validate_submission(written,ev,ex); return {'path':str(path),'rows':int(len(written)),'unique_rowids':int(written.rowid.nunique()),'duplicates':int(written.rowid.duplicated().sum()),'min':float(score.min()),'max':float(score.max()),'mean':float(score.mean()),'trees':TREES,'seeds':list(SEEDS)}


def main():
    import sys
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); out={'params':PARAMS,'trees':TREES,'seeds':list(SEEDS),'folds':{}}
    cached=OUTPUT_DIR/'lmf_v2_dm24.json'
    if '--continue' in sys.argv and cached.exists():
        prior=json.loads(cached.read_text()); out.update({k:v for k,v in prior.items() if k not in {'folds'}}); out['folds'].update(prior.get('folds',{}))
    else:
        t=time.time(); train,val,evalf=build_fold(data,4); out['folds']['DM24']=full_fold(train,val,evalf,'DM24'); out['folds']['DM24']['elapsed_sec']=time.time()-t
    (OUTPUT_DIR/'lmf_v2_dm24.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
    if out['folds']['DM24']['dm'] < 0.8011938531181393 + 0.001:
        print('DM24 full gate not reached; no SAFE_LOGMF_v2 built',flush=True); return
    for name,day in (("DM22",2),("DM23",3)):
        t=time.time(); train,val,evalf=build_fold(data,day); out['folds'][name]=full_fold(train,val,evalf,name); out['folds'][name]['elapsed_sec']=time.time()-t; print(out['folds'][name],flush=True)
    out['full_DM_mean3']=float(np.mean([out['folds'][n]['dm'] for n in ('DM22','DM23','DM24')]))
    out['baseline_full_DM_mean3']=0.7920908228267501
    out['full_gain']=float(out['full_DM_mean3']-out['baseline_full_DM_mean3'])
    if out['full_gain'] >= 0.001:
        out['production']=build_submission(data)
    else:
        out['production']={'status':'not_built','reason':'full DM_mean3 gate failed'}
    (OUTPUT_DIR/'lmf_v2_results.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))

if __name__=='__main__': main()
