"""Run only the A3 half-day weighted final model after A2 diagnostics."""
from __future__ import annotations
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR,CUT,D0,DAY,EXAMPLE_PATH,OUTPUT_DIR
from .dataset import join_global
from .features_a2 import build_features,build_stats
from .phase_a2 import add_choice,importance_weights
from .run_phase1 import first_events,subset
from .svd_model import fit_impression_svd,fit_svd
from .submit import write_submission
from .train import fit_binary

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    est=build_stats(data,CUT); s11=fit_svd(est['click_log'],data.n_users,data.n_items,rank=11); s32=fit_svd(est['click_log'],data.n_users,data.n_items,rank=32); imp=fit_impression_svd(data.train,CUT,data.n_users,data.n_items,rank=10)
    eval_raw=build_features(data,data.evaluation,CUT,s11,est,svd_alt=s32,impression_svd=imp)
    half_bounds=[D0+int(i*DAY/2) for i in range(1,12)]
    parts=[]
    for b in half_bounds:
        target=subset(data.train,first_events(data.train,b,CUT)); st=build_stats(data,b); a=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11); c=fit_svd(st['click_log'],data.n_users,data.n_items,rank=32); q=fit_impression_svd(data.train,b,data.n_users,data.n_items,rank=10)
        parts.append(build_features(data,target,b,a,st,svd_alt=c,impression_svd=q)); print('half window',b,target.n_events,flush=True)
    train=pd.concat(parts,ignore_index=True)
    b=D0+6*DAY; target=subset(data.train,first_events(data.train,b,CUT)); st=build_stats(data,b); a=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11); c=fit_svd(st['click_log'],data.n_users,data.n_items,rank=32); q=fit_impression_svd(data.train,b,data.n_users,data.n_items,rank=10); val=build_features(data,target,b,a,st,svd_alt=c,impression_svd=q)
    choice=fit_choice(train,maxiter=120); train=add_choice(train,choice); val=add_choice(val,choice); ev=add_choice(eval_raw,choice)
    w=importance_weights(train,eval_raw)
    model,cols,pred,auc,imp=fit_binary(train,val,train_weight=w)
    report={'A3_standard_auc':auc,'A3_standard_weighted':float(roc_auc_score(val['label'],pred,sample_weight=np.ones(len(val)))),'best_iteration':int(model.best_iteration_),'train_rows':len(train),'weights_min':float(w.min()),'weights_max':float(w.max()),'top_features':imp[:20]}
    # The validation-fitted booster already uses only the half-day training
    # windows and has its early-stopped tree count; reuse it for evaluation
    # rather than fitting a second full 5M-row booster.
    score=np.clip(model.predict_proba(ev[cols],num_iteration=model.best_iteration_)[:,1].astype(float),1e-7,1-1e-7); rowids=data.evaluation.rowids.reshape(-1); evaluation=pd.DataFrame({'rowid':rowids}); example=pd.read_csv(EXAMPLE_PATH); out=OUTPUT_DIR/'SAFE_A2.csv'; write_submission(rowids,score,evaluation,out,example)
    report.update({'output':str(out),'mean':float(score.mean()),'history_mean':float(score[eval_raw['history_count'].to_numpy()>0].mean()),'other_mean':float(score[eval_raw['history_count'].to_numpy()==0].mean())})
    (OUTPUT_DIR/'phase_a3_results.json').write_text(json.dumps(report,indent=2,default=float)); print(json.dumps(report,indent=2,default=float))
if __name__=='__main__':main()
