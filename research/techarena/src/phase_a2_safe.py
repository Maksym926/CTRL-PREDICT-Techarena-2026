"""Produce a checked A2 candidate from the completed five-window path."""
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
from .phase_a2 import add_choice
from .run_phase1 import first_events,subset
from .svd_model import fit_impression_svd,fit_svd
from .submit import write_submission
from .train import fit_binary

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    est=build_stats(data,CUT);s=fit_svd(est['click_log'],data.n_users,data.n_items,11);s32=fit_svd(est['click_log'],data.n_users,data.n_items,32);imp=fit_impression_svd(data.train,CUT,data.n_users,data.n_items,10);evraw=build_features(data,data.evaluation,CUT,s,est,svd_alt=s32,impression_svd=imp)
    parts=[]
    for b in [D0+i*DAY for i in range(1,6)]:
        t=subset(data.train,first_events(data.train,b,CUT));st=build_stats(data,b);a=fit_svd(st['click_log'],data.n_users,data.n_items,11);c=fit_svd(st['click_log'],data.n_users,data.n_items,32);q=fit_impression_svd(data.train,b,data.n_users,data.n_items,10);parts.append(build_features(data,t,b,a,st,svd_alt=c,impression_svd=q));print('window',b,flush=True)
    train=pd.concat(parts,ignore_index=True);b=D0+6*DAY;t=subset(data.train,first_events(data.train,b,CUT));st=build_stats(data,b);a=fit_svd(st['click_log'],data.n_users,data.n_items,11);c=fit_svd(st['click_log'],data.n_users,data.n_items,32);q=fit_impression_svd(data.train,b,data.n_users,data.n_items,10);val=build_features(data,t,b,a,st,svd_alt=c,impression_svd=q)
    ch=fit_choice(train,maxiter=120);train=add_choice(train,ch);val=add_choice(val,ch);ev=add_choice(evraw,ch);model,cols,pred,auc,imp=fit_binary(train,val);score=np.clip(model.predict_proba(ev[cols],num_iteration=model.best_iteration_)[:,1],1e-7,1-1e-7)
    rowids=data.evaluation.rowids.reshape(-1);out=OUTPUT_DIR/'SAFE_A2.csv';write_submission(rowids,score,pd.DataFrame({'rowid':rowids}),out,pd.read_csv(EXAMPLE_PATH));report={'output':str(out),'standard_auc':auc,'best_iteration':int(model.best_iteration_),'mean':float(score.mean()),'history_mean':float(score[evraw['history_count'].to_numpy()>0].mean()),'other_mean':float(score[evraw['history_count'].to_numpy()==0].mean()),'top_features':imp[:20]};(OUTPUT_DIR/'phase_a2_safe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
