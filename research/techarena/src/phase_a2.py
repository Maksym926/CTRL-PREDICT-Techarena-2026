"""Phase A2/A3: fixed-window features, long-gap folds and SAFE_A2 output."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import join_global
from .features_a2 import build_features, build_stats
from .phase_a import adversarial_report, bucket_labels, bucket_auc, weighted_auc
from .run_phase1 import first_events, subset
from .svd_model import fit_impression_svd, fit_svd
from .submit import write_submission
from .train import feature_columns, fit_binary


def frame_for_boundary(data, boundary, train_boundaries, feature_builder=build_features):
    parts=[]
    for b in train_boundaries:
        target=subset(data.train, first_events(data.train,b,CUT))
        st=build_stats(data,b); s11=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11); s32=fit_svd(st['click_log'],data.n_users,data.n_items,rank=32); imp=fit_impression_svd(data.train,b,data.n_users,data.n_items,rank=10)
        parts.append(feature_builder(data,target,b, s11,st,svd_alt=s32,impression_svd=imp))
        print(f'window {b}: {target.n_events:,} events',flush=True)
    train=pd.concat(parts,ignore_index=True)
    target=subset(data.train,first_events(data.train,boundary,CUT))
    st=build_stats(data,boundary); s11=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11); s32=fit_svd(st['click_log'],data.n_users,data.n_items,rank=32); imp=fit_impression_svd(data.train,boundary,data.n_users,data.n_items,rank=10)
    val=feature_builder(data,target,boundary,s11,st,svd_alt=s32,impression_svd=imp)
    return train,val


def add_choice(frame, choice):
    u,p,n=choice.predict(frame); out=frame.copy(); out['choice_utility']=u; out['choice_probability']=p; out['choice_no_click']=np.repeat(n,10); return out


def importance_weights(train, evaluation):
    th,tg=bucket_labels(train); eh,eg=bucket_labels(evaluation)
    tk=np.asarray(th+'|'+tg); ek=np.asarray(eh+'|'+eg)
    e_share=pd.Series(ek).value_counts(normalize=True); t_share=pd.Series(tk).value_counts(normalize=True)
    weights=np.ones(len(train),dtype=np.float32)
    for key,share in e_share.items():
        mask=tk==key
        if mask.any(): weights[mask]=float(share/max(t_share.get(key,1e-12),1e-12))
    return weights


def train_choice_gbdt(train,val,eval_frame,weights=None):
    choice=fit_choice(train,maxiter=120)
    train=add_choice(train,choice); val=add_choice(val,choice); eval_frame=add_choice(eval_frame,choice)
    model,cols,pred,auc,importance=fit_binary(train,val,train_weight=weights)
    return choice,train,val,eval_frame,model,cols,pred,auc,importance


def eval_fold(data,boundary,train_boundaries,eval_frame):
    train,val=frame_for_boundary(data,boundary,train_boundaries)
    # Fit choice/GBDT; DM is always computed against evaluation bucket weights.
    choice,train,val,_,model,cols,pred,auc,imp=train_choice_gbdt(train,val,eval_frame)
    return {'train':train,'val':val,'pred':pred,'auc':auc,'importance':imp,'model':model,'cols':cols,'choice':choice,'reweight':weighted_auc(val,pred,eval_frame),'buckets':bucket_auc(val,pred)}


def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    # Evaluation features are built once using the A2 fixed windows.
    est=build_stats(data,CUT); s11=fit_svd(est['click_log'],data.n_users,data.n_items,rank=11); s32=fit_svd(est['click_log'],data.n_users,data.n_items,rank=32); imp=fit_impression_svd(data.train,CUT,data.n_users,data.n_items,rank=10)
    eval_raw=build_features(data,data.evaluation,CUT,s11,est,svd_alt=s32,impression_svd=imp)

    folds={}
    for name,day in [('DM23',3),('DM24',4),('STANDARD26',6)]:
        b=D0+day*DAY; train_bounds=[D0+i*DAY for i in range(1,day)]
        folds[name]=eval_fold(data,b,train_bounds,eval_raw)
        print(name,folds[name]['auc'],folds[name]['reweight'],flush=True)
    dm23=folds['DM23']['reweight']['joint_auc_supported']; dm24=folds['DM24']['reweight']['joint_auc_supported']; dmmean=float((dm23+dm24)/2)
    report={'DM23':dm23,'DM24':dm24,'DM_mean':dmmean,'standard_validation_auc':folds['STANDARD26']['auc'],'standard_reweighting':folds['STANDARD26']['reweight'],'adversarial_overall':adversarial_report(folds['STANDARD26']['val'],eval_raw)}
    # Exclude the two deliberately shifted fields from adversarial validation.
    adv_val=folds['STANDARD26']['val'].drop(columns=['gap_hours','history_length'])
    adv_eval=eval_raw.drop(columns=['gap_hours','history_length'])
    report['adversarial_excluding_gap_history']=adversarial_report(adv_val,adv_eval)
    report['DM23_buckets']=folds['DM23']['buckets']; report['DM24_buckets']=folds['DM24']['buckets']

    # A3: half-day targets and evaluation-matched row weights.
    half_bounds=[D0+int(i*DAY/2) for i in range(1,12)]
    train_parts=[]
    for b in half_bounds:
        target=subset(data.train,first_events(data.train,b,CUT)); st=build_stats(data,b); s11=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11); s32=fit_svd(st['click_log'],data.n_users,data.n_items,rank=32); imp=fit_impression_svd(data.train,b,data.n_users,data.n_items,rank=10)
        train_parts.append(build_features(data,target,b,s11,st,svd_alt=s32,impression_svd=imp))
    half_train=pd.concat(train_parts,ignore_index=True)
    # Use the standard boundary as the held-out report for this weighted model.
    standard_val=folds['STANDARD26']['val'].copy()
    ch=fit_choice(half_train,maxiter=120); half_aug=add_choice(half_train,ch); val_aug=add_choice(standard_val,ch)
    w=importance_weights(half_aug,eval_raw)
    weighted_model,cols,pred,weighted_auc_std,weighted_imp=fit_binary(half_aug,val_aug,train_weight=w)
    report['A3_weighted_standard_auc']=weighted_auc_std; report['A3_weighted_standard_reweighting']=weighted_auc(val_aug,pred,eval_raw); report['A3_weighted_buckets']=bucket_auc(val_aug,pred)

    # Fit final SAFE_A2 from the half-day weighted targets and predict evaluation.
    final_model=lgb.LGBMClassifier(objective='binary',n_estimators=max(50,round(weighted_model.best_iteration_*1.15)),learning_rate=0.05,num_leaves=63,min_child_samples=300,colsample_bytree=0.7,subsample=0.8,subsample_freq=1,reg_lambda=10.0,random_state=20261024,n_jobs=4,verbosity=-1)
    final_model.fit(half_aug[cols],half_aug['label'].astype(int),sample_weight=w)
    score=final_model.predict_proba(add_choice(eval_raw,ch)[cols])[:,1]
    score=np.clip(score,1e-7,1-1e-7); rowids=data.evaluation.rowids.reshape(-1); evaluation=pd.DataFrame({'rowid':rowids}); example=pd.read_csv(EXAMPLE_PATH)
    safe=OUTPUT_DIR/'SAFE_A2.csv'; write_submission(rowids,score,evaluation,safe,example)
    report['safe_output']=str(safe); report['safe_mean']=float(score.mean()); report['safe_history_mean']=float(score[eval_raw['history_count'].to_numpy()>0].mean()); report['safe_other_mean']=float(score[eval_raw['history_count'].to_numpy()==0].mean())
    (OUTPUT_DIR/'phase_a2_results.json').write_text(json.dumps(report,indent=2,default=float))
    print(json.dumps(report,indent=2,default=float))


if __name__=='__main__': main()
