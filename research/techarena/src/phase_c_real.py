"""Laptop experiment harness for A3 and real-data C1-C3.

It caches the original production features per boundary, samples complete
negative events for fast experiments, and evaluates every candidate on DM23,
DM24 and their mean.
"""
from __future__ import annotations

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .features import build_features,build_stats
from .phase_a import bucket_auc,weighted_auc
from .run_phase1 import first_events,subset
from .svd_model import fit_impression_svd,fit_svd
from .train import DROP_COLUMNS,feature_columns


def _paths(boundary):
    d=CACHE_DIR/'features_orig';d.mkdir(parents=True,exist_ok=True)
    return d/f'{int(boundary)}.parquet'


def cached_frame(data,boundary,which='train'):
    path=_paths(boundary)
    if which=='eval': path=CACHE_DIR/'features_orig'/'evaluation.parquet'
    if path.exists(): return pd.read_parquet(path)
    if which=='eval': target=data.evaluation
    else: target=subset(data.train,first_events(data.train,boundary,CUT))
    st=build_stats(data,boundary);s=fit_svd(st['click_log'],data.n_users,data.n_items,11);s32=fit_svd(st['click_log'],data.n_users,data.n_items,32);imp=fit_impression_svd(data.train,boundary,data.n_users,data.n_items,10)
    frame=build_features(data,target,boundary,s,st,svd_alt=s32,impression_svd=imp)
    frame.to_parquet(path,index=False);print('cached',path,flush=True);return frame


def target_for(data,b): return subset(data.train,first_events(data.train,b,CUT))


def base_train(data,bounds):
    frames=[]
    for wi,b in enumerate(bounds):
        f=cached_frame(data,b).copy();f['window_id']=wi;frames.append(f)
    return pd.concat(frames,ignore_index=True)


def downsample_events(frame,rate=.25,seed=20261024):
    rng=np.random.default_rng(seed)
    keys=frame['window_id'].to_numpy(np.int64)*10_000_000+frame['event_index'].to_numpy(np.int64)
    y=frame['label'].to_numpy()
    # Keep the original deterministic policy: group keys are sorted and the
    # RNG is consumed only for all-negative events, exactly as the first
    # experiment run did.
    grouped=pd.DataFrame({'key':keys,'y':y}).groupby('key',sort=True)['y'].max()
    values=grouped.to_numpy()
    keep=(values==1)
    negative=~keep
    keep[negative]=rng.random(int(negative.sum()))<rate
    keep_keys=grouped.index.to_numpy()[keep]
    mask=np.isin(keys,keep_keys)
    out=frame.loc[mask].reset_index(drop=True)
    out['sampling_weight']=1.0
    return out


def fit_fast(train,val,weights=None,seed=20261024):
    cols=feature_columns(train)
    model=lgb.LGBMClassifier(objective='binary',n_estimators=1200,learning_rate=.1,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.0,random_state=seed,n_jobs=4,verbosity=-1)
    model.fit(train[cols],train['label'].astype(int),sample_weight=weights,eval_set=[(val[cols],val['label'].astype(int))],eval_metric='auc',callbacks=[lgb.early_stopping(50,verbose=False)])
    pred=model.predict_proba(val[cols],num_iteration=model.best_iteration_)[:,1]
    return model,cols,pred,float(roc_auc_score(val['label'],pred))


def importance_weights(train,eval_frame):
    def bins(f):
        h=pd.cut(f['history_length'],[-1,0,9,19,20],labels=['0','1-9','10-19','20']).astype(str);g=pd.cut(f['gap_hours'],[-np.inf,6,24,72,168,np.inf],labels=['<6h','6-24h','1-3d','3-7d','>7d']).astype(str);return np.asarray(h+'|'+g)
    tk=bins(train);ek=bins(eval_frame);ts=pd.Series(tk).value_counts(normalize=True);es=pd.Series(ek).value_counts(normalize=True);w=np.ones(len(train),np.float32)
    for key,share in es.items():
        m=tk==key
        if m.any():w[m]=float(share/max(ts.get(key,1e-12),1e-12))
    return w


def shrink_extra(data,target,boundary,k=10.0,uncertainty=False):
    st=build_stats(data,boundary); s=fit_svd(st['click_log'],data.n_users,data.n_items,11); out=[]
    for e in range(target.n_events):
        h=target.history_items[e];h=h[h>=0]; n=len(h); w=n/(n+k) if n else 0.0;uv=s.user_vector(int(target.user_id[e]),h);hv=s.item_vectors[h].mean(axis=0) if len(h) else np.zeros(s.rank);u=w*uv+(1-w)*hv;iv=s.item_vectors[target.candidates[e]];dot=iv@u;cos=dot/((np.linalg.norm(iv,axis=1)+1e-8)*(np.linalg.norm(u)+1e-8));std=1/np.sqrt(n+k);out.extend(np.c_[dot,cos,np.full(10,np.linalg.norm(u)),np.full(10,std),np.abs(dot)*std])
    a=np.asarray(out,dtype=np.float32)
    result={'shrink_dot':a[:,0],'shrink_cos':a[:,1],'shrink_norm':a[:,2]}
    # C1 is the point estimate. C2 explicitly adds uncertainty columns; the
    # previous implementation returned them for both modes, which made C2
    # identical to C1 and prevented a valid audit.
    if uncertainty:
        result.update({'posterior_std':a[:,3],'posterior_utility_std':a[:,4]})
    return pd.DataFrame(result)


def ease_extra(data,target,boundary,lam=500.0):
    st=build_stats(data,boundary);log=st['click_log']; users={}
    for u,i,t in log: users.setdefault(int(u),set()).add(int(i))
    degree={};pairs={}
    for items in users.values():
        # Exact EASE needs an item-item inverse over the complete catalogue.
        # For the laptop experiment use the same bounded history signal the
        # production model can observe: at most the 20 most recent unique
        # clicks per user.  This keeps the pair table sparse while retaining
        # the useful sequential/co-click signal.
        a=list(items)[-20:]
        for i in a:degree[i]=degree.get(i,0)+1
        for i in a:
            for j in a:
                if i!=j:pairs[(i,j)]=pairs.get((i,j),0)+1
    scores=[]
    for e in range(target.n_events):
        h=target.history_items[e];h=h[h>=0];vals=[]
        for c in target.candidates[e]:
            c=int(c);v=0.0
            for src in h:
                v+=pairs.get((int(src),c),0.0)/(degree.get(int(src),0.0)+lam)
            vals.append(v)
        scores.extend(vals)
    return pd.DataFrame({'ease_score':np.asarray(scores,dtype=np.float32)})


def add_choice(train,val):
    ch=fit_choice(train,maxiter=80);u,p,n=ch.predict(train);train=train.copy();train['choice_probability']=p;train['choice_utility']=u;train['choice_no_click']=np.repeat(n,10);u,p,n=ch.predict(val);val=val.copy();val['choice_probability']=p;val['choice_utility']=u;val['choice_no_click']=np.repeat(n,10);return train,val


def run_fold(data,b,train_bounds,eval_frame,method='base',k=10.0,lam=500.0,weighted=False):
    train=base_train(data,train_bounds);val=cached_frame(data,b).copy();
    if method in ('c1','c2'):
        train_extra=[];val_extra=[]
        for wi,bb in enumerate(train_bounds):
            train_extra.append(shrink_extra(data,target_for(data,bb),bb,k=k,uncertainty=(method=='c2')))
        for bb in train_bounds:pass
        # concatenate extras in the same window order
        extra_train=pd.concat(train_extra,ignore_index=True);extra_val=shrink_extra(data,target_for(data,b),b,k=k,uncertainty=(method=='c2'));train=pd.concat([train.reset_index(drop=True),extra_train],axis=1);val=pd.concat([val.reset_index(drop=True),extra_val],axis=1)
    elif method=='c3':
        train_e=[]
        for bb in train_bounds:train_e.append(ease_extra(data,target_for(data,bb),bb,lam))
        train=pd.concat([train.reset_index(drop=True),pd.concat(train_e,ignore_index=True)],axis=1);val=pd.concat([val.reset_index(drop=True),ease_extra(data,target_for(data,b),b,lam)],axis=1)
    if method in ('c1','c2'):
        train,val=add_choice(train,val)
    sample=downsample_events(train)
    w=importance_weights(sample,eval_frame) if weighted else None
    _,_,pred,auc=fit_fast(sample,val,weights=w)
    return {'auc':auc,'reweight':weighted_auc(val,pred,eval_frame),'buckets':bucket_auc(val,pred)}


def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'));eval_frame=cached_frame(data,CUT,'eval');results={}
    # A3 half-day fold experiment mode.
    for name,day in [('DM23',3),('DM24',4)]:
        b=D0+day*DAY;bounds=[D0+int(i*DAY/2) for i in range(1,2*day)]
        results.setdefault('A3',{})[name]=run_fold(data,b,bounds,eval_frame,weighted=True);print('A3',name,results['A3'][name],flush=True)
    results['A3']['DM_mean']=float((results['A3']['DM23']['reweight']['joint_auc_supported']+results['A3']['DM24']['reweight']['joint_auc_supported'])/2)
    for method in ['base','c1','c2','c3']:
        key=method;results[key]={}
        for name,day in [('DM23',3),('DM24',4)]:
            b=D0+day*DAY;bounds=[D0+i*DAY for i in range(1,day)];kwargs={'method':method,'eval_frame':eval_frame}
            results[key][name]=run_fold(data,b,bounds,eval_frame,**{k:v for k,v in kwargs.items() if k!='eval_frame'});print(method,name,results[key][name],flush=True)
        results[key]['DM_mean']=float((results[key]['DM23']['reweight']['joint_auc_supported']+results[key]['DM24']['reweight']['joint_auc_supported'])/2)
    (OUTPUT_DIR/'phase_c_real_results.json').write_text(json.dumps(results,indent=2,default=float));print(json.dumps(results,indent=2,default=float))
if __name__=='__main__':main()
