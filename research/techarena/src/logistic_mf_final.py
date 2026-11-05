"""Final CPU logistic matrix-factorisation attempt for the TechArena harness.

The model is deliberately boundary-safe: impressions come only from labelled
train events before B; SVD and C1 initialisation use click_log_before(data, B).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from .cache import load_cache
from .config import D0, DAY, CUT
from .dataset import join_global, click_log_before
from .resume_handover import CACHE, OUT, read_parquet, weighted_dm, safe_auc, feature_names, matrix
from .svd_model import fit_svd

RANK=11
LAMBDAS=(1.0,10.0,100.0)
CHUNK=100_000

def first_events(ev,start,end):
    mask=(ev.timestamp>=start)&(ev.timestamp<end); seen=set(); out=[]
    for i in np.flatnonzero(mask):
        u=int(ev.user_id[i])
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out,dtype=np.int64)

def sigmoid(x):
    x=np.clip(x,-35.0,35.0)
    return 1.0/(1.0+np.exp(-x))

def make_rows(data,boundary,recent_days=None):
    m=data.train.timestamp<int(boundary)
    if recent_days is not None:
        m &= data.train.timestamp >= int(boundary) - int(recent_days * DAY)
    e=np.flatnonzero(m)
    users=np.repeat(data.train.user_id[e].astype(np.int32),10)
    items=data.train.candidates[e].reshape(-1).astype(np.int32)
    labels=data.train.labels[e].reshape(-1).astype(np.float64)
    return users,items,labels

def init_c1(data,boundary,log,svd):
    d=svd.rank
    sums=np.zeros((data.n_users,d),dtype=np.float64)
    counts=np.bincount(log[:,0].astype(np.int64),minlength=data.n_users).astype(np.float64)
    np.add.at(sums,log[:,0].astype(np.int64),svd.item_vectors[log[:,1].astype(np.int64)].astype(np.float64))
    hmean=sums/np.maximum(counts[:,None],1.0)
    w=counts/(counts+10.0)
    u0=w[:,None]*svd.user_vectors.astype(np.float64)+(1.0-w[:,None])*hmean
    return u0.astype(np.float64),svd.item_vectors.astype(np.float64).copy()

def accumulate(A,rhs,ids,X,w,z):
    d=X.shape[1]
    outer=X[:,:,None]*X[:,None,:]
    np.add.at(A,ids,outer*w[:,None,None])
    np.add.at(rhs,ids,X*(w*z)[:,None])

def solve_update(A,rhs,init,lam):
    d=rhs.shape[1]
    A[:,:d-1,:d-1]+=lam*np.eye(d-1,dtype=np.float64)[None,:,:]
    rhs[:,:d-1]+=lam*init
    A[:,-1,-1]+=1e-3
    return np.linalg.solve(A,rhs[...,None])[...,0]

def fit_model(data,boundary,lam,alternations=3,rank=RANK,recent_days=None):
    log=click_log_before(data,int(boundary)); svd=fit_svd(log,data.n_users,data.n_items,rank=rank)
    users,items,y=make_rows(data,boundary,recent_days=recent_days); u0,v0=init_c1(data,boundary,log,svd)
    U=u0.copy(); V=v0.copy(); cu=np.zeros(data.n_users,dtype=np.float64); bi=np.zeros(data.n_items,dtype=np.float64)
    d=rank+1; t0=time.time(); n=len(y)
    for alt in range(alternations):
        # User Newton/IRLS update for (u_u, c_u), batched over all users.
        A=np.zeros((data.n_users,d,d),dtype=np.float64); rhs=np.zeros((data.n_users,d),dtype=np.float64)
        for s in range(0,n,CHUNK):
            q=slice(s,min(s+CHUNK,n)); sc=np.sum(U[users[q]]*V[items[q]],axis=1)+cu[users[q]]+bi[items[q]]; p=sigmoid(sc); w=np.maximum(p*(1-p),1e-5); z=sc+(y[q]-p)/w; X=np.empty((len(sc),d),dtype=np.float64); X[:,:rank]=V[items[q]]; X[:,-1]=1.0
            accumulate(A,rhs,users[q],X,w,z)
        sol=solve_update(A,rhs,u0,lam); U=sol[:,:rank]; cu=sol[:,-1]
        # Item Newton/IRLS update for (v_i, b_i), batched over all items.
        A=np.zeros((data.n_items,d,d),dtype=np.float64); rhs=np.zeros((data.n_items,d),dtype=np.float64)
        for s in range(0,n,CHUNK):
            q=slice(s,min(s+CHUNK,n)); sc=np.sum(U[users[q]]*V[items[q]],axis=1)+cu[users[q]]+bi[items[q]]; p=sigmoid(sc); w=np.maximum(p*(1-p),1e-5); z=sc+(y[q]-p)/w; X=np.empty((len(sc),d),dtype=np.float64); X[:,:rank]=U[users[q]]; X[:,-1]=1.0
            # item prior is lambda*(v-v_init)^2; bias has a small ridge.
            accumulate(A,rhs,items[q],X,w,z)
        sol=solve_update(A,rhs,v0,lam); V=sol[:,:rank]; bi=sol[:,-1]
        print(f'lambda={lam:g} alternation={alt+1}/{alternations} rows={n} elapsed={time.time()-t0:.1f}s',flush=True)
    return {'U':U.astype(np.float32),'V':V.astype(np.float32),'cu':cu.astype(np.float32),'bi':bi.astype(np.float32),'u0':u0.astype(np.float32),'v0':v0.astype(np.float32),'log_rows':int(len(log)),'fit_sec':time.time()-t0}

def score(data,target,model,cosine=False):
    U,V,cu,bi=model['U'],model['V'],model['cu'],model['bi']; out=np.zeros(len(target)*10,dtype=np.float32)
    for z,e in enumerate(target):
        u=U[int(data.train.user_id[e])]; v=V[data.train.candidates[e]]
        if cosine:
            out[z*10:(z+1)*10]=(v@u)/((np.linalg.norm(v,axis=1)+1e-8)*(np.linalg.norm(u)+1e-8))
        else:
            out[z*10:(z+1)*10]=v@u+bi[data.train.candidates[e]]+cu[int(data.train.user_id[e])]
    return out

def residual_gain(tr,va,st,sv,evalf):
    cols=feature_names(tr); X=matrix(tr,cols); XV=matrix(va,cols); y=tr['label'].astype(np.int8)
    base=lgb.LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,reg_lambda=10.0,random_state=20261025,n_jobs=4,verbosity=-1)
    base.fit(X,y); p0=base.predict_proba(XV)[:,1]; dm0=weighted_dm(va,p0,evalf)
    ext=lgb.LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,reg_lambda=10.0,random_state=20261026,n_jobs=4,verbosity=-1)
    ext.fit(np.c_[X,st],y); p1=ext.predict_proba(np.c_[XV,sv])[:,1]; dm1=weighted_dm(va,p1,evalf)
    return float(dm1-dm0),float(dm0),float(dm1)

def main():
    data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz')); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); folds={}
    for name,day in [('DM22',2),('DM23',3),('DM24',4)]:
        B=D0+day*DAY; target=first_events(data.train,B,CUT); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); folds[name]={'B':B,'target':target,'frame':frame}
    results={'rank':RANK,'lambdas':list(LAMBDAS),'alternations':3,'folds':{},'environment':'CPU (MPS unavailable)','models':{}}
    for lam in LAMBDAS:
        key=f'lambda_{lam:g}'; results['models'][key]={}
        for name in ('DM22','DM23','DM24'):
            B=folds[name]['B']; model=fit_model(data,B,lam,3); folds[name].setdefault('models',{})[key]=model
            fr=folds[name]['frame']; target=folds[name]['target']; dot=score(data,target,model,False); cos=score(data,target,model,True)
            results['models'][key][name]={'dot':{'dm':float(weighted_dm(fr,dot,evalf)),'raw_auc':float(safe_auc(fr['label'],dot)),'light':float(safe_auc(fr['label'][(fr['history_length']>=1)&(fr['history_length']<=9)],dot[(fr['history_length']>=1)&(fr['history_length']<=9)])),'corr_svd':float(spearmanr(dot,fr['svd_cos']).statistic)},'cosine':{'dm':float(weighted_dm(fr,cos,evalf)),'raw_auc':float(safe_auc(fr['label'],cos)),'light':float(safe_auc(fr['label'][(fr['history_length']>=1)&(fr['history_length']<=9)],cos[(fr['history_length']>=1)&(fr['history_length']<=9)])),'corr_svd':float(spearmanr(cos,fr['svd_cos']).statistic)},'fit_sec':model['fit_sec'],'log_rows':model['log_rows']}
            np.savez_compressed(OUT/f'logistic_mf_{key}_{name}.npz',dot=dot,cosine=cos)
            print(key,name,results['models'][key][name],flush=True)
        for score_name in ('dot','cosine'):
            vals=[results['models'][key][n][score_name]['dm'] for n in ('DM22','DM23','DM24')]
            results['models'][key][score_name+'_DM_mean3']=float(np.mean(vals))
    # residual gains use DM23 as training fold and DM24 as validation fold.
    tr=folds['DM23']['frame']; va=folds['DM24']['frame']; results['residuals']={}
    for key in results['models']:
        results['residuals'][key]={}
        for score_name in ('dot','cosine'):
            st=np.load(OUT/f'logistic_mf_{key}_DM23.npz')[score_name]; sv=np.load(OUT/f'logistic_mf_{key}_DM24.npz')[score_name]
            rg,dm0,dm1=residual_gain(tr,va,st,sv,evalf); results['residuals'][key][score_name]={'base_dm24':dm0,'with_dm24':dm1,'gain':rg}
            print('residual',key,score_name,rg,flush=True)
    (OUT/'logistic_mf_final_results.json').write_text(json.dumps(results,indent=2,default=float)); print(json.dumps(results,indent=2,default=float))
if __name__=='__main__': main()
