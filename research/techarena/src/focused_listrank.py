"""Focused event-aware listwise ranker with an explicit no-click row.

Each event becomes an 11-choice group: ten candidates plus a synthetic slot 0
(no click). All event context is built from train events before B, while row
features come from the existing SAFE_v2 feature frames.
"""
from __future__ import annotations
import argparse,json,time
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .phase_c_revision import add_choice_pair, add_family, bounds_for, cached_frame, C2_COLS, C1_COLS
from .resume_handover import feature_names, matrix, weighted_dm, read_parquet, safe_auc
from .run_phase1 import first_events


def event_context(data, frame, boundary):
    n=len(frame)//10; a={c:frame[c].to_numpy().reshape(n,10) for c in frame.columns if c in frame}
    out={}
    def stat(name, fn, default=0.0):
        x=a.get(name)
        return np.asarray(fn(x,axis=1) if x is not None else np.full(n,default),dtype=np.float32)
    for name in ('svd_cos','history_mean_cos','history_max_cos','item_ctr','transition_mean','transition_max','choice_utility','choice_probability','choice_no_click','history_age_days','history_count'):
        out[f'evt_{name}_max']=stat(name,np.max); out[f'evt_{name}_mean']=stat(name,np.mean); out[f'evt_{name}_std']=stat(name,np.std)
    if 'choice_utility' in a:
        s=np.sort(a['choice_utility'],axis=1); out['evt_choice_top1']=s[:,-1].astype(np.float32); out['evt_choice_top2']=s[:,-2].astype(np.float32); out['evt_choice_gap']=(s[:,-1]-s[:,-2]).astype(np.float32)
        p=np.clip(a['choice_probability'],1e-8,1); out['evt_choice_entropy']=(-p*np.log(p)).sum(1).astype(np.float32)
    out['evt_history_length']=a['history_length'][:,0].astype(np.float32)
    out['evt_gap_hours']=a['gap_hours'][:,0].astype(np.float32)
    # Time-safe user/session state: only labelled train events strictly before B.
    by=defaultdict(list)
    m=data.train.timestamp<int(boundary)
    for e in np.flatnonzero(m): by[int(data.train.user_id[e])].append((int(data.train.timestamp[e]),int(data.train.labels[e].sum()>0)))
    for u in by: by[u].sort()
    user=a['user_id'][:,0].astype(np.int64); ts=a['timestamp'][:,0].astype(np.int64)
    vals={k:[] for k in ('user_rate','user_events','last_clicked','no_click_streak','prev_gap_h','last_click_gap_h','events_1h','events_6h','events_24h')}
    for u,t in zip(user,ts):
        hist=by.get(int(u),[]); times=np.asarray([q[0] for q in hist],dtype=np.int64); ys=np.asarray([q[1] for q in hist],dtype=np.int8); cnt=len(ys); vals['user_events'].append(cnt); vals['user_rate'].append(float(ys.mean()) if cnt else 0.0); vals['last_clicked'].append(int(ys[-1]) if cnt else 0); streak=0
        for y in ys[::-1]:
            if y: break
            streak+=1
        vals['no_click_streak'].append(streak); prev=int(times[-1]) if cnt else int(boundary); vals['prev_gap_h'].append(max(0,int(t)-prev)/3600.0); clicks=times[ys==1]; vals['last_click_gap_h'].append(max(0,int(t)-int(clicks[-1]))/3600.0 if len(clicks) else 1e3)
        vals['events_1h'].append(int(((times>=int(t)-3600)&(times<int(boundary))).sum())); vals['events_6h'].append(int(((times>=int(t)-6*3600)&(times<int(boundary))).sum())); vals['events_24h'].append(int(((times>=int(t)-24*3600)&(times<int(boundary))).sum()))
    for k,v in vals.items(): out['evt_'+k]=np.asarray(v,dtype=np.float32)
    out['evt_outside_logit']=np.log(np.clip(1-out['evt_user_rate'],1e-5,1)/np.clip(out['evt_user_rate'],1e-5,1)).astype(np.float32)
    return pd.DataFrame(out)

def prepare_frame(frame, context, row_cols):
    x=matrix(frame,row_cols); ev=np.repeat(context.to_numpy(dtype=np.float32),10,axis=0); return np.c_[x,ev,np.zeros((len(x),1),dtype=np.float32)]

def make_groups(frame, context, row_cols, include_no):
    x=prepare_frame(frame,context,row_cols); y=frame['label'].to_numpy(np.float32); n=len(frame)//10; groups=[11]*n
    if not include_no: return x,y,groups
    nox=np.zeros((n,x.shape[1]),dtype=np.float32); nox[:,-1]=1.0; nox[:,:context.shape[1]]=0.0; nox[:,len(row_cols):len(row_cols)+context.shape[1]]=context.to_numpy(dtype=np.float32)
    yy=(frame['label'].to_numpy().reshape(n,10).sum(1)==0).astype(np.float32); xx=np.empty((n,11,x.shape[1]),dtype=np.float32); xx[:,:10]=x.reshape(n,10,-1); xx[:,10]=nox
    return xx.reshape(-1,x.shape[1]),np.c_[y.reshape(n,10),yy].reshape(-1),groups

def build_fold(data, boundary, family='a3_c1_c2'):
    bs=bounds_for(int(round((boundary-D0)/DAY)),half=True); parts=[]
    for wi,b in enumerate(bs):
        f=add_family(data,cached_frame(data,b).copy(),b,family); f['window_id']=wi; parts.append(f)
    train=pd.concat(parts,ignore_index=True); val=add_family(data,cached_frame(data,boundary).copy(),boundary,family); train,val,_=add_choice_pair(train,val)
    # Context must align with each individual window. Rebuild it from the
    # corresponding row slices after choice features have been added.
    trctx=[]; start=0
    for f,b in zip(parts,bs):
        n=len(f); trctx.append(event_context(data,train.iloc[start:start+n],b)); start+=n
    train_ctx=pd.concat(trctx,ignore_index=True); val_ctx=event_context(data,val,boundary)
    row_cols=[c for c in feature_names(train) if c not in {'window_id'}]
    return train,val,train_ctx,val_ctx,row_cols

def fit_fold(data,boundary,eval_frame,trees=700):
    train,val,trctx,vctx,row_cols=build_fold(data,boundary); Xtr,ytr,groups=make_groups(train,trctx,row_cols,True); Xv,yv,vgroups=make_groups(val,vctx,row_cols,True)
    model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',label_gain=[0.0,1.0],n_estimators=trees,learning_rate=.03,num_leaves=31,min_child_samples=300,reg_lambda=10.0,random_state=20261041,n_jobs=4,verbosity=-1)
    model.fit(Xtr,ytr,group=groups,eval_set=[(Xv,yv)],eval_group=[vgroups],callbacks=[lgb.early_stopping(80,verbose=False)])
    pred_all=model.predict(Xv,num_iteration=model.best_iteration_); pred=pred_all.reshape(-1,11)[:,:10].reshape(-1); dm=weighted_dm(val,pred,eval_frame); auc=safe_auc(val['label'].to_numpy(),pred); light=safe_auc(val['label'].to_numpy()[(val['history_length']>=1)&(val['history_length']<=9)],pred[(val['history_length']>=1)&(val['history_length']<=9)])
    return {'dm':float(dm),'auc':auc,'light':light,'best_iteration':int(model.best_iteration_),'pred':pred,'val':val,'model':model,'row_cols':row_cols,'train_rows':len(train),'groups':len(groups)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--folds',default='DM22,DM23,DM24'); ap.add_argument('--trees',type=int,default=700); args=ap.parse_args(); data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); eval_frame=read_parquet(CACHE_DIR/'features_orig'/'evaluation.parquet'); out={'model':'11-choice LightGBM LambdaRank with explicit no-click row','folds':{},'device':'CPU'}
    for name,day in [('DM22',2),('DM23',3),('DM24',4)]:
        if name not in args.folds.split(','): continue
        t=time.time(); r=fit_fold(data,D0+day*DAY,eval_frame,args.trees); out['folds'][name]={k:v for k,v in r.items() if k not in {'pred','val','model','row_cols'}}; np.save(OUTPUT_DIR/f'focused_listrank_{name}.npy',r['pred']); print(name,out['folds'][name], 'sec',time.time()-t,flush=True)
    vals=[out['folds'][k]['dm'] for k in ('DM22','DM23','DM24') if k in out['folds']]; out['DM_mean3']=float(np.mean(vals)) if len(vals)==3 else None; (OUTPUT_DIR/'focused_listrank_results.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
