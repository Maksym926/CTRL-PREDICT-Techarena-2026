"""Train and compare R1/R2 protocol regimes on a materialized DM24 table."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .resume_handover import read_parquet,weighted_dm,safe_auc
from .run_phase1 import first_events
from .cache import load_cache
from .dataset import join_global

DROP={'event_index','candidate_index','user_id','timestamp','label'}
PARAMS={'objective':'binary','metric':'auc','learning_rate':0.03,'num_leaves':63,'min_data_in_leaf':300,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':1,'lambda_l2':10.0,'num_threads':8,'verbosity':-1}

def cell(h,g):
 hs=np.where(h<=0,'0',np.where(h<=9,'1-9',np.where(h<=19,'10-19','20'))); gs=np.where(g<6,'<6h',np.where(g<24,'6-24h',np.where(g<72,'1-3d',np.where(g<168,'3-7d','>7d')))); return np.char.add(np.char.add(hs,'|'),gs)
def cols(df): return [c for c in df.columns if c not in DROP]
def sample_mask(df,all_event_random):
 y=df['label'].to_numpy(np.int8); ev=df['event_index'].to_numpy(np.int64); clicked=pd.Series(y).groupby(ev,sort=False).transform('max').to_numpy().astype(bool); rand=np.asarray([all_event_random.get(int(e),0.0) for e in ev]); return clicked|(rand<0.35)
def user_weights(df,mask,evalf):
 work=df.loc[mask].copy(); ev=work['event_index'].to_numpy(np.int64); u=work['user_id'].to_numpy(np.int64); cnt=pd.Series(ev).groupby(u).nunique(); w=np.asarray([1.0/max(1,int(cnt.get(int(x),1))) for x in u],np.float32)
 # Evaluation history-share correction; gap cells are reported but training rows
 # before B have negative B-relative gaps, so unsupported gap cells receive 0.
 hk=cell(work['history_length'].to_numpy(),work['gap_hours'].to_numpy()); eh=evalf['history_length'] if isinstance(evalf,dict) else evalf['history_length'].to_numpy(); eg=evalf['gap_hours'] if isinstance(evalf,dict) else evalf['gap_hours'].to_numpy(); ek=cell(eh,eg); es=pd.Series(ek).value_counts(normalize=True); vs=pd.Series(hk).value_counts(normalize=True)
 for k,share in es.items(): w[hk==k]*=float(share/max(vs.get(k,1e-12),1e-12))
 return w

def train_regime(name,train,val,evalf,randoms,trees=1600):
 mask=sample_mask(train,randoms); c=cols(train); X=train.loc[mask,c].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32); y=train.loc[mask,'label'].to_numpy(np.int8); Xv=val[c].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32); yv=val['label'].to_numpy(np.int8); w=user_weights(train,mask,evalf); dtrain=lgb.Dataset(X,label=y,weight=w,feature_name=c,params={'max_bin':63},free_raw_data=True); dval=lgb.Dataset(Xv,label=yv,reference=dtrain,feature_name=c,params={'max_bin':63},free_raw_data=True); binpath=CACHE_DIR/'protocol'/f'{name}_DM24.bin'; dtrain.save_binary(binpath); model=lgb.train(PARAMS,dtrain,num_boost_round=trees,valid_sets=[dval],valid_names=['val'],callbacks=[lgb.early_stopping(100,verbose=False)]); p=model.predict(Xv,num_iteration=model.best_iteration); dm=weighted_dm(val,p,evalf); light=safe_auc(yv[(val['history_length'].to_numpy()>=1)&(val['history_length'].to_numpy()<=9)],p[(val['history_length'].to_numpy()>=1)&(val['history_length'].to_numpy()<=9)]); longm=val['gap_hours'].to_numpy()>24; long=safe_auc(yv[longm],p[longm]); result={'regime':name,'dm24':float(dm),'raw_auc':float(roc_auc_score(yv,p)),'light':light,'long_gap':long,'train_rows':int(mask.sum()),'train_events':int(train.loc[mask,'event_index'].nunique()),'best_iteration':int(model.best_iteration),'features':c,'sampled_nonclick_rate':0.35,'binary_dataset':str(binpath)}; np.save(OUTPUT_DIR/f'protocol_{name}_DM24.npy',p); return result

def main():
 data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); B=D0+4*DAY; train=pd.read_parquet(CACHE_DIR/'protocol'/f'train_all_{B}.parquet'); val=pd.read_parquet(CACHE_DIR/'protocol'/f'val_{B}.parquet'); evalf=read_parquet(CACHE_DIR/'features_orig'/'evaluation.parquet'); events=np.sort(train['event_index'].unique()); rng=np.random.default_rng(20261024); randoms={int(e):float(x) for e,x in zip(events,rng.random(len(events)))}
 # R2: same random draw, restricted to first-event windows before B.
 selected=set();
 for b in [D0+int(i*DAY/2) for i in range(1,2*4)]: selected.update(map(int,first_events(data.train,b,CUT)))
 r2=train[train['event_index'].isin(selected)].copy(); t=time.time(); r1=train_regime('R1_all_events',train,val,evalf,randoms); print(r1,flush=True); r2m=train_regime('R2_first_event',r2,val,evalf,randoms); print(r2m,flush=True)
 # Distribution report for taste score.
 dist={}
 for label,df in [('R1_sample',train.loc[sample_mask(train,randoms)]),('R2_sample',r2.loc[sample_mask(r2,randoms)]),('validation',val),('evaluation',evalf)]: dist[label]={'mean':float(df['svd_cos'].mean()),'std':float(df['svd_cos'].std()),'rows':int(len(df))}
 out={'boundary':B,'R0_reference':{'DM22':0.7859872506,'DM23':0.7959620121,'DM24':0.8055218787,'DM_mean3':0.7958237138},'R1':r1,'R2':r2m,'svd_cos_distribution':dist,'time_sec':time.time()-t}; (OUTPUT_DIR/'protocol_stage1_results.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
