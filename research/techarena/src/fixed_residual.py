import json,numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from .fixed_rejected import *
from .taste_fixed_probe import first_events as first_events2

def mat(f,cols): return np.column_stack([np.nan_to_num(f[c].astype(np.float32),nan=0,posinf=0,neginf=0) for c in cols])
def main():
 data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz')); folds={}
 for name,day in [('DM23',3),('DM24',4)]:
  B=D0+day*DAY; target=first_events(data.train,B,CUT); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); log=click_log_before(data,B); seqs=defaultdict(list)
  for u,it,ts in log: seqs[int(u)].append((int(ts),int(it)))
  scores={}
  for kind in ('CV_seq','CV_recent'):
   neigh=pair_neighbours(seqs,kind,B); best=None
   for agg in ('sum','max','recency'):
    p=aggregate(data.train.candidates,data.train.history_items,target,neigh,agg); dm=weighted_dm(frame,p,evalf)
    if best is None or dm>best[0]: best=(dm,p)
   scores[kind]=best[1]
  scores['slate_frequency']=slate_frequency(data.train.candidates,target,data.n_items); scores['surprise']=frame['svd_cos']-frame['impression_svd_cos']; folds[name]=(frame,scores)
 cols=['svd_cos','history_mean_cos','history_count','item_ctr']; tr,sc=folds['DM23']; va,_=folds['DM24']; y=tr['label'].astype(np.int8)
 base=lgb.LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,reg_lambda=10,verbosity=-1,n_jobs=4,random_state=20261025)
 base.fit(mat(tr,cols),y); p0=base.predict_proba(mat(va,cols))[:,1]; dm0=weighted_dm(va,p0,read_parquet(CACHE/'features_orig'/'evaluation.parquet'))
 out={}
 for s in folds['DM24'][1]:
  ext=lgb.LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,reg_lambda=10,verbosity=-1,n_jobs=4,random_state=20261026)
  ext.fit(np.c_[mat(tr,cols),sc[s]],y); p=ext.predict_proba(np.c_[mat(va,cols),folds['DM24'][1][s]])[:,1]; dm=weighted_dm(va,p,read_parquet(CACHE/'features_orig'/'evaluation.parquet')); out[s]={'base_dm24':float(dm0),'with_dm24':float(dm),'residual_gain':float(dm-dm0)}
 print(json.dumps(out,indent=2)); (OUT/'fixed_residual_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
