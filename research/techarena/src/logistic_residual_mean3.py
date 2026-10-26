"""Leakage-safe adjacent-fold LightGBM residual confirmation for logistic MF."""
from __future__ import annotations
import json,numpy as np,lightgbm as lgb
from .cache import load_cache
from .config import D0,DAY,CUT
from .dataset import join_global
from .resume_handover import CACHE,OUT,read_parquet,feature_names,matrix,weighted_dm
from .logistic_mf_final import fit_model,first_events,score

def fit_gain(tr,va,st,sv,evalf):
 cols=feature_names(tr); X=matrix(tr,cols); XV=matrix(va,cols); y=tr['label'].astype(np.int8)
 b=lgb.LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,reg_lambda=10,random_state=20261025,n_jobs=4,verbosity=-1); b.fit(X,y); p0=b.predict_proba(XV)[:,1]; dm0=weighted_dm(va,p0,evalf)
 e=lgb.LGBMClassifier(objective='binary',n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,reg_lambda=10,random_state=20261026,n_jobs=4,verbosity=-1); e.fit(np.c_[X,st],y); p1=e.predict_proba(np.c_[XV,sv])[:,1]; dm1=weighted_dm(va,p1,evalf); return float(dm0),float(dm1),float(dm1-dm0)
def main():
 data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz')); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); lam=10.; bs=[D0+DAY,D0+2*DAY,D0+3*DAY,D0+4*DAY]; scores={}
 # Need the first boundary model; later folds already have saved score arrays.
 m=fit_model(data,bs[0],lam,3); t=first_events(data.train,bs[0],CUT); scores[bs[0]]=score(data,t,m,True); np.savez_compressed(OUT/'logistic_mf_lambda_10_DM21.npz',cosine=scores[bs[0]])
 for b,name in zip(bs[1:],('DM22','DM23','DM24')): scores[b]=np.load(OUT/f'logistic_mf_lambda_10_{name}.npz')['cosine']
 out=[]
 for i in range(3):
  trb, vab=bs[i],bs[i+1]; tr=read_parquet(CACHE/'features_orig'/f'{trb}.parquet'); va=read_parquet(CACHE/'features_orig'/f'{vab}.parquet'); dm0,dm1,g=fit_gain(tr,va,scores[trb],scores[vab],evalf); out.append({'train_boundary':trb,'val_boundary':vab,'base_dm':dm0,'with_dm':dm1,'gain':g}); print(out[-1],flush=True)
 result={'lambda':lam,'transitions':out,'mean3_gain':float(np.mean([x['gain'] for x in out]))}; (OUT/'logistic_mf_lambda10_residual_mean3.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
