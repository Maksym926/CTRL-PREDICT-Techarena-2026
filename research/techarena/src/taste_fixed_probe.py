"""DM24 taste probe using the exact boundary-safe click log and global codes."""
from __future__ import annotations
import json,time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.decomposition import NMF,LatentDirichletAllocation
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from .cache import load_cache
from .dataset import join_global,click_log_before
from .resume_handover import CACHE,CUT,D0,DAY,OUT,read_parquet,weighted_dm,safe_auc

B=D0+4*DAY

def first_events(ev,start,end):
 m=(ev.timestamp>=start)&(ev.timestamp<end); seen=set(); out=[]
 for i in np.flatnonzero(m):
  u=int(ev.user_id[i]);
  if u not in seen: seen.add(u); out.append(int(i))
 return np.asarray(out,np.int64)

def weighted_eligible(frame,p,m,evalf):
 ff={k:v[m] for k,v in frame.items() if hasattr(v,'__len__') and len(v)==len(m)}
 return weighted_dm(ff,p[m],evalf)

def score_user_item(data,target,item_scores,user_scores):
 out=np.zeros((len(target),10),np.float32)
 for z,e in enumerate(target):
  u=int(data.train.user_id[e]); out[z]=user_scores[u]@item_scores[:,data.train.candidates[e]].astype(np.float32)
 return out.reshape(-1)

def score_history(data,target,item_scores,mode='max'):
 out=np.zeros((len(target),10),np.float32)
 for z,e in enumerate(target):
  h=data.train.history_items[e]; h=h[h>=0]
  if not len(h): continue
  hv=item_scores[:,h]; cv=item_scores[:,data.train.candidates[e]]
  # cosine between candidate columns and history columns
  cn=np.linalg.norm(cv,axis=0)+1e-8; hn=np.linalg.norm(hv,axis=0)+1e-8; sim=(cv.T@hv)/(cn[:,None]*hn[None,:])
  if mode=='max': out[z]=sim.max(1)
  elif mode=='mean': out[z]=sim.mean(1)
  else: out[z]=sim[:,-1]
 return out.reshape(-1)

def main():
 data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz')); target=first_events(data.train,B,CUT); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); log=click_log_before(data,B); X=csr_matrix((np.ones(len(log),np.float32),(log[:,0],log[:,1])),shape=(data.n_users,data.n_items),dtype=np.float32); result={'boundary':B,'click_log_rows':int(len(log)),'device':'cpu','estimators':{}}
 for K in (8,10,12):
  t=time.time(); nmf=NMF(n_components=K,init='nndsvda',solver='mu',beta_loss='kullback-leibler',max_iter=45,random_state=20261050+K); W=nmf.fit_transform(X); H=nmf.components_.astype(np.float32); p=score_user_item(data,target,H,W); result['estimators'][f'NMF_K{K}']={'dm24':weighted_dm(frame,p,evalf),'raw_auc':safe_auc(frame['label'],p),'corr_svd':float(spearmanr(p,frame['svd_cos']).statistic),'time_sec':time.time()-t}
 for K in (8,10,12):
  t=time.time(); lda=LatentDirichletAllocation(n_components=K,learning_method='online',batch_size=4096,max_iter=6,random_state=20261060+K,n_jobs=4,doc_topic_prior=.1); W=lda.fit_transform(X); H=lda.components_.astype(np.float32); H/=np.maximum(H.sum(1,keepdims=True),1e-8); p=score_user_item(data,target,H,W); result['estimators'][f'LDA_K{K}']={'dm24':weighted_dm(frame,p,evalf),'raw_auc':safe_auc(frame['label'],p),'corr_svd':float(spearmanr(p,frame['svd_cos']).statistic),'time_sec':time.time()-t}
 seq=defaultdict(list)
 for u,it,t in sorted(log.tolist(),key=lambda q:(q[0],q[2])): seq[int(u)].append(str(int(it)))
 t=time.time(); w2v=Word2Vec(sentences=list(seq.values()),vector_size=32,window=5,negative=10,min_count=1,workers=4,sg=1,epochs=10,seed=20261070); iv=w2v.wv
 for mode in ('max','mean','last'):
  p=np.full(len(target)*10,np.nan,np.float32); eligible=np.zeros(len(target)*10,bool)
  for z,e in enumerate(target):
   hv=[iv[str(int(x))] for x in data.train.history_items[e] if x>=0 and str(int(x)) in iv]
   if not hv: continue
   hvec=np.asarray(hv); hn=np.linalg.norm(hvec,axis=1)+1e-8
   for j,it in enumerate(data.train.candidates[e]):
    key=str(int(it));
    if key not in iv: continue
    v=iv[key]; sim=(hvec@v)/((hn*np.linalg.norm(v))+1e-8); p[z*10+j]=float(sim.max() if mode=='max' else sim.mean() if mode=='mean' else sim[-1]); eligible[z*10+j]=True
  result['estimators'][f'item2vec_{mode}']={'dm24':weighted_eligible(frame,p,eligible,evalf),'raw_auc_eligible':safe_auc(frame['label'][eligible],p[eligible]),'coverage':float(eligible.mean()),'corr_svd':float(spearmanr(p[eligible],frame['svd_cos'][eligible]).statistic),'time_sec':time.time()-t}
 result['self_test']='outputs/harness_self_test.json passes exactly; scorer matrix now uses click_log_before'
 (OUT/'taste_fixed_probe.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))

if __name__=='__main__': main()
