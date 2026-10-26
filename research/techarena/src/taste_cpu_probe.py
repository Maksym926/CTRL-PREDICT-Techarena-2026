"""CPU fallback probe for all-item taste estimators after project venv setup."""
from __future__ import annotations
import json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from resume_handover import CACHE, CUT, D0, DAY, OUT, read_parquet, weighted_dm, safe_auc

B = D0 + 4 * DAY

def load(name):
    with np.load(CACHE / name, allow_pickle=True) as d: return {k:d[k] for k in d.files}

def first_events(raw,start,end):
    m=(raw['timestamp']>=start)&(raw['timestamp']<end); seen=set(); out=[]
    for i in np.flatnonzero(m):
        u=int(raw['user_id'][i])
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out,np.int64)

def setup():
    tr=load('train.npz'); ev=load('evaluation.npz'); keys=np.unique(np.concatenate([tr['item_keys'],ev['item_keys']])); im={int(k):i for i,k in enumerate(keys)}; users=np.unique(np.concatenate([tr['user_keys'],ev['user_keys']])); um={int(k):i for i,k in enumerate(users)}
    ti=np.asarray([um[int(tr['user_keys'][u])] for u in tr['user_id']],np.int64); cand=np.asarray([im[int(k)] for k in tr['item_keys']],np.int32)[tr['candidates']]; hist=np.full(tr['history_items'].shape,-1,np.int32); m=tr['history_items']>=0; ids=np.asarray([im[int(k)] for k in tr['item_keys']],np.int32); hist[m]=ids[tr['history_items'][m]]
    target=first_events(tr,B,CUT); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet')
    rows=[]; cols=[]
    seq=defaultdict(list)
    for e in np.flatnonzero(tr['timestamp']<B):
        u=int(ti[e]); t=int(tr['timestamp'][e])
        for j,it in enumerate(cand[e]):
            if tr['labels'][e,j]==1: rows.append(u); cols.append(int(it)); seq[u].append((t,int(it)))
    X=csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(users),len(keys)),dtype=np.float32)
    for u in seq: seq[u]=[it for _,it in sorted(seq[u])]
    return tr,ev,ti,cand,hist,target,frame,evalf,X,seq,im,um

def score_matrix(frame,tr,cand,hist,target,um,im,item_scores,user_scores):
    n=len(target); out=np.full((n,10),np.nan,np.float32); eligible=np.zeros((n,10),bool)
    for z,e in enumerate(target):
        u=int(um[int(tr['user_keys'][tr['user_id'][e]])]);
        for j,it in enumerate(cand[e]):
            if user_scores is not None and u < user_scores.shape[0]: out[z,j]=float(user_scores[u]@item_scores[:,int(it)]); eligible[z,j]=True
    return out.reshape(-1),eligible.reshape(-1)

def weighted_eligible(frame,p,mask,evalf):
    ff={k:v[mask] for k,v in frame.items() if hasattr(v,'__len__') and len(v)==len(mask)}
    return weighted_dm(ff,p[mask],evalf)

def main():
    tr,ev,ti,cand,hist,target,frame,evalf,X,seq,im,um=setup(); result={'device':'cpu','boundary':B,'estimators':{}}
    for K in (8,):
        t=time.time(); model=NMF(n_components=K,init='nndsvda',solver='mu',beta_loss='kullback-leibler',max_iter=35,random_state=20261040); W=model.fit_transform(X); H=model.components_.astype(np.float32); p,elig=score_matrix(frame,tr,cand,hist,target,um,im,H,W); m=elig & np.isfinite(p); result['estimators'][f'NMF_K{K}']={'dm24':weighted_eligible(frame,p,m,evalf),'raw_auc_eligible':safe_auc(frame['label'][m],p[m]),'coverage':float(m.mean()),'corr_svd':float(spearmanr(p[m],frame['svd_cos'][m]).statistic),'time_sec':time.time()-t}
        np.save(OUT/f'nmf_dm24_k{K}.npy',p)
    # LDA uses the same all-item sparse documents; keep the first K for the CPU budget.
    t=time.time(); lda=LatentDirichletAllocation(n_components=8,learning_method='online',batch_size=4096,max_iter=5,random_state=20261041,n_jobs=4,doc_topic_prior=0.1); W=lda.fit_transform(X); H=lda.components_.astype(np.float32); H/=np.maximum(H.sum(1,keepdims=True),1e-8); p,elig=score_matrix(frame,tr,cand,hist,target,um,im,H,W); m=elig & np.isfinite(p); result['estimators']['LDA_K8']={'dm24':weighted_eligible(frame,p,m,evalf),'raw_auc_eligible':safe_auc(frame['label'][m],p[m]),'coverage':float(m.mean()),'corr_svd':float(spearmanr(p[m],frame['svd_cos'][m]).statistic),'time_sec':time.time()-t}; np.save(OUT/'lda_dm24_k8.npy',p)
    # Gensim SGNS item2vec on source click sequences.
    try:
        from gensim.models import Word2Vec
        t=time.time(); sentences=[[str(x) for x in v] for v in seq.values() if len(v)>=2]; w2v=Word2Vec(sentences=sentences,vector_size=32,window=5,negative=10,min_count=1,workers=4,sg=1,epochs=10,seed=20261042); iv=w2v.wv; out=np.full((len(target),10),np.nan,np.float32); elig=np.zeros((len(target),10),bool)
        for z,e in enumerate(target):
            hv=[iv[str(int(x))] for x in hist[e] if x>=0 and str(int(x)) in iv];
            if not hv: continue
            hvec=np.asarray(hv); mean=hvec.mean(0); last=hvec[-1]
            for j,it in enumerate(cand[e]):
                if str(int(it)) not in iv: continue
                v=iv[str(int(it))]; nv=np.linalg.norm(v)+1e-8; out[z,j]=float(np.max((hvec@v)/(np.linalg.norm(hvec,axis=1)*nv+1e-8))); elig[z,j]=True
        m=elig.reshape(-1); p=out.reshape(-1); result['estimators']['item2vec_max']={'dm24':weighted_eligible(frame,p,m,evalf),'raw_auc_eligible':safe_auc(frame['label'][m],p[m]),'coverage':float(m.mean()),'corr_svd':float(spearmanr(p[m],frame['svd_cos'][m]).statistic),'time_sec':time.time()-t}
        np.save(OUT/'item2vec_dm24.npy',p)
    except Exception as e: result['estimators']['item2vec_max']={'skipped':repr(e)}
    (OUT/'taste_cpu_probe.json').write_text(json.dumps(result,indent=2,default=float)); print(json.dumps(result,indent=2,default=float))

if __name__=='__main__': main()
