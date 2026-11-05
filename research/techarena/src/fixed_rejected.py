"""DM22/23/24 rejected scorers rebuilt with the canonical boundary-safe click log."""
from __future__ import annotations
import json, math, time
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .dataset import join_global, click_log_before
from .resume_handover import CACHE, CUT, D0, DAY, OUT, read_parquet, weighted_dm, safe_auc

def first_events(ev,start,end):
    m=(ev.timestamp>=start)&(ev.timestamp<end); seen=set(); out=[]
    for i in np.flatnonzero(m):
        u=int(ev.user_id[i])
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out,np.int64)

def pair_neighbours(seqs,kind,B,topk=50):
    pairs=defaultdict(float); out=defaultdict(float)
    for seq in seqs.values():
        for _,a in seq: out[a]+=1
        for i,(ta,a) in enumerate(seq):
            for j in range(i+1,min(len(seq),i+6)):
                tb,b=seq[j]; dt=tb-ta
                if a==b: continue
                if kind=='CV_recent': w=1.0+3.0*max(0,min(1,(tb-D0)/max(1,B-D0)))
                elif kind=='CV_seq': w=1.0/(j-i)
                else: w=1.0
                pairs[(a,b)]+=w; pairs[(b,a)]+=w
    rows=defaultdict(list)
    for (a,b),w in pairs.items(): rows[a].append((b,w/max(1e-8,out[a])))
    return {a:sorted(v,key=lambda q:q[1],reverse=True)[:topk] for a,v in rows.items()}

def aggregate(cands,hist,target,neigh,agg):
    out=np.zeros((len(target),10),np.float32)
    for z,e in enumerate(target):
        h=hist[e]; valid=np.flatnonzero(h>=0); pos={int(x):j for j,x in enumerate(cands[e])}; vals=np.zeros(10); mx=np.zeros(10)
        for kk,k in enumerate(valid):
            item=int(h[k]); rw=0.8**max(0,len(valid)-1-kk) if agg=='recency' else 1.0
            for b,w in neigh.get(item,[]):
                j=pos.get(int(b))
                if j is None: continue
                ww=w*rw; vals[j]+=ww; mx[j]=max(mx[j],ww)
        out[z]=mx if agg=='max' else vals
    return out.reshape(-1)

def slate_frequency(cands,target,n_items):
    sl=cands[target]; freq=np.bincount(sl.reshape(-1),minlength=n_items).astype(np.float64)/max(1,len(sl)); return freq[sl.reshape(-1)].astype(np.float32)

def eval_one(frame,p,evalf):
    light=(frame['history_length']>=1)&(frame['history_length']<=9)
    return {'DM24':float(weighted_dm(frame,p,evalf)),'raw_auc':float(safe_auc(frame['label'],p)),'light':float(safe_auc(frame['label'][light],p[light]))}

def main():
    data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz')); allfold={}
    for name,day in [('DM22',2),('DM23',3),('DM24',4)]:
        B=D0+day*DAY; target=first_events(data.train,B,CUT); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); log=click_log_before(data,B)
        seqs=defaultdict(list)
        for u,it,ts in log: seqs[int(u)].append((int(ts),int(it)))
        scores={}; times={}
        for kind in ('CV_seq','CV_recent'):
            t=time.time(); neigh=pair_neighbours(seqs,kind,B); best=None
            for agg in ('sum','max','recency'):
                p=aggregate(data.train.candidates,data.train.history_items,target,neigh,agg); dm=weighted_dm(frame,p,evalf)
                if best is None or dm>best[0]: best=(dm,agg,p)
            scores[kind]=best[2]; times[kind]=time.time()-t
        scores['slate_frequency']=slate_frequency(data.train.candidates,target,data.n_items)
        scores['surprise']=frame['svd_cos']-frame['impression_svd_cos']; times['slate_frequency']=0.; times['surprise']=0.
        allfold[name]={'frame':frame,'evalf':evalf,'scores':scores,'times':times,'log_rows':len(log)}
        print(name,{k:round(eval_one(frame,v,evalf)['DM24'],6) for k,v in scores.items()},flush=True)
    rows={}
    for s in allfold['DM24']['scores']:
        v24=eval_one(allfold['DM24']['frame'],allfold['DM24']['scores'][s],allfold['DM24']['evalf']); vals=[eval_one(allfold[k]['frame'],allfold[k]['scores'][s],allfold[k]['evalf'])['DM24'] for k in ('DM22','DM23','DM24')]
        corr=float(spearmanr(allfold['DM24']['scores'][s],allfold['DM24']['frame']['svd_cos']).statistic)
        rows[s]={**v24,'DM_mean3':float(np.mean(vals)),'corr_svd':corr,'time_sec':allfold['DM24']['times'][s]}
    out={'boundary':D0+4*DAY,'click_log_rows':allfold['DM24']['log_rows'],'scorers':rows,'fold_dm24':{k:{s:eval_one(v['frame'],v['scores'][s],v['evalf'])['DM24'] for s in v['scores']} for k,v in allfold.items()},'note':'all click-log scorers use click_log_before; slate_frequency and surprise are label-free/frame-native'}
    (OUT/'fixed_rejected_results.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
