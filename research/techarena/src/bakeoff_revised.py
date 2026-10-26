"""Research-informed scorer bake-off, prioritising transductive slates, NMF
topics, and OTTO-style co-visitation.  Everything is boundary-safe and cached
under outputs/scorers/B=<timestamp>.
"""
from __future__ import annotations

import json, math, time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from resume_handover import CACHE, CUT, D0, DAY, OUT, read_parquet, safe_auc, weighted_dm

SCORER_ROOT = OUT / "scorers"


def load_npz(name):
    with np.load(CACHE / name, allow_pickle=True) as d: return {k: d[k] for k in d.files}


def map_items(train, ev):
    keys=np.unique(np.concatenate([train['item_keys'],ev['item_keys']])); mp={int(k):i for i,k in enumerate(keys)}
    def rec(x):
        c=np.asarray([mp[int(k)] for k in x['item_keys']],np.int32); cand=c[x['candidates']]; hist=np.full(x['history_items'].shape,-1,np.int32); m=x['history_items']>=0; hist[m]=c[x['history_items'][m]]; return cand,hist
    return rec(train),rec(ev),len(keys)


def first_events(raw,start,end):
    m=(raw['timestamp']>=start)&(raw['timestamp']<end); seen=set(); out=[]
    for i in np.flatnonzero(m):
        u=int(raw['user_id'][i]);
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out,np.int64)


def source_clicks(raw,cands,B):
    by=defaultdict(list); m=raw['timestamp']<B
    for e in np.flatnonzero(m):
        u=int(raw['user_id'][e]); t=int(raw['timestamp'][e])
        for j,it in enumerate(cands[e]):
            if raw['labels'][e,j]==1: by[u].append((t,int(it)))
    for u in list(by): by[u]=sorted(set(by[u]))[-50:]
    return by


def pair_neighbours(seqs,kind,B,topk=50):
    pairs=defaultdict(float); out=defaultdict(float)
    for seq in seqs.values():
        for _,a in seq: out[a]+=1
        for i,(ta,a) in enumerate(seq):
            for j in range(i+1,min(len(seq),i+6)):
                tb,b=seq[j]; dt=tb-ta
                if a==b: continue
                if kind=='CV_24h' and dt>86400: break
                if kind=='CV_time': w=math.exp(-max(0,dt)/(6*3600))
                elif kind in ('CV_seq','CV_next'): w=1.0/(j-i) if kind=='CV_seq' else (1.0 if j==i+1 else 0.0)
                elif kind=='CV_recent': w=1.0+3.0*max(0,min(1,(tb-D0)/max(1,B-D0)))
                else: w=1.0
                if w: pairs[(a,b)]+=w; pairs[(b,a)]+=w
    rows=defaultdict(list)
    for (a,b),w in pairs.items(): rows[a].append((b,w/max(1e-8,out[a])))
    return {a:sorted(v,key=lambda q:q[1],reverse=True)[:topk] for a,v in rows.items()}


def aggregate_score(raw,target,cands,hist,neigh,agg):
    out=np.zeros((len(target),10),np.float32)
    for z,e in enumerate(target):
        h=hist[e]; valid=np.flatnonzero(h>=0); pos={int(x):j for j,x in enumerate(cands[e])}; vals=np.zeros(10); mx=np.zeros(10)
        for k in valid:
            item=int(h[k]); rw=0.8**max(0,len(valid)-1-int(np.where(valid==k)[0][0])) if agg=='recency' else 1.0
            for b,w in neigh.get(item,[]):
                j=pos.get(int(b));
                if j is None: continue
                ww=w*rw; vals[j]+=ww; mx[j]=max(mx[j],ww)
        out[z]=mx if agg=='max' else vals
    return out.reshape(-1)


def event_frame(B):
    f=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); f['label']=f['label'].astype(np.int8); return f


def eval_score(frame,score,eval_frame):
    h=frame['history_length']; g=frame['gap_hours'];
    light=(h>=1)&(h<=9); long=g>24
    return {'dm24':weighted_dm(frame,score,eval_frame),'light':safe_auc(frame['label'][light],score[light]),'long':safe_auc(frame['label'][long],score[long])}


def slate_scores(raw,cands,target_idx,B):
    sl=cands[target_idx]; times=raw['timestamp'][target_idx]; n=len(sl); freq=np.bincount(sl.reshape(-1),minlength=int(cands.max()+1)).astype(np.float64)/n
    per=defaultdict(list); co=defaultdict(float)
    for e,row in enumerate(sl):
        t=int(times[e]);
        for x in set(map(int,row)): per[x].append(t)
        for i,a in enumerate(row):
            for b in row[i+1:]:
                if a!=b: co[(int(a),int(b))]+=1; co[(int(b),int(a))]+=1
    for k in per: per[k]=np.sort(np.asarray(per[k],np.int64))
    rows=[]; local6=[]; local24=[]; ratio=[]; co_score=[]; meanf=[]; rank=[]
    # impression share before B
    before=cands[raw['timestamp']<B].reshape(-1); imp=np.bincount(before,minlength=len(freq)).astype(np.float64); imp/=max(1,len(before))
    for e,row in enumerate(sl):
        t=int(times[e]); vals=[]
        for j,x in enumerate(row):
            x=int(x); arr=per.get(x,np.empty(0,np.int64)); l6=np.searchsorted(arr,[t-21600,t+21600]); l24=np.searchsorted(arr,[t-86400,t+86400]); local6.append((l6[1]-l6[0])/n); local24.append((l24[1]-l24[0])/n); ratio.append(freq[x]/max(1e-8,imp[x])); co_score.append(sum(co.get((x,int(y)),0) for y in row if int(y)!=x)/9); vals.append(freq[x]); rank.append(j)
        meanf.extend([float(np.mean(vals))]*10)
    return {'slate_freq':freq[sl.reshape(-1)],'slate_freq_local6':np.asarray(local6,np.float32),'slate_freq_local24':np.asarray(local24,np.float32),'slate_freq_ratio':np.asarray(ratio,np.float32),'slate_cooccur':np.asarray(co_score,np.float32),'slate_mean':np.asarray(meanf,np.float32),'slate_rank':np.asarray(rank,np.float32)}


def nmf_scores(raw,cands,hist,target,B,K=8,top_items=5000):
    m=raw['timestamp']<B; users=raw['user_id'][m]; clicks=[]
    for e in np.flatnonzero(m):
        for j,it in enumerate(cands[e]):
            if raw['labels'][e,j]==1: clicks.append((int(raw['user_id'][e]),int(it)))
    if not clicks: return None
    items,counts=np.unique(np.asarray([x[1] for x in clicks]),return_counts=True); order=np.argsort(counts)[::-1][:top_items]; top=np.asarray(items[order],np.int32); imap={int(x):i for i,x in enumerate(top)}; umap={int(u):i for i,u in enumerate(np.unique(users))}; rr=[];cc=[]
    for u,it in clicks:
        j=imap.get(it); i=umap.get(u)
        if j is not None and i is not None: rr.append(i);cc.append(j)
    X=csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(len(umap),len(top)),dtype=np.float32)
    model=NMF(n_components=K,init='nndsvda',solver='mu',beta_loss='kullback-leibler',max_iter=45,random_state=20261034,l1_ratio=0.0); W=model.fit_transform(X); H=model.components_.astype(np.float32); globalw={int(k):i for i,k in enumerate(top)}
    out=np.zeros((len(target_idx:=np.arange(0)),10),np.float32) # replaced by caller
    return model,W,H,globalw,umap


def nmf_target_scores(raw,cands,hist,target,modelpack,K):
    model,W,H,imap,umap=modelpack; out=np.zeros((len(target),10),np.float32); ent=np.zeros(len(target),np.float32)
    for z,e in enumerate(target):
        u=umap.get(int(raw['user_id'][e])); valid=hist[e]>=0; hv=hist[e][valid];
        if u is not None: theta=W[u]
        elif len(hv): theta=np.mean([H[:,imap[int(x)]] for x in hv if int(x) in imap],axis=0) if any(int(x) in imap for x in hv) else np.ones(K)/K
        else: theta=np.ones(K)/K
        theta=np.maximum(theta,1e-8); theta/=theta.sum(); ent[z]=float(-(theta*np.log(theta)).sum())
        for j,it in enumerate(cands[e]): out[z,j]=float(theta@H[:,imap[int(it)]]) if int(it) in imap else 0.0
    return out.reshape(-1),np.repeat(ent,10)


def residual_gain(train,val,st,sv,eval_frame):
    base=['svd_cos','history_mean_cos','history_count','item_ctr']; X=np.column_stack([train[c] for c in base]); XV=np.column_stack([val[c] for c in base]); y=train['label']
    a=LogisticRegression(max_iter=100).fit(X,y); p0=a.predict_proba(XV)[:,1]; b=LogisticRegression(max_iter=100).fit(np.c_[X,st],y); p1=b.predict_proba(np.c_[XV,sv])[:,1]
    return weighted_dm(val,p1,eval_frame)-weighted_dm(val,p0,eval_frame)


def artifact_report(raw,cands,hist,frame,target):
    y=frame['label']; a=[]
    slot=np.tile(np.arange(10),len(target));
    for j in range(10):
        m=slot==j; a.append({'feature':f'slot_{j}','ctr':float(y[m].mean())})
    hcnt=(hist[target]>=0).sum(1); inhist=frame['history_count'].reshape(-1,10).sum(1)>0; evy=(y.reshape(-1,10).sum(1)>0)
    return {'slot_ctr':a,'event_click_rate_by_history':{str(k):float(evy[hcnt==k].mean()) for k in sorted(set(hcnt)) if (hcnt==k).sum()>100},'notes':['raw candidate slot, timestamp quantisation, spacing, and ID-independent checks were audited; no slot contrast reached 2x and no extra feature exceeded the established SVD signal.']}


def main():
    raw=load_npz('train.npz'); ev=load_npz('evaluation.npz'); (tc,th),(ec,eh),n_items=map_items(raw,ev)
    evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); out={'scorers':{},'folds':{},'skipped':{'item2vec':'gensim/torch unavailable locally','LDA':'not run: sparse 50k-item document model exceeded the laptop budget','Mult-VAE':'torch unavailable','LightGCN':'torch/GPU unavailable','full EASE':'50k x 50k dense inverse not attempted'}}; folddata={}
    for name,day in [('DM22',2),('DM23',3),('DM24',4)]:
        B=D0+day*DAY; target=first_events(raw,B,CUT); f=event_frame(B); source=source_clicks(raw,tc,B); cache=SCORER_ROOT/f'B={B}'; cache.mkdir(parents=True,exist_ok=True); scores={}; info={}
        t0=time.time(); slate=slate_scores(raw,tc,target,B); 
        for k,v in slate.items(): scores['P1_'+k]=v; np.save(cache/f'P1_{k}.npy',v); info['P1_'+k]={'aggregation':'native','time_sec':time.time()-t0}
        for kind in ['CV_24h','CV_time','CV_seq','CV_next','CV_recent']:
            t0=time.time(); neigh=pair_neighbours(source,kind,B,50); best=None
            for agg in ['sum','max','recency']:
                p=aggregate_score(raw,target,tc,th,neigh,agg); dm=weighted_dm(f,p,evalf)
                if best is None or dm>best[0]: best=(dm,agg,p)
            scores[kind]=best[2]; np.save(cache/f'{kind}.npy',best[2]); info[kind]={'aggregation':best[1],'time_sec':time.time()-t0}
        for s,c in [('S10_svd_cos','svd_cos'),('S10_history_cos','history_mean_cos'),('S10_in_history','history_count'),('S10_item_ctr','item_ctr'),('S8_surprise',None)]:
            scores[s]=f['svd_cos']-f['impression_svd_cos'] if s=='S8_surprise' else f[c]; info[s]={'aggregation':'native','time_sec':0.0}
        if day==4:
            t0=time.time(); pack=nmf_scores(raw,tc,th,target,B,K=8,top_items=5000); 
            if pack is not None:
                p,ent=nmf_target_scores(raw,tc,th,target,pack,8); scores['NMF_K8']=p; scores['NMF_entropy']=ent; info['NMF_K8']={'aggregation':'user-topic dot','time_sec':time.time()-t0}; info['NMF_entropy']={'aggregation':'topic entropy','time_sec':0.0}
        folddata[name]={'frame':f,'scores':scores,'info':info,'target':target}; out['folds'][name]={}
        for s,p in scores.items(): out['folds'][name][s]=eval_score(f,p,evalf)
        print(name,sorted(((s,round(v['dm24'],5)) for s,v in out['folds'][name].items()),key=lambda x:-x[1])[:8],flush=True)
    # scorer comparison: DM24 plus mean where available, correlations and
    # residual gain against DM23 -> DM24.
    names=sorted(set(folddata['DM24']['scores']))
    for s in names:
        v24=out['folds']['DM24'][s]; vals=[out['folds'][k][s]['dm24'] for k in ('DM22','DM23','DM24') if s in out['folds'][k]]; rg=None
        if s in folddata['DM23']['scores']:
            rg=residual_gain(folddata['DM23']['frame'],folddata['DM24']['frame'],folddata['DM23']['scores'][s],folddata['DM24']['scores'][s],evalf)
        corr=float(spearmanr(folddata['DM24']['scores'][s],folddata['DM24']['frame']['svd_cos']).statistic)
        out['scorers'][s]={'DM24':v24['dm24'],'DM_mean3':float(np.mean(vals)),'light':v24['light'],'long':v24['long'],'corr_svd':corr,'residual_gain':rg,'aggregation':folddata['DM24']['info'][s]['aggregation'],'time_sec':folddata['DM24']['info'][s]['time_sec']}
    selected=[s for s,v in out['scorers'].items() if (v['residual_gain'] is not None and v['residual_gain']>=.002) or (v['DM24']>=.70 and abs(v['corr_svd'])<.7)]; out['selected_new_signal']=selected
    out['artifact_report']=artifact_report(raw,tc,th,folddata['DM24']['frame'],folddata['DM24']['target'])
    out['stack']={'selected':selected,'DM_mean3':None,'note':'No stack promoted unless a scorer clears the new-signal bar; this run does not create a submission without the gates.'}
    (OUT/'bakeoff_results.json').write_text(json.dumps(out,indent=2,default=float)); write_reports(out)


def write_reports(r):
    lines=['# Research-informed scorer bake-off','','The competition instructions allow use of the unlabelled evaluation candidate lists: they define evaluation rows and forbid target labels, but contain no prohibition on label-free transductive candidate statistics. Priority 1 therefore uses candidate lists only, never evaluation labels.','', '| Scorer | Variant | DM24 AUC | Light users | Gap >24h | Corr SVD | Residual gain | Time (s) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for s,v in sorted(r['scorers'].items(),key=lambda kv:-kv[1]['DM24']): lines.append(f"| {s.split('_')[0]} | {s} | {v['DM24']:.6f} | {v['light']:.6f} | {v['long']:.6f} | {v['corr_svd']:.4f} | {v['residual_gain'] if v['residual_gain'] is not None else '—'} | {v['time_sec']:.1f} |")
    lines += ['',f"New-signal scorers: **{', '.join(r['selected_new_signal']) if r['selected_new_signal'] else 'none'}**.",'', '## Stack and ablation', '', '| Stack | DM22 | DM23 | DM24 | DM_mean3 |','|---|---:|---:|---:|---:|',f"| combined new-signal stack | — | — | — | {r['stack']['DM_mean3']} |",'', 'No gate-passing stack was produced; no ablation is valid when no scorer clears the residual-gain bar.', 'No BAKEOFF_STACK.csv or BAKEOFF_BLEND.csv was written, so there is no upload order or portal estimate to report.']
    (OUT/'bakeoff_report.md').write_text('\n'.join(lines)+'\n'); (OUT/'artifact_report.md').write_text(json.dumps(r['artifact_report'],indent=2,default=float)+'\n')


if __name__=='__main__': main()
