"""Build Stage-1 protocol features for the final TechArena hypothesis.

R1 uses five user-fold SVD item spaces fit from clicks before B excluding the
held-out user's fold. Popularity and exposure statistics are updated online at
each row timestamp. The output is chunked float32 Parquet for bounded memory.
"""
from __future__ import annotations
import argparse, json, math, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global,click_log_before
from .run_phase1 import first_events
from .svd_model import fit_svd,fit_impression_svd

FOLDS=5; RANKS=(11,32); CHUNK_EVENTS=50_000

def fit_crossfold(data,boundary):
    log=click_log_before(data,boundary); uf=(np.arange(data.n_users,dtype=np.int64)%FOLDS); models={}
    for r in RANKS:
        models[r]=[]
        for f in range(FOLDS):
            t=time.time(); keep=uf[log[:,0]]!=f; m=fit_svd(log[keep],data.n_users,data.n_items,rank=r); models[r].append(m.item_vectors.astype(np.float32)); print('crossfit SVD',r,f,'rows',int(keep.sum()),'sec',round(time.time()-t,1),flush=True)
    t=time.time(); imp=fit_impression_svd(data.train,boundary,data.n_users,data.n_items,rank=10); print('impression SVD sec',round(time.time()-t,1),flush=True)
    return models,imp.item_vectors.astype(np.float32),int(len(log))

def hist_for(events,e,t):
    h=events.history_items[e]; ht=events.history_times[e]; m=(h>=0)&(ht<int(t)); return h[m].astype(np.int32),ht[m].astype(np.int64)

def safe_cos(v,u):
    return (v@u)/((np.linalg.norm(v,axis=1)+1e-8)*(np.linalg.norm(u)+1e-8))

def make_scores(cands,h,vec):
    if len(h): u=vec[h].mean(0); w=len(h)/(len(h)+10.0); u=(w*u).astype(np.float32)
    else: u=np.zeros(vec.shape[1],np.float32)
    iv=vec[cands]; dot=iv@u; cos=dot/((np.linalg.norm(iv,axis=1)+1e-8)*(np.linalg.norm(u)+1e-8))
    if len(h):
        hv=vec[h]; pair=iv@hv.T/((np.linalg.norm(iv,axis=1)[:,None]+1e-8)*(np.linalg.norm(hv,axis=1)[None,:]+1e-8)); weights=np.power(.8,np.arange(len(h)-1,-1,-1,dtype=np.float32)); weighted=(pair@weights)/weights.sum()
        mean=pair.mean(1); mx=pair.max(1); last=pair[:,-1]
    else: mean=mx=last=weighted=np.zeros(len(cands),np.float32)
    return dot.astype(np.float32),cos.astype(np.float32),mean.astype(np.float32),mx.astype(np.float32),last.astype(np.float32),weighted.astype(np.float32),u

class Online:
    def __init__(self,n_users,n_items):
        self.item_imp=np.zeros(n_items,np.int32); self.item_click=np.zeros(n_items,np.int32); self.user_imp=np.zeros(n_users,np.int32); self.user_click=np.zeros(n_users,np.int32); self.user_events=[[] for _ in range(n_users)]; self.user_click_times=[[] for _ in range(n_users)]; self.shown=defaultdict(int); self.clicked=defaultdict(int); self.last_shown={}; self.recent={1:deque(),3:deque(),7:deque()}; self.recent_counts={1:np.zeros(n_items,np.int32),3:np.zeros(n_items,np.int32),7:np.zeros(n_items,np.int32)}; self.trans=defaultdict(float); self.trans_out=defaultdict(float)
    def ev_features(self,u,t,cands,labels=None):
        hist=self.user_events[u]; ts=np.asarray([x[0] for x in hist],np.int64) if hist else np.empty(0,np.int64); ys=np.asarray([x[1] for x in hist],np.int8) if hist else np.empty(0,np.int8); cnt=len(ys); prev=int(ts[-1]) if cnt else int(t); clicks=ts[ys==1] if cnt else np.empty(0,np.int64); streak=0
        for y in ys[::-1]:
            if y: break
            streak+=1
        session_start=prev; session_events=0
        for j in range(len(hist)-1,-1,-1):
            if t-hist[j][0]>1800: break
            session_start=hist[j][0]; session_events+=1
        return {'user_event_count':float(cnt),'user_click_rate':float(ys.mean()) if cnt else 0.0,'last_event_clicked':float(ys[-1]) if cnt else 0.0,'no_click_streak':float(streak),'time_since_prev_event_h':max(0,t-prev)/3600.0,'time_since_last_click_h':max(0,t-int(clicks[-1]))/3600.0 if len(clicks) else 1e3,'session_events':float(session_events),'session_age_h':max(0,t-session_start)/3600.0,'events_1h':float(((ts>=t-3600)&(ts<t)).sum()) if cnt else 0.0,'events_6h':float(((ts>=t-21600)&(ts<t)).sum()) if cnt else 0.0,'events_24h':float(((ts>=t-86400)&(ts<t)).sum()) if cnt else 0.0}
    def row(self,u,t,cands,labels):
        ev=self.ev_features(u,t,cands,labels); gctr=(self.item_click.sum()+1)/(self.item_imp.sum()+20+1e-8); out=[]; hist_items=[]
        for j,c in enumerate(cands):
            c=int(c); imp=int(self.item_imp[c]); clk=int(self.item_click[c]); key=(int(u),c); age=max(0,t-int(self.last_shown.get(key,t))) if key in self.last_shown else 1e9; trans=[]
            for src in hist_items[-20:]: trans.append(self.trans.get((int(src),c),0.0)/max(1.0,self.trans_out.get(int(src),0.0)))
            out.append((imp,clk,(clk+20*gctr)/(imp+20),float(self.user_imp[u]),float(self.user_click[u]),(self.user_click[u]+20*gctr)/(self.user_imp[u]+20),float(key in self.shown),float(key in self.clicked),float(age/3600.0 if age<1e8 else 1e3),float(self.recent_counts[1][c]),float(self.recent_counts[3][c]),float(self.recent_counts[7][c]),float(np.mean(trans) if trans else 0.0),float(np.max(trans) if trans else 0.0)))
            hist_items.append(c)
        return ev,out
    def update(self,u,t,cands,labels):
        prev=[x[1] for x in self.user_events[u][-3:] if x[1]]
        for j,c in enumerate(cands):
            c=int(c); key=(int(u),c); self.item_imp[c]+=1; self.user_imp[u]+=1; self.shown[key]+=1; self.last_shown[key]=int(t)
            if int(labels[j]): self.item_click[c]+=1; self.user_click[u]+=1; self.clicked[key]+=1
            if int(labels[j]):
                for src in prev: self.trans[(int(src),c)]+=1.0; self.trans_out[int(src)]+=1.0
        clicked=int(np.asarray(labels).sum()>0); self.user_events[u].append((int(t),clicked));
        if clicked: self.user_click_times[u].append(int(t))
        # Recent item click windows are updated only with click labels.
        for j,c in enumerate(cands):
            if int(labels[j]):
                for d in (1,3,7): self.recent[d].append((int(t),int(c))); self.recent_counts[d][int(c)]+=1
        for d in (1,3,7):
            q=self.recent[d]; cutoff=int(t)-d*86400
            while q and q[0][0]<cutoff:
                _,c=q.popleft(); self.recent_counts[d][c]-=1

def base_row_names():
    return ['svd_dot','svd_cos','shrink_dot','shrink_cos','shrink_norm','posterior_std','posterior_utility_std','svd32_dot','svd32_cos','history_mean_cos','history_max_cos','history_last_cos','history_weighted_cos','impression_svd_cos','impression_svd_dot','history_count','history_age_days','item_impressions','item_clicks','item_ctr','user_impressions','user_clicks','user_ctr','shown_before','clicked_before','item_age_since_shown_h','item_click_1d','item_click_3d','item_click_7d','transition_mean','transition_max','history_length','gap_hours','event_score_max','event_score_mean','event_score_gap','user_event_count','user_click_rate','last_event_clicked','no_click_streak','time_since_prev_event_h','time_since_last_click_h','session_events','session_age_h','events_1h','events_6h','events_24h','cand_short_minus_long','short_long_cos']

def build_rows(data,events,indices,boundary,models,impvec,online,update,fold_cache=None):
    names=base_row_names(); rows={k:[] for k in ['event_index','candidate_index','user_id','timestamp','label']+names}; model11=models[11]; model32=models[32]
    indices=list(map(int,indices))
    for ii,e in enumerate(indices):
        u=int(events.user_id[e]); t=int(events.timestamp[e]); cands=events.candidates[e].astype(np.int32); labels=events.labels[e].astype(np.int8); h,ht=hist_for(events,e,t); f=u%FOLDS; v11=model11[f]; v32=model32[f]; s11=make_scores(cands,h,v11); s32=make_scores(cands,h,v32); si=make_scores(cands,h,impvec); ev,stats=online.row(u,t,cands,labels)
        hset=set(map(int,h)); short=h[-3:] if len(h) else h; longu=s11[-1] if False else s11[-1]  # retained for explicit naming below
        if len(h): shortv=v11[h[-3:]].mean(0); longv=s11[-1]
        else: shortv=np.zeros(v11.shape[1],np.float32); longv=np.zeros(v11.shape[1],np.float32)
        cand_short=(v11[cands]@shortv)/(np.linalg.norm(v11[cands],axis=1)+1e-8)/(np.linalg.norm(shortv)+1e-8) if len(h) else np.zeros(10,np.float32)
        cand_long=s11[1]; shortlong=float((shortv@longv)/(np.linalg.norm(shortv)+1e-8)/(np.linalg.norm(longv)+1e-8)) if len(h) else 0.0
        for j,c in enumerate(cands):
            imp,clk,ctr,ui,uc,uctr,shown,clicked,age,c1,c3,c7,tm,tx=stats[j]; pstd=1.0/math.sqrt(len(h)+10.0); vals=dict(ev); vals.update({'svd_dot':s11[0][j],'svd_cos':s11[1][j],'shrink_dot':s11[0][j],'shrink_cos':s11[1][j],'shrink_norm':float(np.linalg.norm(s11[-1])),'posterior_std':pstd,'posterior_utility_std':float(abs(s11[0][j])*pstd),'svd32_dot':s32[0][j],'svd32_cos':s32[1][j],'history_mean_cos':s11[2][j],'history_max_cos':s11[3][j],'history_last_cos':s11[4][j],'history_weighted_cos':s11[5][j],'impression_svd_cos':si[1][j],'impression_svd_dot':si[0][j],'history_count':float(np.sum(h==c)),'history_age_days':float((t-int(ht[np.flatnonzero(h==c)[-1]]) if np.any(h==c) else 1e9)/86400.0),'item_impressions':float(imp),'item_clicks':float(clk),'item_ctr':float(ctr),'user_impressions':ui,'user_clicks':uc,'user_ctr':uctr,'shown_before':shown,'clicked_before':clicked,'item_age_since_shown_h':age,'item_click_1d':c1,'item_click_3d':c3,'item_click_7d':c7,'transition_mean':tm,'transition_max':tx,'history_length':float(len(h)),'gap_hours':float((t-boundary)/3600.0),'event_score_max':float(s11[1].max()),'event_score_mean':float(s11[1].mean()),'event_score_gap':float(s11[1][j]-s11[1].max()),'cand_short_minus_long':float(cand_short[j]-cand_long[j]),'short_long_cos':shortlong}); rows['event_index'].append(e); rows['candidate_index'].append(j); rows['user_id'].append(u); rows['timestamp'].append(t); rows['label'].append(int(labels[j])); [rows[k].append(vals[k]) for k in names]
        if update: online.update(u,t,cands,labels)
    return pd.DataFrame(rows)

def write_frame(df,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); table=pa.Table.from_pandas(df,preserve_index=False); pq.write_table(table,path,compression='zstd')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--boundary',type=int,default=D0+4*DAY); args=ap.parse_args(); data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); B=args.boundary; t=time.time(); models,impvec,logrows=fit_crossfold(data,B); print('fit all',time.time()-t,flush=True)
    online=Online(data.n_users,data.n_items); order=np.flatnonzero(data.train.timestamp<B); order=order[np.argsort(data.train.timestamp[order],kind='stable')]; outpath=CACHE_DIR/'protocol'/f'train_all_{B}.parquet'; parts=[]
    for s in range(0,len(order),CHUNK_EVENTS):
        df=build_rows(data,data.train,order[s:s+CHUNK_EVENTS],B,models,impvec,online,True); parts.append(df); print('train chunk',s,len(df),flush=True)
    train=pd.concat(parts,ignore_index=True); write_frame(train,outpath)
    # Validation targets use the state frozen at B and only the first event per user.
    target=first_events(data.train,B,CUT); val=build_rows(data,data.train,target,B,models,impvec,online,False); valpath=CACHE_DIR/'protocol'/f'val_{B}.parquet'; write_frame(val,valpath)
    meta={'boundary':B,'log_rows':logrows,'train_rows':int(len(train)),'train_events':int(len(order)),'val_rows':int(len(val)),'val_events':int(len(target)),'train_path':str(outpath),'val_path':str(valpath),'sec':time.time()-t}; (CACHE_DIR/'protocol'/f'meta_{B}.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
