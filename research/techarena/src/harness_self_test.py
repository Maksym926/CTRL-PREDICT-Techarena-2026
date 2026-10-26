"""Decisive DM24 SVD parity test for the bake-off harness."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .resume_handover import CACHE, CUT, D0, DAY, OUT, read_parquet, weighted_dm
from .cache import load_cache
from .dataset import join_global, click_log_before
from .svd_model import fit_svd

B = D0 + 4 * DAY

def first_events(ev,start,end):
    m=(ev.timestamp>=start)&(ev.timestamp<end); seen=set(); out=[]
    for i in np.flatnonzero(m):
        u=int(ev.user_id[i])
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out,np.int64)

def main():
    train_raw=np.load(CACHE/'train.npz',allow_pickle=True); eval_raw=np.load(CACHE/'evaluation.npz',allow_pickle=True)
    data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz'))
    target=first_events(data.train,B,CUT); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet')
    log=click_log_before(data,B); svd=fit_svd(log,data.n_users,data.n_items,rank=11)
    p=np.zeros(len(target)*10,np.float32)
    for z,e in enumerate(target):
        h=data.train.history_items[e]; h=h[h>=0]; u=svd.user_vector(int(data.train.user_id[e]),h); iv=svd.item_vectors[data.train.candidates[e]]; p[z*10:(z+1)*10]=(iv@u)/((np.linalg.norm(iv,axis=1)+1e-8)*(np.linalg.norm(u)+1e-8))
    cached=frame['svd_cos'].astype(np.float32); y=frame['label'].astype(np.int8)
    corr=float(spearmanr(p,cached).statistic); auc=float(weighted_dm(frame,p,evalf)); cached_auc=float(weighted_dm(frame,cached,evalf));
    # Print three target events with original opaque identifiers.
    keys_i=np.unique(np.concatenate([train_raw['item_keys'],eval_raw['item_keys']])); keys_u=np.unique(np.concatenate([train_raw['user_keys'],eval_raw['user_keys']]))
    rows=[]
    for z,e in enumerate(target[:3]):
        ukey=int(train_raw['user_keys'][train_raw['user_id'][e]]); ckeys=[int(train_raw['item_keys'][x]) for x in train_raw['candidates'][e]]; hm=train_raw['history_items'][e]>=0; hkeys=[int(train_raw['item_keys'][x]) for x in train_raw['history_items'][e][hm]]
        rows.append({'event':z,'user_id':ukey,'candidate_item_ids':ckeys,'history_item_ids':hkeys,'labels':y[z*10:(z+1)*10].tolist(),'harness_score':p[z*10:(z+1)*10].round(6).tolist(),'cached_svd_cos':cached[z*10:(z+1)*10].round(6).tolist()})
    out={'boundary':B,'click_log_rows':int(len(log)),'harness_weighted_auc':auc,'cached_weighted_auc':cached_auc,'auc_delta':auc-cached_auc,'spearman_cached':corr,'pass_auc':abs(auc-cached_auc)<=.003,'pass_corr':corr>.99,'events':rows}
    (OUT/'harness_self_test.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))

if __name__=='__main__': main()
