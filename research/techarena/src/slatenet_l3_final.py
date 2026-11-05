"""Final SlateNet L3 attempt: trainable 11-D item embeddings with SVD anchor.

The state is a mean-pool SVD L2 model. L3 unfreezes item vectors and adds an
L2 penalty toward the SVD initialization while learning the 11-way choice.
"""
from __future__ import annotations
import json,time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import D0,DAY,CUT
from .dataset import join_global, click_log_before
from .resume_handover import CACHE,OUT,read_parquet,weighted_dm,safe_auc
from .svd_model import fit_svd

B=D0+4*DAY
RANK=11
BATCH=2048
EPOCHS=2
LAMBDAS=(1e-3,1e-2)

def first_events(ev,start,end):
    m=(ev.timestamp>=start)&(ev.timestamp<end); seen=set(); out=[]
    for i in np.flatnonzero(m):
        u=int(ev.user_id[i])
        if u not in seen: seen.add(u); out.append(int(i))
    return np.asarray(out,np.int64)

def make_arrays(events,indices):
    idx=np.asarray(indices,np.int64); return {'c':events.candidates[idx].astype(np.int64),'h':events.history_items[idx].astype(np.int64),'y':events.labels[idx].astype(np.float32),'u':events.user_id[idx].astype(np.int64),'ts':events.timestamp[idx].astype(np.int64),'hlen':events.history_length[idx].astype(np.int64)}

def choice_target(y):
    cls=np.zeros(len(y),dtype=np.int64); clicked=y.sum(1)>0
    cls[clicked]=np.argmax(y[clicked],axis=1)+1
    return cls

def l2_scores(events,indices,item):
    arr=make_arrays(events,indices); h=arr['h']; mask=h>=0; hc=np.maximum(h,0); hv=item[hc]; hm=mask[:,:,None].astype(np.float32); u=(hv*hm).sum(1)/np.maximum(hm.sum(1),1.0); cv=item[arr['c']]; return np.sum(cv*u[:,None,:],axis=2).reshape(-1).astype(np.float32)

class MeanPoolL3(nn.Module):
    def __init__(self,item_init):
        super().__init__(); self.item=nn.Embedding.from_pretrained(torch.as_tensor(item_init,dtype=torch.float32),freeze=False); self.bias=nn.Embedding(item_init.shape[0],1); nn.init.zeros_(self.bias.weight); self.no_click=nn.Parameter(torch.tensor(0.0)); self.no_len=nn.Parameter(torch.tensor(0.0))
    def forward(self,c,h):
        mask=(h>=0).float(); hv=self.item(torch.clamp(h,min=0)); u=(hv*mask.unsqueeze(-1)).sum(1)/(mask.sum(1,keepdim=True)+1e-6); cv=self.item(c); logits=(cv*u.unsqueeze(1)).sum(-1)+self.bias(c).squeeze(-1); no=self.no_click+self.no_len*(mask.sum(1)/20.0); return logits,no

def predict(model,arr,device):
    model.eval(); out=[]
    with torch.no_grad():
        for s in range(0,len(arr['c']),BATCH):
            q=slice(s,min(s+BATCH,len(arr['c']))); c=torch.as_tensor(arr['c'][q],device=device); h=torch.as_tensor(arr['h'][q],device=device); logits,no=model(c,h); out.append(logits.cpu().numpy())
    return np.concatenate(out).reshape(-1).astype(np.float32)

def run(lam):
    data=join_global(load_cache(CACHE/'train.npz'),load_cache(CACHE/'evaluation.npz')); log=click_log_before(data,B); svd=fit_svd(log,data.n_users,data.n_items,RANK); item_init=svd.item_vectors.astype(np.float32); m=data.train.timestamp<B; all_idx=np.flatnonzero(m); order=all_idx[np.argsort(data.train.timestamp[all_idx],kind='stable')]; split=int(len(order)*.8); train_idx=order[:split]; hold_idx=order[split:]; target_idx=first_events(data.train,B,CUT); train=make_arrays(data.train,train_idx); hold=make_arrays(data.train,hold_idx); target=make_arrays(data.train,target_idx); frame=read_parquet(CACHE/'features_orig'/f'{B}.parquet'); evalf=read_parquet(CACHE/'features_orig'/'evaluation.parquet'); device=torch.device('cpu'); model=MeanPoolL3(item_init).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=0.0); cls=torch.as_tensor(choice_target(train['y']),dtype=torch.long,device=device); hcls=torch.as_tensor(choice_target(hold['y']),dtype=torch.long,device=device); base=l2_scores(data.train,target_idx,item_init); rows=[]; rng=np.random.default_rng(20261024+int(lam*100000))
    base_dm=weighted_dm(frame,base,evalf)
    for ep in range(1,EPOCHS+1):
        model.train(); perm=rng.permutation(len(train['c'])); losses=[]; t=time.time()
        for s in range(0,len(perm),BATCH):
            ii=perm[s:s+BATCH]; c=torch.as_tensor(train['c'][ii],device=device); h=torch.as_tensor(train['h'][ii],device=device); ycls=cls[ii]; logits,no=model(c,h); ce=F.cross_entropy(torch.cat([no[:,None],logits],1),ycls); penalty=lam*torch.mean((model.item.weight-torch.as_tensor(item_init,device=device))**2); loss=ce+penalty; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach()))
        dm_pred=predict(model,target,device); dm=weighted_dm(frame,dm_pred,evalf); hold_pred=predict(model,hold,device); held_auc=roc_auc_score(hold['y'].reshape(-1),hold_pred); rows.append({'epoch':ep,'dm24':float(dm),'delta_vs_l2':float(dm-base_dm),'heldout_train_auc':float(held_auc),'train_loss':float(np.mean(losses)),'epoch_sec':time.time()-t}); print('L3',lam,rows[-1],flush=True)
    return {'lambda':lam,'l2_dm24':float(base_dm),'l2_auc':float(safe_auc(frame['label'],base)),'history':rows}

def main():
    out={'boundary':B,'rank':RANK,'batch_events':BATCH,'epochs':EPOCHS,'device':'cpu','results':{}}
    for lam in LAMBDAS: out['results'][str(lam)]=run(lam)
    (OUT/'slatenet_l3_final_results.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
