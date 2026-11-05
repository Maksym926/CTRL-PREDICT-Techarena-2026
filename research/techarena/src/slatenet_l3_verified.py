"""SlateNet L3: verified SlateNet architecture with trainable rank-11 items.

This starts from the existing SVD-initialized SlateNet path, changes the item
embedding to rank 11, and unfreezes it with an explicit L2 anchor penalty.
"""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .phase_a import weighted_auc
from .resume_handover import weighted_dm,read_parquet
from .svd_model import fit_svd
from .features import build_stats
from .slatenet import _event_features,_dataset_class,_model_class,_arrays_to_torch,_predict_pair,EventArrays
from .run_phase1 import first_events

RANK=11; BATCH=2048; EPOCHS=2; LAMBDAS=(1e-3,1e-2)

def train_fold(data,boundary,lam,epochs=2):
    torch.manual_seed(20261024+int(lam*100000)); np.random.seed(20261024+int(lam*100000)); source=data.train.timestamp<boundary
    train_all=_event_features(data.train,source); split=int(len(train_all.candidates)*.8); train_idx=np.arange(split,dtype=np.int64); hold_idx=np.arange(split,len(train_all.candidates),dtype=np.int64)
    def take(a,ix): return EventArrays(*(getattr(a,n)[ix] for n in ('candidates','history','history_age_days','item_log_clicks','repeat','item_age_days','labels','user_id','timestamp','history_length')))
    train_a=take(train_all,train_idx); hold_a=take(train_all,hold_idx); target_idx=np.asarray(first_events(data.train,boundary,CUT),dtype=np.int64); val_a=_event_features(data.train,source,target_idx)
    torch_,nn_,DataLoader,Dataset=__import__('src.slatenet',fromlist=['_torch'])._torch(); EventDataset=_dataset_class(torch_,Dataset); Model=_model_class(torch_,nn_)
    device=torch_.device('cpu'); model=Model(data.n_items,data.n_users,dim=RANK).to(device); stats=build_stats(data,boundary); svd=fit_svd(stats['click_log'],data.n_users,data.n_items,rank=RANK)
    with torch_.no_grad(): model.item.weight[1:,:].copy_(torch_.as_tensor(svd.item_vectors,device=device))
    init=torch_.as_tensor(svd.item_vectors,device=device); opt=torch_.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-5)
    loader=DataLoader(EventDataset(train_a),batch_size=BATCH,shuffle=True,num_workers=0); history=[]
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for batch in loader:
            c,h,ha,il,r,ia,y,uid=_arrays_to_torch(torch_,batch,device); opt.zero_grad(set_to_none=True); logits,no,_=model(c,h,ha,il,r,ia,uid.long()); clicked=y.sum(1)>0; cls=torch_.where(clicked,y.argmax(1)+1,torch_.zeros(len(y),device=device,dtype=torch_.long)); ce=nn_.functional.cross_entropy(torch_.cat([no[:,None],logits],1),cls); penalty=lam*torch_.mean((model.item.weight[1:]-init)**2); loss=ce+.3*nn_.functional.binary_cross_entropy_with_logits(logits,y)+penalty; loss.backward(); torch_.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        pred_sig,pred_choice=_predict_pair(model,val_a,device,torch_); frame={'history_length':np.repeat(val_a.history_length,10),'gap_hours':np.repeat((val_a.timestamp-boundary)/3600.0,10),'label':val_a.labels.reshape(-1)}; dm=weighted_dm(frame,pred_choice,read_parquet(CACHE_DIR/'features_orig'/'evaluation.parquet')); held=_predict_pair(model,hold_a,device,torch_)[1]; held_auc=roc_auc_score(hold_a.labels.reshape(-1),held); history.append({'epoch':ep,'dm24':float(dm),'heldout_train_auc':float(held_auc),'train_loss':float(np.mean(losses))}); print('L3 verified',lam,history[-1],flush=True)
    return history

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); frame=read_parquet(CACHE_DIR/'features_orig'/f'{D0+4*DAY}.parquet'); evalf=read_parquet(CACHE_DIR/'features_orig'/'evaluation.parquet'); out={'boundary':D0+4*DAY,'rank':RANK,'batch_events':BATCH,'epochs':EPOCHS,'device':'cpu','l2_dm24':float(weighted_dm(frame,frame['svd_cos'],evalf)),'results':{}}
    for lam in LAMBDAS: out['results'][str(lam)]={'lambda':lam,'history':train_fold(data,D0+4*DAY,lam,EPOCHS)}
    (OUTPUT_DIR/'slatenet_l3_verified_results.json').write_text(json.dumps(out,indent=2,default=float)); print(json.dumps(out,indent=2,default=float))
if __name__=='__main__': main()
