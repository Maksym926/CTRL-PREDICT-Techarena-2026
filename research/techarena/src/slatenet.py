"""Leakage-safe supervised choice network for the TechArena CTR task.

The module is intentionally self-contained so it can run in the GPU venv on
the DCU server. It trains on every labelled event strictly before a fold
boundary B and evaluates on first-event-per-user targets after B.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import CachedEvents, load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import GlobalDataset, join_global
from .features import build_stats
from .phase_a import bucket_auc, weighted_auc
from .run_phase1 import first_events, subset
from .svd_model import fit_svd


def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("SlateNet requires torch; run scripts/setup_server.sh on the GPU host") from exc
    return torch, nn, DataLoader, Dataset


@dataclass
class EventArrays:
    candidates: np.ndarray
    history: np.ndarray
    history_age_days: np.ndarray
    item_log_clicks: np.ndarray
    repeat: np.ndarray
    item_age_days: np.ndarray
    labels: np.ndarray
    user_id: np.ndarray
    timestamp: np.ndarray
    history_length: np.ndarray


def _event_features(events: CachedEvents, source_mask: np.ndarray, target_indices: np.ndarray | None = None) -> EventArrays:
    """Build online item statistics using only source events before B."""
    n_items = int(max(events.candidates.max(), events.history_items.max()) + 1)
    click_counts = np.zeros(n_items, dtype=np.float32)
    last_click = np.full(n_items, -1, dtype=np.int64)
    order = np.flatnonzero(source_mask)
    order = order[np.argsort(events.timestamp[order], kind="stable")]
    target_set = set(int(x) for x in target_indices) if target_indices is not None else None
    rows = []
    for e in order:
        if target_set is not None and int(e) not in target_set:
            # Source events only advance the online state. They are not rows
            # returned by this call unless target_indices is omitted.
            for j, item in enumerate(events.candidates[e]):
                if events.labels[e, j] == 1:
                    item = int(item); click_counts[item] += 1; last_click[item] = int(events.timestamp[e])
            continue
        rows.append(int(e))
    if target_indices is not None:
        rows = [int(x) for x in target_indices]
    else:
        rows = [int(x) for x in order]
    # Rebuild the source state once, then calculate target features at B. For
    # training rows, we update immediately after each event.
    click_counts[:] = 0; last_click[:] = -1
    features = {}
    source_set = set(int(x) for x in order)
    for e in order:
        cands = events.candidates[e].astype(np.int64, copy=False)
        hist = events.history_items[e].astype(np.int64, copy=False)
        ts = int(events.timestamp[e])
        valid = hist >= 0
        h = hist[valid]
        repeat = np.asarray([float(int(c) in set(int(x) for x in h)) for c in cands], dtype=np.float32)
        age = np.asarray([(ts - int(last_click[int(c)])) / DAY if last_click[int(c)] >= 0 else 999.0 for c in cands], dtype=np.float32)
        feat = (np.log1p(click_counts[cands]), repeat, age)
        if target_indices is None:
            features[int(e)] = feat
        if int(e) in source_set:
            for j, item in enumerate(cands):
                if events.labels[e, j] == 1:
                    item = int(item); click_counts[item] += 1; last_click[item] = ts
    if target_indices is not None:
        # At a fold boundary, source state contains every event before B.
        click_counts[:] = 0; last_click[:] = -1
        for e in order:
            for j, item in enumerate(events.candidates[e]):
                if events.labels[e, j] == 1:
                    item = int(item); click_counts[item] += 1; last_click[item] = int(events.timestamp[e])
        for e in rows:
            cands = events.candidates[e].astype(np.int64, copy=False)
            hist = events.history_items[e].astype(np.int64, copy=False)
            ts = int(events.timestamp[e]); h = hist[hist >= 0]
            repeat = np.asarray([float(int(c) in set(int(x) for x in h)) for c in cands], dtype=np.float32)
            age = np.asarray([(ts - int(last_click[int(c)])) / DAY if last_click[int(c)] >= 0 else 999.0 for c in cands], dtype=np.float32)
            features[int(e)] = (np.log1p(click_counts[cands]), repeat, age)

    cand = events.candidates[np.asarray(rows)].astype(np.int64)
    hist = events.history_items[np.asarray(rows)].astype(np.int64)
    hts = events.history_times[np.asarray(rows)].astype(np.int64)
    ts = events.timestamp[np.asarray(rows)].astype(np.int64)
    h_age = np.where(hist >= 0, np.maximum(0, ts[:, None] - hts) / DAY, 999.0).astype(np.float32)
    item_log = np.stack([features[e][0] for e in rows]).astype(np.float32)
    repeat = np.stack([features[e][1] for e in rows]).astype(np.float32)
    item_age = np.stack([features[e][2] for e in rows]).astype(np.float32)
    labels = events.labels[np.asarray(rows)].astype(np.float32)
    return EventArrays(cand, hist, h_age, item_log, repeat, item_age, labels, events.user_id[np.asarray(rows)], ts, events.history_length[np.asarray(rows)])


def _dataset_class(torch, Dataset):
    class EventDataset(Dataset):
        def __init__(self, arrays): self.a = arrays
        def __len__(self): return len(self.a.candidates)
        def __getitem__(self, i):
            a=self.a
            return (a.candidates[i], a.history[i], a.history_age_days[i], a.item_log_clicks[i], a.repeat[i], a.item_age_days[i], a.labels[i], a.user_id[i])
    return EventDataset


def _model_class(torch, nn):
    class SlateNet(nn.Module):
        def __init__(self, n_items, n_users, dim=32, user_dim=16, use_user=True, dropout=.15):
            super().__init__(); self.dim=dim; self.use_user=use_user
            self.item=nn.Embedding(n_items+1, dim, padding_idx=0)
            self.user=nn.Embedding(n_users, user_dim) if use_user else None
            att_in=dim*4+1
            self.att=nn.Sequential(nn.Linear(att_in,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1))
            state_dim=dim+(user_dim if use_user else 0)+1
            self.state=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,dim))
            cand_dim=dim+dim+dim+4
            self.score=nn.Sequential(nn.Linear(cand_dim+dim,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,1))
            self.no_click=nn.Sequential(nn.Linear(dim+dim+2,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,cands,hist,h_age,item_log,repeat,item_age,user_id):
            ce=self.item(cands+1); he=self.item(torch.clamp(hist+1,min=0)); mask=(hist>=0).float()
            rec=torch.exp(-torch.clamp(h_age,min=0,max=999)/7.0)*mask
            hmean=(he*rec.unsqueeze(-1)).sum(1)/(rec.sum(1,keepdim=True)+1e-6)
            hlen=mask.sum(1,keepdim=True)/20.0
            if self.use_user: state_in=torch.cat([hmean,self.user(user_id),hlen],-1)
            else: state_in=torch.cat([hmean,hlen],-1)
            ustate=self.state(state_in)
            hp=he.unsqueeze(1).expand(-1,10,-1,-1); cp=ce.unsqueeze(2).expand(-1,-1,20,-1)
            att_in=torch.cat([hp,cp,hp*cp,hp-cp,torch.log1p(h_age).unsqueeze(1).unsqueeze(-1).expand(-1,10,-1,-1)],-1)
            aw=self.att(att_in).squeeze(-1).masked_fill(mask.unsqueeze(1)==0,-1e9)
            aw=torch.softmax(aw,-1); ah=(aw.unsqueeze(-1)*hp).sum(2)
            slate=ce.mean(1)
            hist_dot=(ce*hmean.unsqueeze(1)).sum(-1)
            cand=torch.cat([ce,ah,ce*ustate.unsqueeze(1),item_log.unsqueeze(-1),repeat.unsqueeze(-1),torch.log1p(item_age).unsqueeze(-1),hist_dot.unsqueeze(-1)],-1)
            logits=self.score(torch.cat([cand,ustate.unsqueeze(1).expand(-1,10,-1)],-1)).squeeze(-1) + 2.0*hist_dot
            no=self.no_click(torch.cat([ustate,slate,hlen,torch.log1p(h_age.masked_fill(mask==0,999).min(1).values).unsqueeze(-1)],-1)).squeeze(-1)
            return logits,no,ustate
    return SlateNet


def _arrays_to_torch(torch, batch, device):
    return [torch.as_tensor(x,device=device) for x in batch]


def train_fold(data: GlobalDataset, boundary: int, epochs=2, dim=32, batch_size=2048, seed=20261024):
    torch, nn, DataLoader, Dataset = _torch(); torch.manual_seed(seed); np.random.seed(seed)
    source = data.train.timestamp < boundary
    train_a = _event_features(data.train, source)
    target_idx = np.asarray(first_events(data.train, boundary, CUT), dtype=np.int64)
    val_a = _event_features(data.train, source, target_idx)
    EventDataset=_dataset_class(torch,Dataset); Model=_model_class(torch,nn)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Model(data.n_items,data.n_users,dim=dim).to(device)
    stats=build_stats(data,boundary); svd=fit_svd(stats["click_log"],data.n_users,data.n_items,rank=11)
    with torch.no_grad():
        d=min(dim,svd.item_vectors.shape[1]); model.item.weight[1:, :d].copy_(torch.as_tensor(svd.item_vectors[:, :d],device=device))
        if d < dim: model.item.weight[1:, d:].zero_()
    model.item.weight.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-5)
    loader=DataLoader(EventDataset(train_a),batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=device.type=="cuda")
    best=None; history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for batch in loader:
            c,h,ha,il,r,ia,y,uid=_arrays_to_torch(torch,batch,device); opt.zero_grad(set_to_none=True)
            logits,no,_=model(c,h,ha,il,r,ia,uid.long())
            clicked=y.sum(1)>0; cls=torch.where(clicked,y.argmax(1)+1,torch.zeros(len(y),device=device,dtype=torch.long))
            ce=nn.functional.cross_entropy(torch.cat([no[:,None],logits],1),cls); bce=nn.functional.binary_cross_entropy_with_logits(logits,y); loss=ce+.3*bce; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        pred=_predict(model,val_a,device,torch)
        val_frame=pd.DataFrame({"history_length":np.repeat(val_a.history_length,10),"gap_hours":np.repeat((val_a.timestamp-boundary)/3600.0,10),"label":val_a.labels.reshape(-1)})
        pred_auc=float(roc_auc_score(val_frame.label,pred)); history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"auc":pred_auc})
        if best is None or pred_auc>best["auc"]: best={"auc":pred_auc,"state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"epoch":epoch+1}
    model.load_state_dict(best["state"])
    return model, val_a, history, device


def _predict(model, arrays, device, torch):
    return _predict_pair(model, arrays, device, torch)[0]


def _predict_pair(model, arrays, device, torch):
    model.eval(); out=[]; choice_out=[]
    with torch.no_grad():
        for start in range(0,len(arrays.candidates),2048):
            sl=slice(start,start+2048); batch=(arrays.candidates[sl],arrays.history[sl],arrays.history_age_days[sl],arrays.item_log_clicks[sl],arrays.repeat[sl],arrays.item_age_days[sl],arrays.labels[sl]); c,h,ha,il,r,ia,y=[torch.as_tensor(x,device=device) for x in batch]; logits,no,_=model(c,h,ha,il,r,ia,torch.as_tensor(arrays.user_id[sl],device=device)); out.append(torch.sigmoid(logits).cpu().numpy()); choice_out.append(torch.softmax(torch.cat([no[:,None],logits],1),1)[:,1:].cpu().numpy())
    return np.concatenate(out).reshape(-1), np.concatenate(choice_out).reshape(-1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--dim",type=int,default=32); ap.add_argument("--folds",default="DM22,DM23,DM24"); args=ap.parse_args()
    data=join_global(load_cache(CACHE_DIR/"train.npz"),load_cache(CACHE_DIR/"evaluation.npz")); eval_frame=None; results={}
    for name,day in [("DM22",2),("DM23",3),("DM24",4)]:
        if name not in args.folds.split(","): continue
        model,val,history,device=train_fold(data,D0+day*DAY,args.epochs,args.dim,seed=20261024+day)
        pred_sig,pred_choice=_predict_pair(model,val,device,_torch()[0]); frame=pd.DataFrame({"history_length":np.repeat(val.history_length,10),"gap_hours":np.repeat((val.timestamp-(D0+day*DAY))/3600.0,10),"label":val.labels.reshape(-1)})
        if eval_frame is None:
            from .phase_c_real import cached_frame as cf
            eval_frame=cf(data,CUT,"eval")
        chosen="softmax_choice" if roc_auc_score(frame.label,pred_choice)>roc_auc_score(frame.label,pred_sig) else "sigmoid"
        chosen_pred=pred_choice if chosen=="softmax_choice" else pred_sig
        results[name]={"auc_sigmoid":float(roc_auc_score(frame.label,pred_sig)),"auc_softmax_choice":float(roc_auc_score(frame.label,pred_choice)),"dm_sigmoid":weighted_auc(frame,pred_sig,eval_frame),"dm_softmax_choice":weighted_auc(frame,pred_choice,eval_frame),"chosen_output":chosen,"history":history,"prediction":chosen_pred.tolist()}
        print("SlateNet",name,results[name]["auc_sigmoid"],results[name]["auc_softmax_choice"],flush=True)
    (OUTPUT_DIR/"slatenet_results.json").write_text(json.dumps(results,indent=2,default=float)); print(json.dumps({k:{"auc_sigmoid":v["auc_sigmoid"],"auc_softmax_choice":v["auc_softmax_choice"],"chosen_output":v["chosen_output"],"epochs":len(v["history"])} for k,v in results.items()},indent=2))

if __name__=="__main__": main()
