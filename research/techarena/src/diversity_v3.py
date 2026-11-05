"""Final-mode model-diversity audit on the three real long-gap folds."""
from __future__ import annotations
import json
import numpy as np
import lightgbm as lgb
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .phase_a import bucket_auc, weighted_auc
from .phase_v3 import build_train_val, frame_for
from .phase_c_real import cached_frame
from .train import feature_columns

SEEDS=(20261024,20261025,20261026,20261027,20261028)

def fit_binary(train,val,config,seed,trees=300):
    cols=feature_columns(train)
    if config=="config2":
        model=lgb.LGBMClassifier(objective="binary",n_estimators=trees,learning_rate=.03,num_leaves=127,max_bin=127,min_child_samples=500,colsample_bytree=.5,subsample=.8,subsample_freq=1,reg_lambda=10.0,random_state=seed,n_jobs=4,verbosity=-1)
    else:
        model=lgb.LGBMClassifier(objective="binary",n_estimators=trees,learning_rate=.03,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.0,random_state=seed,n_jobs=4,verbosity=-1)
    model.fit(train[cols],train["label"].astype(int))
    return model.predict_proba(val[cols])[:,1]

def fit_ranker(train,val,trees=300):
    cols=feature_columns(train)
    model=lgb.LGBMRanker(objective="lambdarank",metric="auc",n_estimators=trees,learning_rate=.03,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,random_state=20261024,n_jobs=4,verbosity=-1)
    model.fit(train[cols],train["label"].astype(int),group=[10]*(len(train)//10))
    return model.predict(val[cols])

def score(pred,val,eval_frame):
    return {"auc":float(roc_auc_score(val["label"],pred)),"reweight":weighted_auc(val,pred,eval_frame),"buckets":bucket_auc(val,pred)}

def main():
    data=join_global(load_cache(CACHE_DIR/"train.npz"),load_cache(CACHE_DIR/"evaluation.npz"))
    ev=frame_for(data,CUT,data.evaluation,"c2")
    folds={}
    all_rows=[]
    for name,day in [("DM22",2),("DM23",3),("DM24",4)]:
        train,val,_=build_train_val(data,D0+day*DAY,"c2")
        preds={}
        preds["binary5"]=[fit_binary(train,val,"binary",s) for s in SEEDS]
        preds["binary5_mean"]=np.mean(preds["binary5"],axis=0)
        preds["config2"]=[fit_binary(train,val,"config2",s) for s in SEEDS[:3]]
        preds["config2_mean"]=np.mean(preds["config2"],axis=0)
        preds["lambdarank"]=fit_ranker(train,val)
        fold_scores={}
        for key in ["binary5_mean","config2_mean","lambdarank"]:
            fold_scores[key]=score(preds[key],val,ev)
            all_rows.append({"fold":name,"key":key,"pred":preds[key],"label":val["label"].to_numpy(),"frame":val})
        folds[name]=fold_scores
        print("diversity",name,{k:v["reweight"]["joint_auc_supported"] for k,v in fold_scores.items()},flush=True)

    keys=["binary5_mean","config2_mean","lambdarank"]
    blend_rows={k:[] for k in keys}; labels=[]
    for row in all_rows:
        blend_rows[row["key"]].append(row["pred"])
        labels.append(row["label"])
    P=np.column_stack([np.concatenate(blend_rows[k]) for k in keys])
    y=np.concatenate(labels)
    best={"logit":None,"rank":None}
    for a in np.linspace(0,1,11):
        for b in np.linspace(0,1-a,11):
            ww=np.array([a,b,1-a-b])
            lp=expit(np.sum(ww[None,:]*logit(np.clip(P,1e-6,1-1e-6)),axis=1))
            ranks=np.column_stack([rankdata(P[:,j],method="average") for j in range(P.shape[1])])
            rp=np.sum(ww[None,:]*ranks,axis=1)
            for typ,pred in [("logit",lp),("rank",rp)]:
                auc=float(roc_auc_score(y,pred))
                if best[typ] is None or auc>best[typ]["auc"]:
                    best[typ]={"auc":auc,"weights":ww.tolist()}
    out={"folds":folds,"blend_search":best,"catboost":{"available":False,"reason":"catboost package unavailable in offline bundled runtime"},"trees":300,"seeds":list(SEEDS)}
    (OUTPUT_DIR/"diversity_v3_results.json").write_text(json.dumps(out,indent=2,default=float))
    print(json.dumps(out,indent=2,default=float))

if __name__=="__main__": main()
