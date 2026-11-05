"""Three-fold v3 experiments and final-mode candidate evaluation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import join_global
from .phase_a import bucket_auc, weighted_auc
from .phase_c_real import cached_frame, downsample_events, importance_weights, shrink_extra
from .phase_c_revision import C1_COLS, C2_COLS, FULL_SEEDS, add_choice_pair, bounds_for, cached_c2_extra
from .run_phase1 import first_events, subset
from .train import feature_columns
from .v3_features import cached_v3

EXP_BASELINE3 = 0.7959716354531262
SAFE_V2_FULL3_UNKNOWN = None


def target_for(data, boundary):
    return subset(data.train, first_events(data.train, boundary, CUT))


def c2_eval(data):
    path=CACHE_DIR/"features_c"/"c2_evaluation.parquet"
    if path.exists(): extra=pd.read_parquet(path)
    else:
        extra=shrink_extra(data,data.evaluation,CUT,k=10.0,uncertainty=True)
        extra.to_parquet(path,index=False)
    return extra[C2_COLS].reset_index(drop=True)


def frame_for(data, boundary, target, family):
    base=cached_frame(data,boundary,"eval" if target is data.evaluation else "train").copy()
    if boundary==CUT and target is data.evaluation:
        extra=c2_eval(data)
    else:
        extra=cached_c2_extra(data,boundary)[C2_COLS].reset_index(drop=True)
    frame=pd.concat([base.reset_index(drop=True),extra],axis=1)
    if family!="c2":
        v=cached_v3(data,target,boundary,family)
        frame=pd.concat([frame.reset_index(drop=True),v.reset_index(drop=True)],axis=1)
    return frame


def build_train_val(data, boundary, family):
    day=int(round((boundary-D0)/DAY))
    bounds=bounds_for(day,half=True)
    parts=[]
    for wi,b in enumerate(bounds):
        t=target_for(data,b)
        f=frame_for(data,b,t,family)
        f["window_id"]=wi
        parts.append(f)
    train=pd.concat(parts,ignore_index=True)
    val=frame_for(data,boundary,target_for(data,boundary),family)
    train,val,choice=add_choice_pair(train,val)
    return train,val,choice


def fit_experiment(train,val,eval_frame):
    sample=downsample_events(train)
    w=importance_weights(sample,eval_frame)
    cols=feature_columns(sample)
    model=lgb.LGBMClassifier(objective="binary",n_estimators=1200,learning_rate=.1,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.0,random_state=20261024,n_jobs=4,verbosity=-1)
    model.fit(sample[cols],sample["label"].astype(int),sample_weight=w,eval_set=[(val[cols],val["label"].astype(int))],eval_metric="auc",callbacks=[lgb.early_stopping(50,verbose=False)])
    pred=model.predict_proba(val[cols],num_iteration=model.best_iteration_)[:,1]
    return {"auc":float(roc_auc_score(val["label"],pred)),"reweight":weighted_auc(val,pred,eval_frame),"buckets":bucket_auc(val,pred),"model":model,"cols":cols,"train":sample,"val":val,"pred":pred}


def run_family(data,eval_frame,family):
    out={}
    for name,day in [("DM22",2),("DM23",3),("DM24",4)]:
        b=D0+day*DAY
        train,val,_=build_train_val(data,b,family)
        out[name]=fit_experiment(train,val,eval_frame)
        print("v3",family,name,"auc",out[name]["auc"],"dm",out[name]["reweight"]["joint_auc_supported"],flush=True)
    out["DM_mean3"]=float(np.mean([out[k]["reweight"]["joint_auc_supported"] for k in ["DM22","DM23","DM24"]]))
    return out


def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items() if k not in {"model","cols","train","val","pred","choice"}}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,(np.floating,np.integer)): return x.item()
    return x


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--families",default="c2,bag,multi,gap,all")
    args=ap.parse_args()
    data=join_global(load_cache(CACHE_DIR/"train.npz"),load_cache(CACHE_DIR/"evaluation.npz"))
    eval_frame=frame_for(data,CUT,data.evaluation,"c2")
    results={"experiment_baseline_DM_mean3":EXP_BASELINE3,"families":{}}
    for fam in args.families.split(","):
        results["families"][fam]=run_family(data,eval_frame,fam)
        serial=clean(results["families"][fam])
        (OUTPUT_DIR/f"phase_v3_{fam}.json").write_text(json.dumps(serial,indent=2,default=float))
    (OUTPUT_DIR/"phase_v3_experiments.json").write_text(json.dumps(clean(results),indent=2,default=float))
    print(json.dumps(clean(results),indent=2,default=float))


if __name__=="__main__": main()
