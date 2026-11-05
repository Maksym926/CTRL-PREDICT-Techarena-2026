"""A2 feature-shift audit without fitting a CTR model."""
from __future__ import annotations
import json
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .features_a2 import build_features,build_stats
from .phase_a import adversarial_report
from .run_phase1 import first_events,subset
from .svd_model import fit_impression_svd,fit_svd

def make(data,b,eval_target=False):
    if eval_target:
        target=data.evaluation
    else:
        target=subset(data.train,first_events(data.train,b,CUT))
    st=build_stats(data,b);s11=fit_svd(st['click_log'],data.n_users,data.n_items,rank=11);s32=fit_svd(st['click_log'],data.n_users,data.n_items,rank=32);imp=fit_impression_svd(data.train,b,data.n_users,data.n_items,rank=10)
    return build_features(data,target,b,s11,st,svd_alt=s32,impression_svd=imp)

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    val=make(data,D0+6*DAY); ev=make(data,CUT,True)
    out={'overall':adversarial_report(val,ev),'without_gap_history':adversarial_report(val.drop(columns=['gap_hours','history_length']),ev.drop(columns=['gap_hours','history_length']))}
    (OUTPUT_DIR/'phase_a2_adversarial.json').write_text(json.dumps(out,indent=2,default=float));print(json.dumps(out,indent=2,default=float))
if __name__=='__main__':main()
