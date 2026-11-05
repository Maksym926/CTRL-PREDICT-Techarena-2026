"""Lightweight signal-scale calibration for the synthetic twin."""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .features import build_stats
from .run_phase1 import first_events,subset
from .phase_b_twin import _calibrate_no_click,_utility_arrays
from .svd_model import fit_svd

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz')); st=build_stats(data,CUT); svd=fit_svd(st['click_log'],data.n_users,data.n_items,11); val=subset(data.train,first_events(data.train,D0+6*DAY,CUT)); rng=np.random.default_rng(20261024); rows=[]
    for scale in [2,4,6,8,10,12,16]:
        for pop in [0.0,0.02,0.05,0.08,0.12,0.2]:
            bias=_calibrate_no_click(data.train,svd,st['item_clicks'],2.0,float(scale),float(pop)); util,rep,probs=_utility_arrays(val,svd,st['item_clicks'],2.0,bias,float(scale),float(pop)); labels=np.zeros((val.n_events,10),dtype=np.int8)
            for e in range(val.n_events):
                pick=int(rng.choice(11,p=probs[e]));
                if pick>0: labels[e,pick-1]=1
            y=labels.reshape(-1); svd_item=svd.item_vectors[val.candidates]; uv=svd.user_vectors[val.user_id]; cos=np.einsum('erd,ed->er',svd_item,uv)/((np.linalg.norm(svd_item,axis=2)+1e-8)*(np.linalg.norm(uv,axis=1)[:,None]+1e-8))
            rows.append({'scale':scale,'pop':pop,'no_click_bias':bias,'no_click_rate':float(probs[:,0].mean()),'repeat_ctr':float(probs[:,1:][rep].mean()) if rep.any() else 0.0,'other_ctr':float(probs[:,1:][~rep].mean()) if (~rep).any() else 0.0,'oracle_auc':float(roc_auc_score(y,probs[:,1:].reshape(-1))),'svd_auc':float(roc_auc_score(y,cos.reshape(-1)))})
    rows=sorted(rows,key=lambda x:abs(x['svd_auc']-0.79)); result={'target_svd_auc':0.79,'target_pipeline_auc':0.818,'grid':rows,'best_by_svd':rows[:10],'pipeline_auc':'not run: no calibrated setting can be accepted until the SVD twin gate matches'}; (OUTPUT_DIR/'phase_b2_calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
