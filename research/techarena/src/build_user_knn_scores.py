"""Build user-kNN arrays for the seven DM24 residual windows only."""
from pathlib import Path
import numpy as np
from .cache import load_cache
from .config import CACHE_DIR, CUT, D0, DAY, OUTPUT_DIR
from .dataset import click_log_before, join_global
from .graph_fold_safe import _sequences, _user_knn_score
from .run_phase1 import first_events, subset
from .phase_c_revision import bounds_for

def main():
    data=join_global(load_cache(CACHE_DIR/'train.npz'),load_cache(CACHE_DIR/'evaluation.npz'))
    for b in bounds_for(4,half=True):
        out=OUTPUT_DIR/f'graph_scores_{int(b)}.npz'
        if out.exists() and 'user_knn' in np.load(out).files:
            continue
        log=click_log_before(data,b); seq=_sequences(log); idx=first_events(data.train,b,CUT); target=subset(data.train,idx)
        s=_user_knn_score(data,target,seq)
        old={}
        if out.exists():
            z=np.load(out); old={k:z[k] for k in z.files}
        old['user_knn']=s; np.savez_compressed(out,**old)
        print('user_knn',b,len(s),flush=True)
if __name__=='__main__': main()
