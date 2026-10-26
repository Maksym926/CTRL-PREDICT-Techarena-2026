"""Full-mode SAFE_v2 DM22 fold."""
from __future__ import annotations
import json
from .cache import load_cache
from .config import CACHE_DIR,CUT,D0,DAY,OUTPUT_DIR
from .dataset import join_global
from .phase_c_real import cached_frame
from .phase_c_revision import final_fold,bounds_for

def main():
    data=join_global(load_cache(CACHE_DIR/"train.npz"),load_cache(CACHE_DIR/"evaluation.npz"))
    ev=cached_frame(data,CUT,"eval")
    out=final_fold(data,D0+2*DAY,bounds_for(2,half=True),ev,"a3_c1_c2")
    clean={"auc":out["auc"],"reweight":out["reweight"],"buckets":out["buckets"],"best_iterations":out["best_iterations"]}
    (OUTPUT_DIR/"safe_v2_dm22_full.json").write_text(json.dumps(clean,indent=2,default=float))
    print(json.dumps(clean,indent=2,default=float))
if __name__=="__main__": main()
