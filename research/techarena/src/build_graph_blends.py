"""Build global rank-average blends once the old graph CSV is supplied."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import EVAL_PATH, EXAMPLE_PATH, OUTPUT_DIR
from .submit import write_submission

def main():
    new_path=OUTPUT_DIR/'SAFE_v2.csv' if (OUTPUT_DIR/'SAFE_v2.csv').exists() else OUTPUT_DIR/'SAFE_A2.csv'
    old_path=OUTPUT_DIR/'techarena_submission_meta_graph80.csv'
    if not old_path.exists():
        raise FileNotFoundError(f'missing {old_path}; place the old graph CSV there before running')
    new=pd.read_csv(new_path); old=pd.read_csv(old_path); evaluation=pd.read_csv(EVAL_PATH,usecols=['rowid']); example=pd.read_csv(EXAMPLE_PATH)
    if not new.rowid.equals(old.rowid): old=old.set_index('rowid').loc[new.rowid].reset_index()
    corr=float(spearmanr(new.predicted_probability,old.predicted_probability).statistic)
    nr=new.predicted_probability.rank(method='average',pct=True).to_numpy();orank=old.predicted_probability.rank(method='average',pct=True).to_numpy()
    outputs={'spearman':corr,'weights':{}}
    for w in (0.90,0.85,0.75):
        score=w*nr+(1-w)*orank; path=OUTPUT_DIR/f'BLEND_G{int(w*100):02d}.csv'; write_submission(new.rowid,score,evaluation,path,example); outputs['weights'][str(w)]={'path':str(path),'new_weight':w,'old_weight':1-w}
    (OUTPUT_DIR/'graph_blend_results.json').write_text(json.dumps(outputs,indent=2));print(json.dumps(outputs,indent=2))
if __name__=='__main__':main()
