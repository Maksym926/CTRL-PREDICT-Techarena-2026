"""Label-free diagnosis of SAFE_LOGMF10 versus graph80 on evaluation rows."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import OUTPUT_DIR, EXAMPLE_PATH


def bucket_history(x):
    return np.where(x<=0,'0',np.where(x<=9,'1-9',np.where(x<=19,'10-19','20')))

def bucket_gap(x):
    return np.where(x<6,'<6h',np.where(x<24,'6-24h',np.where(x<72,'1-3d',np.where(x<168,'3-7d','>7d'))))

def dist_report(labels, mask):
    out={}
    for k in np.unique(labels):
        out[str(k)]={'rows':int((labels==k).sum()),'share':float((labels[mask]==k).mean()) if mask.any() else 0.0}
    return out

def main():
    safe=pd.read_csv(OUTPUT_DIR/'SAFE_LOGMF10.csv'); graph=pd.read_csv(OUTPUT_DIR/'techarena_submission_meta_graph80.csv')
    ex=pd.read_csv(EXAMPLE_PATH)
    if not safe.rowid.equals(ex.rowid): safe=safe.set_index('rowid').loc[ex.rowid].reset_index()
    if not graph.rowid.equals(ex.rowid): graph=graph.set_index('rowid').loc[ex.rowid].reset_index()
    if len(safe)!=529390 or len(graph)!=529390: raise ValueError('unexpected row count')
    s=safe.predicted_probability.to_numpy().reshape(-1,10); g=graph.predicted_probability.to_numpy().reshape(-1,10)
    feat=pd.read_parquet(Path(__file__).resolve().parents[1]/'cache/features_orig/evaluation.parquet')
    F=feat.to_numpy()
    h=feat.history_length.to_numpy(); gap=feat.gap_hours.to_numpy(); seen=h>0; inhist=feat.history_count.to_numpy()>0
    pop=feat.item_impressions.to_numpy(); svd=feat.svd_cos.to_numpy()
    # Per-event rank agreement and correlations.
    event_r=[]; top_same=[]
    srank=np.argsort(np.argsort(-s,axis=1),axis=1); grank=np.argsort(np.argsort(-g,axis=1),axis=1)
    for i in range(len(s)):
        event_r.append(float(spearmanr(s[i],g[i]).statistic) if np.unique(s[i]).size>1 and np.unique(g[i]).size>1 else 0.0)
        top_same.append(int(np.argmax(s[i])==np.argmax(g[i])))
    evmean_s=s.mean(1); evmean_g=g.mean(1); evmax_s=s.max(1); evmax_g=g.max(1)
    row_abs=np.abs(srank-grank).reshape(-1); topmask=row_abs>=np.quantile(row_abs,.90)
    labels={
      'history_length':bucket_history(h), 'gap':bucket_gap(gap),
      'seen_user':np.where(seen,'seen','unseen'), 'in_history_candidate':np.where(inhist,'in_history','new'),
      'item_popularity_decile':np.asarray(pd.qcut(pop,10,labels=False,duplicates='drop')).astype(str),
      'svd_cos_decile':np.asarray(pd.qcut(svd,10,labels=False,duplicates='drop')).astype(str),
    }
    result={
      'rows':int(len(safe)), 'global_spearman':float(spearmanr(safe.predicted_probability,graph.predicted_probability).statistic),
      'within_event_mean_spearman':float(np.mean(event_r)), 'within_event_median_spearman':float(np.median(event_r)),
      'same_top1_share':float(np.mean(top_same)),
      'event_mean_spearman':float(spearmanr(evmean_s,evmean_g).statistic), 'event_max_spearman':float(spearmanr(evmax_s,evmax_g).statistic),
      'rank_diff_quantiles':{str(q):float(np.quantile(row_abs,q)) for q in (.5,.75,.9,.95,.99)},
      'top10_row_share':float(topmask.mean()), 'disagreement_map':{}
    }
    for name,lab in labels.items(): result['disagreement_map'][name]={'all':dist_report(lab,np.ones(len(lab),bool)),'top10_abs_rank_diff':dist_report(lab,topmask)}
    (OUTPUT_DIR/'graph_signal_diagnosis.json').write_text(json.dumps(result,indent=2,default=float))
    md=['# Graph signal diagnosis','',f"Global Spearman: **{result['global_spearman']:.6f}**.",f"Mean within-event Spearman: **{result['within_event_mean_spearman']:.6f}**; same top-1 candidate: **{result['same_top1_share']:.4%}**.",f"Event-mean Spearman: **{result['event_mean_spearman']:.6f}**; event-max Spearman: **{result['event_max_spearman']:.6f}**.",'', 'The JSON contains all disagreement-map buckets. Rows in the largest 10% absolute within-event rank disagreement were compared with all evaluation rows.']
    (OUTPUT_DIR/'graph_signal_diagnosis.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(result,indent=2,default=float))

if __name__=='__main__': main()
