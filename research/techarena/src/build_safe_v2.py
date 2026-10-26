"""Build SAFE_v2 only after the full-data gate has passed."""
from __future__ import annotations
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

from .cache import load_cache
from .choice import fit_choice
from .config import CACHE_DIR, CUT, D0, DAY, EXAMPLE_PATH, OUTPUT_DIR
from .dataset import join_global
from .phase_c_real import cached_frame, shrink_extra
from .phase_c_revision import C2_COLS, FULL_SEEDS, add_choice_pair, add_family, bounds_for
from .run_phase1 import first_events
from .submit import validate_submission, write_submission


def _target_for(data, boundary):
    return subset(data.train, first_events(data.train, boundary, CUT))


def main():
    report_path = OUTPUT_DIR / "phase_c_revision_results.json"
    report = json.loads(report_path.read_text())
    final = report["final"]
    if not final.get("beats_production_gate", False):
        raise SystemExit("full-data gate did not pass; SAFE_v2 was not built")
    family = final["family"]
    if family != "a3_c1_c2":
        raise SystemExit(f"unsupported final family for SAFE_v2: {family}")

    data = join_global(load_cache(CACHE_DIR / "train.npz"), load_cache(CACHE_DIR / "evaluation.npz"))
    train_parts=[]
    for wi,b in enumerate(bounds_for(6, half=True)):
        f=add_family(data, cached_frame(data,b).copy(), b, family)
        f["window_id"]=wi
        train_parts.append(f)
        print("training window", b, len(f), flush=True)
    # Add the final half-day boundary at Jan-25.5; bounds_for(6) covers
    # 20.5 through 25.5 Jan and therefore uses the complete training period.
    train=pd.concat(train_parts,ignore_index=True)
    eval_frame=cached_frame(data,CUT,"eval").copy()
    # Evaluation has no train target window; compute its C2 extras directly
    # from the evaluation histories at the cutoff.
    eval_extra=shrink_extra(data,data.evaluation,CUT,k=10.0,uncertainty=True)
    eval_frame=pd.concat([eval_frame.reset_index(drop=True),eval_extra[C2_COLS].reset_index(drop=True)],axis=1)
    train, eval_frame, choice = add_choice_pair(train, eval_frame)
    cols=[c for c in train.columns if c not in {"label","event_index","candidate_index","user_id","timestamp","window_id","sampling_weight"}]
    best=[]
    for f in final["folds"].values(): best.extend(f.get("best_iterations", []))
    # The validation folds ran to the 2,500-tree ceiling, but that count is
    # impractical for three full-data fits on this laptop. Keep the requested
    # lr=0.03/full-data/three-seed regime with a bounded 1,200-tree final fit.
    n_trees=min(1200, max(50,int(round(float(np.mean(best))))) if best else 1200)
    scores=np.zeros(len(eval_frame),dtype=np.float64)
    for seed in FULL_SEEDS:
        model=lgb.LGBMClassifier(objective="binary",n_estimators=n_trees,learning_rate=.03,num_leaves=63,max_bin=127,min_child_samples=300,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10.0,random_state=seed,n_jobs=4,verbosity=-1)
        model.fit(train[cols],train["label"].astype(int))
        scores += model.predict_proba(eval_frame[cols])[:,1]
        print("seed complete",seed,flush=True)
    scores=np.clip(scores/len(FULL_SEEDS),1e-7,1-1e-7)
    rowids=data.evaluation.rowids.reshape(-1)
    evaluation=pd.DataFrame({"rowid":rowids})
    example=pd.read_csv(EXAMPLE_PATH)
    out=OUTPUT_DIR/"SAFE_v2.csv"
    write_submission(rowids,scores,evaluation,out,example)
    written=pd.read_csv(out)
    validate_submission(written,evaluation,example)
    checks={"path":str(out),"rows":int(len(written)),"unique_rowids":int(written.rowid.nunique()),"missing_rowids":int(len(set(example.rowid)-set(written.rowid))),"duplicate_rowids":int(written.rowid.duplicated().sum()),"min":float(written.predicted_probability.min()),"max":float(written.predicted_probability.max()),"mean":float(written.predicted_probability.mean()),"trees":n_trees,"seeds":list(FULL_SEEDS),"family":family}
    (OUTPUT_DIR/"SAFE_v2_checks.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__ == "__main__": main()
