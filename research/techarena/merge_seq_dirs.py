"""Average sequence-model feature folders (seq_<boundary>.parquet + .json each) into one, weighted by their seed counts.

Usage (from the repo root):
    python research/techarena/merge_seq_dirs.py research/techarena/cache/seq/stage3_h50_s10 \
        research/techarena/cache/seq/stage3_h50 research/techarena/cache/seq/stage3_h50_more
The seed counts come from each folder's JSONs (the Kaggle merge writes "seeds": [3, 2], i.e. 5). Every folder must hold
the same 14 boundaries with identical rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out, ins = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
out.mkdir(parents=True, exist_ok=True)
names = sorted(f.name for f in ins[0].glob("seq_*.parquet"))
assert len(names) == 14, f"{ins[0]}: {len(names)} files, 14 expected"
for f in names:
    frames = [pd.read_parquet(d / f) for d in ins]
    infos = [json.loads((d / f.replace(".parquet", ".json")).read_text()) for d in ins]
    seeds = np.array([sum(i["seeds"]) if isinstance(i.get("seeds"), list) else 1 for i in infos], float)
    for fr in frames[1:]:
        assert (fr["rowid"].to_numpy() == frames[0]["rowid"].to_numpy()).all(), f"{f}: rows differ"
    w = seeds / seeds.sum()
    prob = sum(wi * fr["seq_prob"].to_numpy() for wi, fr in zip(w, frames))
    noclick = sum(wi * fr["seq_noclick"].to_numpy() for wi, fr in zip(w, frames))
    rank = ((-prob.reshape(-1, 10)).argsort(1).argsort(1) + 1).ravel().astype(np.float32)
    pd.DataFrame({"rowid": frames[0]["rowid"], "seq_prob": prob, "seq_noclick": noclick, "seq_rank": rank}).to_parquet(
        out / f, index=False)
    (out / f.replace(".parquet", ".json")).write_text(json.dumps({"seeds": [int(s) for s in seeds],
                                                                  "sources": [str(d) for d in ins]}, indent=1))
print(f"merged {len(names)} boundaries from {[str(d) for d in ins]} (seeds {seeds.astype(int).tolist()}) into {out}")
