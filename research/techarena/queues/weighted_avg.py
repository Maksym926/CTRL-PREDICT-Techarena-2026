"""Weighted average of prediction files (parquet or csv with rowid, predicted_probability), matched by rowid:

    python research/techarena/queues/weighted_avg.py OUT FILE1:W1 FILE2:W2 ...

OUT ends in .parquet or .csv. Used for package G's third model (4 Nov): the context MLP lr 2e-4 at weight 2 and the
attention MLP lr 2e-4 at weight 1, for both the validation predictions and the submission files.
"""
import sys

import pandas as pd

out, ins = sys.argv[1], [(a.rsplit(":", 1)[0], float(a.rsplit(":", 1)[1])) for a in sys.argv[2:]]
total = sum(w for _, w in ins)
base = None
for path, w in ins:
    f = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    f = f.assign(rowid=f["rowid"].astype(str))[["rowid", "predicted_probability"]]
    if base is None:
        base = f[["rowid"]].assign(predicted_probability=0.0)
    m = base[["rowid"]].merge(f, on="rowid", how="left")
    assert len(m) == len(base) and m["predicted_probability"].notna().all(), f"{path}: rowids differ"
    base["predicted_probability"] += w / total * m["predicted_probability"].to_numpy()
    print(f"  {path}: weight {w / total:.3f}")
base.to_parquet(out, index=False) if out.endswith(".parquet") else base.to_csv(out, index=False)
print(f"wrote {out}: {len(base):,} rows")
