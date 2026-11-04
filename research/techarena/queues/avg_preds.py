"""Average predicted_probability over several prediction files (parquet or csv) by rowid: avg_preds.py OUT IN1 IN2 ..."""
import sys

import pandas as pd

out, ins = sys.argv[1], sys.argv[2:]
frames = [pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p) for p in ins]
d = frames[0].assign(rowid=frames[0]["rowid"].astype(str)).rename(columns={"predicted_probability": "p0"})
for i, f in enumerate(frames[1:], 1):
    d = d.merge(f.assign(rowid=f["rowid"].astype(str)).rename(columns={"predicted_probability": f"p{i}"}), on="rowid")
assert len(d) == len(frames[0]), "rowids differ between the files"
d["predicted_probability"] = d[[f"p{i}" for i in range(len(ins))]].mean(1)
print(d[[f"p{i}" for i in range(len(ins))]].corr().round(4))
res = d[["rowid", "predicted_probability"]]
res.to_parquet(out, index=False) if out.endswith(".parquet") else res.to_csv(out, index=False)
print(f"wrote {out}: {len(res):,} rows")
