"""Publish the three-fold v3 audit and cheap blend artifact."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import xlsxwriter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
BASE_EXP={"model":"SAFE_v2 baseline","mode":"experiment","DM22":0.7788258635490015,"DM23":0.7905611569718828,"DM24":0.8013821139343696}
BASE_FULL={"model":"SAFE_v2 baseline","mode":"full","DM22":0.7859872505557265,"DM23":0.7959620121431473,"DM24":0.8055218787124601}
families={
 "bag":{"DM22":0.7784524093037567,"DM23":0.7888796807226376,"DM24":0.8024094038650889},
 "multi":{"DM22":0.778777531058251,"DM23":0.7896844882210461,"DM24":0.8027197072257555},
 "gap":{"DM22":0.7784023827560878,"DM23":0.790178335529438,"DM24":0.8019343639299742},
}
rows=[BASE_EXP,BASE_FULL]
for name,x in families.items(): rows.append({"model":name,"mode":"experiment","DM22":x["DM22"],"DM23":x["DM23"],"DM24":x["DM24"]})
rows += [
 {"model":"binary5","mode":"final diagnostic (300 trees)","DM22":0.7799786918698051,"DM23":0.7881453353830151,"DM24":None},
 {"model":"config2","mode":"final diagnostic (300 trees)","DM22":0.7815151027026166,"DM23":0.7899241411424363,"DM24":None},
 {"model":"lambdarank","mode":"final diagnostic (300 trees)","DM22":0.7748253725518841,"DM23":0.7834444547535935,"DM24":None},
 {"model":"CatBoost","mode":"unavailable","DM22":None,"DM23":None,"DM24":None},
 {"model":"BLEND_SAFE_OLD","mode":"file only","DM22":None,"DM23":None,"DM24":None,"file":"outputs/BLEND_SAFE_OLD.csv"},
]
for r in rows:
    vals=[r.get(k) for k in ("DM22","DM23","DM24")]
    r["DM_mean3"]=sum(vals)/3 if all(v is not None for v in vals) else None
    r["expected_portal_range"]=f"{r['DM_mean3']+0.007:.6f} to {r['DM_mean3']+0.011:.6f}" if r["DM_mean3"] is not None else ""
    r.setdefault("file","")
df=pd.DataFrame(rows)
df.to_csv(OUT/"phase_v3_summary.csv",index=False)
with xlsxwriter.Workbook(OUT/"phase_v3_report.xlsx",{"nan_inf_to_errors":True}) as wb:
    h=wb.add_format({"bold":True,"bg_color":"#D9EAF7","border":1}); n=wb.add_format({"num_format":"0.000000","border":1}); t=wb.add_format({"border":1})
    ws=wb.add_worksheet("models")
    for c,col in enumerate(df.columns): ws.write(0,c,col,h)
    for i,row in enumerate(df.itertuples(index=False),1):
        for c,v in enumerate(row):
            if isinstance(v,(float,int)) and pd.notna(v): ws.write_number(i,c,float(v),n)
            else: ws.write(i,c,str(v),t)
    ws.freeze_panes(1,0);ws.autofilter(0,0,len(df),len(df.columns)-1);ws.set_column(0,len(df.columns)-1,22)
    ws=wb.add_worksheet("decisions")
    decisions=[("experiment gate", "PASS only when DM_mean3 >= 0.792256; all v3 feature families rejected"),("full gate","No SAFE_v3/BEST_v3: no completed candidate beats SAFE_v2 by +0.005"),("CatBoost","Unavailable: package cannot be installed in offline runtime"),("graph blends","Not run: old graph file is absent"),("cheap blend","BLEND_SAFE_OLD.csv written and structurally validated")]
    for i,(a,b) in enumerate(decisions): ws.write(i,0,a,h if i==0 else t); ws.write(i,1,b,t)
    ws.set_column(0,1,70)
print(df.to_string(index=False))
