"""Write the Phase C experiment log, workbook, and notebook appendix."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULTS = json.loads((OUT / "phase_c_real_results.json").read_text())
LOG_PATH = ROOT / "experiment_log.csv"

ORIGINAL_DM = 0.797383140197548

def dm(method):
    x = RESULTS[method]
    return float(x["DM_mean"])

def row(method, label=None, kept="rejected", notes=""):
    x = RESULTS[method]
    dm23 = float(x["DM23"]["reweight"]["joint_auc_supported"])
    dm24 = float(x["DM24"]["reweight"]["joint_auc_supported"])
    ordinary = (float(x["DM23"]["auc"]) + float(x["DM24"]["auc"])) / 2.0
    hist23 = x["DM23"]["buckets"]["history_length"]
    hist24 = x["DM24"]["buckets"]["history_length"]
    h19 = (float(hist23["1-9"]["auc"]) + float(hist24["1-9"]["auc"])) / 2.0
    detail = f"DM23={dm23:.6f}; DM24={dm24:.6f}; DM_mean={dm(method):.6f}; 1-9 AUC={h19:.6f}; {notes}"
    return {
        "phase": "A3" if method == "A3" else ("C" if method in {"c1", "c2", "c3"} else "experiment"),
        "experiment": label or method,
        "standard_val": ordinary,
        "reweighted_val": dm(method),
        "long_gap_val_DM": dm(method),
        "twin_oracle": "",
        "twin_pipeline": "",
        "twin_vector_cosine": "",
        "kept": kept,
        "notes": detail,
    }

# Preserve previous rows exactly and append a measured row for every real-data run.
old = pd.read_csv(LOG_PATH)
new_rows = [
    row("A3", "half-day targets + eval-matched importance weights", "rejected", "experiment mode: 25% event-coherent negative downsample; lower than original production DM"),
    row("base", "original features, daily targets, experiment baseline", "reference", "experiment mode baseline for C1-C3"),
    row("c1", "C1 shrinkage + global no-click choice features", "rejected", "below A3 and original production DM; no >=0.003 gain"),
    row("c2", "C2 shrinkage uncertainty features", "rejected", "same measured AUC as C1; no >=0.003 gain"),
    row("c3", "C3 bounded recent-history EASE co-click feature", "rejected", "bounded 20-item/user approximation; below baseline"),
]
new_df = pd.DataFrame(new_rows)
# The first report attempt writes the log before creating the workbook. Make
# this script idempotent so a retry cannot duplicate the measured rows.
if not old.empty:
    new_df = new_df[~new_df["experiment"].isin(old["experiment"].astype(str))]
new = pd.concat([old, new_df], ignore_index=True)
new.to_csv(LOG_PATH, index=False, float_format="%.12f")

# Tidy fold and bucket tables for review.
fold_rows=[]
bucket_rows=[]
for method in ["A3","base","c1","c2","c3"]:
    x=RESULTS[method]
    for fold in ["DM23","DM24"]:
        f=x[fold]
        fold_rows.append({"method":method,"fold":fold,"ordinary_auc":f["auc"],"reweighted_auc":f["reweight"]["joint_auc_supported"],"history_only_auc":f["reweight"]["history_only_auc"],"supported_eval_mass":f["reweight"]["joint_eval_mass_supported"]})
        for b,v in f["buckets"]["history_length"].items():
            bucket_rows.append({"method":method,"fold":fold,"bucket_type":"history_length","bucket":b,"rows":v["rows"],"share":v["share"],"auc":v["auc"]})
        for b,v in f["buckets"]["gap"].items():
            bucket_rows.append({"method":method,"fold":fold,"bucket_type":"gap","bucket":b,"rows":v["rows"],"share":v["share"],"auc":v["auc"]})

summary_rows=[]
for method in ["A3","base","c1","c2","c3"]:
    x=RESULTS[method]
    summary_rows.append({"method":method,"DM23":x["DM23"]["reweight"]["joint_auc_supported"],"DM24":x["DM24"]["reweight"]["joint_auc_supported"],"DM_mean":x["DM_mean"],"delta_vs_original_production":x["DM_mean"]-ORIGINAL_DM,"decision":"keep" if x["DM_mean"]-ORIGINAL_DM>=0.003 else "reject"})
summary=pd.DataFrame(summary_rows)
fold=pd.DataFrame(fold_rows)
buckets=pd.DataFrame(bucket_rows)

workbook_path=OUT/"phase_c_experiment_report.xlsx"
with xlsxwriter.Workbook(workbook_path, {"nan_inf_to_errors": True}) as wb:
    fmt=wb.add_format({"bold":True,"bg_color":"#D9EAF7","border":1})
    num=wb.add_format({"num_format":"0.000000","border":1})
    txt=wb.add_format({"border":1})
    for name,df in [("summary",summary),("fold_results",fold),("bucket_results",buckets),("experiment_log",new)]:
        ws=wb.add_worksheet(name)
        for c,col in enumerate(df.columns): ws.write(0,c,col,fmt)
        for r_idx,rowv in enumerate(df.itertuples(index=False),1):
            for c,val in enumerate(rowv):
                if isinstance(val,(float,int)) and not isinstance(val,bool) and pd.notna(val): ws.write_number(r_idx,c,float(val),num)
                else: ws.write(r_idx,c,str(val),txt)
        ws.freeze_panes(1,0); ws.autofilter(0,0,len(df),len(df.columns)-1)
        ws.set_column(0,len(df.columns)-1,18)
    ws=wb.add_worksheet("decision")
    ws.write(0,0,"Decision",fmt);ws.write(1,0,"Original production DM_mean",txt);ws.write_number(1,1,ORIGINAL_DM,num)
    ws.write(2,0,"A3 DM_mean",txt);ws.write_number(2,1,RESULTS["A3"]["DM_mean"],num)
    ws.write(3,0,"Best C method DM_mean",txt);ws.write_number(3,1,max(RESULTS[m]["DM_mean"] for m in ["c1","c2","c3"]),num)
    ws.write(4,0,"Combined C gain gate",txt);ws.write(4,1,"FAIL: no C1-C3 method adds +0.003 DM_mean",txt)
    ws.write(5,0,"Next phase",txt);ws.write(5,1,"STOP after C3; do not run D/E or create new submission",txt)
    ws.set_column(0,1,48)

# Save a compact JSON summary next to the raw run.
(OUT/"phase_c_summary.json").write_text(json.dumps({"original_production_DM_mean":ORIGINAL_DM,"summary":summary_rows,"workbook":str(workbook_path)},indent=2))

# Append an explicit results appendix to the documentation notebook.
nb_path=ROOT/"notebooks"/"run_all.ipynb"
nb=json.loads(nb_path.read_text())
append=[
 {"cell_type":"markdown","metadata":{},"source":["## Phase C: real-data light-user experiments\n","The twin and adversarial diagnostics are retired. DM_mean is the mean of the evaluation-matched long-gap folds at B=23 Jan and B=24 Jan. The original production feature path is the reference. Experiment mode uses event-coherent 25% negative-event downsampling, LightGBM learning rate 0.1, 63 leaves and max_bin 127.\n\n","A3 DM_mean was 0.792807. C1 shrinkage plus a global no-click choice feature scored 0.790130, C2 added uncertainty features but was identical at 0.790130, and bounded recent-history EASE scored 0.787933. None clears the +0.003 gate against the original production DM_mean 0.797383, so Phase D/E are intentionally stopped."]},
 {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import json\n","from pathlib import Path\n","report=json.loads(Path('outputs/phase_c_real_results.json').read_text())\n","{k: report[k]['DM_mean'] for k in ['A3','base','c1','c2','c3']}\n"]},
 {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import pandas as pd\n","pd.read_excel('outputs/phase_c_experiment_report.xlsx', sheet_name='summary')\n"]},
]
nb['cells'].extend(append)
nb_path.write_text(json.dumps(nb,indent=2))
print(json.dumps({"log":str(LOG_PATH),"workbook":str(workbook_path),"notebook":str(nb_path),"rows_added":len(new_df),"summary":summary_rows},indent=2))
