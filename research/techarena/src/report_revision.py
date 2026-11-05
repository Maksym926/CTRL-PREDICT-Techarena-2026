"""Publish corrected Phase C results and the C2 audit."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
D = json.loads((OUT / "phase_c_revision_results.json").read_text())
LOG = ROOT / "experiment_log.csv"

def rw(x):
    return float(x["reweight"]["joint_auc_supported"])

def ordinary(x):
    return float(x["auc"])

rows = []
for key, label in [("a3", "A3 half-day weighted"), ("a3_c1", "A3 + C1"), ("a3_c1_c2", "A3 + C1 + corrected C2")]:
    x = D["experiment"][key]
    rows.append({
        "phase": "C-correction", "experiment": label,
        "standard_val": (ordinary(x["DM23"]) + ordinary(x["DM24"])) / 2,
        "reweighted_val": x["DM_mean"], "long_gap_val_DM": x["DM_mean"],
        "twin_oracle": "", "twin_pipeline": "", "twin_vector_cosine": "",
        "kept": "PASS" if x["DM_mean"] - D["experiment_baseline_DM_mean"] >= .004 else "reject",
        "notes": f"DM23={rw(x['DM23']):.6f}; DM24={rw(x['DM24']):.6f}; DM_mean={x['DM_mean']:.6f}; gain_vs_exp_baseline={x['DM_mean']-D['experiment_baseline_DM_mean']:.6f}",
    })
p = D["production_full"]
rows.append({
    "phase": "production", "experiment": "original production full-data baseline",
    "standard_val": (ordinary(p["DM23"]) + ordinary(p["DM24"])) / 2,
    "reweighted_val": p["DM_mean"], "long_gap_val_DM": p["DM_mean"],
    "twin_oracle": "", "twin_pipeline": "", "twin_vector_cosine": "",
    "kept": "reference",
    "notes": f"DM23={rw(p['DM23']):.6f}; DM24={rw(p['DM24']):.6f}; DM_mean={p['DM_mean']:.6f}",
})
f = D["final"]
rows.append({
    "phase": "final", "experiment": "A3 + C1 + corrected C2, 3-seed lr=.03",
    "standard_val": (ordinary(f["folds"]["DM23"]) + ordinary(f["folds"]["DM24"])) / 2,
    "reweighted_val": f["DM_mean"], "long_gap_val_DM": f["DM_mean"],
    "twin_oracle": "", "twin_pipeline": "", "twin_vector_cosine": "",
    "kept": "SAFE_v2 built" if f["beats_production_gate"] else "reject",
    "notes": f"DM23={rw(f['folds']['DM23']):.6f}; DM24={rw(f['folds']['DM24']):.6f}; DM_mean={f['DM_mean']:.6f}; full-data gate delta={f['DM_mean']-p['DM_mean']:.6f}",
})
old = pd.read_csv(LOG)
new_df = pd.DataFrame(rows)
new_df = new_df[~new_df["experiment"].isin(old["experiment"].astype(str))]
log = pd.concat([old, new_df], ignore_index=True)
log.to_csv(LOG, index=False, float_format="%.12f")

summary = []
for key, label, mode, file in [
    ("a3", "A3", "experiment", ""), ("a3_c1", "A3 + C1", "experiment", ""),
    ("a3_c1_c2", "A3 + C1 + corrected C2", "experiment", ""),
    ("production_full", "original production", "full-data", "outputs/submission_choice_features.csv"),
    ("final", "A3 + C1 + corrected C2", "final 3-seed", "outputs/SAFE_v2.csv"),
]:
    x = D["production_full"] if key == "production_full" else D["final"] if key == "final" else D["experiment"][key]
    if key == "final":
        d23, d24, dm = x["folds"]["DM23"], x["folds"]["DM24"], x["DM_mean"]
    else:
        d23, d24, dm = x["DM23"], x["DM24"], x["DM_mean"]
    summary.append({"model": label, "mode": mode, "DM23": rw(d23), "DM24": rw(d24), "DM_mean": dm, "file": file})
summary_df = pd.DataFrame(summary)

bucket_rows = []
for key, label in [("a3", "A3"), ("a3_c1", "A3 + C1"), ("a3_c1_c2", "A3 + C1 + corrected C2")]:
    x = D["experiment"][key]
    for fold in ["DM23", "DM24"]:
        for typ in ["history_length", "gap"]:
            for bucket, val in x[fold]["buckets"][typ].items():
                bucket_rows.append({"model": label, "mode": "experiment", "fold": fold, "bucket_type": typ, "bucket": bucket, "rows": val["rows"], "share": val["share"], "auc": val["auc"]})
for fold in ["DM23", "DM24"]:
    x = D["final"]["folds"][fold]
    for typ in ["history_length", "gap"]:
        for bucket, val in x["buckets"][typ].items():
            bucket_rows.append({"model": "A3 + C1 + corrected C2", "mode": "final 3-seed", "fold": fold, "bucket_type": typ, "bucket": bucket, "rows": val["rows"], "share": val["share"], "auc": val["auc"]})
buckets = pd.DataFrame(bucket_rows)

audit_rows = []
C2_COLUMNS = ["shrink_dot", "shrink_cos", "shrink_norm", "posterior_std", "posterior_utility_std"]
for fold, vals in D["c2_audit"]["folds"].items():
    for feature in C2_COLUMNS:
        audit_rows.append({"fold": fold, "feature": feature, "variance": vals["variance"][feature], "single_feature_auc": vals["single_feature_auc"][feature], "lightgbm_gain": vals["lightgbm_gain"][feature], "reached_training_matrix": feature in vals["training_columns"]})
audit = pd.DataFrame(audit_rows)

workbook = OUT / "phase_c_revision_report.xlsx"
with xlsxwriter.Workbook(workbook, {"nan_inf_to_errors": True}) as wb:
    header = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    num = wb.add_format({"num_format": "0.000000", "border": 1})
    txt = wb.add_format({"border": 1})
    for name, df in [("model_summary", summary_df), ("bucket_breakdown", buckets), ("c2_audit", audit), ("experiment_log", log)]:
        ws = wb.add_worksheet(name)
        for c, col in enumerate(df.columns):
            ws.write(0, c, col, header)
        for r, rowv in enumerate(df.itertuples(index=False), 1):
            for c, val in enumerate(rowv):
                if isinstance(val, (float, int)) and pd.notna(val):
                    ws.write_number(r, c, float(val), num)
                else:
                    ws.write(r, c, str(val), txt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        ws.set_column(0, len(df.columns) - 1, 20)
    ws = wb.add_worksheet("decision")
    ws.write(0, 0, "experiment baseline DM_mean", header)
    ws.write_number(0, 1, D["experiment_baseline_DM_mean"], num)
    ws.write(1, 0, "production full-data DM_mean", txt)
    ws.write_number(1, 1, D["production_full"]["DM_mean"], num)
    ws.write(2, 0, "final full-data DM_mean", txt)
    ws.write_number(2, 1, D["final"]["DM_mean"], num)
    ws.write(3, 0, "SAFE_v2 checks", txt)
    ws.write(3, 1, "outputs/SAFE_v2_checks.json", txt)
    ws.set_column(0, 1, 42)

summary_path = OUT / "phase_c_revision_summary.csv"
summary_df.to_csv(summary_path, index=False)

nb_path = ROOT / "notebooks" / "run_all.ipynb"
nb = json.loads(nb_path.read_text())
nb["cells"].extend([
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Corrected Phase C gate and final-mode result\n",
        "The previous comparison used production DM24 as if it were DM_mean. The corrected experiment baseline is DM_mean=0.788117. C2 was corrected: C1 has shrink_dot, shrink_cos and shrink_norm; C2 adds posterior_std and posterior_utility_std.\n\n",
        "A3+C1+C2 reaches experiment DM_mean=0.795972 (+0.007855). Full-data production is DM_mean=0.791842; the final three-seed lr=0.03 model reaches DM_mean=0.800742 (+0.008900), so SAFE_v2 was built and mechanically checked."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import pandas as pd\n",
        "pd.read_csv('outputs/phase_c_revision_summary.csv')\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import json\n",
        "json.load(open('outputs/SAFE_v2_checks.json'))\n"
    ]},
])
nb_path.write_text(json.dumps(nb, indent=2))
print(json.dumps({"log": str(LOG), "workbook": str(workbook), "summary_csv": str(summary_path), "notebook": str(nb_path), "rows_added": len(new_df)}, indent=2))
