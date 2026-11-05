#!/usr/bin/env bash
# After queue_d.sh: if package D was built, apply final_blend.py (factorised whether + conditional-logit which with the
# raw sequence shares) to D's two models -> submissions/safe_v2_step7_d_final.csv. Log: logs/queue_d.log
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_d.log
say() { echo "[$(date +%H:%M:%S)] final: $*" | tee -a "$Q"; }

until grep -qE "^[[0-9:]+] queue finished$" "$Q" 2>/dev/null; do sleep 60; done
if [ ! -f submissions/safe_v2_step7_d.csv ] || [ ! -f submissions/safe_v2_step7_softmax_d.csv ]; then
    say "package D was not built; the final file stays submissions/safe_v2_step7_s30_final_c.csv"
    exit 0
fi
$PY research/techarena/final_blend.py --val-a data/validation/safe_v2_step7_d_val_c4.parquet \
    --val-b data/validation/softmax_step7_d_val_c4.parquet --sub-a submissions/safe_v2_step7_d.csv \
    --sub-b submissions/safe_v2_step7_softmax_d.csv --seq50 stage3_h50_s30 \
    --out submissions/safe_v2_step7_d_final.csv > logs/final_blend_d.log 2>&1 || { say "final blend FAILED"; exit 1; }
$PY src/check_submission.py submissions/safe_v2_step7_d_final.csv data/evaluation.csv > logs/check_d_final.log 2>&1
say "$(grep -E 'final_on_validation|held_out_gains' -A2 logs/final_blend_d.log | tr -d '\n ' ) -> submissions/safe_v2_step7_d_final.csv; check: $(tail -1 logs/check_d_final.log)"
