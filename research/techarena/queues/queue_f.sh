#!/usr/bin/env bash
# Package F: C's binary + the tuned softmax partner (package E's) + the MLP stack (nn_step7.py), combined by
# final_blend.py (validation: held-out 0.80788 / 0.81009, about +0.0012 over package C's rank blend).
# Runs after queue_e.sh (needs its partner submission and this PC's CPU). Log: logs/queue_f.log
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_f.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }

until grep -qE "queue E finished|FAILED" logs/queue_e.log 2>/dev/null; do sleep 60; done
grep -q "FAILED" logs/queue_e.log && { say "F: queue E failed, stopping"; exit 1; }
say "F: NN submission (3 seeds, epochs <= 8)"
$PY research/techarena/nn_step7.py --seq stage3_h50_s30 --tag _s30 --max-epochs 8 --threads 10 --submit \
    > logs/nn_step7_s30_submit.log 2>&1 || { say "F: NN submission FAILED"; exit 1; }
say "F: $(grep -E 'chose|wrote' logs/nn_step7_s30_submit.log | tr '\n' ' ' | cut -c1-300)"
$PY research/techarena/final_blend.py --val-a data/validation/safe_v2_step7_s30_val_c4.parquet \
    --val-b data/validation/softmax_step7_s30_opt4_val_c4.parquet --val-c data/validation/nn_step7_s30_val_c4.parquet \
    --sub-a submissions/safe_v2_step7_s30.csv --sub-b submissions/safe_v2_step7_softmax_s30_opt4.csv \
    --sub-c submissions/safe_v2_step7_nn_s30.csv --seq50 stage3_h50_s30 \
    --out submissions/safe_v2_step7_s30_final_f.csv > logs/final_blend_f.log 2>&1 || { say "F: final blend FAILED"; exit 1; }
$PY src/check_submission.py submissions/safe_v2_step7_s30_final_f.csv data/evaluation.csv > logs/check_f.log 2>&1
say "F: final_on_validation $(grep -o '"final_on_validation": [0-9.]*' logs/final_blend_f.log | cut -d' ' -f2); check: $(tail -1 logs/check_f.log)"
say "queue F finished"
