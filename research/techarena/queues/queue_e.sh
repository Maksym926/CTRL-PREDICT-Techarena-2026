#!/usr/bin/env bash
# Package E: package C's binary model + the softmax partner with step 4's tuned LightGBM parameters (validation: alone
# 0.807807 vs 0.807347, final blend held-out +0.0002 over C's final blend), combined by final_blend.py.
# Waits for the NN validation (nn_step7.py) to free the CPU. Log: logs/queue_e.log
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_e.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }

until [ -f research/techarena/outputs/nn_step7_s30.json ] || grep -q "Traceback" logs/nn_step7_s30_val.log 2>/dev/null; do
    sleep 30
done
say "E: partner submission with the tuned parameters (3 seeds, tree cap 4000)"
$PY research/techarena/softmax_step7.py --seq stage3_h50_s30 --params research/techarena/outputs/optuna_step4.json \
    --max-trees 4000 --tag _s30_opt4 --submit > logs/softmax_step7_s30_opt4_submit.log 2>&1 \
    || { say "E: partner submission FAILED"; exit 1; }
say "E: $(grep -E 'inner holdout chose|wrote' logs/softmax_step7_s30_opt4_submit.log | tr '\n' ' ' | cut -c1-300)"
$PY research/techarena/final_blend.py --val-a data/validation/safe_v2_step7_s30_val_c4.parquet \
    --val-b data/validation/softmax_step7_s30_opt4_val_c4.parquet --sub-a submissions/safe_v2_step7_s30.csv \
    --sub-b submissions/safe_v2_step7_softmax_s30_opt4.csv --seq50 stage3_h50_s30 \
    --out submissions/safe_v2_step7_s30_final_e.csv > logs/final_blend_e.log 2>&1 || { say "E: final blend FAILED"; exit 1; }
$PY src/check_submission.py submissions/safe_v2_step7_s30_final_e.csv data/evaluation.csv > logs/check_e.log 2>&1
say "E: final_on_validation $(grep -o '"final_on_validation": [0-9.]*' logs/final_blend_e.log | cut -d' ' -f2); held-out gains over factorised $(grep -A2 held_out_gains logs/final_blend_e.log | tail -2 | tr -d ' \n'); check: $(tail -1 logs/check_e.log)"
say "queue E finished"
