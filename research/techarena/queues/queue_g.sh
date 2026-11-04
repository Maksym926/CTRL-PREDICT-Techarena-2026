#!/usr/bin/env bash
# Reserve MLP submissions (4 Nov morning): after the context MLP lr 3e-4 submission (started 05:15 by hand), build the
# context MLP lr 2e-4 submission. Both only become part of a final file if their validation blend passes the rule
# (held-out >= +0.0001 on both halves over package F). Log: logs/queue_g.log
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_g.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }
until grep -q "wrote .*safe_v2_step7_nn_s30_ctx_lr3.csv" logs/nn_step7_s30_ctx_lr3_submit.log 2>/dev/null; do sleep 60; done
say "G: context lr3 submission done; starting context lr2 submission (3 seeds, patience 5)"
$PY research/techarena/nn_step7.py --seq stage3_h50_s30 --context --lr 0.0002 --max-epochs 30 --patience 5 --threads 10 \
    --tag _s30_ctx_lr2 --submit > logs/nn_step7_s30_ctx_lr2_submit.log 2>&1 || { say "G: context lr2 submission FAILED"; exit 1; }
say "G: $(grep -E 'chose|wrote' logs/nn_step7_s30_ctx_lr2_submit.log | tr '\n' ' ' | cut -c1-300)"
say "queue G finished"
