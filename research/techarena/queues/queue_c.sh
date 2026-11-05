#!/usr/bin/env bash
# After queue B: softmax partner retrained on the 30-seed sequence features; its submission only if its blend with
# step 7 s30 beats package B (0.807593) by >= 0.0001 (fixed before the results).
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_c.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }

until grep -q "queue finished" logs/queue_b.log 2>/dev/null; do sleep 30; done
grep -q "package B:" logs/queue_b.log || { say "package B was not built (see logs/queue_b.log); stopping"; exit 1; }
say "C: softmax partner on the 30-seed sequence features, validation"
$PY research/techarena/softmax_step7.py --seq stage3_h50_s30 --binary data/validation/safe_v2_step7_s30_val_c4.parquet \
    --tag _s30 > logs/softmax_step7_s30_val.log 2>&1 || { say "partner s30 validation FAILED"; exit 1; }
$PY research/techarena/blend_val.py data/validation/safe_v2_step7_s30_val_c4.parquet \
    data/validation/softmax_step7_s30_val_c4.parquet > logs/blend_c.log 2>&1
BL=$(grep "50/50 rank blend" logs/blend_c.log | awk '{print $NF}')
say "blend(step 7 s30, partner s30) = $BL (package B 0.807593; build if >= 0.807693)"
if $PY -c "import sys; sys.exit(0 if float('$BL') >= 0.807693 else 1)"; then
    say "C: GO, building the partner s30 submission"
    $PY research/techarena/softmax_step7.py --submit --seq stage3_h50_s30 --tag _s30 > logs/softmax_step7_s30_submit.log 2>&1 \
        || { say "partner s30 submission FAILED"; exit 1; }
    $PY research/techarena/blend_csv.py submissions/safe_v2_step7_s30.csv submissions/safe_v2_step7_softmax_s30.csv \
        submissions/safe_v2_step7_s30_blend_c.csv >> "$Q" 2>&1
    $PY src/check_submission.py submissions/safe_v2_step7_s30_blend_c.csv data/evaluation.csv > logs/check_c.log 2>&1
    say "package C: submissions/safe_v2_step7_s30_blend_c.csv (local $BL); check: $(tail -1 logs/check_c.log)"
else
    say "C: STOP (below +0.0001 over B); final candidate stays package B"
fi
say "queue finished"
