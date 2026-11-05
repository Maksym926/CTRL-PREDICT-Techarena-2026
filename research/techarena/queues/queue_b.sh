#!/usr/bin/env bash
# After the softmax partner build: check it, blend it with step 7 (package A), validate step 7 with the 30-seed
# sequence features, and build that submission only if its blend beats A by >= 0.0001 (fixed before the results).
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_b.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }

until grep -q "^exit" logs/softmax_step7_submit.log 2>/dev/null; do sleep 30; done
grep -q "^exit 0" logs/softmax_step7_submit.log || { say "partner build FAILED (logs/softmax_step7_submit.log)"; exit 1; }
say "partner build done: $(grep -v Warning logs/softmax_step7_submit.log | grep wrote | tail -1)"

$PY src/check_submission.py submissions/safe_v2_step7_softmax.csv data/evaluation.csv > logs/check_softmax7.log 2>&1
say "check partner: $(tail -1 logs/check_softmax7.log)"
$PY research/techarena/blend_csv.py submissions/safe_v2_step7.csv submissions/safe_v2_step7_softmax.csv \
    submissions/safe_v2_step7_blend2.csv >> "$Q" 2>&1
$PY src/check_submission.py submissions/safe_v2_step7_blend2.csv data/evaluation.csv > logs/check_blend2.log 2>&1
say "package A (step 7 + new partner, local 0.807361): submissions/safe_v2_step7_blend2.csv; check: $(tail -1 logs/check_blend2.log)"

say "B: step 7 validation with the 30-seed sequence features"
$PY research/techarena/safe_v2_step7.py --seq stage3_h50_s30 --tag _s30 > logs/safe_v2_step7_s30_val.log 2>&1 \
    || { say "step 7 s30 validation FAILED"; exit 1; }
say "$(grep -v Warning logs/safe_v2_step7_s30_val.log | grep -E 'STEP 7|change vs step 7' | tr '\n' ' ')"
$PY research/techarena/blend_val.py data/validation/safe_v2_step7_s30_val_c4.parquet \
    data/validation/softmax_step7_val_c4.parquet > logs/blend_s30.log 2>&1
BL=$(grep "50/50 rank blend" logs/blend_s30.log | awk '{print $NF}')
say "blend(step 7 s30, new partner) = $BL (package A 0.807361; build if >= 0.807461)"
if $PY -c "import sys; sys.exit(0 if float('$BL') >= 0.807461 else 1)"; then
    say "B: GO, building the step 7 s30 submission"
    $PY research/techarena/safe_v2_step7.py --seq stage3_h50_s30 --tag _s30 --submit > logs/safe_v2_step7_s30_submit.log 2>&1 \
        || { say "step 7 s30 submission FAILED"; exit 1; }
    $PY research/techarena/blend_csv.py submissions/safe_v2_step7_s30.csv submissions/safe_v2_step7_softmax.csv \
        submissions/safe_v2_step7_s30_blend.csv >> "$Q" 2>&1
    $PY src/check_submission.py submissions/safe_v2_step7_s30_blend.csv data/evaluation.csv > logs/check_s30_blend.log 2>&1
    say "package B: submissions/safe_v2_step7_s30_blend.csv (local $BL); check: $(tail -1 logs/check_s30_blend.log)"
else
    say "B: STOP (below +0.0001 over A); final candidate stays package A"
fi
say "queue finished"
