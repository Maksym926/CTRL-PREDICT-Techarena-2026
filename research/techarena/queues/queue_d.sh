#!/usr/bin/env bash
# Package D: the four choice-model features averaged over seeds (cmf 5, cmf12 10, cmf16 5, cmfp 5) swapped into step 7
# (30-seed sequence features) and its softmax partner. Built only if the blend reaches package C (0.807766) + 0.0001.
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=logs/queue_d.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }
SW="cmf=choicemf_v2_avg cmf12=choicemf_d12_avg cmf16=choicemf_d16_avg cmfp=choicemf_pop_avg"

until grep -q "queue finished" logs/queue_c.log 2>/dev/null; do sleep 30; done
say "package C queue finished: $(grep 'package C' logs/queue_c.log | tail -1)"

for s in 1 2 3 4; do
    $PY research/techarena/cmf_seeds.py --model pop --seed $s > logs/cmf_pop_s$s.log 2>&1 || { say "cmfp seed $s FAILED"; exit 1; }
    say "cmfp seed $s done"
done
$PY research/techarena/cmf_seeds.py --model pop --merge 1 2 3 4 > logs/cmf_pop_merge.log 2>&1 || { say "cmfp merge FAILED"; exit 1; }
say "cmfp: 5 seeds averaged"

until [ -f logs/queue_d_transfer.done ]; do sleep 30; done
for spec in d12:9 d16:4 v2:4; do
    m=${spec%%:*}; n=${spec##*:}
    $PY research/techarena/cmf_seeds.py --model $m --merge --with choicemf_${m}_part:$n > logs/cmf_${m}_merge.log 2>&1 \
        || { say "$m merge FAILED"; exit 1; }
    say "$m: seed 0 + $n seeds from the second PC averaged"
done

say "D: step 7 validation with the averaged choice features"
$PY research/techarena/safe_v2_step7.py --seq stage3_h50_s30 --swap $SW --tag _d > logs/safe_v2_step7_d_val.log 2>&1 \
    || { say "step 7 D validation FAILED"; exit 1; }
say "$(grep -v Warning logs/safe_v2_step7_d_val.log | grep -E 'STEP 7|change vs step 7' | tr '\n' ' ')"
$PY research/techarena/softmax_step7.py --seq stage3_h50_s30 --swap $SW --binary data/validation/safe_v2_step7_d_val_c4.parquet \
    --tag _d > logs/softmax_step7_d_val.log 2>&1 || { say "partner D validation FAILED"; exit 1; }
$PY research/techarena/blend_val.py data/validation/safe_v2_step7_d_val_c4.parquet data/validation/softmax_step7_d_val_c4.parquet \
    > logs/blend_d.log 2>&1
say "$(tr '\n' ' ' < logs/blend_d.log)"
BL=$(grep "50/50 rank blend" logs/blend_d.log | awk '{print $NF}')
say "blend D = $BL (package C 0.807766; build if >= 0.807866)"
if $PY -c "import sys; sys.exit(0 if float('$BL') >= 0.807866 else 1)"; then
    say "D: GO, building both submissions"
    $PY research/techarena/safe_v2_step7.py --seq stage3_h50_s30 --swap $SW --tag _d --submit > logs/safe_v2_step7_d_submit.log 2>&1 \
        || { say "step 7 D submission FAILED"; exit 1; }
    $PY research/techarena/softmax_step7.py --seq stage3_h50_s30 --swap $SW --tag _d --submit > logs/softmax_step7_d_submit.log 2>&1 \
        || { say "partner D submission FAILED"; exit 1; }
    $PY research/techarena/blend_csv.py submissions/safe_v2_step7_d.csv submissions/safe_v2_step7_softmax_d.csv \
        submissions/safe_v2_step7_d_blend.csv >> "$Q" 2>&1
    $PY src/check_submission.py submissions/safe_v2_step7_d_blend.csv data/evaluation.csv > logs/check_d.log 2>&1
    say "package D: submissions/safe_v2_step7_d_blend.csv (local $BL); check: $(tail -1 logs/check_d.log)"
else
    say "D: STOP (below +0.0001 over C); package C stays the final candidate"
fi
say "queue finished"
