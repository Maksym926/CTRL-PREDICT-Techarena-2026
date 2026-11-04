#!/usr/bin/env bash
# Package G (4 Nov): F's binary + tuned softmax partner + a new third model = (2 x context MLP lr 2e-4, 24 seeds +
# 1 x attention MLP lr 2e-4, 12 seeds) / 3. Validation side is already built
# (data/validation/nn_step7_s30_g_mix_val_c4.parquet; held-out gain over F +0.000123 / +0.000190). The two submission
# MLPs are built on the second PC's GPU by D:\HuaweiTechArena\lgb_exp13.bat (log there: logs\gpu_exp.log).
# This script: waits for that batch, fetches the two submission files as compact float32 arrays (the Tailscale link is
# sometimes slow), rebuilds them here in the row order of submissions/safe_v2_step7_nn_s30.csv (checked by md5), mixes
# them 2:1, runs final_blend.py and the checker. Safe to re-run. Log: logs/build_g.log
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
R=yulia@100.103.24.101
RR='D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026'
Q=logs/build_g.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }

say "G: waiting for batch 13 on the second PC"
until ssh -o BatchMode=yes -o ConnectTimeout=20 "$R" "type $RR\\logs\\gpu_exp.log" 2>/dev/null | grep -q "batch 13 (GPU) done"; do sleep 60; done
ssh -o BatchMode=yes "$R" "type $RR\\logs\\gpu_exp.log" 2>/dev/null | tr -d '\r' | sed -n '/batch 13 (GPU) start/,$p' | tee -a "$Q"

# remote: CSV -> float32 .npy in the file's own row order + md5 of that order
ENC=$($PY -c "
import base64
src = r'''
import hashlib, numpy as np, pandas as pd
S = 'D:/HuaweiTechArena/CTRL-PREDICT-Techarena-2026/submissions/'
for t in ('ctx_lr2_s24', 'attn1_lr2_s12'):
    f = pd.read_csv(S + f'safe_v2_step7_nn_s30_{t}.csv')
    np.save(S + f'nn_sub_{t}.npy', f['predicted_probability'].to_numpy().astype(np.float32))
    print(t, len(f), hashlib.md5('|'.join(f['rowid'].astype(str)).encode()).hexdigest())
'''
print(base64.b64encode(src.encode()).decode())")
REMOTE=$(ssh -o BatchMode=yes "$R" "cd /d $RR && .venv\\Scripts\\python.exe -c \"import base64;exec(base64.b64decode('$ENC'))\"" 2>&1 | tr -d '\r')
echo "$REMOTE" | tee -a "$Q"
LOCAL=$($PY -c "
import hashlib, pandas as pd
r = pd.read_csv('submissions/safe_v2_step7_nn_s30.csv')['rowid'].astype(str)
print(hashlib.md5('|'.join(r).encode()).hexdigest())")
[ "$(echo "$REMOTE" | grep -c "$LOCAL")" = "2" ] || { say "G: row order differs from submissions/safe_v2_step7_nn_s30.csv ($LOCAL): STOP"; exit 1; }
say "G: row order matches ($LOCAL)"
for t in ctx_lr2_s24 attn1_lr2_s12; do
    until scp -q -o BatchMode=yes -o ConnectTimeout=20 "$R:D:/HuaweiTechArena/CTRL-PREDICT-Techarena-2026/submissions/nn_sub_$t.npy" submissions/; do
        say "G: copy of nn_sub_$t.npy failed, retrying"; sleep 30
    done
done
$PY - <<'EOF' 2>&1 | tee -a "$Q"
import hashlib, numpy as np, pandas as pd
ref = pd.read_csv("submissions/safe_v2_step7_nn_s30.csv")
for t in ("ctx_lr2_s24", "attn1_lr2_s12"):
    v = np.load(f"submissions/nn_sub_{t}.npy").astype(float)
    assert len(v) == len(ref), (t, len(v), len(ref))
    pd.DataFrame({"rowid": ref["rowid"], "predicted_probability": v}).to_csv(f"submissions/safe_v2_step7_nn_s30_{t}.csv", index=False)
    print("rebuilt", t, round(v.mean(), 4))
EOF
$PY research/techarena/queues/weighted_avg.py submissions/safe_v2_step7_nn_s30_g_mix.csv \
    submissions/safe_v2_step7_nn_s30_ctx_lr2_s24.csv:2 submissions/safe_v2_step7_nn_s30_attn1_lr2_s12.csv:1 2>&1 | tee -a "$Q"
$PY research/techarena/final_blend.py --val-a data/validation/safe_v2_step7_s30_val_c4.parquet \
    --val-b data/validation/softmax_step7_s30_opt4_val_c4.parquet --val-c data/validation/nn_step7_s30_g_mix_val_c4.parquet \
    --sub-a submissions/safe_v2_step7_s30.csv --sub-b submissions/safe_v2_step7_softmax_s30_opt4.csv \
    --sub-c submissions/safe_v2_step7_nn_s30_g_mix.csv --seq50 stage3_h50_s30 \
    --out submissions/safe_v2_step7_s30_final_g.csv > logs/final_blend_g.log 2>&1 || { say "G: final blend FAILED"; exit 1; }
$PY src/check_submission.py submissions/safe_v2_step7_s30_final_g.csv data/evaluation.csv > logs/check_g.log 2>&1
say "G: held-out $(tr -d '\n ' < logs/final_blend_g.log | grep -o '"held_out":[0-9.]*' | cut -d: -f2 | tr '\n' ' ')(F: 0.807880 0.810090); $(grep -o '"final_on_validation": [0-9.]*' logs/final_blend_g.log) (F: 0.809056); check: $(tail -1 logs/check_g.log)"
$PY -c "
import pandas as pd
a = pd.read_csv('submissions/safe_v2_step7_s30_final_g.csv'); b = pd.read_csv('submissions/safe_v2_step7_s30_final_f.csv')
m = a.merge(b, on='rowid'); print('Spearman G vs F', round(m.iloc[:, 1].corr(m.iloc[:, 2], method='spearman'), 5))" 2>&1 | tee -a "$Q"
say "package G finished"
