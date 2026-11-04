#!/usr/bin/env bash
# Wait until the eight files sent via Google Drive are on the second PC with their exact sizes (and the attention MLP's
# validation seeds are done), then run lgb_exp13.bat there (the many-seed MLP submissions on the GPU).
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
R=yulia@100.103.24.101
RR='D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026'
L=logs/second_pc_lgb.log
say() { echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
want="2166553822 205091961 19492050 18262669 17469760 14663166 12542423 21926275"
files='research\techarena\cache\lab\step5_full_train.parquet research\techarena\cache\lab\step5_full_test.parquet research\techarena\cache\seq\stage3_h50_s30\seq_b1737720000.parquet research\techarena\cache\seq\stage3_h50_s30\seq_b1737763200.parquet research\techarena\cache\seq\stage3_h50_s30\seq_b1737806400.parquet research\techarena\cache\seq\stage3_h50_s30\seq_b1737849600.parquet research\techarena\cache\seq\stage3_h50_s30\seq_b1737892800.parquet research\techarena\cache\seq\stage3_h50_s30\seq_full_eval.parquet'
sizes() {
    local out=""
    for f in $files; do
        s=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$R" "for %A in ($RR\$f) do @echo %~zA" 2>/dev/null | tr -d '\r')
        out="$out ${s:-0}"
    done
    echo $out
}
say "waiting for the Google Drive files on the second PC"
until [ "$(sizes)" = "$want" ]; do sleep 60; done
say "all eight files present with the right sizes"
until ssh -o BatchMode=yes -o ConnectTimeout=20 "$R" "type $RR\logs\nn_s30_attn1_lr2_o3s9.log" 2>/dev/null | grep -q "NN STEP 7"; do sleep 30; done
say "batch 13 (GPU submissions): starting"
ssh -o BatchMode=yes -o ServerAliveInterval=30 "$R" "D:\HuaweiTechArena\lgb_exp13.bat" >> "$L" 2>&1
say "batch 13 returned $?"
ssh -o BatchMode=yes "$R" "type $RR\logs\gpu_exp.log" 2>&1 | tr -d '\r' | grep -v "^$" | sed -n '/batch 13 (GPU) start/,$p'
