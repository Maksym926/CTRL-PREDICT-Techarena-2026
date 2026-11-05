#!/usr/bin/env bash
# Wait for the second PC's partial choice-model averages, then copy them into this PC's cache.
cd "G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026" || exit 1
R=yulia@100.103.24.101
RC=/D:/HuaweiTechArena/CTRL-PREDICT-Techarena-2026/research/techarena/cache
Q=logs/queue_d.log
say() { echo "[$(date +%H:%M:%S)] transfer: $*" | tee -a "$Q"; }
rm -f logs/queue_d_transfer.done

until ssh -o BatchMode=yes -o ConnectTimeout=20 "$R" \
    "if exist D:\\HuaweiTechArena\\CTRL-PREDICT-Techarena-2026\\research\\techarena\\cache\\choicemf_v2_part\\seeds.json (echo READY)" \
    2>/dev/null | grep -q READY; do sleep 60; done
say "second PC finished its seeds; copying the partial averages"
for m in d12 d16 v2; do
    rm -rf "research/techarena/cache/choicemf_${m}_part"
    scp -r -o BatchMode=yes -o ServerAliveInterval=30 "$R:$RC/choicemf_${m}_part" research/techarena/cache/ \
        || { say "copy of choicemf_${m}_part FAILED"; exit 1; }
    say "choicemf_${m}_part: $(ls research/techarena/cache/choicemf_${m}_part | wc -l) files, $(cat research/techarena/cache/choicemf_${m}_part/seeds.json)"
done
touch logs/queue_d_transfer.done
say "done"
