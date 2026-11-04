@echo off
rem Batch 9 on the second PC (4 Nov): the MLP stack with self-attention over the event's 10 candidates (--attn), as a
rem stronger form of --context. Validation runs; summary lines: logs\lgb_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 10 --patience 4
echo [%date% %time%] batch 9 start >> logs\lgb_exp.log
echo [%date% %time%] nn attn1 lr3 started >> logs\lgb_exp.log
%PY% %NN% --attn 1 --lr 0.0003 --max-epochs 20 --tag _s30_attn1_lr3 > logs\nn_s30_attn1_lr3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_attn1_lr3.log >> logs\lgb_exp.log
echo [%date% %time%] nn attn2 lr3 started >> logs\lgb_exp.log
%PY% %NN% --attn 2 --lr 0.0003 --max-epochs 20 --tag _s30_attn2_lr3 > logs\nn_s30_attn2_lr3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_attn2_lr3.log >> logs\lgb_exp.log
echo [%date% %time%] nn attn1 started >> logs\lgb_exp.log
%PY% %NN% --attn 1 --max-epochs 12 --tag _s30_attn1 > logs\nn_s30_attn1.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_attn1.log >> logs\lgb_exp.log
echo [%date% %time%] batch 9 done >> logs\lgb_exp.log
exit /b 0
