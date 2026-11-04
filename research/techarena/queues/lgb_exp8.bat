@echo off
rem Batch 8 on the second PC: context MLP at learning rate 1e-4 (the blend gain grew as the rate fell: 5e-4 -> 3e-4 ->
rem 2e-4), and 2e-4 again with refit seeds 3-5 (seed check, and a 6-seed average). Summary lines: logs\lgb_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 10 --context --patience 5
echo [%date% %time%] batch 8 start >> logs\lgb_exp.log
echo [%date% %time%] nn ctx lr1 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0001 --max-epochs 40 --tag _s30_ctx_lr1 > logs\nn_s30_ctx_lr1.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_lr1.log >> logs\lgb_exp.log
echo [%date% %time%] nn ctx lr2 seeds 3-5 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0002 --max-epochs 30 --seed-offset 3 --tag _s30_ctx_lr2_o3 > logs\nn_s30_ctx_lr2_o3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_lr2_o3.log >> logs\lgb_exp.log
echo [%date% %time%] batch 8 done >> logs\lgb_exp.log
exit /b 0
