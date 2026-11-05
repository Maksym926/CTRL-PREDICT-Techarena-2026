@echo off
rem Batch 5 on the second PC: the context MLP at learning rate 5e-4 (best alone in batch 4, 0.807872) with 3 new refit
rem seeds (luck check), at 3e-4, and 512 wide / dropout 0.2. Summary lines: logs\lgb_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 10 --context
echo [%date% %time%] batch 5 start >> logs\lgb_exp.log
echo [%date% %time%] nn ctx lr5 seeds 3-5 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0005 --max-epochs 16 --seed-offset 3 --tag _s30_ctx_lr5_o3 > logs\nn_s30_ctx_lr5_o3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_lr5_o3.log >> logs\lgb_exp.log
echo [%date% %time%] nn ctx lr3 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0003 --max-epochs 20 --tag _s30_ctx_lr3 > logs\nn_s30_ctx_lr3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_lr3.log >> logs\lgb_exp.log
echo [%date% %time%] nn ctx w512 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0005 --max-epochs 16 --hidden 512 --dropout 0.2 --tag _s30_ctx_w512 > logs\nn_s30_ctx_w512.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_w512.log >> logs\lgb_exp.log
echo [%date% %time%] batch 5 done >> logs\lgb_exp.log
exit /b 0
