@echo off
rem Batch 7 on the second PC: lower learning rates for the MLP stack (context head at 2e-4; plain head at 3e-4, the
rem plain MLP being the better blend partner so far). Summary lines: logs\lgb_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 10
echo [%date% %time%] batch 7 start >> logs\lgb_exp.log
echo [%date% %time%] nn lr3 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0003 --max-epochs 20 --tag _s30_lr3 > logs\nn_s30_lr3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_lr3.log >> logs\lgb_exp.log
echo [%date% %time%] nn ctx lr2 started >> logs\lgb_exp.log
%PY% %NN% --context --lr 0.0002 --max-epochs 30 --tag _s30_ctx_lr2 > logs\nn_s30_ctx_lr2.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_lr2.log >> logs\lgb_exp.log
echo [%date% %time%] batch 7 done >> logs\lgb_exp.log
exit /b 0
