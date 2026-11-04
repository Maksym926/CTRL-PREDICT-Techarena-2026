@echo off
rem Batch 4 on the second PC: MLP-stack variants (validation): candidate head with event context (two learning
rem rates) and a wider, more regularised body. Summary lines: logs\lgb_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 10
echo [%date% %time%] batch 4 start >> logs\lgb_exp.log
echo [%date% %time%] nn ctx started >> logs\lgb_exp.log
%PY% %NN% --context --tag _s30_ctx > logs\nn_s30_ctx.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx.log >> logs\lgb_exp.log
echo [%date% %time%] nn ctx lr5 started >> logs\lgb_exp.log
%PY% %NN% --context --lr 0.0005 --max-epochs 16 --tag _s30_ctx_lr5 > logs\nn_s30_ctx_lr5.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_ctx_lr5.log >> logs\lgb_exp.log
echo [%date% %time%] nn d3 started >> logs\lgb_exp.log
%PY% %NN% --hidden 512 --dropout 0.3 --lr 0.0005 --max-epochs 16 --tag _s30_d3 > logs\nn_s30_d3.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_d3.log >> logs\lgb_exp.log
echo [%date% %time%] batch 4 done >> logs\lgb_exp.log
exit /b 0
