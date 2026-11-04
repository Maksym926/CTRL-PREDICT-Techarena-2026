@echo off
rem Batch 6 on the second PC: the MLP stack trained per candidate (--binary: logistic loss, sigmoid outputs) at two
rem learning rates, as a more different third model for the final blend. Summary lines: logs\lgb_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 10 --binary
echo [%date% %time%] batch 6 start >> logs\lgb_exp.log
echo [%date% %time%] nn bin started >> logs\lgb_exp.log
%PY% %NN% --tag _s30_bin > logs\nn_s30_bin.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_bin.log >> logs\lgb_exp.log
echo [%date% %time%] nn bin lr5 started >> logs\lgb_exp.log
%PY% %NN% --lr 0.0005 --max-epochs 16 --tag _s30_bin_lr5 > logs\nn_s30_bin_lr5.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_bin_lr5.log >> logs\lgb_exp.log
echo [%date% %time%] batch 6 done >> logs\lgb_exp.log
exit /b 0
