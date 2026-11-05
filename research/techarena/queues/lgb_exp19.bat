@echo off
rem Batch 19 on the second PC GPU (4 Nov 17:35): global pairwise AUC loss (--auc-loss) on the wday context MLP, validation.
rem reference: ctx lr 2e-4 --window-day 3 seeds 0.808138 alone.
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda --window-day
echo [%date% %time%] batch 19 (GPU) start >> logs\gpu_exp.log
call :run ctx_lr2_wday_auc1 --context --lr 0.0002 --max-epochs 30 --patience 5 --auc-loss 1.0
call :run ctx_lr2_wday_auc03 --context --lr 0.0002 --max-epochs 30 --patience 5 --auc-loss 0.3
echo [%date% %time%] batch 19 (GPU) done >> logs\gpu_exp.log
exit /b 0

:run
set TAG=%1
shift
set EXTRA=
:collect
if "%~1"=="" goto go
set EXTRA=%EXTRA% %1
shift
goto collect
:go
echo [%date% %time%] %TAG% started:%EXTRA% >> logs\gpu_exp.log
%PY% %NN%%EXTRA% --tag _s30_%TAG% > logs\nn_s30_%TAG%.log 2>&1
findstr /C:"NN STEP 7" /C:"chose" /C:"wrote" /C:"Error" logs\nn_s30_%TAG%.log >> logs\gpu_exp.log
exit /b 0
