@echo off
rem Batch 15 on the second PC's GPU (4 Nov 16:30): does the MLP stack under-use late (data-rich) taste features?
rem --window-day (window boundary as an input) and --window-weight (later windows weigh more), context MLP lr 2e-4
rem (G's main third-model type; its plain 3-seed validation run: 0.807954 alone). Validation, 3 seeds each.
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda --context --lr 0.0002 --max-epochs 30 --patience 5
echo [%date% %time%] batch 15 (GPU) start >> logs\gpu_exp.log
call :run ctx_lr2_wday --window-day
call :run ctx_lr2_ww1 --window-weight 1
call :run ctx_lr2_wday_ww1 --window-day --window-weight 1
echo [%date% %time%] batch 15 (GPU) done >> logs\gpu_exp.log
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
