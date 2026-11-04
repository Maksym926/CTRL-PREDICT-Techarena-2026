@echo off
rem Batch 18 on the second PC's GPU (4 Nov 16:40): --window-day MLPs for a G-like mix, submission side (full frames).
rem writes submissionssafe_v2_step7_nn_s30_ctx_lr2_wday_s24.csv and ..._attn1_lr2_wday_s12.csv there.
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda --window-day
echo [%date% %time%] batch 18 (GPU) start >> logs\gpu_exp.log
call :run ctx_lr2_wday_s24 --context --lr 0.0002 --max-epochs 30 --patience 5 --seeds 24 --submit
call :run attn1_lr2_wday_s12 --attn 1 --lr 0.0002 --max-epochs 30 --patience 4 --seeds 12 --submit
echo [%date% %time%] batch 18 (GPU) done >> logs\gpu_exp.log
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
