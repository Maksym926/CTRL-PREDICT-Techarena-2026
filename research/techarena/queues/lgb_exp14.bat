@echo off
rem Batch 14 on the second PC's GPU (4 Nov 16:15): MLP stack with piecewise-linear input encoding (nn_step7.py --ple),
rem validation runs (3 seeds each), as new MLP types for the final blend's third model (package G = context x2 + attn).
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda
echo [%date% %time%] batch 14 (GPU) start >> logs\gpu_exp.log
call :run ple16 --ple 16 --max-epochs 20 --patience 4
call :run ple16_ctx_lr2 --ple 16 --context --lr 0.0002 --max-epochs 30 --patience 5
call :run ple16_attn1_lr2 --ple 16 --attn 1 --lr 0.0002 --max-epochs 30 --patience 4
echo [%date% %time%] batch 14 (GPU) done >> logs\gpu_exp.log
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
