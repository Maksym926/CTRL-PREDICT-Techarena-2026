@echo off
rem Batch 10 on the second PC's GPU (4 Nov): MLP stack with self-attention over the candidates (--attn), on CUDA, next to batch 9 (CPU: attn1 lr3, attn2 lr3, attn1 lr 1e-3).
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda --patience 4
echo [%date% %time%] batch 10 (GPU) start >> logs\gpu_exp.log
call :run attn1_lr2 --attn 1 --lr 0.0002 --max-epochs 30
call :run attn1ctx_lr3 --attn 1 --context --lr 0.0003 --max-epochs 20
call :run attn1w_lr3 --attn 1 --hidden 512 --dropout 0.2 --lr 0.0003 --max-epochs 20
echo [%date% %time%] batch 10 (GPU) done >> logs\gpu_exp.log
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
findstr /C:"NN STEP 7" /C:"chose" /C:"Error" logs\nn_s30_%TAG%.log >> logs\gpu_exp.log
exit /b 0
