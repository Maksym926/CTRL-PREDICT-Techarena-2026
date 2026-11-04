@echo off
rem Batch 16 on the second PC's GPU (4 Nov 16:45): day-6 validation of package G's third-model MLPs
rem (nn_step7.py --val6: train on the submission frames' windows 0.5-5.5 with targets before day 6, score window 6.0).
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda --val6
echo [%date% %time%] batch 16 (GPU, day-6 validation) start >> logs\gpu_exp.log
call :run ctx_lr2_s24 --context --lr 0.0002 --max-epochs 30 --patience 5 --seeds 24
call :run attn1_lr2_s12 --attn 1 --lr 0.0002 --max-epochs 30 --patience 4 --seeds 12
echo [%date% %time%] batch 16 (GPU, day-6 validation) done >> logs\gpu_exp.log
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
echo [%date% %time%] %TAG% (val6) started:%EXTRA% >> logs\gpu_exp.log
%PY% %NN%%EXTRA% --tag _s30_%TAG% > logs\nn_s30_%TAG%_val6.log 2>&1
findstr /C:"DAY-6" /C:"chose" /C:"Error" logs\nn_s30_%TAG%_val6.log >> logs\gpu_exp.log
exit /b 0
