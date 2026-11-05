@echo off
rem Batch 12 on the second PC's GPU (4 Nov): more refit seeds of the two MLPs closest to the rule (base; context lr 2e-4),
rem 9 seeds per run, so their blend can be judged without seed noise. Summary lines: logs\gpu_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda --seeds 9
echo [%date% %time%] batch 12 (GPU) start >> logs\gpu_exp.log
call :run o3s9 --seed-offset 3
call :run o12s9 --seed-offset 12
call :run ctx_lr2_o6s9 --context --lr 0.0002 --max-epochs 30 --patience 5 --seed-offset 6
call :run ctx_lr2_o15s9 --context --lr 0.0002 --max-epochs 30 --patience 5 --seed-offset 15
echo [%date% %time%] batch 12 (GPU) done >> logs\gpu_exp.log
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
