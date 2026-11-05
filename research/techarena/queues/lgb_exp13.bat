@echo off
rem Batch 13 on the second PC's GPU (4 Nov): the many-seed MLPs for the submission (needs cache\lab\step5_full_*.parquet
rem and all 14 stage3_h50_s30 files, sent via Google Drive). (The 9 extra validation seeds of the attention MLP ran on
rem their own, tag _s30_attn1_lr2_o3s9.) Submission builds: context lr 2e-4 x 24 seeds, attention lr 2e-4 x 12 (package G).
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set NN=research\techarena\nn_step7.py --seq stage3_h50_s30 --threads 2 --device cuda
echo [%date% %time%] batch 13 (GPU) start >> logs\gpu_exp.log
call :run ctx_lr2_s24 --context --lr 0.0002 --max-epochs 30 --patience 5 --seeds 24 --submit
call :run attn1_lr2_s12 --attn 1 --lr 0.0002 --max-epochs 30 --patience 4 --seeds 12 --submit
echo [%date% %time%] batch 13 (GPU) done >> logs\gpu_exp.log
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
