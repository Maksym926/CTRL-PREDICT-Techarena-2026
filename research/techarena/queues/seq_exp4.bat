@echo off
rem Sequence-model screening 4 on the second PC's GPU (4 Nov): the other direction from screen 2 (item vectors
rem learning faster, a higher learning rate, a wider model). Inner holdout of the evaluation fit, 3 seeds each. Summary lines: logs\seq_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set SEQ=research\techarena\seq_model.py --data data\parquet --out research\techarena\cache\seq_screen --mode full --inner-only --hist 50 --seeds 3 --max-epochs 4 --patience 2
echo [%date% %time%] seq screen 4 start >> logs\seq_exp.log
call :run il2 --item-lr-mult 2
call :run il3 --item-lr-mult 3
call :run lr2x --lr 0.002
call :run d96 --dim 96
echo [%date% %time%] seq screen 4 done >> logs\seq_exp.log
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
echo [%date% %time%] %TAG% started:%EXTRA% >> logs\seq_exp.log
%PY% %SEQ%%EXTRA% --tag _%TAG% > logs\seq_%TAG%.log 2>&1
findstr /C:"inner epoch" /C:"ensemble inner AUC" /C:"Error" logs\seq_%TAG%.log >> logs\seq_exp.log
exit /b 0
