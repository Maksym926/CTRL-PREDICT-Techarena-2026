@echo off
rem Sequence-model screening 2 on the second PC's GPU (4 Nov): item vectors learning slower or frozen, a smaller model,
rem a lower learning rate. Inner holdout of the evaluation fit, 3 seeds each. Summary lines: logs\seq_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set SEQ=research\techarena\seq_model.py --data data\parquet --out research\techarena\cache\seq_screen --mode full --inner-only --hist 50 --seeds 3 --max-epochs 4 --patience 2
echo [%date% %time%] seq screen 2 start >> logs\seq_exp.log
call :run il03 --item-lr-mult 0.3
call :run il01 --item-lr-mult 0.1
call :run il0 --item-lr-mult 0
call :run d32 --dim 32
call :run lr5 --lr 0.0005
echo [%date% %time%] seq screen 2 done >> logs\seq_exp.log
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
