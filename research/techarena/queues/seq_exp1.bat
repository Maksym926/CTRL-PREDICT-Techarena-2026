@echo off
rem Sequence-model screening on the second PC's GPU (4 Nov): anti-overfitting variants of the 50-click model, scored on
rem the inner holdout of the evaluation fit (29,940 events), 3 seeds each. Summary lines: logs\seq_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set SEQ=research\techarena\seq_model.py --data data\parquet --out research\techarena\cache\seq_screen --mode full --inner-only --hist 50 --seeds 3 --max-epochs 4 --patience 2
echo [%date% %time%] seq screen 1 start >> logs\seq_exp.log
call :run base
call :run nouser --no-user
call :run hd20 --hist-drop 0.2
call :run nouser_hd20 --no-user --hist-drop 0.2
call :run l2 --emb-l2 1.0
call :run do30 --dropout 0.3
echo [%date% %time%] seq screen 1 done >> logs\seq_exp.log
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
