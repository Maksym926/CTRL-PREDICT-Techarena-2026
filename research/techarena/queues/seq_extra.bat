@echo off
rem Extra sequence models for the final blend only (4 Nov): 10 new seeds each of three variants (base, weight
rem average 0.99, histories of 100 clicks), fitted at the validation boundary (day 4) and at the evaluation boundary, so their
rem within-event shares can join final_blend.py --seq-extra without retraining the stack. Summary: logs\seq_exp.log
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set SEQX=research\techarena\seq_model.py --data data\parquet --hist 50 --seeds 10 --max-epochs 4 --patience 2
set OUT=research\techarena\cache\seq
echo [%date% %time%] seq extra start >> logs\seq_exp.log
call :run base 100 seqx_base
call :run ema 200 seqx_ema --ema 0.99
call :run h100 300 seqx_h100 --hist 100
echo [%date% %time%] seq extra done >> logs\seq_exp.log
exit /b 0

:run
set NAME=%1
set SEED=%2
set DIR=%3
shift
shift
shift
set EXTRA=
:collect
if "%~1"=="" goto go
set EXTRA=%EXTRA% %1
shift
goto collect
:go
for %%M in (val full) do (
  echo [%date% %time%] %NAME% %%M started: seeds %SEED%+ %EXTRA% >> logs\seq_exp.log
  %PY% %SEQX% --mode %%M --seed %SEED% --out %OUT%\%DIR%%EXTRA% > logs\seqx_%NAME%_%%M.log 2>&1
  findstr /C:"inner epoch" /C:"VALIDATION AUC" /C:"wrote" /C:"Error" logs\seqx_%NAME%_%%M.log >> logs\seq_exp.log
)
exit /b 0
