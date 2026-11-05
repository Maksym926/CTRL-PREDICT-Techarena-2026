@echo off
rem Mult-VAE regularisation screen on the second PC's GPU (4 Nov 16:00): validation boundary, 1 seed each.
rem The base run (dim 32, target all, beta 0.2, drop 0.5, no wd) overfits after 5 epochs (inner which 0.779 -> 0.751).
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set M=research\techarena\mvae_model.py --data data\parquet --mode val --seeds 1 --threads 2
echo [%date% %time%] mvae screen start >> logs\mvae_screen.log
call :run held --target held
call :run d12 --dim 12
call :run d12held --dim 12 --target held
call :run wd1 --wd 1.0
call :run beta1 --beta 1.0
call :run drop8held --drop 0.8 --target held
call :run h200d16 --hidden 200 --dim 16 --target held --wd 0.3
echo [%date% %time%] mvae screen done >> logs\mvae_screen.log
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
%PY% %M%%EXTRA% --out research\techarena\cache\seq\mvae_%TAG% > logs\mvae_%TAG%.log 2>&1
echo [%date% %time%] %TAG%:%EXTRA% >> logs\mvae_screen.log
findstr /C:"start:" /C:"best inner" /C:"averaged" /C:"Error" logs\mvae_%TAG%.log >> logs\mvae_screen.log
exit /b 0
