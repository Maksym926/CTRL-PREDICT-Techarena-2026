@echo off
rem Batch 11 on the second PC's CPU (4 Nov): the softmax partner with more randomised trees, for diversity in the blend.
cd /d D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
set SM=research\techarena\softmax_step7.py --seq stage3_h50_s30 --binary data\validation\safe_v2_step7_s30_val_c4.parquet --max-trees 4000 --threads 10
echo [%date% %time%] batch 11 (CPU) start >> logs\lgb_exp.log
echo [%date% %time%] partner extra_trees started >> logs\lgb_exp.log
%PY% %SM% --params research\techarena\outputs\opt4_et.json --tag _s30_opt4_et > logs\softmax_s30_opt4_et.log 2>&1
findstr /C:"softmax7_alone" /C:"chose" /C:"Error" logs\softmax_s30_opt4_et.log >> logs\lgb_exp.log
echo [%date% %time%] partner colsample 0.2 started >> logs\lgb_exp.log
%PY% %SM% --params research\techarena\outputs\opt4_cs02.json --tag _s30_opt4_cs02 > logs\softmax_s30_opt4_cs02.log 2>&1
findstr /C:"softmax7_alone" /C:"chose" /C:"Error" logs\softmax_s30_opt4_cs02.log >> logs\lgb_exp.log
echo [%date% %time%] batch 11 (CPU) done >> logs\lgb_exp.log
exit /b 0
