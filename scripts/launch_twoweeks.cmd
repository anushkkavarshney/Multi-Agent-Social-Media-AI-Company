@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Users\HP\Desktop\task
.venv\Scripts\python.exe -u scripts\run_two_weeks.py --reject-once --short > logs\two_week_run.log 2> logs\two_week_run.err