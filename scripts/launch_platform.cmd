@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Users\HP\Desktop\task
.venv\Scripts\python.exe -m platform.main > logs\platform_server.log 2> logs\platform_server.err