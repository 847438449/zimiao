@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt streamlit
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" check_env.py
pause
