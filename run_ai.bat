@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Local virtual environment not found. Run setup_local.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py
pause
