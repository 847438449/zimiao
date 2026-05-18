@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Local virtual environment not found.
  echo Please run setup_local.bat first.
  pause
  exit /b 1
)

if not exist "launcher_ui.py" (
  echo [ERROR] launcher_ui.py not found in %cd%.
  pause
  exit /b 1
)

echo [INFO] Starting YOLOv8 Dashboard Launcher...
echo [INFO] URL: http://127.0.0.1:7860/
echo [INFO] Keep this window open while using the dashboard.

start "" "http://127.0.0.1:7860/"
".venv\Scripts\python.exe" launcher_ui.py

if errorlevel 1 (
  echo.
  echo [ERROR] Dashboard exited with an error.
  pause
  exit /b %errorlevel%
)

endlocal
