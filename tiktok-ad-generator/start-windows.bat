@echo off
REM ============================================================
REM  TikTok Shop Ad Maker - one-click start for Windows
REM  Just double-click this file.
REM ============================================================
cd /d "%~dp0"
title TikTok Shop Ad Maker

echo.
echo  Setting up (first time takes a few minutes)...
echo.

REM Find Python
where python >nul 2>nul
if errorlevel 1 (
  echo  [!] Python is not installed.
  echo      Please install Python 3.10+ from https://www.python.org/downloads/
  echo      IMPORTANT: tick "Add Python to PATH" during install, then run this again.
  echo.
  pause
  exit /b
)

REM Create virtual environment on first run
if not exist ".venv\Scripts\python.exe" (
  echo  Creating environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul
echo  Installing components (only the first time)...
python -m pip install -r requirements.txt

echo.
echo  Starting the app... your browser will open automatically.
echo  (Keep this black window open while you use the app. Close it to stop.)
echo.
python app.py

pause
