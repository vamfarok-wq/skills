@echo off
setlocal EnableDelayedExpansion

echo =================================================
echo   YouTube ^& TikTok Video Automation -- Setup
echo =================================================
echo.

:: ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Install Python 3.10+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo Python: %PY_VER%

:: ── [1/5] Virtual environment ─────────────────────────────────────────────────
echo.
echo [1/5] Setting up virtual environment ...
if not exist ".venv\" (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Created .venv
) else (
    echo       .venv already exists.
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo       Virtual environment active.

:: ── [2/5] Python dependencies ─────────────────────────────────────────────────
echo.
echo [2/5] Installing Python dependencies ...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Some packages failed to install. Check requirements.txt.
) else (
    echo       All dependencies installed.
)

:: ── [3/5] ffmpeg ──────────────────────────────────────────────────────────────
echo.
echo [3/5] Checking ffmpeg ...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    :: imageio-ffmpeg bundles its own ffmpeg, so videos will still render.
    :: A system ffmpeg gives better codec options.
    echo       System ffmpeg not found ^(optional^).
    echo       imageio-ffmpeg is bundled -- videos will render without it.
    echo.
    echo       For best results, install ffmpeg system-wide:
    echo         Option A: winget install Gyan.FFmpeg        ^(run in PowerShell as Admin^)
    echo         Option B: choco install ffmpeg              ^(if Chocolatey is installed^)
    echo         Option C: https://www.gyan.dev/ffmpeg/builds/
    echo                   Download, extract, add bin\ folder to PATH.
    echo.
) else (
    for /f "tokens=1-3" %%a in ('ffmpeg -version 2^>^&1 ^| findstr /i "ffmpeg version"') do (
        echo       %%a %%b %%c
    )
)

:: ── [4/5] Font ────────────────────────────────────────────────────────────────
echo.
echo [4/5] Checking font ...
if not exist "assets\fonts\" mkdir assets\fonts
if not exist "assets\fonts\Roboto-Bold.ttf" (
    echo       Downloading Roboto Bold font ...
    powershell -Command ^
        "$url='https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Bold.ttf';" ^
        "try { Invoke-WebRequest $url -OutFile 'assets\fonts\Roboto-Bold.ttf' -UseBasicParsing;" ^
        "Write-Host '      Font downloaded.' } catch { Write-Host '      Could not download font -- Windows system fonts will be used.' }"
) else (
    echo       Font already present.
)

:: ── [5/5] Configuration ───────────────────────────────────────────────────────
echo.
echo [5/5] Configuration ...
if not exist ".env" (
    copy .env.example .env >nul
    echo       Created .env from .env.example
    echo       *** IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY ***
) else (
    echo       .env already exists.
)

if not exist "output\"      mkdir output
if not exist "credentials\" mkdir credentials

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo =================================================
echo   Setup complete!
echo.
echo   REQUIRED: Open .env and add your ANTHROPIC_API_KEY
echo.
echo   Launch the GUI:
echo     Double-click  launch.bat
echo     -- or --
echo     .venv\Scripts\activate ^& python gui.py
echo.
echo   CLI commands:
echo     python main.py niches
echo     python main.py generate --niche tech
echo     python main.py run
echo =================================================
echo.
pause
