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
    echo Install Python 3.11 or 3.12 from:
    echo   https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo Python detected: %PY_VER%

:: ── Python version check ──────────────────────────────────────────────────────
:: Extract major.minor for comparison
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

echo.
if %PY_MAJOR% GEQ 3 if %PY_MINOR% GEQ 13 (
    echo [WARNING] You are using Python %PY_VER%
    echo.
    echo Python 3.13 and 3.14 are TOO NEW for some packages.
    echo moviepy, numpy, and Pillow may not have pre-built wheels
    echo for Python %PY_VER% yet, and compiling from source requires
    echo Microsoft Visual C++ Build Tools.
    echo.
    echo RECOMMENDED: Install Python 3.12 instead:
    echo   https://www.python.org/downloads/release/python-3128/
    echo   ^(scroll down, download "Windows installer 64-bit"^)
    echo.
    echo You can have multiple Python versions installed side-by-side.
    echo After installing 3.12, run:
    echo   py -3.12 -m venv .venv
    echo.
    choice /C YN /M "Continue anyway with Python %PY_VER%? (Y=yes, N=exit)"
    if errorlevel 2 (
        echo.
        echo Exiting. Please install Python 3.12 and re-run setup.bat
        pause
        exit /b 0
    )
    echo Continuing with Python %PY_VER% ...
    echo.
)

:: ── [1/5] Virtual environment ─────────────────────────────────────────────────
echo [1/5] Setting up virtual environment ...

if exist ".venv\" (
    echo       .venv already exists. Deleting and recreating to avoid cache issues...
    rmdir /s /q .venv
)

python -m venv .venv
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment.
    echo         Try: pip install virtualenv  then  virtualenv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo       Done.

:: ── [2/5] Python dependencies ─────────────────────────────────────────────────
echo.
echo [2/5] Installing Python dependencies ...
echo       Clearing pip cache to avoid deserialization warnings...
python -m pip cache purge >nul 2>&1

echo       Upgrading pip...
python -m pip install --upgrade pip --no-cache-dir --quiet

echo       Installing packages (this may take 2-5 minutes)...
pip install -r requirements.txt ^
    --no-cache-dir ^
    --prefer-binary ^
    --timeout 120

if errorlevel 1 (
    echo.
    echo [ERROR] Some packages failed to install.
    echo.
    echo Common fixes:
    echo   1. Use Python 3.12 instead of %PY_VER%
    echo      https://www.python.org/downloads/release/python-3128/
    echo.
    echo   2. Install Microsoft C++ Build Tools ^(for packages without wheels^):
    echo      https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo      Run the installer, select "Desktop development with C++"
    echo.
    echo   3. Try running this command manually to see the full error:
    echo      pip install -r requirements.txt --no-cache-dir --prefer-binary
    echo.
    pause
    exit /b 1
)
echo       All dependencies installed successfully.

:: ── [3/5] ffmpeg ──────────────────────────────────────────────────────────────
echo.
echo [3/5] Checking ffmpeg ...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo       System ffmpeg not found.
    echo       NOTE: imageio-ffmpeg ^(installed above^) bundles its own ffmpeg,
    echo       so videos WILL still render without a system install.
    echo.
    echo       For best quality, install ffmpeg system-wide ^(optional^):
    echo         winget install Gyan.FFmpeg    ^(run in PowerShell as Admin^)
    echo         or download: https://www.gyan.dev/ffmpeg/builds/
) else (
    for /f "tokens=1-3" %%a in ('ffmpeg -version 2^>^&1 ^| findstr /i "ffmpeg version"') do echo       %%a %%b %%c
)

:: ── [4/5] Font ────────────────────────────────────────────────────────────────
echo.
echo [4/5] Downloading Roboto Bold font ...
if not exist "assets\fonts\" mkdir assets\fonts

if not exist "assets\fonts\Roboto-Bold.ttf" (
    powershell -NoProfile -Command ^
        "try { Invoke-WebRequest 'https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Bold.ttf' -OutFile 'assets\fonts\Roboto-Bold.ttf' -UseBasicParsing; Write-Host '      Downloaded.' } catch { Write-Host '      Could not download — Windows system fonts (Arial) will be used instead.' }"
) else (
    echo       Already present.
)

:: ── [5/5] Configuration ───────────────────────────────────────────────────────
echo.
echo [5/5] Configuration ...
if not exist ".env" (
    copy .env.example .env >nul
    echo       Created .env
    echo.
    echo       =====================================================
    echo        IMPORTANT: Open .env and set ANTHROPIC_API_KEY
    echo        Without it the app cannot generate video scripts.
    echo       =====================================================
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
echo   Next steps:
echo   1. Open .env in Notepad and add ANTHROPIC_API_KEY
echo.
echo   2. Launch the GUI:
echo      Double-click  launch.bat
echo      -- or in this window --
echo      python gui.py
echo.
echo   3. Browser opens at http://localhost:5000
echo      Enter all API keys in the "API Settings" tab
echo =================================================
echo.
pause
