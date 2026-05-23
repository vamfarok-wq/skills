@echo off
:: Launch the Video Automation GUI
:: Double-click this file to start the web interface.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

if not exist ".env" (
    echo [WARNING] .env file not found. Copying from .env.example ...
    copy .env.example .env >nul
    echo Open .env and add your ANTHROPIC_API_KEY before continuing.
    notepad .env
    pause
)

echo Starting Video Automation GUI ...
echo Browser will open at http://localhost:5000
echo.
echo Press Ctrl+C to stop the server.
echo.
python gui.py
pause
