@echo off
title Win Go Predictor — Launcher
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║        WIN GO PREDICTION SYSTEM — LAUNCHER       ║
echo  ║        One-click start: Server + Scraper          ║
echo  ╚═══════════════════════════════════════════════════╝
echo.

:: Navigate to the project directory
cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found! Please install Python and add to PATH.
    pause
    exit /b 1
)

:: Kill any existing instances
echo  [1/4] Cleaning up old processes...
taskkill /F /FI "WINDOWTITLE eq WinGo-Server" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq WinGo-Scraper" >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start the FastAPI server in a new window
echo  [2/4] Starting API server...
start "WinGo-Server" /min cmd /k "cd /d "%~dp0" && python run.py"

:: Wait for server to be ready
echo  [3/4] Waiting for server to start...
timeout /t 4 /nobreak >nul

:: Check if server is actually running
curl -s http://127.0.0.1:8000/ >nul 2>&1
if errorlevel 1 (
    echo  Server still starting, waiting a bit more...
    timeout /t 4 /nobreak >nul
)

:: Open the dashboard in default browser
echo  [4/4] Opening dashboard...
start http://127.0.0.1:8000/

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║  Server running at: http://127.0.0.1:8000        ║
echo  ║  Dashboard opened in browser                      ║
echo  ╚═══════════════════════════════════════════════════╝
echo.

:: Start the scraper (all timers, auto-login)
echo  Starting scraper with auto-login (all timers)...
echo  ─────────────────────────────────────────────────────
echo.

python scraper/run_scraper.py --timer all --wait 120

:: If scraper exits, keep window open
echo.
echo  Scraper stopped. Press any key to exit...
pause >nul
