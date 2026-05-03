@echo off
title Win Go Predictor — Launcher
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║        WIN GO PREDICTION SYSTEM — LAUNCHER       ║
echo  ║        Server + API Poller (no scraper needed)    ║
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
echo  [1/3] Cleaning up old processes...
taskkill /F /FI "WINDOWTITLE eq WinGo-Server" >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start the FastAPI server (includes built-in API poller)
echo  [2/3] Starting server + API poller...
start "WinGo-Server" /min cmd /k "cd /d "%~dp0" && python run.py"

:: Wait for server to be ready
echo  [3/3] Waiting for server to start...
timeout /t 4 /nobreak >nul

:: Check if server is actually running
curl -s http://127.0.0.1:8000/ >nul 2>&1
if errorlevel 1 (
    echo  Server still starting, waiting a bit more...
    timeout /t 4 /nobreak >nul
)

:: Open the dashboard in default browser
start http://127.0.0.1:8000/

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║  Server running at: http://127.0.0.1:8000        ║
echo  ║  Dashboard opened in browser                      ║
echo  ║  API Poller: Auto-fetching results (no Chrome!)   ║
echo  ╚═══════════════════════════════════════════════════╝
echo.
echo  Everything is running. Press any key to exit launcher.
echo  (Server continues in background)
pause >nul
