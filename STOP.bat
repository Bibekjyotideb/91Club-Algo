@echo off
title Win Go Predictor — Shutdown
color 0C

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║        STOPPING WIN GO PREDICTION SYSTEM         ║
echo  ╚═══════════════════════════════════════════════════╝
echo.

echo  [1/2] Stopping server...
taskkill /F /FI "WINDOWTITLE eq WinGo-Server" >nul 2>&1

echo  [2/2] Stopping scraper...
taskkill /F /FI "WINDOWTITLE eq WinGo-Scraper" >nul 2>&1

:: Also kill any stray python processes running our scripts
taskkill /F /FI "WINDOWTITLE eq Win Go Predictor*" >nul 2>&1

echo.
echo  ✓ All processes stopped.
echo.
pause
