@echo off
title Win Go Predictor — Shutdown
color 0C

echo.
echo  ╔═══════════════════════════════════════════════════╗
echo  ║        STOPPING WIN GO PREDICTION SYSTEM         ║
echo  ╚═══════════════════════════════════════════════════╝
echo.

echo  [1/3] Stopping server...
taskkill /F /FI "WINDOWTITLE eq WinGo-Server" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Win Go Predictor*" >nul 2>&1

echo  [2/3] Stopping scraper + Chrome...
:: Kill chromedriver (which also kills the Chrome it spawned)
taskkill /F /IM chromedriver.exe >nul 2>&1
:: Kill any leftover chrome from selenium (has --test-type flag)
wmic process where "name='chrome.exe' and commandline like '%%--test-type%%'" call terminate >nul 2>&1

echo  [3/3] Cleaning up Python processes...
:: Kill python processes running our specific scripts
wmic process where "name='python.exe' and commandline like '%%run.py%%'" call terminate >nul 2>&1
wmic process where "name='python.exe' and commandline like '%%run_scraper%%'" call terminate >nul 2>&1
wmic process where "name='python.exe' and commandline like '%%uvicorn%%'" call terminate >nul 2>&1

echo.
echo  ✓ All processes stopped.
echo.
timeout /t 3 /nobreak >nul
