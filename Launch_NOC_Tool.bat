@echo off
title Network Cell Health Monitor
chcp 65001 >nul

echo.
echo  =========================================
echo   Network Cell Health Monitor v1.0
echo   AI-powered 5G/LTE KPI Analysis
echo  =========================================
echo.
echo  Starting API server...

cd /d C:\Users\%USERNAME%\Desktop\rfai

start /min "NOC_API" cmd /k "C:\Users\SujitR\Desktop\rfai\rfai_env\Scripts\python.exe -m uvicorn network_health_api:app --host 0.0.0.0 --port 8000"

echo  Waiting for server to initialize...
timeout /t 5 /nobreak >nul

echo  Opening dashboard...
start "" "http://localhost:8000/dashboard"

echo.
echo  Network Cell Health Monitor is running
echo  Dashboard opened in your browser
echo.
echo  NOTE: Keep this window open while using the tool
echo  Press any key to stop the server and exit...
pause >nul

echo  Stopping server...
taskkill /f /fi "WINDOWTITLE eq NOC_API*" >nul 2>&1
taskkill /f /im uvicorn.exe >nul 2>&1
echo  Stopped.
timeout /t 2 /nobreak >nul