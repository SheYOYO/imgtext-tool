@echo off
chcp 65001 >nul
title 同步代码到 GitHub
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo   正在把当前代码同步到 GitHub...
echo ============================================================
echo.
"%PY%" "%~dp0sync_to_github.py" %*
echo.
echo ============================================================
pause
