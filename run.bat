@echo off
chcp 65001 >nul
echo ============================================
echo   图片文字替换工具 - 源码直接运行
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/2] 检查依赖...
python -c "import cv2, numpy, PIL, tkinter" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 首次运行，正在安装依赖...
    python -m pip install -r requirements.txt -q
)

echo [2/2] 启动工具...
python main.py
if %errorlevel% neq 0 (
    echo [错误] 启动失败，请检查依赖或查看上方报错
    pause
)
