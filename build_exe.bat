@echo off
chcp 65001 >nul
echo ============================================
echo   图片文字替换工具 - 一键打包 exe
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/3] 安装依赖...
python -m pip install -r requirements.txt pyinstaller -q
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo [2/3] 开始打包...
pyinstaller --noconfirm --clean --windowed --name 图片文字替换工具 ^
    --add-data "fonts;fonts" ^
    --hidden-import app.config ^
    --hidden-import app.feature_extract ^
    --hidden-import app.erase ^
    --hidden-import app.render ^
    --hidden-import app.font_match ^
    --hidden-import app.io_utils ^
    --hidden-import app.gui ^
    main.py
if %errorlevel% neq 0 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo [3/3] 打包完成！
echo 可执行文件位于: dist\图片文字替换工具\图片文字替换工具.exe
echo.
pause
