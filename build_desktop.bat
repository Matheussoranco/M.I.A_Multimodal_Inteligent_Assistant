@echo off
REM Build M.I.A Desktop Application (.exe)
REM This creates a standalone executable that runs without a browser

echo.
echo ===============================================================================
echo   Building M.I.A Desktop Application
echo ===============================================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check for PyInstaller
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing PyInstaller...
    pip install pyinstaller
)

REM Check for pywebview
python -c "import webview" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing pywebview...
    pip install pywebview
)

echo.
echo [*] Building executable...
echo.

REM Build the desktop app
pyinstaller --clean mia_desktop.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo   Build Complete!
echo ===============================================================================
echo.
echo   The executable is located at:
echo   dist\MIA-Desktop.exe
echo.
echo   You can run it directly - no browser needed!
echo.
pause
