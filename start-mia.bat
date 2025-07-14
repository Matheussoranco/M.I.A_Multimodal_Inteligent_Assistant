@echo off
REM M.I.A Startup Script for Windows
REM This script activates the virtual environment and starts M.I.A

setlocal enabledelayedexpansion

echo.
echo =================================
echo      M.I.A Startup Script
echo =================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Change to the M.I.A directory
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run the installation script first:
    echo   scripts\install\install.bat
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if main.py exists
if not exist "main.py" (
    echo [ERROR] main.py not found!
    echo Please ensure you are in the correct directory.
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting M.I.A...
echo.
echo 🚀 M.I.A - The successor of pseudoJarvis
echo ==========================================
echo.

REM Start M.I.A with any command line arguments
python main.py %*

REM Keep window open if there's an error
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] M.I.A exited with an error
    pause
)
