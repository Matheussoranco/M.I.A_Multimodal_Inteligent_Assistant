@echo off
setlocal enabledelayedexpansion

echo.
echo =================================
echo   M.I.A Quick Start Options
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

REM Check if main.py exists
if not exist "main.py" (
    echo [ERROR] main.py not found!
    echo Please ensure you are in the correct directory.
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo Choose how to start M.I.A:
echo.
echo 1. Interactive Mode Selection (Recommended)
echo 2. Text-Only Mode (Fast)
echo 3. Audio Mode
echo 4. Mixed Mode
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting M.I.A with interactive mode selection...
    python main.py
) else if "%choice%"=="2" (
    echo Starting M.I.A in text-only mode...
    python main.py --text-only --skip-mode-selection
) else if "%choice%"=="3" (
    echo Starting M.I.A in audio mode...
    python main.py --audio-mode --skip-mode-selection
) else if "%choice%"=="4" (
    echo Starting M.I.A in mixed mode...
    python main.py --skip-mode-selection
) else (
    echo Invalid choice. Starting with interactive mode selection...
    python main.py
)

REM Keep window open if there's an error
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] M.I.A exited with an error
    pause
)
