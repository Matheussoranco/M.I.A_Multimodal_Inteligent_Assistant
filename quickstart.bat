@echo off
REM Quick Start Script for M.I.A v0.1.0
REM This script provides easy access to M.I.A functionality

echo.
echo ==========================================
echo     M.I.A v0.1.0 - Quick Start Menu
echo ==========================================
echo.

REM Check if M.I.A is installed via pip
mia --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ M.I.A is installed and ready!
    goto :menu
)

echo ⚠️  M.I.A not found via pip.
echo.
echo To install M.I.A, run:
echo   pip install mia-successor
echo.
echo Or build from source:
echo   .\build-release.bat
echo   pip install dist\mia_successor-0.1.0-py3-none-any.whl
echo.
pause
exit /b 1

:menu
echo Choose an option:
echo.
echo 1. Show M.I.A Info
echo 2. Start Text-Only Mode
echo 3. Start Audio Mode  
echo 4. Start Mixed Mode
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 📋 M.I.A Information:
    mia --info
    echo.
    pause
    goto :menu
)

if "%choice%"=="2" (
    echo.
    echo 🔤 Starting M.I.A in Text-Only Mode...
    mia --text-only
    goto :menu
)

if "%choice%"=="3" (
    echo.
    echo 🎤 Starting M.I.A in Audio Mode...
    mia --audio-mode
    goto :menu
)

if "%choice%"=="4" (
    echo.
    echo 🔄 Starting M.I.A in Mixed Mode...
    mia
    goto :menu
)

if "%choice%"=="5" (
    echo.
    echo 👋 Goodbye!
    exit /b 0
)

echo Invalid choice. Please try again.
goto :menu
