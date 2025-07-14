@echo off
echo M.I.A Quick Start Options
echo ========================
echo.
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

pause
