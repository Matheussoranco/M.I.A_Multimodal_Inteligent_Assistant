@echo off
echo Starting M.I.A v0.1.0 - Multimodal Intelligent Assistant...

REM Check if M.I.A is installed via pip
mia --version >nul 2>&1
if %errorlevel% equ 0 (
    echo M.I.A is installed via pip. Starting...
    mia %*
    goto :end
)

REM Fall back to local development mode
if not exist "venv\Scripts\activate.bat" (
    echo M.I.A not found via pip and no virtual environment found!
    echo Please install M.I.A first:
    echo   pip install mia-successor
    echo Or run: scripts\install\install.bat
    pause
    exit /b 1
)

echo Running in development mode...
call venv\Scripts\activate.bat

echo.
echo M.I.A is starting...
python main.py %*

:end
echo.
echo M.I.A session ended.
pause
