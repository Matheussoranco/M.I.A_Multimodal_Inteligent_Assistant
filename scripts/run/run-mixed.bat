@echo off
echo Starting M.I.A v0.1.0 - Mixed Mode...

REM Check if M.I.A is installed via pip
mia --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting M.I.A in mixed mode...
    mia %*
    goto :end
)

REM Fall back to development mode
if not exist "venv\Scripts\activate.bat" (
    echo M.I.A not found. Please install first:
    echo   pip install mia-successor
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo Starting M.I.A in mixed mode (development)...
python -m mia.main %*

:end
echo.
echo M.I.A session ended.
pause
