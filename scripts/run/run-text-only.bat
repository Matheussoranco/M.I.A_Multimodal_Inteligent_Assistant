@echo off
echo Starting M.I.A v0.1.0 - Text Mode Only...

REM Check if M.I.A is installed via pip
mia --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting M.I.A in text-only mode...
    mia --text-only %*
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
echo Starting M.I.A in text-only mode (development)...
python -m mia.main --text-only %*

:end
pause
    echo Please copy .env.example to .env and configure your API keys
)

echo.
echo M.I.A is starting in text-only mode...
python -m main_modules.main --text-only --skip-mode-selection %*

echo.
echo M.I.A session ended.
pause
