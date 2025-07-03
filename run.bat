@echo off
echo Starting M.I.A - Multimodal Intelligent Assistant...

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run install.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist ".env" (
    echo Warning: .env file not found!
    echo Using default configuration...
    echo Please copy .env.example to .env and configure your API keys
)

echo.
echo M.I.A is starting...
python -m main_modules.main %*

echo.
echo M.I.A session ended.
pause
