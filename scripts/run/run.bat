@echo off
echo Starting M.I.A - Multimodal Intelligent Assistant...

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run scripts\install\install.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist "config\.env" (
    echo Warning: config\.env file not found!
    echo Using default configuration...
    echo Please copy config\.env.example to config\.env and configure your API keys
)

echo.
echo M.I.A is starting...
python main.py %*

echo.
echo M.I.A session ended.
pause
