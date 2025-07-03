@echo off
echo =====================================
echo M.I.A - Installation Script (Windows)
echo =====================================

echo.
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo Python found!

echo.
echo [2/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)
echo Virtual environment ready!

echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/6] Installing dependencies...
echo Installing core dependencies (this may take a while)...
python -m pip install -r requirements.txt

echo.
echo [6/6] Installing M.I.A package...
python -m pip install -e .

echo.
echo =====================================
echo Installation Complete!
echo =====================================
echo.
echo To run M.I.A:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Configure .env file with your API keys
echo 3. Run: python -m main_modules.main
echo.
echo Or use the provided run.bat script
echo.
echo Note: PyAudio might require additional system dependencies
echo If you encounter audio issues, install PyAudio manually:
echo pip install PyAudio
echo.
pause
