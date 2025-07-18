@echo off
REM M.I.A Installation Script for Windows v0.1.0
REM This script installs M.I.A and its dependencies via pip

setlocal enablecho To run M.I.A, use one of these commands:
echo   mia --info         (show version info)
echo   mia --text-only    (text mode)
echo   mia --audio-mode   (with audio)
echo.
echo Or use the run scripts:
echo   scripts\run\run.bat
echo.
echo For help and documentation, check README.md
echo.
echo Note: Some features require Ollama to be installed.
echo Visit https://ollama.ai for installation instructions.
echo.
echo Press any key to exit...
pause >nul

echo.
echo =================================
echo   M.I.A Installation Script
echo =================================
echo Installing M.I.A v0.1.0 - Multimodal Intelligent Assistant
echo.

REM Check if Python is installed
echo [STEP] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher from https://python.org
    echo [ERROR] Make sure to add Python to your PATH during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo [INFO] Python %PYTHON_VERSION% found

REM Check Python version (basic check)
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8 or higher required. Found %PYTHON_VERSION%
    pause
    exit /b 1
)

REM Check if pip is installed
echo [STEP] Checking pip installation...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found. Please install pip
    pause
    exit /b 1
)
echo [INFO] pip found

REM Create virtual environment
echo [STEP] Creating virtual environment...
if exist "venv" (
    echo [WARNING] Virtual environment already exists
    set /p "RECREATE=Do you want to recreate it? (y/n): "
    if /i "!RECREATE!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q venv
    ) else (
        echo [INFO] Using existing virtual environment
        goto :activate_venv
    )
)

python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [INFO] Virtual environment created

:activate_venv
REM Activate virtual environment
echo [STEP] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment activation script not found
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo [INFO] Virtual environment activated

REM Upgrade pip
echo [STEP] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip, continuing...
)

REM Install dependencies
echo [STEP] Installing dependencies...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [INFO] Dependencies installed

REM Install M.I.A in development mode
echo [STEP] Installing M.I.A in development mode...
if exist "setup.py" (
    python -m pip install -e .
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to install M.I.A in development mode
    ) else (
        echo [INFO] M.I.A installed in development mode
    )
) else (
    echo [WARNING] setup.py not found, skipping development install
)

REM Test installation
echo [STEP] Testing installation...
python -c "import sys; print('Python', sys.version)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python test failed
    pause
    exit /b 1
)
echo [INFO] Python test passed

REM Test M.I.A import
python -c "from src.mia import main; print('M.I.A import successful')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] M.I.A import test failed (this might be normal during first setup)
) else (
    echo [INFO] M.I.A import test passed
)

echo.
echo =================================
echo   Installation Complete!
echo =================================
echo.
echo To start M.I.A, run:
echo   start-mia.bat
echo.
echo Or manually:
echo   venv\Scripts\activate.bat
echo   python main.py
echo.
echo For help and documentation, check the docs\ directory.
echo.
echo Press any key to exit...
pause >nul
echo Or use the provided run.bat script
echo.
echo Note: PyAudio might require additional system dependencies
echo If you encounter audio issues, install PyAudio manually:
echo pip install PyAudio
echo.
pause
