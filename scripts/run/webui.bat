@echo off
:: M.I.A Web UI Launcher for Windows
:: Starts the Ollama-style Web Interface

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     🧠  M.I.A - Multimodal Intelligent Assistant            ║
echo ║         Ollama-Style Web UI Launcher                         ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check for dependencies
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install fastapi uvicorn[standard]
)

:: Set the working directory to the script's location
cd /d "%~dp0"

:: Start the web UI
echo Starting M.I.A Web UI on http://localhost:8080
echo Press Ctrl+C to stop the server
echo.

python -m mia --web --port 8080

pause
