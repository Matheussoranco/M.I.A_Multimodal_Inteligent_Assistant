@echo off
:: Build M.I.A as a standalone Windows executable
:: This script creates a portable .exe file

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║     🧠  M.I.A - Build Script                                 ║
echo ║         Creating standalone Windows executable               ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Install PyInstaller if needed
echo [1/4] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: Install required dependencies
echo [2/4] Installing dependencies...
pip install fastapi uvicorn[standard] requests aiohttp python-dotenv pyyaml colorama openai numpy

:: Set path
cd /d "%~dp0"
set PYTHONPATH=%~dp0src

:: Build the executable
echo [3/4] Building executable...
echo This may take a few minutes...
echo.

pyinstaller mia.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ❌ Build failed! Trying alternative method...
    echo.
    
    :: Fallback: simpler build command
    pyinstaller --onefile --name MIA --add-data "src/mia;mia" --add-data "config;config" --hidden-import uvicorn --hidden-import fastapi --hidden-import starlette --hidden-import pydantic --hidden-import anyio mia_launcher.py
)

echo.
echo [4/4] Build complete!
echo.

if exist "dist\MIA.exe" (
    echo ✅ SUCCESS! Executable created at:
    echo    %~dp0dist\MIA.exe
    echo.
    echo To run M.I.A, double-click the MIA.exe file
    echo or run it from command line: dist\MIA.exe
) else (
    echo ⚠️  Executable not found in expected location.
    echo    Check the dist folder for the output.
)

echo.
pause
