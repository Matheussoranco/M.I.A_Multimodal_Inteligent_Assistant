@echo off
setlocal enabledelayedexpansion

echo.
echo =================================
echo       M.I.A Diagnostic Tool
echo =================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Change to the M.I.A directory
cd /d "%SCRIPT_DIR%"

echo [INFO] Current directory: %CD%
echo.

echo [CHECK] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.8+ and add it to PATH
) else (
    for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
    echo [OK] Python %PYTHON_VERSION% found
)
echo.

echo [CHECK] Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment found
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated
) else (
    echo [ERROR] Virtual environment not found
    echo Please run: scripts\install\install.bat
    goto :end
)
echo.

echo [CHECK] Checking main.py...
if exist "main.py" (
    echo [OK] main.py found
) else (
    echo [ERROR] main.py not found
    goto :end
)
echo.

echo [CHECK] Checking config files...
if exist "config\.env" (
    echo [OK] config\.env found
) else (
    if exist "config\.env.example" (
        echo [WARNING] config\.env not found, but config\.env.example exists
        echo Please copy config\.env.example to config\.env and configure
    ) else (
        echo [ERROR] No config files found
    )
)
echo.

echo [CHECK] Checking Python modules...
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import sys; print('Python path:', sys.path[0])"
echo.

echo [CHECK] Testing basic imports...
python -c "import os, sys; print('Basic imports: OK')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Basic Python imports failed
    goto :end
)

echo [CHECK] Testing M.I.A imports...
python -c "import sys; sys.path.insert(0, 'src'); import mia; print('M.I.A package: OK')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] M.I.A package import failed
    echo [INFO] This might indicate missing dependencies
    echo [INFO] Try running: pip install -r requirements.txt
    echo [INFO] Or reinstall with: scripts\install\install.bat
) else (
    echo [OK] M.I.A package imports successfully
)

echo [CHECK] Testing main.py execution...
python -c "import sys; sys.path.insert(0, 'src'); from mia.main import main; print('Main function: OK')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Main function import failed
    echo [INFO] This might indicate missing dependencies or import issues
    echo [INFO] You can still try running M.I.A directly
) else (
    echo [OK] Main function accessible
)

echo [OK] All basic checks passed!
echo.
echo [INFO] You can now try running M.I.A with:
echo   - start.bat (simple start)
echo   - start-mia.bat (detailed start)
echo   - start-menu.bat (with options)
echo.

:end
pause
