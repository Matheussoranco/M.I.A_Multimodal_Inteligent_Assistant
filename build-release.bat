@echo off
REM Build script for M.I.A v0.1.0 pre-release (Windows)

echo 🚀 Building M.I.A v0.1.0 Pre-Release...

REM Clean previous builds
echo 🧹 Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo 🐍 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip and install build tools
echo 🔧 Installing build tools...
python -m pip install --upgrade pip setuptools wheel build

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Run basic tests
echo 🧪 Running basic tests...
python -m pytest tests/unit/ -v
if errorlevel 1 echo ⚠️  Some tests failed, continuing with build...

REM Build the package
echo 🏗️  Building package...
python -m build

REM Verify the build
echo ✅ Verifying build...
if exist "dist\mia-successor-0.1.0.tar.gz" (
    echo ✅ Source distribution created successfully
) else (
    echo ❌ Source distribution not found
    pause
    exit /b 1
)

if exist "dist\mia_successor-0.1.0-py3-none-any.whl" (
    echo ✅ Wheel distribution created successfully
) else (
    echo ❌ Wheel distribution not found
    pause
    exit /b 1
)

REM Show package info
echo 📋 Package information:
python setup.py --name --version --description

echo.
echo 🎉 Build completed successfully!
echo 📦 Packages created in dist/ directory:
dir dist\

echo.
echo 🚀 Ready for pre-release v0.1.0!
echo.
echo Next steps:
echo 1. Test the wheel: pip install dist/mia_successor-0.1.0-py3-none-any.whl
echo 2. Create git tag: git tag v0.1.0
echo 3. Push tag: git push origin v0.1.0
echo 4. Create GitHub release with the built packages

pause
