@echo off
echo Installing ffmpeg for Windows...
echo.
echo Option 1: Using winget (Windows Package Manager)
winget install --id=Gyan.FFmpeg -e

echo.
echo Option 2: Using chocolatey (if installed)
echo choco install ffmpeg

echo.
echo Option 3: Manual installation
echo 1. Download ffmpeg from https://ffmpeg.org/download.html
echo 2. Extract to C:\ffmpeg
echo 3. Add C:\ffmpeg\bin to your PATH environment variable

echo.
echo After installation, restart your terminal and try running M.I.A again.
pause
