# M.I.A Startup Script for Windows PowerShell
# This script activates the virtual environment and starts M.I.A

param(
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Info = "Blue"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Type = "Info"
    )
    
    $color = $Colors[$Type]
    Write-Host "[$Type] $Message" -ForegroundColor $color
}

function Write-Header {
    Write-Host ""
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host "     M.I.A Startup Script" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host ""
}

Write-Header

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to the M.I.A directory
Set-Location $ScriptDir

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\activate.bat")) {
    Write-ColoredOutput "Virtual environment not found!" "Error"
    Write-ColoredOutput "Please run the installation script first:" "Error"
    Write-ColoredOutput "  scripts\install\install.bat" "Error"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-ColoredOutput "Activating virtual environment..." "Info"

# Activate virtual environment
& "venv\Scripts\activate.bat"

# Check if main.py exists
if (-not (Test-Path "main.py")) {
    Write-ColoredOutput "main.py not found!" "Error"
    Write-ColoredOutput "Please ensure you are in the correct directory." "Error"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for config file
if (-not (Test-Path "config\.env")) {
    if (Test-Path ".env") {
        Write-ColoredOutput "Migrating .env to config\.env..." "Info"
        Move-Item ".env" "config\.env"
    } else {
        Write-ColoredOutput "config\.env file not found!" "Warning"
        Write-ColoredOutput "Using default configuration..." "Warning"
        Write-ColoredOutput "Please copy config\.env.example to config\.env and configure your API keys" "Warning"
    }
}

Write-ColoredOutput "Starting M.I.A..." "Info"
Write-Host ""
Write-Host "🚀 M.I.A - The successor of pseudoJarvis" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Start M.I.A with any command line arguments
try {
    if ($Arguments.Count -gt 0) {
        & python "main.py" $Arguments
    } else {
        & python "main.py"
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColoredOutput "M.I.A exited with error code $LASTEXITCODE" "Error"
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    } else {
        Write-ColoredOutput "M.I.A session ended successfully" "Success"
        Read-Host "Press Enter to exit"
    }
} catch {
    Write-ColoredOutput "Error starting M.I.A: $($_.Exception.Message)" "Error"
    Read-Host "Press Enter to exit"
    exit 1
}
