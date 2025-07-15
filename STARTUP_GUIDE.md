# M.I.A Startup Scripts Guide

This document explains how to use the various startup scripts for M.I.A (Multimodal Intelligent Assistant).

## Available Scripts

### Windows Scripts

#### `start.bat` - Simple Quick Start
- **Purpose**: Simple, straightforward way to start M.I.A
- **Features**: 
  - Activates virtual environment
  - Handles basic configuration
  - Starts M.I.A with default settings
  - Proper error handling and user feedback
- **Usage**: Double-click or run `start.bat` from command line

#### `start-mia.bat` - Detailed Start
- **Purpose**: More verbose startup with detailed logging
- **Features**:
  - Detailed status messages
  - Better error handling
  - Activates virtual environment
  - Comprehensive checks
  - Proper directory handling
- **Usage**: Double-click or run `start-mia.bat` from command line

#### `start-menu.bat` - Interactive Menu
- **Purpose**: Provides a menu to choose M.I.A startup mode
- **Features**:
  - Interactive mode selection
  - Text-only mode
  - Audio mode
  - Mixed mode
  - Virtual environment activation
  - Proper error handling
- **Usage**: Double-click and follow the menu prompts

#### `start-mia.ps1` - PowerShell Version
- **Purpose**: PowerShell version with enhanced features
- **Features**:
  - Colored output
  - Better error handling
  - PowerShell-specific optimizations
  - Enhanced user feedback
- **Usage**: Run from PowerShell: `.\start-mia.ps1`

#### `diagnose.bat` - Diagnostic Tool
- **Purpose**: Troubleshooting and system verification
- **Features**:
  - Checks Python installation
  - Verifies virtual environment
  - Tests imports and dependencies
  - Validates configuration
  - Provides detailed feedback
- **Usage**: Run when experiencing issues

### Linux/macOS Scripts

#### `start-mia.sh` - Unix Start Script
- **Purpose**: Cross-platform startup script for Unix-like systems
- **Features**:
  - Colored output
  - Comprehensive error handling
  - Virtual environment activation
  - Configuration checks
  - Proper exit codes
- **Usage**: `./start-mia.sh`

## Quick Start

1. **First time setup**: Run `diagnose.bat` to check your system
2. **Daily use**: Use `start.bat` for quick startup
3. **Troubleshooting**: Use `diagnose.bat` to identify issues

## ✅ Recent Fixes Applied

The following issues have been resolved:

- **Import errors**: Fixed relative import issues in the codebase
- **Virtual environment activation**: All scripts now properly activate the virtual environment
- **Error handling**: Improved error messages and user feedback
- **Path handling**: Fixed directory navigation issues
- **Missing exceptions**: Added missing exception classes
- **Configuration**: Better handling of config files

## Prerequisites

Before using any startup script, ensure:

1. **Python 3.8+** is installed and in PATH
2. **Virtual environment** is created by running:
   ```bash
   # Windows
   scripts\install\install.bat
   
   # Linux/macOS
   scripts/install/install.sh
   ```
3. **Dependencies** are installed (handled by install script)
4. **Configuration** is set up (copy `config\.env.example` to `config\.env`)

## Troubleshooting

### Common Issues

1. **"Virtual environment not found"**
   - Run the installation script first
   - Check that `venv` folder exists in project root

2. **"Python not found"**
   - Install Python 3.8+ from python.org
   - Add Python to your system PATH

3. **"main.py not found"**
   - Ensure you're running the script from the project root directory
   - Check that `main.py` exists in the project root

4. **Import errors**
   - Run `diagnose.bat` to check imports
   - Reinstall dependencies if needed

### Diagnostic Steps

1. Run `diagnose.bat` to check system status
2. Check Python version: `python --version`
3. Verify virtual environment exists: `venv\Scripts\activate.bat`
4. Test basic imports: `python -c "import src.mia.main"`

## Configuration

### Environment Variables

Copy `config\.env.example` to `config\.env` and configure:

```env
# API Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Other configurations...
```

### Startup Options

When using `start-menu.bat`, you can choose from:

1. **Interactive Mode**: Full M.I.A experience with mode selection
2. **Text-Only Mode**: Faster startup, text-based interaction only
3. **Audio Mode**: Voice-enabled interaction
4. **Mixed Mode**: Combination of text and audio

## Command Line Arguments

All scripts support passing arguments to M.I.A:

```bash
# Windows
start-mia.bat --text-only --debug

# Linux/macOS
./start-mia.sh --audio-mode --verbose
```

## Script Selection Guide

| Use Case | Recommended Script |
|----------|-------------------|
| First-time setup | `diagnose.bat` |
| Quick daily use | `start.bat` |
| Troubleshooting | `diagnose.bat` |
| Mode selection | `start-menu.bat` |
| Detailed logging | `start-mia.bat` |
| PowerShell users | `start-mia.ps1` |
| Linux/macOS | `start-mia.sh` |

## Getting Help

If you continue to experience issues:

1. Run `diagnose.bat` and share the output
2. Check the project README.md for additional setup instructions
3. Verify all dependencies are installed correctly
4. Ensure your Python environment is properly configured

## Advanced Usage

### Custom Virtual Environment

If you have a custom virtual environment location:

1. Edit the script to point to your venv path
2. Ensure the activation script exists at the specified path

### Development Mode

For development, you might want to:

1. Use `start-mia.bat` for detailed logging
2. Add `--debug` flag to command line arguments
3. Check `diagnose.bat` output for import verification
