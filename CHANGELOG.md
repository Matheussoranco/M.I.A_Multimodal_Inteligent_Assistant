# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-07-03 - Major Bug Fixes and Improvements

### 🐛 Fixed
- **Critical**: Fixed None checking errors in main.py that caused crashes
- **Critical**: Corrected syntax errors and indentation issues in llm_manager.py  
- **Critical**: Fixed import resolution issues with optional dependencies
- **Critical**: Resolved response validation and error handling problems
- **Critical**: Fixed setup.py to properly read requirements.txt

### 🔧 Improved
- **Dependencies**: Made requirements.txt more flexible with version ranges
- **Error Handling**: Added comprehensive exception handling throughout
- **Logging**: Implemented structured logging system
- **Configuration**: Enhanced environment variable management
- **Architecture**: Improved LLM provider initialization and availability checking

### ✨ Added
- **Installation**: Created Windows installation script (install.bat)
- **Runtime**: Added run.bat for easy execution on Windows
- **Configuration**: Created proper .env file with all needed variables
- **Documentation**: Updated README.md with better installation instructions
- **Documentation**: Enhanced USAGE.md with detailed examples and troubleshooting
- **Development**: Improved requirements-dev.txt for development tools

### 🔄 Changed
- **Dependencies**: Made PyTorch, OpenAI, and Transformers optional imports
- **Code Quality**: Added type hints and better documentation
- **Security**: Enhanced input validation and sanitization
- **Performance**: Reduced import overhead with optional loading

## [0.1.0] - 2025-06-29
### Added
- Initial public release of M.I.A - The Successor of PseudoJarvis.
- Modular architecture for LLM-powered virtual assistant.
- Speech recognition and synthesis modules.
- File management, web automation, notifications, and plugin system.
- Example tests and CI workflow.
