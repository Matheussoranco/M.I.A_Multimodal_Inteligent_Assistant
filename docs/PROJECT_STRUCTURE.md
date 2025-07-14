# M.I.A Project Structure

## 🏗️ New Organized Structure

```
M.I.A-The-successor-of-pseudoJarvis/
├── main.py                 # 🚀 Main entry point
├── start.bat               # 🎯 Quick start script (Windows)
├── requirements.txt        # 📦 Python dependencies
├── requirements-dev.txt    # 🔧 Development dependencies
├── requirements-windows.txt # 🪟 Windows-specific dependencies
├── setup.py               # 📋 Package configuration
├── LICENSE                # 📄 License file
├── .gitignore            # 🚫 Git ignore rules
├── Dockerfile            # 🐳 Docker configuration
├── docker-compose.yml    # 🐙 Docker Compose setup
├── Makefile              # 🔨 Build automation
│
├── src/mia/              # 🧠 Main source code
│   ├── __init__.py       # 📚 Package initialization
│   ├── main.py           # 🎮 Core application logic
│   ├── audio/            # 🎵 Audio processing modules
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   ├── speech_processor.py
│   │   └── speech_generator.py
│   ├── core/             # 🧬 Core cognitive architecture
│   │   ├── __init__.py
│   │   └── cognitive_architecture.py
│   ├── llm/              # 🤖 LLM integration
│   │   ├── __init__.py
│   │   └── llm_manager.py
│   ├── memory/           # 🧠 Memory management
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py
│   │   └── long_term_memory.py
│   ├── multimodal/       # 🎨 Multimodal processing
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   └── vision_processor.py
│   ├── plugins/          # 🔌 Plugin system
│   │   ├── __init__.py
│   │   └── plugin_manager.py
│   ├── security/         # 🛡️ Security management
│   │   ├── __init__.py
│   │   └── security_manager.py
│   ├── tools/            # 🔧 Tool execution
│   │   ├── __init__.py
│   │   └── action_executor.py
│   ├── utils/            # 🛠️ Utility functions
│   │   ├── __init__.py
│   │   └── automation_util.py
│   ├── learning/         # 📚 User learning
│   │   ├── __init__.py
│   │   └── user_learning.py
│   ├── planning/         # 📅 Planning and scheduling
│   │   ├── __init__.py
│   │   └── calendar_integration.py
│   ├── deployment/       # 🚀 Deployment management
│   │   ├── __init__.py
│   │   └── deployment_manager.py
│   ├── langchain/        # 🔗 LangChain integration
│   │   ├── __init__.py
│   │   └── langchain_verifier.py
│   └── system/           # 💻 System control
│       ├── __init__.py
│       └── system_control.py
│
├── scripts/              # 📜 Utility scripts
│   ├── install/          # 📦 Installation scripts
│   │   ├── install.bat
│   │   ├── install.sh
│   │   ├── install_ffmpeg.bat
│   │   └── uninstall.sh
│   ├── run/              # 🏃 Run scripts
│   │   ├── run.bat
│   │   ├── run-audio.bat
│   │   ├── run-mixed.bat
│   │   ├── run-text-only.bat
│   │   └── run.sh
│   └── development/      # 🔧 Development tools
│       ├── check-system.sh
│       ├── dev.sh
│       ├── quickstart.sh
│       └── test_ollama.py
│
├── docs/                 # 📖 Documentation
│   ├── README.md         # 📄 Main documentation
│   ├── CHANGELOG.md      # 📝 Change history
│   ├── PROJECT_STRUCTURE.md # 🏗️ This file
│   ├── user/             # 👤 User documentation
│   │   ├── USAGE.md
│   │   └── USAGE_GUIDE.md
│   └── developer/        # 👨‍💻 Developer documentation
│       └── API.md
│
├── config/               # ⚙️ Configuration files
│   └── .env.example      # 🔧 Environment template
│
├── tests/                # 🧪 Test files
│   ├── unit/             # 🔬 Unit tests
│   ├── integration/      # 🔗 Integration tests
│   └── __init__.py
│
└── venv/                 # 🐍 Virtual environment (local)
```

## 🎯 Key Improvements

### 1. **Clear Entry Points**
- **`main.py`**: Primary entry point at root level
- **`start.bat`**: Quick start script for users
- **`scripts/run/`**: Multiple run options organized

### 2. **Organized Source Code**
- **`src/mia/`**: All Python modules under single namespace
- **Proper package structure**: Each module has `__init__.py`
- **Relative imports**: Clean import statements

### 3. **Logical Script Organization**
- **`scripts/install/`**: Installation and setup scripts
- **`scripts/run/`**: Different run configurations
- **`scripts/development/`**: Development and testing tools

### 4. **Better Documentation**
- **`docs/user/`**: User-facing documentation
- **`docs/developer/`**: Technical documentation
- **`docs/`**: Main project documentation

### 5. **Configuration Management**
- **`config/`**: Centralized configuration files
- **Environment variables**: Organized in dedicated directory

## 🚀 Running the Application

### Quick Start
```bash
# Windows
start.bat

# Linux/Mac
python main.py
```

### Specific Modes
```bash
# Text-only mode
python main.py --text-only

# Audio mode
python main.py --audio-mode

# Mixed mode
python main.py

# With specific model
python main.py --model-id gemma3:4b-it-qat
```

### Using Scripts
```bash
# Windows
scripts\run\run.bat
scripts\run\run-text-only.bat
scripts\run\run-audio.bat

# Linux/Mac
scripts/run/run.sh
```

## 📦 Installation

### Windows
```bash
scripts\install\install.bat
```

### Linux/Mac
```bash
scripts/install/install.sh
```

## 🔧 Development

### Setup Development Environment
```bash
# Windows
scripts\development\dev.sh

# Linux/Mac
scripts/development/dev.sh
```

### Testing
```bash
# Test Ollama connection
python scripts/development/test_ollama.py

# Run tests
python -m pytest tests/
```

## 🗂️ Module Organization

### Core Modules
- **`main.py`**: Application entry point and main loop
- **`core/`**: Cognitive architecture and core AI logic
- **`llm/`**: Language model integration and management

### I/O Modules
- **`audio/`**: Audio input/output processing
- **`multimodal/`**: Vision and multimodal processing
- **`memory/`**: Knowledge graphs and long-term memory

### System Modules
- **`tools/`**: Action execution and system tools
- **`security/`**: Security and permission management
- **`plugins/`**: Plugin system and extensions

### Utility Modules
- **`utils/`**: General utility functions
- **`learning/`**: User learning and adaptation
- **`planning/`**: Calendar and scheduling integration
- **`deployment/`**: Deployment and configuration management

## 🎨 Benefits of New Structure

1. **🧹 Cleaner imports**: Relative imports reduce coupling
2. **📚 Better organization**: Related functionality grouped together
3. **🔧 Easier maintenance**: Clear separation of concerns
4. **🚀 Simpler deployment**: Single entry point and clear structure
5. **📖 Better documentation**: Organized docs for different audiences
6. **🧪 Improved testing**: Clear test organization
7. **🔌 Plugin support**: Well-organized plugin system
8. **⚙️ Configuration management**: Centralized config handling

## 🔄 Migration Guide

### For Users
1. Use `start.bat` instead of `run.bat`
2. Configuration moved to `config/` directory
3. Documentation moved to `docs/` directory

### For Developers
1. Import from `src.mia.module` instead of `module`
2. Scripts moved to `scripts/` subdirectories
3. Tests organized in `tests/` with proper structure

This new structure provides a much cleaner, more maintainable, and professional codebase! 🎉
