#!/usr/bin/env python3
"""
Project Reorganization Script for M.I.A
This script reorganizes the project structure for better maintainability
"""

import os
import shutil
import sys
from pathlib import Path

def create_new_structure():
    """Create the new organized directory structure"""
    base_dir = Path(".")
    
    # New directory structure
    new_dirs = [
        "src/mia",
        "src/mia/audio",
        "src/mia/core", 
        "src/mia/llm",
        "src/mia/memory",
        "src/mia/multimodal",
        "src/mia/plugins",
        "src/mia/security",
        "src/mia/tools",
        "src/mia/utils",
        "src/mia/learning",
        "src/mia/planning",
        "src/mia/deployment",
        "src/mia/langchain",
        "src/mia/system",
        "scripts/install",
        "scripts/run", 
        "scripts/development",
        "docs/user",
        "docs/developer",
        "config",
        "tests/unit",
        "tests/integration"
    ]
    
    # Create directories
    for dir_path in new_dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def move_files():
    """Move files to their new organized locations"""
    base_dir = Path(".")
    
    # File movements - (source, destination)
    movements = [
        # Main module becomes main entry point
        ("main_modules/main.py", "src/mia/main.py"),
        
        # Core modules
        ("audio/", "src/mia/audio/"),
        ("core/", "src/mia/core/"),
        ("llm/", "src/mia/llm/"),
        ("memory/", "src/mia/memory/"),
        ("multimodal/", "src/mia/multimodal/"),
        ("plugins/", "src/mia/plugins/"),
        ("security/", "src/mia/security/"),
        ("tools/", "src/mia/tools/"),
        ("utils/", "src/mia/utils/"),
        ("learning/", "src/mia/learning/"),
        ("planning/", "src/mia/planning/"),
        ("deployment/", "src/mia/deployment/"),
        ("langchain/", "src/mia/langchain/"),
        ("system/", "src/mia/system/"),
        
        # Scripts
        ("run.bat", "scripts/run/run.bat"),
        ("run-audio.bat", "scripts/run/run-audio.bat"),
        ("run-mixed.bat", "scripts/run/run-mixed.bat"),
        ("run-text-only.bat", "scripts/run/run-text-only.bat"),
        ("run.sh", "scripts/run/run.sh"),
        ("install.bat", "scripts/install/install.bat"),
        ("install.sh", "scripts/install/install.sh"),
        ("install_ffmpeg.bat", "scripts/install/install_ffmpeg.bat"),
        ("uninstall.sh", "scripts/install/uninstall.sh"),
        ("check-system.sh", "scripts/development/check-system.sh"),
        ("dev.sh", "scripts/development/dev.sh"),
        ("quickstart.sh", "scripts/development/quickstart.sh"),
        ("test_ollama.py", "scripts/development/test_ollama.py"),
        
        # Documentation
        ("README.md", "docs/README.md"),
        ("CHANGELOG.md", "docs/CHANGELOG.md"),
        ("USAGE.md", "docs/user/USAGE.md"),
        ("USAGE_GUIDE.md", "docs/user/USAGE_GUIDE.md"),
        
        # Configuration
        (".env.example", "config/.env.example"),
        
        # Requirements
        ("requirements.txt", "requirements.txt"),  # Keep at root
        ("requirements-dev.txt", "requirements-dev.txt"),  # Keep at root
        ("requirements-windows.txt", "requirements-windows.txt"),  # Keep at root
        
        # Tests
        ("tests/", "tests/"),  # Keep existing test structure
    ]
    
    for src, dest in movements:
        src_path = base_dir / src
        dest_path = base_dir / dest
        
        if src_path.exists():
            if src_path.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
                print(f"Moved directory: {src} -> {dest}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"Moved file: {src} -> {dest}")
        else:
            print(f"Warning: Source not found: {src}")

def create_new_files():
    """Create new configuration and entry point files"""
    base_dir = Path(".")
    
    # Create new main entry point
    main_entry = '''#!/usr/bin/env python3
"""
M.I.A - Multimodal Intelligent Assistant
Main entry point for the application
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main
from mia.main import main

if __name__ == "__main__":
    main()
'''
    
    (base_dir / "main.py").write_text(main_entry)
    print("Created new main entry point: main.py")
    
    # Create __init__.py files
    init_dirs = [
        "src/mia",
        "src/mia/audio",
        "src/mia/core",
        "src/mia/llm",
        "src/mia/memory",
        "src/mia/multimodal",
        "src/mia/plugins",
        "src/mia/security",
        "src/mia/tools",
        "src/mia/utils",
        "src/mia/learning",
        "src/mia/planning",
        "src/mia/deployment",
        "src/mia/langchain",
        "src/mia/system"
    ]
    
    for dir_path in init_dirs:
        init_file = base_dir / dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""M.I.A module"""')
            print(f"Created __init__.py in {dir_path}")
    
    # Create new project structure documentation
    structure_doc = '''# M.I.A Project Structure

## Directory Organization

```
M.I.A-The-successor-of-pseudoJarvis/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Development dependencies
├── requirements-windows.txt # Windows-specific dependencies
├── setup.py               # Package configuration
├── LICENSE                # License file
├── .gitignore            # Git ignore rules
├── .env.example          # Environment variables template
│
├── src/mia/              # Main source code
│   ├── __init__.py
│   ├── main.py           # Core application logic
│   ├── audio/            # Audio processing modules
│   ├── core/             # Core cognitive architecture
│   ├── llm/              # LLM integration
│   ├── memory/           # Memory management
│   ├── multimodal/       # Multimodal processing
│   ├── plugins/          # Plugin system
│   ├── security/         # Security management
│   ├── tools/            # Tool execution
│   ├── utils/            # Utility functions
│   ├── learning/         # User learning
│   ├── planning/         # Planning and scheduling
│   ├── deployment/       # Deployment management
│   ├── langchain/        # LangChain integration
│   └── system/           # System control
│
├── scripts/              # Utility scripts
│   ├── install/          # Installation scripts
│   ├── run/              # Run scripts
│   └── development/      # Development tools
│
├── docs/                 # Documentation
│   ├── user/             # User documentation
│   └── developer/        # Developer documentation
│
├── config/               # Configuration files
│   └── .env.example      # Environment template
│
├── tests/                # Test files
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
│
└── venv/                 # Virtual environment (local)
```

## Key Changes

1. **Centralized source code**: All Python modules under `src/mia/`
2. **Organized scripts**: Installation, run, and development scripts separated
3. **Better documentation**: User and developer docs separated
4. **Clear entry point**: Single `main.py` at root level
5. **Proper package structure**: All modules have `__init__.py` files

## Running the Application

- **Main entry**: `python main.py`
- **Quick start**: `python scripts/run/run.py`
- **Development**: `python scripts/development/dev.py`
'''
    
    (base_dir / "docs" / "PROJECT_STRUCTURE.md").write_text(structure_doc)
    print("Created project structure documentation")

def update_imports():
    """Update import statements in moved files"""
    print("Import updates would need to be done manually for each moved file")
    print("Key changes needed:")
    print("- Update relative imports to use new src/mia structure")
    print("- Update script paths in batch files")
    print("- Update any hardcoded paths")

def cleanup_old_files():
    """Remove old files and directories after successful move"""
    base_dir = Path(".")
    
    # Directories to remove after successful move
    old_dirs = [
        "main_modules",
        "audio",
        "core", 
        "llm",
        "memory",
        "multimodal",
        "plugins",
        "security",
        "tools",
        "utils",
        "learning",
        "planning",
        "deployment",
        "langchain",
        "system"
    ]
    
    # Files to remove
    old_files = [
        "run.bat",
        "run-audio.bat", 
        "run-mixed.bat",
        "run-text-only.bat",
        "run.sh",
        "install.bat",
        "install.sh",
        "install_ffmpeg.bat",
        "uninstall.sh",
        "check-system.sh",
        "dev.sh",
        "quickstart.sh",
        "test_ollama.py",
        "USAGE.md",
        "USAGE_GUIDE.md"
    ]
    
    print("\\nWARNING: This will remove old files and directories!")
    print("Make sure the reorganization worked correctly first.")
    print("\\nOld directories to remove:")
    for dir_name in old_dirs:
        if (base_dir / dir_name).exists():
            print(f"  - {dir_name}/")
    
    print("\\nOld files to remove:")
    for file_name in old_files:
        if (base_dir / file_name).exists():
            print(f"  - {file_name}")

def main():
    """Main reorganization function"""
    print("M.I.A Project Reorganization")
    print("=" * 40)
    
    print("\\n1. Creating new directory structure...")
    create_new_structure()
    
    print("\\n2. Moving files to new locations...")
    move_files()
    
    print("\\n3. Creating new configuration files...")
    create_new_files()
    
    print("\\n4. Import updates needed...")
    update_imports()
    
    print("\\n5. Cleanup information...")
    cleanup_old_files()
    
    print("\\n" + "=" * 40)
    print("Reorganization complete!")
    print("\\nNext steps:")
    print("1. Review the new structure")
    print("2. Update import statements in moved files")
    print("3. Test the application")
    print("4. Run cleanup script if everything works")

if __name__ == "__main__":
    main()
