#!/usr/bin/env python3
"""
M.I.A - Multimodal Intelligent Assistant
Main entry point for the application
"""

import sys
import os
import importlib
from pathlib import Path

# Get the directory containing this script
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'

# Add src directory to Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and run main
if __name__ == "__main__":
    try:
        mia_main = importlib.import_module("mia.main")
        mia_main.main()
    except ImportError as e:
        print(f"Error importing main module: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Source directory: {src_dir}")
        print(f"Python path: {sys.path}")
        sys.exit(1)
