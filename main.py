#!/usr/bin/env python3
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
