"""
Test configuration for M.I.A project
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add project root to path as well
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
