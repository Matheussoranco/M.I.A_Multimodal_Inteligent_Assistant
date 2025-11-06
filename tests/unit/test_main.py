import os
import sys
import unittest
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from mia.main import main  # type: ignore


class TestMainEntryPoint(unittest.TestCase):
    def test_main_exists(self):
        self.assertTrue(callable(main), "main should be callable")


if __name__ == "__main__":
    unittest.main()
