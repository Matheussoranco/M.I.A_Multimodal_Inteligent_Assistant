import sys
import os
import unittest

# Ensure src is in the path for test discovery
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    unittest.main(module=None, argv=[sys.argv[0], 'discover', '-s', 'tests', '-p', 'test_*.py'])
