import unittest
from mia import main

class TestMainEntryPoint(unittest.TestCase):
    def test_main_exists(self):
        self.assertTrue(callable(main.main), "main.main should be callable")

if __name__ == "__main__":
    unittest.main()
