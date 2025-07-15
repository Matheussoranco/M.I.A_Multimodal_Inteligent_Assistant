import unittest
from mia.llm.llm_manager import LLMManager
from unittest.mock import patch, MagicMock

class TestLLMManager(unittest.TestCase):
    def test_init(self):
        with patch('mia.llm.llm_manager.ConfigManager') as MockConfig:
            manager = LLMManager()
            self.assertIsNotNone(manager)

if __name__ == "__main__":
    unittest.main()
