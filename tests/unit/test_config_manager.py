import unittest
from mia.config_manager import LLMConfig
from mia.exceptions import ValidationError

class TestLLMConfig(unittest.TestCase):
    def test_valid_config(self):
        config = LLMConfig(provider='ollama', max_tokens=512, temperature=0.5)
        # Should not raise
        config.validate()

    def test_invalid_provider(self):
        config = LLMConfig(provider='invalid', max_tokens=512, temperature=0.5)
        with self.assertRaises(ValidationError):
            config.validate()

    def test_invalid_max_tokens(self):
        config = LLMConfig(provider='ollama', max_tokens=0, temperature=0.5)
        with self.assertRaises(ValidationError):
            config.validate()

    def test_invalid_temperature(self):
        config = LLMConfig(provider='ollama', max_tokens=512, temperature=3.0)
        with self.assertRaises(ValidationError):
            config.validate()

if __name__ == "__main__":
    unittest.main()
