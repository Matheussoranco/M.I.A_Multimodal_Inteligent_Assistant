import unittest
from mia.config_manager import LLMConfig
from mia.exceptions import ValidationError

class TestLLMConfigEdgeCases(unittest.TestCase):
    def test_empty_provider(self):
        config = LLMConfig(provider='', max_tokens=512, temperature=0.5)
        with self.assertRaises(ValidationError):
            config.validate()

    def test_unsupported_provider(self):
        config = LLMConfig(provider='fooai', max_tokens=512, temperature=0.5)
        with self.assertRaises(ValidationError):
            config.validate()

    def test_negative_max_tokens(self):
        config = LLMConfig(provider='ollama', max_tokens=-10, temperature=0.5)
        with self.assertRaises(ValidationError):
            config.validate()

    def test_temperature_below_zero(self):
        config = LLMConfig(provider='ollama', max_tokens=512, temperature=-0.1)
        with self.assertRaises(ValidationError):
            config.validate()

if __name__ == "__main__":
    unittest.main()
