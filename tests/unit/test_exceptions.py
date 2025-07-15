import unittest
from mia.exceptions import MIAException, LLMProviderError, AudioProcessingError, VisionProcessingError, SecurityError, ConfigurationError

class TestExceptions(unittest.TestCase):
    def test_base_exception(self):
        e = MIAException("msg", error_code="E1", details={"foo": 1})
        self.assertEqual(str(e), "[E1] msg")
        self.assertEqual(e.details["foo"], 1)

    def test_subclasses(self):
        self.assertTrue(issubclass(LLMProviderError, MIAException))
        self.assertTrue(issubclass(AudioProcessingError, MIAException))
        self.assertTrue(issubclass(VisionProcessingError, MIAException))
        self.assertTrue(issubclass(SecurityError, MIAException))
        self.assertTrue(issubclass(ConfigurationError, MIAException))

if __name__ == "__main__":
    unittest.main()
