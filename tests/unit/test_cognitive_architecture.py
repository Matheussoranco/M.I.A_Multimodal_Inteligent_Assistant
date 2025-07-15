import unittest
from mia.core.cognitive_architecture import MIACognitiveCore
from unittest.mock import MagicMock

class TestMIACognitiveCore(unittest.TestCase):
    def test_init(self):
        mock_llm = MagicMock()
        core = MIACognitiveCore(mock_llm, device="cpu")
        self.assertIs(core.llm, mock_llm)
        self.assertEqual(core.device, "cpu")
        self.assertIsInstance(core.working_memory, list)

if __name__ == "__main__":
    unittest.main()
