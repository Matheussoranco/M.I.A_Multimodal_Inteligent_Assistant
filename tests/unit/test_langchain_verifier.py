import unittest
from langchain.langchain_verifier import LangChainVerifier

class TestLangChainVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = LangChainVerifier()

    def test_exact_match(self):
        result = self.verifier.verify("hello", expected="hello")
        self.assertIn("Verified: True", result)

    def test_mismatch(self):
        result = self.verifier.verify("hello", expected="world")
        self.assertIn("Verified: False", result)

    def test_no_expected(self):
        result = self.verifier.verify("hello")
        self.assertIn("LangChain output", result)

if __name__ == "__main__":
    unittest.main()
