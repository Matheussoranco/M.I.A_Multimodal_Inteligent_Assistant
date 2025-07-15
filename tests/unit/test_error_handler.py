import unittest
from mia.error_handler import ErrorHandler

class DummyException(Exception):
    pass

def dummy_strategy():
    return "recovered"

class TestErrorHandler(unittest.TestCase):
    def test_register_and_handle(self):
        handler = ErrorHandler()
        handler.register_recovery_strategy(DummyException, dummy_strategy)
        self.assertIn(DummyException, handler.recovery_strategies)

    def test_handle_error_logging(self):
        handler = ErrorHandler()
        try:
            raise DummyException("fail")
        except DummyException as e:
            result = handler.handle_error(e)
            self.assertIsNone(result)  # No recovery strategy by default

if __name__ == "__main__":
    unittest.main()
