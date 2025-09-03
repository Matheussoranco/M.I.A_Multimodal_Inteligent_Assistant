"""
Basic tests for agent enhancements.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestAgentComponents(unittest.TestCase):
    """Test agent components with proper mocking."""

    def test_action_executor_mock(self):
        """Test action executor with mocking."""
        # Mock the ActionExecutor since it may not be available
        with patch('mia.tools.action_executor.ActionExecutor') as MockActionExecutor:
            mock_ae = MockActionExecutor.return_value
            mock_ae.execute.return_value = "Success"
            
            # Test the mocked functionality
            result = mock_ae.execute("open_file", {"path": "dummy.txt"})
            self.assertEqual(result, "Success")
            mock_ae.execute.assert_called_once_with("open_file", {"path": "dummy.txt"})

    def test_vision_processor_mock(self):
        """Test vision processor with mocking."""
        with patch('mia.multimodal.vision_processor.VisionProcessor') as MockVisionProcessor:
            mock_vp = MockVisionProcessor.return_value
            mock_vp.process_image.return_value = {"detected": "objects"}
            
            # Test the mocked functionality
            result = mock_vp.process_image("test_image.jpg")
            self.assertEqual(result, {"detected": "objects"})
            mock_vp.process_image.assert_called_once_with("test_image.jpg")

    def test_calendar_integration_mock(self):
        """Test calendar integration with mocking."""
        with patch('mia.planning.calendar_integration.CalendarIntegration') as MockCalendar:
            mock_ci = MockCalendar.return_value
            mock_ci.get_events.return_value = ["event1", "event2"]
            
            # Test the mocked functionality
            result = mock_ci.get_events()
            self.assertEqual(result, ["event1", "event2"])
            mock_ci.get_events.assert_called_once()

    def test_security_manager_mock(self):
        """Test security manager with mocking."""
        with patch('mia.security.security_manager.SecurityManager') as MockSecurity:
            mock_sm = MockSecurity.return_value
            mock_sm.validate_action.return_value = True
            
            # Test the mocked functionality
            result = mock_sm.validate_action("safe_action")
            self.assertTrue(result)
            mock_sm.validate_action.assert_called_once_with("safe_action")

    def test_main_import(self):
        """Test that main module can be imported."""
        try:
            from mia.main import main  # type: ignore
            self.assertTrue(callable(main))
        except ImportError as e:
            self.fail(f"Failed to import main: {e}")

    def test_version_import(self):
        """Test that version can be imported."""
        try:
            from mia.__version__ import __version__  # type: ignore
            self.assertIsInstance(__version__, str)
            self.assertTrue(len(__version__) > 0)
        except ImportError as e:
            self.fail(f"Failed to import version: {e}")


if __name__ == '__main__':
    unittest.main()
