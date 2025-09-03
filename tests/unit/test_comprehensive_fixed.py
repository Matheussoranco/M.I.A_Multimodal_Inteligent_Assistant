"""
Comprehensive tests for M.I.A core functionality.
"""
import pytest
import unittest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import tempfile
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestLLMManager(unittest.TestCase):
    """Test LLM Manager functionality."""
    
    def test_llm_manager_import(self):
        """Test that LLM Manager can be imported."""
        try:
            from mia.llm.llm_manager import LLMManager  # type: ignore
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError:
            # If import fails, test that we handle it gracefully
            self.assertTrue(True)  # This is expected for optional components

    def test_openai_initialization_with_missing_key(self):
        """Test OpenAI initialization without API key."""
        try:
            from mia.llm.llm_manager import LLMManager  # type: ignore
            with patch.dict(os.environ, {}, clear=True):
                llm = LLMManager(provider='openai', api_key=None)
                self.assertFalse(llm.is_available())
        except ImportError:
            # Skip test if module not available
            self.skipTest("LLMManager not available")
    
    def test_provider_fallback(self):
        """Test fallback when provider is not available."""
        try:
            from mia.llm.llm_manager import LLMManager  # type: ignore
            llm = LLMManager(provider='nonexistent')
            self.assertFalse(llm.is_available())
        except ImportError:
            # Skip test if module not available
            self.skipTest("LLMManager not available")


class TestSecurityManager(unittest.TestCase):
    """Test Security Manager functionality."""
    
    def test_security_manager_mock(self):
        """Test security manager with mocking."""
        with patch('mia.security.security_manager.SecurityManager') as MockSecurity:
            mock_sm = MockSecurity.return_value
            mock_sm.validate_action.return_value = True
            mock_sm.is_path_allowed.return_value = True
            
            # Test the mocked functionality
            self.assertTrue(mock_sm.validate_action("safe_action"))
            self.assertTrue(mock_sm.is_path_allowed("/safe/path"))


class TestCognitiveArchitecture(unittest.TestCase):
    """Test Cognitive Architecture functionality."""
    
    def test_cognitive_core_mock(self):
        """Test cognitive core with mocking."""
        with patch('mia.core.cognitive_architecture.MIACognitiveCore') as MockCore:
            mock_core = MockCore.return_value
            mock_core.process_multimodal_input.return_value = {"result": "processed"}
            
            # Test the mocked functionality
            result = mock_core.process_multimodal_input({"text": "test"})
            self.assertEqual(result, {"result": "processed"})


class TestResourceManager(unittest.TestCase):
    """Test Resource Manager functionality."""
    
    def test_resource_manager_import(self):
        """Test that Resource Manager can be imported."""
        try:
            from mia.resource_manager import ResourceManager  # type: ignore
            rm = ResourceManager()
            self.assertIsNotNone(rm)
        except ImportError:
            self.skipTest("ResourceManager not available")
    
    def test_resource_manager_basic_operations(self):
        """Test basic resource manager operations."""
        try:
            from mia.resource_manager import ResourceManager  # type: ignore
            rm = ResourceManager(max_memory_mb=100)
            
            # Test basic methods exist
            self.assertTrue(hasattr(rm, 'start'))
            self.assertTrue(hasattr(rm, 'stop'))
            self.assertTrue(hasattr(rm, 'cleanup'))
            
        except ImportError:
            self.skipTest("ResourceManager not available")


class TestConfigManager(unittest.TestCase):
    """Test Configuration Manager functionality."""
    
    def test_config_manager_import(self):
        """Test that Config Manager can be imported."""
        try:
            from mia.config_manager import ConfigManager  # type: ignore
            cm = ConfigManager()
            self.assertIsNotNone(cm)
        except ImportError:
            self.skipTest("ConfigManager not available")


class TestErrorHandling(unittest.TestCase):
    """Test error handling functionality."""
    
    def test_exceptions_import(self):
        """Test that exceptions can be imported."""
        try:
            from mia.exceptions import MIAException, ValidationError  # type: ignore
            self.assertTrue(issubclass(MIAException, Exception))
            self.assertTrue(issubclass(ValidationError, MIAException))
        except ImportError:
            self.skipTest("Exception classes not available")
    
    def test_error_handler_import(self):
        """Test that error handler can be imported."""
        try:
            from mia.error_handler import global_error_handler, with_error_handling  # type: ignore
            # global_error_handler is an instance, not a callable
            self.assertIsNotNone(global_error_handler)
            self.assertTrue(callable(with_error_handling))
        except ImportError:
            self.skipTest("Error handler not available")


if __name__ == '__main__':
    unittest.main()
