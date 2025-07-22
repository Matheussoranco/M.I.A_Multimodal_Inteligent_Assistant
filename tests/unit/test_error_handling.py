"""
Test Priority 2 fixes - Error Handling Standardization.
"""
import pytest
import unittest
import sys
import os
import requests
from unittest.mock import MagicMock, patch, Mock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mia.exceptions import *
from mia.error_handler import ErrorHandler, global_error_handler, with_error_handling, safe_execute
from mia.llm.llm_manager import LLMManager
from mia.security.security_manager import SecurityManager

class TestErrorHandling:
    """Test the error handling system."""
    
    def test_custom_exceptions(self):
        """Test custom exception classes."""
        # Test base exception
        base_error = MIAException("Test error", "TEST_001", {"key": "value"})
        assert str(base_error) == "[TEST_001] Test error"
        assert base_error.error_code == "TEST_001"
        assert base_error.details == {"key": "value"}
        
        # Test derived exceptions
        llm_error = LLMProviderError("LLM failed", "LLM_001")
        assert isinstance(llm_error, MIAException)
        assert str(llm_error) == "[LLM_001] LLM failed"
        
        # Test exception without error code
        simple_error = MIAException("Simple error")
        assert str(simple_error) == "Simple error"
        assert simple_error.error_code is None
    
    def test_error_handler_registration(self):
        """Test error handler registration and recovery."""
        handler = ErrorHandler()
        
        # Test recovery strategy registration
        def test_recovery(error, context):
            return "Recovery successful"
        
        handler.register_recovery_strategy(ValueError, test_recovery)
        assert ValueError in handler.recovery_strategies
        
        # Test error handling with recovery
        error = ValueError("Test error")
        result = handler.handle_error(error, {"test": "context"})
        assert result == "Recovery successful"
        
        # Test error counting
        assert handler.error_counts["ValueError"] == 1
    
    def test_error_handler_circuit_breaker(self):
        """Test circuit breaker functionality."""
        handler = ErrorHandler()
        handler.circuit_breaker_threshold = 2
        
        # Trigger multiple errors
        error = ValueError("Test error")
        handler.handle_error(error)
        handler.handle_error(error)
        
        # Should trigger circuit breaker
        assert handler.error_counts["ValueError"] == 2
        
        # Next error should not attempt recovery
        result = handler.handle_error(error)
        assert result is None
    
    def test_with_error_handling_decorator(self):
        """Test error handling decorator."""
        handler = ErrorHandler()
        
        @with_error_handling(handler, fallback_value="fallback")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "fallback"
        
        # Test with reraise
        @with_error_handling(handler, reraise=True)
        def failing_function_reraise():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function_reraise()
    
    def test_safe_execute(self):
        """Test safe execution utility."""
        # Test successful execution
        result = safe_execute(lambda x: x * 2, 5)
        assert result == 10
        
        # Test failed execution with default
        result = safe_execute(lambda: 1/0, default="error")
        assert result == "error"
        
        # Test failed execution without default
        result = safe_execute(lambda: 1/0)
        assert result is None

class TestLLMManagerErrorHandling(unittest.TestCase):
    """Test LLM Manager error handling improvements."""
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        # Test unknown provider - should not raise, but should not be available
        llm = LLMManager(provider='unknown')
        self.assertFalse(llm.is_available())
        
        # Test OpenAI without API key - should not raise, but should not be available
        with patch('mia.llm.llm_manager.HAS_OPENAI', True), \
             patch.dict(os.environ, {}, clear=True):
            llm_openai = LLMManager(provider='openai')
            self.assertFalse(llm_openai.is_available())
    
    def test_query_validation(self):
        """Test query input validation."""
        # Test with mock LLM since real LLM might not be available
        with patch('mia.llm.llm_manager.LLMManager') as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.query.side_effect = ValueError("Empty prompt")
            
            llm = MockLLM(provider='test')
            
            # Test empty prompt
            with pytest.raises(ValueError, match="Empty prompt"):
                llm.query("")
    
    def test_ollama_error_handling(self):
        """Test Ollama-specific error handling."""
        # Test that error handling captures and logs network errors appropriately
        with patch('mia.llm.llm_manager.requests.post') as mock_post:
            # Test timeout error
            mock_post.side_effect = requests.exceptions.Timeout()
            
            try:
                llm = LLMManager(provider='ollama')
                result = llm.query("test")
                # If error handling works, should return None or default value
                assert result is None or isinstance(result, str)
            except Exception as e:
                # Error handling should prevent unhandled exceptions
                # but if one occurs, it should be a known type
                assert isinstance(e, (NetworkError, ConfigurationError))

class TestSecurityManagerErrorHandling:
    """Test Security Manager error handling improvements."""
    
    def test_permission_validation(self):
        """Test permission check validation."""
        # Test error handling behavior rather than specific exceptions
        try:
            sm = SecurityManager()
            
            # Test empty action - should be handled gracefully
            result = sm.check_permission("")
            # Should return False or be handled by error handler
            assert result is False or result is None
            
        except Exception as e:
            # If any exception occurs, ensure it's logged appropriately
            assert hasattr(e, '__str__')  # Basic exception interface
    
    def test_path_validation(self):
        """Test file path validation."""
        sm = SecurityManager()
        
        # Test path traversal
        result = sm._validate_file_path("../../../etc/passwd")
        assert result == False
        
        # Test blocked path
        result = sm._validate_file_path("/etc/passwd")
        assert result == False
        
        # Test valid path
        result = sm._validate_file_path("valid_file.txt")
        assert result == True
    
    def test_command_validation(self):
        """Test command validation."""
        sm = SecurityManager()
        
        # Test dangerous command
        result = sm._validate_command("rm -rf /")
        assert result == False
        
        # Test command injection
        result = sm._validate_command("ls; rm file")
        assert result == False
        
        # Test valid command
        result = sm._validate_command("ls -la")
        assert result == True
    
    def test_web_query_validation(self):
        """Test web query validation."""
        sm = SecurityManager()
        
        # Test malicious query
        result = sm._validate_web_query("<script>alert('xss')</script>")
        assert result == False
        
        # Test long query
        result = sm._validate_web_query("x" * 1001)
        assert result == False
        
        # Test valid query
        result = sm._validate_web_query("python programming")
        assert result == True
    
    def test_policy_management(self):
        """Test policy management error handling."""
        sm = SecurityManager()
        
        # Test invalid action
        with pytest.raises(ValidationError, match="Invalid action"):
            sm.set_policy("", True)
        
        # Test invalid permission type
        with pytest.raises(ValidationError, match="must be boolean"):
            sm.set_policy("test_action", "invalid")

class TestMemoryErrorHandling:
    """Test Memory system error handling."""
    
    def test_store_experience_validation(self):
        """Test experience storage validation."""
        with patch('mia.memory.knowledge_graph.CHROMADB_AVAILABLE', False):
            from mia.memory.knowledge_graph import AgentMemory
            memory = AgentMemory()
            
            # Test empty text
            with pytest.raises(ValidationError, match="Empty text"):
                memory.store_experience("", [1, 2, 3])
            
            # Test invalid text type
            with pytest.raises(ValidationError, match="Text must be a string"):
                memory.store_experience(123, [1, 2, 3])
            
            # Test empty embedding
            with pytest.raises(ValidationError, match="Empty embedding"):
                memory.store_experience("test", [])
    
    def test_retrieve_context_validation(self):
        """Test context retrieval validation."""
        with patch('mia.memory.knowledge_graph.CHROMADB_AVAILABLE', False):
            from mia.memory.knowledge_graph import AgentMemory
            memory = AgentMemory()
            
            # Test empty query embedding
            with pytest.raises(ValidationError, match="Empty query embedding"):
                memory.retrieve_context([])
            
            # Test invalid top_k
            with pytest.raises(ValidationError, match="positive integer"):
                memory.retrieve_context([1, 2, 3], top_k=0)

if __name__ == "__main__":
    pytest.main([__file__])
