"""
Comprehensive tests for M.I.A core functionality.
"""
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import tempfile
import os

from mia.llm.llm_manager import LLMManager
from security.security_manager import SecurityManager
from core.cognitive_architecture import MIACognitiveCore


class TestLLMManager:
    """Test LLM Manager functionality."""
    
    def test_openai_initialization_with_missing_key(self):
        """Test OpenAI initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            llm = LLMManager(provider='openai', api_key=None)
            assert not llm.is_available()
    
    def test_provider_fallback(self):
        """Test fallback when provider is not available."""
        llm = LLMManager(provider='nonexistent')
        assert not llm.is_available()
    
    @mock.patch('llm.llm_manager.requests.post')
    def test_ollama_query(self, mock_post):
        """Test Ollama API query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'Test response'}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        llm = LLMManager(provider='ollama', api_key='test')
        result = llm.query("Test prompt")
        assert result == "Test response"


class TestSecurityManager:
    """Test Security Manager functionality."""
    
    def test_default_permissions(self):
        """Test default permission settings."""
        sm = SecurityManager()
        assert not sm.check_permission("read_file")
        assert sm.check_permission("web_search")
    
    def test_blocked_paths(self):
        """Test that sensitive paths are blocked."""
        sm = SecurityManager()
        sm.set_policy("read_file", True)
        
        # Should be blocked
        result = sm.check_permission("read_file", {"path": "/etc/passwd"})
        assert not result
        
        # Should be allowed
        result = sm.check_permission("read_file", {"path": "/tmp/safe_file.txt"})
        assert result
    
    def test_dangerous_commands(self):
        """Test that dangerous commands are blocked."""
        sm = SecurityManager()
        sm.set_policy("execute_command", True)
        
        # Should be blocked
        result = sm.check_permission("execute_command", {"command": "rm -rf /"})
        assert not result
        
        # Should be allowed
        result = sm.check_permission("execute_command", {"command": "ls -la"})
        assert result
    
    def test_audit_trail(self):
        """Test audit trail functionality."""
        sm = SecurityManager()
        sm.check_permission("read_file")
        sm.check_permission("web_search")
        
        trail = sm.get_audit_trail()
        assert len(trail) == 2
        assert trail[0]["action"] == "read_file"
        assert trail[1]["action"] == "web_search"


class TestCognitiveArchitecture:
    """Test Cognitive Architecture functionality."""
    
    def test_multimodal_processing(self):
        """Test multimodal input processing."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Processed response"
        
        with patch('core.cognitive_architecture.CLIPProcessor'), \
             patch('core.cognitive_architecture.CLIPModel'):
            core = MIACognitiveCore(mock_llm)
            
            inputs = {
                'text': 'Test input',
                'audio': 'test audio data'
            }
            
            result = core.process_multimodal_input(inputs)
            assert result == "Processed response"


class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_llm_security_integration(self):
        """Test LLM with security manager."""
        sm = SecurityManager()
        llm = LLMManager(provider='ollama', api_key='test')
        
        # Test that security is checked before LLM query
        action = "execute_command"
        context = {"command": "rm -rf /"}
        
        if sm.check_permission(action, context):
            # Should not reach here due to security block
            assert False, "Security manager should have blocked dangerous command"
        else:
            # This should happen
            assert True


if __name__ == "__main__":
    pytest.main([__file__])
