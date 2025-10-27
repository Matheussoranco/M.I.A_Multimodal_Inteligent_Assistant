import unittest
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from mia.llm.llm_manager import LLMManager
from mia.exceptions import LLMProviderError, ConfigurationError, InitializationError


class TestLLMManager(unittest.TestCase):
    """Comprehensive unit tests for LLMManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Set testing environment variable to re-raise exceptions
        os.environ['TESTING'] = 'true'
        
        self.mock_config = MagicMock()
        self.mock_config.config = MagicMock()
        self.mock_config.config.llm = MagicMock()
        self.mock_config.config.llm.provider = 'openai'
        self.mock_config.config.llm.model_id = 'test-model'
        self.mock_config.config.llm.api_key = 'test-key'
        self.mock_config.config.llm.url = 'http://test-url'
        self.mock_config.config.llm.max_tokens = 1000
        self.mock_config.config.llm.temperature = 0.7
        self.mock_config.config.llm.timeout = 30

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_init_with_config(self, mock_config_class):
        """Test initialization with configuration."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager()
        self.assertIsNotNone(manager)
        self.assertEqual(manager.provider, 'openai')
        self.assertEqual(manager.model_id, 'test-model')
        self.assertEqual(manager.api_key, 'test-key')

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_init_with_custom_params(self, mock_config_class):
        """Test initialization with custom parameters."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager(
            provider='openai',
            model_id='gpt-4',
            api_key='custom-key'
        )
        self.assertEqual(manager.provider, 'openai')
        self.assertEqual(manager.model_id, 'gpt-4')
        self.assertEqual(manager.api_key, 'custom-key')

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.HAS_OPENAI', True)
    @patch('mia.llm.llm_manager.OpenAI')
    def test_initialize_openai_success(self, mock_openai_class, mock_config_class):
        """Test successful OpenAI initialization."""
        mock_config_class.return_value = self.mock_config
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        manager = LLMManager(provider='openai', api_key='test-key')
        self.assertEqual(manager.client, mock_client)

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.HAS_OPENAI', False)
    def test_initialize_openai_no_package(self, mock_config_class):
        """Test OpenAI initialization when package not available."""
        mock_config_class.return_value = self.mock_config

        with self.assertRaises(InitializationError) as context:
            LLMManager(provider='openai')
        self.assertIn("OpenAI package not installed", str(context.exception))

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_initialize_openai_no_api_key(self, mock_config_class):
        """Test OpenAI initialization without API key."""
        mock_config_class.return_value = self.mock_config
        self.mock_config.config.llm.api_key = None

        with patch('mia.llm.llm_manager.HAS_OPENAI', True):
            with patch('os.getenv', return_value=None):
                with self.assertRaises(ConfigurationError) as context:
                    LLMManager(provider='openai')
                self.assertIn("API key not provided", str(context.exception))

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.HAS_TRANSFORMERS', True)
    @patch('mia.llm.llm_manager.pipeline')
    def test_initialize_huggingface_success(self, mock_pipeline, mock_config_class):
        """Test successful HuggingFace initialization."""
        mock_config_class.return_value = self.mock_config
        mock_client = MagicMock()
        mock_pipeline.return_value = mock_client

        manager = LLMManager(provider='huggingface', model_id='test-model')
        self.assertEqual(manager.client, mock_client)

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.HAS_TRANSFORMERS', False)
    def test_initialize_huggingface_no_package(self, mock_config_class):
        """Test HuggingFace initialization when package not available."""
        mock_config_class.return_value = self.mock_config

        with self.assertRaises(InitializationError) as context:
            LLMManager(provider='huggingface')
        self.assertIn("Transformers package not installed", str(context.exception))

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_initialize_huggingface_no_model_id(self, mock_config_class):
        """Test HuggingFace initialization without model ID."""
        mock_config_class.return_value = self.mock_config
        self.mock_config.config.llm.model_id = None

        with patch('mia.llm.llm_manager.HAS_TRANSFORMERS', True):
            with self.assertRaises(ConfigurationError) as context:
                LLMManager(provider='huggingface')
            self.assertIn("Model ID required", str(context.exception))

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_initialize_api_provider_ollama(self, mock_config_class):
        """Test API provider initialization for Ollama."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager(provider='ollama')
        self.assertEqual(manager.provider, 'ollama')

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_initialize_unknown_provider(self, mock_config_class):
        """Test initialization with unknown provider."""
        mock_config_class.return_value = self.mock_config

        with self.assertRaises(ConfigurationError) as context:
            LLMManager(provider='unknown')
        self.assertIn("Unknown provider", str(context.exception))

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.HAS_AIOHTTP', True)
    @patch('mia.llm.llm_manager.aiohttp')
    @patch('mia.llm.llm_manager.asyncio')
    def test_query_async_success(self, mock_asyncio, mock_aiohttp, mock_config_class):
        """Test successful async query."""
        mock_config_class.return_value = self.mock_config

        # Mock async context
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={'response': 'test response'})
        mock_session.post = MagicMock(return_value=mock_response)
        mock_aiohttp.ClientSession.return_value.__aenter__ = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientSession.return_value.__aexit__ = MagicMock(return_value=None)

        manager = LLMManager(provider='ollama')
        result = manager.query("test prompt")

        self.assertIsNotNone(result)

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_query_empty_prompt(self, mock_config_class):
        """Test query with empty prompt."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager()
        with self.assertRaises(ValueError):
            manager.query("")

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_query_whitespace_prompt(self, mock_config_class):
        """Test query with whitespace-only prompt."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager()
        with self.assertRaises(ValueError):
            manager.query("   ")

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.requests')
    @patch('mia.llm.llm_manager.HAS_AIOHTTP', False)
    def test_query_sync_fallback(self, mock_requests, mock_config_class):
        """Test sync query fallback."""
        mock_config_class.return_value = self.mock_config

        # Mock sync request
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'sync response'}
        mock_requests.post.return_value = mock_response

        manager = LLMManager(provider='ollama')
        result = manager.query("test prompt")

        self.assertIsNotNone(result)
        self.assertEqual(result, 'sync response')

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.requests')
    def test_query_timeout_error(self, mock_requests, mock_config_class):
        """Test query with timeout error."""
        mock_config_class.return_value = self.mock_config

        mock_requests.post.side_effect = TimeoutError("Request timed out")

        manager = LLMManager(provider='ollama')
        result = manager.query("test prompt")

        # Should return None on error due to error handler
        self.assertIsNone(result)

    @patch('mia.llm.llm_manager.ConfigManager')
    @patch('mia.llm.llm_manager.requests')
    def test_query_connection_error(self, mock_requests, mock_config_class):
        """Test query with connection error."""
        mock_config_class.return_value = self.mock_config

        mock_requests.post.side_effect = ConnectionError("Connection failed")

        manager = LLMManager(provider='ollama')
        result = manager.query("test prompt")

        # Should return None on error due to error handler
        self.assertIsNone(result)

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_query_model_alias(self, mock_config_class):
        """Test query_model alias method."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager()
        manager.query = MagicMock(return_value="test response")

        result = manager.query_model("test prompt")
        self.assertEqual(result, "test response")
        manager.query.assert_called_once_with("test prompt")

    @patch('mia.llm.llm_manager.ConfigManager')
    def test_provider_availability(self, mock_config_class):
        """Test provider availability checking."""
        mock_config_class.return_value = self.mock_config

        manager = LLMManager()
        # Initially should be available
        self.assertTrue(manager._available)

        # Test with failed initialization
        with patch.object(manager, '_initialize_provider', side_effect=Exception("Init failed")):
            manager._available = False
            self.assertFalse(manager._available)


if __name__ == "__main__":
    unittest.main()
