"""
Integration tests for configuration and resource management
"""
import unittest
import os
import tempfile
import shutil
import json
import yaml
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
from src.mia.config_manager import ConfigManager, LLMConfig, AudioConfig, VisionConfig, SystemConfig
from src.mia.resource_manager import ResourceManager, ManagedResource
from src.mia.audio.audio_resource_manager import AudioResourceManager, AudioResource
from src.mia.multimodal.vision_resource_manager import VisionResourceManager, VisionResource
from src.mia.llm.llm_manager import LLMManager
from src.mia.exceptions import ConfigurationError, ResourceError

class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        
        # Create test configuration
        self.test_config = {
            'llm': {
                'provider': 'openai',
                'model_id': 'gpt-3.5-turbo',
                'api_key': 'test-key',
                'url': 'https://api.openai.com/v1/chat/completions',
                'max_tokens': 2048,
                'temperature': 0.8,
                'timeout': 60
            },
            'audio': {
                'enabled': True,
                'sample_rate': 16000,
                'chunk_size': 1024,
                'device_id': None,
                'input_threshold': 0.01,
                'speech_model': 'whisper-base',
                'tts_enabled': True
            },
            'vision': {
                'enabled': True,
                'model': 'clip-vit-base-patch32',
                'device': 'auto',
                'max_image_size': 1024,
                'supported_formats': ['jpg', 'png', 'gif']
            },
            'system': {
                'debug': False,
                'log_level': 'INFO',
                'max_workers': 4,
                'request_timeout': 30,
                'retry_attempts': 3,
                'cache_enabled': True,
                'cache_ttl': 3600
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        config_manager = ConfigManager(self.config_path)
        
        # Test LLM configuration
        self.assertEqual(config_manager.config.llm.provider, 'openai')
        self.assertEqual(config_manager.config.llm.model_id, 'gpt-3.5-turbo')
        self.assertEqual(config_manager.config.llm.max_tokens, 2048)
        
        # Test audio configuration
        self.assertTrue(config_manager.config.audio.enabled)
        self.assertEqual(config_manager.config.audio.sample_rate, 16000)
        
        # Test vision configuration
        self.assertTrue(config_manager.config.vision.enabled)
        self.assertEqual(config_manager.config.vision.model, 'clip-vit-base-patch32')
        
        # Test system configuration
        self.assertEqual(config_manager.config.system.log_level, 'INFO')
        self.assertEqual(config_manager.config.system.max_workers, 4)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager(self.config_path)
        
        # Test valid configuration
        self.assertTrue(config_manager.validate_config())
        
        # Test invalid configuration
        config_manager.config.llm.max_tokens = -1
        with self.assertRaises(ConfigurationError):
            config_manager.validate_config()
    
    def test_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {'MIA_LLM_API_KEY': 'env-key'}):
            config_manager = ConfigManager(self.config_path)
            # Environment variable should override config file
            self.assertEqual(config_manager.config.llm.api_key, 'env-key')
    
    def test_config_update(self):
        """Test configuration updates."""
        config_manager = ConfigManager(self.config_path)
        
        # Test updating configuration
        config_manager.update_config('llm.temperature', 0.9)
        self.assertEqual(config_manager.config.llm.temperature, 0.9)
        
        # Test invalid path
        with self.assertRaises(ConfigurationError):
            config_manager.update_config('invalid.path', 'value')

class TestResourceManagement(unittest.TestCase):
    """Test resource management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.resource_manager = ResourceManager(max_memory_mb=100, cleanup_interval=1)
    
    def tearDown(self):
        """Clean up test environment."""
        self.resource_manager.cleanup()
    
    def test_resource_acquisition(self):
        """Test resource acquisition and release."""
        # Create a mock resource
        mock_data = Mock()
        
        # Acquire resource
        with self.resource_manager.acquire_resource("test_resource") as resource:
            resource.set_data(mock_data)
            self.assertEqual(resource.data, mock_data)
            self.assertIn("test_resource", self.resource_manager.resources)
        
        # Resource should be released after context
        self.assertNotIn("test_resource", self.resource_manager.resources)
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        # Create a resource that uses memory
        class MemoryResource(ManagedResource):
            def get_memory_usage(self):
                return 50 * 1024 * 1024  # 50MB
        
        resource = MemoryResource("memory_test", Mock())
        self.resource_manager.resources["memory_test"] = resource
        
        # Check memory usage
        total_memory = self.resource_manager.get_total_memory_usage()
        self.assertEqual(total_memory, 50 * 1024 * 1024)
    
    def test_cleanup_thread(self):
        """Test cleanup thread functionality."""
        # This test is complex due to threading, so we'll test the basics
        self.assertTrue(self.resource_manager.cleanup_thread.is_alive())
        
        # Test cleanup
        self.resource_manager.cleanup()
        self.assertFalse(self.resource_manager.cleanup_thread.is_alive())

class TestAudioResourceManager(unittest.TestCase):
    """Test audio resource management."""
    
    def setUp(self):
        """Set up test environment."""
        self.audio_manager = AudioResourceManager(max_memory_mb=100, cleanup_interval=1)
    
    def tearDown(self):
        """Clean up test environment."""
        self.audio_manager.cleanup()
    
    def test_audio_resource_acquisition(self):
        """Test audio resource acquisition."""
        mock_audio_component = Mock()
        
        # Acquire audio resource
        resource = self.audio_manager.acquire_audio_resource("test_audio", mock_audio_component)
        self.assertIsInstance(resource, AudioResource)
        self.assertEqual(resource.data, mock_audio_component)
        self.assertIn("test_audio", self.audio_manager.audio_resources)
    
    def test_recording_management(self):
        """Test recording state management."""
        mock_audio_component = Mock()
        resource = self.audio_manager.acquire_audio_resource("test_audio", mock_audio_component)
        
        # Test recording state
        self.assertFalse(self.audio_manager.is_recording_active())
        
        resource.start_recording()
        self.assertTrue(self.audio_manager.is_recording_active())
        
        resource.stop_recording()
        self.assertFalse(self.audio_manager.is_recording_active())
    
    def test_audio_cleanup(self):
        """Test audio resource cleanup."""
        mock_audio_component = Mock()
        mock_audio_component.stop_recording = Mock()
        mock_audio_component.close = Mock()
        
        resource = self.audio_manager.acquire_audio_resource("test_audio", mock_audio_component)
        self.audio_manager.release_audio_resource("test_audio")
        
        # Verify cleanup was called
        mock_audio_component.stop_recording.assert_called_once()
        mock_audio_component.close.assert_called_once()

class TestVisionResourceManager(unittest.TestCase):
    """Test vision resource management."""
    
    def setUp(self):
        """Set up test environment."""
        self.vision_manager = VisionResourceManager(max_memory_mb=500, cleanup_interval=2)
    
    def tearDown(self):
        """Clean up test environment."""
        self.vision_manager.cleanup()
    
    def test_vision_resource_acquisition(self):
        """Test vision resource acquisition."""
        mock_vision_component = Mock()
        
        # Acquire vision resource
        resource = self.vision_manager.acquire_vision_resource("test_vision", mock_vision_component)
        self.assertIsInstance(resource, VisionResource)
        self.assertEqual(resource.data, mock_vision_component)
        self.assertIn("test_vision", self.vision_manager.vision_resources)
    
    def test_processing_management(self):
        """Test processing state management."""
        mock_vision_component = Mock()
        resource = self.vision_manager.acquire_vision_resource("test_vision", mock_vision_component)
        
        # Test processing state
        self.assertFalse(self.vision_manager.is_processing_active())
        
        resource.start_processing()
        self.assertTrue(self.vision_manager.is_processing_active())
        
        resource.stop_processing()
        self.assertFalse(self.vision_manager.is_processing_active())

class TestIntegration(unittest.TestCase):
    """Test integration between configuration and resource management."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        
        # Create minimal configuration
        config_data = {
            'llm': {
                'provider': 'ollama',
                'model_id': 'test-model',
                'api_key': 'test-key',
                'url': 'http://localhost:11434',
                'max_tokens': 1024,
                'temperature': 0.7,
                'timeout': 30
            },
            'system': {
                'debug': True,
                'log_level': 'DEBUG'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    @patch('src.mia.llm.llm_manager.requests')
    def test_llm_manager_with_config(self, mock_requests):
        """Test LLM manager with configuration."""
        config_manager = ConfigManager(self.config_path)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'test response'}
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response
        
        # Create LLM manager with configuration
        llm_manager = LLMManager(config_manager=config_manager)
        
        # Verify configuration was applied
        self.assertEqual(llm_manager.provider, 'ollama')
        self.assertEqual(llm_manager.model_id, 'test-model')
        self.assertEqual(llm_manager.url, 'http://localhost:11434')
        self.assertEqual(llm_manager.max_tokens, 1024)
        self.assertEqual(llm_manager.temperature, 0.7)

if __name__ == '__main__':
    unittest.main()
