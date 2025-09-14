"""
Test fixtures and utilities for integration testing.
Provides real component instances and test data for comprehensive testing.
"""
import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Import actual components for integration testing
from mia.config_manager import ConfigManager
from mia.resource_manager import ResourceManager
from mia.cache_manager import CacheManager
from mia.performance_monitor import PerformanceMonitor


@pytest.fixture(scope="session")
def test_config_dir():
    """Create a temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        'llm': {
            'provider': 'openai',
            'model_id': 'gpt-3.5-turbo',
            'api_key': 'test-key',
            'max_tokens': 1000,
            'temperature': 0.7,
            'timeout': 30
        },
        'audio': {
            'enabled': True,
            'sample_rate': 16000,
            'chunk_size': 1024,
            'device_id': None,
            'input_threshold': 0.01
        },
        'vision': {
            'enabled': True,
            'model': 'clip-vit-base-patch32',
            'device': 'cpu',
            'max_image_size': 512
        },
        'system': {
            'debug': False,
            'log_level': 'INFO',
            'max_workers': 2,
            'cache_enabled': True,
            'cache_ttl': 300
        }
    }


@pytest.fixture
def config_manager(test_config_dir, sample_config_data):
    """Real ConfigManager instance for integration testing."""
    config_path = os.path.join(test_config_dir, "test_config.yaml")

    # Write sample config
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_data, f)

    manager = ConfigManager(config_path)
    return manager


@pytest.fixture
def resource_manager():
    """Real ResourceManager instance for integration testing."""
    return ResourceManager(max_memory_mb=50)  # Small limit for testing


@pytest.fixture
def cache_manager(test_data_dir):
    """Real CacheManager instance for integration testing."""
    # Create a mock config manager for the cache manager
    mock_config = Mock()
    mock_config.get.return_value = None  # Return None for any config requests

    # We'll need to patch the PersistentCache to use our test directory
    with patch('mia.cache_manager.PersistentCache') as MockPersistentCache:
        mock_persistent = MockPersistentCache.return_value
        mock_persistent.get.return_value = None
        mock_persistent.put.return_value = None

        manager = CacheManager(config_manager=mock_config)
        # Override the persistent cache to use our test directory
        manager.persistent_cache = MockPersistentCache(cache_dir=test_data_dir, max_size_mb=10)
        return manager


@pytest.fixture
def performance_monitor():
    """Real PerformanceMonitor instance for integration testing."""
    return PerformanceMonitor()


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        'text': 'This is a test response from the LLM.',
        'tokens_used': 50,
        'finish_reason': 'stop'
    }


@pytest.fixture
def mock_image_data():
    """Mock image data for testing."""
    return {
        'path': '/fake/path/test_image.jpg',
        'size': (800, 600),
        'format': 'JPEG'
    }


@pytest.fixture
def mock_audio_data():
    """Mock audio data for testing."""
    return {
        'path': '/fake/path/test_audio.wav',
        'duration': 5.0,
        'sample_rate': 16000,
        'channels': 1
    }


@pytest.fixture
def test_user_query():
    """Sample user query for testing."""
    return "What is the capital of France?"


@pytest.fixture
def test_conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
    ]


@pytest.fixture
def mock_external_api():
    """Mock external API responses."""
    def _mock_response(service_name, endpoint, data=None):
        responses = {
            'openai': {
                'chat/completions': {
                    'choices': [{'message': {'content': 'Mock response'}}],
                    'usage': {'total_tokens': 100}
                }
            },
            'huggingface': {
                'inference': {'generated_text': 'Mock generated text'}
            },
            'google_vision': {
                'annotate': {'responses': [{'textAnnotations': []}]}
            }
        }
        return responses.get(service_name, {}).get(endpoint, {})

    return _mock_response


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup logic here if needed


# Test utilities
def create_test_file(directory, filename, content=""):
    """Create a test file with given content."""
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def create_test_image(directory, filename="test.jpg"):
    """Create a mock test image file."""
    filepath = os.path.join(directory, filename)
    # Create a minimal valid image file for testing
    with open(filepath, 'wb') as f:
        # Minimal JPEG header
        f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x00\x10\x00\x10\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4')
    return filepath


def create_test_audio(directory, filename="test.wav"):
    """Create a mock test audio file."""
    filepath = os.path.join(directory, filename)
    # Create a minimal WAV file header
    with open(filepath, 'wb') as f:
        # RIFF header
        f.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
    return filepath


def assert_cache_hit(cache_manager, key):
    """Assert that a cache key exists and has been accessed."""
    assert key in cache_manager.cache
    entry = cache_manager.cache[key]
    assert entry.access_count > 0


def assert_performance_metrics(performance_monitor, operation, min_calls=1):
    """Assert that performance metrics were recorded for an operation."""
    metrics = performance_monitor.get_metrics()
    assert operation in metrics
    assert metrics[operation]['call_count'] >= min_calls