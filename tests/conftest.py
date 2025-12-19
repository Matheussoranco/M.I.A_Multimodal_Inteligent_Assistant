"""
Test configuration for M.I.A project
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add project root to path as well
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for testing."""
    mock = MagicMock()
    mock.generate.return_value = "Test response from LLM"
    mock.stream_generate.return_value = iter(["Test ", "streaming ", "response"])
    return mock


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for testing."""
    mock = MagicMock()
    mock.embed.return_value = [[0.1] * 384]  # Standard embedding dimension
    mock.dimension = 384
    return mock


@pytest.fixture
def mock_action_executor():
    """Mock action executor for testing."""
    mock = MagicMock()
    result = MagicMock()
    result.success = True
    result.output = "Action executed successfully"
    mock.execute.return_value = result
    return mock


@pytest.fixture
def mock_security_manager():
    """Mock security manager for testing."""
    mock = MagicMock()
    mock.check_permission.return_value = True
    mock.is_path_allowed.return_value = True
    return mock


@pytest.fixture
def mock_memory():
    """Mock memory system for testing."""
    mock = MagicMock()
    mock.store.return_value = "memory_id_123"
    mock.retrieve.return_value = [
        {"id": "memory_id_123", "content": "Test memory", "similarity": 0.9}
    ]
    return mock


@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        "version": "1.0.0",
        "environment": "testing",
        "debug": True,
        "llm": {
            "default_provider": "openai",
            "openai": {
                "api_key": "test_key",
                "model": "gpt-3.5-turbo",
            }
        },
        "memory": {
            "enabled": True,
            "vector_store_path": "./test_memory",
        },
        "vision": {
            "enabled": True,
            "provider": "blip",
        },
        "security": {
            "sandbox_enabled": True,
            "filesystem_access": True,
        }
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_text():
    """Sample text for testing chunking and RAG."""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    as opposed to natural intelligence displayed by animals including humans.
    AI research has been defined as the field of study of intelligent agents,
    which refers to any system that perceives its environment and takes actions
    that maximize its chance of achieving its goals.
    
    The term "artificial intelligence" had previously been used to describe
    machines that mimic and display "human" cognitive skills that are associated
    with the human mind, such as "learning" and "problem-solving".
    
    Machine learning is a subset of artificial intelligence that provides systems
    the ability to automatically learn and improve from experience without being
    explicitly programmed. Deep learning is a subset of machine learning that
    uses neural networks with many layers.
    """


@pytest.fixture
def sample_image_bytes():
    """Sample image bytes for testing vision processing."""
    # 1x1 pixel PNG
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'

