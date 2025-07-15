"""
M.I.A - Multimodal Intelligent Assistant
Main package initialization with error handling support.
"""

# Export core components
from .main import main
from .llm.llm_manager import LLMManager
from .core.cognitive_architecture import MIACognitiveCore
from .security.security_manager import SecurityManager
from .memory.knowledge_graph import AgentMemory

# Export error handling components
from .exceptions import *
from .error_handler import ErrorHandler, global_error_handler, with_error_handling, safe_execute

__version__ = "2.0.0"
__author__ = "M.I.A Development Team"
__description__ = "A multimodal intelligent assistant with text and audio capabilities"

# Package metadata
__all__ = [
    'main',
    'LLMManager',
    'MIACognitiveCore', 
    'SecurityManager',
    'AgentMemory',
    'ErrorHandler',
    'global_error_handler',
    'with_error_handling',
    'safe_execute',
    'MIAException',
    'LLMProviderError',
    'AudioProcessingError',
    'VisionProcessingError',
    'SecurityError',
    'ConfigurationError',
    'MemoryError',
    'ActionExecutionError',
    'InitializationError',
    'NetworkError',
    'ValidationError'
]
