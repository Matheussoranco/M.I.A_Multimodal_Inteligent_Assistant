"""
M.I.A - Multimodal Intelligent Assistant
Main package initialization with minimal imports to avoid circular dependencies.
"""

# Import version information
from .__version__ import __version__

# Minimal exports to avoid circular imports during build
__all__ = ['__version__']

# Lazy imports for main functionality
def get_main():
    """Lazy import of main function to avoid circular imports."""
    from .main import main
    return main

def get_llm_manager():
    """Lazy import of LLM manager."""
    from .llm.llm_manager import LLMManager
    return LLMManager

def get_agent_memory():
    """Lazy import of agent memory."""
    from .memory.knowledge_graph import AgentMemory
    return AgentMemory

# Lazy imports for error handling
def get_error_handler():
    """Lazy import of error handler."""
    from .error_handler import ErrorHandler, global_error_handler, with_error_handling, safe_execute
    return ErrorHandler, global_error_handler, with_error_handling, safe_execute

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
