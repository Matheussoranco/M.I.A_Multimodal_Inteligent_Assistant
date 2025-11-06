"""
M.I.A - Multimodal Intelligent Assistant
Package initialization with safe lazy exports to avoid circular dependencies
and accidental side effects at import time.
"""

from typing import Any

# Version information
from .__version__ import __version__

__author__ = "M.I.A Development Team"
__description__ = "A multimodal intelligent assistant with text and audio capabilities"


# Public API (only symbols we can reliably provide)
__all__ = [
    "__version__",
    # Lazy accessors
    "get_main",
    "get_llm_manager", 
    "get_agent_memory",
    "get_error_handler",
]


# Lazily resolve heavy or optional imports
def get_main():
    """Return the CLI entrypoint function lazily."""
    from .main import main

    return main


def get_llm_manager():
    """Return the LLMManager class lazily."""
    from .llm.llm_manager import LLMManager

    return LLMManager


def get_agent_memory():
    """Return the AgentMemory class lazily."""
    from .memory.knowledge_graph import AgentMemory

    return AgentMemory


def get_error_handler():
    """Return error handler helpers lazily."""
    from .error_handler import (
        ErrorHandler,
        global_error_handler,
        with_error_handling,
        safe_execute,
    )

    return ErrorHandler, global_error_handler, with_error_handling, safe_execute


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name == "MIACognitiveCore":
        from .core.cognitive_architecture import MIACognitiveCore

        return MIACognitiveCore
    if name == "SecurityManager":
        from .security.security_manager import SecurityManager

        return SecurityManager
    if name == "AgentMemory":
        from .memory.knowledge_graph import AgentMemory

        return AgentMemory

    # Exceptions
    if name in {
        "MIAException",
        "LLMProviderError",
        "AudioProcessingError",
        "VisionProcessingError",
        "SecurityError",
        "ConfigurationError",
        "ActionExecutionError",
        "InitializationError",
        "NetworkError",
        "ValidationError",
    }:
        from . import exceptions as _exc

        return getattr(_exc, name)
    if name == "MIAMemoryError":
        from . import exceptions as _exc

        return _exc.MemoryError

    raise AttributeError(f"module 'mia' has no attribute {name!r}")
