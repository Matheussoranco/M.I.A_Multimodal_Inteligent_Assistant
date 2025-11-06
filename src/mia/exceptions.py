"""
Custom exception classes for M.I.A - Multimodal Intelligent Assistant
Provides structured error handling across all modules.
"""

from typing import Any, Dict, Optional


class MIAException(Exception):
    """Base exception for all M.I.A errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class LLMProviderError(MIAException):
    """Raised when LLM provider encounters an error."""

    pass


class AudioProcessingError(MIAException):
    """Raised when audio processing fails."""

    pass


class VisionProcessingError(MIAException):
    """Raised when vision processing fails."""

    pass


class SecurityError(MIAException):
    """Raised when security validation fails."""

    pass


class ConfigurationError(MIAException):
    """Raised when configuration is invalid."""

    pass


class MemoryError(MIAException):
    """Raised when memory operations fail."""

    pass


class ActionExecutionError(MIAException):
    """Raised when action execution fails."""

    pass


class InitializationError(MIAException):
    """Raised when component initialization fails."""

    pass


class NetworkError(MIAException):
    """Raised when network operations fail."""

    pass


class ValidationError(MIAException):
    """Raised when input validation fails."""

    pass


class PerformanceError(MIAException):
    """Raised when performance-related issues occur."""

    pass


class ResourceError(MIAException):
    """Raised when resource-related issues occur."""

    pass


class AudioError(MIAException):
    """Raised when audio-related issues occur."""

    pass


class VisionError(MIAException):
    """Raised when vision-related issues occur."""

    pass


class CacheError(MIAException):
    """Raised when cache-related issues occur."""

    pass
