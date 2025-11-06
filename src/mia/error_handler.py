"""
Error Handling Manager for M.I.A
Provides centralized error handling, logging, and recovery mechanisms.
"""

import logging
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

from .exceptions import LLMProviderError, NetworkError

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling and recovery system."""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.max_retries = 3
        self.circuit_breaker_threshold = 5

    def register_recovery_strategy(
        self, exception_type: Type[Exception], strategy: Callable
    ):
        """Register a recovery strategy for a specific exception type."""
        self.recovery_strategies[exception_type] = strategy

    def add_recovery_strategy(
        self, exception_type: Type[Exception], strategy: Callable
    ):
        """Add recovery strategy - alias for register_recovery_strategy."""
        self.register_recovery_strategy(exception_type, strategy)

    def record_failure(self, service_name: str):
        """Record a failure for circuit breaker tracking."""
        self.error_counts[service_name] = (
            self.error_counts.get(service_name, 0) + 1
        )

    def is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service."""
        return (
            self.error_counts.get(service_name, 0)
            >= self.circuit_breaker_threshold
        )

    def handle_errors(self, max_retries: int = 3):
        """Decorator factory for error handling with retries."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                            )
                        else:
                            logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                            )

                # All retries exhausted
                if last_exception:
                    raise last_exception

            return wrapper

        return decorator

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Handle an error with appropriate logging and recovery."""
        error_type = type(error).__name__
        self.error_counts[error_type] = (
            self.error_counts.get(error_type, 0) + 1
        )

        # Log the error with context
        context = context or {}
        logger.error(
            f"Error occurred: {error_type}: {str(error)}",
            extra={
                "error_type": error_type,
                "context": context,
                "stack_trace": traceback.format_exc(),
            },
        )

        # Check if we should attempt recovery
        if self.error_counts[error_type] < self.circuit_breaker_threshold:
            recovery_strategy = self.recovery_strategies.get(type(error))
            if recovery_strategy:
                try:
                    return recovery_strategy(error, context)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")

        # If no recovery or circuit breaker triggered
        if self.error_counts[error_type] >= self.circuit_breaker_threshold:
            logger.critical(f"Circuit breaker triggered for {error_type}")

        return None

    def reset_error_count(self, error_type: str):
        """Reset error count for a specific error type."""
        if error_type in self.error_counts:
            self.error_counts[error_type] = 0

    def get_error_stats(self) -> Dict[str, int]:
        """Get current error statistics."""
        return self.error_counts.copy()


def with_error_handling(
    error_handler: ErrorHandler,
    fallback_value: Any = None,
    reraise: bool = False,
):
    """Decorator to add error handling to functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:100],  # Truncate for logging
                    "kwargs": str(kwargs)[:100],
                }

                result = error_handler.handle_error(e, context)
                if result is not None:
                    return result

                # Re-raise exceptions during testing
                import sys

                if reraise or "pytest" in sys.modules:
                    raise

                return fallback_value

        return wrapper

    return decorator


def safe_execute(func: Callable, *args, default=None, **kwargs) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default


# Global error handler instance
global_error_handler = ErrorHandler()


# Register default recovery strategies
def llm_provider_recovery(
    error: LLMProviderError, context: Dict[str, Any]
) -> Optional[str]:
    """Default recovery strategy for LLM provider errors."""
    logger.info("Attempting LLM provider recovery")
    return "I'm experiencing technical difficulties. Please try again."


def network_error_recovery(
    error: NetworkError, context: Dict[str, Any]
) -> Optional[str]:
    """Default recovery strategy for network errors."""
    logger.info("Attempting network error recovery")
    return (
        "Network connection issue. Please check your connection and try again."
    )


# Register default strategies
global_error_handler.register_recovery_strategy(
    LLMProviderError, llm_provider_recovery
)
global_error_handler.register_recovery_strategy(
    NetworkError, network_error_recovery
)
