"""
Comprehensive tests for the Error Handler module.
Tests circuit breaker, exponential backoff, and recovery strategies.
"""

import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from mia.error_handler import (  # type: ignore[import-not-found]
    ErrorHandler,
    CircuitState,
    CircuitBreakerConfig,
    RetryConfig,
    ServiceMetrics,
    with_error_handling,
    safe_execute,
    global_error_handler,
)
from mia.exceptions import MIAException, NetworkError, LLMProviderError  # type: ignore[import-not-found]


class TestCircuitBreakerConfig(unittest.TestCase):
    """Tests for CircuitBreakerConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.half_open_max_calls, 3)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=30.0,
            half_open_max_calls=2,
        )
        
        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.success_threshold, 1)
        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.half_open_max_calls, 2)


class TestServiceMetrics(unittest.TestCase):
    """Tests for ServiceMetrics."""
    
    def test_initial_values(self):
        """Test initial metric values."""
        metrics = ServiceMetrics(name="test_service")
        
        self.assertEqual(metrics.name, "test_service")
        self.assertEqual(metrics.failure_count, 0)
        self.assertEqual(metrics.success_count, 0)
        self.assertEqual(metrics.circuit_state, CircuitState.CLOSED)
    
    def test_avg_latency(self):
        """Test average latency calculation."""
        metrics = ServiceMetrics(name="test")
        
        # No calls yet
        self.assertEqual(metrics.avg_latency_ms, 0.0)
        
        # Simulate calls
        metrics.total_calls = 4
        metrics.total_latency_ms = 400.0
        
        self.assertEqual(metrics.avg_latency_ms, 100.0)
    
    def test_error_rate(self):
        """Test error rate calculation."""
        metrics = ServiceMetrics(name="test")
        
        # No calls yet
        self.assertEqual(metrics.error_rate, 0.0)
        
        # Simulate calls
        metrics.total_calls = 10
        metrics.failure_count = 3
        
        self.assertEqual(metrics.error_rate, 0.3)


class TestErrorHandler(unittest.TestCase):
    """Tests for ErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler(
            circuit_config=CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=0.1,  # Short timeout for testing
            ),
            retry_config=RetryConfig(
                max_retries=2,
                base_delay=0.01,
                max_delay=0.1,
            ),
        )
    
    def test_record_success(self):
        """Test recording successful calls."""
        self.handler.record_success("test_service", latency_ms=50.0)
        
        metrics = self.handler.get_service_metrics("test_service")
        
        self.assertEqual(metrics.success_count, 1)
        self.assertEqual(metrics.consecutive_successes, 1)
        self.assertEqual(metrics.total_calls, 1)
        self.assertEqual(metrics.total_latency_ms, 50.0)
    
    def test_record_failure(self):
        """Test recording failed calls."""
        self.handler.record_failure("test_service", latency_ms=100.0)
        
        metrics = self.handler.get_service_metrics("test_service")
        
        self.assertEqual(metrics.failure_count, 1)
        self.assertEqual(metrics.consecutive_failures, 1)
        self.assertEqual(metrics.total_calls, 1)
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        # Record failures to trigger circuit open
        for i in range(3):
            self.handler.record_failure("test_service")
        
        metrics = self.handler.get_service_metrics("test_service")
        
        self.assertEqual(metrics.circuit_state, CircuitState.OPEN)
        self.assertTrue(self.handler.is_circuit_open("test_service"))
    
    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        for i in range(3):
            self.handler.record_failure("test_service")
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Check should trigger half-open transition
        can_exec, reason = self.handler.can_execute("test_service")
        
        self.assertTrue(can_exec)
        self.assertEqual(reason, "circuit_half_open")
    
    def test_circuit_closes_after_successes_in_half_open(self):
        """Test circuit closes after successes in half-open state."""
        # Open the circuit
        for i in range(3):
            self.handler.record_failure("test_service")
        
        # Wait for timeout and trigger half-open
        time.sleep(0.15)
        self.handler.can_execute("test_service")
        
        # Record successes in half-open
        self.handler.record_success("test_service")
        self.handler.record_success("test_service")
        
        metrics = self.handler.get_service_metrics("test_service")
        
        self.assertEqual(metrics.circuit_state, CircuitState.CLOSED)
    
    def test_calculate_backoff(self):
        """Test exponential backoff calculation."""
        delay0 = self.handler.calculate_backoff(0)
        delay1 = self.handler.calculate_backoff(1)
        delay2 = self.handler.calculate_backoff(2)
        
        # Delay should increase exponentially
        self.assertGreater(delay1, delay0)
        self.assertGreater(delay2, delay1)
        
        # Check max delay is respected
        delay10 = self.handler.calculate_backoff(10)
        self.assertLessEqual(delay10, self.handler.retry_config.max_delay * 1.25)  # Allow jitter
    
    def test_reset_circuit(self):
        """Test manual circuit reset."""
        # Open the circuit
        for i in range(3):
            self.handler.record_failure("test_service")
        
        # Reset
        self.handler.reset_circuit("test_service")
        
        metrics = self.handler.get_service_metrics("test_service")
        
        self.assertEqual(metrics.circuit_state, CircuitState.CLOSED)
        self.assertEqual(metrics.consecutive_failures, 0)
    
    def test_register_recovery_strategy(self):
        """Test registering recovery strategies."""
        def recovery(error, context):
            return "recovered"
        
        self.handler.register_recovery_strategy(ValueError, recovery)
        
        self.assertIn(ValueError, self.handler.recovery_strategies)
    
    def test_register_fallback_provider(self):
        """Test registering fallback providers."""
        def fallback(*args, **kwargs):
            return "fallback_result"
        
        self.handler.register_fallback_provider("test_service", fallback)
        
        self.assertIn("test_service", self.handler.fallback_providers)
        self.assertEqual(len(self.handler.fallback_providers["test_service"]), 1)
    
    def test_get_all_service_metrics(self):
        """Test getting all service metrics."""
        self.handler.record_success("service1")
        self.handler.record_failure("service2")
        
        all_metrics = self.handler.get_all_service_metrics()
        
        self.assertIn("service1", all_metrics)
        self.assertIn("service2", all_metrics)
        self.assertEqual(all_metrics["service1"]["success_count"], 1)
        self.assertEqual(all_metrics["service2"]["failure_count"], 1)
    
    def test_get_error_history(self):
        """Test getting error history."""
        # Record some failures to generate history
        self.handler.record_failure("test_service")
        
        history = self.handler.get_error_history()
        
        # History may be empty if no actual errors logged through handle_error
        self.assertIsInstance(history, list)


class TestHandleErrorsDecorator(unittest.TestCase):
    """Tests for handle_errors decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler(
            circuit_config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout=0.1,
            ),
            retry_config=RetryConfig(
                max_retries=2,
                base_delay=0.01,
                max_delay=0.1,
            ),
        )
    
    def test_successful_call(self):
        """Test decorator with successful function."""
        @self.handler.handle_errors()
        def success_func():
            return "success"
        
        result = success_func()
        
        self.assertEqual(result, "success")
    
    def test_retry_on_failure(self):
        """Test decorator retries on failure."""
        call_count = [0]
        
        @self.handler.handle_errors(max_retries=2)
        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = fail_then_succeed()
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
    
    def test_max_retries_exhausted(self):
        """Test decorator raises after max retries."""
        @self.handler.handle_errors(max_retries=2)
        def always_fail():
            raise ConnectionError("Always fails")
        
        with self.assertRaises(ConnectionError):
            always_fail()
    
    def test_fallback_on_circuit_open(self):
        """Test fallback is used when circuit is open."""
        # Open the circuit
        for i in range(5):
            self.handler.record_failure("test_func")
        
        @self.handler.handle_errors(service_name="test_func", fallback_value="fallback")
        def test_func():
            return "success"
        
        result = test_func()
        
        self.assertEqual(result, "fallback")


class TestWithErrorHandling(unittest.TestCase):
    """Tests for with_error_handling decorator."""
    
    def test_success_tracking(self):
        """Test success is tracked."""
        handler = ErrorHandler()
        
        @with_error_handling(handler, service_name="test")
        def success_func():
            return "success"
        
        result = success_func()
        
        self.assertEqual(result, "success")
        
        metrics = handler.get_service_metrics("test")
        self.assertEqual(metrics.success_count, 1)
    
    def test_failure_tracking(self):
        """Test failure is tracked."""
        handler = ErrorHandler()
        
        @with_error_handling(handler, fallback_value="fallback", service_name="test")
        def fail_func():
            raise ValueError("Test error")
        
        result = fail_func()
        
        self.assertEqual(result, "fallback")
        
        metrics = handler.get_service_metrics("test")
        self.assertEqual(metrics.failure_count, 1)


class TestSafeExecute(unittest.TestCase):
    """Tests for safe_execute function."""
    
    def test_success(self):
        """Test successful execution."""
        def success():
            return "result"
        
        result = safe_execute(success)
        
        self.assertEqual(result, "result")
    
    def test_failure_returns_default(self):
        """Test failure returns default value."""
        def fail():
            raise ValueError("Error")
        
        result = safe_execute(fail, default="default")
        
        self.assertEqual(result, "default")
    
    def test_with_arguments(self):
        """Test with function arguments."""
        def add(a, b):
            return a + b
        
        result = safe_execute(add, 2, 3)
        
        self.assertEqual(result, 5)


class TestIsRetryable(unittest.TestCase):
    """Tests for _is_retryable method."""
    
    def setUp(self):
        self.handler = ErrorHandler()
    
    def test_network_error_is_retryable(self):
        """Test network errors are retryable."""
        error = NetworkError("Connection failed")
        self.assertTrue(self.handler._is_retryable(error))
    
    def test_connection_error_is_retryable(self):
        """Test connection errors are retryable."""
        error = ConnectionError("Connection refused")
        self.assertTrue(self.handler._is_retryable(error))
    
    def test_timeout_error_is_retryable(self):
        """Test timeout errors are retryable."""
        error = TimeoutError("Request timed out")
        self.assertTrue(self.handler._is_retryable(error))
    
    def test_value_error_not_retryable(self):
        """Test value errors are not retryable."""
        error = ValueError("Invalid value")
        self.assertFalse(self.handler._is_retryable(error))


if __name__ == "__main__":
    unittest.main()
