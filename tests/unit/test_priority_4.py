"""
Comprehensive Test Suite for M.I.A
"""
import unittest
import asyncio
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from mia.config_manager import ConfigManager
    from mia.performance_monitor import PerformanceMonitor
    from mia.cache_manager import CacheManager, LRUCache
    from mia.resource_manager import ResourceManager
    from mia.error_handler import ErrorHandler
    from mia.exceptions import (
        MIAException, LLMProviderError, AudioProcessingError, 
        VisionProcessingError, SecurityError, ConfigurationError,
        MemoryError, ActionExecutionError, InitializationError,
        NetworkError, ValidationError
    )
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for testing
    class ConfigManager:
        def __init__(self):
            self.config = None
    class PerformanceMonitor:
        pass
    class CacheManager:
        pass
    class LRUCache:
        pass
    class ResourceManager:
        pass
    class ErrorHandler:
        pass
    # Mock exceptions
    class MIAException(Exception):
        pass
    ValidationError = ValueError

class TestConfigManager(unittest.TestCase):
    """Test configuration manager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)
        
    def test_config_initialization(self):
        """Test configuration initialization."""
        config_manager = ConfigManager()
        self.assertIsNotNone(config_manager.config)
        self.assertIsNotNone(config_manager.config.llm)
        self.assertIsNotNone(config_manager.config.audio)
        self.assertIsNotNone(config_manager.config.vision)
        
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Test valid configuration
        self.assertTrue(config_manager.validate_config())
        
        # Test invalid configuration
        config_manager.config.llm.max_tokens = -1
        self.assertFalse(config_manager.validate_config())
        
    def test_config_save_load(self):
        """Test configuration save and load."""
        config_manager = ConfigManager()
        
        # Modify configuration
        config_manager.config.llm.model_id = "test-model"
        config_manager.config.audio.sample_rate = 22050
        
        # Save configuration
        config_manager.save_config(self.config_path)
        
        # Load configuration
        new_config_manager = ConfigManager(self.config_path)
        self.assertEqual(new_config_manager.config.llm.model_id, "test-model")
        self.assertEqual(new_config_manager.config.audio.sample_rate, 22050)

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.perf_monitor = PerformanceMonitor()
        
    def tearDown(self):
        """Clean up test environment."""
        self.perf_monitor.cleanup()
        
    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = self.perf_monitor._collect_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.cpu_percent, 0)
        self.assertGreater(metrics.memory_percent, 0)
        self.assertGreater(metrics.memory_used_mb, 0)
        self.assertGreater(metrics.timestamp, 0)
        
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        self.perf_monitor.start_monitoring()
        self.assertTrue(self.perf_monitor.monitoring_active)
        
        # Wait for some metrics
        import time
        time.sleep(1)
        
        # Check metrics were collected
        metrics = self.perf_monitor.get_current_metrics()
        self.assertIsNotNone(metrics)
        
        # Stop monitoring
        self.perf_monitor.stop_monitoring()
        self.assertFalse(self.perf_monitor.monitoring_active)
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Start monitoring and collect some metrics
        self.perf_monitor.start_monitoring()
        import time
        time.sleep(1)
        
        summary = self.perf_monitor.get_performance_summary()
        
        self.assertIn('average_cpu_percent', summary)
        self.assertIn('average_memory_percent', summary)
        self.assertIn('memory_used_mb', summary)
        
        self.perf_monitor.stop_monitoring()

class TestCacheManager(unittest.TestCase):
    """Test caching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache_manager = CacheManager()
        
    def tearDown(self):
        """Clean up test environment."""
        self.cache_manager.clear_all()
        
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations."""
        cache = LRUCache(max_size=3, default_ttl=10)
        
        # Test put and get
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
        
        # Test miss
        self.assertIsNone(cache.get("nonexistent"))
        
        # Test LRU eviction
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "value2")
        
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
        
        # Wait for TTL to expire
        import time
        time.sleep(0.2)
        
        self.assertIsNone(cache.get("key1"))
        
    def test_cache_manager_operations(self):
        """Test cache manager operations."""
        # Test put and get
        self.cache_manager.put("test_key", "test_value")
        self.assertEqual(self.cache_manager.get("test_key"), "test_value")
        
        # Test stats
        stats = self.cache_manager.get_stats()
        self.assertIn('memory_cache', stats)
        self.assertIn('persistent_cache', stats)
        
    def test_cache_decorator(self):
        """Test cache decorator functionality."""
        call_count = 0
        
        @self.cache_manager.cached(ttl=10)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
            
        # First call - should execute function
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)
        
        # Second call - should use cache
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Should not increment

class TestResourceManager(unittest.TestCase):
    """Test resource management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.resource_manager = ResourceManager()
        
    def tearDown(self):
        """Clean up test environment."""
        self.resource_manager.stop()
        
    def test_resource_acquisition(self):
        """Test resource acquisition and release."""
        # Test resource acquisition
        with self.resource_manager.acquire_resource("test_resource") as resource:
            self.assertIsNotNone(resource)
            resource.set_data("test_data")
            self.assertEqual(resource.get_data(), "test_data")
            
        # Resource should be released after context
        self.assertNotIn("test_resource", self.resource_manager.resources)
        
    def test_resource_cleanup(self):
        """Test resource cleanup."""
        # Create mock resource with cleanup method
        mock_resource = Mock()
        mock_resource.cleanup = Mock()
        
        with self.resource_manager.acquire_resource("test_resource") as resource:
            resource.set_data(mock_resource)
            
        # Cleanup should be called
        mock_resource.cleanup.assert_called_once()
        
    def test_memory_monitoring(self):
        """Test memory monitoring."""
        # Get initial memory usage
        initial_usage = self.resource_manager.get_memory_usage()
        self.assertGreaterEqual(initial_usage, 0)
        
        # Create resource and check memory usage
        with self.resource_manager.acquire_resource("test_resource") as resource:
            resource.set_data("large_data" * 1000)
            
        # Memory usage should be tracked
        stats = self.resource_manager.get_stats()
        self.assertIn('total_memory_mb', stats)
        self.assertIn('active_resources', stats)

class TestErrorHandler(unittest.TestCase):
    """Test error handling functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.error_handler = ErrorHandler()
        
    def test_error_handling_decorator(self):
        """Test error handling decorator."""
        @self.error_handler.handle_errors(max_retries=2)
        def failing_function():
            raise ValueError("Test error")
            
        # Should handle the error and return None
        result = failing_function()
        self.assertIsNone(result)
        
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        # Simulate multiple failures
        for _ in range(6):  # Default threshold is 5
            self.error_handler.record_failure("test_service")
            
        # Circuit should be open
        self.assertTrue(self.error_handler.is_circuit_open("test_service"))
        
        # Should fail fast
        with self.assertRaises(Exception):
            self.error_handler.check_circuit_breaker("test_service")
            
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test recovery strategy
        def recovery_func():
            return "recovered"
            
        self.error_handler.add_recovery_strategy(ValueError, recovery_func)
        
        @self.error_handler.handle_errors()
        def failing_function():
            raise ValueError("Test error")
            
        result = failing_function()
        self.assertEqual(result, "recovered")

class TestIntegration(unittest.TestCase):
    """Integration tests for M.I.A components."""
    
    def test_config_performance_integration(self):
        """Test configuration and performance monitoring integration."""
        config_manager = ConfigManager()
        perf_monitor = PerformanceMonitor(config_manager)
        
        # Start monitoring
        perf_monitor.start_monitoring()
        
        # Wait for metrics
        import time
        time.sleep(1)
        
        # Check metrics
        metrics = perf_monitor.get_current_metrics()
        self.assertIsNotNone(metrics)
        
        # Stop monitoring
        perf_monitor.stop_monitoring()
        perf_monitor.cleanup()
        
    def test_cache_performance_integration(self):
        """Test cache and performance integration."""
        cache_manager = CacheManager()
        perf_monitor = PerformanceMonitor()
        
        # Start monitoring
        perf_monitor.start_monitoring()
        
        # Perform cache operations
        for i in range(100):
            cache_manager.put(f"key_{i}", f"value_{i}")
            
        for i in range(100):
            cache_manager.get(f"key_{i}")
            
        # Check cache stats
        stats = cache_manager.get_stats()
        self.assertGreater(stats['memory_cache']['hits'], 0)
        
        # Clean up
        cache_manager.clear_all()
        perf_monitor.stop_monitoring()
        perf_monitor.cleanup()

def run_performance_tests():
    """Run performance-specific tests."""
    print("Running performance tests...")
    
    # Test performance under load
    cache_manager = CacheManager()
    perf_monitor = PerformanceMonitor()
    
    perf_monitor.start_monitoring()
    
    # Simulate load
    import time
    start_time = time.time()
    
    for i in range(1000):
        cache_manager.put(f"perf_key_{i}", f"perf_value_{i}")
        
    for i in range(1000):
        cache_manager.get(f"perf_key_{i}")
        
    end_time = time.time()
    
    print(f"Cache operations completed in {end_time - start_time:.2f} seconds")
    
    # Check performance metrics
    metrics = perf_monitor.get_current_metrics()
    if metrics:
        print(f"CPU usage: {metrics.cpu_percent:.1f}%")
        print(f"Memory usage: {metrics.memory_percent:.1f}%")
        
    # Clean up
    cache_manager.clear_all()
    perf_monitor.stop_monitoring()
    perf_monitor.cleanup()
    
    print("Performance tests completed")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
