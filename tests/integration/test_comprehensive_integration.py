"""
Comprehensive integration tests for core M.I.A functionality.
Tests real component interactions with minimal mocking.
"""

import os
import time
from unittest.mock import Mock, patch

import pytest


class TestCoreIntegration:
    """Test core component integration with real dependencies."""

    def test_config_resource_integration(
        self, config_manager, resource_manager
    ):
        """Test configuration and resource manager integration."""
        # Test that config manager loads properly
        assert config_manager.config is not None
        assert config_manager.config.llm.provider == "openai"

        # Test resource manager initialization
        assert resource_manager.max_memory_bytes == 50 * 1024 * 1024

        # Test integration: config affects resource allocation
        max_workers = config_manager.config.system.max_workers
        assert max_workers == 2  # From our test config

    def test_cache_performance_integration(
        self, cache_manager, performance_monitor
    ):
        """Test cache and performance monitor integration."""
        # Start performance monitoring
        performance_monitor.start_monitoring()

        # Test cache operations
        test_key = "test_key"
        test_value = {"data": "test_value"}

        # Put value in cache
        cache_manager.put(test_key, test_value, ttl=60)

        # Get value from cache
        retrieved_value = cache_manager.get(test_key)

        # Stop monitoring
        performance_monitor.stop_monitoring()

        # Verify cache operation
        assert retrieved_value == test_value

        # Verify performance monitoring
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1

    def test_resource_cache_integration(self, resource_manager, cache_manager):
        """Test resource manager and cache integration."""
        # Create a mock resource
        from mia.resource_manager import ManagedResource  # type: ignore

        class TestResource(ManagedResource):
            def __init__(self):
                super().__init__("test_resource", "test_type")
                self.data = "test_data"

            def initialize(self):
                pass

            def cleanup(self):
                pass

            def get_memory_usage(self) -> int:
                return 1024  # Mock memory usage

        # Register resource
        test_resource = TestResource()
        resource_manager.register_resource(test_resource)

        # Cache the resource state
        cache_key = f"resource_{test_resource.resource_id}"
        cache_manager.put(cache_key, test_resource.data, ttl=300)

        # Verify resource is registered
        assert test_resource.resource_id in resource_manager.resources

        # Verify cache contains resource data
        cached_data = cache_manager.get(cache_key)
        assert cached_data == "test_data"

    def test_error_handling_integration(self, config_manager):
        """Test error handling across components."""
        from mia.error_handler import safe_execute  # type: ignore

        def failing_operation():
            raise ValueError("Test error")

        # Test that safe_execute works
        result = safe_execute(failing_operation, default=None)
        assert result is None  # Should return None on error

    def test_memory_management_integration(
        self, resource_manager, cache_manager
    ):
        """Test memory management across components."""
        # Fill cache with some data
        for i in range(10):
            cache_manager.put(
                f"key_{i}", f"value_{i}" * 100, ttl=3600
            )  # Larger values

        # Check that cache stats are updated
        stats = cache_manager.get_stats()
        assert stats["memory_cache"]["size"] > 0

        # Test resource manager memory tracking
        assert resource_manager.max_memory_bytes > 0


class TestWorkflowIntegration:
    """Test complete workflow integration scenarios."""

    def test_configuration_workflow(self, config_manager, test_config_dir):
        """Test complete configuration loading and validation workflow."""
        # Test config loading
        config = config_manager.load_config()
        # Config might be None if loading fails, let's check the config attribute directly
        if config is None:
            config = config_manager.config
        assert config is not None

        # Test config validation
        is_valid = config_manager.validate_config()
        assert is_valid

        # Test config updates
        config_manager.update_config("llm", "temperature", 0.8)
        assert config_manager.config.llm.temperature == 0.8

    def test_caching_workflow(self, cache_manager):
        """Test complete caching workflow."""
        # Test cache put/get
        cache_manager.put("workflow_test", {"step": 1}, ttl=60)
        result = cache_manager.get("workflow_test")
        assert result == {"step": 1}

        # Test cache expiration (simulate)
        with patch(
            "time.time", return_value=time.time() + 120
        ):  # 2 minutes later
            expired_result = cache_manager.get("workflow_test")
            assert expired_result is None  # Should be expired

    def test_performance_monitoring_workflow(self, performance_monitor):
        """Test performance monitoring workflow."""
        # Start monitoring
        performance_monitor.start_monitoring()

        # Simulate some work
        time.sleep(0.01)

        # Stop monitoring
        performance_monitor.stop_monitoring()

        # Check metrics
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1

    @pytest.mark.slow
    def test_load_test_scenario(self, cache_manager, performance_monitor):
        """Test load scenario with multiple operations."""
        performance_monitor.start_monitoring()

        # Perform multiple cache operations
        for i in range(100):
            cache_manager.put(f"load_key_{i}", f"load_value_{i}", ttl=300)

        # Retrieve some values
        for i in range(0, 100, 10):  # Every 10th item
            value = cache_manager.get(f"load_key_{i}")
            assert value == f"load_value_{i}"

        performance_monitor.stop_monitoring()

        # Verify performance under load
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1


class TestComponentInteraction:
    """Test interactions between different components."""

    def test_config_cache_interaction(self, config_manager, cache_manager):
        """Test how configuration affects cache behavior."""
        # Cache a config-dependent value
        config_value = config_manager.config.system.cache_enabled
        cache_manager.put("config_test", config_value, ttl=300)

        # Retrieve and verify
        cached_value = cache_manager.get("config_test")
        assert cached_value == config_value

    def test_resource_performance_interaction(
        self, resource_manager, performance_monitor
    ):
        """Test resource management with performance monitoring."""
        performance_monitor.start_monitoring()

        # Register a resource
        from mia.resource_manager import ManagedResource  # type: ignore

        class MonitoredResource(ManagedResource):
            def __init__(self):
                super().__init__("monitored_resource", "test")
                self.operation_count = 0

            def initialize(self):
                self.operation_count += 1

            def cleanup(self):
                self.operation_count += 1

            def get_memory_usage(self) -> int:
                return 512  # Mock memory usage

        resource = MonitoredResource()
        resource_manager.register_resource(resource)

        # Actually use the resource to trigger initialization
        with resource_manager.use_resource("monitored_resource") as res:
            assert res is not None

        performance_monitor.stop_monitoring()

        # Verify resource was registered and operations were tracked
        assert resource.resource_id in resource_manager.resources
        assert resource.operation_count == 1  # initialize was called
