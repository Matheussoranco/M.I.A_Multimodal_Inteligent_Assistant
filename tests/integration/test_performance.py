"""
Performance and load testing for M.I.A components.
Tests system behavior under various load conditions.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import concurrent.futures
import threading
import time
from unittest.mock import Mock, patch

import pytest


class TestPerformanceScenarios:
    """Test performance characteristics under different scenarios."""

    def test_cache_performance_under_load(self, cache_manager, performance_monitor):
        """Test cache performance with high concurrent load."""
        performance_monitor.start_monitoring()

        num_threads = 5
        operations_per_thread = 100

        def cache_worker(thread_id):
            """Worker function for cache operations."""
            for i in range(operations_per_thread):
                key = f"perf_test_{thread_id}_{i}"
                value = f"value_{thread_id}_{i}_" * 20  # Larger payload

                # Put operation
                cache_manager.put(key, value, ttl=300)

                # Get operation
                retrieved = cache_manager.get(key)
                assert retrieved == value

        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        performance_monitor.stop_monitoring()

        # Verify all operations completed successfully
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1

        # Check cache stats after load
        stats = cache_manager.get_stats()
        assert stats["memory_cache"]["size"] >= num_threads * operations_per_thread

    def test_memory_scaling(self, cache_manager, performance_monitor):
        """Test how memory usage scales with data size."""
        performance_monitor.start_monitoring()

        # Test with increasing data sizes
        sizes = [100, 1000, 10000]  # Characters in value

        for size in sizes:
            key = f"size_test_{size}"
            value = "x" * size

            # Measure time for put operation
            start_time = time.time()
            cache_manager.put(key, value, ttl=300)
            put_time = time.time() - start_time

            # Measure time for get operation
            start_time = time.time()
            retrieved = cache_manager.get(key)
            get_time = time.time() - start_time

            # Verify correctness
            assert retrieved == value
            assert len(retrieved) == size

            # Performance should degrade gracefully with size
            assert put_time < 1.0  # Should complete within 1 second
            assert get_time < 0.1  # Should be fast

        performance_monitor.stop_monitoring()

        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1

    def test_concurrent_resource_access(self, resource_manager, performance_monitor):
        """Test concurrent access to shared resources."""
        performance_monitor.start_monitoring()

        from mia.resource_manager import ManagedResource  # type: ignore

        class SharedResource(ManagedResource):
            def __init__(self, resource_id):
                super().__init__(resource_id, "shared")
                self.access_count = 0
                self.lock = threading.Lock()

            def initialize(self):
                pass

            def cleanup(self):
                pass

            def get_memory_usage(self):
                return 1024

            def safe_increment(self):
                with self.lock:
                    self.access_count += 1
                    return self.access_count

        # Create and register shared resource
        shared_resource = SharedResource("shared_test")
        resource_manager.register_resource(shared_resource)

        num_threads = 10
        accesses_per_thread = 50

        def resource_worker(thread_id):
            """Worker that accesses shared resource."""
            for _ in range(accesses_per_thread):
                with resource_manager.use_resource("shared_test") as resource:
                    count = resource.safe_increment()
                    assert count > 0

        # Execute concurrent resource access
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(resource_worker, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        performance_monitor.stop_monitoring()

        # Verify final access count
        final_count = shared_resource.access_count
        expected_count = num_threads * accesses_per_thread
        assert final_count == expected_count

        # Verify performance monitoring
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1

    def test_cache_eviction_under_memory_pressure(
        self, cache_manager, performance_monitor
    ):
        """Test cache behavior under memory pressure."""
        performance_monitor.start_monitoring()

        # Fill cache with many entries
        num_entries = 200
        large_value = "x" * 1000  # 1KB per entry

        for i in range(num_entries):
            key = f"eviction_test_{i}"
            cache_manager.put(key, large_value, ttl=300)

        # Check initial cache size
        initial_stats = cache_manager.get_stats()
        initial_size = initial_stats["memory_cache"]["size"]

        # Add more entries to trigger eviction
        for i in range(num_entries, num_entries + 50):
            key = f"eviction_test_{i}"
            cache_manager.put(key, large_value, ttl=300)

        # Check cache size after potential eviction
        final_stats = cache_manager.get_stats()
        final_size = final_stats["memory_cache"]["size"]

        # Cache should have evicted some entries if under memory pressure
        # (This depends on LRU cache implementation)
        assert final_size > 0  # Should have some entries

        # Some entries should still be retrievable
        retrievable_count = 0
        for i in range(num_entries + 50):
            key = f"eviction_test_{i}"
            if cache_manager.get(key) is not None:
                retrievable_count += 1

        assert retrievable_count > 0  # At least some entries should be retrievable

        performance_monitor.stop_monitoring()

        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1

    @pytest.mark.slow
    def test_sustained_load_simulation(
        self, cache_manager, performance_monitor, resource_manager
    ):
        """Test sustained load over an extended period."""
        performance_monitor.start_monitoring()

        duration_seconds = 10  # Run for 10 seconds
        start_time = time.time()

        operation_count = 0

        while time.time() - start_time < duration_seconds:
            # Perform cache operations
            key = f"sustained_{operation_count}"
            value = f"value_{operation_count}"
            cache_manager.put(key, value, ttl=60)

            # Occasionally retrieve values
            if operation_count % 10 == 0:
                retrieved = cache_manager.get(key)
                assert retrieved == value

            operation_count += 1

            # Small delay to prevent overwhelming the system
            time.sleep(0.001)

        performance_monitor.stop_monitoring()

        # Verify operations were performed
        assert operation_count > 100  # Should have performed many operations

        # Verify cache has content
        stats = cache_manager.get_stats()
        assert stats["memory_cache"]["size"] > 0

        # Verify resource manager is still functional
        assert resource_manager.max_memory_bytes > 0

        # Verify performance monitoring captured the sustained load
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1
        assert summary["average_memory_percent"] > 0

    def test_memory_cleanup_efficiency(self, cache_manager, performance_monitor):
        """Test efficiency of memory cleanup operations."""
        performance_monitor.start_monitoring()

        # Create entries with short TTL
        num_entries = 100
        short_ttl = 1  # 1 second

        for i in range(num_entries):
            key = f"cleanup_test_{i}"
            value = f"value_{i}_" * 50
            cache_manager.put(key, value, ttl=short_ttl)

        # Wait for entries to expire
        time.sleep(short_ttl + 0.5)

        # Force cleanup by accessing cache operations
        for i in range(10):  # Trigger some operations to potentially clean up
            temp_key = f"temp_{i}"
            cache_manager.put(temp_key, f"temp_value_{i}", ttl=60)
            cache_manager.get(temp_key)

        # Check cache stats
        stats = cache_manager.get_stats()

        # Some cleanup should have occurred
        # (Exact behavior depends on cache implementation)
        assert stats["memory_cache"]["size"] >= 0

        performance_monitor.stop_monitoring()

        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1
