"""
Performance Monitor - System performance monitoring and optimization
"""

import gc
import logging
import time
import tracemalloc
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

import psutil

from .config_manager import ConfigManager

# Import resource manager for integration
from .resource_manager import resource_manager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    active_threads: int
    open_files: int
    gpu_usage: Optional[float] = None
    gpu_memory_used: Optional[float] = None


class PerformanceMonitor:
    """Performance monitoring and optimization system."""

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        collection_interval: float = 5.0,
    ):
        self.config_manager = config_manager or ConfigManager()
        self.metrics_history: List[PerformanceMetrics] = []
        self.lock = Lock()
        self.monitoring_active = False
        self.monitoring_thread: Optional[Thread] = None
        self.collection_interval = collection_interval  # seconds
        self.max_history_size = 1000

        # Performance thresholds
        self.cpu_threshold = 80.0  # percent
        self.memory_threshold = 85.0  # percent
        self.disk_threshold = 90.0  # percent

        # Initialize tracemalloc for memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        # Collect initial metrics immediately
        initial_metrics = self._collect_metrics()
        with self.lock:
            self.metrics_history.append(initial_metrics)

        self.monitoring_thread = Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()

                with self.lock:
                    self.metrics_history.append(metrics)

                    # Limit history size
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[
                            -self.max_history_size :
                        ]

                # Check for performance issues
                self._check_performance_thresholds(metrics)

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.collection_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_io_write_mb = (
            disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        )

        # Network
        network_io = psutil.net_io_counters()
        network_bytes_sent = network_io.bytes_sent if network_io else 0
        network_bytes_recv = network_io.bytes_recv if network_io else 0

        # Process info
        process = psutil.Process()
        active_threads = process.num_threads()
        open_files = len(process.open_files())

        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory_used = None
        try:
            import GPUtil  # type: ignore

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory_used = gpu.memoryUsed
        except ImportError:
            pass

        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_threads=active_threads,
            open_files=open_files,
            gpu_usage=gpu_usage,
            gpu_memory_used=gpu_memory_used,
        )

    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger optimizations."""
        # CPU threshold
        if metrics.cpu_percent > self.cpu_threshold:
            logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            self._trigger_cpu_optimization()

        # Memory threshold
        if metrics.memory_percent > self.memory_threshold:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
            self._trigger_memory_optimization()

        # Disk threshold
        disk_usage = psutil.disk_usage("/")
        if disk_usage.percent > self.disk_threshold:
            logger.warning(f"High disk usage: {disk_usage.percent:.1f}%")
            self._trigger_disk_optimization()

    def _trigger_cpu_optimization(self):
        """Trigger CPU optimization measures."""
        # Force garbage collection
        gc.collect()

        # Log CPU-intensive processes
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                if proc.info["cpu_percent"] > 10:
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if processes:
            logger.info(f"High CPU processes: {processes}")

    def _trigger_memory_optimization(self):
        """Trigger memory optimization measures."""
        # Force garbage collection
        gc.collect()

        # Log memory usage
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(
                f"Memory usage: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB"
            )

        # Integrate with resource manager for model cleanup
        self._optimize_resource_usage()

        # Log memory-intensive processes
        processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
            try:
                if proc.info["memory_percent"] > 5:
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if processes:
            logger.info(f"High memory processes: {processes}")

    def _trigger_disk_optimization(self):
        """Trigger disk optimization measures."""
        # Log disk usage by directory
        import os

        for root, dirs, files in os.walk("."):
            size = sum(
                os.path.getsize(os.path.join(root, file))
                for file in files
                if os.path.exists(os.path.join(root, file))
            )
            if size > 100 * 1024 * 1024:  # > 100MB
                logger.info(
                    f"Large directory: {root} ({size/1024/1024:.1f}MB)"
                )

    def _optimize_resource_usage(self):
        """Optimize resource usage by cleaning up unused resources."""
        try:
            # Get resource manager stats
            resource_stats = resource_manager.get_stats()
            total_memory_usage = resource_stats.get("total_memory_usage", 0)
            max_memory_limit = resource_stats.get(
                "max_memory_limit", 1024 * 1024 * 1024
            )  # 1GB default

            memory_usage_percent = (
                total_memory_usage / max_memory_limit
            ) * 100

            if (
                memory_usage_percent > 70
            ):  # If using more than 70% of memory limit
                logger.info(
                    f"High resource memory usage: {memory_usage_percent:.1f}%, triggering cleanup"
                )

                # Trigger resource manager cleanup
                resource_manager._cleanup_idle_resources()

                # Log cleanup results
                after_stats = resource_manager.get_stats()
                after_memory_usage = after_stats.get("total_memory_usage", 0)
                after_percent = (after_memory_usage / max_memory_limit) * 100

                freed_memory = total_memory_usage - after_memory_usage
                logger.info(
                    f"Resource cleanup freed {freed_memory/1024/1024:.1f}MB, "
                    f"memory usage now: {after_percent:.1f}%"
                )

        except Exception as e:
            logger.error(f"Error during resource optimization: {e}")

    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get resource manager metrics integrated with performance metrics."""
        try:
            resource_stats = resource_manager.get_stats()
            resource_info = resource_manager.get_resource_info()

            # Group resources by type
            resource_counts = {}
            memory_by_type = {}

            for info in resource_info:
                resource_counts[info.type] = (
                    resource_counts.get(info.type, 0) + 1
                )
                memory_by_type[info.type] = (
                    memory_by_type.get(info.type, 0) + info.memory_usage
                )

            return {
                "resource_counts": resource_counts,
                "memory_by_type": memory_by_type,
                "total_resource_memory": resource_stats.get(
                    "total_memory_usage", 0
                ),
                "resource_memory_percent": resource_stats.get(
                    "memory_usage_percent", 0
                ),
                "active_resources": len(resource_info),
            }
        except Exception as e:
            logger.error(f"Error getting resource metrics: {e}")
            return {}

    def optimize_performance(self):
        """Perform general performance optimization."""
        logger.info("Starting performance optimization...")

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")

        # Optimize resource usage
        self._optimize_resource_usage()

        # Clear metrics history if too large
        with self.lock:
            if len(self.metrics_history) > self.max_history_size // 2:
                self.metrics_history = self.metrics_history[
                    -self.max_history_size // 2 :
                ]
                logger.info("Cleared old performance metrics")

        # Log current memory usage
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(
                f"Memory after optimization: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB"
            )

        logger.info("Performance optimization completed")

    def get_memory_top_consumers(
        self, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top memory consumers."""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        consumers = []
        for stat in top_stats[:limit]:
            consumers.append(
                {
                    "filename": stat.traceback.format()[0],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
            )

        return consumers

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Get the most recent performance metrics.

        Returns:
            PerformanceMetrics: The latest metrics, or None if no metrics available
        """
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            else:
                # Return current metrics if monitoring is active
                if self.monitoring_active:
                    return self._collect_metrics()
                return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics over the monitoring period.

        Returns:
            Dict containing performance summary statistics
        """
        with self.lock:
            if not self.metrics_history:
                return {
                    "status": "no_data",
                    "message": "No performance metrics available",
                }

            # Calculate summary statistics
            metrics_count = len(self.metrics_history)
            if metrics_count == 0:
                return {
                    "status": "no_data",
                    "message": "No performance metrics available",
                }

            # Extract metric values
            cpu_values = [
                m.cpu_percent
                for m in self.metrics_history
                if m.cpu_percent is not None
            ]
            memory_values = [
                m.memory_percent
                for m in self.metrics_history
                if m.memory_percent is not None
            ]
            memory_used_values = [
                m.memory_used_mb
                for m in self.metrics_history
                if m.memory_used_mb is not None
            ]

            summary = {
                "status": "active" if self.monitoring_active else "stopped",
                "total_samples": metrics_count,
                "time_range": {
                    "start": self.metrics_history[0].timestamp,
                    "end": self.metrics_history[-1].timestamp,
                    "duration_seconds": self.metrics_history[-1].timestamp
                    - self.metrics_history[0].timestamp,
                },
            }

            # CPU statistics
            if cpu_values:
                summary["cpu"] = {
                    "current": cpu_values[-1],
                    "average": sum(cpu_values) / len(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values),
                }
                # Add flat keys for backward compatibility
                summary["average_cpu_percent"] = summary["cpu"]["average"]
                summary["current_cpu_percent"] = summary["cpu"]["current"]
                summary["max_cpu_percent"] = summary["cpu"]["max"]
                summary["min_cpu_percent"] = summary["cpu"]["min"]

            # Memory statistics
            memory_section: Dict[str, Any] = {}
            memory_section_has_data = False

            if memory_values:
                memory_section_has_data = True
                memory_section.update(
                    {
                        "current_percent": memory_values[-1],
                        "average_percent": sum(memory_values)
                        / len(memory_values),
                        "max_percent": max(memory_values),
                        "min_percent": min(memory_values),
                    }
                )
                # Add flat keys for backward compatibility
                summary["average_memory_percent"] = memory_section[
                    "average_percent"
                ]
                summary["current_memory_percent"] = memory_section[
                    "current_percent"
                ]
                summary["max_memory_percent"] = memory_section["max_percent"]
                summary["min_memory_percent"] = memory_section["min_percent"]

            if memory_used_values:
                memory_section_has_data = True
                memory_section.update(
                    {
                        "current_mb": memory_used_values[-1],
                        "average_mb": sum(memory_used_values)
                        / len(memory_used_values),
                        "max_mb": max(memory_used_values),
                        "min_mb": min(memory_used_values),
                    }
                )
                # Add flat keys for backward compatibility
                summary["memory_used_mb"] = memory_section["current_mb"]
                summary["average_memory_mb"] = memory_section["average_mb"]
                summary["max_memory_mb"] = memory_section["max_mb"]
                summary["min_memory_mb"] = memory_section["min_mb"]

            if memory_section_has_data:
                summary["memory"] = memory_section

            # Add metrics_count for test compatibility
            summary["metrics_count"] = metrics_count

            # Performance issues
            issues = []
            if cpu_values and max(cpu_values) > 90:
                issues.append("High CPU usage detected")
            if memory_values and max(memory_values) > 90:
                issues.append("High memory usage detected")

            summary["issues"] = issues
            summary["issues_count"] = len(issues)

            return summary

    def cleanup(self):
        """Cleanup performance monitor."""
        self.stop_monitoring()

        if tracemalloc.is_tracing():
            tracemalloc.stop()

        logger.info("Performance monitor cleanup completed")
