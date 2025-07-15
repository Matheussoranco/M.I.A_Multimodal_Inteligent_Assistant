"""
Performance Monitor - System performance monitoring and optimization
"""
import psutil
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from threading import Thread, Lock
import gc
import tracemalloc

from .exceptions import PerformanceError
from .config_manager import ConfigManager

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
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.metrics_history: List[PerformanceMetrics] = []
        self.lock = Lock()
        self.monitoring_active = False
        self.monitoring_thread: Optional[Thread] = None
        self.collection_interval = 5.0  # seconds
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
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
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
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                
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
        disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        
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
            import GPUtil
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
            gpu_memory_used=gpu_memory_used
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
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > self.disk_threshold:
            logger.warning(f"High disk usage: {disk_usage.percent:.1f}%")
            self._trigger_disk_optimization()
            
    def _trigger_cpu_optimization(self):
        """Trigger CPU optimization measures."""
        # Force garbage collection
        gc.collect()
        
        # Log CPU-intensive processes
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                if proc.info['cpu_percent'] > 10:
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
            logger.info(f"Memory usage: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
            
        # Log memory-intensive processes
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                if proc.info['memory_percent'] > 5:
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        if processes:
            logger.info(f"High memory processes: {processes}")
            
    def _trigger_disk_optimization(self):
        """Trigger disk optimization measures."""
        # Log disk usage by directory
        import os
        for root, dirs, files in os.walk('.'):
            size = sum(os.path.getsize(os.path.join(root, file)) for file in files)
            if size > 100 * 1024 * 1024:  # > 100MB
                logger.info(f"Large directory: {root} ({size/1024/1024:.1f}MB)")
                
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
            
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            if not self.metrics_history:
                return {}
                
            recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
            
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_threads = sum(m.active_threads for m in recent_metrics) / len(recent_metrics)
            
            return {
                'average_cpu_percent': avg_cpu,
                'average_memory_percent': avg_memory,
                'average_active_threads': avg_threads,
                'current_open_files': recent_metrics[-1].open_files,
                'memory_used_mb': recent_metrics[-1].memory_used_mb,
                'memory_available_mb': recent_metrics[-1].memory_available_mb,
                'gpu_usage': recent_metrics[-1].gpu_usage,
                'gpu_memory_used': recent_metrics[-1].gpu_memory_used,
                'metrics_count': len(self.metrics_history)
            }
            
    def optimize_performance(self):
        """Perform general performance optimization."""
        logger.info("Starting performance optimization...")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear metrics history if too large
        with self.lock:
            if len(self.metrics_history) > self.max_history_size // 2:
                self.metrics_history = self.metrics_history[-self.max_history_size // 2:]
                logger.info("Cleared old performance metrics")
                
        # Log current memory usage
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"Memory after optimization: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
            
        logger.info("Performance optimization completed")
        
    def get_memory_top_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory consumers."""
        if not tracemalloc.is_tracing():
            return []
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        consumers = []
        for stat in top_stats[:limit]:
            consumers.append({
                'filename': stat.traceback.format()[0],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
            
        return consumers
        
    def cleanup(self):
        """Cleanup performance monitor."""
        self.stop_monitoring()
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            
        logger.info("Performance monitor cleanup completed")
