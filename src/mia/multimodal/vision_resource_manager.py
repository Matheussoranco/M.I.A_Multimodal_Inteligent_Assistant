"""
Vision Resource Manager - Specialized resource management for vision components
"""
import logging
from typing import Dict, Any
import threading
import time

from ..resource_manager import ManagedResource, ResourceManager, ResourceState

logger = logging.getLogger(__name__)

class VisionResource(ManagedResource):
    """Managed resource for vision components."""
    
    def __init__(self, name: str, vision_component: Any, config_manager=None):
        super().__init__(name, "vision")
        self.data = vision_component
        self.config_manager = config_manager
        self.is_processing = False
        self.last_activity = time.time()
        
    def initialize(self) -> None:
        """Initialize the vision resource."""
        self.state = ResourceState.INITIALIZED
        logger.info(f"Vision resource {self.resource_id} initialized")
        
    def cleanup(self):
        """Clean up vision resources."""
        try:
            if hasattr(self.data, 'stop_processing'):
                self.data.stop_processing()
            if hasattr(self.data, 'close'):
                self.data.close()
            logger.info(f"Vision resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up vision resource {self.resource_id}: {e}")
    
    def get_memory_usage(self) -> int:
        """Get memory usage of vision resource."""
        try:
            # Estimate memory usage based on vision buffers
            memory_usage = 0
            if hasattr(self.data, 'buffer_size'):
                memory_usage += getattr(self.data, 'buffer_size', 0)
            if hasattr(self.data, 'model_memory'):
                memory_usage += getattr(self.data, 'model_memory', 0)
            return memory_usage
        except Exception:
            return 0
    
    def start_processing(self):
        """Start vision processing."""
        self.is_processing = True
        self.last_activity = time.time()
        
    def stop_processing(self):
        """Stop vision processing."""
        self.is_processing = False
        self.last_activity = time.time()

class VisionResourceManager(ResourceManager):
    """Specialized resource manager for vision components."""
    
    def __init__(self, max_memory_mb: int = 500):
        super().__init__(max_memory_mb)
        self.vision_resources: Dict[str, VisionResource] = {}
        self.processing_lock = threading.Lock()
        
    def acquire_vision_resource(self, name: str, vision_component: Any, config_manager=None) -> VisionResource:
        """Acquire a vision resource."""
        with self._lock:
            if name in self.vision_resources:
                return self.vision_resources[name]
            
            resource = VisionResource(name, vision_component, config_manager)
            self.vision_resources[name] = resource
            self.resources[name] = resource
            
            logger.info(f"Vision resource {name} acquired")
            return resource
    
    def release_vision_resource(self, name: str):
        """Release a vision resource."""
        with self._lock:
            if name in self.vision_resources:
                resource = self.vision_resources[name]
                resource.cleanup()
                del self.vision_resources[name]
                if name in self.resources:
                    del self.resources[name]
                logger.info(f"Vision resource {name} released")
    
    def is_processing_active(self) -> bool:
        """Check if any vision resource is currently processing."""
        with self._lock:
            return any(resource.is_processing for resource in self.vision_resources.values())
    
    def stop_all_processing(self):
        """Stop all processing activities."""
        with self.processing_lock:
            with self._lock:
                for resource in self.vision_resources.values():
                    if resource.is_processing:
                        resource.stop_processing()
                logger.info("All processing stopped")
    
    def get_vision_status(self) -> Dict[str, Any]:
        """Get status of all vision resources."""
        with self._lock:
            status = {
                'total_resources': len(self.vision_resources),
                'processing_active': self.is_processing_active(),
                'resources': {}
            }
            
            for name, resource in self.vision_resources.items():
                status['resources'][name] = {
                    'is_processing': resource.is_processing,
                    'last_activity': resource.last_activity,
                    'memory_usage': resource.get_memory_usage()
                }
            
            return status
    
    def cleanup_idle_resources(self):
        """Clean up idle vision resources."""
        current_time = time.time()
        idle_timeout = 300  # 5 minutes
        
        with self._lock:
            idle_resources = []
            for name, resource in self.vision_resources.items():
                if (not resource.is_processing and 
                    current_time - resource.last_activity > idle_timeout):
                    idle_resources.append(name)
            
            for name in idle_resources:
                logger.info(f"Cleaning up idle vision resource: {name}")
                self.release_vision_resource(name)
    
    def _cleanup_thread(self):
        """Enhanced cleanup thread for vision resources."""
        while self._running:
            try:
                # Run parent cleanup
                self._cleanup_idle_resources()
                self._check_memory_usage()
                
                # Run vision-specific cleanup
                self.cleanup_idle_resources()
                
                # Check memory usage
                total_memory = sum(resource.get_memory_usage() for resource in self.vision_resources.values())
                if total_memory > self.max_memory_bytes:
                    logger.warning(f"Vision memory usage ({total_memory / 1024 / 1024:.1f}MB) exceeds limit")
                    self.cleanup_idle_resources()
                
            except Exception as e:
                logger.error(f"Error in vision cleanup thread: {e}")
            
            # Wait for cleanup interval
            time.sleep(self._cleanup_interval)
