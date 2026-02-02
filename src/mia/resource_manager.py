"""
Resource Management System for M.I.A
Provides resource lifecycle management, cleanup, and monitoring.
"""

import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .error_handler import global_error_handler, with_error_handling
from .exceptions import ConfigurationError, MemoryError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceState(Enum):
    """Resource lifecycle states."""

    CREATED = "created"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLEANING_UP = "cleaning_up"
    DESTROYED = "destroyed"


@dataclass
class ResourceInfo:
    """Information about a managed resource."""

    id: str
    type: str
    state: ResourceState
    created_at: float
    last_used: float
    memory_usage: int = 0
    reference_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ManagedResource(ABC, Generic[T]):
    """Base class for managed resources."""

    def __init__(self, resource_id: str, resource_type: str):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.state = ResourceState.CREATED
        self.created_at = time.time()
        self.last_used = time.time()
        self._lock = threading.RLock()
        self._cleanup_callbacks: List[Callable] = []

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the resource."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up the resource."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        pass

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a cleanup callback."""
        self._cleanup_callbacks.append(callback)

    def update_last_used(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()

    def is_active(self) -> bool:
        """Check if resource is active."""
        return self.state == ResourceState.ACTIVE

    def __enter__(self):
        """Context manager entry."""
        with self._lock:
            if self.state == ResourceState.CREATED:
                self.initialize()
                self.state = ResourceState.INITIALIZED
            if self.state == ResourceState.INITIALIZED:
                self.state = ResourceState.ACTIVE
            self.update_last_used()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Resources are managed by ResourceManager, not auto-cleaned here
        pass


class AudioResource(ManagedResource[Any]):
    """Managed audio resource."""

    def __init__(self, resource_id: str, audio_device=None):
        super().__init__(resource_id, "audio")
        self.audio_device = audio_device
        self._stream = None

    def initialize(self) -> None:
        """Initialize audio resource."""
        try:
            # Audio device initialization would go here
            logger.info(f"Audio resource {self.resource_id} initialized")
        except Exception as e:
            logger.error(
                f"Failed to initialize audio resource {self.resource_id}: {e}"
            )
            raise

    def cleanup(self) -> None:
        """Clean up audio resource."""
        try:
            if self._stream:
                self._stream.close()
                self._stream = None

            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")

            logger.info(f"Audio resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(
                f"Failed to cleanup audio resource {self.resource_id}: {e}"
            )

    def get_memory_usage(self) -> int:
        """Get memory usage estimate."""
        return 1024 * 1024  # 1MB estimate for audio buffers


class ModelResource(ManagedResource[Any]):
    """Managed model resource."""

    def __init__(self, resource_id: str, model_path: str, model_type: str):
        super().__init__(resource_id, "model")
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self._estimated_size = 0

    def initialize(self) -> None:
        """Initialize model resource."""
        try:
            # Model loading would go here
            logger.info(
                f"Model resource {self.resource_id} ({self.model_type}) initialized"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize model resource {self.resource_id}: {e}"
            )
            raise

    def cleanup(self) -> None:
        """Clean up model resource."""
        try:
            if self.model:
                # Model cleanup would go here
                self.model = None

            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")

            logger.info(f"Model resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(
                f"Failed to cleanup model resource {self.resource_id}: {e}"
            )

    def get_memory_usage(self) -> int:
        """Get memory usage estimate."""
        return (
            self._estimated_size or 100 * 1024 * 1024
        )  # 100MB default estimate


class DatabaseResource(ManagedResource[Any]):
    """Managed database resource."""

    def __init__(self, resource_id: str, db_path: str):
        super().__init__(resource_id, "database")
        self.db_path = db_path
        self.connection = None

    def initialize(self) -> None:
        """Initialize database resource."""
        try:
            # Database connection would go here
            logger.info(f"Database resource {self.resource_id} initialized")
        except Exception as e:
            logger.error(
                f"Failed to initialize database resource {self.resource_id}: {e}"
            )
            raise

    def cleanup(self) -> None:
        """Clean up database resource."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None

            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")

            logger.info(f"Database resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(
                f"Failed to cleanup database resource {self.resource_id}: {e}"
            )

    def get_memory_usage(self) -> int:
        """Get memory usage estimate."""
        return 10 * 1024 * 1024  # 10MB estimate for database connections


class WhisperModelResource(ManagedResource[Any]):
    """Managed Whisper model resource for speech processing."""

    # Memory usage estimates for different Whisper model sizes (in MB)
    MODEL_MEMORY_ESTIMATES = {
        "tiny": 100,
        "base": 200,
        "small": 500,
        "medium": 1500,
        "large": 3000,
        "large-v2": 3000,
        "large-v3": 3000,
    }

    def __init__(self, model_name: str):
        resource_id = f"whisper_{model_name}"
        super().__init__(resource_id, "whisper_model")
        self.model_name = model_name
        self.model = None
        self._estimated_size = (
            self.MODEL_MEMORY_ESTIMATES.get(model_name, 500) * 1024 * 1024
        )  # Convert to bytes

    def initialize(self) -> None:
        """Initialize Whisper model resource."""
        try:
            # Import whisper here to avoid circular imports
            try:
                import whisper

                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name)
                logger.info(
                    f"Whisper model {self.model_name} loaded successfully"
                )
            except ImportError:
                raise MemoryError(
                    "Whisper package not available", "MISSING_WHISPER"
                )
            except Exception as e:
                raise MemoryError(
                    f"Failed to load Whisper model {self.model_name}: {str(e)}",
                    "MODEL_LOAD_FAILED",
                )
        except Exception as e:
            logger.error(
                f"Failed to initialize Whisper model resource {self.resource_id}: {e}"
            )
            raise

    def cleanup(self) -> None:
        """Clean up Whisper model resource."""
        try:
            if self.model:
                # Clear the model from memory
                del self.model
                self.model = None

                # Force garbage collection
                import gc

                gc.collect()

            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")

            logger.info(
                f"Whisper model resource {self.resource_id} cleaned up"
            )
        except Exception as e:
            logger.error(
                f"Failed to cleanup Whisper model resource {self.resource_id}: {e}"
            )

    def get_memory_usage(self) -> int:
        """Get memory usage estimate."""
        return self._estimated_size

    def transcribe(self, audio_data, **kwargs):
        """Transcribe audio using the loaded model."""
        if self.model is None:
            raise RuntimeError(f"Whisper model {self.model_name} not loaded")

        try:
            result = self.model.transcribe(audio_data, **kwargs)
            self.update_last_used()
            return result
        except Exception as e:
            logger.error(
                f"Error transcribing with Whisper model {self.model_name}: {e}"
            )
            raise


class ResourceManager:
    """Central resource manager for M.I.A."""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.resources: Dict[str, ManagedResource] = {}
        self.resource_refs: Dict[str, weakref.ref] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._running = False
        self._cleanup_interval = 60  # 60 seconds
        self._max_idle_time = 300  # 5 minutes
        self._memory_pressure_callbacks: List[Callable] = []

    def start(self) -> None:
        """Start the resource manager."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_thread.start()
            logger.info("Resource manager started")

    def register_memory_pressure_callback(self, callback: Callable) -> None:
        """Register a callback to be called when system memory pressure is high."""
        with self._lock:
            self._memory_pressure_callbacks.append(callback)

    def stop(self) -> None:
        """Stop the resource manager and clean up all resources."""
        with self._lock:
            if not self._running:
                return

            self._running = False

            # Clean up all resources
            for resource_id in list(self.resources.keys()):
                try:
                    self.release_resource(resource_id)
                except Exception as e:
                    logger.error(
                        f"Error releasing resource {resource_id}: {e}"
                    )

            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)

            logger.info("Resource manager stopped")

    def cleanup(self) -> None:
        """Cleanup alias for stop() method for semantic clarity."""
        self.stop()

    @with_error_handling(global_error_handler, fallback_value=None)
    def register_resource(self, resource: ManagedResource) -> str:
        """Register a new resource."""
        with self._lock:
            if resource.resource_id in self.resources:
                raise ConfigurationError(
                    f"Resource {resource.resource_id} already registered",
                    "RESOURCE_ALREADY_REGISTERED",
                )

            self.resources[resource.resource_id] = resource

            # Create weak reference for cleanup detection
            def cleanup_callback(ref):
                self._handle_resource_cleanup(resource.resource_id)

            self.resource_refs[resource.resource_id] = weakref.ref(
                resource, cleanup_callback
            )

            logger.info(
                f"Resource {resource.resource_id} ({resource.resource_type}) registered"
            )
            return resource.resource_id

    @with_error_handling(global_error_handler, fallback_value=None)
    def get_resource(self, resource_id: str) -> Optional[ManagedResource]:
        """Get a registered resource."""
        with self._lock:
            resource = self.resources.get(resource_id)
            if resource:
                resource.update_last_used()
            return resource

    @contextmanager
    def use_resource(self, resource_id: str):
        """Context manager for using a resource."""
        resource = self.get_resource(resource_id)
        if not resource:
            raise ConfigurationError(
                f"Resource {resource_id} not found", "RESOURCE_NOT_FOUND"
            )

        try:
            with resource:
                yield resource
        finally:
            # Resource cleanup is handled by the manager
            pass

    @contextmanager
    def acquire_resource(self, resource_id: str):
        """Acquire resource - alias for use_resource for compatibility."""
        with self.use_resource(resource_id) as resource:
            yield resource

    def get_memory_usage(self) -> int:
        """Get current memory usage - alias for get_total_memory_usage."""
        return self.get_total_memory_usage()

    @with_error_handling(global_error_handler, fallback_value=False)
    def release_resource(self, resource_id: str) -> bool:
        """Release a resource."""
        with self._lock:
            resource = self.resources.get(resource_id)
            if not resource:
                return False

            try:
                resource.state = ResourceState.CLEANING_UP
                resource.cleanup()
                resource.state = ResourceState.DESTROYED

                # Remove from tracking
                del self.resources[resource_id]
                if resource_id in self.resource_refs:
                    del self.resource_refs[resource_id]

                logger.info(f"Resource {resource_id} released")
                return True

            except Exception as e:
                logger.error(f"Error releasing resource {resource_id}: {e}")
                return False

    def _handle_resource_cleanup(self, resource_id: str) -> None:
        """Handle cleanup when resource is garbage collected."""
        with self._lock:
            if resource_id in self.resources:
                logger.warning(
                    f"Resource {resource_id} was garbage collected without proper cleanup"
                )
                try:
                    del self.resources[resource_id]
                except KeyError:
                    pass

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                self._cleanup_idle_resources()
                self._check_memory_usage()

                for _ in range(int(self._cleanup_interval)):
                    if not self._running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(5)  # Wait before retrying

    def _cleanup_idle_resources(self) -> None:
        """Clean up idle resources."""
        current_time = time.time()
        idle_resources = []

        with self._lock:
            for resource_id, resource in self.resources.items():
                if (current_time - resource.last_used) > self._max_idle_time:
                    idle_resources.append(resource_id)

        for resource_id in idle_resources:
            logger.info(f"Cleaning up idle resource: {resource_id}")
            self.release_resource(resource_id)

    def _check_memory_usage(self) -> None:
        """Check and manage memory usage using psutil if available."""
        # Check system memory first if psutil is available
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90.0:  # High memory pressure
                logger.warning(f"System memory critical: {mem.percent}% used")
                self._trigger_memory_pressure_callbacks()
                self._free_memory()
                return
        except ImportError:
            pass

        total_memory = self.get_total_memory_usage()

        if total_memory > self.max_memory_bytes:
            logger.warning(
                f"Managed memory usage ({total_memory / 1024 / 1024:.1f}MB) exceeds limit"
            )
            self._free_memory()

    def _trigger_memory_pressure_callbacks(self) -> None:
        """Trigger registered memory pressure callbacks."""
        with self._lock:
            callbacks = list(self._memory_pressure_callbacks)
        
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Memory pressure callback failed: {e}")

    def _free_memory(self) -> None:
        """Free memory by releasing least recently used resources."""
        with self._lock:
            # Sort resources by last used time
            sorted_resources = sorted(
                self.resources.items(), key=lambda x: x[1].last_used
            )

            # Release oldest resources until memory is under limit
            for resource_id, resource in sorted_resources:
                if (
                    self.get_total_memory_usage()
                    <= self.max_memory_bytes * 0.8
                ):
                    break

                logger.info(
                    f"Freeing memory by releasing resource: {resource_id}"
                )
                self.release_resource(resource_id)

    def get_total_memory_usage(self) -> int:
        """Get total memory usage of all resources."""
        with self._lock:
            total = 0
            for resource in self.resources.values():
                try:
                    total += resource.get_memory_usage()
                except Exception as e:
                    logger.error(
                        f"Error getting memory usage for {resource.resource_id}: {e}"
                    )
            return total

    def get_resource_info(self) -> List[ResourceInfo]:
        """Get information about all resources."""
        with self._lock:
            info_list = []
            for resource in self.resources.values():
                try:
                    info = ResourceInfo(
                        id=resource.resource_id,
                        type=resource.resource_type,
                        state=resource.state,
                        created_at=resource.created_at,
                        last_used=resource.last_used,
                        memory_usage=resource.get_memory_usage(),
                        reference_count=(
                            1
                            if resource.resource_id in self.resource_refs
                            else 0
                        ),
                        metadata={},
                    )
                    info_list.append(info)
                except Exception as e:
                    logger.error(
                        f"Error getting info for resource {resource.resource_id}: {e}"
                    )
            return info_list

    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        with self._lock:
            resource_counts = {}
            for resource in self.resources.values():
                resource_counts[resource.resource_type] = (
                    resource_counts.get(resource.resource_type, 0) + 1
                )

            return {
                "total_resources": len(self.resources),
                "resource_counts": resource_counts,
                "total_memory_usage": self.get_total_memory_usage(),
                "max_memory_limit": self.max_memory_bytes,
                "memory_usage_percent": (
                    self.get_total_memory_usage() / self.max_memory_bytes
                )
                * 100,
                "running": self._running,
            }


# Global resource manager instance
resource_manager = ResourceManager()
