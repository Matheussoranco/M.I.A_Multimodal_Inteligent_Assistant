"""
Audio Resource Manager - Specialized resource management for audio components
"""
import logging
from typing import Dict, Any
import threading
import time

from ..resource_manager import ManagedResource, ResourceManager, ResourceState

logger = logging.getLogger(__name__)

class AudioResource(ManagedResource):
    """Managed resource for audio components."""
    
    def __init__(self, name: str, audio_component: Any, config_manager=None):
        super().__init__(name, "audio")
        self.data = audio_component
        self.config_manager = config_manager
        self.is_recording = False
        self.is_playing = False
        self.last_activity = time.time()
        
    def initialize(self) -> None:
        """Initialize the audio resource."""
        self.state = ResourceState.INITIALIZED
        logger.info(f"Audio resource {self.resource_id} initialized")
        
    def cleanup(self):
        """Clean up audio resources."""
        try:
            if hasattr(self.data, 'stop_recording'):
                self.data.stop_recording()
            if hasattr(self.data, 'stop_playback'):
                self.data.stop_playback()
            if hasattr(self.data, 'close'):
                self.data.close()
            logger.info(f"Audio resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up audio resource {self.resource_id}: {e}")
    
    def get_memory_usage(self) -> int:
        """Get memory usage of audio resource."""
        try:
            # Estimate memory usage based on audio buffers
            memory_usage = 0
            if hasattr(self.data, 'buffer_size'):
                memory_usage += getattr(self.data, 'buffer_size', 0)
            if hasattr(self.data, 'model_memory'):
                memory_usage += getattr(self.data, 'model_memory', 0)
            return memory_usage
        except Exception:
            return 0
    
    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.last_activity = time.time()
        
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        self.last_activity = time.time()
        
    def start_playback(self):
        """Start audio playback."""
        self.is_playing = True
        self.last_activity = time.time()
        
    def stop_playback(self):
        """Stop audio playback."""
        self.is_playing = False
        self.last_activity = time.time()

class AudioResourceManager(ResourceManager):
    """Specialized resource manager for audio components."""
    
    def __init__(self, max_memory_mb: int = 500):
        super().__init__(max_memory_mb)
        self.max_memory_mb = max_memory_mb
        self.audio_resources: Dict[str, AudioResource] = {}
        self.recording_lock = threading.Lock()
        
    def acquire_audio_resource(self, name: str, audio_component: Any, config_manager=None) -> AudioResource:
        """Acquire an audio resource."""
        with self._lock:
            if name in self.audio_resources:
                return self.audio_resources[name]
            
            resource = AudioResource(name, audio_component, config_manager)
            self.audio_resources[name] = resource
            self.resources[name] = resource
            
            logger.info(f"Audio resource {name} acquired")
            return resource
    
    def release_audio_resource(self, name: str):
        """Release an audio resource."""
        with self._lock:
            if name in self.audio_resources:
                resource = self.audio_resources[name]
                resource.cleanup()
                del self.audio_resources[name]
                if name in self.resources:
                    del self.resources[name]
                logger.info(f"Audio resource {name} released")
    
    def is_recording_active(self) -> bool:
        """Check if any audio resource is currently recording."""
        with self._lock:
            return any(resource.is_recording for resource in self.audio_resources.values())
    
    def is_playback_active(self) -> bool:
        """Check if any audio resource is currently playing."""
        with self._lock:
            return any(resource.is_playing for resource in self.audio_resources.values())
    
    def stop_all_recording(self):
        """Stop all recording activities."""
        with self.recording_lock:
            with self._lock:
                for resource in self.audio_resources.values():
                    if resource.is_recording:
                        resource.stop_recording()
                logger.info("All recording stopped")
    
    def stop_all_playback(self):
        """Stop all playback activities."""
        with self._lock:
            for resource in self.audio_resources.values():
                if resource.is_playing:
                    resource.stop_playback()
            logger.info("All playback stopped")
    
    def get_audio_status(self) -> Dict[str, Any]:
        """Get status of all audio resources."""
        with self._lock:
            status = {
                'total_resources': len(self.audio_resources),
                'recording_active': self.is_recording_active(),
                'playback_active': self.is_playback_active(),
                'resources': {}
            }
            
            for name, resource in self.audio_resources.items():
                status['resources'][name] = {
                    'is_recording': resource.is_recording,
                    'is_playing': resource.is_playing,
                    'last_activity': resource.last_activity,
                    'memory_usage': resource.get_memory_usage()
                }
            
            return status
    
    def cleanup_idle_resources(self):
        """Clean up idle audio resources."""
        current_time = time.time()
        idle_timeout = 300  # 5 minutes
        
        with self._lock:
            idle_resources = []
            for name, resource in self.audio_resources.items():
                if (not resource.is_recording and 
                    not resource.is_playing and 
                    current_time - resource.last_activity > idle_timeout):
                    idle_resources.append(name)
            
            for name in idle_resources:
                logger.info(f"Cleaning up idle audio resource: {name}")
                self.release_audio_resource(name)
    
    def _audio_cleanup_thread(self):
        """Enhanced cleanup thread for audio resources."""
        while self._running:
            try:
                # Run parent cleanup
                self._cleanup_idle_resources()
                self._check_memory_usage()
                
                # Run audio-specific cleanup
                self.cleanup_idle_resources()
                
                # Check memory usage
                total_memory = sum(resource.get_memory_usage() for resource in self.audio_resources.values())
                if total_memory > self.max_memory_bytes:
                    logger.warning(f"Audio memory usage ({total_memory / 1024 / 1024:.1f}MB) exceeds limit")
                    self.cleanup_idle_resources()
                
            except Exception as e:
                logger.error(f"Error in audio cleanup thread: {e}")
            
            # Wait for cleanup interval
            time.sleep(self._cleanup_interval)
