"""
Enhanced Caching System - Intelligent caching for improved performance
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    timestamp: float
    ttl: float
    hit_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_size_bytes": 0}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None

            # Update access info
            entry.hit_count += 1
            entry.last_access = time.time()
            self.stats["hits"] += 1

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            ttl = ttl or self.default_ttl

            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1000  # Estimate

            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
            )

            # Add to cache
            self.cache[key] = entry
            self.access_order.append(key)
            self.stats["total_size_bytes"] += size_bytes

            # Evict if necessary
            while len(self.cache) > self.max_size:
                self._evict_lru()

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.stats["total_size_bytes"] -= entry.size_bytes
            del self.cache[key]

        if key in self.access_order:
            self.access_order.remove(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_order:
            return

        lru_key = self.access_order[0]
        self._remove_entry(lru_key)
        self.stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats["total_size_bytes"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_size_bytes": self.stats["total_size_bytes"],
                "avg_size_bytes": (
                    self.stats["total_size_bytes"] / len(self.cache)
                    if self.cache
                    else 0
                ),
            }


class PersistentCache:
    """Persistent cache using disk storage."""

    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.index_file = self.cache_dir / "index.json"
        self.lock = Lock()
        self.index: Dict[str, Dict[str, Any]] = {}

        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, "r") as f:
                    self.index = json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache index: {e}")
            self.index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        with self.lock:
            if key not in self.index:
                return None

            entry_info = self.index[key]

            # Check TTL
            if time.time() - entry_info["timestamp"] > entry_info["ttl"]:
                self._remove_entry(key)
                return None

            # Load from disk
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, "rb") as f:
                    value = pickle.load(f)

                # Update access info
                entry_info["hit_count"] += 1
                entry_info["last_access"] = time.time()
                self._save_index()

                return value
            except Exception as e:
                logger.error(f"Error loading cache entry {key}: {e}")
                self._remove_entry(key)
                return None

    def put(self, key: str, value: Any, ttl: float = 3600) -> None:
        """Put value in persistent cache."""
        with self.lock:
            cache_path = self._get_cache_path(key)

            try:
                # Save to disk
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)

                # Update index
                file_size = cache_path.stat().st_size
                self.index[key] = {
                    "timestamp": time.time(),
                    "ttl": ttl,
                    "size_bytes": file_size,
                    "hit_count": 0,
                    "last_access": time.time(),
                }

                self._save_index()

                # Clean up if cache is too large
                self._cleanup_if_needed()

            except Exception as e:
                logger.error(f"Error saving cache entry {key}: {e}")

    def _remove_entry(self, key: str) -> None:
        """Remove entry from persistent cache."""
        if key in self.index:
            del self.index[key]

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limit."""
        total_size = sum(entry["size_bytes"] for entry in self.index.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024

        if total_size > max_size_bytes:
            # Remove oldest entries
            sorted_entries = sorted(
                self.index.items(), key=lambda x: x[1]["last_access"]
            )

            for key, entry in sorted_entries:
                self._remove_entry(key)
                total_size -= entry["size_bytes"]

                if total_size <= max_size_bytes * 0.8:  # Clean to 80% capacity
                    break

            self._save_index()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()

            # Clear index
            self.index.clear()
            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry["size_bytes"] for entry in self.index.values())
            total_hits = sum(entry["hit_count"] for entry in self.index.values())

            return {
                "entries": len(self.index),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / 1024 / 1024,
                "total_hits": total_hits,
                "cache_dir": str(self.cache_dir),
            }


class CacheManager:
    """Unified cache management system."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.memory_cache = LRUCache(max_size=1000, default_ttl=3600)
        self.persistent_cache = PersistentCache(cache_dir="cache", max_size_mb=100)

    def get(self, key: str, use_persistent: bool = True) -> Optional[Any]:
        """Get value from cache (memory first, then persistent)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try persistent cache
        if use_persistent:
            value = self.persistent_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                self.memory_cache.put(key, value)
                return value

        return None

    def put(
        self, key: str, value: Any, ttl: float = 3600, use_persistent: bool = True
    ) -> None:
        """Put value in cache."""
        # Always store in memory cache
        self.memory_cache.put(key, value, ttl)

        # Store in persistent cache if enabled
        if use_persistent:
            self.persistent_cache.put(key, value, ttl)

    def cached(
        self, key: Optional[str] = None, ttl: float = 3600, use_persistent: bool = True
    ):
        """Decorator for caching function results."""

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                # Generate cache key
                cache_key = key or f"{func.__name__}:{hash(str(args) + str(kwargs))}"

                # Try to get from cache
                result = self.get(cache_key, use_persistent)
                if result is not None:
                    return result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl, use_persistent)

                return result

            return wrapper

        return decorator

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        self.memory_cache._remove_entry(key)
        self.persistent_cache._remove_entry(key)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.persistent_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "memory_cache": self.memory_cache.get_stats(),
            "persistent_cache": self.persistent_cache.get_stats(),
        }

    def optimize(self) -> None:
        """Optimize cache performance."""
        # Clear expired entries
        current_time = time.time()

        # Memory cache cleanup
        with self.memory_cache.lock:
            expired_keys = []
            for key, entry in self.memory_cache.cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                self.memory_cache._remove_entry(key)

        # Persistent cache cleanup
        self.persistent_cache._cleanup_if_needed()

        logger.info("Cache optimization completed")


# Global cache instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
