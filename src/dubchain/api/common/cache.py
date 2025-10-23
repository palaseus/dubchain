"""
Caching Infrastructure for DubChain.

This module provides comprehensive caching capabilities with multiple
backends, TTL support, and cache invalidation strategies.
"""

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""

    max_size: int = 10000
    default_ttl: int = 300  # 5 minutes
    cleanup_interval: int = 60  # 1 minute
    enable_compression: bool = True
    enable_serialization: bool = True


class CacheEntry:
    """Cache entry with metadata."""

    def __init__(self, value: Any, ttl: int, created_at: float = None):
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl

    def access(self):
        """Record access to entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """Least Recently Used cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check if expired
        if entry.is_expired():
            del self.cache[key]
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.access()

        return entry.value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in cache."""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]

        # Add new entry
        entry = CacheEntry(value, ttl)
        self.cache[key] = entry

        # Remove oldest if over capacity
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self.cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)


class CacheManager:
    """Main cache manager with multiple backends."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager."""
        self.config = config or CacheConfig()
        self.lru_cache = LRUCache(self.config.max_size)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "cleanups": 0,
        }

        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.lru_cache.get(key)
        if value is not None:
            self.stats["hits"] += 1
            return value
        else:
            self.stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.config.default_ttl

        # Serialize if enabled
        if self.config.enable_serialization:
            try:
                value = json.dumps(value)
            except (TypeError, ValueError):
                pass  # Keep original value if not serializable

        self.lru_cache.set(key, value, ttl)
        self.stats["sets"] += 1

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        success = self.lru_cache.delete(key)
        if success:
            self.stats["deletes"] += 1
        return success

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.lru_cache.clear()
        self.stats = {k: 0 for k in self.stats}

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.lru_cache.get(key) is not None

    async def get_or_set(
        self, key: str, factory_func, ttl: Optional[int] = None
    ) -> Any:
        """Get value or set using factory function."""
        value = await self.get(key)
        if value is not None:
            return value

        # Generate value using factory
        if asyncio.iscoroutinefunction(factory_func):
            value = await factory_func()
        else:
            value = factory_func()

        await self.set(key, value, ttl)
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "size": self.lru_cache.size(),
            "max_size": self.config.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "cleanups": self.stats["cleanups"],
            "config": {
                "default_ttl": self.config.default_ttl,
                "cleanup_interval": self.config.cleanup_interval,
                "enable_compression": self.config.enable_compression,
                "enable_serialization": self.config.enable_serialization,
            },
        }

    def _start_cleanup_task(self):
        """Start cleanup task for expired entries."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, skip cleanup task for now
            self._cleanup_task = None
            return

        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(self.config.cleanup_interval)
                    expired_count = self.lru_cache.cleanup_expired()
                    self.stats["cleanups"] += expired_count
                except Exception as e:
                    logger.info(f"Cache cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup())

    def stop(self):
        """Stop cache manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()


# Global cache manager instance
cache_manager = CacheManager()
