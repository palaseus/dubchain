"""
Comprehensive tests for cache core module.

This module tests the core caching interfaces and data structures including:
- Cache exceptions and error handling
- Cache configuration and policies
- Cache entry management
- Cache statistics and monitoring
"""

import logging

logger = logging.getLogger(__name__)
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.cache.core import (
    CacheBackend,
    CacheConfig,
    CacheEntry,
    CacheError,
    CacheExpiredError,
    CacheFullError,
    CacheLevel,
    CacheMissError,
    CachePolicy,
    CacheStats,
    EvictionPolicy,
)


class TestCacheExceptions:
    """Test cache exception classes."""

    def test_cache_error(self):
        """Test base cache error."""
        error = CacheError("Test cache error")
        assert str(error) == "Test cache error"
        assert isinstance(error, Exception)

    def test_cache_miss_error(self):
        """Test cache miss error."""
        error = CacheMissError("Key not found")
        assert str(error) == "Key not found"
        assert isinstance(error, CacheError)

    def test_cache_full_error(self):
        """Test cache full error."""
        error = CacheFullError("Cache is full")
        assert str(error) == "Cache is full"
        assert isinstance(error, CacheError)

    def test_cache_expired_error(self):
        """Test cache expired error."""
        error = CacheExpiredError("Cache entry expired")
        assert str(error) == "Cache entry expired"
        assert isinstance(error, CacheError)


class TestCacheLevel:
    """Test CacheLevel enum."""

    def test_cache_level_values(self):
        """Test cache level values."""
        assert CacheLevel.L1.value == "l1"
        assert CacheLevel.L2.value == "l2"
        assert CacheLevel.L3.value == "l3"
        assert CacheLevel.L4.value == "l4"

    def test_cache_level_comparison(self):
        """Test cache level comparison."""
        assert CacheLevel.L1 != CacheLevel.L2
        assert CacheLevel.L2 != CacheLevel.L3
        assert CacheLevel.L3 != CacheLevel.L4


class TestEvictionPolicy:
    """Test EvictionPolicy enum."""

    def test_eviction_policy_values(self):
        """Test eviction policy values."""
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"
        assert EvictionPolicy.FIFO.value == "fifo"
        assert EvictionPolicy.LIFO.value == "lifo"
        assert EvictionPolicy.TTL.value == "ttl"

    def test_eviction_policy_comparison(self):
        """Test eviction policy comparison."""
        assert EvictionPolicy.LRU != EvictionPolicy.LFU
        assert EvictionPolicy.FIFO != EvictionPolicy.LIFO


class TestCachePolicy:
    """Test CachePolicy functionality."""

    def test_cache_policy_creation(self):
        """Test creating cache policy."""
        policy = CachePolicy(
            eviction_policy=EvictionPolicy.LRU,
            max_size=1000,
            max_memory_bytes=100 * 1024 * 1024,
            default_ttl=3600,
            enable_compression=True,
        )

        assert policy.eviction_policy == EvictionPolicy.LRU
        assert policy.max_size == 1000
        assert policy.max_memory_bytes == 100 * 1024 * 1024
        assert policy.default_ttl == 3600
        assert policy.enable_compression is True

    def test_cache_policy_defaults(self):
        """Test cache policy with default values."""
        policy = CachePolicy()

        assert policy.eviction_policy == EvictionPolicy.LRU
        assert policy.max_size == 1000
        assert policy.max_memory_bytes == 100 * 1024 * 1024
        assert policy.default_ttl is None
        assert policy.enable_compression is False

    def test_cache_policy_validation(self):
        """Test cache policy validation."""
        # Valid policy
        policy = CachePolicy(max_size=1000, max_memory_bytes=100 * 1024 * 1024)
        assert policy.max_size == 1000
        assert policy.max_memory_bytes == 100 * 1024 * 1024


class TestCacheConfig:
    """Test CacheConfig functionality."""

    def test_cache_config_creation(self):
        """Test creating cache config."""
        policy = CachePolicy()
        config = CacheConfig(
            name="test_cache",
            level=CacheLevel.L1,
            policy=policy,
            enable_monitoring=True,
            metrics_interval=30.0,
        )

        assert config.name == "test_cache"
        assert config.level == CacheLevel.L1
        assert config.policy == policy
        assert config.enable_monitoring is True
        assert config.metrics_interval == 30.0

    def test_cache_config_defaults(self):
        """Test cache config with default values."""
        config = CacheConfig()

        assert config.name == "default"
        assert config.level == CacheLevel.L2
        assert isinstance(config.policy, CachePolicy)
        assert config.enable_monitoring is True
        assert config.metrics_interval == 60.0

    def test_cache_config_validation(self):
        """Test cache config validation."""
        # Valid config
        config = CacheConfig(name="test", metrics_interval=30.0)
        assert config.name == "test"
        assert config.metrics_interval == 30.0


class TestCacheEntry:
    """Test CacheEntry functionality."""

    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        current_time = time.time()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=current_time,
            accessed_at=current_time,
            ttl=3600,
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at > 0
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert entry.accessed_at == entry.created_at

    def test_cache_entry_defaults(self):
        """Test cache entry with default values."""
        entry = CacheEntry(key="test", value="value")

        assert entry.key == "test"
        assert entry.value == "value"
        assert entry.created_at > 0
        assert entry.ttl is None
        assert entry.access_count == 0
        # accessed_at and created_at may be slightly different due to timing
        assert abs(entry.accessed_at - entry.created_at) < 0.01

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration check."""
        # Non-expiring entry
        entry = CacheEntry(key="test", value="value")
        assert not entry.is_expired()

        # Expired entry
        entry = CacheEntry(
            key="test", value="value", created_at=time.time() - 3700, ttl=3600
        )
        assert entry.is_expired()

        # Non-expired entry
        entry = CacheEntry(
            key="test", value="value", created_at=time.time() - 1800, ttl=3600
        )
        assert not entry.is_expired()

    def test_cache_entry_access(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(key="test", value="value")
        initial_access_count = entry.access_count
        initial_accessed_at = entry.accessed_at

        # Access entry
        entry.update_access()

        assert entry.access_count == initial_access_count + 1
        assert entry.accessed_at > initial_accessed_at

    def test_cache_entry_size_bytes(self):
        """Test cache entry size bytes."""
        entry = CacheEntry(key="test", value="value", size_bytes=100)
        assert entry.size_bytes == 100


class TestCacheStats:
    """Test CacheStats functionality."""

    def test_cache_stats_creation(self):
        """Test creating cache stats."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.total_requests == 0
        assert stats.total_size_bytes == 0
        assert stats.entry_count == 0
        assert stats.memory_usage_bytes == 0

    def test_cache_stats_hit_rate(self):
        """Test cache stats hit rate calculation."""
        stats = CacheStats()

        # No operations
        assert stats.hit_rate == 0.0

        # Only misses
        stats.misses = 10
        stats.total_requests = 10
        stats.update_hit_rate()
        assert stats.hit_rate == 0.0

        # Only hits
        stats.hits = 10
        stats.misses = 0
        stats.total_requests = 10
        stats.update_hit_rate()
        assert stats.hit_rate == 1.0

        # Mixed
        stats.hits = 7
        stats.misses = 3
        stats.update_hit_rate()
        assert stats.hit_rate == 0.7

    def test_cache_stats_miss_rate(self):
        """Test cache stats miss rate calculation."""
        stats = CacheStats()

        # No operations
        assert stats.miss_rate == 0.0

        # Only hits
        stats.hits = 10
        stats.total_requests = 10
        stats.update_hit_rate()
        assert stats.miss_rate == 0.0

        # Only misses
        stats.hits = 0
        stats.misses = 10
        stats.total_requests = 10
        stats.update_hit_rate()
        assert stats.miss_rate == 1.0

        # Mixed
        stats.hits = 3
        stats.misses = 7
        stats.update_hit_rate()
        assert stats.miss_rate == 0.7

    def test_cache_stats_total_requests(self):
        """Test cache stats total requests calculation."""
        stats = CacheStats()

        # No operations
        assert stats.total_requests == 0

        # With operations
        stats.hits = 5
        stats.misses = 3
        stats.total_requests = 8

        assert stats.total_requests == 8


class TestCacheBackend:
    """Test CacheBackend abstract base class."""

    def test_cache_backend_creation(self):
        """Test creating cache backend."""
        config = CacheConfig()

        # Create a concrete implementation
        class TestCacheBackend(CacheBackend):
            def get(self, key: str):
                return "test_value"

            def set(self, key: str, value: Any, ttl: Optional[int] = None):
                pass

            def delete(self, key: str):
                pass

            def exists(self, key: str) -> bool:
                return True

            def clear(self):
                pass

            def size(self) -> int:
                return 1

            def keys(self):
                return ["test_key"]

        backend = TestCacheBackend(config)

        assert backend.config == config
        assert isinstance(backend.stats, CacheStats)
        assert backend._lock is not None

    def test_cache_backend_abstract_methods(self):
        """Test that CacheBackend is abstract."""
        config = CacheConfig()

        with pytest.raises(TypeError):
            CacheBackend(config)

    def test_cache_backend_stats_tracking(self):
        """Test cache backend stats tracking."""

        class TestCacheBackend(CacheBackend):
            def get(self, key: str):
                return "test_value"

            def set(self, key: str, value: Any, ttl: Optional[int] = None):
                pass

            def delete(self, key: str):
                pass

            def exists(self, key: str) -> bool:
                return True

            def clear(self):
                pass

            def size(self) -> int:
                return 1

            def keys(self):
                return ["test_key"]

        config = CacheConfig()
        backend = TestCacheBackend(config)

        # Test stats tracking
        backend._update_stats_on_hit()
        assert backend.stats.hits == 1

        backend._update_stats_on_miss()
        assert backend.stats.misses == 1

        backend._update_stats_on_set()
        assert backend.stats.sets == 1

        backend._update_stats_on_delete()
        assert backend.stats.deletes == 1

        backend._update_stats_on_evict()
        assert backend.stats.evictions == 1

    def test_cache_backend_thread_safety(self):
        """Test cache backend thread safety."""

        class TestCacheBackend(CacheBackend):
            def __init__(self, config):
                super().__init__(config)
                self._counter = 0

            def get(self, key: str):
                return "test_value"

            def set(self, key: str, value: Any, ttl: Optional[int] = None):
                pass

            def delete(self, key: str):
                pass

            def exists(self, key: str) -> bool:
                return True

            def clear(self):
                pass

            def size(self) -> int:
                return 1

            def keys(self):
                return ["test_key"]

            def increment_counter(self):
                with self._lock:
                    self._counter += 1
                    return self._counter

        config = CacheConfig()
        backend = TestCacheBackend(config)

        # Test thread safety
        results = []

        def worker():
            for _ in range(100):
                results.append(backend.increment_counter())

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be unique (no race conditions)
        assert len(set(results)) == len(results)
        assert max(results) == 500


# Additional test classes for CacheEntry functionality
class TestCacheEntryAdvanced:
    """Test advanced CacheEntry functionality."""

    def test_cache_entry_update_access(self):
        """Test cache entry access update."""
        entry = CacheEntry(key="test", value="value")
        initial_access_count = entry.access_count
        initial_accessed_at = entry.accessed_at

        # Update access
        entry.update_access()

        assert entry.access_count == initial_access_count + 1
        assert entry.accessed_at > initial_accessed_at

    def test_cache_entry_get_age(self):
        """Test cache entry age calculation."""
        entry = CacheEntry(key="test", value="value")
        initial_age = entry.get_age()

        # Wait a bit
        time.sleep(0.01)

        new_age = entry.get_age()
        assert new_age > initial_age

    def test_cache_entry_get_idle_time(self):
        """Test cache entry idle time calculation."""
        entry = CacheEntry(key="test", value="value")
        initial_idle_time = entry.get_idle_time()

        # Wait a bit
        time.sleep(0.01)

        new_idle_time = entry.get_idle_time()
        assert new_idle_time > initial_idle_time

    def test_cache_entry_with_tags(self):
        """Test cache entry with tags."""
        entry = CacheEntry(key="test", value="value", tags={"tag1", "tag2"})

        assert "tag1" in entry.tags
        assert "tag2" in entry.tags
        assert len(entry.tags) == 2

    def test_cache_entry_with_metadata(self):
        """Test cache entry with metadata."""
        metadata = {"source": "test", "version": "1.0"}
        entry = CacheEntry(key="test", value="value", metadata=metadata)

        assert entry.metadata == metadata
        assert entry.metadata["source"] == "test"
        assert entry.metadata["version"] == "1.0"
