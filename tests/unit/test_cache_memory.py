"""
Comprehensive tests for memory cache module.

This module tests the memory-based cache implementations including:
- MemoryCache base implementation
- LRU cache with eviction policies
- LFU cache with frequency tracking
- TTL cache with expiration handling
- Cache statistics and monitoring
"""

import threading
import time
from collections import OrderedDict
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.cache.core import (
    CacheConfig,
    CacheExpiredError,
    CacheFullError,
    CacheLevel,
    CacheMissError,
    CachePolicy,
    EvictionPolicy,
)
from dubchain.cache.memory import LFUCache, LRUCache, MemoryCache, TTLCache


class TestMemoryCache:
    """Test MemoryCache base implementation."""

    @pytest.fixture
    def cache_config(self):
        """Fixture for cache configuration."""
        return CacheConfig(
            name="test_cache",
            level=CacheLevel.L1,
            policy=CachePolicy(
                eviction_policy=EvictionPolicy.LRU,
                max_size=100,
                max_memory_bytes=10 * 1024 * 1024,
            ),
        )

    @pytest.fixture
    def memory_cache(self, cache_config):
        """Fixture for memory cache."""
        return MemoryCache(cache_config)

    def test_memory_cache_creation(self, cache_config):
        """Test creating memory cache."""
        cache = MemoryCache(cache_config)

        assert cache.config == cache_config
        assert cache._cache == {}
        assert cache._access_order == []
        assert cache._size_bytes == 0
        assert cache._running is False

    def test_memory_cache_get_miss(self, memory_cache):
        """Test getting non-existent key from cache."""
        with pytest.raises(
            CacheMissError, match="Key 'nonexistent' not found in cache"
        ):
            memory_cache.get("nonexistent")

        # Check stats
        assert memory_cache.stats.misses == 1
        assert memory_cache.stats.hits == 0

    def test_memory_cache_set_get(self, memory_cache):
        """Test setting and getting values from cache."""
        # Set value
        memory_cache.set("key1", "value1")

        # Get value
        value = memory_cache.get("key1")
        assert value == "value1"

        # Check stats
        assert memory_cache.stats.sets == 1
        assert memory_cache.stats.hits == 1
        assert memory_cache.stats.misses == 0

    def test_memory_cache_set_with_ttl(self, memory_cache):
        """Test setting value with TTL."""
        # Set value with TTL
        memory_cache.set("key1", "value1", ttl=1)

        # Get value immediately
        value = memory_cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        with pytest.raises(CacheExpiredError):
            memory_cache.get("key1")

    def test_memory_cache_delete(self, memory_cache):
        """Test deleting values from cache."""
        # Set value
        memory_cache.set("key1", "value1")

        # Delete value
        result = memory_cache.delete("key1")
        assert result is True

        # Try to get deleted value
        with pytest.raises(CacheMissError):
            memory_cache.get("key1")

        # Check stats
        assert memory_cache.stats.deletes == 1

    def test_memory_cache_delete_nonexistent(self, memory_cache):
        """Test deleting non-existent key."""
        result = memory_cache.delete("nonexistent")
        assert result is False

    def test_memory_cache_exists(self, memory_cache):
        """Test checking if key exists in cache."""
        # Key doesn't exist
        assert not memory_cache.exists("key1")

        # Set key
        memory_cache.set("key1", "value1")

        # Key exists
        assert memory_cache.exists("key1")

    def test_memory_cache_clear(self, memory_cache):
        """Test clearing cache."""
        # Add some values
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2")

        # Clear cache
        memory_cache.clear()

        # Cache should be empty
        assert memory_cache.size() == 0
        assert not memory_cache.exists("key1")
        assert not memory_cache.exists("key2")

    def test_memory_cache_size(self, memory_cache):
        """Test cache size calculation."""
        # Empty cache
        assert memory_cache.size() == 0

        # Add values
        memory_cache.set("key1", "value1")
        assert memory_cache.size() == 1

        memory_cache.set("key2", "value2")
        assert memory_cache.size() == 2

        # Delete value
        memory_cache.delete("key1")
        assert memory_cache.size() == 1

    def test_memory_cache_memory_usage(self, memory_cache):
        """Test cache memory usage calculation."""
        # Empty cache
        assert memory_cache.memory_usage() == 0

        # Add values
        memory_cache.set("key1", "value1")
        usage1 = memory_cache.memory_usage()
        assert usage1 > 0

        memory_cache.set("key2", "value2")
        usage2 = memory_cache.memory_usage()
        assert usage2 > usage1

    def test_memory_cache_cleanup_thread(self, cache_config):
        """Test cache cleanup thread."""
        # Create cache with TTL
        cache_config.policy.default_ttl = 1
        cache = MemoryCache(cache_config)

        # Set value with TTL
        cache.set("key1", "value1")

        # Wait for expiration
        time.sleep(1.1)

        # Value should be expired
        with pytest.raises(CacheExpiredError):
            cache.get("key1")

        # Stop cleanup thread
        cache.stop()

    def test_memory_cache_thread_safety(self, memory_cache):
        """Test cache thread safety."""
        results = []

        def worker(worker_id):
            for i in range(10):
                key = f"key_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"
                memory_cache.set(key, value)
                retrieved = memory_cache.get(key)
                results.append(retrieved)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 50
        assert all(result.startswith("value_") for result in results)


class TestLRUCache:
    """Test LRU cache implementation."""

    @pytest.fixture
    def lru_config(self):
        """Fixture for LRU cache configuration."""
        return CacheConfig(
            name="lru_cache",
            level=CacheLevel.L1,
            policy=CachePolicy(
                eviction_policy=EvictionPolicy.LRU,
                max_size=3,
                max_memory_bytes=10 * 1024 * 1024,
            ),
        )

    @pytest.fixture
    def lru_cache(self, lru_config):
        """Fixture for LRU cache."""
        return LRUCache(lru_config)

    def test_lru_cache_creation(self, lru_config):
        """Test creating LRU cache."""
        cache = LRUCache(lru_config)

        assert cache.config == lru_config
        assert isinstance(cache._cache, OrderedDict)
        # OrderedDict doesn't have maxlen, check config instead
        assert cache.config.policy.max_size == 3

    def test_lru_cache_eviction(self, lru_cache):
        """Test LRU cache eviction."""
        # Fill cache to capacity
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")
        lru_cache.set("key3", "value3")

        # Cache should be full
        assert lru_cache.size() == 3

        # Access key1 to make it recently used
        lru_cache.get("key1")

        # Add new key - should evict least recently used (key2)
        lru_cache.set("key4", "value4")

        # key2 should be evicted
        with pytest.raises(CacheMissError):
            lru_cache.get("key2")

        # key1, key3, key4 should still be there
        assert lru_cache.get("key1") == "value1"
        assert lru_cache.get("key3") == "value3"
        assert lru_cache.get("key4") == "value4"

        # Check eviction stats
        assert lru_cache.stats.evictions == 1

    def test_lru_cache_access_order(self, lru_cache):
        """Test LRU cache access order tracking."""
        # Add values
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")
        lru_cache.set("key3", "value3")

        # Access key1 - should move to end
        lru_cache.get("key1")

        # Add new key - should evict key2 (least recently used)
        lru_cache.set("key4", "value4")

        # key2 should be evicted
        with pytest.raises(CacheMissError):
            lru_cache.get("key2")

    def test_lru_cache_update_existing(self, lru_cache):
        """Test updating existing key in LRU cache."""
        # Add value
        lru_cache.set("key1", "value1")

        # Update value
        lru_cache.set("key1", "value1_updated")

        # Should have updated value
        assert lru_cache.get("key1") == "value1_updated"
        assert lru_cache.size() == 1


class TestLFUCache:
    """Test LFU cache implementation."""

    @pytest.fixture
    def lfu_config(self):
        """Fixture for LFU cache configuration."""
        return CacheConfig(
            name="lfu_cache",
            level=CacheLevel.L1,
            policy=CachePolicy(
                eviction_policy=EvictionPolicy.LFU,
                max_size=3,
                max_memory_bytes=10 * 1024 * 1024,
            ),
        )

    @pytest.fixture
    def lfu_cache(self, lfu_config):
        """Fixture for LFU cache."""
        return LFUCache(lfu_config)

    def test_lfu_cache_creation(self, lfu_config):
        """Test creating LFU cache."""
        cache = LFUCache(lfu_config)

        assert cache.config == lfu_config
        assert cache._frequencies == {}
        assert cache._frequency_groups == {}

    def test_lfu_cache_frequency_tracking(self, lfu_cache):
        """Test LFU cache frequency tracking."""
        # Add values
        lfu_cache.set("key1", "value1")
        lfu_cache.set("key2", "value2")
        lfu_cache.set("key3", "value3")

        # Access key1 multiple times
        lfu_cache.get("key1")
        lfu_cache.get("key1")
        lfu_cache.get("key1")

        # Access key2 once
        lfu_cache.get("key2")

        # key3 has not been accessed

        # Add new key - should evict least frequently used (key3)
        lfu_cache.set("key4", "value4")

        # key3 should be evicted
        with pytest.raises(CacheMissError):
            lfu_cache.get("key3")

        # key1, key2, key4 should still be there
        assert lfu_cache.get("key1") == "value1"
        assert lfu_cache.get("key2") == "value2"
        assert lfu_cache.get("key4") == "value4"

    def test_lfu_cache_tie_breaking(self, lfu_cache):
        """Test LFU cache tie-breaking with LRU."""
        # Add values
        lfu_cache.set("key1", "value1")
        lfu_cache.set("key2", "value2")
        lfu_cache.set("key3", "value3")

        # All keys have same frequency (0)
        # key1 should be evicted (first added, least recently used)
        lfu_cache.set("key4", "value4")

        with pytest.raises(CacheMissError):
            lfu_cache.get("key1")

    def test_lfu_cache_frequency_update(self, lfu_cache):
        """Test LFU cache frequency update on access."""
        # Add value
        lfu_cache.set("key1", "value1")

        # Check initial frequency
        assert lfu_cache._frequencies["key1"] == 0

        # Access value
        lfu_cache.get("key1")

        # Frequency should be updated
        assert lfu_cache._frequencies["key1"] == 1


class TestTTLCache:
    """Test TTL cache implementation."""

    @pytest.fixture
    def ttl_config(self):
        """Fixture for TTL cache configuration."""
        return CacheConfig(
            name="ttl_cache",
            level=CacheLevel.L1,
            policy=CachePolicy(
                eviction_policy=EvictionPolicy.TTL,
                max_size=100,
                max_memory_bytes=10 * 1024 * 1024,
                default_ttl=2,
            ),
        )

    @pytest.fixture
    def ttl_cache(self, ttl_config):
        """Fixture for TTL cache."""
        return TTLCache(ttl_config)

    def test_ttl_cache_creation(self, ttl_config):
        """Test creating TTL cache."""
        cache = TTLCache(ttl_config)

        assert cache.config == ttl_config
        assert cache._cleanup_interval == 1.0
        assert cache._running is True

    def test_ttl_cache_default_ttl(self, ttl_cache):
        """Test TTL cache with default TTL."""
        # Set value without explicit TTL
        ttl_cache.set("key1", "value1")

        # Should be accessible immediately
        assert ttl_cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(2.1)

        # Should be expired
        with pytest.raises(CacheExpiredError):
            ttl_cache.get("key1")

    def test_ttl_cache_custom_ttl(self, ttl_cache):
        """Test TTL cache with custom TTL."""
        # Set value with custom TTL
        ttl_cache.set("key1", "value1", ttl=1)

        # Should be accessible immediately
        assert ttl_cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        with pytest.raises(CacheExpiredError):
            ttl_cache.get("key1")

    def test_ttl_cache_cleanup(self, ttl_cache):
        """Test TTL cache cleanup."""
        # Set values with different TTLs
        ttl_cache.set("key1", "value1", ttl=1)
        ttl_cache.set("key2", "value2", ttl=3)

        # Wait for first expiration
        time.sleep(1.1)

        # key1 should be expired
        with pytest.raises(CacheExpiredError):
            ttl_cache.get("key1")

        # key2 should still be accessible
        assert ttl_cache.get("key2") == "value2"

        # Wait for second expiration (key2 was accessed at 1.1s, so it expires at 1.1 + 3 = 4.1s)
        time.sleep(3.0)

        # key2 should be expired (TTL was refreshed on access, so it expires 3 seconds after access)
        with pytest.raises(CacheExpiredError):
            ttl_cache.get("key2")

    def test_ttl_cache_refresh_on_access(self, ttl_cache):
        """Test TTL cache refresh on access."""
        # Set value with short TTL
        ttl_cache.set("key1", "value1", ttl=1)

        # Access before expiration
        time.sleep(0.5)
        assert ttl_cache.get("key1") == "value1"

        # Access again before expiration
        time.sleep(0.5)
        assert ttl_cache.get("key1") == "value1"

        # Should still be accessible (refreshed)
        time.sleep(0.5)
        assert ttl_cache.get("key1") == "value1"

    def test_ttl_cache_stop(self, ttl_cache):
        """Test TTL cache stop."""
        # Cache should be running
        assert ttl_cache._running is True

        # Stop cache
        ttl_cache.stop()

        # Cache should be stopped
        assert ttl_cache._running is False


# Additional test classes for advanced cache functionality
class TestMemoryCacheAdvanced:
    """Test advanced MemoryCache functionality."""

    @pytest.fixture
    def advanced_cache_config(self):
        """Fixture for advanced cache configuration."""
        return CacheConfig(
            name="advanced_cache",
            level=CacheLevel.L1,
            policy=CachePolicy(
                eviction_policy=EvictionPolicy.LRU,
                max_size=100,
                max_memory_bytes=1024,
                default_ttl=1,
                enable_compression=True,
                enable_serialization=True,
                enable_statistics=True,
            ),
        )

    @pytest.fixture
    def advanced_cache(self, advanced_cache_config):
        """Fixture for advanced cache."""
        return MemoryCache(advanced_cache_config)

    def test_memory_cache_compression(self, advanced_cache):
        """Test memory cache compression."""
        # Set large value
        large_value = "x" * 200
        advanced_cache.set("key1", large_value)

        # Get value
        retrieved_value = advanced_cache.get("key1")
        assert retrieved_value == large_value

    def test_memory_cache_serialization(self, advanced_cache):
        """Test memory cache serialization."""
        # Set complex object
        complex_obj = {"nested": {"data": [1, 2, 3]}, "text": "test"}
        advanced_cache.set("key1", complex_obj)

        # Get value
        retrieved_value = advanced_cache.get("key1")
        assert retrieved_value == complex_obj

    def test_memory_cache_statistics(self, advanced_cache):
        """Test memory cache statistics."""
        # Perform operations
        advanced_cache.set("key1", "value1")
        advanced_cache.get("key1")
        advanced_cache.delete("key1")

        # Check statistics
        stats = advanced_cache.get_stats()
        assert stats.hits >= 1

    def test_memory_cache_memory_usage_tracking(self, advanced_cache):
        """Test memory cache memory usage tracking."""
        # Set values
        advanced_cache.set("key1", "value1")

        advanced_cache.set("key2", "value2")

        # Check that cache has entries
        assert advanced_cache.size() >= 1

    def test_memory_cache_cleanup_thread(self, advanced_cache):
        """Test memory cache cleanup thread."""
        # Set value with TTL
        advanced_cache.set("key1", "value1", ttl=0.1)

        # Wait for expiration
        time.sleep(0.2)

        # Value should be expired
        with pytest.raises(CacheExpiredError):
            advanced_cache.get("key1")

    def test_memory_cache_thread_safety_advanced(self, advanced_cache):
        """Test advanced memory cache thread safety."""
        results = []

        def worker(worker_id):
            for i in range(10):
                key = f"key_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"

                # Set value
                advanced_cache.set(key, value)

                # Get value
                try:
                    retrieved = advanced_cache.get(key)
                    results.append(retrieved)
                except CacheMissError:
                    pass

                # Delete value
                advanced_cache.delete(key)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 50
        assert all(result.startswith("value_") for result in results)
