"""
Comprehensive tests for cache manager module.
"""

import logging

logger = logging.getLogger(__name__)
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.cache.core import (
    CacheConfig,
    CacheError,
    CacheLevel,
    CacheMissError,
    CachePolicy,
    CacheStats,
    EvictionPolicy,
)
from dubchain.cache.manager import CacheHierarchy, CacheManager
from dubchain.cache.memory import MemoryCache


class TestCacheHierarchy:
    """Test CacheHierarchy functionality."""

    def test_cache_hierarchy_creation_defaults(self):
        """Test creating cache hierarchy with default values."""
        hierarchy = CacheHierarchy()
        assert hierarchy.levels == [
            CacheLevel.L1,
            CacheLevel.L2,
            CacheLevel.L3,
            CacheLevel.L4,
        ]
        assert hierarchy.promote_on_hit is True
        assert hierarchy.demote_on_evict is True
        assert hierarchy.cross_level_invalidation is True
        assert hierarchy.async_operations is False
        assert hierarchy.batch_operations is True
        assert hierarchy.batch_size == 100
        assert hierarchy.enable_hierarchy_stats is True
        assert hierarchy.stats_interval == 60.0

    def test_cache_hierarchy_creation_custom(self):
        """Test creating cache hierarchy with custom values."""
        custom_levels = [CacheLevel.L1, CacheLevel.L2]
        hierarchy = CacheHierarchy(
            levels=custom_levels,
            promote_on_hit=False,
            demote_on_evict=False,
            cross_level_invalidation=False,
            async_operations=True,
            batch_operations=False,
            batch_size=50,
            enable_hierarchy_stats=False,
            stats_interval=30.0,
        )
        assert hierarchy.levels == custom_levels
        assert hierarchy.promote_on_hit is False
        assert hierarchy.demote_on_evict is False
        assert hierarchy.cross_level_invalidation is False
        assert hierarchy.async_operations is True
        assert hierarchy.batch_operations is False
        assert hierarchy.batch_size == 50
        assert hierarchy.enable_hierarchy_stats is False
        assert hierarchy.stats_interval == 30.0


class TestCacheManager:
    """Test CacheManager functionality."""

    @pytest.fixture
    def hierarchy_no_stats(self):
        """Create a hierarchy with stats disabled."""
        return CacheHierarchy(enable_hierarchy_stats=False)

    @pytest.fixture
    def manager_no_stats(self, hierarchy_no_stats):
        """Create a manager with stats disabled."""
        return CacheManager(hierarchy_no_stats)

    def test_cache_manager_creation(self, manager_no_stats):
        """Test creating cache manager."""
        manager = manager_no_stats
        assert manager._caches == {}
        assert manager._running is False

    def test_cache_manager_creation_with_stats_disabled(self, manager_no_stats):
        """Test creating cache manager with stats disabled."""
        manager = manager_no_stats
        assert manager._stats_thread is None
        assert manager._running is False

    def test_add_cache(self, manager_no_stats):
        """Test adding a cache to the hierarchy."""
        manager = manager_no_stats
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        assert CacheLevel.L1 in manager._caches
        assert manager._caches[CacheLevel.L1] == cache

    def test_remove_cache(self, manager_no_stats):
        """Test removing a cache from the hierarchy."""
        manager = manager_no_stats
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        assert CacheLevel.L1 in manager._caches

        manager.remove_cache(CacheLevel.L1)
        assert CacheLevel.L1 not in manager._caches

    def test_remove_cache_nonexistent(self, manager_no_stats):
        """Test removing a nonexistent cache."""
        manager = manager_no_stats

        # Should not raise exception
        manager.remove_cache(CacheLevel.L1)

    def test_get_cache(self):
        """Test getting a cache from the hierarchy."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        retrieved_cache = manager.get_cache(CacheLevel.L1)
        assert retrieved_cache == cache

    def test_get_cache_nonexistent(self):
        """Test getting a nonexistent cache."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        retrieved_cache = manager.get_cache(CacheLevel.L1)
        assert retrieved_cache is None

    def test_get_from_specific_level(self):
        """Test getting a value from a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")

        value = manager.get("key1", level=CacheLevel.L1)
        assert value == "value1"

    def test_get_from_specific_level_nonexistent_cache(self):
        """Test getting from a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        with pytest.raises(CacheError, match="No cache at level"):
            manager.get("key1", level=CacheLevel.L1)

    def test_get_from_specific_level_miss(self):
        """Test getting a miss from a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        with pytest.raises(CacheMissError):
            manager.get("nonexistent", level=CacheLevel.L1)

    def test_get_from_hierarchy_hit(self):
        """Test getting a value from the cache hierarchy."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache2.set("key1", "value1")

        value = manager.get("key1")
        assert value == "value1"

    def test_get_from_hierarchy_miss(self):
        """Test getting a miss from the cache hierarchy."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        with pytest.raises(CacheMissError, match="Key 'nonexistent' not found"):
            manager.get("nonexistent")

    def test_get_from_hierarchy_with_promotion(self):
        """Test getting from hierarchy with promotion enabled."""
        hierarchy = CacheHierarchy(promote_on_hit=True)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache2.set("key1", "value1")

        value = manager.get("key1")
        assert value == "value1"

        # Check that value was promoted to L1
        assert cache1.get("key1") == "value1"

    def test_get_from_hierarchy_without_promotion(self):
        """Test getting from hierarchy with promotion disabled."""
        hierarchy = CacheHierarchy(promote_on_hit=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache2.set("key1", "value1")

        value = manager.get("key1")
        assert value == "value1"

        # Check that value was not promoted to L1
        with pytest.raises(CacheMissError):
            cache1.get("key1")

    def test_set_in_specific_level(self):
        """Test setting a value in a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        manager.set("key1", "value1", level=CacheLevel.L1)

        assert cache.get("key1") == "value1"

    def test_set_in_specific_level_nonexistent_cache(self):
        """Test setting in a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        with pytest.raises(CacheError, match="No cache at level"):
            manager.set("key1", "value1", level=CacheLevel.L1)

    def test_set_in_all_levels(self):
        """Test setting a value in all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        manager.set("key1", "value1")

        assert cache1.get("key1") == "value1"
        assert cache2.get("key1") == "value1"

    def test_set_with_ttl_and_tags(self):
        """Test setting a value with TTL and tags."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        manager.set(
            "key1", "value1", level=CacheLevel.L1, ttl=60.0, tags={"tag1", "tag2"}
        )

        assert cache.get("key1") == "value1"

    def test_delete_from_specific_level(self):
        """Test deleting a value from a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")

        result = manager.delete("key1", level=CacheLevel.L1)
        assert result is True

        with pytest.raises(CacheMissError):
            cache.get("key1")

    def test_delete_from_specific_level_nonexistent_cache(self):
        """Test deleting from a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        result = manager.delete("key1", level=CacheLevel.L1)
        assert result is False

    def test_delete_from_all_levels(self):
        """Test deleting a value from all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("key1", "value1")
        cache2.set("key1", "value1")

        result = manager.delete("key1")
        assert result is True

        with pytest.raises(CacheMissError):
            cache1.get("key1")
        with pytest.raises(CacheMissError):
            cache2.get("key1")

    def test_delete_nonexistent_key(self):
        """Test deleting a nonexistent key."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        result = manager.delete("nonexistent", level=CacheLevel.L1)
        assert result is False

    def test_exists_in_specific_level(self):
        """Test checking existence in a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")

        assert manager.exists("key1", level=CacheLevel.L1) is True
        assert manager.exists("nonexistent", level=CacheLevel.L1) is False

    def test_exists_in_specific_level_nonexistent_cache(self):
        """Test checking existence in a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        assert manager.exists("key1", level=CacheLevel.L1) is False

    def test_exists_in_all_levels(self):
        """Test checking existence in all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache2.set("key1", "value1")

        assert manager.exists("key1") is True
        assert manager.exists("nonexistent") is False

    def test_clear_specific_level(self):
        """Test clearing a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")

        manager.clear(level=CacheLevel.L1)

        with pytest.raises(CacheMissError):
            cache.get("key1")

    def test_clear_specific_level_nonexistent_cache(self):
        """Test clearing a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        # Should not raise exception
        manager.clear(level=CacheLevel.L1)

    def test_clear_all_levels(self):
        """Test clearing all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        manager.clear()

        with pytest.raises(CacheMissError):
            cache1.get("key1")
        with pytest.raises(CacheMissError):
            cache2.get("key2")

    def test_size_specific_level(self):
        """Test getting size of a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        size = manager.size(level=CacheLevel.L1)
        assert size == 2

    def test_size_specific_level_nonexistent_cache(self):
        """Test getting size of a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        size = manager.size(level=CacheLevel.L1)
        assert size == 0

    def test_size_all_levels(self):
        """Test getting total size of all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        cache2.set("key3", "value3")

        total_size = manager.size()
        assert total_size == 3

    def test_keys_specific_level(self):
        """Test getting keys from a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        keys = manager.keys(level=CacheLevel.L1)
        assert set(keys) == {"key1", "key2"}

    def test_keys_specific_level_nonexistent_cache(self):
        """Test getting keys from a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        keys = manager.keys(level=CacheLevel.L1)
        assert keys == []

    def test_keys_all_levels(self):
        """Test getting keys from all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        cache2.set("key3", "value3")

        keys = manager.keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_invalidate_by_tag_specific_level(self):
        """Test invalidating by tag in a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1", tags={"tag1"})
        cache.set("key2", "value2", tags={"tag2"})

        invalidated = manager.invalidate_by_tag("tag1", level=CacheLevel.L1)
        assert invalidated == 1

        with pytest.raises(CacheMissError):
            cache.get("key1")
        assert cache.get("key2") == "value2"

    def test_invalidate_by_tag_specific_level_nonexistent_cache(self):
        """Test invalidating by tag in a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        invalidated = manager.invalidate_by_tag("tag1", level=CacheLevel.L1)
        assert invalidated == 0

    def test_invalidate_by_tag_all_levels(self):
        """Test invalidating by tag in all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("key1", "value1", tags={"tag1"})
        cache2.set("key2", "value2", tags={"tag1"})
        cache2.set("key3", "value3", tags={"tag2"})

        invalidated = manager.invalidate_by_tag("tag1")
        assert invalidated == 2

        with pytest.raises(CacheMissError):
            cache1.get("key1")
        with pytest.raises(CacheMissError):
            cache2.get("key2")
        assert cache2.get("key3") == "value3"

    def test_invalidate_by_pattern_specific_level(self):
        """Test invalidating by pattern in a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("user_1", "value1")
        cache.set("user_2", "value2")
        cache.set("admin_1", "value3")

        invalidated = manager.invalidate_by_pattern("user_*", level=CacheLevel.L1)
        assert invalidated == 2

        with pytest.raises(CacheMissError):
            cache.get("user_1")
        with pytest.raises(CacheMissError):
            cache.get("user_2")
        assert cache.get("admin_1") == "value3"

    def test_invalidate_by_pattern_specific_level_nonexistent_cache(self):
        """Test invalidating by pattern in a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        invalidated = manager.invalidate_by_pattern("user_*", level=CacheLevel.L1)
        assert invalidated == 0

    def test_invalidate_by_pattern_all_levels(self):
        """Test invalidating by pattern in all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("user_1", "value1")
        cache2.set("user_2", "value2")
        cache2.set("admin_1", "value3")

        invalidated = manager.invalidate_by_pattern("user_*")
        assert invalidated == 2

        with pytest.raises(CacheMissError):
            cache1.get("user_1")
        with pytest.raises(CacheMissError):
            cache2.get("user_2")
        assert cache2.get("admin_1") == "value3"

    def test_warm_up_specific_level(self):
        """Test warming up a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        warmed = manager.warm_up(["key1", "key2"], level=CacheLevel.L1)
        assert warmed == 0  # MemoryCache doesn't implement warm_up

    def test_warm_up_specific_level_nonexistent_cache(self):
        """Test warming up a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        warmed = manager.warm_up(["key1", "key2"], level=CacheLevel.L1)
        assert warmed == 0

    def test_warm_up_all_levels(self):
        """Test warming up all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)

        warmed = manager.warm_up(["key1", "key2"])
        assert warmed == 0  # MemoryCache doesn't implement warm_up

    def test_optimize_specific_level(self):
        """Test optimizing a specific cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        # Should not raise exception
        manager.optimize(level=CacheLevel.L1)

    def test_optimize_specific_level_nonexistent_cache(self):
        """Test optimizing a nonexistent cache level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        # Should not raise exception
        manager.optimize(level=CacheLevel.L1)

    def test_optimize_all_levels(self):
        """Test optimizing all cache levels."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)

        # Should not raise exception
        manager.optimize()

    def test_get_hierarchy_stats(self):
        """Test getting hierarchy statistics."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")

        stats = manager.get_hierarchy_stats()

        assert CacheLevel.L1 in stats
        assert isinstance(stats[CacheLevel.L1], CacheStats)

    def test_get_combined_stats(self):
        """Test getting combined statistics."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        combined_stats = manager.get_combined_stats()

        assert isinstance(combined_stats, CacheStats)
        assert combined_stats.entry_count >= 0

    def test_health_check(self):
        """Test health check."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)

        health = manager.health_check()

        assert "status" in health
        assert "levels" in health
        assert "combined_stats" in health
        assert "timestamp" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_shutdown(self):
        """Test shutdown."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)
        cache = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache)
        cache.set("key1", "value1")

        manager.shutdown()

        # Cache should be cleared after shutdown
        with pytest.raises(CacheMissError):
            cache.get("key1")

    def test_context_manager(self):
        """Test context manager functionality."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        cache = MemoryCache(CacheConfig())

        with CacheManager(hierarchy) as manager:
            manager.add_cache(CacheLevel.L1, cache)
            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"

        # Cache should be cleared after context exit
        with pytest.raises(CacheMissError):
            cache.get("key1")

    def test_stats_worker_disabled(self, manager_no_stats):
        """Test that stats worker is disabled when stats are disabled."""
        manager = manager_no_stats
        assert manager._stats_thread is None
        assert manager._running is False

    def test_promote_to_higher_levels(self):
        """Test promotion to higher levels."""
        hierarchy = CacheHierarchy(promote_on_hit=True)
        manager = CacheManager(hierarchy)
        cache1 = MemoryCache(CacheConfig())
        cache2 = MemoryCache(CacheConfig())

        manager.add_cache(CacheLevel.L1, cache1)
        manager.add_cache(CacheLevel.L2, cache2)
        cache2.set("key1", "value1")

        # Manually call promotion
        manager._promote_to_higher_levels("key1", "value1", CacheLevel.L2)

        # Value should be promoted to L1
        assert cache1.get("key1") == "value1"

    def test_promote_to_higher_levels_invalid_level(self):
        """Test promotion with invalid level."""
        hierarchy = CacheHierarchy(enable_hierarchy_stats=False)
        manager = CacheManager(hierarchy)

        # Should not raise exception
        manager._promote_to_higher_levels("key1", "value1", CacheLevel.L1)
