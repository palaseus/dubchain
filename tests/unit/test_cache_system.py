"""Tests for the DubChain caching system."""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.cache.analytics import CacheAnalytics, CacheMetrics, CacheProfiler
from dubchain.cache.core import (
    CacheBackend,
    CacheConfig,
    CacheEntry,
    CacheError,
    CacheLevel,
    CachePolicy,
    CacheStats,
    EvictionPolicy,
)
from dubchain.cache.distributed import DistributedCache, MemcachedCache, RedisCache
from dubchain.cache.manager import CacheHierarchy, CacheManager
from dubchain.cache.memory import LFUCache, LRUCache, MemoryCache, TTLCache
from dubchain.cache.warming import (
    CacheWarmer,
    DatabaseWarmingDataSource,
    FileWarmingDataSource,
    WarmingConfig,
    WarmingDataSource,
    WarmingStrategy,
)


class TestCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(
            key="test_key", value="test_value", ttl=3600.0, tags={"tag1", "tag2"}
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600.0
        assert entry.tags == {"tag1", "tag2"}
        assert entry.access_count == 0
        assert entry.size_bytes == 0
        assert entry.created_at > 0
        assert entry.accessed_at > 0

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        entry = CacheEntry(key="test_key", value="test_value", ttl=0.1)  # 100ms TTL

        assert not entry.is_expired()

        time.sleep(0.2)
        assert entry.is_expired()

    def test_cache_entry_access_update(self):
        """Test cache entry access update."""
        entry = CacheEntry(key="test_key", value="test_value")
        initial_access_count = entry.access_count
        initial_accessed_at = entry.accessed_at

        time.sleep(0.01)  # Small delay
        entry.update_access()

        assert entry.access_count == initial_access_count + 1
        assert entry.accessed_at > initial_accessed_at

    def test_cache_entry_age_calculation(self):
        """Test cache entry age calculation."""
        entry = CacheEntry(key="test_key", value="test_value")

        time.sleep(0.1)
        age = entry.get_age()

        assert age >= 0.1
        assert age < 0.2  # Should be close to 0.1


class TestCachePolicy:
    """Test cache policy functionality."""

    def test_default_policy(self):
        """Test default cache policy."""
        policy = CachePolicy()

        assert policy.max_size == 1000
        assert policy.max_memory_bytes == 100 * 1024 * 1024
        assert policy.eviction_policy == EvictionPolicy.LRU
        assert policy.default_ttl is None
        assert policy.enable_compression is False
        assert policy.enable_serialization is True
        assert policy.enable_statistics is True
        assert policy.enable_analytics is False

    def test_custom_policy(self):
        """Test custom cache policy."""
        policy = CachePolicy(
            max_size=5000,
            max_memory_bytes=200 * 1024 * 1024,
            eviction_policy=EvictionPolicy.LFU,
            default_ttl=7200.0,
            enable_compression=True,
            enable_serialization=False,
            enable_statistics=False,
            enable_analytics=True,
        )

        assert policy.max_size == 5000
        assert policy.max_memory_bytes == 200 * 1024 * 1024
        assert policy.eviction_policy == EvictionPolicy.LFU
        assert policy.default_ttl == 7200.0
        assert policy.enable_compression is True
        assert policy.enable_serialization is False
        assert policy.enable_statistics is False
        assert policy.enable_analytics is True


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()

        assert config.name == "default"
        assert config.level == CacheLevel.L2
        assert config.backend_type == "memory"
        assert config.enable_async is False
        assert config.max_concurrent_operations == 100
        assert config.operation_timeout == 5.0
        assert config.enable_monitoring is True
        assert config.metrics_interval == 60.0
        assert config.log_level == "INFO"
        assert config.enable_clustering is False
        assert config.replication_factor == 1
        assert config.consistency_level == "eventual"

    def test_custom_config(self):
        """Test custom cache configuration."""
        config = CacheConfig(
            name="custom_cache",
            level=CacheLevel.L1,
            backend_type="redis",
            enable_async=True,
            max_concurrent_operations=200,
            operation_timeout=10.0,
            enable_monitoring=False,
            metrics_interval=120.0,
            log_level="DEBUG",
            enable_clustering=True,
            cluster_nodes=["node1", "node2"],
            replication_factor=3,
            consistency_level="strong",
        )

        assert config.name == "custom_cache"
        assert config.level == CacheLevel.L1
        assert config.backend_type == "redis"
        assert config.enable_async is True
        assert config.max_concurrent_operations == 200
        assert config.operation_timeout == 10.0
        assert config.enable_monitoring is False
        assert config.metrics_interval == 120.0
        assert config.log_level == "DEBUG"
        assert config.enable_clustering is True
        assert config.cluster_nodes == ["node1", "node2"]
        assert config.replication_factor == 3
        assert config.consistency_level == "strong"


class TestMemoryCache:
    """Test memory cache implementation."""

    @pytest.fixture
    def cache_config(self):
        """Create cache configuration."""
        return CacheConfig(
            name="test_memory_cache",
            level=CacheLevel.L2,
            policy=CachePolicy(max_size=100, max_memory_bytes=1024 * 1024),
        )

    @pytest.fixture
    def memory_cache(self, cache_config):
        """Create memory cache."""
        return MemoryCache(cache_config)

    def test_memory_cache_initialization(self, memory_cache):
        """Test memory cache initialization."""
        assert memory_cache.config.name == "test_memory_cache"
        assert memory_cache.config.level == CacheLevel.L2
        assert memory_cache.size() == 0
        assert len(memory_cache.keys()) == 0

    def test_memory_cache_set_get(self, memory_cache):
        """Test setting and getting values."""
        # Set value
        memory_cache.set("key1", "value1")

        # Get value
        value = memory_cache.get("key1")
        assert value == "value1"

        # Check size
        assert memory_cache.size() == 1
        assert "key1" in memory_cache.keys()

    def test_memory_cache_miss(self, memory_cache):
        """Test cache miss."""
        with pytest.raises(CacheError):
            memory_cache.get("nonexistent_key")

    def test_memory_cache_delete(self, memory_cache):
        """Test cache deletion."""
        # Set value
        memory_cache.set("key1", "value1")
        assert memory_cache.size() == 1

        # Delete value
        result = memory_cache.delete("key1")
        assert result is True
        assert memory_cache.size() == 0

        # Try to delete non-existent key
        result = memory_cache.delete("nonexistent_key")
        assert result is False

    def test_memory_cache_exists(self, memory_cache):
        """Test cache existence check."""
        # Set value
        memory_cache.set("key1", "value1")

        # Check existence
        assert memory_cache.exists("key1") is True
        assert memory_cache.exists("nonexistent_key") is False

    def test_memory_cache_clear(self, memory_cache):
        """Test cache clearing."""
        # Set multiple values
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2")
        assert memory_cache.size() == 2

        # Clear cache
        memory_cache.clear()
        assert memory_cache.size() == 0
        assert len(memory_cache.keys()) == 0

    def test_memory_cache_ttl(self, memory_cache):
        """Test cache TTL."""
        # Set value with TTL
        memory_cache.set("key1", "value1", ttl=0.1)

        # Should be available immediately
        assert memory_cache.exists("key1") is True

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert memory_cache.exists("key1") is False

    def test_memory_cache_tags(self, memory_cache):
        """Test cache tags."""
        # Set value with tags
        memory_cache.set("key1", "value1", tags={"tag1", "tag2"})

        # Invalidate by tag
        invalidated = memory_cache.invalidate_by_tag("tag1")
        assert invalidated == 1
        assert memory_cache.exists("key1") is False

    def test_memory_cache_pattern_invalidation(self, memory_cache):
        """Test cache pattern invalidation."""
        # Set multiple values
        memory_cache.set("user:1", "value1")
        memory_cache.set("user:2", "value2")
        memory_cache.set("post:1", "value3")

        # Invalidate by pattern
        invalidated = memory_cache.invalidate_by_pattern("user:.*")
        assert invalidated == 2
        assert memory_cache.exists("user:1") is False
        assert memory_cache.exists("user:2") is False
        assert memory_cache.exists("post:1") is True

    def test_memory_cache_stats(self, memory_cache):
        """Test cache statistics."""
        # Set and get values
        memory_cache.set("key1", "value1")
        memory_cache.get("key1")  # Hit

        # Try to get non-existent key (will raise exception)
        try:
            memory_cache.get("key2")  # Miss
        except CacheError:
            pass  # Expected to fail

        stats = memory_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.total_requests == 2
        assert stats.hit_rate == 0.5

    def test_memory_cache_eviction(self, memory_cache):
        """Test cache eviction."""
        # Set max size to 2
        memory_cache.config.policy.max_size = 2

        # Add 3 values
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2")
        memory_cache.set("key3", "value3")

        # Should have evicted one entry
        assert memory_cache.size() <= 2

    def test_memory_cache_context_manager(self, memory_cache):
        """Test cache as context manager."""
        with memory_cache as cache:
            assert cache is memory_cache
            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"


class TestLRUCache:
    """Test LRU cache implementation."""

    @pytest.fixture
    def lru_cache(self):
        """Create LRU cache."""
        config = CacheConfig(
            name="test_lru_cache",
            policy=CachePolicy(max_size=3, eviction_policy=EvictionPolicy.LRU),
        )
        return LRUCache(config)

    def test_lru_eviction(self, lru_cache):
        """Test LRU eviction policy."""
        # Add 3 values
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")
        lru_cache.set("key3", "value3")

        # Access key1 to make it recently used
        lru_cache.get("key1")

        # Add 4th value - should evict key2 (least recently used)
        lru_cache.set("key4", "value4")

        # key1 should still be there (recently accessed)
        assert lru_cache.exists("key1") is True
        # key2 should be evicted
        assert lru_cache.exists("key2") is False
        # key3 and key4 should be there
        assert lru_cache.exists("key3") is True
        assert lru_cache.exists("key4") is True


class TestLFUCache:
    """Test LFU cache implementation."""

    @pytest.fixture
    def lfu_cache(self):
        """Create LFU cache."""
        config = CacheConfig(
            name="test_lfu_cache",
            policy=CachePolicy(max_size=3, eviction_policy=EvictionPolicy.LFU),
        )
        return LFUCache(config)

    def test_lfu_eviction(self, lfu_cache):
        """Test LFU eviction policy."""
        # Add 3 values
        lfu_cache.set("key1", "value1")
        lfu_cache.set("key2", "value2")
        lfu_cache.set("key3", "value3")

        # Access key1 multiple times to increase frequency
        lfu_cache.get("key1")
        lfu_cache.get("key1")
        lfu_cache.get("key1")

        # Access key2 once
        lfu_cache.get("key2")

        # Add 4th value - should evict key3 (least frequently used)
        lfu_cache.set("key4", "value4")

        # key1 should still be there (most frequently used)
        assert lfu_cache.exists("key1") is True
        # key2 should still be there (accessed once)
        assert lfu_cache.exists("key2") is True
        # key3 should be evicted (never accessed)
        assert lfu_cache.exists("key3") is False
        # key4 should be there
        assert lfu_cache.exists("key4") is True


class TestTTLCache:
    """Test TTL cache implementation."""

    @pytest.fixture
    def ttl_cache(self):
        """Create TTL cache."""
        config = CacheConfig(name="test_ttl_cache")
        return TTLCache(config, default_ttl=0.1)

    def test_ttl_expiration(self, ttl_cache):
        """Test TTL expiration."""
        # Set value
        ttl_cache.set("key1", "value1")

        # Should be available immediately
        assert ttl_cache.exists("key1") is True
        assert ttl_cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert ttl_cache.exists("key1") is False
        with pytest.raises(CacheError):
            ttl_cache.get("key1")

    def test_ttl_override(self, ttl_cache):
        """Test TTL override."""
        # Set value with custom TTL
        ttl_cache.set("key1", "value1", ttl=0.2)

        # Should be available
        assert ttl_cache.exists("key1") is True

        # Wait for default TTL but not custom TTL
        time.sleep(0.15)

        # Should still be available
        assert ttl_cache.exists("key1") is True

        # Wait for custom TTL
        time.sleep(0.1)

        # Should be expired
        assert ttl_cache.exists("key1") is False


class TestCacheManager:
    """Test cache manager functionality."""

    @pytest.fixture
    def cache_hierarchy(self):
        """Create cache hierarchy."""
        return CacheHierarchy(
            levels=[CacheLevel.L1, CacheLevel.L2],
            promote_on_hit=True,
            demote_on_evict=True,
        )

    @pytest.fixture
    def cache_manager(self, cache_hierarchy):
        """Create cache manager."""
        return CacheManager(cache_hierarchy)

    @pytest.fixture
    def l1_cache(self):
        """Create L1 cache."""
        config = CacheConfig(name="l1_cache", level=CacheLevel.L1)
        return MemoryCache(config)

    @pytest.fixture
    def l2_cache(self):
        """Create L2 cache."""
        config = CacheConfig(name="l2_cache", level=CacheLevel.L2)
        return MemoryCache(config)

    def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization."""
        assert cache_manager.hierarchy is not None
        assert len(cache_manager._caches) == 0

    def test_add_remove_cache(self, cache_manager, l1_cache, l2_cache):
        """Test adding and removing caches."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        assert len(cache_manager._caches) == 2
        assert cache_manager.get_cache(CacheLevel.L1) is l1_cache
        assert cache_manager.get_cache(CacheLevel.L2) is l2_cache

        # Remove cache
        cache_manager.remove_cache(CacheLevel.L1)
        assert len(cache_manager._caches) == 1
        assert cache_manager.get_cache(CacheLevel.L1) is None

    def test_hierarchy_get_set(self, cache_manager, l1_cache, l2_cache):
        """Test hierarchy get/set operations."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        # Set value in hierarchy
        cache_manager.set("key1", "value1")

        # Should be in both caches
        assert l1_cache.exists("key1") is True
        assert l2_cache.exists("key1") is True

        # Get value from hierarchy
        value = cache_manager.get("key1")
        assert value == "value1"

    def test_hierarchy_miss(self, cache_manager, l1_cache, l2_cache):
        """Test hierarchy miss."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        # Try to get non-existent key
        with pytest.raises(CacheError):
            cache_manager.get("nonexistent_key")

    def test_hierarchy_delete(self, cache_manager, l1_cache, l2_cache):
        """Test hierarchy delete."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        # Set value
        cache_manager.set("key1", "value1")

        # Delete from hierarchy
        result = cache_manager.delete("key1")
        assert result is True

        # Should be deleted from both caches
        assert l1_cache.exists("key1") is False
        assert l2_cache.exists("key1") is False

    def test_hierarchy_clear(self, cache_manager, l1_cache, l2_cache):
        """Test hierarchy clear."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        # Set values
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        # Clear hierarchy
        cache_manager.clear()

        # Should be cleared from both caches
        assert l1_cache.size() == 0
        assert l2_cache.size() == 0

    def test_hierarchy_stats(self, cache_manager, l1_cache, l2_cache):
        """Test hierarchy statistics."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        # Set and get values
        cache_manager.set("key1", "value1")
        cache_manager.get("key1")

        # Get hierarchy stats
        stats = cache_manager.get_combined_stats()
        assert stats.total_requests > 0

    def test_hierarchy_health_check(self, cache_manager, l1_cache, l2_cache):
        """Test hierarchy health check."""
        # Add caches
        cache_manager.add_cache(CacheLevel.L1, l1_cache)
        cache_manager.add_cache(CacheLevel.L2, l2_cache)

        # Perform health check
        health = cache_manager.health_check()
        assert health["status"] == "healthy"
        assert "levels" in health
        assert "combined_stats" in health

    def test_cache_manager_context_manager(self, cache_manager):
        """Test cache manager as context manager."""
        with cache_manager as cm:
            assert cm is cache_manager

        # Should not raise exception
        assert True


class TestCacheAnalytics:
    """Test cache analytics functionality."""

    @pytest.fixture
    def cache_backend(self):
        """Create cache backend."""
        config = CacheConfig(name="test_analytics_cache")
        return MemoryCache(config)

    @pytest.fixture
    def cache_analytics(self, cache_backend):
        """Create cache analytics."""
        return CacheAnalytics(cache_backend)

    def test_analytics_initialization(self, cache_analytics):
        """Test analytics initialization."""
        assert cache_analytics.cache_backend is not None
        assert cache_analytics.metrics is not None
        assert cache_analytics.profiler is not None

    def test_analytics_monitoring(self, cache_analytics):
        """Test analytics monitoring."""
        # Start monitoring
        cache_analytics.start_monitoring()
        assert cache_analytics._running is True

        # Stop monitoring
        cache_analytics.stop_monitoring()
        assert cache_analytics._running is False

    def test_analytics_record_operation(self, cache_analytics):
        """Test recording operations."""
        # Record operations
        cache_analytics.record_operation("get", 0.001, True)
        cache_analytics.record_operation("get", 0.002, False)
        cache_analytics.record_operation("set", 0.003, True)

        # Check metrics
        assert cache_analytics.metrics.hits == 1
        assert cache_analytics.metrics.misses == 1
        assert cache_analytics.metrics.total_requests == 2

    def test_analytics_performance_report(self, cache_analytics):
        """Test performance report generation."""
        # Record some operations
        cache_analytics.record_operation("get", 0.001, True)
        cache_analytics.record_operation("set", 0.002, True)

        # Generate report
        report = cache_analytics.get_performance_report()

        assert "cache_name" in report
        assert "cache_level" in report
        assert "metrics" in report
        assert "operation_stats" in report
        assert "slow_operations" in report
        assert "historical_trends" in report

    def test_analytics_recommendations(self, cache_analytics):
        """Test analytics recommendations."""
        # Get recommendations
        recommendations = cache_analytics.get_recommendations()

        assert isinstance(recommendations, list)

    def test_analytics_export(self, cache_analytics):
        """Test analytics export."""
        # Record some operations
        cache_analytics.record_operation("get", 0.001, True)

        # Export as JSON
        json_export = cache_analytics.export_metrics("json")
        assert isinstance(json_export, str)

        # Export as CSV
        csv_export = cache_analytics.export_metrics("csv")
        assert isinstance(csv_export, str)

    def test_analytics_reset(self, cache_analytics):
        """Test analytics reset."""
        # Record some operations
        cache_analytics.record_operation("get", 0.001, True)

        # Reset analytics
        cache_analytics.reset_analytics()

        # Check that data is reset
        assert cache_analytics.metrics.hits == 0
        assert cache_analytics.metrics.misses == 0
        assert cache_analytics.metrics.total_requests == 0

    def test_analytics_context_manager(self, cache_analytics):
        """Test analytics as context manager."""
        with cache_analytics as analytics:
            assert analytics is cache_analytics
            assert analytics._running is True

        # Should be stopped after context exit
        assert cache_analytics._running is False


class TestCacheWarmer:
    """Test cache warming functionality."""

    @pytest.fixture
    def cache_backend(self):
        """Create cache backend."""
        config = CacheConfig(name="test_warming_cache")
        return MemoryCache(config)

    @pytest.fixture
    def warming_config(self):
        """Create warming configuration."""
        return WarmingConfig(
            strategy=WarmingStrategy.MANUAL,
            enabled=True,
            warming_batch_size=10,
            warming_timeout=5.0,
        )

    @pytest.fixture
    def cache_warmer(self, cache_backend, warming_config):
        """Create cache warmer."""
        return CacheWarmer(cache_backend, warming_config)

    @pytest.fixture
    def mock_data_source(self):
        """Create mock data source."""
        data_source = Mock(spec=WarmingDataSource)
        data_source.get_data.return_value = {"key1": "value1", "key2": "value2"}
        data_source.get_frequent_keys.return_value = ["key1", "key2"]
        data_source.get_related_keys.return_value = ["key4", "key5"]
        return data_source

    def test_warmer_initialization(self, cache_warmer):
        """Test warmer initialization."""
        assert cache_warmer.cache_backend is not None
        assert cache_warmer.config is not None
        assert cache_warmer.data_source is None

    def test_warmer_set_data_source(self, cache_warmer, mock_data_source):
        """Test setting data source."""
        cache_warmer.set_data_source(mock_data_source)
        assert cache_warmer.data_source is mock_data_source

    def test_warmer_manual_warming(self, cache_warmer, mock_data_source):
        """Test manual warming."""
        cache_warmer.set_data_source(mock_data_source)

        # Warm cache with specific keys
        result = cache_warmer.warm_cache(["key1", "key2"])

        assert result.strategy == WarmingStrategy.MANUAL
        assert result.keys_warmed == 2
        assert result.success_count == 2
        assert result.failure_count == 0
        assert result.duration > 0

        # Check that values are in cache
        assert cache_warmer.cache_backend.exists("key1") is True
        assert cache_warmer.cache_backend.exists("key2") is True

    def test_warmer_no_data_source(self, cache_warmer):
        """Test warming without data source."""
        result = cache_warmer.warm_cache(["key1", "key2"])

        assert result.keys_warmed == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert len(result.errors) > 0

    def test_warmer_related_keys(self, cache_warmer, mock_data_source):
        """Test warming related keys."""
        cache_warmer.set_data_source(mock_data_source)

        # Warm related keys
        result = cache_warmer.warm_related_keys("key1", limit=5)

        assert result.strategy == WarmingStrategy.MANUAL
        assert result.keys_warmed == 2  # key4, key5
        assert result.success_count == 2

    def test_warmer_stats(self, cache_warmer, mock_data_source):
        """Test warming statistics."""
        cache_warmer.set_data_source(mock_data_source)

        # Perform warming
        cache_warmer.warm_cache(["key1", "key2"])

        # Get stats
        stats = cache_warmer.get_warming_stats()

        assert stats["total_warming_operations"] == 1
        assert stats["total_keys_warmed"] == 2
        assert stats["total_success_count"] == 2
        assert stats["total_failure_count"] == 0
        assert stats["success_rate"] == 1.0

    def test_warmer_recommendations(self, cache_warmer):
        """Test warming recommendations."""
        recommendations = cache_warmer.get_warming_recommendations()

        assert isinstance(recommendations, list)
        assert (
            len(recommendations) > 0
        )  # Should have recommendation about no data source

    def test_warmer_context_manager(self, cache_warmer):
        """Test warmer as context manager."""
        with cache_warmer as warmer:
            assert warmer is cache_warmer
            assert warmer._running is True

        # Should be stopped after context exit
        assert cache_warmer._running is False


class TestWarmingDataSource:
    """Test warming data source implementations."""

    def test_database_data_source(self):
        """Test database data source."""
        # Mock database connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("key1", "value1"), ("key2", "value2")]

        # Create data source
        data_source = DatabaseWarmingDataSource(
            mock_connection, "test_table", "key_col", "value_col"
        )

        # Test get_data
        data = data_source.get_data(["key1", "key2"])
        assert data == {"key1": "value1", "key2": "value2"}

        # Test get_frequent_keys
        frequent_keys = data_source.get_frequent_keys(10)
        assert isinstance(frequent_keys, list)

        # Test get_related_keys
        related_keys = data_source.get_related_keys("key1", 5)
        assert isinstance(related_keys, list)

    def test_file_data_source(self):
        """Test file data source."""
        import json
        import tempfile

        # Create temporary file with test data
        test_data = [
            {"key": "key1", "value": "value1"},
            {"key": "key2", "value": "value2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Create data source
            data_source = FileWarmingDataSource(temp_file, "key", "value")

            # Test get_data
            data = data_source.get_data(["key1", "key2"])
            assert data == {"key1": "value1", "key2": "value2"}

            # Test get_frequent_keys
            frequent_keys = data_source.get_frequent_keys(10)
            assert isinstance(frequent_keys, list)

            # Test get_related_keys
            related_keys = data_source.get_related_keys("key1", 5)
            assert isinstance(related_keys, list)

        finally:
            import os

            os.unlink(temp_file)
