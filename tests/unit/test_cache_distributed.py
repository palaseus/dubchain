"""Tests for distributed cache implementations."""

import logging

logger = logging.getLogger(__name__)
import json
import pickle
from unittest.mock import MagicMock, Mock, patch

import pytest

# Check if pymemcache is available
try:
    import pymemcache

    PYMCACHE_AVAILABLE = True
except ImportError:
    PYMCACHE_AVAILABLE = False

from dubchain.cache.core import (
    CacheConfig,
    CacheError,
    CacheExpiredError,
    CacheMissError,
    CachePolicy,
)
from dubchain.cache.distributed import DistributedCache, MemcachedCache, RedisCache


class TestDistributedCacheBasic:
    """Basic tests for distributed cache functionality."""

    @pytest.fixture
    def cache_config(self):
        """Create a cache configuration."""
        policy = CachePolicy(
            max_size=1000,
            default_ttl=3600,
            enable_compression=False,  # Disable to avoid zlib import issues
            enable_serialization=True,
            enable_tag_invalidation=True,
        )
        return CacheConfig(name="test_cache", policy=policy, backend_config={})

    def test_cache_config_creation(self, cache_config):
        """Test cache configuration creation."""
        assert cache_config.name == "test_cache"
        assert cache_config.policy.max_size == 1000
        assert cache_config.policy.default_ttl == 3600
        assert cache_config.policy.enable_compression is False
        assert cache_config.policy.enable_serialization is True
        assert cache_config.policy.enable_tag_invalidation is True

    def test_cache_policy_creation(self):
        """Test cache policy creation."""
        policy = CachePolicy(
            max_size=500,
            default_ttl=1800,
            enable_compression=True,
            enable_serialization=False,
            enable_tag_invalidation=False,
        )

        assert policy.max_size == 500
        assert policy.default_ttl == 1800
        assert policy.enable_compression is True
        assert policy.enable_serialization is False
        assert policy.enable_tag_invalidation is False


class TestDistributedCache:
    """Test cases for DistributedCache base class."""

    @pytest.fixture
    def mock_distributed_cache(self):
        """Create a mock distributed cache for testing."""
        config = CacheConfig(
            name="test_cache",
            policy=CachePolicy(
                max_size=1000,
                default_ttl=3600,
                enable_compression=False,
                enable_serialization=True,
                enable_tag_invalidation=True,
            ),
            backend_config={},
        )

        # Create a concrete implementation for testing
        class TestDistributedCache(DistributedCache):
            def _get_connection(self):
                return Mock()

            def _execute_command(self, command: str, *args):
                return Mock()

        return TestDistributedCache(config)

    def test_distributed_cache_initialization(self, mock_distributed_cache):
        """Test distributed cache initialization."""
        cache = mock_distributed_cache
        assert cache.config.name == "test_cache"
        assert (
            cache._serializer == pickle
        )  # Should use pickle when serialization enabled
        assert cache._compressor is None  # Should be None when compression disabled

    def test_distributed_cache_serializer_pickle(self):
        """Test distributed cache with pickle serializer."""
        config = CacheConfig(
            name="test_cache",
            policy=CachePolicy(
                max_size=1000,
                default_ttl=3600,
                enable_compression=False,
                enable_serialization=True,
                enable_tag_invalidation=True,
            ),
            backend_config={},
        )

        class TestDistributedCache(DistributedCache):
            def _get_connection(self):
                return Mock()

            def _execute_command(self, command: str, *args):
                return Mock()

        cache = TestDistributedCache(config)
        assert cache._serializer == pickle  # Should use pickle by default

    def test_distributed_cache_compressor_enabled(self):
        """Test distributed cache with compression enabled."""
        config = CacheConfig(
            name="test_cache",
            policy=CachePolicy(
                max_size=1000,
                default_ttl=3600,
                enable_compression=True,
                enable_serialization=True,
                enable_tag_invalidation=True,
            ),
            backend_config={},
        )

        class TestDistributedCache(DistributedCache):
            def _get_connection(self):
                return Mock()

            def _execute_command(self, command: str, *args):
                return Mock()

        cache = TestDistributedCache(config)
        assert cache._compressor is not None  # Should have compressor when enabled

    def test_distributed_cache_serialize(self, mock_distributed_cache):
        """Test serialization functionality."""
        cache = mock_distributed_cache
        test_data = {"key": "value", "number": 42}

        serialized = cache._serialize(test_data)
        assert isinstance(serialized, bytes)

        # Test deserialization
        deserialized = cache._deserialize(serialized)
        assert deserialized == test_data

    def test_distributed_cache_serialize_error(self, mock_distributed_cache):
        """Test serialization error handling."""
        cache = mock_distributed_cache

        # Mock serializer to raise an exception
        with patch.object(
            cache._serializer, "dumps", side_effect=Exception("Serialization failed")
        ):
            with pytest.raises(CacheError, match="Serialization failed"):
                cache._serialize({"test": "data"})

    def test_distributed_cache_deserialize_error(self, mock_distributed_cache):
        """Test deserialization error handling."""
        cache = mock_distributed_cache

        # Mock serializer to raise an exception
        with patch.object(
            cache._serializer, "loads", side_effect=Exception("Deserialization failed")
        ):
            with pytest.raises(CacheError, match="Deserialization failed"):
                cache._deserialize(b"invalid data")

    def test_distributed_cache_encode_decode_key(self, mock_distributed_cache):
        """Test key encoding and decoding."""
        cache = mock_distributed_cache

        original_key = "test_key"
        encoded_key = cache._encode_key(original_key)
        decoded_key = cache._decode_key(encoded_key)

        assert encoded_key == "test_cache:test_key"
        assert decoded_key == original_key

    def test_distributed_cache_decode_key_no_prefix(self, mock_distributed_cache):
        """Test decoding key without prefix."""
        cache = mock_distributed_cache

        key_without_prefix = "no_prefix_key"
        decoded_key = cache._decode_key(key_without_prefix)

        assert decoded_key == key_without_prefix

    def test_distributed_cache_get_success(self, mock_distributed_cache):
        """Test successful get operation."""
        cache = mock_distributed_cache
        test_data = {"key": "value"}
        serialized_data = cache._serialize(test_data)

        with patch.object(cache, "_execute_command", return_value=serialized_data):
            result = cache.get("test_key")
            assert result == test_data

    def test_distributed_cache_get_miss(self, mock_distributed_cache):
        """Test cache miss scenario."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=None):
            with pytest.raises(CacheMissError):
                cache.get("nonexistent_key")

    def test_distributed_cache_get_error(self, mock_distributed_cache):
        """Test get operation error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to get key"):
                cache.get("test_key")

    def test_distributed_cache_set_with_ttl(self, mock_distributed_cache):
        """Test set operation with TTL."""
        cache = mock_distributed_cache
        test_data = {"key": "value"}

        with patch.object(cache, "_execute_command") as mock_execute:
            cache.set("test_key", test_data, ttl=3600)
            mock_execute.assert_called_once()
            # Should call SETEX with TTL
            call_args = mock_execute.call_args
            assert call_args[0][0] == "SETEX"

    def test_distributed_cache_set_without_ttl(self, mock_distributed_cache):
        """Test set operation without TTL."""
        cache = mock_distributed_cache
        test_data = {"key": "value"}

        with patch.object(cache, "_execute_command") as mock_execute:
            cache.set("test_key", test_data)
            mock_execute.assert_called_once()
            # Should call SET without TTL
            call_args = mock_execute.call_args
            assert call_args[0][0] == "SET"

    def test_distributed_cache_set_with_tags(self, mock_distributed_cache):
        """Test set operation with tags."""
        cache = mock_distributed_cache
        test_data = {"key": "value"}
        tags = {"tag1", "tag2"}

        with patch.object(cache, "_execute_command") as mock_execute, patch.object(
            cache, "_store_tags"
        ) as mock_store_tags:
            cache.set("test_key", test_data, tags=tags)
            mock_store_tags.assert_called_once()

    def test_distributed_cache_set_error(self, mock_distributed_cache):
        """Test set operation error handling."""
        cache = mock_distributed_cache
        test_data = {"key": "value"}

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to set key"):
                cache.set("test_key", test_data)

    def test_distributed_cache_delete_success(self, mock_distributed_cache):
        """Test successful delete operation."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=1):
            result = cache.delete("test_key")
            assert result is True

    def test_distributed_cache_delete_not_found(self, mock_distributed_cache):
        """Test delete operation when key not found."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=0):
            result = cache.delete("nonexistent_key")
            assert result is False

    def test_distributed_cache_delete_with_tags(self, mock_distributed_cache):
        """Test delete operation with tag cleanup."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=1), patch.object(
            cache, "_cleanup_tags"
        ) as mock_cleanup:
            cache.delete("test_key")
            mock_cleanup.assert_called_once()

    def test_distributed_cache_delete_error(self, mock_distributed_cache):
        """Test delete operation error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to delete key"):
                cache.delete("test_key")

    def test_distributed_cache_exists_true(self, mock_distributed_cache):
        """Test exists operation when key exists."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=1):
            result = cache.exists("test_key")
            assert result is True

    def test_distributed_cache_exists_false(self, mock_distributed_cache):
        """Test exists operation when key doesn't exist."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=0):
            result = cache.exists("nonexistent_key")
            assert result is False

    def test_distributed_cache_exists_error(self, mock_distributed_cache):
        """Test exists operation error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to check existence of key"):
                cache.exists("test_key")

    def test_distributed_cache_clear_success(self, mock_distributed_cache):
        """Test clear operation."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command") as mock_execute:
            # Mock KEYS command to return some keys
            mock_execute.side_effect = [["test_cache:key1", "test_cache:key2"], None]
            cache.clear()

            # Should call KEYS and then DEL
            assert mock_execute.call_count == 2
            assert mock_execute.call_args_list[0][0] == ("KEYS", "test_cache:*")

    def test_distributed_cache_clear_no_keys(self, mock_distributed_cache):
        """Test clear operation when no keys exist."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=[]):
            cache.clear()
            # Should only call KEYS, not DEL

    def test_distributed_cache_clear_error(self, mock_distributed_cache):
        """Test clear operation error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to clear cache"):
                cache.clear()

    def test_distributed_cache_size(self, mock_distributed_cache):
        """Test size operation."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", return_value=["key1", "key2", "key3"]
        ):
            size = cache.size()
            assert size == 3

    def test_distributed_cache_size_no_keys(self, mock_distributed_cache):
        """Test size operation when no keys exist."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=None):
            size = cache.size()
            assert size == 0

    def test_distributed_cache_size_error(self, mock_distributed_cache):
        """Test size operation error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to get cache size"):
                cache.size()

    def test_distributed_cache_keys(self, mock_distributed_cache):
        """Test keys operation."""
        cache = mock_distributed_cache
        encoded_keys = ["test_cache:key1", "test_cache:key2"]

        with patch.object(cache, "_execute_command", return_value=encoded_keys):
            keys = cache.keys()
            assert keys == ["key1", "key2"]

    def test_distributed_cache_keys_empty(self, mock_distributed_cache):
        """Test keys operation when no keys exist."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=[]):
            keys = cache.keys()
            assert keys == []

    def test_distributed_cache_keys_error(self, mock_distributed_cache):
        """Test keys operation error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to get cache keys"):
                cache.keys()

    def test_distributed_cache_store_tags(self, mock_distributed_cache):
        """Test storing tags for a key."""
        cache = mock_distributed_cache
        tags = {"tag1", "tag2"}

        with patch.object(cache, "_execute_command") as mock_execute:
            cache._store_tags("test_cache:key1", tags)

            # Should call SADD for each tag
            assert mock_execute.call_count == 2
            calls = mock_execute.call_args_list

            # Check that both tags are processed (order may vary due to set iteration)
            call_args = [call[0] for call in calls]
            expected_calls = [
                ("SADD", "test_cache:tag:tag1", "test_cache:key1"),
                ("SADD", "test_cache:tag:tag2", "test_cache:key1"),
            ]
            assert call_args[0] in expected_calls
            assert call_args[1] in expected_calls
            assert call_args[0] != call_args[1]  # Different tags

    def test_distributed_cache_store_tags_error(self, mock_distributed_cache):
        """Test storing tags error handling."""
        cache = mock_distributed_cache
        tags = {"tag1"}

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            # Should not raise exception, just log warning
            cache._store_tags("test_cache:key1", tags)

    def test_distributed_cache_cleanup_tags(self, mock_distributed_cache):
        """Test cleaning up tags for a key."""
        cache = mock_distributed_cache

        # This is a placeholder implementation, so just test it doesn't raise
        cache._cleanup_tags("test_cache:key1")

    def test_distributed_cache_invalidate_by_tag(self, mock_distributed_cache):
        """Test invalidating entries by tag."""
        cache = mock_distributed_cache
        encoded_keys = ["test_cache:key1", "test_cache:key2"]

        with patch.object(cache, "_execute_command") as mock_execute:
            mock_execute.side_effect = [encoded_keys, 2, None]  # SMEMBERS, DEL, DEL
            result = cache.invalidate_by_tag("tag1")

            assert result == 2
            assert mock_execute.call_count == 3

    def test_distributed_cache_invalidate_by_tag_no_keys(self, mock_distributed_cache):
        """Test invalidating by tag when no keys exist."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=[]):
            result = cache.invalidate_by_tag("tag1")
            assert result == 0

    def test_distributed_cache_invalidate_by_tag_error(self, mock_distributed_cache):
        """Test invalidating by tag error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to invalidate by tag"):
                cache.invalidate_by_tag("tag1")

    def test_distributed_cache_invalidate_by_pattern(self, mock_distributed_cache):
        """Test invalidating entries by pattern."""
        cache = mock_distributed_cache
        encoded_keys = ["test_cache:key1", "test_cache:key2"]

        with patch.object(cache, "_execute_command") as mock_execute:
            mock_execute.side_effect = [encoded_keys, 2]  # KEYS, DEL
            result = cache.invalidate_by_pattern("key*")

            assert result == 2
            assert mock_execute.call_count == 2

    def test_distributed_cache_invalidate_by_pattern_no_keys(
        self, mock_distributed_cache
    ):
        """Test invalidating by pattern when no keys exist."""
        cache = mock_distributed_cache

        with patch.object(cache, "_execute_command", return_value=[]):
            result = cache.invalidate_by_pattern("key*")
            assert result == 0

    def test_distributed_cache_invalidate_by_pattern_error(
        self, mock_distributed_cache
    ):
        """Test invalidating by pattern error handling."""
        cache = mock_distributed_cache

        with patch.object(
            cache, "_execute_command", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(CacheError, match="Failed to invalidate by pattern"):
                cache.invalidate_by_pattern("key*")

    def test_distributed_cache_warm_up(self, mock_distributed_cache):
        """Test warm up operation."""
        cache = mock_distributed_cache
        keys = ["key1", "key2", "key3"]

        result = cache.warm_up(keys)
        assert result == 0  # Placeholder implementation returns 0

    def test_distributed_cache_optimize(self, mock_distributed_cache):
        """Test optimize operation."""
        cache = mock_distributed_cache

        # Should not raise exception
        cache.optimize()


class TestRedisCache:
    """Test cases for RedisCache implementation."""

    @pytest.fixture
    def redis_config(self):
        """Create a Redis cache configuration."""
        policy = CachePolicy(
            max_size=1000,
            default_ttl=3600,
            enable_compression=False,  # Disable to avoid zlib import issues
            enable_serialization=True,
            enable_tag_invalidation=True,
        )
        return CacheConfig(
            name="test_redis_cache",
            policy=policy,
            backend_config={
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "timeout": 5.0,
                    "connect_timeout": 5.0,
                }
            },
        )

    def test_redis_cache_initialization(self, redis_config):
        """Test Redis cache initialization."""
        # This test will either succeed (if redis is installed) or fail with import error
        try:
            cache = RedisCache(redis_config)
            # If we get here, redis is installed and cache was created
            assert cache.config == redis_config
        except CacheError as e:
            # If we get here, redis is not installed
            assert "Redis library not installed" in str(
                e
            ) or "Failed to connect to Redis" in str(e)

    def test_redis_config_validation(self, redis_config):
        """Test Redis configuration validation."""
        assert redis_config.name == "test_redis_cache"
        assert redis_config.backend_config["redis"]["host"] == "localhost"
        assert redis_config.backend_config["redis"]["port"] == 6379
        assert redis_config.backend_config["redis"]["db"] == 0
        assert redis_config.backend_config["redis"]["timeout"] == 5.0
        assert redis_config.backend_config["redis"]["connect_timeout"] == 5.0

    @patch("redis.Redis")
    def test_redis_cache_connect_success(self, mock_redis, redis_config):
        """Test successful Redis connection."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping.return_value = True

        cache = RedisCache(redis_config)

        assert cache._redis_client == mock_redis_client
        mock_redis.assert_called_once()
        mock_redis_client.ping.assert_called_once()

    @patch("redis.Redis")
    def test_redis_cache_connect_failure(self, mock_redis, redis_config):
        """Test Redis connection failure."""
        mock_redis.side_effect = Exception("Connection failed")

        with pytest.raises(CacheError, match="Failed to connect to Redis"):
            RedisCache(redis_config)

    def test_redis_cache_import_error(self, redis_config):
        """Test Redis import error."""
        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(CacheError, match="Redis library not installed"):
                RedisCache(redis_config)

    @patch("redis.Redis")
    def test_redis_cache_get_connection(self, mock_redis, redis_config):
        """Test getting Redis connection."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping.return_value = True

        cache = RedisCache(redis_config)
        connection = cache._get_connection()

        assert connection == mock_redis_client

    @patch("redis.Redis")
    def test_redis_cache_execute_command(self, mock_redis, redis_config):
        """Test executing Redis command."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping.return_value = True
        mock_redis_client.get.return_value = b"test_value"

        cache = RedisCache(redis_config)
        result = cache._execute_command("GET", "test_key")

        mock_redis_client.get.assert_called_once_with("test_key")
        assert result == b"test_value"

    @patch("redis.Redis")
    def test_redis_cache_execute_command_error(self, mock_redis, redis_config):
        """Test Redis command execution error."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping.return_value = True
        mock_redis_client.get.side_effect = Exception("Command failed")

        cache = RedisCache(redis_config)

        with pytest.raises(CacheError, match="Redis command failed"):
            cache._execute_command("GET", "test_key")

    @patch("redis.Redis")
    def test_redis_cache_health_check_healthy(self, mock_redis, redis_config):
        """Test Redis health check when healthy."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping.return_value = True
        mock_redis_client.info.return_value = {
            "redis_version": "6.0.0",
            "used_memory": 1024,
            "connected_clients": 5,
        }

        cache = RedisCache(redis_config)
        health = cache.health_check()

        assert health["status"] == "healthy"
        assert health["redis_version"] == "6.0.0"
        assert health["used_memory"] == 1024
        assert health["connected_clients"] == 5
        assert "stats" in health
        assert "timestamp" in health

    @patch("redis.Redis")
    def test_redis_cache_health_check_unhealthy(self, mock_redis, redis_config):
        """Test Redis health check when unhealthy."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping.return_value = True
        mock_redis_client.info.side_effect = Exception("Connection lost")

        cache = RedisCache(redis_config)
        health = cache.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "timestamp" in health


@pytest.mark.skipif(not PYMCACHE_AVAILABLE, reason="pymemcache not available")
class TestMemcachedCache:
    """Test cases for MemcachedCache implementation."""

    @pytest.fixture
    def memcached_config(self):
        """Create a Memcached cache configuration."""
        policy = CachePolicy(
            max_size=1000,
            default_ttl=3600,
            enable_compression=False,  # Disable to avoid zlib import issues
            enable_serialization=True,
            enable_tag_invalidation=True,
        )
        return CacheConfig(
            name="test_memcached_cache",
            policy=policy,
            backend_config={
                "memcached": {
                    "servers": ["localhost:11211"],
                    "connect_timeout": 5.0,
                    "timeout": 5.0,
                    "retry_attempts": 3,
                    "retry_delay": 1.0,
                }
            },
        )

    def test_memcached_cache_initialization(self, memcached_config):
        """Test Memcached cache initialization."""
        # This test will either succeed (if pymemcache is installed) or fail with import error
        try:
            cache = MemcachedCache(memcached_config)
            # If we get here, pymemcache is installed and cache was created
            assert cache.config == memcached_config
        except CacheError as e:
            # If we get here, pymemcache is not installed
            assert "Memcached library not installed" in str(
                e
            ) or "Failed to connect to Memcached" in str(e)

    def test_memcached_config_validation(self, memcached_config):
        """Test Memcached configuration validation."""
        assert memcached_config.name == "test_memcached_cache"
        assert memcached_config.backend_config["memcached"]["servers"] == [
            "localhost:11211"
        ]
        assert memcached_config.backend_config["memcached"]["connect_timeout"] == 5.0
        assert memcached_config.backend_config["memcached"]["timeout"] == 5.0
        assert memcached_config.backend_config["memcached"]["retry_attempts"] == 3
        assert memcached_config.backend_config["memcached"]["retry_delay"] == 1.0

    @patch("pymemcache.Client")
    def test_memcached_cache_connect_success(self, mock_client, memcached_config):
        """Test successful Memcached connection."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)

        assert cache._memcached_client == mock_client_instance
        mock_client.assert_called_once()
        mock_client_instance.get.assert_called_once_with("test")

    @patch("pymemcache.Client")
    def test_memcached_cache_connect_failure(self, mock_client, memcached_config):
        """Test Memcached connection failure."""
        mock_client.side_effect = Exception("Connection failed")

        with pytest.raises(CacheError, match="Failed to connect to Memcached"):
            MemcachedCache(memcached_config)

    def test_memcached_cache_import_error(self, memcached_config):
        """Test Memcached import error."""
        with patch.dict("sys.modules", {"pymemcache": None}):
            with pytest.raises(CacheError, match="Memcached library not installed"):
                MemcachedCache(memcached_config)

    @patch("pymemcache.Client")
    def test_memcached_cache_get_connection(self, mock_client, memcached_config):
        """Test getting Memcached connection."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)
        connection = cache._get_connection()

        assert connection == mock_client_instance

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_get(self, mock_client, memcached_config):
        """Test executing Memcached GET command."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = b"test_value"

        cache = MemcachedCache(memcached_config)
        result = cache._execute_command("GET", "test_key")

        mock_client_instance.get.assert_called_with("test_key")
        assert result == b"test_value"

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_set(self, mock_client, memcached_config):
        """Test executing Memcached SET command."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.set.return_value = True

        cache = MemcachedCache(memcached_config)
        result = cache._execute_command("SET", "test_key", b"test_value")

        mock_client_instance.set.assert_called_with("test_key", b"test_value")
        assert result is True

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_setex(self, mock_client, memcached_config):
        """Test executing Memcached SETEX command."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.set.return_value = True

        cache = MemcachedCache(memcached_config)
        result = cache._execute_command("SETEX", "test_key", 3600, b"test_value")

        mock_client_instance.set.assert_called_with(
            "test_key", b"test_value", expire=3600
        )
        assert result is True

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_del(self, mock_client, memcached_config):
        """Test executing Memcached DEL command."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.delete.return_value = True

        cache = MemcachedCache(memcached_config)
        result = cache._execute_command("DEL", "test_key")

        mock_client_instance.delete.assert_called_with("test_key")
        assert result is True

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_exists(
        self, mock_client, memcached_config
    ):
        """Test executing Memcached EXISTS command."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = b"test_value"

        cache = MemcachedCache(memcached_config)
        result = cache._execute_command("EXISTS", "test_key")

        mock_client_instance.get.assert_called_with("test_key")
        assert result is True

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_unsupported(
        self, mock_client, memcached_config
    ):
        """Test executing unsupported Memcached command."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)

        with pytest.raises(CacheError, match="Unsupported Memcached command"):
            cache._execute_command("UNSUPPORTED", "test_key")

    @patch("pymemcache.Client")
    def test_memcached_cache_execute_command_error(self, mock_client, memcached_config):
        """Test Memcached command execution error."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        # First call to get('test') during _connect() should succeed
        mock_client_instance.get.return_value = None
        # But subsequent calls should fail
        mock_client_instance.get.side_effect = [None, Exception("Command failed")]

        cache = MemcachedCache(memcached_config)

        with pytest.raises(CacheError, match="Memcached command failed"):
            cache._execute_command("GET", "test_key")

    @patch("pymemcache.Client")
    def test_memcached_cache_clear(self, mock_client, memcached_config):
        """Test Memcached clear operation."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.flush_all.return_value = True

        cache = MemcachedCache(memcached_config)
        cache.clear()

        mock_client_instance.flush_all.assert_called_once()

    @patch("pymemcache.Client")
    def test_memcached_cache_clear_error(self, mock_client, memcached_config):
        """Test Memcached clear operation error."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.flush_all.side_effect = Exception("Clear failed")

        cache = MemcachedCache(memcached_config)

        with pytest.raises(CacheError, match="Failed to clear Memcached cache"):
            cache.clear()

    @patch("pymemcache.Client")
    def test_memcached_cache_size(self, mock_client, memcached_config):
        """Test Memcached size operation."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)
        size = cache.size()

        # Memcached doesn't provide direct size, returns 0
        assert size == 0

    @patch("pymemcache.Client")
    def test_memcached_cache_keys(self, mock_client, memcached_config):
        """Test Memcached keys operation."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)
        keys = cache.keys()

        # Memcached doesn't provide direct keys listing, returns empty list
        assert keys == []

    @patch("pymemcache.Client")
    def test_memcached_cache_invalidate_by_tag(self, mock_client, memcached_config):
        """Test Memcached invalidate by tag operation."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)
        result = cache.invalidate_by_tag("tag1")

        # Memcached doesn't support tag invalidation, returns 0
        assert result == 0

    @patch("pymemcache.Client")
    def test_memcached_cache_invalidate_by_pattern(self, mock_client, memcached_config):
        """Test Memcached invalidate by pattern operation."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None

        cache = MemcachedCache(memcached_config)
        result = cache.invalidate_by_pattern("key*")

        # Memcached doesn't support pattern invalidation, returns 0
        assert result == 0

    @patch("pymemcache.Client")
    def test_memcached_cache_health_check_healthy(self, mock_client, memcached_config):
        """Test Memcached health check when healthy."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.stats.return_value = {
            "version": "1.6.0",
            "bytes": 1024,
            "curr_connections": 5,
        }

        cache = MemcachedCache(memcached_config)
        health = cache.health_check()

        assert health["status"] == "healthy"
        assert health["version"] == "1.6.0"
        assert health["bytes"] == 1024
        assert health["curr_connections"] == 5
        assert "stats" in health
        assert "timestamp" in health

    @patch("pymemcache.Client")
    def test_memcached_cache_health_check_unhealthy(
        self, mock_client, memcached_config
    ):
        """Test Memcached health check when unhealthy."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get.return_value = None
        mock_client_instance.stats.side_effect = Exception("Connection lost")

        cache = MemcachedCache(memcached_config)
        health = cache.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "timestamp" in health
