"""
Unit tests for cache analytics module.
"""

import logging

logger = logging.getLogger(__name__)
import json
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.cache.analytics import CacheAnalytics, CacheMetrics, CacheProfiler
from dubchain.cache.core import CacheBackend, CacheConfig, CacheLevel, CacheStats


class TestCacheMetrics:
    """Test CacheMetrics class."""

    def test_cache_metrics_creation(self):
        """Test creating cache metrics."""
        metrics = CacheMetrics()
        assert metrics.total_requests == 0
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.expirations == 0
        assert metrics.average_response_time == 0.0
        assert metrics.min_response_time == float("inf")
        assert metrics.max_response_time == 0.0
        assert len(metrics.response_time_percentiles) == 0
        assert metrics.current_size == 0
        assert metrics.max_size == 0
        assert metrics.size_utilization == 0.0
        assert metrics.memory_usage_bytes == 0
        assert metrics.max_memory_bytes == 0
        assert metrics.memory_utilization == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 0.0
        assert metrics.eviction_rate == 0.0
        assert metrics.expiration_rate == 0.0
        assert metrics.cache_efficiency == 0.0
        assert metrics.memory_efficiency == 0.0
        assert metrics.compression_ratio == 0.0
        assert metrics.prefetch_hit_rate == 0.0
        assert len(metrics.response_times) == 0
        assert len(metrics.hit_rates) == 0
        assert len(metrics.size_history) == 0

    def test_update_percentiles_empty(self):
        """Test updating percentiles with empty data."""
        metrics = CacheMetrics()
        metrics.update_percentiles()
        assert len(metrics.response_time_percentiles) == 0

    def test_update_percentiles_single_value(self):
        """Test updating percentiles with single value."""
        metrics = CacheMetrics()
        metrics.response_times = [1.0]
        metrics.update_percentiles()
        # Percentiles are set to 0.0 when there's only one value
        assert len(metrics.response_time_percentiles) == 4
        assert all(
            metrics.response_time_percentiles[p] == 0.0 for p in [50, 90, 95, 99]
        )

    def test_update_percentiles_multiple_values(self):
        """Test updating percentiles with multiple values."""
        metrics = CacheMetrics()
        metrics.response_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics.update_percentiles()

        assert 50 in metrics.response_time_percentiles
        assert 90 in metrics.response_time_percentiles
        assert 95 in metrics.response_time_percentiles
        assert 99 in metrics.response_time_percentiles

        # Check that percentiles are reasonable
        assert 0 <= metrics.response_time_percentiles[50] <= 5.0
        assert (
            0 <= metrics.response_time_percentiles[90] <= 6.0
        )  # Allow some margin for quantile calculation
        assert 0 <= metrics.response_time_percentiles[95] <= 6.0
        assert 0 <= metrics.response_time_percentiles[99] <= 6.0

    def test_add_response_time(self):
        """Test adding response time measurements."""
        metrics = CacheMetrics()

        # Add first response time
        metrics.add_response_time(1.5)
        assert len(metrics.response_times) == 1
        assert metrics.response_times[0] == 1.5
        assert metrics.average_response_time == 1.5
        assert metrics.min_response_time == 1.5
        assert metrics.max_response_time == 1.5

        # Add second response time
        metrics.add_response_time(2.5)
        assert len(metrics.response_times) == 2
        assert metrics.average_response_time == 2.0
        assert metrics.min_response_time == 1.5
        assert metrics.max_response_time == 2.5

    def test_add_response_time_limit(self):
        """Test response time list size limit."""
        metrics = CacheMetrics()

        # Add more than 1000 response times
        for i in range(1005):
            metrics.add_response_time(float(i))

        # Should keep only the last 1000
        assert len(metrics.response_times) == 1000
        assert metrics.response_times[0] == 5.0  # First kept value
        assert metrics.response_times[-1] == 1004.0  # Last value

    def test_add_hit_rate(self):
        """Test adding hit rate measurements."""
        metrics = CacheMetrics()

        metrics.add_hit_rate(0.8)
        assert len(metrics.hit_rates) == 1
        assert metrics.hit_rates[0] == 0.8

        metrics.add_hit_rate(0.9)
        assert len(metrics.hit_rates) == 2
        assert metrics.hit_rates[1] == 0.9

    def test_add_hit_rate_limit(self):
        """Test hit rate list size limit."""
        metrics = CacheMetrics()

        # Add more than 100 hit rates
        for i in range(105):
            metrics.add_hit_rate(float(i) / 100.0)

        # Should keep only the last 100
        assert len(metrics.hit_rates) == 100
        assert metrics.hit_rates[0] == 0.05  # First kept value
        assert metrics.hit_rates[-1] == 1.04  # Last value

    def test_add_size_measurement(self):
        """Test adding size measurements."""
        metrics = CacheMetrics()

        metrics.add_size_measurement(100)
        assert len(metrics.size_history) == 1
        assert metrics.size_history[0] == 100

        metrics.add_size_measurement(200)
        assert len(metrics.size_history) == 2
        assert metrics.size_history[1] == 200

    def test_add_size_measurement_limit(self):
        """Test size measurement list size limit."""
        metrics = CacheMetrics()

        # Add more than 100 size measurements
        for i in range(105):
            metrics.add_size_measurement(i)

        # Should keep only the last 100
        assert len(metrics.size_history) == 100
        assert metrics.size_history[0] == 5  # First kept value
        assert metrics.size_history[-1] == 104  # Last value


class TestCacheProfiler:
    """Test CacheProfiler class."""

    def test_cache_profiler_creation(self):
        """Test creating cache profiler."""
        profiler = CacheProfiler()
        assert profiler.enabled is True
        assert profiler.sample_rate == 0.1
        assert profiler.max_samples == 10000
        assert len(profiler.operation_times) == 0
        assert len(profiler.operation_counts) == 0
        assert len(profiler.slow_operations) == 0

    def test_start_profiling_enabled(self):
        """Test starting profiling when enabled."""
        profiler = CacheProfiler()
        profiler.enabled = True

        with patch("random.random", return_value=0.05):  # Below sample rate
            start_time = profiler.start_profiling("test_operation")
            assert start_time is not None
            assert isinstance(start_time, float)

    def test_start_profiling_disabled(self):
        """Test starting profiling when disabled."""
        profiler = CacheProfiler()
        profiler.enabled = False

        start_time = profiler.start_profiling("test_operation")
        assert start_time is None

    def test_start_profiling_above_sample_rate(self):
        """Test starting profiling when above sample rate."""
        profiler = CacheProfiler()
        profiler.enabled = True

        with patch("random.random", return_value=0.15):  # Above sample rate
            start_time = profiler.start_profiling("test_operation")
            assert start_time is None

    def test_end_profiling_no_start_time(self):
        """Test ending profiling with no start time."""
        profiler = CacheProfiler()

        # Should not raise any exceptions
        profiler.end_profiling("test_operation", None)
        assert len(profiler.operation_times) == 0
        assert len(profiler.operation_counts) == 0

    def test_end_profiling_with_start_time(self):
        """Test ending profiling with start time."""
        profiler = CacheProfiler()

        start_time = time.time() - 0.1  # 100ms ago
        profiler.end_profiling("test_operation", start_time)

        assert "test_operation" in profiler.operation_times
        assert len(profiler.operation_times["test_operation"]) == 1
        assert profiler.operation_times["test_operation"][0] >= 0.1
        assert profiler.operation_counts["test_operation"] == 1

    def test_end_profiling_slow_operation(self):
        """Test ending profiling for slow operation."""
        profiler = CacheProfiler()

        start_time = time.time() - 1.5  # 1.5 seconds ago (slow)
        profiler.end_profiling("slow_operation", start_time)

        assert len(profiler.slow_operations) == 1
        operation, duration, timestamp = profiler.slow_operations[0]
        assert operation == "slow_operation"
        assert duration >= 1.5
        assert isinstance(timestamp, str)

    def test_end_profiling_max_samples(self):
        """Test operation times max samples limit."""
        profiler = CacheProfiler()
        profiler.max_samples = 3

        # Add more than max_samples operations
        for i in range(5):
            start_time = time.time() - 0.1
            profiler.end_profiling("test_operation", start_time)

        # Should keep only the last max_samples
        assert len(profiler.operation_times["test_operation"]) == 3
        assert profiler.operation_counts["test_operation"] == 5

    def test_end_profiling_slow_operations_limit(self):
        """Test slow operations list size limit."""
        profiler = CacheProfiler()

        # Add more than 100 slow operations
        for i in range(105):
            start_time = time.time() - 1.5
            profiler.end_profiling(f"slow_operation_{i}", start_time)

        # Should keep only the last 100
        assert len(profiler.slow_operations) == 100
        assert profiler.slow_operations[0][0] == "slow_operation_5"  # First kept
        assert profiler.slow_operations[-1][0] == "slow_operation_104"  # Last

    def test_get_operation_stats_empty(self):
        """Test getting operation stats for non-existent operation."""
        profiler = CacheProfiler()
        stats = profiler.get_operation_stats("non_existent")
        assert stats == {}

    def test_get_operation_stats_single_value(self):
        """Test getting operation stats with single value."""
        profiler = CacheProfiler()

        start_time = time.time() - 0.1
        profiler.end_profiling("test_operation", start_time)

        stats = profiler.get_operation_stats("test_operation")

        assert stats["count"] == 1
        assert stats["average_time"] >= 0.1
        assert stats["min_time"] >= 0.1
        assert stats["max_time"] >= 0.1
        assert stats["median_time"] >= 0.1
        assert stats["std_dev"] == 0.0  # Single value has no std dev
        assert 50 in stats["percentiles"]
        assert 90 in stats["percentiles"]
        assert 95 in stats["percentiles"]
        assert 99 in stats["percentiles"]

    def test_get_operation_stats_multiple_values(self):
        """Test getting operation stats with multiple values."""
        profiler = CacheProfiler()

        # Add multiple operations
        for i in range(5):
            start_time = time.time() - (0.1 + i * 0.01)
            profiler.end_profiling("test_operation", start_time)

        stats = profiler.get_operation_stats("test_operation")

        assert stats["count"] == 5
        assert stats["average_time"] > 0
        assert stats["min_time"] > 0
        assert stats["max_time"] > 0
        assert stats["median_time"] > 0
        assert stats["std_dev"] >= 0.0
        # Allow some margin for quantile calculation
        assert all(
            0 <= stats["percentiles"][p] <= stats["max_time"] * 1.2
            for p in [50, 90, 95, 99]
        )

    def test_get_all_operation_stats(self):
        """Test getting all operation stats."""
        profiler = CacheProfiler()

        # Add operations for different operation types
        start_time = time.time() - 0.1
        profiler.end_profiling("get_operation", start_time)
        profiler.end_profiling("set_operation", start_time)

        all_stats = profiler.get_all_operation_stats()

        assert "get_operation" in all_stats
        assert "set_operation" in all_stats
        assert all_stats["get_operation"]["count"] == 1
        assert all_stats["set_operation"]["count"] == 1

    def test_get_slow_operations(self):
        """Test getting slow operations."""
        profiler = CacheProfiler()

        # Add a slow operation
        start_time = time.time() - 1.5
        profiler.end_profiling("slow_operation", start_time)

        slow_ops = profiler.get_slow_operations()
        assert len(slow_ops) == 1
        assert slow_ops[0][0] == "slow_operation"

    def test_reset(self):
        """Test resetting profiler data."""
        profiler = CacheProfiler()

        # Add some data
        start_time = time.time() - 0.1
        profiler.end_profiling("test_operation", start_time)

        assert len(profiler.operation_times) > 0
        assert len(profiler.operation_counts) > 0

        profiler.reset()

        assert len(profiler.operation_times) == 0
        assert len(profiler.operation_counts) == 0
        assert len(profiler.slow_operations) == 0


class TestCacheAnalytics:
    """Test CacheAnalytics class."""

    @pytest.fixture
    def mock_cache_backend(self):
        """Create a mock cache backend."""
        backend = Mock(spec=CacheBackend)
        backend.config = Mock()
        backend.config.name = "test_cache"
        backend.config.level = CacheLevel.L1

        # Mock cache stats
        cache_stats = Mock(spec=CacheStats)
        cache_stats.total_requests = 100
        cache_stats.hits = 80
        cache_stats.misses = 20
        cache_stats.evictions = 5
        cache_stats.expirations = 2
        cache_stats.average_execution_time = 0.05
        cache_stats.entry_count = 50
        cache_stats.memory_usage = 1024
        cache_stats.hit_rate = 0.8
        cache_stats.miss_rate = 0.2
        cache_stats.cache_efficiency = 0.9
        cache_stats.memory_efficiency = 0.85

        backend.get_stats.return_value = cache_stats
        return backend

    @pytest.fixture
    def cache_analytics(self, mock_cache_backend):
        """Create cache analytics instance."""
        return CacheAnalytics(mock_cache_backend)

    def test_cache_analytics_creation(self, cache_analytics, mock_cache_backend):
        """Test creating cache analytics."""
        assert cache_analytics.cache_backend == mock_cache_backend
        assert isinstance(cache_analytics.metrics, CacheMetrics)
        assert isinstance(cache_analytics.profiler, CacheProfiler)
        assert cache_analytics._monitoring_thread is None
        assert cache_analytics._running is False
        assert cache_analytics._monitoring_interval == 60.0
        assert len(cache_analytics._historical_data) == 0
        assert cache_analytics._max_historical_entries == 1000

    def test_start_monitoring(self, cache_analytics):
        """Test starting monitoring."""
        cache_analytics.start_monitoring()

        assert cache_analytics._running is True
        assert cache_analytics._monitoring_thread is not None
        assert cache_analytics._monitoring_thread.is_alive()

        # Clean up
        cache_analytics.stop_monitoring()

    def test_start_monitoring_already_running(self, cache_analytics):
        """Test starting monitoring when already running."""
        cache_analytics.start_monitoring()
        original_thread = cache_analytics._monitoring_thread

        # Try to start again
        cache_analytics.start_monitoring()

        # Should be the same thread
        assert cache_analytics._monitoring_thread == original_thread

        # Clean up
        cache_analytics.stop_monitoring()

    def test_stop_monitoring(self, cache_analytics):
        """Test stopping monitoring."""
        cache_analytics.start_monitoring()
        assert cache_analytics._running is True

        cache_analytics.stop_monitoring()
        assert cache_analytics._running is False

    def test_collect_metrics(self, cache_analytics, mock_cache_backend):
        """Test collecting metrics."""
        cache_analytics._collect_metrics()

        # Verify that get_stats was called
        mock_cache_backend.get_stats.assert_called_once()

        # Verify metrics were updated
        assert cache_analytics.metrics.total_requests == 100
        assert cache_analytics.metrics.hits == 80
        assert cache_analytics.metrics.misses == 20
        assert cache_analytics.metrics.evictions == 5
        assert cache_analytics.metrics.expirations == 2
        assert cache_analytics.metrics.average_response_time == 0.05
        assert cache_analytics.metrics.current_size == 50
        assert cache_analytics.metrics.memory_usage_bytes == 1024
        assert cache_analytics.metrics.hit_rate == 0.8
        assert cache_analytics.metrics.miss_rate == 0.2
        assert cache_analytics.metrics.cache_efficiency == 0.9
        assert cache_analytics.metrics.memory_efficiency == 0.85

    def test_store_historical_data(self, cache_analytics):
        """Test storing historical data."""
        cache_analytics._store_historical_data()

        assert len(cache_analytics._historical_data) == 1
        data_point = cache_analytics._historical_data[0]

        assert "timestamp" in data_point
        assert "hit_rate" in data_point
        assert "miss_rate" in data_point
        assert "current_size" in data_point
        assert "memory_usage" in data_point
        assert "average_response_time" in data_point
        assert "evictions" in data_point
        assert "expirations" in data_point

    def test_store_historical_data_limit(self, cache_analytics):
        """Test historical data size limit."""
        cache_analytics._max_historical_entries = 3

        # Add more than the limit
        for i in range(5):
            cache_analytics._store_historical_data()

        # Should keep only the last 3
        assert len(cache_analytics._historical_data) == 3

    def test_record_operation_get_hit(self, cache_analytics):
        """Test recording a successful get operation."""
        cache_analytics.record_operation("get", 0.05, success=True)

        assert cache_analytics.metrics.hits == 1
        assert cache_analytics.metrics.misses == 0
        assert cache_analytics.metrics.total_requests == 1
        assert len(cache_analytics.metrics.response_times) == 1
        assert cache_analytics.metrics.response_times[0] == 0.05

    def test_record_operation_get_miss(self, cache_analytics):
        """Test recording a failed get operation."""
        cache_analytics.record_operation("get", 0.03, success=False)

        assert cache_analytics.metrics.hits == 0
        assert cache_analytics.metrics.misses == 1
        assert cache_analytics.metrics.total_requests == 1
        assert len(cache_analytics.metrics.response_times) == 1
        assert cache_analytics.metrics.response_times[0] == 0.03

    def test_record_operation_evict(self, cache_analytics):
        """Test recording an evict operation."""
        cache_analytics.record_operation("evict", 0.01, success=True)

        assert cache_analytics.metrics.evictions == 1
        assert cache_analytics.metrics.hits == 0
        assert cache_analytics.metrics.misses == 0
        assert cache_analytics.metrics.total_requests == 0

    def test_record_operation_expire(self, cache_analytics):
        """Test recording an expire operation."""
        cache_analytics.record_operation("expire", 0.02, success=True)

        assert cache_analytics.metrics.expirations == 1
        assert cache_analytics.metrics.hits == 0
        assert cache_analytics.metrics.misses == 0
        assert cache_analytics.metrics.total_requests == 0

    def test_get_performance_report(self, cache_analytics, mock_cache_backend):
        """Test getting performance report."""
        # Update metrics from cache backend first
        cache_analytics._collect_metrics()

        # Add some historical data
        cache_analytics._store_historical_data()
        cache_analytics._store_historical_data()

        report = cache_analytics.get_performance_report()

        assert report["cache_name"] == "test_cache"
        assert report["cache_level"] == CacheLevel.L1.value
        assert "timestamp" in report
        assert "metrics" in report
        assert "operation_stats" in report
        assert "slow_operations" in report
        assert "historical_trends" in report

        # Check metrics
        metrics = report["metrics"]
        assert metrics["hit_rate"] == 0.8
        assert metrics["miss_rate"] == 0.2
        assert metrics["average_response_time"] == 0.05
        assert metrics["current_size"] == 50
        assert metrics["memory_usage"] == 1024
        assert metrics["cache_efficiency"] == 0.9
        assert metrics["memory_efficiency"] == 0.85

    def test_calculate_trend_increasing(self, cache_analytics):
        """Test calculating increasing trend."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = cache_analytics._calculate_trend(values)
        assert trend == "increasing"

    def test_calculate_trend_decreasing(self, cache_analytics):
        """Test calculating decreasing trend."""
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = cache_analytics._calculate_trend(values)
        assert trend == "decreasing"

    def test_calculate_trend_stable(self, cache_analytics):
        """Test calculating stable trend."""
        values = [1.0, 1.0, 1.0, 1.0, 1.0]
        trend = cache_analytics._calculate_trend(values)
        assert trend == "stable"

    def test_calculate_trend_insufficient_data(self, cache_analytics):
        """Test calculating trend with insufficient data."""
        values = [1.0]
        trend = cache_analytics._calculate_trend(values)
        assert trend == "stable"

    def test_get_historical_trends_insufficient_data(self, cache_analytics):
        """Test getting historical trends with insufficient data."""
        trends = cache_analytics._get_historical_trends()
        assert trends == {}

    def test_get_historical_trends_sufficient_data(self, cache_analytics):
        """Test getting historical trends with sufficient data."""
        # Add enough historical data
        for i in range(12):
            cache_analytics._store_historical_data()

        trends = cache_analytics._get_historical_trends()

        assert "hit_rate_trend" in trends
        assert "size_trend" in trends
        assert "response_time_trend" in trends
        assert "memory_usage_trend" in trends

        # All trends should be 'stable' since we're using the same data
        assert trends["hit_rate_trend"] == "stable"
        assert trends["size_trend"] == "stable"
        assert trends["response_time_trend"] == "stable"
        assert trends["memory_usage_trend"] == "stable"

    def test_get_recommendations_low_hit_rate(self, cache_analytics):
        """Test getting recommendations for low hit rate."""
        cache_analytics.metrics.hit_rate = 0.7  # Below 0.8 threshold

        recommendations = cache_analytics.get_recommendations()

        assert len(recommendations) >= 1
        assert any("cache size" in rec.lower() for rec in recommendations)

    def test_get_recommendations_high_memory_usage(self, cache_analytics):
        """Test getting recommendations for high memory usage."""
        cache_analytics.metrics.memory_utilization = 0.95  # Above 0.9 threshold

        recommendations = cache_analytics.get_recommendations()

        assert len(recommendations) >= 1
        assert any("memory" in rec.lower() for rec in recommendations)

    def test_get_recommendations_high_response_time(self, cache_analytics):
        """Test getting recommendations for high response time."""
        cache_analytics.metrics.average_response_time = 0.15  # Above 0.1 threshold

        recommendations = cache_analytics.get_recommendations()

        assert len(recommendations) >= 1
        assert any("response time" in rec.lower() for rec in recommendations)

    def test_get_recommendations_high_eviction_rate(self, cache_analytics):
        """Test getting recommendations for high eviction rate."""
        cache_analytics.metrics.eviction_rate = 0.15  # Above 0.1 threshold

        recommendations = cache_analytics.get_recommendations()

        assert len(recommendations) >= 1
        assert any("eviction" in rec.lower() for rec in recommendations)

    def test_get_recommendations_high_size_utilization(self, cache_analytics):
        """Test getting recommendations for high size utilization."""
        cache_analytics.metrics.size_utilization = 0.98  # Above 0.95 threshold

        recommendations = cache_analytics.get_recommendations()

        assert len(recommendations) >= 1
        assert any("size utilization" in rec.lower() for rec in recommendations)

    def test_export_metrics_json(self, cache_analytics):
        """Test exporting metrics as JSON."""
        json_data = cache_analytics.export_metrics("json")

        # Should be valid JSON
        parsed_data = json.loads(json_data)
        assert "cache_name" in parsed_data
        assert "metrics" in parsed_data
        assert "timestamp" in parsed_data

    def test_export_metrics_csv(self, cache_analytics):
        """Test exporting metrics as CSV."""
        # Add some historical data
        cache_analytics._store_historical_data()
        cache_analytics._store_historical_data()

        csv_data = cache_analytics.export_metrics("csv")

        # Should be valid CSV with header
        lines = csv_data.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row
        assert "timestamp" in lines[0]
        assert "hit_rate" in lines[0]
        assert "miss_rate" in lines[0]

    def test_export_metrics_unsupported_format(self, cache_analytics):
        """Test exporting metrics with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            cache_analytics.export_metrics("xml")

    def test_reset_analytics(self, cache_analytics):
        """Test resetting analytics data."""
        # Add some data
        cache_analytics.record_operation("get", 0.05, success=True)
        cache_analytics._store_historical_data()

        assert cache_analytics.metrics.hits == 1
        assert len(cache_analytics._historical_data) == 1

        cache_analytics.reset_analytics()

        assert cache_analytics.metrics.hits == 0
        assert len(cache_analytics._historical_data) == 0
        assert len(cache_analytics.profiler.operation_times) == 0

    def test_context_manager(self, cache_analytics):
        """Test using cache analytics as context manager."""
        with cache_analytics as analytics:
            assert analytics._running is True
            assert analytics._monitoring_thread is not None

        # Should be stopped after context exit
        assert analytics._running is False
