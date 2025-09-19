"""Test cases for network/performance.py module."""

import time
from unittest.mock import Mock, patch

import pytest

from dubchain.network.performance import (
    MetricType,
    NetworkPerformance,
    PerformanceConfig,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceOptimizer,
)


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.THROUGHPUT.value == "throughput"
        assert MetricType.BANDWIDTH.value == "bandwidth"
        assert MetricType.CONNECTION_COUNT.value == "connection_count"
        assert MetricType.MESSAGE_COUNT.value == "message_count"


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""

    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation."""
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY, value=100.5, timestamp=time.time()
        )

        assert metric.metric_type == MetricType.LATENCY
        assert metric.value == 100.5
        assert metric.peer_id is None
        assert metric.metadata == {}

    def test_performance_metric_with_peer_id(self):
        """Test PerformanceMetric creation with peer_id."""
        metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        assert metric.peer_id == "peer1"

    def test_performance_metric_with_metadata(self):
        """Test PerformanceMetric creation with metadata."""
        metadata = {"source": "test", "version": "1.0"}
        metric = PerformanceMetric(
            metric_type=MetricType.BANDWIDTH,
            value=1000.0,
            timestamp=time.time(),
            peer_id="peer1",
            metadata=metadata,
        )

        assert metric.metadata == metadata


class TestPerformanceConfig:
    """Test PerformanceConfig dataclass."""

    def test_performance_config_defaults(self):
        """Test PerformanceConfig with default values."""
        config = PerformanceConfig()

        assert config.monitoring_interval == 10.0
        assert config.metrics_retention == 3600.0
        assert config.enable_auto_optimization is True
        assert config.optimization_threshold == 0.8
        assert config.metadata == {}

    def test_performance_config_custom_values(self):
        """Test PerformanceConfig with custom values."""
        metadata = {"custom_setting": "value"}
        config = PerformanceConfig(
            monitoring_interval=5.0,
            metrics_retention=1800.0,
            enable_auto_optimization=False,
            optimization_threshold=0.9,
            metadata=metadata,
        )

        assert config.monitoring_interval == 5.0
        assert config.metrics_retention == 1800.0
        assert config.enable_auto_optimization is False
        assert config.optimization_threshold == 0.9
        assert config.metadata == metadata


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_performance_monitor_creation(self):
        """Test PerformanceMonitor creation."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        assert monitor.config == config
        assert monitor.metrics == []
        assert monitor.peer_metrics == {}

    def test_record_metric_without_peer_id(self):
        """Test record_metric without peer_id."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY, value=100.0, timestamp=time.time()
        )

        monitor.record_metric(metric)

        assert len(monitor.metrics) == 1
        assert monitor.metrics[0] == metric
        assert monitor.peer_metrics == {}

    def test_record_metric_with_peer_id(self):
        """Test record_metric with peer_id."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(metric)

        assert len(monitor.metrics) == 1
        assert monitor.metrics[0] == metric
        assert "peer1" in monitor.peer_metrics
        assert len(monitor.peer_metrics["peer1"]) == 1
        assert monitor.peer_metrics["peer1"][0] == metric

    def test_record_metric_multiple_peers(self):
        """Test record_metric with multiple peers."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            peer_id="peer2",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)

        assert len(monitor.metrics) == 2
        assert "peer1" in monitor.peer_metrics
        assert "peer2" in monitor.peer_metrics
        assert len(monitor.peer_metrics["peer1"]) == 1
        assert len(monitor.peer_metrics["peer2"]) == 1

    def test_get_metrics_all(self):
        """Test get_metrics without filters."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            peer_id="peer2",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)

        metrics = monitor.get_metrics()

        assert len(metrics) == 2
        assert metric1 in metrics
        assert metric2 in metrics

    def test_get_metrics_by_type(self):
        """Test get_metrics filtered by metric_type."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            peer_id="peer2",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)

        latency_metrics = monitor.get_metrics(metric_type=MetricType.LATENCY)

        assert len(latency_metrics) == 1
        assert latency_metrics[0] == metric1

    def test_get_metrics_by_peer_id(self):
        """Test get_metrics filtered by peer_id."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            peer_id="peer2",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)

        peer1_metrics = monitor.get_metrics(peer_id="peer1")

        assert len(peer1_metrics) == 1
        assert peer1_metrics[0] == metric1

    def test_get_metrics_by_type_and_peer_id(self):
        """Test get_metrics filtered by both metric_type and peer_id."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=150.0,
            timestamp=time.time(),
            peer_id="peer2",
        )

        metric3 = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)
        monitor.record_metric(metric3)

        peer1_latency_metrics = monitor.get_metrics(
            metric_type=MetricType.LATENCY, peer_id="peer1"
        )

        assert len(peer1_latency_metrics) == 1
        assert peer1_latency_metrics[0] == metric1

    def test_get_average_metric(self):
        """Test get_average_metric."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=200.0,
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)

        avg_latency = monitor.get_average_metric(MetricType.LATENCY, "peer1")

        assert avg_latency == 150.0

    def test_get_average_metric_no_metrics(self):
        """Test get_average_metric with no metrics."""
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)

        avg_latency = monitor.get_average_metric(MetricType.LATENCY)

        assert avg_latency == 0.0

    @patch("time.time")
    def test_cleanup_old_metrics(self, mock_time):
        """Test _cleanup_old_metrics."""
        config = PerformanceConfig(metrics_retention=100.0)
        monitor = PerformanceMonitor(config)

        # Mock current time
        current_time = 1000.0
        mock_time.return_value = current_time

        # Add old metric
        old_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            timestamp=current_time - 200.0,  # 200 seconds ago
            peer_id="peer1",
        )

        # Add recent metric
        recent_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=200.0,
            timestamp=current_time - 50.0,  # 50 seconds ago
            peer_id="peer1",
        )

        monitor.record_metric(old_metric)
        monitor.record_metric(recent_metric)

        # Manually call cleanup
        monitor._cleanup_old_metrics()

        assert len(monitor.metrics) == 1
        assert monitor.metrics[0] == recent_metric
        assert len(monitor.peer_metrics["peer1"]) == 1
        assert monitor.peer_metrics["peer1"][0] == recent_metric


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class."""

    def test_performance_optimizer_creation(self):
        """Test PerformanceOptimizer creation."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        assert optimizer.config == config
        assert optimizer.optimization_history == []

    def test_analyze_performance_good_health(self):
        """Test analyze_performance with good performance."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)
        monitor = PerformanceMonitor(config)

        # Add good performance metrics
        metric1 = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=50.0,  # Low latency
            timestamp=time.time(),
            peer_id="peer1",
        )

        metric2 = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=200.0,  # High throughput
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(metric1)
        monitor.record_metric(metric2)

        analysis = optimizer.analyze_performance(monitor)

        assert analysis["overall_health"] == "good"
        assert len(analysis["bottlenecks"]) == 0
        assert len(analysis["recommendations"]) == 0

    def test_analyze_performance_high_latency(self):
        """Test analyze_performance with high latency."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)
        monitor = PerformanceMonitor(config)

        # Add high latency metric
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=1500.0,  # High latency
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(metric)

        analysis = optimizer.analyze_performance(monitor)

        assert analysis["overall_health"] == "good"
        assert "high_latency" in analysis["bottlenecks"]
        assert "optimize_network_routing" in analysis["recommendations"]

    def test_analyze_performance_low_throughput(self):
        """Test analyze_performance with low throughput."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)
        monitor = PerformanceMonitor(config)

        # Add low throughput metric
        metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,  # Low throughput
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(metric)

        analysis = optimizer.analyze_performance(monitor)

        assert analysis["overall_health"] == "good"
        assert "low_throughput" in analysis["bottlenecks"]
        assert "increase_connection_pool" in analysis["recommendations"]

    def test_analyze_performance_multiple_bottlenecks(self):
        """Test analyze_performance with multiple bottlenecks."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)
        monitor = PerformanceMonitor(config)

        # Add high latency and low throughput metrics
        latency_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=1500.0,  # High latency
            timestamp=time.time(),
            peer_id="peer1",
        )

        throughput_metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=50.0,  # Low throughput
            timestamp=time.time(),
            peer_id="peer1",
        )

        monitor.record_metric(latency_metric)
        monitor.record_metric(throughput_metric)

        analysis = optimizer.analyze_performance(monitor)

        assert analysis["overall_health"] == "good"
        assert "high_latency" in analysis["bottlenecks"]
        assert "low_throughput" in analysis["bottlenecks"]
        assert "optimize_network_routing" in analysis["recommendations"]
        assert "increase_connection_pool" in analysis["recommendations"]

    def test_optimize_network_no_bottlenecks(self):
        """Test optimize_network with no bottlenecks."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        analysis = {"overall_health": "good", "bottlenecks": [], "recommendations": []}

        optimizations = optimizer.optimize_network(analysis)

        assert len(optimizations) == 0

    def test_optimize_network_high_latency(self):
        """Test optimize_network with high latency bottleneck."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        analysis = {
            "overall_health": "good",
            "bottlenecks": ["high_latency"],
            "recommendations": ["optimize_network_routing"],
        }

        optimizations = optimizer.optimize_network(analysis)

        assert "routing_optimized" in optimizations

    def test_optimize_network_low_throughput(self):
        """Test optimize_network with low throughput bottleneck."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        analysis = {
            "overall_health": "good",
            "bottlenecks": ["low_throughput"],
            "recommendations": ["increase_connection_pool"],
        }

        optimizations = optimizer.optimize_network(analysis)

        assert "connection_pool_increased" in optimizations

    def test_optimize_network_multiple_bottlenecks(self):
        """Test optimize_network with multiple bottlenecks."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        analysis = {
            "overall_health": "good",
            "bottlenecks": ["high_latency", "low_throughput"],
            "recommendations": ["optimize_network_routing", "increase_connection_pool"],
        }

        optimizations = optimizer.optimize_network(analysis)

        assert "routing_optimized" in optimizations
        assert "connection_pool_increased" in optimizations
        assert len(optimizations) == 2


class TestNetworkPerformance:
    """Test NetworkPerformance class."""

    def test_network_performance_creation(self):
        """Test NetworkPerformance creation."""
        config = PerformanceConfig()
        network_perf = NetworkPerformance(config)

        assert network_perf.config == config
        assert network_perf.monitor is not None
        assert network_perf.optimizer is not None

    @patch("time.time")
    def test_record_latency(self, mock_time):
        """Test record_latency method."""
        config = PerformanceConfig()
        network_perf = NetworkPerformance(config)

        current_time = 1000.0
        mock_time.return_value = current_time

        network_perf.record_latency("peer1", 100.5)

        metrics = network_perf.monitor.get_metrics(
            metric_type=MetricType.LATENCY, peer_id="peer1"
        )

        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.LATENCY
        assert metrics[0].value == 100.5
        assert metrics[0].peer_id == "peer1"
        assert metrics[0].timestamp == current_time

    @patch("time.time")
    def test_record_throughput(self, mock_time):
        """Test record_throughput method."""
        config = PerformanceConfig()
        network_perf = NetworkPerformance(config)

        current_time = 1000.0
        mock_time.return_value = current_time

        network_perf.record_throughput("peer1", 50.0)

        metrics = network_perf.monitor.get_metrics(
            metric_type=MetricType.THROUGHPUT, peer_id="peer1"
        )

        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.THROUGHPUT
        assert metrics[0].value == 50.0
        assert metrics[0].peer_id == "peer1"
        assert metrics[0].timestamp == current_time

    def test_get_performance_analysis(self):
        """Test get_performance_analysis method."""
        config = PerformanceConfig()
        network_perf = NetworkPerformance(config)

        # Add some metrics
        network_perf.record_latency("peer1", 100.0)
        network_perf.record_throughput("peer1", 200.0)

        analysis = network_perf.get_performance_analysis()

        assert "overall_health" in analysis
        assert "bottlenecks" in analysis
        assert "recommendations" in analysis
        assert analysis["overall_health"] == "good"
