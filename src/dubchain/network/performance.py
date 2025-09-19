"""
Performance monitoring and optimization for GodChain P2P network.
"""

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .peer import Peer, PeerInfo


class MetricType(Enum):
    """Performance metric types."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BANDWIDTH = "bandwidth"
    CONNECTION_COUNT = "connection_count"
    MESSAGE_COUNT = "message_count"


@dataclass
class PerformanceMetric:
    """Performance metric data."""

    metric_type: MetricType
    value: float
    timestamp: float
    peer_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""

    monitoring_interval: float = 10.0
    metrics_retention: float = 3600.0
    enable_auto_optimization: bool = True
    optimization_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Performance monitoring system."""

    def __init__(self, config: PerformanceConfig):
        """Initialize performance monitor."""
        self.config = config
        self.metrics: List[PerformanceMetric] = []
        self.peer_metrics: Dict[str, List[PerformanceMetric]] = {}

    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record performance metric."""
        self.metrics.append(metric)

        if metric.peer_id:
            if metric.peer_id not in self.peer_metrics:
                self.peer_metrics[metric.peer_id] = []
            self.peer_metrics[metric.peer_id].append(metric)

        # Clean old metrics
        self._cleanup_old_metrics()

    def get_metrics(
        self, metric_type: Optional[MetricType] = None, peer_id: Optional[str] = None
    ) -> List[PerformanceMetric]:
        """Get performance metrics."""
        filtered_metrics = self.metrics

        if metric_type:
            filtered_metrics = [
                m for m in filtered_metrics if m.metric_type == metric_type
            ]

        if peer_id:
            filtered_metrics = [m for m in filtered_metrics if m.peer_id == peer_id]

        return filtered_metrics

    def get_average_metric(
        self, metric_type: MetricType, peer_id: Optional[str] = None
    ) -> float:
        """Get average metric value."""
        metrics = self.get_metrics(metric_type, peer_id)
        if not metrics:
            return 0.0

        return statistics.mean(m.value for m in metrics)

    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics."""
        current_time = time.time()
        cutoff_time = current_time - self.config.metrics_retention

        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

        for peer_id in self.peer_metrics:
            self.peer_metrics[peer_id] = [
                m for m in self.peer_metrics[peer_id] if m.timestamp > cutoff_time
            ]


class PerformanceOptimizer:
    """Performance optimization system."""

    def __init__(self, config: PerformanceConfig):
        """Initialize performance optimizer."""
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []

    def analyze_performance(self, monitor: PerformanceMonitor) -> Dict[str, Any]:
        """Analyze network performance."""
        analysis = {"overall_health": "good", "bottlenecks": [], "recommendations": []}

        # Analyze latency
        avg_latency = monitor.get_average_metric(MetricType.LATENCY)
        if avg_latency > 1000:  # 1 second
            analysis["bottlenecks"].append("high_latency")
            analysis["recommendations"].append("optimize_network_routing")

        # Analyze throughput
        avg_throughput = monitor.get_average_metric(MetricType.THROUGHPUT)
        if avg_throughput < 100:  # 100 messages/second
            analysis["bottlenecks"].append("low_throughput")
            analysis["recommendations"].append("increase_connection_pool")

        return analysis

    def optimize_network(self, analysis: Dict[str, Any]) -> List[str]:
        """Optimize network based on analysis."""
        optimizations = []

        if "high_latency" in analysis["bottlenecks"]:
            optimizations.append("routing_optimized")

        if "low_throughput" in analysis["bottlenecks"]:
            optimizations.append("connection_pool_increased")

        return optimizations


class NetworkPerformance:
    """Network performance manager."""

    def __init__(self, config: PerformanceConfig):
        """Initialize network performance."""
        self.config = config
        self.monitor = PerformanceMonitor(config)
        self.optimizer = PerformanceOptimizer(config)

    def record_latency(self, peer_id: str, latency: float) -> None:
        """Record latency metric."""
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=latency,
            timestamp=time.time(),
            peer_id=peer_id,
        )
        self.monitor.record_metric(metric)

    def record_throughput(self, peer_id: str, throughput: float) -> None:
        """Record throughput metric."""
        metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=throughput,
            timestamp=time.time(),
            peer_id=peer_id,
        )
        self.monitor.record_metric(metric)

    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get performance analysis."""
        return self.optimizer.analyze_performance(self.monitor)
