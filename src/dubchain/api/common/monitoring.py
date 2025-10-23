"""
Monitoring and Metrics Infrastructure for DubChain.

This module provides comprehensive monitoring capabilities with metrics
collection, aggregation, and reporting for system performance and health.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Metric value with metadata."""

    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricConfig:
    """Metric configuration."""

    retention_period: int = 3600  # 1 hour
    aggregation_interval: int = 60  # 1 minute
    max_samples: int = 1000
    enable_histograms: bool = True
    enable_timers: bool = True


class Counter:
    """Counter metric."""

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.value = 0
        self.last_reset = time.time()

    def increment(self, amount: int = 1):
        """Increment counter."""
        self.value += amount

    def reset(self):
        """Reset counter."""
        self.value = 0
        self.last_reset = time.time()

    def get_value(self) -> int:
        """Get current value."""
        return self.value


class Gauge:
    """Gauge metric."""

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.value = 0.0
        self.last_update = time.time()

    def set(self, value: Union[int, float]):
        """Set gauge value."""
        self.value = float(value)
        self.last_update = time.time()

    def increment(self, amount: Union[int, float] = 1):
        """Increment gauge."""
        self.value += float(amount)
        self.last_update = time.time()

    def decrement(self, amount: Union[int, float] = 1):
        """Decrement gauge."""
        self.value -= float(amount)
        self.last_update = time.time()

    def get_value(self) -> float:
        """Get current value."""
        return self.value


class Histogram:
    """Histogram metric."""

    def __init__(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.labels = labels or {}
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        self.samples = deque(maxlen=1000)
        self.count = 0
        self.sum = 0.0

    def observe(self, value: Union[int, float]):
        """Observe a value."""
        value = float(value)
        self.samples.append(value)
        self.count += 1
        self.sum += value

    def get_stats(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        if not self.samples:
            return {
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "buckets": {str(bucket): 0 for bucket in self.buckets},
            }

        samples = list(self.samples)
        samples.sort()

        # Calculate percentiles
        count = len(samples)
        p50 = samples[int(count * 0.5)] if count > 0 else 0
        p90 = samples[int(count * 0.9)] if count > 0 else 0
        p95 = samples[int(count * 0.95)] if count > 0 else 0
        p99 = samples[int(count * 0.99)] if count > 0 else 0

        # Calculate bucket counts
        bucket_counts = {}
        for bucket in self.buckets:
            bucket_counts[str(bucket)] = sum(1 for s in samples if s <= bucket)

        return {
            "count": self.count,
            "sum": self.sum,
            "min": min(samples),
            "max": max(samples),
            "mean": self.sum / count if count > 0 else 0,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "buckets": bucket_counts,
        }


class Timer:
    """Timer metric."""

    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.histogram = Histogram(f"{name}_duration", labels=labels)
        self.active_timers = {}

    def start(self, timer_id: str = "default"):
        """Start timer."""
        self.active_timers[timer_id] = time.time()

    def stop(self, timer_id: str = "default"):
        """Stop timer and record duration."""
        if timer_id in self.active_timers:
            duration = time.time() - self.active_timers[timer_id]
            self.histogram.observe(duration)
            del self.active_timers[timer_id]
            return duration
        return 0.0

    def time(self, func):
        """Decorator to time function execution."""

        def wrapper(*args, **kwargs):
            self.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.stop()

        return wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get timer statistics."""
        return self.histogram.get_stats()


class MetricsCollector:
    """Main metrics collector."""

    def __init__(self, config: Optional[MetricConfig] = None):
        """Initialize metrics collector."""
        self.config = config or MetricConfig()
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.timers: Dict[str, Timer] = {}

        # System metrics
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": {"bytes_sent": 0, "bytes_received": 0},
            "process_count": 0,
            "uptime": 0.0,
        }

        # Start collection task
        self._collection_task = None
        self._start_collection_task()

    def counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create counter."""
        key = self._get_key(name, labels)
        if key not in self.counters:
            self.counters[key] = Counter(name, labels)
        return self.counters[key]

    def gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create gauge."""
        key = self._get_key(name, labels)
        if key not in self.gauges:
            self.gauges[key] = Gauge(name, labels)
        return self.gauges[key]

    def histogram(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Histogram:
        """Get or create histogram."""
        key = self._get_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = Histogram(name, buckets, labels)
        return self.histograms[key]

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> Timer:
        """Get or create timer."""
        key = self._get_key(name, labels)
        if key not in self.timers:
            self.timers[key] = Timer(name, labels)
        return self.timers[key]

    def _get_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Get unique key for metric."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    async def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        metrics = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {},
            "system": self.system_metrics,
            "timestamp": time.time(),
        }

        # Collect counter metrics
        for key, counter in self.counters.items():
            metrics["counters"][key] = {
                "value": counter.get_value(),
                "labels": counter.labels,
                "last_reset": counter.last_reset,
            }

        # Collect gauge metrics
        for key, gauge in self.gauges.items():
            metrics["gauges"][key] = {
                "value": gauge.get_value(),
                "labels": gauge.labels,
                "last_update": gauge.last_update,
            }

        # Collect histogram metrics
        for key, histogram in self.histograms.items():
            metrics["histograms"][key] = {
                "stats": histogram.get_stats(),
                "labels": histogram.labels,
            }

        # Collect timer metrics
        for key, timer in self.timers.items():
            metrics["timers"][key] = {
                "stats": timer.get_stats(),
                "labels": timer.labels,
            }

        return metrics

    def get_metric_summary(self) -> Dict[str, Any]:
        """Get metric summary."""
        return {
            "total_metrics": len(self.counters)
            + len(self.gauges)
            + len(self.histograms)
            + len(self.timers),
            "counters": len(self.counters),
            "gauges": len(self.gauges),
            "histograms": len(self.histograms),
            "timers": len(self.timers),
            "system_metrics": len(self.system_metrics),
        }

    def reset_metrics(self):
        """Reset all metrics."""
        for counter in self.counters.values():
            counter.reset()

        for gauge in self.gauges.values():
            gauge.set(0)

        for histogram in self.histograms.values():
            histogram.samples.clear()
            histogram.count = 0
            histogram.sum = 0.0

        for timer in self.timers.values():
            timer.histogram.samples.clear()
            timer.histogram.count = 0
            timer.histogram.sum = 0.0

    def _start_collection_task(self):
        """Start system metrics collection task."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, skip collection task for now
            self._collection_task = None
            return

        async def collect_system_metrics():
            while True:
                try:
                    await asyncio.sleep(10)  # Collect every 10 seconds
                    await self._collect_system_metrics()
                except Exception as e:
                    logger.info(f"System metrics collection error: {e}")

        self._collection_task = asyncio.create_task(collect_system_metrics())

    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            import psutil

            # CPU usage
            self.system_metrics["cpu_usage"] = psutil.cpu_percent()

            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics["memory_usage"] = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            self.system_metrics["disk_usage"] = (disk.used / disk.total) * 100

            # Network I/O
            net_io = psutil.net_io_counters()
            self.system_metrics["network_io"]["bytes_sent"] = net_io.bytes_sent
            self.system_metrics["network_io"]["bytes_received"] = net_io.bytes_recv

            # Process count
            self.system_metrics["process_count"] = len(psutil.pids())

            # Uptime
            self.system_metrics["uptime"] = time.time() - psutil.boot_time()

        except ImportError:
            # psutil not available, use mock values
            self.system_metrics["cpu_usage"] = 0.0
            self.system_metrics["memory_usage"] = 0.0
            self.system_metrics["disk_usage"] = 0.0
            self.system_metrics["process_count"] = 1
            self.system_metrics["uptime"] = time.time()

    def stop(self):
        """Stop metrics collector."""
        if self._collection_task:
            self._collection_task.cancel()


# Global metrics collector instance
metrics_collector = MetricsCollector()
