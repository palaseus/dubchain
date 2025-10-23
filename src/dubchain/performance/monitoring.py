"""
Performance monitoring and alerting system for DubChain.

This module provides:
- Real-time performance metrics collection
- Performance alerting and threshold monitoring
- Performance dashboard and reporting
- Integration with external monitoring systems
- Performance trend analysis and forecasting
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import psutil
from datetime import datetime, timedelta
import statistics

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class MetricType(Enum):
    """Types of performance metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """A performance metric."""
    
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """A performance alert."""
    
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    severity: AlertSeverity
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    description: str = ""


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.metric_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
        
    def record_metric(self, metric: Metric) -> None:
        """Record a performance metric."""
        with self._lock:
            self.metrics.append(metric)
            self.metric_series[metric.name].append(metric)
            
    def record_counter(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
        
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
        
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric."""
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
        
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
        
    def get_metric_series(self, name: str, time_window: Optional[float] = None) -> List[Metric]:
        """Get metric series for a specific metric name."""
        with self._lock:
            if name not in self.metric_series:
                return []
                
            series = list(self.metric_series[name])
            
            if time_window:
                cutoff_time = time.time() - time_window
                series = [m for m in series if m.timestamp >= cutoff_time]
                
            return series
            
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest metric for a specific name."""
        with self._lock:
            if name not in self.metric_series or not self.metric_series[name]:
                return None
            return self.metric_series[name][-1]
            
    def get_metric_statistics(self, name: str, time_window: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a metric series."""
        series = self.get_metric_series(name, time_window)
        
        if not series:
            return {}
            
        values = [m.value for m in series]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }
        
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


class AlertManager:
    """Manages performance alerts and thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a performance threshold."""
        with self._lock:
            self.thresholds[threshold.metric_name] = threshold
            
    def remove_threshold(self, metric_name: str) -> None:
        """Remove a performance threshold."""
        with self._lock:
            self.thresholds.pop(metric_name, None)
            
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add an alert callback."""
        self.alert_callbacks.append(callback)
        
    def check_thresholds(self) -> List[Alert]:
        """Check all thresholds and generate alerts."""
        new_alerts = []
        
        with self._lock:
            for metric_name, threshold in self.thresholds.items():
                if not threshold.enabled:
                    continue
                    
                latest_metric = self.metrics_collector.get_latest_metric(metric_name)
                if not latest_metric:
                    continue
                    
                # Check if threshold is violated
                if self._check_threshold_violation(latest_metric.value, threshold):
                    # Check cooldown
                    if self._is_in_cooldown(metric_name, threshold.cooldown_seconds):
                        continue
                        
                    # Create alert
                    alert = Alert(
                        name=f"{metric_name}_threshold_violation",
                        severity=threshold.severity,
                        message=f"{metric_name} {threshold.comparison_operator} {threshold.threshold_value} (current: {latest_metric.value})",
                        metric_name=metric_name,
                        threshold=threshold.threshold_value,
                        current_value=latest_metric.value,
                        timestamp=time.time(),
                        metadata={"threshold_description": threshold.description}
                    )
                    
                    # Check if this is a new alert or update existing
                    if metric_name in self.active_alerts:
                        # Update existing alert
                        existing_alert = self.active_alerts[metric_name]
                        existing_alert.current_value = latest_metric.value
                        existing_alert.timestamp = time.time()
                    else:
                        # New alert
                        self.active_alerts[metric_name] = alert
                        self.alert_history.append(alert)
                        new_alerts.append(alert)
                        
                else:
                    # Threshold not violated, resolve alert if exists
                    if metric_name in self.active_alerts:
                        alert = self.active_alerts[metric_name]
                        alert.resolved = True
                        alert.timestamp = time.time()
                        del self.active_alerts[metric_name]
                        
        # Notify callbacks
        for alert in new_alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.info(f"Alert callback failed: {e}")
                    
        return new_alerts
        
    def _check_threshold_violation(self, value: float, threshold: PerformanceThreshold) -> bool:
        """Check if a value violates a threshold."""
        if threshold.comparison_operator == '>':
            return value > threshold.threshold_value
        elif threshold.comparison_operator == '<':
            return value < threshold.threshold_value
        elif threshold.comparison_operator == '>=':
            return value >= threshold.threshold_value
        elif threshold.comparison_operator == '<=':
            return value <= threshold.threshold_value
        elif threshold.comparison_operator == '==':
            return value == threshold.threshold_value
        elif threshold.comparison_operator == '!=':
            return value != threshold.threshold_value
        else:
            return False
            
    def _is_in_cooldown(self, metric_name: str, cooldown_seconds: int) -> bool:
        """Check if an alert is in cooldown period."""
        if metric_name not in self.active_alerts:
            return False
            
        alert = self.active_alerts[metric_name]
        return (time.time() - alert.timestamp) < cooldown_seconds
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
            
    def get_alert_history(self, time_window: Optional[float] = None) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            if time_window:
                cutoff_time = time.time() - time_window
                return [a for a in self.alert_history if a.timestamp >= cutoff_time]
            return list(self.alert_history)


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.monitoring_enabled = True
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # System metrics collection
        self.process = psutil.Process()
        
        # Initialize default thresholds
        self._setup_default_thresholds()
        
    def _setup_default_thresholds(self) -> None:
        """Setup default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="cpu_usage_percent",
                threshold_value=80.0,
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                description="High CPU usage detected"
            ),
            PerformanceThreshold(
                metric_name="memory_usage_mb",
                threshold_value=1000.0,
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                description="High memory usage detected"
            ),
            PerformanceThreshold(
                metric_name="block_creation_latency_ms",
                threshold_value=1000.0,
                comparison_operator=">",
                severity=AlertSeverity.ERROR,
                description="Block creation latency too high"
            ),
            PerformanceThreshold(
                metric_name="transaction_throughput_tps",
                threshold_value=100.0,
                comparison_operator="<",
                severity=AlertSeverity.WARNING,
                description="Transaction throughput too low"
            ),
        ]
        
        for threshold in default_thresholds:
            self.alert_manager.add_threshold(threshold)
            
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start performance monitoring."""
        if self.monitoring_enabled and not self._stop_event.is_set():
            return
            
        self.monitoring_enabled = True
        self._stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_enabled = False
        self._stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while not self._stop_event.wait(interval):
            try:
                self._collect_system_metrics()
                self.alert_manager.check_thresholds()
            except Exception as e:
                logger.info(f"Monitoring error: {e}")
                
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        self.metrics_collector.record_gauge("cpu_usage_percent", cpu_percent)
        
        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.metrics_collector.record_gauge("memory_usage_mb", memory_mb)
        
        # Memory percentage
        memory_percent = self.process.memory_percent()
        self.metrics_collector.record_gauge("memory_usage_percent", memory_percent)
        
        # Thread count
        thread_count = self.process.num_threads()
        self.metrics_collector.record_gauge("thread_count", thread_count)
        
        # File descriptor count
        try:
            fd_count = self.process.num_fds()
            self.metrics_collector.record_gauge("file_descriptor_count", fd_count)
        except (psutil.AccessDenied, AttributeError):
            pass  # Not available on all systems
            
    def record_block_creation_time(self, duration: float) -> None:
        """Record block creation time."""
        self.metrics_collector.record_timer("block_creation_latency_ms", duration * 1000)
        
    def record_transaction_throughput(self, tps: float) -> None:
        """Record transaction throughput."""
        self.metrics_collector.record_gauge("transaction_throughput_tps", tps)
        
    def record_consensus_time(self, duration: float) -> None:
        """Record consensus time."""
        self.metrics_collector.record_timer("consensus_latency_ms", duration * 1000)
        
    def record_network_latency(self, latency: float) -> None:
        """Record network latency."""
        self.metrics_collector.record_timer("network_latency_ms", latency * 1000)
        
    def record_storage_operation_time(self, operation: str, duration: float) -> None:
        """Record storage operation time."""
        self.metrics_collector.record_timer(f"storage_{operation}_latency_ms", duration * 1000)
        
    def get_performance_summary(self, time_window: float = 300) -> Dict[str, Any]:
        """Get performance summary for the last time window."""
        summary = {
            "time_window_seconds": time_window,
            "timestamp": time.time(),
            "system_metrics": {},
            "application_metrics": {},
            "active_alerts": len(self.alert_manager.get_active_alerts()),
        }
        
        # System metrics
        system_metrics = ["cpu_usage_percent", "memory_usage_mb", "memory_usage_percent", "thread_count"]
        for metric_name in system_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric_name, time_window)
            if stats:
                summary["system_metrics"][metric_name] = stats
                
        # Application metrics
        app_metrics = ["block_creation_latency_ms", "transaction_throughput_tps", "consensus_latency_ms"]
        for metric_name in app_metrics:
            stats = self.metrics_collector.get_metric_statistics(metric_name, time_window)
            if stats:
                summary["application_metrics"][metric_name] = stats
                
        return summary


class PerformanceDashboard:
    """Performance dashboard for visualization and reporting."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.output_directory = "performance_dashboard"
        
    def generate_dashboard(self) -> str:
        """Generate performance dashboard HTML."""
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Get performance summary
        summary = self.performance_monitor.get_performance_summary()
        
        # Generate HTML dashboard
        html_content = self._generate_html_dashboard(summary)
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_directory, "dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
            
        return dashboard_path
        
    def _generate_html_dashboard(self, summary: Dict[str, Any]) -> str:
        """Generate HTML dashboard content."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DubChain Performance Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-bottom: 5px; }}
        .alert {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .summary {{ background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>DubChain Performance Dashboard</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Time Window: {summary['time_window_seconds']} seconds</p>
        <p>Active Alerts: {summary['active_alerts']}</p>
    </div>
    
    <h2>System Metrics</h2>
"""
        
        # System metrics
        for metric_name, stats in summary.get("system_metrics", {}).items():
            html += f"""
    <div class="metric-card">
        <div class="metric-label">{metric_name}</div>
        <div class="metric-value">{stats.get('mean', 0):.2f}</div>
        <p>Min: {stats.get('min', 0):.2f} | Max: {stats.get('max', 0):.2f} | P95: {stats.get('p95', 0):.2f}</p>
    </div>
"""
        
        html += """
    <h2>Application Metrics</h2>
"""
        
        # Application metrics
        for metric_name, stats in summary.get("application_metrics", {}).items():
            html += f"""
    <div class="metric-card">
        <div class="metric-label">{metric_name}</div>
        <div class="metric-value">{stats.get('mean', 0):.2f}</div>
        <p>Min: {stats.get('min', 0):.2f} | Max: {stats.get('max', 0):.2f} | P95: {stats.get('p95', 0):.2f}</p>
    </div>
"""
        
        # Active alerts
        active_alerts = self.performance_monitor.alert_manager.get_active_alerts()
        if active_alerts:
            html += """
    <h2>Active Alerts</h2>
"""
            for alert in active_alerts:
                html += f"""
    <div class="alert">
        <strong>{alert.severity.value.upper()}:</strong> {alert.message}
        <br><small>Time: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        return html
        
    def generate_metrics_report(self, time_window: float = 3600) -> str:
        """Generate detailed metrics report."""
        report_lines = [
            "# DubChain Performance Metrics Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Time Window: {time_window} seconds",
            "",
        ]
        
        # Get all metrics
        all_metrics = self.performance_monitor.metrics_collector.metrics
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in all_metrics:
            if time.time() - metric.timestamp <= time_window:
                metric_groups[metric.name].append(metric)
                
        # Generate report for each metric
        for metric_name, metrics in metric_groups.items():
            if not metrics:
                continue
                
            values = [m.value for m in metrics]
            
            report_lines.extend([
                f"## {metric_name}",
                f"- Count: {len(values)}",
                f"- Min: {min(values):.3f}",
                f"- Max: {max(values):.3f}",
                f"- Mean: {statistics.mean(values):.3f}",
                f"- Median: {statistics.median(values):.3f}",
                f"- Std Dev: {statistics.stdev(values) if len(values) > 1 else 0:.3f}",
                f"- P95: {self._percentile(values, 95):.3f}",
                f"- P99: {self._percentile(values, 99):.3f}",
                "",
            ])
            
        # Save report
        report_path = os.path.join(self.output_directory, "metrics_report.md")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
            
        return report_path
        
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
