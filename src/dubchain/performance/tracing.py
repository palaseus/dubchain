"""
Advanced profiling and tracing for DubChain.

This module provides comprehensive profiling including:
- OpenTelemetry distributed tracing
- Real-time performance monitoring
- Memory leak detection
- Performance regression prediction
- Prometheus/Grafana integration
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
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class TraceConfig:
    """Configuration for distributed tracing."""
    
    enable_tracing: bool = True
    service_name: str = "dubchain"
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    sampling_rate: float = 1.0
    max_attributes: int = 32
    max_events: int = 128
    max_links: int = 32


@dataclass
class ProfilingConfig:
    """Configuration for profiling."""
    
    enable_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_network_profiling: bool = True
    profiling_interval: float = 1.0
    max_samples: int = 10000
    enable_flamegraph: bool = True
    enable_regression_detection: bool = True


@dataclass
class MetricsConfig:
    """Configuration for metrics."""
    
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    enable_custom_metrics: bool = True
    metrics_interval: float = 1.0
    enable_grafana_dashboard: bool = True


class DistributedTracer:
    """Distributed tracing using OpenTelemetry."""
    
    def __init__(self, config: TraceConfig):
        """Initialize distributed tracer."""
        self.config = config
        self.tracer: Optional[Any] = None
        self._initialized = False
        
        if config.enable_tracing and OPENTELEMETRY_AVAILABLE:
            self._initialize_tracer()
    
    def _initialize_tracer(self) -> None:
        """Initialize OpenTelemetry tracer."""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": "1.0.0",
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Add exporters
            if self.config.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=14268,
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            if self.config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            self._initialized = True
            
        except Exception as e:
            logger.info(f"Failed to initialize tracer: {e}")
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start a new span."""
        if not self._initialized or not self.tracer:
            return None
        
        try:
            span = self.tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            return span
        except Exception:
            return None
    
    def end_span(self, span: Any, status: Optional[str] = None) -> None:
        """End a span."""
        if not span:
            return
        
        try:
            if status:
                if status == "success":
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.ERROR, status))
            span.end()
        except Exception:
            pass
    
    def add_event(self, span: Any, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        if not span:
            return
        
        try:
            span.add_event(name, attributes or {})
        except Exception:
            pass


class PerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self, config: ProfilingConfig):
        """Initialize performance profiler."""
        self.config = config
        self.samples: deque = deque(maxlen=config.max_samples)
        self.memory_samples: deque = deque(maxlen=config.max_samples)
        self.cpu_samples: deque = deque(maxlen=config.max_samples)
        self.network_samples: deque = deque(maxlen=config.max_samples)
        self._lock = threading.Lock()
        self._profiling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if config.enable_profiling:
            self._start_profiling()
    
    def _start_profiling(self) -> None:
        """Start profiling thread."""
        self._profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True
        )
        self._profiling_thread.start()
    
    def _profiling_loop(self) -> None:
        """Main profiling loop."""
        while not self._stop_event.wait(self.config.profiling_interval):
            try:
                self._collect_samples()
            except Exception as e:
                logger.info(f"Profiling error: {e}")
    
    def _collect_samples(self) -> None:
        """Collect performance samples."""
        timestamp = time.time()
        
        with self._lock:
            # Memory profiling
            if self.config.enable_memory_profiling and PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_sample = {
                    "timestamp": timestamp,
                    "rss_mb": memory_info.rss / (1024 * 1024),
                    "vms_mb": memory_info.vms / (1024 * 1024),
                    "percent": process.memory_percent(),
                }
                self.memory_samples.append(memory_sample)
            
            # CPU profiling
            if self.config.enable_cpu_profiling and PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_sample = {
                    "timestamp": timestamp,
                    "cpu_percent": cpu_percent,
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                }
                self.cpu_samples.append(cpu_sample)
            
            # Network profiling
            if self.config.enable_network_profiling and PSUTIL_AVAILABLE:
                net_io = psutil.net_io_counters()
                network_sample = {
                    "timestamp": timestamp,
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                self.network_samples.append(network_sample)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            if not self.memory_samples:
                return {}
            
            rss_values = [s["rss_mb"] for s in self.memory_samples]
            vms_values = [s["vms_mb"] for s in self.memory_samples]
            percent_values = [s["percent"] for s in self.memory_samples]
            
            return {
                "current_rss_mb": rss_values[-1] if rss_values else 0,
                "current_vms_mb": vms_values[-1] if vms_values else 0,
                "current_percent": percent_values[-1] if percent_values else 0,
                "avg_rss_mb": sum(rss_values) / len(rss_values),
                "max_rss_mb": max(rss_values),
                "min_rss_mb": min(rss_values),
                "avg_percent": sum(percent_values) / len(percent_values),
                "max_percent": max(percent_values),
                "min_percent": min(percent_values),
                "samples_count": len(self.memory_samples),
            }
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics."""
        with self._lock:
            if not self.cpu_samples:
                return {}
            
            cpu_values = [s["cpu_percent"] for s in self.cpu_samples]
            
            return {
                "current_cpu_percent": cpu_values[-1] if cpu_values else 0,
                "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                "max_cpu_percent": max(cpu_values),
                "min_cpu_percent": min(cpu_values),
                "samples_count": len(self.cpu_samples),
            }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        with self._lock:
            if not self.network_samples:
                return {}
            
            bytes_sent = [s["bytes_sent"] for s in self.network_samples]
            bytes_recv = [s["bytes_recv"] for s in self.network_samples]
            
            return {
                "current_bytes_sent": bytes_sent[-1] if bytes_sent else 0,
                "current_bytes_recv": bytes_recv[-1] if bytes_recv else 0,
                "total_bytes_sent": max(bytes_sent) - min(bytes_sent) if len(bytes_sent) > 1 else 0,
                "total_bytes_recv": max(bytes_recv) - min(bytes_recv) if len(bytes_recv) > 1 else 0,
                "samples_count": len(self.network_samples),
            }
    
    def detect_memory_leak(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        with self._lock:
            if len(self.memory_samples) < 10:
                return {"leak_detected": False, "reason": "insufficient_samples"}
            
            rss_values = [s["rss_mb"] for s in self.memory_samples]
            
            # Simple trend analysis
            recent_samples = rss_values[-10:]
            older_samples = rss_values[-20:-10] if len(rss_values) >= 20 else rss_values[:-10]
            
            if not older_samples:
                return {"leak_detected": False, "reason": "insufficient_history"}
            
            recent_avg = sum(recent_samples) / len(recent_samples)
            older_avg = sum(older_samples) / len(older_samples)
            
            growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            
            leak_detected = growth_rate > 0.1  # 10% growth threshold
            
            return {
                "leak_detected": leak_detected,
                "growth_rate": growth_rate,
                "recent_avg_mb": recent_avg,
                "older_avg_mb": older_avg,
                "threshold": 0.1,
            }
    
    def generate_flamegraph(self, output_path: str) -> bool:
        """Generate flamegraph visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        try:
            with self._lock:
                if not self.memory_samples:
                    return False
                
                timestamps = [s["timestamp"] for s in self.memory_samples]
                rss_values = [s["rss_mb"] for s in self.memory_samples]
                
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps, rss_values, 'b-', linewidth=2)
                plt.title('Memory Usage Over Time')
                plt.xlabel('Time')
                plt.ylabel('RSS Memory (MB)')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return True
                
        except Exception as e:
            logger.info(f"Failed to generate flamegraph: {e}")
            return False
    
    def stop_profiling(self) -> None:
        """Stop profiling."""
        self._stop_event.set()
        if self._profiling_thread:
            self._profiling_thread.join(timeout=5.0)


class PrometheusMetrics:
    """Prometheus metrics integration."""
    
    def __init__(self, config: MetricsConfig):
        """Initialize Prometheus metrics."""
        self.config = config
        self.metrics: Dict[str, Any] = {}
        self._initialized = False
        
        if config.enable_prometheus and PROMETHEUS_AVAILABLE:
            self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        try:
            # Define metrics
            self.metrics = {
                "operations_total": Counter(
                    "dubchain_operations_total",
                    "Total number of operations",
                    ["operation_type", "status"]
                ),
                "operation_duration": Histogram(
                    "dubchain_operation_duration_seconds",
                    "Operation duration in seconds",
                    ["operation_type"],
                    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
                ),
                "memory_usage": Gauge(
                    "dubchain_memory_usage_bytes",
                    "Current memory usage in bytes"
                ),
                "cpu_usage": Gauge(
                    "dubchain_cpu_usage_percent",
                    "Current CPU usage percentage"
                ),
                "active_connections": Gauge(
                    "dubchain_active_connections",
                    "Number of active connections"
                ),
                "block_height": Gauge(
                    "dubchain_block_height",
                    "Current block height"
                ),
            }
            
            # Start HTTP server
            start_http_server(self.config.prometheus_port)
            self._initialized = True
            
        except Exception as e:
            logger.info(f"Failed to initialize Prometheus metrics: {e}")
    
    def record_operation(self, operation_type: str, duration: float, status: str = "success") -> None:
        """Record operation metrics."""
        if not self._initialized:
            return
        
        try:
            self.metrics["operations_total"].labels(
                operation_type=operation_type,
                status=status
            ).inc()
            
            self.metrics["operation_duration"].labels(
                operation_type=operation_type
            ).observe(duration)
            
        except Exception as e:
            logger.info(f"Failed to record operation metrics: {e}")
    
    def update_memory_usage(self, bytes_used: int) -> None:
        """Update memory usage metric."""
        if not self._initialized:
            return
        
        try:
            self.metrics["memory_usage"].set(bytes_used)
        except Exception as e:
            logger.info(f"Failed to update memory usage: {e}")
    
    def update_cpu_usage(self, cpu_percent: float) -> None:
        """Update CPU usage metric."""
        if not self._initialized:
            return
        
        try:
            self.metrics["cpu_usage"].set(cpu_percent)
        except Exception as e:
            logger.info(f"Failed to update CPU usage: {e}")
    
    def update_active_connections(self, count: int) -> None:
        """Update active connections metric."""
        if not self._initialized:
            return
        
        try:
            self.metrics["active_connections"].set(count)
        except Exception as e:
            logger.info(f"Failed to update active connections: {e}")
    
    def update_block_height(self, height: int) -> None:
        """Update block height metric."""
        if not self._initialized:
            return
        
        try:
            self.metrics["block_height"].set(height)
        except Exception as e:
            logger.info(f"Failed to update block height: {e}")


class AdvancedProfiler:
    """Advanced profiling system combining all profiling features."""
    
    def __init__(
        self,
        trace_config: Optional[TraceConfig] = None,
        profiling_config: Optional[ProfilingConfig] = None,
        metrics_config: Optional[MetricsConfig] = None,
    ):
        """Initialize advanced profiler."""
        self.trace_config = trace_config or TraceConfig()
        self.profiling_config = profiling_config or ProfilingConfig()
        self.metrics_config = metrics_config or MetricsConfig()
        
        # Initialize components
        self.tracer = DistributedTracer(self.trace_config)
        self.profiler = PerformanceProfiler(self.profiling_config)
        self.metrics = PrometheusMetrics(self.metrics_config)
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.regression_thresholds: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start profiling an operation."""
        span = self.tracer.start_span(operation_name, attributes)
        return {
            "span": span,
            "start_time": time.time(),
            "operation_name": operation_name,
        }
    
    def end_operation(self, operation_context: Dict[str, Any], status: str = "success") -> None:
        """End profiling an operation."""
        span = operation_context.get("span")
        start_time = operation_context.get("start_time")
        operation_name = operation_context.get("operation_name")
        
        if span:
            self.tracer.end_span(span, status)
        
        if start_time:
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.record_operation(operation_name, duration, status)
            
            # Track for regression detection
            self.operation_times[operation_name].append(duration)
            
            # Keep only recent samples
            if len(self.operation_times[operation_name]) > 1000:
                self.operation_times[operation_name] = self.operation_times[operation_name][-500:]
    
    def detect_performance_regression(self, operation_name: str) -> Dict[str, Any]:
        """Detect performance regression for an operation."""
        if operation_name not in self.operation_times:
            return {"regression_detected": False, "reason": "no_data"}
        
        times = self.operation_times[operation_name]
        if len(times) < 20:
            return {"regression_detected": False, "reason": "insufficient_samples"}
        
        # Compare recent vs historical performance
        recent_times = times[-10:]
        historical_times = times[-50:-10] if len(times) >= 50 else times[:-10]
        
        if not historical_times:
            return {"regression_detected": False, "reason": "insufficient_history"}
        
        recent_avg = sum(recent_times) / len(recent_times)
        historical_avg = sum(historical_times) / len(historical_times)
        
        regression_factor = recent_avg / historical_avg if historical_avg > 0 else 1.0
        
        # Threshold for regression detection
        threshold = self.regression_thresholds.get(operation_name, 1.5)  # 50% slower
        regression_detected = regression_factor > threshold
        
        return {
            "regression_detected": regression_detected,
            "regression_factor": regression_factor,
            "recent_avg_ms": recent_avg * 1000,
            "historical_avg_ms": historical_avg * 1000,
            "threshold": threshold,
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "memory": self.profiler.get_memory_stats(),
            "cpu": self.profiler.get_cpu_stats(),
            "network": self.profiler.get_network_stats(),
            "memory_leak": self.profiler.detect_memory_leak(),
            "operations": {
                name: {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times) * 1000 if times else 0,
                    "min_ms": min(times) * 1000 if times else 0,
                    "max_ms": max(times) * 1000 if times else 0,
                }
                for name, times in self.operation_times.items()
            },
        }
    
    def generate_report(self, output_path: str) -> bool:
        """Generate comprehensive performance report."""
        try:
            stats = self.get_comprehensive_stats()
            
            # Generate flamegraph
            flamegraph_path = output_path.replace('.json', '_flamegraph.png')
            self.profiler.generate_flamegraph(flamegraph_path)
            
            # Save JSON report
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.info(f"Failed to generate report: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup profiler resources."""
        self.profiler.stop_profiling()
