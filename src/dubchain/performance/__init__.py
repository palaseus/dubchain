"""
Performance optimization module for DubChain.

This module provides comprehensive performance optimization tools including:
- Automated profiling and hotspot detection
- Performance benchmarking and regression testing
- Optimization implementations across all subsystems
- Feature gates for toggling optimizations
- Performance monitoring and alerting
"""

from .profiling import (
    ProfilingHarness,
    PerformanceProfiler,
    HotspotDetector,
    MemoryProfiler,
    CPUTimeProfiler,
)
from .benchmarks import (
    BenchmarkSuite,
    Microbenchmark,
    SystemBenchmark,
    PerformanceBudget,
    RegressionDetector,
)
from .optimizations import (
    OptimizationManager,
    FeatureGate,
    OptimizationConfig,
    PerformanceMetrics,
)
from .monitoring import (
    PerformanceMonitor,
    MetricsCollector,
    AlertManager,
    PerformanceDashboard,
)

__all__ = [
    # Profiling
    "ProfilingHarness",
    "PerformanceProfiler", 
    "HotspotDetector",
    "MemoryProfiler",
    "CPUTimeProfiler",
    # Benchmarks
    "BenchmarkSuite",
    "Microbenchmark",
    "SystemBenchmark", 
    "PerformanceBudget",
    "RegressionDetector",
    # Optimizations
    "OptimizationManager",
    "FeatureGate",
    "OptimizationConfig",
    "PerformanceMetrics",
    # Monitoring
    "PerformanceMonitor",
    "MetricsCollector",
    "AlertManager",
    "PerformanceDashboard",
]
