"""
Comprehensive performance test suite for DubChain optimizations.

This module provides:
- Performance regression tests
- Optimization effectiveness tests
- Stress tests for performance optimizations
- Integration tests for performance subsystems
- Automated performance validation
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import gc
import json
import os
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import statistics

from src.dubchain.performance.profiling import (
    ProfilingHarness,
    ProfilingConfig,
    PerformanceProfiler,
)
from src.dubchain.performance.benchmarks import (
    BenchmarkSuite,
    BenchmarkConfig,
    Microbenchmark,
    SystemBenchmark,
    RegressionDetector,
)
from src.dubchain.performance.optimizations import (
    OptimizationManager,
    OptimizationConfig,
    OptimizationType,
    ConsensusOptimizations,
    NetworkOptimizations,
    VMOptimizations,
    StorageOptimizations,
    CryptoOptimizations,
    MemoryOptimizations,
    BatchingOptimizations,
)
from src.dubchain.performance.monitoring import (
    PerformanceMonitor,
    MetricsCollector,
    AlertManager,
    PerformanceThreshold,
    AlertSeverity,
)


class PerformanceTestBase:
    """Base class for performance tests."""
    
    @pytest.fixture(autouse=True)
    def setup_performance_test(self):
        """Setup for performance tests."""
        # Force garbage collection before each test
        gc.collect()
        
        # Create test directories
        os.makedirs("test_profiling_artifacts", exist_ok=True)
        os.makedirs("test_benchmark_results", exist_ok=True)
        
        yield
        
        # Cleanup after test
        gc.collect()


class TestProfilingHarness(PerformanceTestBase):
    """Test profiling harness functionality."""
    
    def test_profiling_config(self):
        """Test profiling configuration."""
        config = ProfilingConfig(
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
            output_directory="test_profiling_artifacts",
        )
        
        assert config.enable_cpu_profiling
        assert config.enable_memory_profiling
        assert config.output_directory == "test_profiling_artifacts"
        
    def test_cpu_profiling(self):
        """Test CPU profiling functionality."""
        config = ProfilingConfig(
            enable_cpu_profiling=True,
            enable_memory_profiling=False,
            output_directory="test_profiling_artifacts",
        )
        
        profiler = PerformanceProfiler(config)
        
        def test_function():
            time.sleep(0.01)  # 10ms
            return sum(range(1000))
            
        result = profiler.profile_function(test_function)
        
        assert result.duration > 0
        assert result.total_cpu_time > 0
        assert len(result.cpu_functions) > 0
        
    def test_memory_profiling(self):
        """Test memory profiling functionality."""
        config = ProfilingConfig(
            enable_cpu_profiling=False,
            enable_memory_profiling=True,
            output_directory="test_profiling_artifacts",
        )
        
        profiler = PerformanceProfiler(config)
        
        def memory_intensive_function():
            data = []
            for i in range(10000):
                data.append([i] * 100)
            return len(data)
            
        result = profiler.profile_function(memory_intensive_function)
        
        assert result.memory_peak > 0
        assert result.memory_allocations > 0
        
    def test_hotspot_detection(self):
        """Test hotspot detection."""
        config = ProfilingConfig(
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
            hotspot_threshold=0.1,  # 10%
            output_directory="test_profiling_artifacts",
        )
        
        profiler = PerformanceProfiler(config)
        
        def hotspot_function():
            time.sleep(0.05)  # 50ms - should be detected as hotspot
            return "hotspot"
            
        def normal_function():
            time.sleep(0.001)  # 1ms
            return "normal"
            
        # Profile hotspot function
        result1 = profiler.profile_function(hotspot_function)
        
        # Profile normal function
        result2 = profiler.profile_function(normal_function)
        
        # Hotspot function should have hotspots detected
        assert len(result1.cpu_hotspots) > 0 or len(result1.memory_hotspots) > 0
        
    def test_profiling_context_manager(self):
        """Test profiling context manager."""
        config = ProfilingConfig(
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
            output_directory="test_profiling_artifacts",
        )
        
        profiler = PerformanceProfiler(config)
        
        with profiler.profile_context("test_context"):
            time.sleep(0.01)
            data = [i for i in range(1000)]
            
        assert len(profiler.results) == 1
        result = profiler.results[0]
        assert result.session_id == "test_context"
        assert result.duration > 0


class TestBenchmarkSuite(PerformanceTestBase):
    """Test benchmark suite functionality."""
    
    def test_microbenchmark(self):
        """Test microbenchmark functionality."""
        config = BenchmarkConfig(
            warmup_iterations=2,
            min_iterations=5,
            max_iterations=10,
            output_directory="test_benchmark_results",
        )
        
        benchmark = Microbenchmark(config)
        
        def test_function():
            return sum(range(100))
            
        result = benchmark.benchmark_function(test_function, "test_function")
        
        assert result.name == "test_function"
        assert result.iterations >= config.min_iterations
        assert result.iterations <= config.max_iterations
        assert result.total_time > 0
        assert result.throughput > 0
        assert result.mean_time > 0
        
    def test_system_benchmark(self):
        """Test system benchmark functionality."""
        config = BenchmarkConfig(
            warmup_iterations=1,
            min_iterations=3,
            max_iterations=5,
            output_directory="test_benchmark_results",
        )
        
        benchmark = SystemBenchmark(config)
        
        def system_workflow():
            time.sleep(0.01)
            return "system_result"
            
        result = benchmark.benchmark_workflow(system_workflow, "system_workflow")
        
        assert result.name == "system_workflow"
        assert result.iterations >= config.min_iterations
        assert result.iterations <= config.max_iterations
        
    def test_concurrent_benchmark(self):
        """Test concurrent benchmark functionality."""
        config = BenchmarkConfig(
            warmup_iterations=1,
            min_iterations=2,
            max_iterations=3,
            output_directory="test_benchmark_results",
        )
        
        benchmark = SystemBenchmark(config)
        
        def concurrent_workload():
            time.sleep(0.005)
            return "concurrent_result"
            
        result = benchmark.benchmark_concurrent_workload(
            concurrent_workload, "concurrent_test", num_threads=2, operations_per_thread=5
        )
        
        assert "concurrent" in result.name
        assert result.iterations > 0
        
    def test_performance_budget_violations(self):
        """Test performance budget violation detection."""
        config = BenchmarkConfig(
            max_latency_ms=1.0,  # Very strict budget
            min_throughput_ops_per_sec=10000,  # Very high requirement
            output_directory="test_benchmark_results",
        )
        
        benchmark = Microbenchmark(config)
        
        def slow_function():
            time.sleep(0.01)  # 10ms - violates 1ms budget
            return "slow"
            
        result = benchmark.benchmark_function(slow_function, "slow_function")
        
        # Should have budget violations
        assert len(result.budget_violations) > 0
        
    def test_regression_detection(self):
        """Test regression detection."""
        config = BenchmarkConfig(
            baseline_file="test_baseline.json",
            regression_threshold_percent=5.0,
            output_directory="test_benchmark_results",
        )
        
        detector = RegressionDetector(config)
        
        # Create fake baseline
        baseline_data = {
            "test_function": {
                "name": "test_function",
                "function_name": "test_function",
                "iterations": 10,
                "total_time": 1.0,
                "min_time": 0.1,
                "max_time": 0.1,
                "mean_time": 0.1,
                "median_time": 0.1,
                "std_dev": 0.0,
                "throughput": 10.0,
                "memory_usage_mb": 1.0,
                "memory_peak_mb": 1.0,
                "memory_growth_mb": 0.0,
                "cpu_usage_percent": 10.0,
                "confidence_interval": (0.1, 0.1),
                "confidence_level": 0.95,
                "budget_violations": [],
                "timestamp": time.time(),
                "metadata": {},
            }
        }
        
        # Save baseline
        baseline_dir = os.path.dirname(config.baseline_file)
        if baseline_dir:  # Only create directory if path is not empty
            os.makedirs(baseline_dir, exist_ok=True)
        with open(config.baseline_file, 'w') as f:
            json.dump(baseline_data, f)
            
        # Load baseline
        detector.load_baseline()
        
        # Create current results (with regression)
        from src.dubchain.performance.benchmarks import BenchmarkResult
        
        current_results = [
            BenchmarkResult(
                name="test_function",
                function_name="test_function",
                iterations=10,
                total_time=1.0,
                min_time=0.1,
                max_time=0.1,
                mean_time=0.15,  # 50% regression
                median_time=0.15,
                std_dev=0.0,
                throughput=6.67,  # Regression
                memory_usage_mb=1.0,
                memory_peak_mb=1.0,
                memory_growth_mb=0.0,
                cpu_usage_percent=10.0,
                confidence_interval=(0.15, 0.15),
                confidence_level=0.95,
            )
        ]
        
        regressions = detector.detect_regressions(current_results)
        
        assert len(regressions["regressions"]) > 0
        assert regressions["summary"]["regressions_count"] > 0


class TestOptimizationManager(PerformanceTestBase):
    """Test optimization manager functionality."""
    
    def test_optimization_registration(self):
        """Test optimization registration."""
        manager = OptimizationManager()
        
        config = OptimizationConfig(
            name="test_optimization",
            optimization_type=OptimizationType.CONSENSUS_BATCHING,
            enabled=False,
        )
        
        manager.register_optimization(config)
        
        assert "test_optimization" in manager.optimizations
        assert "test_optimization" in manager.feature_gates
        
    def test_optimization_enable_disable(self):
        """Test optimization enable/disable."""
        manager = OptimizationManager()
        
        config = OptimizationConfig(
            name="test_optimization",
            optimization_type=OptimizationType.CONSENSUS_BATCHING,
            enabled=False,
        )
        
        manager.register_optimization(config)
        
        # Initially disabled
        assert not manager.is_optimization_enabled("test_optimization")
        
        # Enable
        assert manager.enable_optimization("test_optimization")
        assert manager.is_optimization_enabled("test_optimization")
        
        # Disable
        assert manager.disable_optimization("test_optimization")
        assert not manager.is_optimization_enabled("test_optimization")
        
    def test_optimization_dependencies(self):
        """Test optimization dependencies."""
        manager = OptimizationManager()
        
        # Register dependency
        dep_config = OptimizationConfig(
            name="dependency",
            optimization_type=OptimizationType.CONSENSUS_BATCHING,
            enabled=False,
        )
        manager.register_optimization(dep_config)
        
        # Register optimization with dependency
        config = OptimizationConfig(
            name="dependent_optimization",
            optimization_type=OptimizationType.NETWORK_BATCHING,
            enabled=False,
            dependencies=["dependency"],
        )
        manager.register_optimization(config)
        
        # Try to enable without dependency
        assert not manager.enable_optimization("dependent_optimization")
        assert not manager.is_optimization_enabled("dependent_optimization")
        
        # Enable dependency first
        assert manager.enable_optimization("dependency")
        
        # Now should be able to enable dependent optimization
        assert manager.enable_optimization("dependent_optimization")
        assert manager.is_optimization_enabled("dependent_optimization")
        
    def test_optimization_conflicts(self):
        """Test optimization conflicts."""
        manager = OptimizationManager()
        
        # Register first optimization
        config1 = OptimizationConfig(
            name="optimization1",
            optimization_type=OptimizationType.CONSENSUS_BATCHING,
            enabled=False,
        )
        manager.register_optimization(config1)
        
        # Register conflicting optimization
        config2 = OptimizationConfig(
            name="optimization2",
            optimization_type=OptimizationType.NETWORK_BATCHING,
            enabled=False,
            conflicts=["optimization1"],
        )
        manager.register_optimization(config2)
        
        # Enable first optimization
        assert manager.enable_optimization("optimization1")
        
        # Try to enable conflicting optimization
        assert not manager.enable_optimization("optimization2")
        assert not manager.is_optimization_enabled("optimization2")
        
    def test_config_export_import(self):
        """Test configuration export/import."""
        manager = OptimizationManager()
        
        # Export config
        config_path = "test_optimization_config.json"
        manager.export_config(config_path)
        
        assert os.path.exists(config_path)
        
        # Import config
        manager2 = OptimizationManager()
        manager2.import_config(config_path)
        
        # Should have same optimizations
        assert len(manager2.optimizations) == len(manager.optimizations)
        
        # Cleanup
        os.remove(config_path)


class TestOptimizationImplementations(PerformanceTestBase):
    """Test optimization implementations."""
    
    def test_consensus_optimizations(self):
        """Test consensus optimizations."""
        manager = OptimizationManager()
        consensus_opt = ConsensusOptimizations(manager)
        
        # Test with optimization disabled (fallback)
        blocks = [{"data": f"block_{i}"} for i in range(5)]
        results = consensus_opt.batch_block_validation(blocks)
        
        assert len(results) == len(blocks)
        
        # Enable optimization
        manager.enable_optimization("consensus_batching")
        
        # Test with optimization enabled
        results = consensus_opt.batch_block_validation(blocks)
        assert len(results) == len(blocks)
        
    def test_network_optimizations(self):
        """Test network optimizations."""
        manager = OptimizationManager()
        network_opt = NetworkOptimizations(manager)
        
        async def test_network_optimizations():
            messages = [{"data": f"message_{i}"} for i in range(5)]
            
            # Test with optimization disabled
            results = await network_opt.batch_message_sending(messages)
            assert len(results) == len(messages)
            
            # Enable optimization
            manager.enable_optimization("network_batching")
            
            # Test with optimization enabled
            results = await network_opt.batch_message_sending(messages)
            assert len(results) == len(messages)
            
        asyncio.run(test_network_optimizations())
        
    def test_vm_optimizations(self):
        """Test VM optimizations."""
        manager = OptimizationManager()
        vm_opt = VMOptimizations(manager)
        
        # Enable VM bytecode caching optimization
        manager.enable_optimization("vm_bytecode_caching")
        
        # Test bytecode caching
        contract_hash = "test_contract_hash"
        bytecode = {"instructions": ["PUSH", "ADD", "RET"]}
        
        # Cache bytecode
        vm_opt.cache_bytecode(contract_hash, bytecode)
        
        # Retrieve cached bytecode
        cached = vm_opt.get_cached_bytecode(contract_hash)
        assert cached == bytecode
        
        # Test gas optimization
        original_gas = 1000
        optimized_gas = vm_opt.optimize_gas_usage("ADD", original_gas)
        assert optimized_gas <= original_gas
        
    def test_storage_optimizations(self):
        """Test storage optimizations."""
        manager = OptimizationManager()
        storage_opt = StorageOptimizations(manager)
        
        # Test serialization
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        # Serialize
        serialized = storage_opt.serialize_data(test_data)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = storage_opt.deserialize_data(serialized)
        assert deserialized == test_data
        
        # Test batch operations
        operations = [{"op": f"write_{i}"} for i in range(5)]
        results = storage_opt.batch_write_operations(operations)
        assert len(results) == len(operations)
        
    def test_crypto_optimizations(self):
        """Test crypto optimizations."""
        manager = OptimizationManager()
        crypto_opt = CryptoOptimizations(manager)
        
        # Test parallel verification
        signatures = [{"sig": f"signature_{i}"} for i in range(5)]
        results = crypto_opt.parallel_signature_verification(signatures)
        assert len(results) == len(signatures)
        
        # Test cached verification
        signature_hash = "test_hash"
        signature = {"data": "test_signature"}
        
        result1 = crypto_opt.cached_verification(signature_hash, signature)
        result2 = crypto_opt.cached_verification(signature_hash, signature)
        
        assert result1 == result2  # Should be cached
        
    def test_memory_optimizations(self):
        """Test memory optimizations."""
        manager = OptimizationManager()
        memory_opt = MemoryOptimizations(manager)
        
        # Enable memory optimizations
        manager.enable_optimization("memory_buffer_reuse")
        manager.enable_optimization("memory_gc_tuning")
        
        # Test buffer reuse
        size = 1024
        buffer1 = memory_opt.get_reusable_buffer(size)
        assert len(buffer1) == size
        
        # Return buffer
        memory_opt.return_buffer(buffer1)
        
        # Get another buffer of same size
        buffer2 = memory_opt.get_reusable_buffer(size)
        assert len(buffer2) == size
        
        # Test GC tuning
        memory_opt.optimize_gc_settings()
        assert memory_opt.gc_tuning_enabled
        
        memory_opt.restore_gc_settings()
        assert not memory_opt.gc_tuning_enabled
        
    def test_batching_optimizations(self):
        """Test batching optimizations."""
        manager = OptimizationManager()
        batching_opt = BatchingOptimizations(manager)
        
        # Test state writes
        shard_id = "shard_1"
        writes = [{"key": f"key_{i}", "value": f"value_{i}"} for i in range(5)]
        
        results = batching_opt.batch_state_writes(shard_id, writes)
        assert len(results) == len(writes)
        
        # Test signature aggregation
        signatures = [{"sig": f"signature_{i}"} for i in range(5)]
        aggregated = batching_opt.aggregate_signatures(signatures)
        assert aggregated is not None


class TestPerformanceMonitoring(PerformanceTestBase):
    """Test performance monitoring functionality."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        # Record various metrics
        collector.record_counter("test_counter", 1.0)
        collector.record_gauge("test_gauge", 50.0)
        collector.record_timer("test_timer", 0.1)
        collector.record_histogram("test_histogram", 25.0)
        
        # Get latest metrics
        counter = collector.get_latest_metric("test_counter")
        assert counter is not None
        assert counter.value == 1.0
        assert counter.metric_type.value == "counter"
        
        # Get metric series
        series = collector.get_metric_series("test_gauge")
        assert len(series) == 1
        assert series[0].value == 50.0
        
        # Get statistics
        stats = collector.get_metric_statistics("test_timer")
        assert stats["count"] == 1
        assert stats["mean"] == 0.1
        
    def test_alert_manager(self):
        """Test alert management."""
        collector = MetricsCollector()
        alert_manager = AlertManager(collector)
        
        # Add threshold
        threshold = PerformanceThreshold(
            metric_name="test_metric",
            threshold_value=100.0,
            comparison_operator=">",
            severity=AlertSeverity.WARNING,
        )
        alert_manager.add_threshold(threshold)
        
        # Record metric below threshold
        collector.record_gauge("test_metric", 50.0)
        alerts = alert_manager.check_thresholds()
        assert len(alerts) == 0
        
        # Record metric above threshold
        collector.record_gauge("test_metric", 150.0)
        alerts = alert_manager.check_thresholds()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        
    def test_performance_monitor(self):
        """Test performance monitor."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.2)  # Let it collect some metrics
        
        # Record application metrics
        monitor.record_block_creation_time(0.1)
        monitor.record_transaction_throughput(100.0)
        monitor.record_consensus_time(0.05)
        
        # Get performance summary
        summary = monitor.get_performance_summary(time_window=1.0)
        
        assert "system_metrics" in summary
        assert "application_metrics" in summary
        assert "active_alerts" in summary
        
        # Stop monitoring
        monitor.stop_monitoring()


class TestPerformanceIntegration(PerformanceTestBase):
    """Integration tests for performance optimization system."""
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Setup
        manager = OptimizationManager()
        profiler = PerformanceProfiler(ProfilingConfig(output_directory="test_profiling_artifacts"))
        benchmark = Microbenchmark(BenchmarkConfig(output_directory="test_benchmark_results"))
        monitor = PerformanceMonitor()
        
        # Test function
        def test_workload():
            time.sleep(0.01)
            data = [i for i in range(1000)]
            return sum(data)
            
        # Profile baseline
        baseline_result = profiler.profile_function(test_workload)
        
        # Benchmark baseline
        baseline_benchmark = benchmark.benchmark_function(test_workload, "baseline")
        
        # Enable optimizations
        manager.enable_optimization("memory_allocation_reduction")
        
        # Profile optimized
        optimized_result = profiler.profile_function(test_workload)
        
        # Benchmark optimized
        optimized_benchmark = benchmark.benchmark_function(test_workload, "optimized")
        
        # Compare results
        assert baseline_result.duration > 0
        assert optimized_result.duration > 0
        assert baseline_benchmark.mean_time > 0
        assert optimized_benchmark.mean_time > 0
        
        # Monitor should have collected metrics
        summary = monitor.get_performance_summary()
        assert "system_metrics" in summary
        
    def test_performance_regression_detection(self):
        """Test performance regression detection in CI."""
        # Create baseline
        baseline_config = BenchmarkConfig(
            baseline_file="test_baseline.json",
            output_directory="test_benchmark_results",
        )
        
        baseline_benchmark = Microbenchmark(baseline_config)
        
        def baseline_function():
            return sum(range(100))
            
        baseline_result = baseline_benchmark.benchmark_function(baseline_function, "test_function")
        
        # Save baseline
        baseline_benchmark.results = [baseline_result]
        # Create regression detector and save baseline
        detector = RegressionDetector(baseline_config)
        detector.save_baseline(baseline_benchmark.results)
        
        # Simulate regression
        def regressed_function():
            time.sleep(0.01)  # Add delay to simulate regression
            return sum(range(100))
            
        current_benchmark = Microbenchmark(baseline_config)
        current_result = current_benchmark.benchmark_function(regressed_function, "test_function")
        
        # Detect regressions
        detector = RegressionDetector(baseline_config)
        detector.load_baseline()
        regressions = detector.detect_regressions([current_result])
        
        # Should detect regression
        assert len(regressions["regressions"]) > 0
        
        # Cleanup
        if os.path.exists("test_baseline.json"):
            os.remove("test_baseline.json")


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,
]


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
