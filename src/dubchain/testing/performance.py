"""Performance testing infrastructure for DubChain.

This module provides performance testing capabilities including benchmarks,
profiling, load testing, and stress testing.
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

import psutil

from ..logging import get_logger
from .base import (
    BaseTestCase,
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    ExecutionType,
    RunnerManager,
    SuiteManager,
)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    # Timing metrics
    execution_time: float = 0.0
    cpu_time: float = 0.0
    wall_time: float = 0.0

    # Memory metrics
    memory_usage: float = 0.0
    peak_memory: float = 0.0
    memory_growth: float = 0.0

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0

    # I/O metrics
    disk_reads: int = 0
    disk_writes: int = 0
    network_sends: int = 0
    network_recvs: int = 0

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceTestCase(BaseTestCase):
    """Performance test case."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config = ExecutionConfig()
        self.config.test_type = ExecutionType.PERFORMANCE
        self.metrics = PerformanceMetrics()
        self.benchmark_function: Optional[Callable] = None
        self.iterations: int = 1
        self.warmup_iterations: int = 0

    def setup(self) -> None:
        """Setup performance test case."""
        super().setup()
        self._setup_metrics()

    def teardown(self) -> None:
        """Teardown performance test case."""
        self._cleanup_metrics()
        super().teardown()

    def _setup_metrics(self) -> None:
        """Setup performance metrics collection."""
        self.metrics = PerformanceMetrics()
        self.metrics.cpu_count = psutil.cpu_count()

    def _cleanup_metrics(self) -> None:
        """Cleanup performance metrics."""
        pass

    def set_benchmark_function(self, func: Callable) -> None:
        """Set benchmark function."""
        self.benchmark_function = func

    def set_iterations(self, iterations: int) -> None:
        """Set number of iterations."""
        self.iterations = iterations

    def set_warmup_iterations(self, warmup_iterations: int) -> None:
        """Set number of warmup iterations."""
        self.warmup_iterations = warmup_iterations

    def run_test(self) -> None:
        """Run performance test."""
        if not self.benchmark_function:
            raise ValueError("Benchmark function not set")

        # Warmup iterations
        for _ in range(self.warmup_iterations):
            try:
                self.benchmark_function()
            except Exception:
                pass

        # Performance measurement
        start_time = time.time()
        start_cpu_time = time.process_time()
        start_memory = psutil.Process().memory_info().rss

        for _ in range(self.iterations):
            try:
                self.benchmark_function()
            except Exception as e:
                raise AssertionError(f"Benchmark function failed: {e}")

        end_time = time.time()
        end_cpu_time = time.process_time()
        end_memory = psutil.Process().memory_info().rss

        # Calculate metrics
        self.metrics.execution_time = end_time - start_time
        self.metrics.cpu_time = end_cpu_time - start_cpu_time
        self.metrics.wall_time = self.metrics.execution_time
        self.metrics.memory_usage = end_memory
        self.metrics.memory_growth = end_memory - start_memory

        # Update result
        if self.result:
            self.result.duration = self.metrics.execution_time
            self.result.memory_usage = self.metrics.memory_usage
            self.result.custom_metrics = {
                "cpu_time": self.metrics.cpu_time,
                "memory_growth": self.metrics.memory_growth,
                "iterations": self.iterations,
            }


class BenchmarkSuite:
    """Benchmark suite for organizing benchmarks."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.benchmarks: List[PerformanceTestCase] = []
        self.results: List[ExecutionResult] = []

    def add_benchmark(self, benchmark: PerformanceTestCase) -> None:
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)

    def run_benchmarks(self) -> List[ExecutionResult]:
        """Run all benchmarks."""
        self.results.clear()

        for benchmark in self.benchmarks:
            try:
                result = benchmark.run()
                self.results.append(result)
            except Exception as e:
                error_result = ExecutionResult(
                    test_name=benchmark.name,
                    test_type=ExecutionType.PERFORMANCE,
                    status=ExecutionStatus.ERROR,
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0,
                    error_message=str(e),
                )
                self.results.append(error_result)

        return self.results.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {"total_benchmarks": 0}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == ExecutionStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == ExecutionStatus.FAILED)
        error = sum(1 for r in self.results if r.status == ExecutionStatus.ERROR)

        total_duration = sum(r.duration for r in self.results)
        total_memory = sum(r.memory_usage or 0 for r in self.results)

        return {
            "suite_name": self.name,
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / total if total > 0 else 0,
            "total_memory_usage": total_memory,
            "average_memory_usage": total_memory / total if total > 0 else 0,
        }


class ProfilerSuite:
    """Profiler suite for profiling code."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.profiles: List[Dict[str, Any]] = []

    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function."""
        import cProfile
        import io
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

        # Get profile stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats("cumulative")
        stats.print_stats()

        profile_data = {
            "function_name": func.__name__,
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "stats": stats_buffer.getvalue(),
            "timestamp": time.time(),
        }

        self.profiles.append(profile_data)
        return profile_data

    def get_profiles(self) -> List[Dict[str, Any]]:
        """Get all profiles."""
        return self.profiles.copy()

    def clear_profiles(self) -> None:
        """Clear all profiles."""
        self.profiles.clear()


class LoadTestSuite:
    """Load test suite for testing under load."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.load_tests: List[Callable] = []
        self.results: List[Dict[str, Any]] = []

    def add_load_test(self, test_func: Callable) -> None:
        """Add load test function."""
        self.load_tests.append(test_func)

    def run_load_test(
        self, concurrent_users: int = 10, duration: float = 60.0
    ) -> Dict[str, Any]:
        """Run load test with specified concurrent users and duration."""
        import concurrent.futures

        start_time = time.time()
        end_time = start_time + duration

        results = {
            "concurrent_users": concurrent_users,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
        }

        def run_test():
            while time.time() < end_time:
                for test_func in self.load_tests:
                    try:
                        start = time.time()
                        test_func()
                        end = time.time()

                        results["total_requests"] += 1
                        results["successful_requests"] += 1
                        results["response_times"].append(end - start)

                    except Exception as e:
                        results["total_requests"] += 1
                        results["failed_requests"] += 1
                        results["errors"].append(str(e))

        # Run concurrent users
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_users
        ) as executor:
            futures = [executor.submit(run_test) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)

        # Calculate statistics
        if results["response_times"]:
            results["average_response_time"] = sum(results["response_times"]) / len(
                results["response_times"]
            )
            results["min_response_time"] = min(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
        else:
            results["average_response_time"] = 0
            results["min_response_time"] = 0
            results["max_response_time"] = 0

        results["requests_per_second"] = results["total_requests"] / duration
        results["success_rate"] = (
            (results["successful_requests"] / results["total_requests"] * 100)
            if results["total_requests"] > 0
            else 0
        )

        self.results.append(results)
        return results

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all load test results."""
        return self.results.copy()

    def clear_results(self) -> None:
        """Clear all results."""
        self.results.clear()


class StressTestSuite:
    """Stress test suite for testing system limits."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.stress_tests: List[Callable] = []
        self.results: List[Dict[str, Any]] = []

    def add_stress_test(self, test_func: Callable) -> None:
        """Add stress test function."""
        self.stress_tests.append(test_func)

    def run_stress_test(
        self, max_iterations: int = 1000, timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Run stress test with maximum iterations and timeout."""
        start_time = time.time()
        end_time = start_time + timeout

        results = {
            "max_iterations": max_iterations,
            "timeout": timeout,
            "start_time": start_time,
            "end_time": end_time,
            "iterations_completed": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "errors": [],
            "peak_memory": 0,
            "peak_cpu": 0,
        }

        process = psutil.Process()

        for iteration in range(max_iterations):
            if time.time() >= end_time:
                break

            for test_func in self.stress_tests:
                try:
                    test_func()
                    results["successful_iterations"] += 1

                except Exception as e:
                    results["failed_iterations"] += 1
                    results["errors"].append(str(e))

                results["iterations_completed"] += 1

                # Monitor system resources
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()

                results["peak_memory"] = max(results["peak_memory"], memory_info.rss)
                results["peak_cpu"] = max(results["peak_cpu"], cpu_percent)

        # Calculate statistics
        results["success_rate"] = (
            (results["successful_iterations"] / results["iterations_completed"] * 100)
            if results["iterations_completed"] > 0
            else 0
        )
        results["iterations_per_second"] = results["iterations_completed"] / (
            time.time() - start_time
        )

        self.results.append(results)
        return results

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all stress test results."""
        return self.results.copy()

    def clear_results(self) -> None:
        """Clear all results."""
        self.results.clear()


class PerformanceTestSuite(SuiteManager):
    """Performance test suite."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config = ExecutionConfig()
        self.config.test_type = ExecutionType.PERFORMANCE

    def add_test(self, test: PerformanceTestCase) -> None:
        """Add performance test to suite."""
        super().add_test(test)


class PerformanceTestRunner(RunnerManager):
    """Performance test runner."""

    def __init__(self, config: ExecutionConfig = None):
        super().__init__(config)
        if self.config:
            self.config.test_type = ExecutionType.PERFORMANCE

    def run_with_profiling(self) -> List[ExecutionResult]:
        """Run tests with profiling."""
        # This would implement profiling integration
        # For now, we'll just run normally
        return self.run_all()

    def run_with_benchmarking(self, iterations: int = 100) -> List[ExecutionResult]:
        """Run tests with benchmarking."""
        # This would implement benchmarking integration
        # For now, we'll just run normally
        return self.run_all()
