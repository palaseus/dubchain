"""
CUDA-Powered Test Runner for DubChain.

This module provides ultra-fast test execution using CUDA acceleration,
parallel processing, and intelligent test batching.
"""

import time
import threading
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import os

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from .global_accelerator import get_global_accelerator, accelerate_batch, accelerate_async


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    duration: float
    error_message: Optional[str] = None
    gpu_accelerated: bool = False
    batch_id: Optional[str] = None


@dataclass
class TestBatch:
    """Batch of tests to execute together."""
    batch_id: str
    test_files: List[str]
    test_functions: List[str]
    priority: int = 0
    dependencies: List[str] = None
    gpu_accelerated: bool = False


class CUDATestRunner:
    """
    Ultra-fast CUDA-powered test runner.
    
    Features:
    - Parallel test execution across multiple processes
    - GPU acceleration for compatible tests
    - Intelligent test batching and dependency resolution
    - Real-time progress monitoring
    - Automatic retry and fallback mechanisms
    - Performance optimization suggestions
    """
    
    def __init__(self, 
                 max_workers: int = 8,
                 gpu_acceleration: bool = True,
                 batch_size: int = 50,
                 timeout: int = 300):
        """Initialize CUDA test runner."""
        self.max_workers = max_workers
        self.gpu_acceleration = gpu_acceleration
        self.batch_size = batch_size
        self.timeout = timeout
        
        self.accelerator = get_global_accelerator()
        self.results: List[TestResult] = []
        self.batches: List[TestBatch] = []
        
        # Performance tracking
        self.start_time = None
        self.total_tests = 0
        self.completed_tests = 0
        self.failed_tests = 0
        
        # Thread safety
        self._results_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        
        print(f"🚀 CUDA Test Runner initialized")
        print(f"   Max Workers: {max_workers}")
        print(f"   GPU Acceleration: {gpu_acceleration}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Timeout: {timeout}s")
    
    def discover_tests(self, test_dir: str = "tests") -> List[str]:
        """
        Discover all test files in the project.
        
        Args:
            test_dir: Directory to search for tests
            
        Returns:
            List of test file paths
        """
        test_files = []
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"⚠️  Test directory {test_dir} not found")
            return test_files
        
        # Find all test files
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend(test_path.rglob(pattern))
        
        test_files = [str(f) for f in test_files if f.is_file()]
        
        print(f"📁 Discovered {len(test_files)} test files")
        return test_files
    
    def analyze_test_dependencies(self, test_files: List[str]) -> Dict[str, List[str]]:
        """
        Analyze test dependencies to optimize execution order.
        
        Args:
            test_files: List of test file paths
            
        Returns:
            Dictionary mapping test files to their dependencies
        """
        dependencies = {}
        
        for test_file in test_files:
            deps = []
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    
                    # Simple dependency analysis
                    if 'import' in content:
                        # Extract imports that might be test dependencies
                        lines = content.split('\n')
                        for line in lines:
                            if line.strip().startswith('import ') or line.strip().startswith('from '):
                                # Extract module names
                                if 'test_' in line or '_test' in line:
                                    deps.append(line.strip())
            except Exception as e:
                print(f"⚠️  Error analyzing {test_file}: {e}")
            
            dependencies[test_file] = deps
        
        return dependencies
    
    def create_test_batches(self, test_files: List[str]) -> List[TestBatch]:
        """
        Create optimized test batches for parallel execution.
        
        Args:
            test_files: List of test file paths
            
        Returns:
            List of test batches
        """
        batches = []
        dependencies = self.analyze_test_dependencies(test_files)
        
        # Group tests by type and dependencies
        unit_tests = []
        integration_tests = []
        api_tests = []
        bridge_tests = []
        performance_tests = []
        security_tests = []
        
        for test_file in test_files:
            if 'unit' in test_file:
                unit_tests.append(test_file)
            elif 'integration' in test_file:
                integration_tests.append(test_file)
            elif 'api' in test_file:
                api_tests.append(test_file)
            elif 'bridge' in test_file:
                bridge_tests.append(test_file)
            elif 'performance' in test_file:
                performance_tests.append(test_file)
            elif 'security' in test_file:
                security_tests.append(test_file)
            else:
                unit_tests.append(test_file)  # Default to unit tests
        
        # Create batches for each test type
        test_groups = [
            (unit_tests, "unit", 1),
            (integration_tests, "integration", 2),
            (api_tests, "api", 3),
            (bridge_tests, "bridge", 4),
            (performance_tests, "performance", 5),
            (security_tests, "security", 6),
        ]
        
        batch_id = 0
        for test_group, group_name, priority in test_groups:
            if not test_group:
                continue
            
            # Split large groups into smaller batches
            for i in range(0, len(test_group), self.batch_size):
                batch_tests = test_group[i:i + self.batch_size]
                
                batch = TestBatch(
                    batch_id=f"{group_name}_{batch_id}",
                    test_files=batch_tests,
                    test_functions=[],
                    priority=priority,
                    dependencies=dependencies.get(batch_tests[0], []) if batch_tests else [],
                    gpu_accelerated=self.gpu_acceleration and group_name in ['unit', 'performance']
                )
                
                batches.append(batch)
                batch_id += 1
        
        # Sort batches by priority
        batches.sort(key=lambda b: b.priority)
        
        print(f"📦 Created {len(batches)} test batches")
        return batches
    
    def execute_test_batch(self, batch: TestBatch) -> List[TestResult]:
        """
        Execute a batch of tests with CUDA acceleration.
        
        Args:
            batch: Test batch to execute
            
        Returns:
            List of test results
        """
        results = []
        
        if not batch.test_files:
            return results
        
        print(f"🔄 Executing batch {batch.batch_id} ({len(batch.test_files)} tests)")
        
        try:
            if batch.gpu_accelerated and self.accelerator.available:
                # Use GPU acceleration for compatible tests
                results = self._execute_batch_gpu(batch)
            else:
                # Use parallel CPU execution
                results = self._execute_batch_cpu(batch)
            
            print(f"✅ Batch {batch.batch_id} completed: {len(results)} results")
            
        except Exception as e:
            print(f"❌ Batch {batch.batch_id} failed: {e}")
            # Create error results for all tests in batch
            for test_file in batch.test_files:
                result = TestResult(
                    test_name=test_file,
                    status='error',
                    duration=0.0,
                    error_message=str(e),
                    gpu_accelerated=batch.gpu_accelerated,
                    batch_id=batch.batch_id
                )
                results.append(result)
        
        return results
    
    def run_all_tests(self, test_dir: str = "tests") -> Dict[str, Any]:
        """
        Run all tests with CUDA acceleration.
        
        Args:
            test_dir: Directory containing tests
            
        Returns:
            Dictionary with test results and statistics
        """
        print("🚀 Starting CUDA-accelerated test run")
        self.start_time = time.time()
        
        # Discover tests
        test_files = self.discover_tests(test_dir)
        if not test_files:
            print("❌ No tests found")
            return {"error": "No tests found"}
        
        self.total_tests = len(test_files)
        
        # Create batches
        self.batches = self.create_test_batches(test_files)
        
        # Execute batches in parallel
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.execute_test_batch, batch): batch 
                for batch in self.batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.timeout)
                    all_results.extend(batch_results)
                    
                    # Update progress
                    with self._progress_lock:
                        self.completed_tests += len(batch_results)
                        self.failed_tests += sum(1 for r in batch_results if r.status in ['failed', 'error'])
                        
                        progress = (self.completed_tests / self.total_tests) * 100
                        print(f"📊 Progress: {progress:.1f}% ({self.completed_tests}/{self.total_tests})")
                        
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Batch {batch.batch_id} timed out")
                except Exception as e:
                    print(f"❌ Batch {batch.batch_id} failed: {e}")
        
        # Store results
        with self._results_lock:
            self.results = all_results
        
        # Calculate final statistics
        duration = time.time() - self.start_time
        stats = self._calculate_statistics(duration)
        
        print(f"🎉 Test run completed in {duration:.2f}s")
        print(f"   Total: {stats['total']}")
        print(f"   Passed: {stats['passed']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Skipped: {stats['skipped']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        
        if self.accelerator.available:
            metrics = self.accelerator.get_performance_metrics()
            print(f"   GPU Operations: {metrics.gpu_operations}")
            print(f"   Speedup Factor: {metrics.speedup_factor:.1f}x")
        
        return stats
    
    def run_specific_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """
        Run specific test files with CUDA acceleration.
        
        Args:
            test_files: List of test file paths to run
            
        Returns:
            Dictionary with test results and statistics
        """
        print(f"🎯 Running {len(test_files)} specific test files")
        self.start_time = time.time()
        
        # Create batches for specific tests
        self.batches = self.create_test_batches(test_files)
        self.total_tests = len(test_files)
        
        # Execute batches
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.execute_test_batch, batch): batch 
                for batch in self.batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.timeout)
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"❌ Batch {batch.batch_id} failed: {e}")
        
        self.results = all_results
        
        duration = time.time() - self.start_time
        stats = self._calculate_statistics(duration)
        
        return stats
    
    def _execute_batch_gpu(self, batch: TestBatch) -> List[TestResult]:
        """Execute batch with GPU acceleration."""
        results = []
        
        # Create test operations for GPU acceleration
        test_operations = []
        for test_file in batch.test_files:
            def run_test(file_path=test_file):
                return self._run_single_test(file_path)
            test_operations.append(run_test)
        
        # Execute with GPU acceleration
        try:
            batch_results = accelerate_batch(test_operations)
            
            for i, result in enumerate(batch_results):
                test_result = TestResult(
                    test_name=batch.test_files[i],
                    status=result.get('status', 'unknown'),
                    duration=result.get('duration', 0.0),
                    error_message=result.get('error'),
                    gpu_accelerated=True,
                    batch_id=batch.batch_id
                )
                results.append(test_result)
                
        except Exception as e:
            # Fallback to CPU execution
            print(f"⚠️  GPU execution failed for batch {batch.batch_id}, falling back to CPU: {e}")
            results = self._execute_batch_cpu(batch)
        
        return results
    
    def _execute_batch_cpu(self, batch: TestBatch) -> List[TestResult]:
        """Execute batch with CPU parallel processing."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch.test_files), 4)) as executor:
            future_to_test = {
                executor.submit(self._run_single_test, test_file): test_file 
                for test_file in batch.test_files
            }
            
            for future in concurrent.futures.as_completed(future_to_test):
                test_file = future_to_test[future]
                try:
                    result = future.result(timeout=30)
                    test_result = TestResult(
                        test_name=test_file,
                        status=result.get('status', 'unknown'),
                        duration=result.get('duration', 0.0),
                        error_message=result.get('error'),
                        gpu_accelerated=False,
                        batch_id=batch.batch_id
                    )
                    results.append(test_result)
                except Exception as e:
                    test_result = TestResult(
                        test_name=test_file,
                        status='error',
                        duration=0.0,
                        error_message=str(e),
                        gpu_accelerated=False,
                        batch_id=batch.batch_id
                    )
                    results.append(test_result)
        
        return results
    
    def _run_single_test(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file."""
        start_time = time.time()
        
        try:
            # Use pytest to run the test file
            result = subprocess.run([
                sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=60)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = 'passed'
                error_message = None
            else:
                status = 'failed'
                error_message = result.stderr
            
            return {
                'status': status,
                'duration': duration,
                'error': error_message,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': 'Test timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def _calculate_statistics(self, duration: float) -> Dict[str, Any]:
        """Calculate test statistics."""
        if not self.results:
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'success_rate': 0.0,
                'duration': duration,
                'gpu_accelerated': 0,
                'avg_duration': 0.0
            }
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == 'passed')
        failed = sum(1 for r in self.results if r.status == 'failed')
        errors = sum(1 for r in self.results if r.status == 'error')
        skipped = sum(1 for r in self.results if r.status == 'skipped')
        gpu_accelerated = sum(1 for r in self.results if r.gpu_accelerated)
        
        success_rate = (passed / total) * 100 if total > 0 else 0.0
        avg_duration = sum(r.duration for r in self.results) / total if total > 0 else 0.0
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'success_rate': success_rate,
            'duration': duration,
            'gpu_accelerated': gpu_accelerated,
            'avg_duration': avg_duration
        }
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get list of failed tests."""
        return [r for r in self.results if r.status in ['failed', 'error']]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        if not self.results:
            return {}
        
        stats = self._calculate_statistics(0)
        metrics = self.accelerator.get_performance_metrics()
        
        return {
            'test_statistics': stats,
            'acceleration_metrics': {
                'total_operations': metrics.total_operations,
                'gpu_operations': metrics.gpu_operations,
                'cpu_fallbacks': metrics.cpu_fallbacks,
                'speedup_factor': metrics.speedup_factor,
                'avg_gpu_time': metrics.avg_gpu_time,
                'avg_cpu_time': metrics.avg_cpu_time
            },
            'optimization_suggestions': self.accelerator.optimize_performance()
        }


def run_tests_cuda(test_dir: str = "tests", 
                   max_workers: int = 8,
                   gpu_acceleration: bool = True,
                   batch_size: int = 50) -> Dict[str, Any]:
    """
    Convenience function to run tests with CUDA acceleration.
    
    Args:
        test_dir: Directory containing tests
        max_workers: Maximum number of parallel workers
        gpu_acceleration: Enable GPU acceleration
        batch_size: Batch size for test execution
        
    Returns:
        Dictionary with test results and statistics
    """
    runner = CUDATestRunner(
        max_workers=max_workers,
        gpu_acceleration=gpu_acceleration,
        batch_size=batch_size
    )
    
    return runner.run_all_tests(test_dir)


def run_specific_tests_cuda(test_files: List[str],
                          max_workers: int = 8,
                          gpu_acceleration: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run specific tests with CUDA acceleration.
    
    Args:
        test_files: List of test file paths
        max_workers: Maximum number of parallel workers
        gpu_acceleration: Enable GPU acceleration
        
    Returns:
        Dictionary with test results and statistics
    """
    runner = CUDATestRunner(
        max_workers=max_workers,
        gpu_acceleration=gpu_acceleration
    )
    
    return runner.run_specific_tests(test_files)


if __name__ == "__main__":
    # Example usage
    print("🚀 CUDA Test Runner Demo")
    
    runner = CUDATestRunner(max_workers=4, gpu_acceleration=True)
    results = runner.run_all_tests("tests")
    
    print(f"\n📊 Final Results:")
    print(f"   Success Rate: {results['success_rate']:.1f}%")
    print(f"   Duration: {results['duration']:.2f}s")
    
    if results['failed'] > 0:
        print(f"\n❌ Failed Tests:")
        failed_tests = runner.get_failed_tests()
        for test in failed_tests[:5]:  # Show first 5 failed tests
            print(f"   - {test.test_name}: {test.error_message}")
    
    performance_report = runner.get_performance_report()
    print(f"\n🚀 Performance Report:")
    print(f"   GPU Operations: {performance_report['acceleration_metrics']['gpu_operations']}")
    print(f"   Speedup Factor: {performance_report['acceleration_metrics']['speedup_factor']:.1f}x")
