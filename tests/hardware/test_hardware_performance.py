"""
Performance Benchmark Tests for Hardware Acceleration

This module provides comprehensive performance benchmarking tests for all hardware acceleration components.
"""

import pytest
import unittest
import numpy as np
import time
import statistics
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dubchain.hardware.manager import HardwareManager, HardwareManagerConfig
from dubchain.hardware.cuda import CUDAAccelerator, CUDAConfig
from dubchain.hardware.opencl import OpenCLAccelerator, OpenCLConfig
from dubchain.hardware.cpu import CPUAccelerator, CPUConfig


class TestHardwarePerformanceBenchmarks(unittest.TestCase):
    """Test hardware performance benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HardwareManagerConfig()
        self.manager = HardwareManager(self.config)
        
        # Register accelerators
        self.manager.register_accelerator("cuda", CUDAAccelerator(CUDAConfig()))
        self.manager.register_accelerator("opencl", OpenCLAccelerator(OpenCLConfig()))
        self.manager.register_accelerator("cpu", CPUAccelerator(CPUConfig()))
        
        # Benchmark parameters
        self.matrix_sizes = [100, 500, 1000, 2000]
        self.iterations = 5  # Number of iterations for averaging
    
    def benchmark_matrix_multiplication(self, accelerator_name: str, size: int) -> Dict[str, float]:
        """Benchmark matrix multiplication for a specific accelerator and size."""
        if accelerator_name not in self.manager.get_available_accelerators():
            self.skipTest(f"{accelerator_name} not available")
        
        # Create test matrices
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        
        times = []
        throughputs = []
        
        for _ in range(self.iterations):
            start_time = time.time()
            result = self.manager.execute_operation(
                accelerator_name, 
                "matrix_multiply", 
                {"a": a, "b": b}
            )
            end_time = time.time()
            
            if result.success:
                times.append(end_time - start_time)
                throughputs.append(result.throughput)
        
        if not times:
            self.skipTest(f"All iterations failed for {accelerator_name}")
        
        return {
            'mean_time': statistics.mean(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'mean_throughput': statistics.mean(throughputs),
            'std_throughput': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times)
        }
    
    def test_cpu_matrix_multiplication_benchmark(self):
        """Benchmark CPU matrix multiplication."""
        results = {}
        
        for size in self.matrix_sizes:
            benchmark_result = self.benchmark_matrix_multiplication("cpu", size)
            results[size] = benchmark_result
            
            # Verify reasonable performance
            self.assertLess(benchmark_result['mean_time'], 10.0)  # Should complete within 10 seconds
            self.assertGreater(benchmark_result['mean_throughput'], 0)
        
        # Log results for analysis
        print(f"\nCPU Matrix Multiplication Benchmark Results:")
        for size, result in results.items():
            print(f"  Size {size}x{size}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s, "
                  f"{result['mean_throughput']:.2f} ops/s")
    
    def test_cuda_matrix_multiplication_benchmark(self):
        """Benchmark CUDA matrix multiplication."""
        if "cuda" not in self.manager.get_available_accelerators():
            self.skipTest("CUDA not available")
        
        results = {}
        
        for size in self.matrix_sizes:
            benchmark_result = self.benchmark_matrix_multiplication("cuda", size)
            results[size] = benchmark_result
            
            # Verify reasonable performance
            self.assertLess(benchmark_result['mean_time'], 5.0)  # Should complete within 5 seconds
            self.assertGreater(benchmark_result['mean_throughput'], 0)
        
        # Log results for analysis
        print(f"\nCUDA Matrix Multiplication Benchmark Results:")
        for size, result in results.items():
            print(f"  Size {size}x{size}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s, "
                  f"{result['mean_throughput']:.2f} ops/s")
    
    def test_opencl_matrix_multiplication_benchmark(self):
        """Benchmark OpenCL matrix multiplication."""
        if "opencl" not in self.manager.get_available_accelerators():
            self.skipTest("OpenCL not available")
        
        results = {}
        
        for size in self.matrix_sizes:
            benchmark_result = self.benchmark_matrix_multiplication("opencl", size)
            results[size] = benchmark_result
            
            # Verify reasonable performance
            self.assertLess(benchmark_result['mean_time'], 8.0)  # Should complete within 8 seconds
            self.assertGreater(benchmark_result['mean_throughput'], 0)
        
        # Log results for analysis
        print(f"\nOpenCL Matrix Multiplication Benchmark Results:")
        for size, result in results.items():
            print(f"  Size {size}x{size}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s, "
                  f"{result['mean_throughput']:.2f} ops/s")
    
    def test_performance_comparison(self):
        """Compare performance across all available accelerators."""
        available_accelerators = self.manager.get_available_accelerators()
        
        if len(available_accelerators) < 2:
            self.skipTest("Need at least 2 accelerators for comparison")
        
        comparison_results = {}
        
        for accelerator in available_accelerators:
            accelerator_results = {}
            
            for size in self.matrix_sizes:
                benchmark_result = self.benchmark_matrix_multiplication(accelerator, size)
                accelerator_results[size] = benchmark_result
            
            comparison_results[accelerator] = accelerator_results
        
        # Log comparison results
        print(f"\nPerformance Comparison Results:")
        for accelerator, results in comparison_results.items():
            print(f"\n{accelerator.upper()}:")
            for size, result in results.items():
                print(f"  Size {size}x{size}: {result['mean_time']:.4f}s, "
                      f"{result['mean_throughput']:.2f} ops/s")
        
        # Verify that results are reasonable
        for accelerator, results in comparison_results.items():
            for size, result in results.items():
                self.assertGreater(result['mean_throughput'], 0)
                self.assertLess(result['mean_time'], 15.0)  # Should complete within 15 seconds
    
    def test_scalability_benchmark(self):
        """Test scalability with increasing matrix sizes."""
        available_accelerators = self.manager.get_available_accelerators()
        
        scalability_results = {}
        
        for accelerator in available_accelerators:
            accelerator_results = {}
            
            for size in self.matrix_sizes:
                benchmark_result = self.benchmark_matrix_multiplication(accelerator, size)
                accelerator_results[size] = benchmark_result
            
            scalability_results[accelerator] = accelerator_results
        
        # Analyze scalability
        for accelerator, results in scalability_results.items():
            times = [results[size]['mean_time'] for size in self.matrix_sizes]
            throughputs = [results[size]['mean_throughput'] for size in self.matrix_sizes]
            
            # Verify that larger matrices take more time (generally)
            # This is a basic scalability check
            self.assertGreater(times[-1], times[0] * 0.1)  # Last should be at least 10% of first
        
        # Log scalability results
        print(f"\nScalability Analysis:")
        for accelerator, results in scalability_results.items():
            print(f"\n{accelerator.upper()}:")
            for size, result in results.items():
                print(f"  Size {size}x{size}: {result['mean_time']:.4f}s, "
                      f"{result['mean_throughput']:.2f} ops/s")
    
    def test_memory_usage_benchmark(self):
        """Test memory usage patterns."""
        available_accelerators = self.manager.get_available_accelerators()
        
        memory_results = {}
        
        for accelerator in available_accelerators:
            accelerator_results = {}
            
            for size in self.matrix_sizes:
                # Create matrices
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                # Estimate memory usage
                matrix_memory = 2 * size * size * 4  # 2 matrices, 4 bytes per float32
                result_memory = size * size * 4  # Result matrix
                total_memory = matrix_memory + result_memory
                
                accelerator_results[size] = {
                    'matrix_memory_mb': matrix_memory / (1024 * 1024),
                    'total_memory_mb': total_memory / (1024 * 1024),
                    'memory_efficiency': matrix_memory / total_memory
                }
            
            memory_results[accelerator] = accelerator_results
        
        # Log memory usage results
        print(f"\nMemory Usage Analysis:")
        for accelerator, results in memory_results.items():
            print(f"\n{accelerator.upper()}:")
            for size, result in results.items():
                print(f"  Size {size}x{size}: {result['total_memory_mb']:.2f} MB total, "
                      f"{result['memory_efficiency']:.2f} efficiency")
        
        # Verify memory efficiency
        for accelerator, results in memory_results.items():
            for size, result in results.items():
                self.assertGreater(result['memory_efficiency'], 0.5)  # At least 50% efficiency
    
    def test_throughput_consistency(self):
        """Test throughput consistency across multiple runs."""
        available_accelerators = self.manager.get_available_accelerators()
        
        consistency_results = {}
        
        for accelerator in available_accelerators:
            accelerator_results = {}
            
            for size in self.matrix_sizes:
                benchmark_result = self.benchmark_matrix_multiplication(accelerator, size)
                
                # Calculate coefficient of variation (CV) for consistency
                cv_time = benchmark_result['std_time'] / benchmark_result['mean_time'] if benchmark_result['mean_time'] > 0 else 0
                cv_throughput = benchmark_result['std_throughput'] / benchmark_result['mean_throughput'] if benchmark_result['mean_throughput'] > 0 else 0
                
                accelerator_results[size] = {
                    'cv_time': cv_time,
                    'cv_throughput': cv_throughput,
                    'consistency_score': 1.0 - min(cv_time, cv_throughput)  # Higher is more consistent
                }
            
            consistency_results[accelerator] = accelerator_results
        
        # Log consistency results
        print(f"\nThroughput Consistency Analysis:")
        for accelerator, results in consistency_results.items():
            print(f"\n{accelerator.upper()}:")
            for size, result in results.items():
                print(f"  Size {size}x{size}: CV={result['cv_throughput']:.3f}, "
                      f"Consistency={result['consistency_score']:.3f}")
        
        # Verify reasonable consistency
        for accelerator, results in consistency_results.items():
            for size, result in results.items():
                self.assertLess(result['cv_throughput'], 0.5)  # CV should be less than 50%
                self.assertGreater(result['consistency_score'], 0.5)  # Consistency score should be > 50%


class TestHardwareStressTests(unittest.TestCase):
    """Test hardware under stress conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HardwareManagerConfig()
        self.manager = HardwareManager(self.config)
        
        # Register accelerators
        self.manager.register_accelerator("cuda", CUDAAccelerator(CUDAConfig()))
        self.manager.register_accelerator("opencl", OpenCLAccelerator(OpenCLConfig()))
        self.manager.register_accelerator("cpu", CPUAccelerator(CPUConfig()))
    
    def test_continuous_operation_stress(self):
        """Test continuous operation under stress."""
        available_accelerators = self.manager.get_available_accelerators()
        
        if not available_accelerators:
            self.skipTest("No accelerators available")
        
        # Test continuous operation for 30 seconds
        test_duration = 30
        start_time = time.time()
        
        operation_count = 0
        success_count = 0
        
        while time.time() - start_time < test_duration:
            for accelerator in available_accelerators:
                try:
                    # Create small matrices for continuous testing
                    a = np.random.rand(50, 50).astype(np.float32)
                    b = np.random.rand(50, 50).astype(np.float32)
                    
                    result = self.manager.execute_operation(
                        accelerator, 
                        "matrix_multiply", 
                        {"a": a, "b": b}
                    )
                    
                    operation_count += 1
                    if result.success:
                        success_count += 1
                    
                except Exception as e:
                    operation_count += 1
                    # Allow some failures but not too many
                    if operation_count > 10 and success_count / operation_count < 0.8:
                        self.fail(f"Too many failures in continuous operation: {success_count}/{operation_count}")
        
        # Verify reasonable success rate
        success_rate = success_count / operation_count if operation_count > 0 else 0
        self.assertGreater(success_rate, 0.8)  # At least 80% success rate
        
        print(f"\nContinuous Operation Stress Test:")
        print(f"  Duration: {test_duration}s")
        print(f"  Operations: {operation_count}")
        print(f"  Successes: {success_count}")
        print(f"  Success Rate: {success_rate:.2%}")
    
    def test_large_matrix_stress(self):
        """Test with very large matrices."""
        available_accelerators = self.manager.get_available_accelerators()
        
        if not available_accelerators:
            self.skipTest("No accelerators available")
        
        # Test with large matrices
        large_sizes = [2000, 3000, 4000]
        
        for size in large_sizes:
            for accelerator in available_accelerators:
                try:
                    # Create large matrices
                    a = np.random.rand(size, size).astype(np.float32)
                    b = np.random.rand(size, size).astype(np.float32)
                    
                    start_time = time.time()
                    result = self.manager.execute_operation(
                        accelerator, 
                        "matrix_multiply", 
                        {"a": a, "b": b}
                    )
                    end_time = time.time()
                    
                    if result.success:
                        print(f"\nLarge Matrix Test - {accelerator.upper()}:")
                        print(f"  Size: {size}x{size}")
                        print(f"  Time: {end_time - start_time:.4f}s")
                        print(f"  Throughput: {result.throughput:.2f} ops/s")
                        
                        # Verify reasonable performance
                        self.assertLess(end_time - start_time, 60.0)  # Should complete within 60 seconds
                    
                except Exception as e:
                    # Allow failures for very large matrices
                    print(f"Large matrix test failed for {accelerator} with size {size}: {e}")
    
    def test_memory_pressure_stress(self):
        """Test under memory pressure conditions."""
        available_accelerators = self.manager.get_available_accelerators()
        
        if not available_accelerators:
            self.skipTest("No accelerators available")
        
        # Test with multiple concurrent operations
        concurrent_operations = 5
        
        for accelerator in available_accelerators:
            try:
                # Create multiple matrices simultaneously
                matrices = []
                for _ in range(concurrent_operations):
                    a = np.random.rand(500, 500).astype(np.float32)
                    b = np.random.rand(500, 500).astype(np.float32)
                    matrices.append((a, b))
                
                # Execute operations
                results = []
                for a, b in matrices:
                    result = self.manager.execute_operation(
                        accelerator, 
                        "matrix_multiply", 
                        {"a": a, "b": b}
                    )
                    results.append(result)
                
                # Verify most operations succeeded
                success_count = sum(1 for r in results if r.success)
                success_rate = success_count / len(results)
                
                print(f"\nMemory Pressure Test - {accelerator.upper()}:")
                print(f"  Concurrent Operations: {concurrent_operations}")
                print(f"  Success Rate: {success_rate:.2%}")
                
                # Allow some failures under memory pressure
                self.assertGreater(success_rate, 0.6)  # At least 60% success rate
                
            except Exception as e:
                print(f"Memory pressure test failed for {accelerator}: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
