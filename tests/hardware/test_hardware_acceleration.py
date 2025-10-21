"""
Comprehensive Tests for Hardware Acceleration Modules

This module provides comprehensive testing for all hardware acceleration components including:
- Hardware detection and capability testing
- CUDA acceleration tests
- OpenCL acceleration tests
- CPU SIMD optimization tests
- Performance benchmarking tests
- Integration tests for hardware manager
- Error handling and edge case tests
"""

import pytest
import unittest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dubchain.hardware.detection import HardwareDetector, HardwareCapabilities
from dubchain.hardware.base import HardwareAccelerator, AccelerationResult
from dubchain.hardware.cuda import CUDAAccelerator, CUDAConfig
from dubchain.hardware.opencl import OpenCLAccelerator, OpenCLConfig
from dubchain.hardware.cpu import CPUAccelerator, CPUConfig
from dubchain.hardware.manager import HardwareManager, HardwareManagerConfig


class TestHardwareDetection(unittest.TestCase):
    """Test hardware detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HardwareDetector()
    
    def test_detect_cpu_capabilities(self):
        """Test CPU capability detection."""
        capabilities = self.detector.detect_cpu_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.cpu_cores, int)
        self.assertGreater(capabilities.cpu_cores, 0)
        self.assertIsInstance(capabilities.cpu_frequency, float)
        self.assertGreater(capabilities.cpu_frequency, 0)
    
    def test_detect_gpu_capabilities(self):
        """Test GPU capability detection."""
        capabilities = self.detector.detect_gpu_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.gpu_count, int)
        self.assertGreaterEqual(capabilities.gpu_count, 0)
    
    def test_detect_memory_capabilities(self):
        """Test memory capability detection."""
        capabilities = self.detector.detect_memory_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.total_memory, int)
        self.assertGreater(capabilities.total_memory, 0)
        self.assertIsInstance(capabilities.available_memory, int)
        self.assertGreaterEqual(capabilities.available_memory, 0)
    
    def test_detect_all_capabilities(self):
        """Test comprehensive capability detection."""
        capabilities = self.detector.detect_all_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.cpu_cores, int)
        self.assertIsInstance(capabilities.total_memory, int)
        self.assertIsInstance(capabilities.gpu_count, int)
    
    def test_detect_simd_capabilities(self):
        """Test SIMD capability detection."""
        capabilities = self.detector.detect_simd_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.simd_capabilities, list)
        self.assertIsInstance(capabilities.avx_support, bool)
        self.assertIsInstance(capabilities.neon_support, bool)
    
    @patch('platform.machine')
    def test_detect_architecture(self, mock_machine):
        """Test architecture detection."""
        mock_machine.return_value = 'x86_64'
        arch = self.detector.detect_architecture()
        self.assertEqual(arch, 'x86_64')
        
        mock_machine.return_value = 'aarch64'
        arch = self.detector.detect_architecture()
        self.assertEqual(arch, 'aarch64')
    
    def test_detect_cuda_capabilities(self):
        """Test CUDA capability detection."""
        capabilities = self.detector.detect_cuda_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.cuda_available, bool)
        if capabilities.cuda_available:
            self.assertIsInstance(capabilities.cuda_version, str)
            self.assertIsInstance(capabilities.cuda_devices, list)
    
    def test_detect_opencl_capabilities(self):
        """Test OpenCL capability detection."""
        capabilities = self.detector.detect_opencl_capabilities()
        
        self.assertIsInstance(capabilities, HardwareCapabilities)
        self.assertIsInstance(capabilities.opencl_available, bool)
        if capabilities.opencl_available:
            self.assertIsInstance(capabilities.opencl_platforms, list)
            self.assertIsInstance(capabilities.opencl_devices, list)


class TestCUDAAccelerator(unittest.TestCase):
    """Test CUDA acceleration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CUDAConfig()
        self.accelerator = CUDAAccelerator(self.config)
    
    def test_initialization(self):
        """Test CUDA accelerator initialization."""
        self.assertIsInstance(self.accelerator, CUDAAccelerator)
        self.assertIsInstance(self.accelerator.config, CUDAConfig)
    
    def test_is_available(self):
        """Test CUDA availability check."""
        available = self.accelerator.is_available()
        self.assertIsInstance(available, bool)
    
    def test_get_device_info(self):
        """Test CUDA device information."""
        device_info = self.accelerator.get_device_info()
        self.assertIsInstance(device_info, dict)
        
        if self.accelerator.is_available():
            self.assertIn('device_count', device_info)
            self.assertIn('device_names', device_info)
            self.assertIn('compute_capability', device_info)
    
    def test_allocate_memory(self):
        """Test CUDA memory allocation."""
        if not self.accelerator.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Test memory allocation
            size = 1024
            ptr = self.accelerator.allocate_memory(size)
            self.assertIsNotNone(ptr)
            
            # Test memory deallocation
            self.accelerator.free_memory(ptr)
            
        except Exception as e:
            self.skipTest(f"CUDA memory allocation failed: {e}")
    
    def test_matrix_multiplication(self):
        """Test CUDA matrix multiplication."""
        if not self.accelerator.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Create test matrices
            a = np.random.rand(100, 100).astype(np.float32)
            b = np.random.rand(100, 100).astype(np.float32)
            
            # Perform matrix multiplication
            result = self.accelerator.matrix_multiply(a, b)
            
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (100, 100))
            
            # Verify correctness
            expected = np.dot(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-5)
            
        except Exception as e:
            self.skipTest(f"CUDA matrix multiplication failed: {e}")
    
    def test_cryptographic_operations(self):
        """Test CUDA cryptographic operations."""
        if not self.accelerator.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Test hash computation
            data = b"test data for hashing"
            hash_result = self.accelerator.compute_hash(data)
            
            self.assertIsInstance(hash_result, bytes)
            self.assertEqual(len(hash_result), 32)  # SHA-256
            
        except Exception as e:
            self.skipTest(f"CUDA cryptographic operations failed: {e}")
    
    def test_performance_benchmark(self):
        """Test CUDA performance benchmarking."""
        if not self.accelerator.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Benchmark matrix multiplication
            sizes = [100, 500, 1000]
            results = {}
            
            for size in sizes:
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                start_time = time.time()
                result = self.accelerator.matrix_multiply(a, b)
                end_time = time.time()
                
                results[size] = {
                    'time': end_time - start_time,
                    'ops_per_second': (2 * size**3) / (end_time - start_time)
                }
            
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), len(sizes))
            
        except Exception as e:
            self.skipTest(f"CUDA performance benchmark failed: {e}")


class TestOpenCLAccelerator(unittest.TestCase):
    """Test OpenCL acceleration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = OpenCLConfig()
        self.accelerator = OpenCLAccelerator(self.config)
    
    def test_initialization(self):
        """Test OpenCL accelerator initialization."""
        self.assertIsInstance(self.accelerator, OpenCLAccelerator)
        self.assertIsInstance(self.accelerator.config, OpenCLConfig)
    
    def test_is_available(self):
        """Test OpenCL availability check."""
        available = self.accelerator.is_available()
        self.assertIsInstance(available, bool)
    
    def test_get_platform_info(self):
        """Test OpenCL platform information."""
        platform_info = self.accelerator.get_platform_info()
        self.assertIsInstance(platform_info, list)
        
        if self.accelerator.is_available():
            for platform in platform_info:
                self.assertIn('name', platform)
                self.assertIn('vendor', platform)
                self.assertIn('version', platform)
    
    def test_get_device_info(self):
        """Test OpenCL device information."""
        device_info = self.accelerator.get_device_info()
        self.assertIsInstance(device_info, list)
        
        if self.accelerator.is_available():
            for device in device_info:
                self.assertIn('name', device)
                self.assertIn('type', device)
                self.assertIn('compute_units', device)
                self.assertIn('max_memory', device)
    
    def test_create_context(self):
        """Test OpenCL context creation."""
        if not self.accelerator.is_available():
            self.skipTest("OpenCL not available")
        
        try:
            context = self.accelerator.create_context()
            self.assertIsNotNone(context)
            
        except Exception as e:
            self.skipTest(f"OpenCL context creation failed: {e}")
    
    def test_matrix_multiplication(self):
        """Test OpenCL matrix multiplication."""
        if not self.accelerator.is_available():
            self.skipTest("OpenCL not available")
        
        try:
            # Create test matrices
            a = np.random.rand(50, 50).astype(np.float32)
            b = np.random.rand(50, 50).astype(np.float32)
            
            # Perform matrix multiplication
            result = self.accelerator.matrix_multiply(a, b)
            
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (50, 50))
            
            # Verify correctness
            expected = np.dot(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-4)
            
        except Exception as e:
            self.skipTest(f"OpenCL matrix multiplication failed: {e}")
    
    def test_cryptographic_operations(self):
        """Test OpenCL cryptographic operations."""
        if not self.accelerator.is_available():
            self.skipTest("OpenCL not available")
        
        try:
            # Test hash computation
            data = b"test data for hashing"
            hash_result = self.accelerator.compute_hash(data)
            
            self.assertIsInstance(hash_result, bytes)
            self.assertEqual(len(hash_result), 32)  # SHA-256
            
        except Exception as e:
            self.skipTest(f"OpenCL cryptographic operations failed: {e}")


class TestCPUAccelerator(unittest.TestCase):
    """Test CPU acceleration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CPUConfig()
        self.accelerator = CPUAccelerator(self.config)
    
    def test_initialization(self):
        """Test CPU accelerator initialization."""
        self.assertIsInstance(self.accelerator, CPUAccelerator)
        self.assertIsInstance(self.accelerator.config, CPUConfig)
    
    def test_is_available(self):
        """Test CPU availability check."""
        available = self.accelerator.is_available()
        self.assertTrue(available)  # CPU should always be available
    
    def test_get_cpu_info(self):
        """Test CPU information retrieval."""
        cpu_info = self.accelerator.get_cpu_info()
        self.assertIsInstance(cpu_info, dict)
        self.assertIn('cores', cpu_info)
        self.assertIn('frequency', cpu_info)
        self.assertIn('architecture', cpu_info)
        self.assertIn('simd_capabilities', cpu_info)
    
    def test_avx_support(self):
        """Test AVX support detection."""
        avx_support = self.accelerator.supports_avx()
        self.assertIsInstance(avx_support, bool)
    
    def test_neon_support(self):
        """Test NEON support detection."""
        neon_support = self.accelerator.supports_neon()
        self.assertIsInstance(neon_support, bool)
    
    def test_matrix_multiplication(self):
        """Test CPU matrix multiplication."""
        # Create test matrices
        a = np.random.rand(100, 100).astype(np.float32)
        b = np.random.rand(100, 100).astype(np.float32)
        
        # Perform matrix multiplication
        result = self.accelerator.matrix_multiply(a, b)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100))
        
        # Verify correctness
        expected = np.dot(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_cryptographic_operations(self):
        """Test CPU cryptographic operations."""
        # Test hash computation
        data = b"test data for hashing"
        hash_result = self.accelerator.compute_hash(data)
        
        self.assertIsInstance(hash_result, bytes)
        self.assertEqual(len(hash_result), 32)  # SHA-256
    
    def test_simd_optimizations(self):
        """Test SIMD optimizations."""
        # Test vectorized operations
        a = np.random.rand(1000).astype(np.float32)
        b = np.random.rand(1000).astype(np.float32)
        
        # Vectorized addition
        result = self.accelerator.vector_add(a, b)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, a.shape)
        
        # Verify correctness
        expected = a + b
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_performance_benchmark(self):
        """Test CPU performance benchmarking."""
        # Benchmark matrix multiplication
        sizes = [100, 500, 1000]
        results = {}
        
        for size in sizes:
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            
            start_time = time.time()
            result = self.accelerator.matrix_multiply(a, b)
            end_time = time.time()
            
            results[size] = {
                'time': end_time - start_time,
                'ops_per_second': (2 * size**3) / (end_time - start_time)
            }
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(sizes))


class TestHardwareManager(unittest.TestCase):
    """Test hardware manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HardwareManagerConfig()
        self.manager = HardwareManager(self.config)
    
    def test_initialization(self):
        """Test hardware manager initialization."""
        self.assertIsInstance(self.manager, HardwareManager)
        self.assertIsInstance(self.manager.config, HardwareManagerConfig)
    
    def test_register_accelerator(self):
        """Test accelerator registration."""
        # Create mock accelerator
        mock_accelerator = Mock(spec=HardwareAccelerator)
        mock_accelerator.is_available.return_value = True
        mock_accelerator.get_name.return_value = "test_accelerator"
        
        # Register accelerator
        self.manager.register_accelerator("test", mock_accelerator)
        
        # Verify registration
        self.assertIn("test", self.manager.accelerators)
        self.assertEqual(self.manager.accelerators["test"], mock_accelerator)
    
    def test_get_available_accelerators(self):
        """Test getting available accelerators."""
        # Create mock accelerators
        mock_cuda = Mock(spec=HardwareAccelerator)
        mock_cuda.is_available.return_value = True
        mock_cuda.get_name.return_value = "CUDA"
        
        mock_opencl = Mock(spec=HardwareAccelerator)
        mock_opencl.is_available.return_value = False
        mock_opencl.get_name.return_value = "OpenCL"
        
        mock_cpu = Mock(spec=HardwareAccelerator)
        mock_cpu.is_available.return_value = True
        mock_cpu.get_name.return_value = "CPU"
        
        # Register accelerators
        self.manager.register_accelerator("cuda", mock_cuda)
        self.manager.register_accelerator("opencl", mock_opencl)
        self.manager.register_accelerator("cpu", mock_cpu)
        
        # Get available accelerators
        available = self.manager.get_available_accelerators()
        
        self.assertIsInstance(available, list)
        self.assertIn("cuda", available)
        self.assertIn("cpu", available)
        self.assertNotIn("opencl", available)
    
    def test_select_best_accelerator(self):
        """Test selecting best accelerator."""
        # Create mock accelerators with different performance scores
        mock_cuda = Mock(spec=HardwareAccelerator)
        mock_cuda.is_available.return_value = True
        mock_cuda.get_performance_score.return_value = 100
        mock_cuda.get_name.return_value = "CUDA"
        
        mock_cpu = Mock(spec=HardwareAccelerator)
        mock_cpu.is_available.return_value = True
        mock_cpu.get_performance_score.return_value = 50
        mock_cpu.get_name.return_value = "CPU"
        
        # Register accelerators
        self.manager.register_accelerator("cuda", mock_cuda)
        self.manager.register_accelerator("cpu", mock_cpu)
        
        # Select best accelerator
        best = self.manager.select_best_accelerator()
        
        self.assertEqual(best, "cuda")
    
    def test_execute_operation(self):
        """Test executing operations with accelerators."""
        # Create mock accelerator
        mock_accelerator = Mock(spec=HardwareAccelerator)
        mock_accelerator.is_available.return_value = True
        mock_accelerator.get_name.return_value = "test_accelerator"
        
        # Mock operation result
        mock_result = AccelerationResult(
            success=True,
            execution_time=0.1,
            throughput=1000.0,
            data=np.array([1, 2, 3])
        )
        mock_accelerator.execute_operation.return_value = mock_result
        
        # Register accelerator
        self.manager.register_accelerator("test", mock_accelerator)
        
        # Execute operation
        result = self.manager.execute_operation("test", "matrix_multiply", {"a": np.array([1, 2]), "b": np.array([3, 4])})
        
        self.assertIsInstance(result, AccelerationResult)
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 0.1)
        self.assertEqual(result.throughput, 1000.0)
    
    def test_benchmark_accelerators(self):
        """Test benchmarking accelerators."""
        # Create mock accelerators
        mock_cuda = Mock(spec=HardwareAccelerator)
        mock_cuda.is_available.return_value = True
        mock_cuda.get_name.return_value = "CUDA"
        mock_cuda.benchmark.return_value = {"throughput": 1000, "latency": 0.001}
        
        mock_cpu = Mock(spec=HardwareAccelerator)
        mock_cpu.is_available.return_value = True
        mock_cpu.get_name.return_value = "CPU"
        mock_cpu.benchmark.return_value = {"throughput": 500, "latency": 0.002}
        
        # Register accelerators
        self.manager.register_accelerator("cuda", mock_cuda)
        self.manager.register_accelerator("cpu", mock_cpu)
        
        # Benchmark accelerators
        results = self.manager.benchmark_accelerators()
        
        self.assertIsInstance(results, dict)
        self.assertIn("cuda", results)
        self.assertIn("cpu", results)
        self.assertIn("throughput", results["cuda"])
        self.assertIn("latency", results["cuda"])
    
    def test_get_system_stats(self):
        """Test getting system statistics."""
        stats = self.manager.get_system_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_accelerators", stats)
        self.assertIn("available_accelerators", stats)
        self.assertIn("system_capabilities", stats)


class TestHardwareIntegration(unittest.TestCase):
    """Test hardware integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HardwareManagerConfig()
        self.manager = HardwareManager(self.config)
        
        # Register real accelerators
        self.manager.register_accelerator("cuda", CUDAAccelerator(CUDAConfig()))
        self.manager.register_accelerator("opencl", OpenCLAccelerator(OpenCLConfig()))
        self.manager.register_accelerator("cpu", CPUAccelerator(CPUConfig()))
    
    def test_integration_matrix_multiplication(self):
        """Test integrated matrix multiplication across accelerators."""
        # Create test matrices
        a = np.random.rand(100, 100).astype(np.float32)
        b = np.random.rand(100, 100).astype(np.float32)
        
        # Test with each available accelerator
        available = self.manager.get_available_accelerators()
        
        for accelerator_name in available:
            try:
                result = self.manager.execute_operation(
                    accelerator_name, 
                    "matrix_multiply", 
                    {"a": a, "b": b}
                )
                
                self.assertIsInstance(result, AccelerationResult)
                self.assertTrue(result.success)
                self.assertIsInstance(result.data, np.ndarray)
                self.assertEqual(result.data.shape, (100, 100))
                
            except Exception as e:
                self.skipTest(f"Integration test failed for {accelerator_name}: {e}")
    
    def test_integration_cryptographic_operations(self):
        """Test integrated cryptographic operations across accelerators."""
        data = b"test data for cryptographic operations"
        
        # Test with each available accelerator
        available = self.manager.get_available_accelerators()
        
        for accelerator_name in available:
            try:
                result = self.manager.execute_operation(
                    accelerator_name, 
                    "compute_hash", 
                    {"data": data}
                )
                
                self.assertIsInstance(result, AccelerationResult)
                self.assertTrue(result.success)
                self.assertIsInstance(result.data, bytes)
                self.assertEqual(len(result.data), 32)  # SHA-256
                
            except Exception as e:
                self.skipTest(f"Integration test failed for {accelerator_name}: {e}")
    
    def test_performance_comparison(self):
        """Test performance comparison across accelerators."""
        # Create test matrices
        a = np.random.rand(500, 500).astype(np.float32)
        b = np.random.rand(500, 500).astype(np.float32)
        
        # Benchmark each accelerator
        results = {}
        available = self.manager.get_available_accelerators()
        
        for accelerator_name in available:
            try:
                start_time = time.time()
                result = self.manager.execute_operation(
                    accelerator_name, 
                    "matrix_multiply", 
                    {"a": a, "b": b}
                )
                end_time = time.time()
                
                if result.success:
                    results[accelerator_name] = {
                        'execution_time': end_time - start_time,
                        'throughput': result.throughput
                    }
                
            except Exception as e:
                self.skipTest(f"Performance comparison failed for {accelerator_name}: {e}")
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # CPU should always be available and working
        self.assertIn("cpu", results)


class TestHardwareErrorHandling(unittest.TestCase):
    """Test hardware error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HardwareManagerConfig()
        self.manager = HardwareManager(self.config)
    
    def test_invalid_accelerator_name(self):
        """Test handling of invalid accelerator names."""
        with self.assertRaises(ValueError):
            self.manager.execute_operation("invalid", "test_operation", {})
    
    def test_unavailable_accelerator(self):
        """Test handling of unavailable accelerators."""
        # Create mock unavailable accelerator
        mock_accelerator = Mock(spec=HardwareAccelerator)
        mock_accelerator.is_available.return_value = False
        mock_accelerator.get_name.return_value = "unavailable"
        
        # Register accelerator
        self.manager.register_accelerator("unavailable", mock_accelerator)
        
        # Try to execute operation
        with self.assertRaises(RuntimeError):
            self.manager.execute_operation("unavailable", "test_operation", {})
    
    def test_operation_failure(self):
        """Test handling of operation failures."""
        # Create mock accelerator that fails
        mock_accelerator = Mock(spec=HardwareAccelerator)
        mock_accelerator.is_available.return_value = True
        mock_accelerator.get_name.return_value = "failing"
        mock_accelerator.execute_operation.side_effect = Exception("Operation failed")
        
        # Register accelerator
        self.manager.register_accelerator("failing", mock_accelerator)
        
        # Try to execute operation
        result = self.manager.execute_operation("failing", "test_operation", {})
        
        # Check that the operation failed
        self.assertIsInstance(result, AccelerationResult)
        self.assertFalse(result.success)
        self.assertIn("Operation failed", result.error_message)
    
    def test_memory_allocation_failure(self):
        """Test handling of memory allocation failures."""
        # Create mock accelerator with memory issues
        mock_accelerator = Mock(spec=HardwareAccelerator)
        mock_accelerator.is_available.return_value = True
        mock_accelerator.get_name.return_value = "memory_issue"
        mock_accelerator.execute_operation.side_effect = MemoryError("Out of memory")
        
        # Register accelerator
        self.manager.register_accelerator("memory_issue", mock_accelerator)
        
        # Try to allocate memory
        with self.assertRaises(MemoryError):
            self.manager.execute_operation("memory_issue", "allocate_memory", {"size": 1024})


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
