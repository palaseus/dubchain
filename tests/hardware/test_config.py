"""
Hardware Acceleration Test Configuration

This module provides configuration and utilities for hardware acceleration tests.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import unittest
import os
import sys
from typing import Dict, List, Any, Optional
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class HardwareTestConfig:
    """Configuration for hardware acceleration tests."""
    
    def __init__(self):
        self.enable_cuda_tests = os.getenv('ENABLE_CUDA_TESTS', 'true').lower() == 'true'
        self.enable_opencl_tests = os.getenv('ENABLE_OPENCL_TESTS', 'true').lower() == 'true'
        self.enable_cpu_tests = os.getenv('ENABLE_CPU_TESTS', 'true').lower() == 'true'
        self.enable_performance_tests = os.getenv('ENABLE_PERFORMANCE_TESTS', 'true').lower() == 'true'
        self.enable_stress_tests = os.getenv('ENABLE_STRESS_TESTS', 'false').lower() == 'true'
        
        # Performance thresholds
        self.matrix_multiply_timeout = 30.0  # seconds
        self.crypto_operation_timeout = 10.0  # seconds
        self.memory_allocation_timeout = 5.0  # seconds
        
        # Benchmark parameters
        self.benchmark_iterations = int(os.getenv('BENCHMARK_ITERATIONS', '5'))
        self.benchmark_matrix_sizes = [100, 500, 1000]
        self.stress_test_duration = int(os.getenv('STRESS_TEST_DURATION', '30'))
        
        # Test data directory
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)


class HardwareTestUtils:
    """Utilities for hardware acceleration tests."""
    
    @staticmethod
    def create_test_matrix(size: int, dtype: str = 'float32') -> Any:
        """Create a test matrix."""
        import numpy as np
        return np.random.rand(size, size).astype(dtype)
    
    @staticmethod
    def create_test_vector(size: int, dtype: str = 'float32') -> Any:
        """Create a test vector."""
        import numpy as np
        return np.random.rand(size).astype(dtype)
    
    @staticmethod
    def create_test_data(size: int) -> bytes:
        """Create test data for cryptographic operations."""
        import os
        return os.urandom(size)
    
    @staticmethod
    def save_benchmark_results(results: Dict[str, Any], filename: str) -> None:
        """Save benchmark results to file."""
        config = HardwareTestConfig()
        filepath = os.path.join(config.test_data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    @staticmethod
    def load_benchmark_results(filename: str) -> Optional[Dict[str, Any]]:
        """Load benchmark results from file."""
        config = HardwareTestConfig()
        filepath = os.path.join(config.test_data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def compare_performance_results(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current performance results with baseline."""
        comparison = {}
        
        for key in current:
            if key in baseline:
                current_val = current[key]
                baseline_val = baseline[key]
                
                if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                    if baseline_val != 0:
                        improvement = ((current_val - baseline_val) / baseline_val) * 100
                        comparison[key] = {
                            'current': current_val,
                            'baseline': baseline_val,
                            'improvement_percent': improvement
                        }
        
        return comparison


class HardwareTestFixtures:
    """Test fixtures for hardware acceleration tests."""
    
    def __init__(self):
        self.config = HardwareTestConfig()
        self.utils = HardwareTestUtils()
    
    def get_test_matrices(self, size: int) -> tuple:
        """Get test matrices for matrix multiplication tests."""
        a = self.utils.create_test_matrix(size)
        b = self.utils.create_test_matrix(size)
        return a, b
    
    def get_test_vectors(self, size: int) -> tuple:
        """Get test vectors for vector operations."""
        a = self.utils.create_test_vector(size)
        b = self.utils.create_test_vector(size)
        return a, b
    
    def get_test_data(self, size: int) -> bytes:
        """Get test data for cryptographic operations."""
        return self.utils.create_test_data(size)
    
    def get_benchmark_sizes(self) -> List[int]:
        """Get benchmark matrix sizes."""
        return self.config.benchmark_matrix_sizes.copy()
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get performance thresholds."""
        return {
            'matrix_multiply_timeout': self.config.matrix_multiply_timeout,
            'crypto_operation_timeout': self.config.crypto_operation_timeout,
            'memory_allocation_timeout': self.config.memory_allocation_timeout
        }


# Global test configuration
test_config = HardwareTestConfig()
test_utils = HardwareTestUtils()
test_fixtures = HardwareTestFixtures()


def skip_if_accelerator_unavailable(accelerator_name: str):
    """Skip test if accelerator is not available."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would need to be implemented based on actual accelerator availability
            # For now, we'll just run the test
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_performance_tests_disabled():
    """Skip test if performance tests are disabled."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not test_config.enable_performance_tests:
                pytest.skip("Performance tests disabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_stress_tests_disabled():
    """Skip test if stress tests are disabled."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not test_config.enable_stress_tests:
                pytest.skip("Stress tests disabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator
