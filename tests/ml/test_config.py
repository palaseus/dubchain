"""
ML Test Configuration

This module provides configuration and utilities for ML module tests.
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
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class MLTestConfig:
    """Configuration for ML module tests."""
    
    def __init__(self):
        self.enable_infrastructure_tests = os.getenv('ENABLE_INFRASTRUCTURE_TESTS', 'true').lower() == 'true'
        self.enable_feature_tests = os.getenv('ENABLE_FEATURE_TESTS', 'true').lower() == 'true'
        self.enable_topology_tests = os.getenv('ENABLE_TOPOLOGY_TESTS', 'true').lower() == 'true'
        self.enable_routing_tests = os.getenv('ENABLE_ROUTING_TESTS', 'true').lower() == 'true'
        self.enable_anomaly_tests = os.getenv('ENABLE_ANOMALY_TESTS', 'true').lower() == 'true'
        self.enable_optimization_tests = os.getenv('ENABLE_OPTIMIZATION_TESTS', 'true').lower() == 'true'
        self.enable_performance_tests = os.getenv('ENABLE_PERFORMANCE_TESTS', 'true').lower() == 'true'
        self.enable_stress_tests = os.getenv('ENABLE_STRESS_TESTS', 'false').lower() == 'true'
        
        # Test data parameters
        self.test_data_size = int(os.getenv('TEST_DATA_SIZE', '1000'))
        self.test_feature_dim = int(os.getenv('TEST_FEATURE_DIM', '10'))
        self.test_network_size = int(os.getenv('TEST_NETWORK_SIZE', '50'))
        
        # Performance thresholds
        self.feature_extraction_timeout = float(os.getenv('FEATURE_EXTRACTION_TIMEOUT', '30.0'))
        self.topology_optimization_timeout = float(os.getenv('TOPOLOGY_OPTIMIZATION_TIMEOUT', '60.0'))
        self.routing_optimization_timeout = float(os.getenv('ROUTING_OPTIMIZATION_TIMEOUT', '30.0'))
        self.anomaly_detection_timeout = float(os.getenv('ANOMALY_DETECTION_TIMEOUT', '120.0'))
        self.parameter_optimization_timeout = float(os.getenv('PARAMETER_OPTIMIZATION_TIMEOUT', '180.0'))
        
        # Memory limits
        self.max_memory_increase_mb = int(os.getenv('MAX_MEMORY_INCREASE_MB', '1000'))
        
        # Test data directory
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)


class MLTestUtils:
    """Utilities for ML module tests."""
    
    @staticmethod
    def create_test_transactions(count: int = 1000) -> List[Dict[str, Any]]:
        """Create test transactions."""
        from dubchain.bridge.universal import UniversalTransaction, ChainType, TokenType
        
        transactions = []
        for i in range(count):
            tx = UniversalTransaction(
                tx_id=f"0x{i:040x}",
                from_address=f"0x{i:040x}",
                to_address=f"0x{i+1:040x}",
                amount=1000000000000000000,
                token_type=TokenType.NATIVE,
                chain_type=ChainType.ETHEREUM,
                created_at=time.time()
            )
            transactions.append(tx)
        
        return transactions
    
    @staticmethod
    def create_test_network_topology(node_count: int = 50) -> Dict[str, Any]:
        """Create test network topology."""
        from dubchain.ml.network import NetworkTopology, PeerNode
        
        nodes = {}
        for i in range(node_count):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        # Create edges (sparse network)
        edges = []
        for i in range(0, node_count, 2):
            if i + 1 < node_count:
                edges.append((f"node_{i}", f"node_{i+1}"))
        
        return {
            'nodes': nodes,
            'edges': edges,
            'topology': NetworkTopology(nodes=nodes, edges=edges)
        }
    
    @staticmethod
    def create_test_data(samples: int = 1000, features: int = 10) -> np.ndarray:
        """Create test data for ML models."""
        return np.random.rand(samples, features)
    
    @staticmethod
    def create_test_parameter_spaces() -> List[Dict[str, Any]]:
        """Create test parameter spaces."""
        return [
            {
                'parameter_name': 'block_size',
                'parameter_type': 'continuous',
                'bounds': (1000, 10000),
                'default_value': 5000
            },
            {
                'parameter_name': 'consensus_timeout',
                'parameter_type': 'continuous',
                'bounds': (1, 60),
                'default_value': 30
            },
            {
                'parameter_name': 'validator_count',
                'parameter_type': 'discrete',
                'bounds': (3, 20),
                'default_value': 10
            }
        ]
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], filename: str) -> None:
        """Save test results to file."""
        config = MLTestConfig()
        filepath = os.path.join(config.test_data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    @staticmethod
    def load_test_results(filename: str) -> Optional[Dict[str, Any]]:
        """Load test results from file."""
        config = MLTestConfig()
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
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs):
        """Measure memory usage of a function."""
        import psutil
        
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        
        return result, memory_increase
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        return result, execution_time


class MLTestFixtures:
    """Test fixtures for ML module tests."""
    
    def __init__(self):
        self.config = MLTestConfig()
        self.utils = MLTestUtils()
    
    def get_test_transactions(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get test transactions."""
        if count is None:
            count = self.config.test_data_size
        return self.utils.create_test_transactions(count)
    
    def get_test_network_topology(self, node_count: Optional[int] = None) -> Dict[str, Any]:
        """Get test network topology."""
        if node_count is None:
            node_count = self.config.test_network_size
        return self.utils.create_test_network_topology(node_count)
    
    def get_test_data(self, samples: Optional[int] = None, features: Optional[int] = None) -> np.ndarray:
        """Get test data."""
        if samples is None:
            samples = self.config.test_data_size
        if features is None:
            features = self.config.test_feature_dim
        return self.utils.create_test_data(samples, features)
    
    def get_test_parameter_spaces(self) -> List[Dict[str, Any]]:
        """Get test parameter spaces."""
        return self.utils.create_test_parameter_spaces()
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get performance thresholds."""
        return {
            'feature_extraction_timeout': self.config.feature_extraction_timeout,
            'topology_optimization_timeout': self.config.topology_optimization_timeout,
            'routing_optimization_timeout': self.config.routing_optimization_timeout,
            'anomaly_detection_timeout': self.config.anomaly_detection_timeout,
            'parameter_optimization_timeout': self.config.parameter_optimization_timeout
        }


# Global test configuration
test_config = MLTestConfig()
test_utils = MLTestUtils()
test_fixtures = MLTestFixtures()


def skip_if_ml_tests_disabled(test_type: str):
    """Skip test if specific ML test type is disabled."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if test_type == 'infrastructure' and not test_config.enable_infrastructure_tests:
                pytest.skip("ML infrastructure tests disabled")
            elif test_type == 'feature' and not test_config.enable_feature_tests:
                pytest.skip("ML feature tests disabled")
            elif test_type == 'topology' and not test_config.enable_topology_tests:
                pytest.skip("ML topology tests disabled")
            elif test_type == 'routing' and not test_config.enable_routing_tests:
                pytest.skip("ML routing tests disabled")
            elif test_type == 'anomaly' and not test_config.enable_anomaly_tests:
                pytest.skip("ML anomaly tests disabled")
            elif test_type == 'optimization' and not test_config.enable_optimization_tests:
                pytest.skip("ML optimization tests disabled")
            elif test_type == 'performance' and not test_config.enable_performance_tests:
                pytest.skip("ML performance tests disabled")
            elif test_type == 'stress' and not test_config.enable_stress_tests:
                pytest.skip("ML stress tests disabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def performance_test(func):
    """Decorator for performance tests."""
    def wrapper(*args, **kwargs):
        if not test_config.enable_performance_tests:
            pytest.skip("Performance tests disabled")
        return func(*args, **kwargs)
    return wrapper


def stress_test(func):
    """Decorator for stress tests."""
    def wrapper(*args, **kwargs):
        if not test_config.enable_stress_tests:
            pytest.skip("Stress tests disabled")
        return func(*args, **kwargs)
    return wrapper


def memory_test(func):
    """Decorator for memory tests."""
    def wrapper(*args, **kwargs):
        result, memory_increase = test_utils.measure_memory_usage(func, *args, **kwargs)
        
        if memory_increase > test_config.max_memory_increase_mb:
            pytest.fail(f"Memory increase {memory_increase:.2f} MB exceeds limit {test_config.max_memory_increase_mb} MB")
        
        return result
    return wrapper


def timeout_test(timeout: float):
    """Decorator for timeout tests."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result, execution_time = test_utils.measure_execution_time(func, *args, **kwargs)
            
            if execution_time > timeout:
                pytest.fail(f"Execution time {execution_time:.4f}s exceeds timeout {timeout}s")
            
            return result
        return wrapper
    return decorator
