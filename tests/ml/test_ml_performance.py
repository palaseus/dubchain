"""
ML Performance and Stress Tests

This module provides comprehensive performance and stress tests for ML components including:
- Model training performance tests
- Inference performance tests
- Memory usage tests
- Scalability tests
- Stress tests under high load
- Model accuracy tests
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import unittest
import asyncio
import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Any, Optional
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dubchain.ml.infrastructure import MLInfrastructure, ModelConfig
from dubchain.ml.features import FeaturePipeline, FeatureConfig
from dubchain.ml.network import TopologyOptimizer, TopologyConfig
from dubchain.ml.routing import RoutingOptimizer, RoutingConfig
from dubchain.ml.security import AnomalyDetector, AnomalyConfig
from dubchain.ml.optimization import ConsensusParameterOptimizer, ParameterConfig


class TestMLPerformance(unittest.TestCase):
    """Test ML performance benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.infrastructure = MLInfrastructure(ModelConfig())
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.topology_optimizer = TopologyOptimizer(TopologyConfig())
        self.routing_optimizer = RoutingOptimizer(RoutingConfig())
        self.anomaly_detector = AnomalyDetector(AnomalyConfig())
        self.parameter_optimizer = ConsensusParameterOptimizer(ParameterConfig())
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance."""
        # Create large dataset
        from dubchain.ml.universal import UniversalTransaction, ChainType, TokenType
        
        transactions = []
        for i in range(10000):  # 10k transactions
            tx = UniversalTransaction(
                tx_hash=f"0x{i:040x}",
                from_address=f"0x{i:040x}",
                to_address=f"0x{i+1:040x}",
                value=1000000000000000000,
                token_type=TokenType.NATIVE,
                chain_type=ChainType.ETHEREUM,
                timestamp=time.time()
            )
            transactions.append(tx)
        
        # Benchmark feature extraction
        start_time = time.time()
        features = self.feature_pipeline.extract_all_features(transactions)
        end_time = time.time()
        
        extraction_time = end_time - start_time
        transactions_per_second = len(transactions) / extraction_time
        
        logger.info(f"\nFeature Extraction Performance:")
        logger.info(f"  Transactions: {len(transactions)}")
        logger.info(f"  Time: {extraction_time:.4f}s")
        logger.info(f"  Throughput: {transactions_per_second:.2f} tx/s")
        
        # Performance assertions
        self.assertLess(extraction_time, 30.0)  # Should complete within 30 seconds
        self.assertGreater(transactions_per_second, 100)  # At least 100 tx/s
    
    def test_topology_optimization_performance(self):
        """Test topology optimization performance."""
        # Create large network topology
        from dubchain.ml.network import NetworkTopology, PeerNode
        
        nodes = {}
        for i in range(100):  # 100 nodes
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
        for i in range(0, 100, 2):
            if i + 1 < 100:
                edges.append((f"node_{i}", f"node_{i+1}"))
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Initialize model
        self.topology_optimizer.initialize_model(5)
        
        # Benchmark optimization
        start_time = time.time()
        result = self.topology_optimizer.optimize_topology(topology)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        logger.info(f"\nTopology Optimization Performance:")
        logger.info(f"  Nodes: {len(nodes)}")
        logger.info(f"  Edges: {len(edges)}")
        logger.info(f"  Time: {optimization_time:.4f}s")
        logger.info(f"  Improvement Score: {result.improvement_score:.4f}")
        
        # Performance assertions
        self.assertLess(optimization_time, 60.0)  # Should complete within 60 seconds
        self.assertIsInstance(result.improvement_score, float)
    
    def test_routing_optimization_performance(self):
        """Test routing optimization performance."""
        # Create large network topology
        from dubchain.ml.network import NetworkTopology, PeerNode
        
        nodes = {}
        for i in range(50):  # 50 nodes
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        # Create edges
        edges = []
        for i in range(0, 50, 2):
            if i + 1 < 50:
                edges.append((f"node_{i}", f"node_{i+1}"))
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Initialize agents
        self.routing_optimizer.initialize_agents(5, 10)
        
        # Benchmark route optimization
        start_time = time.time()
        route = self.routing_optimizer.optimize_route("node_0", "node_49", topology)
        end_time = time.time()
        
        routing_time = end_time - start_time
        
        logger.info(f"\nRouting Optimization Performance:")
        logger.info(f"  Nodes: {len(nodes)}")
        logger.info(f"  Route Length: {len(route)}")
        logger.info(f"  Time: {routing_time:.4f}s")
        
        # Performance assertions
        self.assertLess(routing_time, 30.0)  # Should complete within 30 seconds
        self.assertIsInstance(route, list)
        self.assertGreater(len(route), 0)
    
    def test_anomaly_detection_performance(self):
        """Test anomaly detection performance."""
        # Create large dataset
        data = np.random.rand(10000, 20)  # 10k samples, 20 features
        
        # Initialize models
        self.anomaly_detector.initialize_models(20)
        
        # Benchmark training
        start_time = time.time()
        self.anomaly_detector.train_models(data)
        training_time = time.time() - start_time
        
        # Benchmark inference
        test_data = np.random.rand(1000, 20)  # 1k test samples
        
        start_time = time.time()
        result = self.anomaly_detector.detect_anomalies(test_data)
        inference_time = time.time() - start_time
        
        samples_per_second = len(test_data) / inference_time
        
        logger.info(f"\nAnomaly Detection Performance:")
        logger.info(f"  Training Samples: {len(data)}")
        logger.info(f"  Training Time: {training_time:.4f}s")
        logger.info(f"  Test Samples: {len(test_data)}")
        logger.info(f"  Inference Time: {inference_time:.4f}s")
        logger.info(f"  Throughput: {samples_per_second:.2f} samples/s")
        logger.info(f"  Anomalies Detected: {len(result.anomaly_scores)}")
        
        # Performance assertions
        self.assertLess(training_time, 120.0)  # Should complete within 2 minutes
        self.assertLess(inference_time, 10.0)  # Should complete within 10 seconds
        self.assertGreater(samples_per_second, 50)  # At least 50 samples/s
    
    def test_parameter_optimization_performance(self):
        """Test parameter optimization performance."""
        # Add parameter spaces
        from dubchain.ml.optimization import ParameterSpace
        
        parameter_spaces = [
            ParameterSpace("block_size", "continuous", (1000, 10000), 5000),
            ParameterSpace("consensus_timeout", "continuous", (1, 60), 30),
            ParameterSpace("validator_count", "discrete", (3, 20), 10)
        ]
        
        for param_space in parameter_spaces:
            self.parameter_optimizer.add_consensus_parameter(param_space)
        
        # Mock performance evaluator
        def performance_evaluator(parameters):
            from dubchain.ml.optimization import PerformanceMetrics
            return PerformanceMetrics(
                throughput=parameters["block_size"] / 1000,
                latency=1000 / parameters["block_size"],
                security_score=0.9,
                energy_efficiency=0.8,
                scalability=0.7
            )
        
        # Benchmark optimization
        start_time = time.time()
        result = self.parameter_optimizer.optimize_consensus_parameters(performance_evaluator)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        logger.info(f"\nParameter Optimization Performance:")
        logger.info(f"  Parameters: {len(parameter_spaces)}")
        logger.info(f"  Iterations: {len(result.optimization_history)}")
        logger.info(f"  Time: {optimization_time:.4f}s")
        logger.info(f"  Best Score: {result.best_score:.4f}")
        
        # Performance assertions
        self.assertLess(optimization_time, 180.0)  # Should complete within 3 minutes
        self.assertIsInstance(result.best_score, float)
    
    def test_memory_usage(self):
        """Test memory usage during ML operations."""
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        data = np.random.rand(5000, 50)  # 5k samples, 50 features
        
        # Initialize models
        self.anomaly_detector.initialize_models(50)
        
        # Train models
        self.anomaly_detector.train_models(data)
        
        # Get memory usage after training
        training_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform inference
        test_data = np.random.rand(1000, 50)
        result = self.anomaly_detector.detect_anomalies(test_data)
        
        # Get memory usage after inference
        inference_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = training_memory - initial_memory
        
        logger.info(f"\nMemory Usage:")
        logger.info(f"  Initial Memory: {initial_memory:.2f} MB")
        logger.info(f"  After Training: {training_memory:.2f} MB")
        logger.info(f"  After Inference: {inference_memory:.2f} MB")
        logger.info(f"  Memory Increase: {memory_increase:.2f} MB")
        
        # Memory assertions
        self.assertLess(memory_increase, 1000.0)  # Should not increase by more than 1GB
        self.assertGreater(memory_increase, 0.0)  # Should use some memory


class TestMLStressTests(unittest.TestCase):
    """Test ML components under stress conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.topology_optimizer = TopologyOptimizer(TopologyConfig())
        self.routing_optimizer = RoutingOptimizer(RoutingConfig())
        self.anomaly_detector = AnomalyDetector(AnomalyConfig())
        self.parameter_optimizer = ConsensusParameterOptimizer(ParameterConfig())
    
    def test_high_volume_feature_extraction(self):
        """Test feature extraction under high volume."""
        # Create very large dataset
        from dubchain.ml.universal import UniversalTransaction, ChainType, TokenType
        
        transactions = []
        for i in range(100000):  # 100k transactions
            tx = UniversalTransaction(
                tx_hash=f"0x{i:040x}",
                from_address=f"0x{i:040x}",
                to_address=f"0x{i+1:040x}",
                value=1000000000000000000,
                token_type=TokenType.NATIVE,
                chain_type=ChainType.ETHEREUM,
                timestamp=time.time()
            )
            transactions.append(tx)
        
        # Test under stress
        start_time = time.time()
        features = self.feature_pipeline.extract_all_features(transactions)
        end_time = time.time()
        
        extraction_time = end_time - start_time
        transactions_per_second = len(transactions) / extraction_time
        
        logger.info(f"\nHigh Volume Feature Extraction:")
        logger.info(f"  Transactions: {len(transactions)}")
        logger.info(f"  Time: {extraction_time:.4f}s")
        logger.info(f"  Throughput: {transactions_per_second:.2f} tx/s")
        
        # Stress test assertions
        self.assertLess(extraction_time, 300.0)  # Should complete within 5 minutes
        self.assertGreater(transactions_per_second, 50)  # At least 50 tx/s
    
    def test_large_network_topology_optimization(self):
        """Test topology optimization with large networks."""
        # Create very large network
        from dubchain.ml.network import NetworkTopology, PeerNode
        
        nodes = {}
        for i in range(1000):  # 1000 nodes
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
        for i in range(0, 1000, 5):
            if i + 1 < 1000:
                edges.append((f"node_{i}", f"node_{i+1}"))
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Initialize model
        self.topology_optimizer.initialize_model(5)
        
        # Test under stress
        start_time = time.time()
        result = self.topology_optimizer.optimize_topology(topology)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        logger.info(f"\nLarge Network Topology Optimization:")
        logger.info(f"  Nodes: {len(nodes)}")
        logger.info(f"  Edges: {len(edges)}")
        logger.info(f"  Time: {optimization_time:.4f}s")
        logger.info(f"  Improvement Score: {result.improvement_score:.4f}")
        
        # Stress test assertions
        self.assertLess(optimization_time, 600.0)  # Should complete within 10 minutes
        self.assertIsInstance(result.improvement_score, float)
    
    def test_continuous_anomaly_detection(self):
        """Test continuous anomaly detection under load."""
        # Initialize models
        self.anomaly_detector.initialize_models(20)
        
        # Train on initial data
        initial_data = np.random.rand(1000, 20)
        self.anomaly_detector.train_models(initial_data)
        
        # Continuous detection for 60 seconds
        test_duration = 60
        start_time = time.time()
        
        detection_count = 0
        total_samples = 0
        
        while time.time() - start_time < test_duration:
            # Create batch of test data
            batch_data = np.random.rand(100, 20)
            
            # Detect anomalies
            result = self.anomaly_detector.detect_anomalies(batch_data)
            
            detection_count += 1
            total_samples += len(batch_data)
            
            # Small delay to simulate real-time processing
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        samples_per_second = total_samples / total_time
        
        logger.info(f"\nContinuous Anomaly Detection:")
        logger.info(f"  Duration: {test_duration}s")
        logger.info(f"  Detections: {detection_count}")
        logger.info(f"  Total Samples: {total_samples}")
        logger.info(f"  Throughput: {samples_per_second:.2f} samples/s")
        
        # Stress test assertions
        self.assertGreater(detection_count, 100)  # At least 100 detections
        self.assertGreater(samples_per_second, 10)  # At least 10 samples/s
    
    def test_concurrent_ml_operations(self):
        """Test concurrent ML operations."""
        import threading
        import queue
        
        # Create shared queue for results
        results_queue = queue.Queue()
        
        def feature_extraction_worker():
            """Worker for feature extraction."""
            from dubchain.ml.universal import UniversalTransaction, ChainType, TokenType
            
            transactions = []
            for i in range(1000):
                tx = UniversalTransaction(
                    tx_hash=f"0x{i:040x}",
                    from_address=f"0x{i:040x}",
                    to_address=f"0x{i+1:040x}",
                    value=1000000000000000000,
                    token_type=TokenType.NATIVE,
                    chain_type=ChainType.ETHEREUM,
                    timestamp=time.time()
                )
                transactions.append(tx)
            
            start_time = time.time()
            features = self.feature_pipeline.extract_all_features(transactions)
            end_time = time.time()
            
            results_queue.put(('feature_extraction', end_time - start_time))
        
        def anomaly_detection_worker():
            """Worker for anomaly detection."""
            data = np.random.rand(1000, 20)
            
            start_time = time.time()
            result = self.anomaly_detector.detect_anomalies(data)
            end_time = time.time()
            
            results_queue.put(('anomaly_detection', end_time - start_time))
        
        def topology_optimization_worker():
            """Worker for topology optimization."""
            from dubchain.ml.network import NetworkTopology, PeerNode
            
            nodes = {}
            for i in range(100):
                node = PeerNode(
                    node_id=f"node_{i}",
                    address=f"0x{i:040x}",
                    features=[0.1, 0.2, 0.3, 0.4, 0.5],
                    priority_score=0.8,
                    health_score=0.9,
                    is_active=True
                )
                nodes[f"node_{i}"] = node
            
            edges = []
            for i in range(0, 100, 2):
                if i + 1 < 100:
                    edges.append((f"node_{i}", f"node_{i+1}"))
            
            topology = NetworkTopology(nodes=nodes, edges=edges)
            
            self.topology_optimizer.initialize_model(5)
            
            start_time = time.time()
            result = self.topology_optimizer.optimize_topology(topology)
            end_time = time.time()
            
            results_queue.put(('topology_optimization', end_time - start_time))
        
        # Start concurrent workers
        threads = []
        for _ in range(5):  # 5 concurrent operations
            threads.append(threading.Thread(target=feature_extraction_worker))
            threads.append(threading.Thread(target=anomaly_detection_worker))
            threads.append(threading.Thread(target=topology_optimization_worker))
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        results = {}
        while not results_queue.empty():
            operation, duration = results_queue.get()
            if operation not in results:
                results[operation] = []
            results[operation].append(duration)
        
        logger.info(f"\nConcurrent ML Operations:")
        logger.info(f"  Total Threads: {len(threads)}")
        logger.info(f"  Total Time: {total_time:.4f}s")
        
        for operation, durations in results.items():
            avg_duration = sum(durations) / len(durations)
            logger.info(f"  {operation}: {len(durations)} operations, avg {avg_duration:.4f}s")
        
        # Stress test assertions
        self.assertLess(total_time, 300.0)  # Should complete within 5 minutes
        self.assertEqual(len(results), 3)  # All three operations should complete
    
    def test_memory_stress_test(self):
        """Test memory usage under stress."""
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple large datasets
        datasets = []
        for i in range(10):
            data = np.random.rand(1000, 100)  # 1k samples, 100 features
            datasets.append(data)
        
        # Process all datasets
        results = []
        for i, data in enumerate(datasets):
            self.anomaly_detector.initialize_models(100)
            self.anomaly_detector.train_models(data)
            result = self.anomaly_detector.detect_anomalies(data)
            results.append(result)
            
            # Check memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            logger.info(f"  Dataset {i+1}: Memory increase {memory_increase:.2f} MB")
        
        # Final memory check
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        logger.info(f"\nMemory Stress Test:")
        logger.info(f"  Initial Memory: {initial_memory:.2f} MB")
        logger.info(f"  Final Memory: {final_memory:.2f} MB")
        logger.info(f"  Total Increase: {total_memory_increase:.2f} MB")
        logger.info(f"  Datasets Processed: {len(datasets)}")
        
        # Stress test assertions
        self.assertLess(total_memory_increase, 2000.0)  # Should not increase by more than 2GB
        self.assertEqual(len(results), 10)  # All datasets should be processed


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
