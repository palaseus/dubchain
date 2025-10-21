"""
Comprehensive Unit and Integration Tests for ML Modules

This module provides comprehensive testing for all ML components including:
- ML infrastructure tests
- Feature engineering tests
- Network topology optimization tests
- Reinforcement learning tests
- Anomaly detection tests
- Bayesian optimization tests
- Model performance tests
- Integration tests
"""

import pytest
import unittest
import asyncio
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import tempfile
import os
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dubchain.ml.infrastructure import MLInfrastructure, ModelConfig, ModelVersion
from dubchain.ml.features import FeaturePipeline, FeatureConfig, TransactionFeatures, NetworkFeatures
from dubchain.ml.network import TopologyOptimizer, TopologyConfig, NetworkTopology, PeerNode
from dubchain.ml.routing import RoutingOptimizer, RoutingConfig, RoutingState, RoutingAction
from dubchain.ml.security import AnomalyDetector, AnomalyConfig, AnomalyScore, ByzantineBehavior
from dubchain.ml.optimization import ConsensusParameterOptimizer, ParameterConfig, ParameterSpace


class TestMLInfrastructure(unittest.TestCase):
    """Test ML infrastructure functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig()
        self.infrastructure = MLInfrastructure(self.config)
    
    def test_infrastructure_initialization(self):
        """Test ML infrastructure initialization."""
        self.assertIsInstance(self.infrastructure, MLInfrastructure)
        self.assertIsInstance(self.infrastructure.config, ModelConfig)
    
    def test_model_versioning(self):
        """Test model versioning functionality."""
        # Create model version
        version = ModelVersion(
            version_id="v1.0.0",
            model_type="GNN",
            created_at=time.time(),
            metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.88}
        )
        
        # Register model version
        self.infrastructure.register_model_version(version)
        
        # Retrieve model version
        retrieved_version = self.infrastructure.get_model_version("v1.0.0")
        
        self.assertEqual(retrieved_version.version_id, "v1.0.0")
        self.assertEqual(retrieved_version.model_type, "GNN")
        self.assertEqual(retrieved_version.metrics["accuracy"], 0.95)
    
    def test_model_serving(self):
        """Test model serving functionality."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.9, 0.7])
        
        # Register model
        self.infrastructure.register_model("test_model", mock_model)
        
        # Test prediction
        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        predictions = self.infrastructure.predict("test_model", input_data)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 3)
        mock_model.predict.assert_called_once()
    
    def test_model_training(self):
        """Test model training functionality."""
        # Mock training data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.score.return_value = 0.85
        
        # Test training
        training_result = self.infrastructure.train_model(
            "test_model",
            mock_model,
            X_train,
            y_train
        )
        
        self.assertIsInstance(training_result, dict)
        self.assertIn('accuracy', training_result)
        self.assertEqual(training_result['accuracy'], 0.85)
        mock_model.fit.assert_called_once()
    
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        # Mock test data
        X_test = np.random.rand(50, 10)
        y_test = np.random.randint(0, 2, 50)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.random.randint(0, 2, 50)
        mock_model.score.return_value = 0.82
        
        # Test evaluation
        evaluation_result = self.infrastructure.evaluate_model(
            "test_model",
            mock_model,
            X_test,
            y_test
        )
        
        self.assertIsInstance(evaluation_result, dict)
        self.assertIn('accuracy', evaluation_result)
        self.assertIn('precision', evaluation_result)
        self.assertIn('recall', evaluation_result)
        self.assertIn('f1_score', evaluation_result)
    
    def test_model_deployment(self):
        """Test model deployment functionality."""
        # Mock model
        mock_model = Mock()
        
        # Test deployment
        deployment_result = self.infrastructure.deploy_model(
            "test_model",
            mock_model,
            "production"
        )
        
        self.assertIsInstance(deployment_result, dict)
        self.assertIn('deployment_id', deployment_result)
        self.assertIn('status', deployment_result)
        self.assertEqual(deployment_result['status'], 'deployed')


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FeatureConfig()
        self.pipeline = FeaturePipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test feature pipeline initialization."""
        self.assertIsInstance(self.pipeline, FeaturePipeline)
        self.assertIsInstance(self.pipeline.config, FeatureConfig)
    
    def test_transaction_feature_extraction(self):
        """Test transaction feature extraction."""
        # Create test transaction
        from dubchain.bridge.universal import UniversalTransaction, ChainType, TokenType
        
        transaction = UniversalTransaction(
            tx_id="0x1234567890abcdef",
            from_address="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            to_address="0x8ba1f109551bD432803012645Hac136c",
            amount=1000000000000000000,
            token_type=TokenType.NATIVE,
            chain_type=ChainType.ETHEREUM,
            created_at=time.time(),
            fee=21000,
            gas_price=20000000000
        )
        
        # Extract features
        features = self.pipeline.transaction_extractor.extract_features(transaction)
        
        self.assertIsInstance(features, TransactionFeatures)
        self.assertEqual(features.tx_id, "0x1234567890abcdef")
        self.assertEqual(features.amount, 1000000000000000000)
        self.assertEqual(features.fee, 21000)
        self.assertGreater(features.amount_log, 0)
        self.assertGreaterEqual(features.fee_ratio, 0)
    
    def test_network_feature_extraction(self):
        """Test network feature extraction."""
        # Create test transactions
        transactions = []
        for i in range(10):
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
        
        # Extract network features
        features = self.pipeline.network_extractor.extract_features(
            "0x0000000000000000000000000000000000000000",
            transactions
        )
        
        self.assertIsInstance(features, NetworkFeatures)
        self.assertEqual(features.node_id, "0x0000000000000000000000000000000000000000")
        self.assertGreaterEqual(features.degree, 0)
        self.assertGreaterEqual(features.betweenness_centrality, 0)
        self.assertGreaterEqual(features.closeness_centrality, 0)
    
    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction."""
        # Create test historical data
        historical_data = []
        base_time = time.time()
        
        for i in range(100):
            data_point = {
                'timestamp': base_time - i * 3600,  # 1 hour intervals
                'value': np.random.rand()
            }
            historical_data.append(data_point)
        
        # Extract temporal features
        features = self.pipeline.temporal_extractor.extract_features(
            base_time,
            historical_data
        )
        
        self.assertIsInstance(features, TemporalFeatures)
        self.assertEqual(features.timestamp, base_time)
        self.assertGreaterEqual(features.hour_of_day, 0)
        self.assertLess(features.hour_of_day, 24)
        self.assertGreaterEqual(features.day_of_week, 0)
        self.assertLess(features.day_of_week, 7)
    
    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        # Create test features
        features = {
            'transaction': [
                TransactionFeatures(
                    tx_id="tx1",
                    amount=1000,
                    fee=10,
                    amount_log=3.0,
                    fee_ratio=0.01,
                    gas_efficiency=0.5,
                    urgency_score=0.8
                ),
                TransactionFeatures(
                    tx_id="tx2",
                    amount=2000,
                    fee=20,
                    amount_log=3.3,
                    fee_ratio=0.01,
                    gas_efficiency=0.6,
                    urgency_score=0.9
                )
            ]
        }
        
        # Scale features
        scaled_features = self.pipeline.scale_features(features)
        
        self.assertIsInstance(scaled_features, dict)
        self.assertIn('transaction', scaled_features)
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Create test features
        features = {
            'transaction': [
                TransactionFeatures(
                    tx_id="tx1",
                    amount=1000,
                    fee=10,
                    amount_log=3.0,
                    fee_ratio=0.01,
                    gas_efficiency=0.5,
                    urgency_score=0.8
                )
            ]
        }
        
        # Create test labels
        labels = [1]
        
        # Select features
        selected_features = self.pipeline.select_features(features, labels)
        
        self.assertIsInstance(selected_features, dict)
    
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction functionality."""
        # Create test features
        features = {
            'transaction': [
                TransactionFeatures(
                    tx_id="tx1",
                    amount=1000,
                    fee=10,
                    amount_log=3.0,
                    fee_ratio=0.01,
                    gas_efficiency=0.5,
                    urgency_score=0.8
                )
            ]
        }
        
        # Reduce dimensions
        reduced_features = self.pipeline.reduce_dimensions(features)
        
        self.assertIsInstance(reduced_features, dict)


class TestNetworkTopologyOptimization(unittest.TestCase):
    """Test network topology optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TopologyConfig()
        self.optimizer = TopologyOptimizer(self.config)
    
    def test_optimizer_initialization(self):
        """Test topology optimizer initialization."""
        self.assertIsInstance(self.optimizer, TopologyOptimizer)
        self.assertIsInstance(self.optimizer.config, TopologyConfig)
    
    def test_gnn_model_initialization(self):
        """Test GNN model initialization."""
        input_dim = 10
        self.optimizer.initialize_model(input_dim)
        
        self.assertIsNotNone(self.optimizer.model)
        self.assertEqual(self.optimizer.model.input_dim, input_dim)
    
    def test_network_topology_creation(self):
        """Test network topology creation."""
        # Create test nodes
        nodes = {}
        for i in range(5):
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
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        # Create topology
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        self.assertIsInstance(topology, NetworkTopology)
        self.assertEqual(len(topology.nodes), 5)
        self.assertEqual(len(topology.edges), 4)
    
    def test_topology_optimization(self):
        """Test topology optimization."""
        # Create test topology
        nodes = {}
        for i in range(5):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Initialize model
        self.optimizer.initialize_model(5)
        
        # Optimize topology
        result = self.optimizer.optimize_topology(topology)
        
        self.assertIsInstance(result, TopologyOptimizationResult)
        self.assertIsInstance(result.optimized_topology, NetworkTopology)
        self.assertIsInstance(result.improvement_score, float)
        self.assertIsInstance(result.removed_edges, list)
        self.assertIsInstance(result.added_edges, list)
        self.assertIsInstance(result.peer_priorities, dict)
        self.assertIsInstance(result.health_scores, dict)
    
    def test_partition_risk_prediction(self):
        """Test partition risk prediction."""
        # Create test topology
        nodes = {}
        for i in range(5):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Predict partition risk
        risk = self.optimizer.predict_partition_risk(topology)
        
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
    
    def test_network_health_calculation(self):
        """Test network health calculation."""
        # Create test topology
        nodes = {}
        for i in range(5):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True,
                connection_count=3,
                latency=50.0,
                bandwidth=1000.0
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Calculate network health
        health_scores = self.optimizer.calculate_network_health(topology)
        
        self.assertIsInstance(health_scores, dict)
        self.assertEqual(len(health_scores), 5)
        
        for node_id, score in health_scores.items():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestReinforcementLearning(unittest.TestCase):
    """Test reinforcement learning functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RoutingConfig()
        self.optimizer = RoutingOptimizer(self.config)
    
    def test_optimizer_initialization(self):
        """Test routing optimizer initialization."""
        self.assertIsInstance(self.optimizer, RoutingOptimizer)
        self.assertIsInstance(self.optimizer.config, RoutingConfig)
    
    def test_agent_initialization(self):
        """Test RL agent initialization."""
        state_dim = 10
        action_dim = 5
        
        self.optimizer.initialize_agents(state_dim, action_dim)
        
        self.assertIsNotNone(self.optimizer.q_learning_agent)
        self.assertIsNotNone(self.optimizer.ppo_agent)
    
    def test_routing_state_creation(self):
        """Test routing state creation."""
        # Create test topology
        nodes = {}
        for i in range(5):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Create routing state
        state = RoutingState(
            current_node="node_0",
            destination_node="node_4",
            available_peers=["node_1", "node_2", "node_3"],
            network_topology=topology,
            traffic_load={},
            latency_matrix={},
            bandwidth_matrix={}
        )
        
        self.assertIsInstance(state, RoutingState)
        self.assertEqual(state.current_node, "node_0")
        self.assertEqual(state.destination_node, "node_4")
        self.assertEqual(len(state.available_peers), 3)
    
    def test_routing_action_creation(self):
        """Test routing action creation."""
        action = RoutingAction(
            selected_peer="node_1",
            route_path=["node_0", "node_1"],
            priority_level=5,
            bandwidth_allocation=0.8
        )
        
        self.assertIsInstance(action, RoutingAction)
        self.assertEqual(action.selected_peer, "node_1")
        self.assertEqual(len(action.route_path), 2)
        self.assertEqual(action.priority_level, 5)
        self.assertEqual(action.bandwidth_allocation, 0.8)
    
    def test_route_optimization(self):
        """Test route optimization."""
        # Create test topology
        nodes = {}
        for i in range(5):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Initialize agents
        self.optimizer.initialize_agents(5, 10)
        
        # Optimize route
        route = self.optimizer.optimize_route("node_0", "node_4", topology)
        
        self.assertIsInstance(route, list)
        self.assertGreater(len(route), 0)
        self.assertEqual(route[0], "node_0")
        self.assertEqual(route[-1], "node_4")
    
    def test_environment_interaction(self):
        """Test environment interaction."""
        # Create test environment
        from dubchain.ml.routing import RoutingEnvironment
        
        env = RoutingEnvironment(self.config)
        
        # Create initial state
        initial_state = RoutingState(
            current_node="node_0",
            destination_node="node_4",
            available_peers=["node_1", "node_2", "node_3"],
            network_topology=NetworkTopology(nodes={}, edges=[]),
            traffic_load={},
            latency_matrix={},
            bandwidth_matrix={}
        )
        
        # Reset environment
        state = env.reset(initial_state)
        
        self.assertEqual(state.current_node, "node_0")
        self.assertEqual(state.destination_node, "node_4")
        
        # Create action
        action = RoutingAction(
            selected_peer="node_1",
            route_path=["node_0", "node_1"],
            priority_level=5,
            bandwidth_allocation=0.8
        )
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        self.assertIsInstance(next_state, RoutingState)
        self.assertIsInstance(reward, RoutingReward)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AnomalyConfig()
        self.detector = AnomalyDetector(self.config)
    
    def test_detector_initialization(self):
        """Test anomaly detector initialization."""
        self.assertIsInstance(self.detector, AnomalyDetector)
        self.assertIsInstance(self.detector.config, AnomalyConfig)
    
    def test_model_initialization(self):
        """Test anomaly detection model initialization."""
        input_dim = 10
        self.detector.initialize_models(input_dim)
        
        self.assertIsNotNone(self.detector.isolation_forest)
        self.assertIsNotNone(self.detector.autoencoder)
        self.assertIsNotNone(self.detector.lstm)
    
    def test_anomaly_score_creation(self):
        """Test anomaly score creation."""
        score = AnomalyScore(
            score=0.8,
            confidence=0.9,
            anomaly_type="isolation_forest",
            features=[0.1, 0.2, 0.3, 0.4, 0.5],
            is_byzantine=True
        )
        
        self.assertIsInstance(score, AnomalyScore)
        self.assertEqual(score.score, 0.8)
        self.assertEqual(score.confidence, 0.9)
        self.assertEqual(score.anomaly_type, "isolation_forest")
        self.assertTrue(score.is_byzantine)
    
    def test_byzantine_behavior_classification(self):
        """Test Byzantine behavior classification."""
        behavior = ByzantineBehavior(
            behavior_type="double_spending",
            severity=0.9,
            confidence=0.95,
            evidence=["High anomaly score", "Suspicious transaction pattern"]
        )
        
        self.assertIsInstance(behavior, ByzantineBehavior)
        self.assertEqual(behavior.behavior_type, "double_spending")
        self.assertEqual(behavior.severity, 0.9)
        self.assertEqual(behavior.confidence, 0.95)
        self.assertEqual(len(behavior.evidence), 2)
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Create test data
        data = np.random.rand(100, 10)
        
        # Initialize models
        self.detector.initialize_models(10)
        
        # Train models
        self.detector.train_models(data)
        
        # Detect anomalies
        result = self.detector.detect_anomalies(data)
        
        self.assertIsInstance(result, AnomalyDetectionResult)
        self.assertIsInstance(result.anomaly_scores, list)
        self.assertIsInstance(result.byzantine_behaviors, list)
        self.assertIsInstance(result.overall_risk_score, float)
        self.assertIsInstance(result.recommendations, list)
    
    def test_byzantine_classification(self):
        """Test Byzantine behavior classification."""
        # Create test anomaly scores
        anomaly_scores = [
            AnomalyScore(
                score=0.9,
                confidence=0.95,
                anomaly_type="isolation_forest",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                is_byzantine=True
            ),
            AnomalyScore(
                score=0.3,
                confidence=0.8,
                anomaly_type="autoencoder",
                features=[0.6, 0.7, 0.8, 0.9, 1.0],
                is_byzantine=False
            )
        ]
        
        # Classify behaviors
        behaviors = self.detector.byzantine_classifier.classify_behavior(anomaly_scores)
        
        self.assertIsInstance(behaviors, list)
        for behavior in behaviors:
            self.assertIsInstance(behavior, ByzantineBehavior)


class TestBayesianOptimization(unittest.TestCase):
    """Test Bayesian optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ParameterConfig()
        self.optimizer = ConsensusParameterOptimizer(self.config)
    
    def test_optimizer_initialization(self):
        """Test parameter optimizer initialization."""
        self.assertIsInstance(self.optimizer, ConsensusParameterOptimizer)
        self.assertIsInstance(self.optimizer.config, ParameterConfig)
    
    def test_parameter_space_creation(self):
        """Test parameter space creation."""
        parameter_space = ParameterSpace(
            parameter_name="block_size",
            parameter_type="continuous",
            bounds=(1000, 10000),
            default_value=5000,
            description="Block size parameter"
        )
        
        self.assertIsInstance(parameter_space, ParameterSpace)
        self.assertEqual(parameter_space.parameter_name, "block_size")
        self.assertEqual(parameter_space.parameter_type, "continuous")
        self.assertEqual(parameter_space.bounds, (1000, 10000))
        self.assertEqual(parameter_space.default_value, 5000)
    
    def test_parameter_optimization(self):
        """Test parameter optimization."""
        # Add parameter spaces
        parameter_space = ParameterSpace(
            parameter_name="block_size",
            parameter_type="continuous",
            bounds=(1000, 10000),
            default_value=5000
        )
        self.optimizer.add_consensus_parameter(parameter_space)
        
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
        
        # Optimize parameters
        result = self.optimizer.optimize_consensus_parameters(performance_evaluator)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.best_parameters, dict)
        self.assertIsInstance(result.best_score, float)
        self.assertIsInstance(result.optimization_history, list)
        self.assertIsInstance(result.convergence_curve, list)
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        # Add parameter spaces
        parameter_space = ParameterSpace(
            parameter_name="block_size",
            parameter_type="continuous",
            bounds=(1000, 10000),
            default_value=5000
        )
        self.optimizer.add_consensus_parameter(parameter_space)
        
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
        
        # Optimize for multiple objectives
        results = self.optimizer.optimize_multi_objective_consensus(performance_evaluator)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIsInstance(result, OptimizationResult)


class TestMLIntegration(unittest.TestCase):
    """Test ML module integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.topology_optimizer = TopologyOptimizer(TopologyConfig())
        self.routing_optimizer = RoutingOptimizer(RoutingConfig())
        self.anomaly_detector = AnomalyDetector(AnomalyConfig())
        self.parameter_optimizer = ConsensusParameterOptimizer(ParameterConfig())
    
    def test_end_to_end_ml_pipeline(self):
        """Test end-to-end ML pipeline."""
        # Create test data
        from dubchain.bridge.universal import UniversalTransaction, ChainType, TokenType
        
        transactions = []
        for i in range(100):
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
        
        # Extract features
        features = self.feature_pipeline.extract_all_features(transactions)
        
        self.assertIsInstance(features, dict)
        self.assertIn('transaction', features)
        
        # Create network topology
        nodes = {}
        for i in range(10):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        # Optimize topology
        self.topology_optimizer.initialize_model(5)
        topology_result = self.topology_optimizer.optimize_topology(topology)
        
        self.assertIsInstance(topology_result, TopologyOptimizationResult)
        
        # Optimize routing
        self.routing_optimizer.initialize_agents(5, 10)
        route = self.routing_optimizer.optimize_route("node_0", "node_4", topology)
        
        self.assertIsInstance(route, list)
        
        # Detect anomalies
        data = np.random.rand(100, 10)
        self.anomaly_detector.initialize_models(10)
        self.anomaly_detector.train_models(data)
        anomaly_result = self.anomaly_detector.detect_anomalies(data)
        
        self.assertIsInstance(anomaly_result, AnomalyDetectionResult)
        
        # Optimize parameters
        parameter_space = ParameterSpace(
            parameter_name="block_size",
            parameter_type="continuous",
            bounds=(1000, 10000),
            default_value=5000
        )
        self.parameter_optimizer.add_consensus_parameter(parameter_space)
        
        def performance_evaluator(parameters):
            from dubchain.ml.optimization import PerformanceMetrics
            return PerformanceMetrics(
                throughput=parameters["block_size"] / 1000,
                latency=1000 / parameters["block_size"],
                security_score=0.9,
                energy_efficiency=0.8,
                scalability=0.7
            )
        
        optimization_result = self.parameter_optimizer.optimize_consensus_parameters(performance_evaluator)
        
        self.assertIsInstance(optimization_result, OptimizationResult)
    
    def test_ml_performance_benchmarking(self):
        """Test ML performance benchmarking."""
        # Benchmark feature extraction
        start_time = time.time()
        features = self.feature_pipeline.extract_all_features([])
        feature_time = time.time() - start_time
        
        self.assertLess(feature_time, 1.0)  # Should complete within 1 second
        
        # Benchmark topology optimization
        nodes = {}
        for i in range(5):
            node = PeerNode(
                node_id=f"node_{i}",
                address=f"0x{i:040x}",
                features=[0.1, 0.2, 0.3, 0.4, 0.5],
                priority_score=0.8,
                health_score=0.9,
                is_active=True
            )
            nodes[f"node_{i}"] = node
        
        edges = [
            ("node_0", "node_1"),
            ("node_1", "node_2"),
            ("node_2", "node_3"),
            ("node_3", "node_4")
        ]
        
        topology = NetworkTopology(nodes=nodes, edges=edges)
        
        start_time = time.time()
        self.topology_optimizer.initialize_model(5)
        topology_result = self.topology_optimizer.optimize_topology(topology)
        topology_time = time.time() - start_time
        
        self.assertLess(topology_time, 5.0)  # Should complete within 5 seconds
        
        # Benchmark anomaly detection
        data = np.random.rand(100, 10)
        
        start_time = time.time()
        self.anomaly_detector.initialize_models(10)
        self.anomaly_detector.train_models(data)
        anomaly_result = self.anomaly_detector.detect_anomalies(data)
        anomaly_time = time.time() - start_time
        
        self.assertLess(anomaly_time, 10.0)  # Should complete within 10 seconds


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
