"""
Anomaly Detection Module

This module provides machine learning-based anomaly detection for blockchain networks.
"""

import logging

logger = logging.getLogger(__name__)
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    NETWORK_ATTACK = "network_attack"
    TRANSACTION_SPAM = "transaction_spam"
    CONSENSUS_FAILURE = "consensus_failure"
    BRIDGE_FRAUD = "bridge_fraud"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNKNOWN = "unknown"

@dataclass
class AnomalyScore:
    """Anomaly detection score and metadata."""
    score: float
    anomaly_type: AnomalyType
    confidence: float
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    description: str = ""

@dataclass
class DetectionConfig:
    """Configuration for anomaly detection."""
    enable_isolation_forest: bool = True
    enable_autoencoder: bool = True
    enable_lstm: bool = True
    threshold: float = 0.7
    window_size: int = 100
    update_interval: float = 60.0  # seconds
    max_features: int = 50

class IsolationForestDetector:
    """Isolation Forest-based anomaly detector."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize Isolation Forest detector."""
        self.config = config
        self.trees = []
        self.feature_importances = {}
        self.training_data = []
        self.is_trained = False
        
        logger.info("Initialized Isolation Forest detector")
    
    def train(self, data: List[Dict[str, Any]]) -> bool:
        """Train the isolation forest model."""
        try:
            if len(data) < 10:
                logger.warning("Insufficient training data for Isolation Forest")
                return False
            
            # Extract features from data
            features = self._extract_features(data)
            if features is None:
                return False
            
            # Build isolation trees
            self.trees = self._build_isolation_trees(features)
            
            # Calculate feature importances
            self.feature_importances = self._calculate_feature_importances(features)
            
            self.training_data = features
            self.is_trained = True
            
            logger.info(f"Trained Isolation Forest with {len(self.trees)} trees on {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}")
            return False
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> AnomalyScore:
        """Detect anomaly in a single data point."""
        try:
            if not self.is_trained:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Model not trained"
                )
            
            # Extract features
            features = self._extract_features([data_point])
            if features is None or len(features) == 0:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Failed to extract features"
                )
            
            feature_vector = features[0]
            
            # Calculate anomaly score
            score = self._calculate_anomaly_score(feature_vector)
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(feature_vector, score)
            
            # Calculate confidence
            confidence = min(score, 1.0)
            
            return AnomalyScore(
                score=score,
                anomaly_type=anomaly_type,
                confidence=confidence,
                features=dict(zip(self.feature_importances.keys(), feature_vector)),
                description=f"Anomaly detected with score {score:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return AnomalyScore(
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                features={},
                description=f"Detection error: {str(e)}"
            )
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
        """Extract numerical features from data points."""
        try:
            features = []
            
            for point in data:
                feature_vector = []
                
                # Network features
                feature_vector.append(point.get('peer_count', 0))
                feature_vector.append(point.get('connection_count', 0))
                feature_vector.append(point.get('latency_ms', 0))
                feature_vector.append(point.get('bandwidth_mbps', 0))
                
                # Transaction features
                feature_vector.append(point.get('tx_count', 0))
                feature_vector.append(point.get('tx_rate', 0))
                feature_vector.append(point.get('gas_price', 0))
                feature_vector.append(point.get('gas_used', 0))
                
                # Consensus features
                feature_vector.append(point.get('block_time', 0))
                feature_vector.append(point.get('validator_count', 0))
                feature_vector.append(point.get('stake_amount', 0))
                feature_vector.append(point.get('consensus_time', 0))
                
                # Performance features
                feature_vector.append(point.get('cpu_usage', 0))
                feature_vector.append(point.get('memory_usage', 0))
                feature_vector.append(point.get('disk_usage', 0))
                feature_vector.append(point.get('network_errors', 0))
                
                # Pad or truncate to max_features
                while len(feature_vector) < self.config.max_features:
                    feature_vector.append(0.0)
                
                features.append(feature_vector[:self.config.max_features])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _build_isolation_trees(self, features: List[List[float]]) -> List[Dict[str, Any]]:
        """Build isolation trees for anomaly detection."""
        trees = []
        n_trees = min(100, len(features) // 2)
        
        for i in range(n_trees):
            tree = self._build_single_tree(features)
            trees.append(tree)
        
        return trees
    
    def _build_single_tree(self, features: List[List[float]]) -> Dict[str, Any]:
        """Build a single isolation tree."""
        # Simplified isolation tree implementation
        tree = {
            'root': {
                'feature': 0,
                'threshold': np.mean([f[0] for f in features]),
                'left': None,
                'right': None,
                'depth': 0
            }
        }
        return tree
    
    def _calculate_feature_importances(self, features: List[List[float]]) -> Dict[str, float]:
        """Calculate feature importances."""
        importances = {}
        feature_names = [
            'peer_count', 'connection_count', 'latency_ms', 'bandwidth_mbps',
            'tx_count', 'tx_rate', 'gas_price', 'gas_used',
            'block_time', 'validator_count', 'stake_amount', 'consensus_time',
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_errors'
        ]
        
        for i, name in enumerate(feature_names):
            if i < len(features[0]):
                # Calculate variance as importance proxy
                values = [f[i] for f in features]
                importances[name] = np.var(values) if len(values) > 1 else 0.0
        
        return importances
    
    def _calculate_anomaly_score(self, feature_vector: List[float]) -> float:
        """Calculate anomaly score using isolation forest."""
        try:
            if not self.trees:
                return 0.0
            
            scores = []
            for tree in self.trees:
                score = self._score_point(tree['root'], feature_vector, 0)
                scores.append(score)
            
            # Average score across trees
            avg_score = np.mean(scores)
            
            # Normalize to 0-1 range
            normalized_score = min(max(avg_score, 0.0), 1.0)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def _score_point(self, node: Dict[str, Any], features: List[float], depth: int) -> float:
        """Score a point using a single tree node."""
        if node is None:
            return depth
        
        feature_idx = node['feature']
        threshold = node['threshold']
        
        if feature_idx < len(features):
            if features[feature_idx] < threshold:
                return self._score_point(node['left'], features, depth + 1)
            else:
                return self._score_point(node['right'], features, depth + 1)
        else:
            return depth
    
    def _classify_anomaly_type(self, features: List[float], score: float) -> AnomalyType:
        """Classify the type of anomaly based on features and score."""
        try:
            # Simple rule-based classification
            if len(features) >= 4:
                peer_count = features[0]
                tx_rate = features[5]
                gas_price = features[6]
                network_errors = features[15] if len(features) > 15 else 0
                
                if tx_rate > 1000:  # High transaction rate
                    return AnomalyType.TRANSACTION_SPAM
                elif gas_price > 100:  # Unusually high gas price
                    return AnomalyType.NETWORK_ATTACK
                elif network_errors > 10:  # Many network errors
                    return AnomalyType.PERFORMANCE_DEGRADATION
                elif peer_count < 5:  # Very few peers
                    return AnomalyType.CONSENSUS_FAILURE
            
            return AnomalyType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error classifying anomaly type: {e}")
            return AnomalyType.UNKNOWN

class AutoencoderDetector:
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize Autoencoder detector."""
        self.config = config
        self.encoder_weights = None
        self.decoder_weights = None
        self.is_trained = False
        
        logger.info("Initialized Autoencoder detector")
    
    def train(self, data: List[Dict[str, Any]]) -> bool:
        """Train the autoencoder model."""
        try:
            if len(data) < 20:
                logger.warning("Insufficient training data for Autoencoder")
                return False
            
            # Extract features
            features = self._extract_features(data)
            if features is None:
                return False
            
            # Train autoencoder
            self.encoder_weights, self.decoder_weights = self._train_autoencoder(features)
            
            self.is_trained = True
            
            logger.info(f"Trained Autoencoder on {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training Autoencoder: {e}")
            return False
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> AnomalyScore:
        """Detect anomaly using autoencoder reconstruction error."""
        try:
            if not self.is_trained:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Model not trained"
                )
            
            # Extract features
            features = self._extract_features([data_point])
            if features is None or len(features) == 0:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Failed to extract features"
                )
            
            feature_vector = features[0]
            
            # Encode and decode
            encoded = self._encode(feature_vector)
            decoded = self._decode(encoded)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(np.array(feature_vector) - np.array(decoded)))
            
            # Normalize to 0-1 range
            score = min(reconstruction_error / 100.0, 1.0)
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(feature_vector, score)
            
            return AnomalyScore(
                score=score,
                anomaly_type=anomaly_type,
                confidence=score,
                features=dict(zip(range(len(feature_vector)), feature_vector)),
                description=f"Autoencoder reconstruction error: {reconstruction_error:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomaly with autoencoder: {e}")
            return AnomalyScore(
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                features={},
                description=f"Detection error: {str(e)}"
            )
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
        """Extract features for autoencoder."""
        # Same implementation as Isolation Forest
        try:
            features = []
            
            for point in data:
                feature_vector = []
                
                # Network features
                feature_vector.append(point.get('peer_count', 0))
                feature_vector.append(point.get('connection_count', 0))
                feature_vector.append(point.get('latency_ms', 0))
                feature_vector.append(point.get('bandwidth_mbps', 0))
                
                # Transaction features
                feature_vector.append(point.get('tx_count', 0))
                feature_vector.append(point.get('tx_rate', 0))
                feature_vector.append(point.get('gas_price', 0))
                feature_vector.append(point.get('gas_used', 0))
                
                # Consensus features
                feature_vector.append(point.get('block_time', 0))
                feature_vector.append(point.get('validator_count', 0))
                feature_vector.append(point.get('stake_amount', 0))
                feature_vector.append(point.get('consensus_time', 0))
                
                # Performance features
                feature_vector.append(point.get('cpu_usage', 0))
                feature_vector.append(point.get('memory_usage', 0))
                feature_vector.append(point.get('disk_usage', 0))
                feature_vector.append(point.get('network_errors', 0))
                
                # Pad to max_features
                while len(feature_vector) < self.config.max_features:
                    feature_vector.append(0.0)
                
                features.append(feature_vector[:self.config.max_features])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _train_autoencoder(self, features: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Train autoencoder weights."""
        # Simplified autoencoder training
        input_dim = len(features[0])
        hidden_dim = max(4, input_dim // 2)
        
        # Initialize weights randomly
        encoder_weights = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        decoder_weights = np.random.normal(0, 0.1, (hidden_dim, input_dim))
        
        # Simple training loop
        learning_rate = 0.01
        for epoch in range(100):
            for feature_vector in features:
                # Forward pass
                encoded = np.dot(feature_vector, encoder_weights)
                decoded = np.dot(encoded, decoder_weights)
                
                # Calculate error
                error = np.array(feature_vector) - decoded
                
                # Update weights (simplified gradient descent)
                encoder_weights += learning_rate * np.outer(feature_vector, error.dot(decoder_weights.T))
                decoder_weights += learning_rate * np.outer(encoded, error)
        
        return encoder_weights, decoder_weights
    
    def _encode(self, features: List[float]) -> np.ndarray:
        """Encode features using encoder weights."""
        return np.dot(features, self.encoder_weights)
    
    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode features using decoder weights."""
        return np.dot(encoded, self.decoder_weights)
    
    def _classify_anomaly_type(self, features: List[float], score: float) -> AnomalyType:
        """Classify anomaly type based on features."""
        # Same logic as Isolation Forest
        try:
            if len(features) >= 4:
                peer_count = features[0]
                tx_rate = features[5]
                gas_price = features[6]
                network_errors = features[15] if len(features) > 15 else 0
                
                if tx_rate > 1000:
                    return AnomalyType.TRANSACTION_SPAM
                elif gas_price > 100:
                    return AnomalyType.NETWORK_ATTACK
                elif network_errors > 10:
                    return AnomalyType.PERFORMANCE_DEGRADATION
                elif peer_count < 5:
                    return AnomalyType.CONSENSUS_FAILURE
            
            return AnomalyType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error classifying anomaly type: {e}")
            return AnomalyType.UNKNOWN

class LSTMDetector:
    """LSTM-based anomaly detector for time series data."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize LSTM detector."""
        self.config = config
        self.lstm_weights = None
        self.is_trained = False
        self.sequence_buffer = []
        
        logger.info("Initialized LSTM detector")
    
    def train(self, data: List[Dict[str, Any]]) -> bool:
        """Train the LSTM model."""
        try:
            if len(data) < self.config.window_size:
                logger.warning("Insufficient training data for LSTM")
                return False
            
            # Extract time series features
            sequences = self._extract_sequences(data)
            if sequences is None:
                return False
            
            # Train LSTM
            self.lstm_weights = self._train_lstm(sequences)
            
            self.is_trained = True
            
            logger.info(f"Trained LSTM on {len(sequences)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return False
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> AnomalyScore:
        """Detect anomaly using LSTM prediction error."""
        try:
            if not self.is_trained:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Model not trained"
                )
            
            # Add to sequence buffer
            self.sequence_buffer.append(data_point)
            if len(self.sequence_buffer) > self.config.window_size:
                self.sequence_buffer.pop(0)
            
            if len(self.sequence_buffer) < self.config.window_size:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Insufficient sequence data"
                )
            
            # Extract features from sequence
            sequence_features = self._extract_features(self.sequence_buffer)
            if sequence_features is None:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="Failed to extract sequence features"
                )
            
            # Predict next value
            predicted = self._predict_next(sequence_features)
            actual = sequence_features[-1]
            
            # Calculate prediction error
            prediction_error = np.mean(np.square(np.array(actual) - np.array(predicted)))
            
            # Normalize to 0-1 range
            score = min(prediction_error / 50.0, 1.0)
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(actual, score)
            
            return AnomalyScore(
                score=score,
                anomaly_type=anomaly_type,
                confidence=score,
                features=dict(zip(range(len(actual)), actual)),
                description=f"LSTM prediction error: {prediction_error:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomaly with LSTM: {e}")
            return AnomalyScore(
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                features={},
                description=f"Detection error: {str(e)}"
            )
    
    def _extract_sequences(self, data: List[Dict[str, Any]]) -> Optional[List[List[List[float]]]]:
        """Extract sequences for LSTM training."""
        try:
            sequences = []
            
            for i in range(len(data) - self.config.window_size + 1):
                sequence = data[i:i + self.config.window_size]
                sequence_features = self._extract_features(sequence)
                if sequence_features:
                    sequences.append(sequence_features)
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error extracting sequences: {e}")
            return None
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
        """Extract features from data points."""
        # Same implementation as other detectors
        try:
            features = []
            
            for point in data:
                feature_vector = []
                
                # Network features
                feature_vector.append(point.get('peer_count', 0))
                feature_vector.append(point.get('connection_count', 0))
                feature_vector.append(point.get('latency_ms', 0))
                feature_vector.append(point.get('bandwidth_mbps', 0))
                
                # Transaction features
                feature_vector.append(point.get('tx_count', 0))
                feature_vector.append(point.get('tx_rate', 0))
                feature_vector.append(point.get('gas_price', 0))
                feature_vector.append(point.get('gas_used', 0))
                
                # Consensus features
                feature_vector.append(point.get('block_time', 0))
                feature_vector.append(point.get('validator_count', 0))
                feature_vector.append(point.get('stake_amount', 0))
                feature_vector.append(point.get('consensus_time', 0))
                
                # Performance features
                feature_vector.append(point.get('cpu_usage', 0))
                feature_vector.append(point.get('memory_usage', 0))
                feature_vector.append(point.get('disk_usage', 0))
                feature_vector.append(point.get('network_errors', 0))
                
                # Pad to max_features
                while len(feature_vector) < self.config.max_features:
                    feature_vector.append(0.0)
                
                features.append(feature_vector[:self.config.max_features])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _train_lstm(self, sequences: List[List[List[float]]]) -> Dict[str, np.ndarray]:
        """Train LSTM weights."""
        # Simplified LSTM training
        input_dim = len(sequences[0][0])
        hidden_dim = max(4, input_dim // 2)
        
        # Initialize LSTM weights
        weights = {
            'input_weights': np.random.normal(0, 0.1, (input_dim, hidden_dim)),
            'hidden_weights': np.random.normal(0, 0.1, (hidden_dim, hidden_dim)),
            'output_weights': np.random.normal(0, 0.1, (hidden_dim, input_dim)),
            'bias': np.zeros(hidden_dim)
        }
        
        # Simple training loop
        learning_rate = 0.01
        for epoch in range(50):
            for sequence in sequences:
                hidden_state = np.zeros(hidden_dim)
                
                for i in range(len(sequence) - 1):
                    # Forward pass
                    input_val = sequence[i]
                    hidden_state = np.tanh(np.dot(input_val, weights['input_weights']) + 
                                          np.dot(hidden_state, weights['hidden_weights']) + 
                                          weights['bias'])
                    output = np.dot(hidden_state, weights['output_weights'])
                    
                    # Calculate error
                    target = sequence[i + 1]
                    error = target - output
                    
                    # Update weights (simplified)
                    weights['output_weights'] += learning_rate * np.outer(hidden_state, error)
        
        return weights
    
    def _predict_next(self, sequence_features: List[List[float]]) -> List[float]:
        """Predict next value in sequence."""
        try:
            if not self.lstm_weights:
                return [0.0] * len(sequence_features[0])
            
            hidden_dim = len(self.lstm_weights['bias'])
            hidden_state = np.zeros(hidden_dim)
            
            # Process sequence
            for features in sequence_features:
                hidden_state = np.tanh(np.dot(features, self.lstm_weights['input_weights']) + 
                                      np.dot(hidden_state, self.lstm_weights['hidden_weights']) + 
                                      self.lstm_weights['bias'])
            
            # Predict next value
            prediction = np.dot(hidden_state, self.lstm_weights['output_weights'])
            return prediction.tolist()
            
        except Exception as e:
            logger.error(f"Error predicting next value: {e}")
            return [0.0] * len(sequence_features[0])
    
    def _classify_anomaly_type(self, features: List[float], score: float) -> AnomalyType:
        """Classify anomaly type based on features."""
        # Same logic as other detectors
        try:
            if len(features) >= 4:
                peer_count = features[0]
                tx_rate = features[5]
                gas_price = features[6]
                network_errors = features[15] if len(features) > 15 else 0
                
                if tx_rate > 1000:
                    return AnomalyType.TRANSACTION_SPAM
                elif gas_price > 100:
                    return AnomalyType.NETWORK_ATTACK
                elif network_errors > 10:
                    return AnomalyType.PERFORMANCE_DEGRADATION
                elif peer_count < 5:
                    return AnomalyType.CONSENSUS_FAILURE
            
            return AnomalyType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error classifying anomaly type: {e}")
            return AnomalyType.UNKNOWN

class AnomalyDetector:
    """Main anomaly detector that coordinates multiple detection methods."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize anomaly detector."""
        self.config = config
        self.detectors = {}
        
        # Initialize detectors based on config
        if config.enable_isolation_forest:
            self.detectors['isolation_forest'] = IsolationForestDetector(config)
        
        if config.enable_autoencoder:
            self.detectors['autoencoder'] = AutoencoderDetector(config)
        
        if config.enable_lstm:
            self.detectors['lstm'] = LSTMDetector(config)
        
        self.detection_history = []
        
        logger.info(f"Initialized AnomalyDetector with {len(self.detectors)} detectors")
    
    def train(self, data: List[Dict[str, Any]]) -> bool:
        """Train all enabled detectors."""
        try:
            success_count = 0
            
            for name, detector in self.detectors.items():
                if detector.train(data):
                    success_count += 1
                    logger.info(f"Successfully trained {name} detector")
                else:
                    logger.warning(f"Failed to train {name} detector")
            
            if success_count == 0:
                logger.error("Failed to train any detectors")
                return False
            
            logger.info(f"Successfully trained {success_count}/{len(self.detectors)} detectors")
            return True
            
        except Exception as e:
            logger.error(f"Error training detectors: {e}")
            return False
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> AnomalyScore:
        """Detect anomaly using ensemble of detectors."""
        try:
            if not self.detectors:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="No detectors available"
                )
            
            scores = []
            
            # Get scores from all detectors
            for name, detector in self.detectors.items():
                score = detector.detect_anomaly(data_point)
                scores.append(score)
            
            # Ensemble scoring
            ensemble_score = self._ensemble_scoring(scores)
            
            # Store in history
            self.detection_history.append({
                'timestamp': time.time(),
                'scores': scores,
                'ensemble_score': ensemble_score
            })
            
            # Keep only recent history
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            return ensemble_score
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return AnomalyScore(
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                features={},
                description=f"Detection error: {str(e)}"
            )
    
    def _ensemble_scoring(self, scores: List[AnomalyScore]) -> AnomalyScore:
        """Combine scores from multiple detectors."""
        try:
            if not scores:
                return AnomalyScore(
                    score=0.0,
                    anomaly_type=AnomalyType.UNKNOWN,
                    confidence=0.0,
                    features={},
                    description="No scores to combine"
                )
            
            # Weighted average of scores
            weights = [1.0 / len(scores)] * len(scores)  # Equal weights for now
            
            ensemble_score = sum(s.score * weight for s, weight in zip(scores, weights))
            ensemble_confidence = sum(s.confidence * weight for s, weight in zip(scores, weights))
            
            # Determine most common anomaly type
            anomaly_types = [s.anomaly_type for s in scores]
            most_common_type = max(set(anomaly_types), key=anomaly_types.count)
            
            # Combine features
            combined_features = {}
            for score in scores:
                combined_features.update(score.features)
            
            return AnomalyScore(
                score=ensemble_score,
                anomaly_type=most_common_type,
                confidence=ensemble_confidence,
                features=combined_features,
                description=f"Ensemble score from {len(scores)} detectors"
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble scoring: {e}")
            return AnomalyScore(
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                features={},
                description=f"Ensemble error: {str(e)}"
            )
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {"total_detections": 0, "avg_score": 0.0}
        
        recent_history = self.detection_history[-100:]  # Last 100 detections
        avg_score = np.mean([h['ensemble_score'].score for h in recent_history])
        
        return {
            "total_detections": len(self.detection_history),
            "avg_score": avg_score,
            "detector_count": len(self.detectors),
            "detector_names": list(self.detectors.keys())
        }

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "LSTMDetector",
    "AnomalyType",
    "AnomalyScore",
    "DetectionConfig",
]