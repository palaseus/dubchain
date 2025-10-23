"""
Anomaly Detection for Byzantine Behavior

This module provides comprehensive anomaly detection for Byzantine behavior including:
- Isolation Forest for outlier detection
- Autoencoders for reconstruction-based anomaly detection
- LSTM-based sequence anomaly detection
- Ensemble methods for robust detection
- Real-time anomaly scoring
- Byzantine behavior classification
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
import json
import hashlib
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...errors import BridgeError, ClientError
from ...logging import get_logger
from ..features import FeaturePipeline, FeatureConfig, TransactionFeatures, NetworkFeatures

logger = get_logger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    enable_isolation_forest: bool = True
    enable_autoencoder: bool = True
    enable_lstm: bool = True
    enable_ensemble: bool = True
    contamination: float = 0.1  # Expected proportion of anomalies
    autoencoder_hidden_dim: int = 32
    autoencoder_latent_dim: int = 16
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    sequence_length: int = 10
    anomaly_threshold: float = 0.5
    enable_real_time: bool = True
    enable_byzantine_classification: bool = True


@dataclass
class AnomalyScore:
    """Anomaly score for a data point."""
    score: float  # 0.0 to 1.0, higher means more anomalous
    confidence: float  # 0.0 to 1.0, confidence in the score
    anomaly_type: str  # Type of anomaly detected
    features: List[float]  # Input features
    timestamp: float = field(default_factory=time.time)
    is_byzantine: bool = False


@dataclass
class ByzantineBehavior:
    """Byzantine behavior classification."""
    behavior_type: str  # double_spending, censorship, sybil, etc.
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Evidence supporting the classification
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""
    anomaly_scores: List[AnomalyScore]
    byzantine_behaviors: List[ByzantineBehavior]
    overall_risk_score: float
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


class Autoencoder(nn.Module):
    """Autoencoder for reconstruction-based anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output."""
        return self.decoder(z)


class LSTMAnomalyDetector(nn.Module):
    """LSTM-based sequence anomaly detector."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, sequence_length: int):
        super(LSTMAnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layers
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
    
    def forward(self, x):
        """Forward pass through LSTM."""
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output for anomaly detection
        last_output = lstm_out[:, -1, :]
        
        # Anomaly score
        anomaly_score = self.anomaly_head(last_output)
        
        # Reconstruction
        reconstruction = self.reconstruction_head(last_output)
        
        return {
            'anomaly_score': anomaly_score,
            'reconstruction': reconstruction,
            'hidden': hidden,
            'cell': cell
        }


class IsolationForestDetector:
    """Isolation Forest-based anomaly detector."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(
                contamination=config.contamination,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
    
    def train(self, data: np.ndarray) -> None:
        """Train the Isolation Forest model."""
        if not SKLEARN_AVAILABLE or self.model is None:
            logger.warning("Scikit-learn not available, skipping Isolation Forest training")
            return
        
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Train the model
            self.model.fit(scaled_data)
            self.is_trained = True
            
            logger.info("Isolation Forest model trained successfully")
            
        except AttributeError as e:
            # Handle scikit-learn version compatibility issues
            logger.warning(f"Scikit-learn compatibility issue: {e}")
            # Try to create a new model with different parameters
            try:
                self.model = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42,
                    n_estimators=50  # Reduce number of estimators
                )
                scaled_data = self.scaler.fit_transform(data)
                self.model.fit(scaled_data)
                self.is_trained = True
                logger.info("Isolation Forest model trained successfully with fallback parameters")
            except Exception as fallback_e:
                logger.error(f"Fallback training also failed: {fallback_e}")
                raise BridgeError(f"Isolation Forest training failed: {fallback_e}")
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest: {e}")
            raise BridgeError(f"Isolation Forest training failed: {e}")
    
    def detect_anomalies(self, data: np.ndarray) -> List[AnomalyScore]:
        """Detect anomalies using Isolation Forest."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return []
        
        try:
            # Scale the data
            scaled_data = self.scaler.transform(data)
            
            # Get anomaly scores
            anomaly_scores = self.model.decision_function(scaled_data)
            
            # Convert to 0-1 scale
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            
            if max_score > min_score:
                normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(anomaly_scores)
            
            # Create anomaly score objects
            results = []
            for i, (score, features) in enumerate(zip(normalized_scores, data)):
                results.append(AnomalyScore(
                    score=float(score),
                    confidence=0.8,  # Fixed confidence for Isolation Forest
                    anomaly_type="isolation_forest",
                    features=features.tolist()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies with Isolation Forest: {e}")
            return []


class AutoencoderDetector:
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, config: AnomalyConfig, input_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.model: Optional[Autoencoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        if TORCH_AVAILABLE:
            self.model = Autoencoder(
                input_dim=input_dim,
                hidden_dim=config.autoencoder_hidden_dim,
                latent_dim=config.autoencoder_latent_dim
            )
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
    
    def train(self, data: np.ndarray, epochs: int = 100) -> None:
        """Train the autoencoder model."""
        if not TORCH_AVAILABLE or not self.model:
            logger.warning("PyTorch not available, skipping autoencoder training")
            return
        
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Convert to PyTorch tensors
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(data_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                
                for batch_data, in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    reconstructed = self.model(batch_data)
                    
                    # Calculate loss
                    loss = criterion(reconstructed, batch_data)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    logger.info(f"Autoencoder Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
            
            self.is_trained = True
            logger.info("Autoencoder model trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train autoencoder: {e}")
            raise BridgeError(f"Autoencoder training failed: {e}")
    
    def detect_anomalies(self, data: np.ndarray) -> List[AnomalyScore]:
        """Detect anomalies using autoencoder reconstruction error."""
        if not self.is_trained or not TORCH_AVAILABLE:
            return []
        
        try:
            # Scale the data
            scaled_data = self.scaler.transform(data)
            
            # Convert to PyTorch tensors
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            
            # Get reconstructions
            self.model.eval()
            with torch.no_grad():
                reconstructed = self.model(data_tensor)
                
                # Calculate reconstruction error
                reconstruction_error = F.mse_loss(reconstructed, data_tensor, reduction='none')
                reconstruction_error = reconstruction_error.mean(dim=1)
                
                # Convert to numpy
                error_scores = reconstruction_error.numpy()
            
            # Normalize scores to 0-1
            min_error = np.min(error_scores)
            max_error = np.max(error_scores)
            
            if max_error > min_error:
                normalized_scores = (error_scores - min_error) / (max_error - min_error)
            else:
                normalized_scores = np.zeros_like(error_scores)
            
            # Create anomaly score objects
            results = []
            for i, (score, features) in enumerate(zip(normalized_scores, data)):
                results.append(AnomalyScore(
                    score=float(score),
                    confidence=0.9,  # Higher confidence for autoencoder
                    anomaly_type="autoencoder",
                    features=features.tolist()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies with autoencoder: {e}")
            return []


class LSTMDetector:
    """LSTM-based sequence anomaly detector."""
    
    def __init__(self, config: AnomalyConfig, input_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.model: Optional[LSTMAnomalyDetector] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        if TORCH_AVAILABLE:
            self.model = LSTMAnomalyDetector(
                input_dim=input_dim,
                hidden_dim=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                sequence_length=config.sequence_length
            )
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
    
    def train(self, sequences: np.ndarray, epochs: int = 100) -> None:
        """Train the LSTM model."""
        if not TORCH_AVAILABLE or not self.model:
            logger.warning("PyTorch not available, skipping LSTM training")
            return
        
        try:
            # Scale the data
            scaled_sequences = self.scaler.fit_transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(sequences.shape)
            
            # Convert to PyTorch tensors
            sequences_tensor = torch.tensor(scaled_sequences, dtype=torch.float32)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(sequences_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                
                for batch_sequences, in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_sequences)
                    
                    # Calculate loss (reconstruction + anomaly)
                    reconstruction_loss = criterion(outputs['reconstruction'], batch_sequences[:, -1, :])
                    anomaly_loss = criterion(outputs['anomaly_score'], torch.zeros_like(outputs['anomaly_score']))
                    
                    total_loss_batch = reconstruction_loss + anomaly_loss
                    
                    # Backward pass
                    total_loss_batch.backward()
                    optimizer.step()
                    
                    total_loss += total_loss_batch.item()
                
                if epoch % 20 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
            
            self.is_trained = True
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train LSTM: {e}")
            raise BridgeError(f"LSTM training failed: {e}")
    
    def detect_anomalies(self, sequences: np.ndarray) -> List[AnomalyScore]:
        """Detect anomalies using LSTM."""
        if not self.is_trained or not TORCH_AVAILABLE:
            return []
        
        try:
            # Scale the data
            scaled_sequences = self.scaler.transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(sequences.shape)
            
            # Convert to PyTorch tensors
            sequences_tensor = torch.tensor(scaled_sequences, dtype=torch.float32)
            
            # Get anomaly scores
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(sequences_tensor)
                anomaly_scores = outputs['anomaly_score'].squeeze().numpy()
            
            # Create anomaly score objects
            results = []
            for i, (score, sequence) in enumerate(zip(anomaly_scores, sequences)):
                results.append(AnomalyScore(
                    score=float(score),
                    confidence=0.85,  # Medium confidence for LSTM
                    anomaly_type="lstm",
                    features=sequence[-1].tolist()  # Use last timestep features
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies with LSTM: {e}")
            return []


class ByzantineClassifier:
    """Classifier for Byzantine behavior types."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.behavior_patterns = {
            "double_spending": {
                "features": ["amount", "fee", "nonce", "timestamp"],
                "threshold": 0.8
            },
            "censorship": {
                "features": ["block_height", "confirmations", "timestamp"],
                "threshold": 0.7
            },
            "sybil": {
                "features": ["degree", "betweenness_centrality", "clustering_coefficient"],
                "threshold": 0.6
            },
            "eclipse": {
                "features": ["latency", "bandwidth", "connection_count"],
                "threshold": 0.75
            }
        }
    
    def classify_behavior(self, anomaly_scores: List[AnomalyScore]) -> List[ByzantineBehavior]:
        """Classify Byzantine behavior types."""
        behaviors = []
        
        for score in anomaly_scores:
            if score.score < self.config.anomaly_threshold:
                continue
            
            # Check each behavior type
            for behavior_type, pattern in self.behavior_patterns.items():
                if self._matches_pattern(score, pattern):
                    behavior = ByzantineBehavior(
                        behavior_type=behavior_type,
                        severity=score.score,
                        confidence=score.confidence,
                        evidence=self._generate_evidence(score, behavior_type)
                    )
                    behaviors.append(behavior)
        
        return behaviors
    
    def _matches_pattern(self, score: AnomalyScore, pattern: Dict[str, Any]) -> bool:
        """Check if anomaly score matches behavior pattern."""
        # Simplified pattern matching
        # Real implementation would use more sophisticated pattern recognition
        
        if len(score.features) < len(pattern["features"]):
            return False
        
        # Check if features exceed threshold
        relevant_features = score.features[:len(pattern["features"])]
        avg_feature_value = np.mean(relevant_features)
        
        return avg_feature_value > pattern["threshold"]
    
    def _generate_evidence(self, score: AnomalyScore, behavior_type: str) -> List[str]:
        """Generate evidence for Byzantine behavior."""
        evidence = []
        
        if behavior_type == "double_spending":
            evidence.append(f"High anomaly score: {score.score:.3f}")
            evidence.append("Suspicious transaction pattern detected")
        
        elif behavior_type == "censorship":
            evidence.append(f"Block confirmation anomaly: {score.score:.3f}")
            evidence.append("Potential transaction censorship")
        
        elif behavior_type == "sybil":
            evidence.append(f"Network topology anomaly: {score.score:.3f}")
            evidence.append("Suspicious peer behavior detected")
        
        elif behavior_type == "eclipse":
            evidence.append(f"Network connectivity anomaly: {score.score:.3f}")
            evidence.append("Potential eclipse attack detected")
        
        return evidence


class AnomalyDetector:
    """Main anomaly detection system."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.isolation_forest = IsolationForestDetector(config)
        self.autoencoder: Optional[AutoencoderDetector] = None
        self.lstm: Optional[LSTMDetector] = None
        self.byzantine_classifier = ByzantineClassifier(config)
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.detection_history = deque(maxlen=1000)
        
    def initialize_models(self, input_dim: int) -> None:
        """Initialize all detection models."""
        if self.config.enable_autoencoder:
            self.autoencoder = AutoencoderDetector(self.config, input_dim)
        
        if self.config.enable_lstm:
            self.lstm = LSTMDetector(self.config, input_dim)
        
        logger.info("Anomaly detection models initialized")
    
    def train_models(self, data: np.ndarray, sequences: Optional[np.ndarray] = None) -> None:
        """Train all detection models."""
        try:
            # Train Isolation Forest
            if self.config.enable_isolation_forest:
                self.isolation_forest.train(data)
            
            # Train Autoencoder
            if self.config.enable_autoencoder and self.autoencoder:
                self.autoencoder.train(data)
            
            # Train LSTM
            if self.config.enable_lstm and self.lstm and sequences is not None:
                self.lstm.train(sequences)
            
            logger.info("All anomaly detection models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detection models: {e}")
            raise BridgeError(f"Model training failed: {e}")
    
    def detect_anomalies(self, data: np.ndarray, sequences: Optional[np.ndarray] = None) -> AnomalyDetectionResult:
        """Detect anomalies using ensemble of methods."""
        all_anomaly_scores = []
        
        try:
            # Isolation Forest detection
            if self.config.enable_isolation_forest:
                if_scores = self.isolation_forest.detect_anomalies(data)
                all_anomaly_scores.extend(if_scores)
            
            # Autoencoder detection
            if self.config.enable_autoencoder and self.autoencoder:
                ae_scores = self.autoencoder.detect_anomalies(data)
                all_anomaly_scores.extend(ae_scores)
            
            # LSTM detection
            if self.config.enable_lstm and self.lstm and sequences is not None:
                lstm_scores = self.lstm.detect_anomalies(sequences)
                all_anomaly_scores.extend(lstm_scores)
            
            # Ensemble scoring
            if self.config.enable_ensemble:
                ensemble_scores = self._ensemble_scoring(all_anomaly_scores)
            else:
                ensemble_scores = all_anomaly_scores
            
            # Byzantine behavior classification
            byzantine_behaviors = []
            if self.config.enable_byzantine_classification:
                byzantine_behaviors = self.byzantine_classifier.classify_behavior(ensemble_scores)
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(ensemble_scores)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(ensemble_scores, byzantine_behaviors)
            
            result = AnomalyDetectionResult(
                anomaly_scores=ensemble_scores,
                byzantine_behaviors=byzantine_behaviors,
                overall_risk_score=overall_risk,
                recommendations=recommendations
            )
            
            # Store in history
            self.detection_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise BridgeError(f"Anomaly detection failed: {e}")
    
    def _ensemble_scoring(self, all_scores: List[AnomalyScore]) -> List[AnomalyScore]:
        """Combine scores from different methods using ensemble."""
        if not all_scores:
            return []
        
        # Group scores by features (assuming same features for same data point)
        score_groups = {}
        for score in all_scores:
            feature_key = tuple(score.features)
            if feature_key not in score_groups:
                score_groups[feature_key] = []
            score_groups[feature_key].append(score)
        
        ensemble_scores = []
        for feature_key, scores in score_groups.items():
            # Calculate ensemble score (weighted average)
            weights = [s.confidence for s in scores]
            weighted_scores = [s.score * w for s, w in zip(scores, weights)]
            
            ensemble_score = sum(weighted_scores) / sum(weights) if sum(weights) > 0 else 0.0
            ensemble_confidence = np.mean([s.confidence for s in scores])
            
            # Determine anomaly type
            anomaly_types = [s.anomaly_type for s in scores]
            most_common_type = max(set(anomaly_types), key=anomaly_types.count)
            
            ensemble_scores.append(AnomalyScore(
                score=ensemble_score,
                confidence=ensemble_confidence,
                anomaly_type=f"ensemble_{most_common_type}",
                features=list(feature_key)
            ))
        
        return ensemble_scores
    
    def _calculate_overall_risk(self, scores: List[AnomalyScore]) -> float:
        """Calculate overall risk score."""
        if not scores:
            return 0.0
        
        # Weight by confidence and take maximum
        risk_scores = [s.score * s.confidence for s in scores]
        return max(risk_scores) if risk_scores else 0.0
    
    def _generate_recommendations(self, scores: List[AnomalyScore], 
                                behaviors: List[ByzantineBehavior]) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []
        
        # High-risk anomalies
        high_risk_scores = [s for s in scores if s.score > 0.8]
        if high_risk_scores:
            recommendations.append("High-risk anomalies detected. Consider immediate investigation.")
        
        # Byzantine behaviors
        if behaviors:
            behavior_types = [b.behavior_type for b in behaviors]
            unique_behaviors = list(set(behavior_types))
            recommendations.append(f"Byzantine behaviors detected: {', '.join(unique_behaviors)}")
        
        # General recommendations
        if len(scores) > 10:
            recommendations.append("High volume of anomalies detected. Consider system-wide review.")
        
        return recommendations
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        recent_results = list(self.detection_history)[-100:]  # Last 100 detections
        
        total_anomalies = sum(len(r.anomaly_scores) for r in recent_results)
        total_byzantine = sum(len(r.byzantine_behaviors) for r in recent_results)
        avg_risk = np.mean([r.overall_risk_score for r in recent_results]) if recent_results else 0.0
        
        return {
            "total_detections": len(self.detection_history),
            "recent_anomalies": total_anomalies,
            "recent_byzantine_behaviors": total_byzantine,
            "average_risk_score": avg_risk,
            "isolation_forest_enabled": self.config.enable_isolation_forest,
            "autoencoder_enabled": self.config.enable_autoencoder,
            "lstm_enabled": self.config.enable_lstm,
            "ensemble_enabled": self.config.enable_ensemble,
            "byzantine_classification_enabled": self.config.enable_byzantine_classification,
            "torch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE
        }
