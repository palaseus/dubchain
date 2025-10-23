"""
Feature Engineering Pipeline for Blockchain Data

This module provides comprehensive feature engineering capabilities including:
- Blockchain data extraction and preprocessing
- Feature scaling and normalization
- Feature selection and dimensionality reduction
- Time series feature engineering
- Graph-based feature extraction
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
import time
from collections import defaultdict, deque

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

# Try to import scikit-learn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_regression
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Feature engineering will be limited.")

# Try to import NumPy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Feature engineering will be limited.")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    window_size: int = 100
    max_features: int = 1000
    scaling_method: str = "standard"  # "standard", "minmax", "robust"
    feature_selection_method: str = "mutual_info"  # "mutual_info", "variance", "correlation"
    n_features_to_select: int = 100
    enable_pca: bool = True
    pca_components: int = 50
    enable_tsne: bool = False
    tsne_components: int = 2


class FeatureExtractor:
    """Extracts features from blockchain data."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_cache: Dict[str, Any] = {}
    
    def extract_transaction_features(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from transaction data."""
        try:
            if not transactions:
                return {}
            
            features = {}
            
            # Basic transaction features
            features["transaction_count"] = len(transactions)
            features["total_value"] = sum(tx.get("value", 0) for tx in transactions)
            features["avg_value"] = features["total_value"] / len(transactions) if transactions else 0
            features["max_value"] = max(tx.get("value", 0) for tx in transactions)
            features["min_value"] = min(tx.get("value", 0) for tx in transactions)
            
            # Time-based features
            timestamps = [tx.get("timestamp", 0) for tx in transactions]
            if timestamps:
                features["time_span"] = max(timestamps) - min(timestamps)
                features["avg_time_interval"] = features["time_span"] / len(transactions) if transactions else 0
            
            # Address-based features
            senders = [tx.get("from", "") for tx in transactions]
            receivers = [tx.get("to", "") for tx in transactions]
            
            features["unique_senders"] = len(set(senders))
            features["unique_receivers"] = len(set(receivers))
            features["unique_addresses"] = len(set(senders + receivers))
            
            # Gas features (if available)
            gas_prices = [tx.get("gas_price", 0) for tx in transactions]
            gas_used = [tx.get("gas_used", 0) for tx in transactions]
            
            if gas_prices and any(gas_prices):
                features["avg_gas_price"] = np.mean(gas_prices)
                features["max_gas_price"] = np.max(gas_prices)
                features["min_gas_price"] = np.min(gas_prices)
            
            if gas_used and any(gas_used):
                features["avg_gas_used"] = np.mean(gas_used)
                features["total_gas_used"] = np.sum(gas_used)
            
            logger.info(f"Extracted {len(features)} transaction features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract transaction features: {e}")
            return {}
    
    def extract_block_features(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from block data."""
        try:
            if not blocks:
                return {}
            
            features = {}
            
            # Basic block features
            features["block_count"] = len(blocks)
            features["avg_block_size"] = np.mean([block.get("size", 0) for block in blocks])
            features["max_block_size"] = np.max([block.get("size", 0) for block in blocks])
            features["min_block_size"] = np.min([block.get("size", 0) for block in blocks])
            
            # Transaction features
            tx_counts = [block.get("transaction_count", 0) for block in blocks]
            features["avg_tx_per_block"] = np.mean(tx_counts)
            features["max_tx_per_block"] = np.max(tx_counts)
            features["min_tx_per_block"] = np.min(tx_counts)
            
            # Time-based features
            timestamps = [block.get("timestamp", 0) for block in blocks]
            if len(timestamps) > 1:
                time_intervals = np.diff(sorted(timestamps))
                features["avg_block_interval"] = np.mean(time_intervals)
                features["min_block_interval"] = np.min(time_intervals)
                features["max_block_interval"] = np.max(time_intervals)
                features["block_interval_std"] = np.std(time_intervals)
            
            # Difficulty features (if available)
            difficulties = [block.get("difficulty", 0) for block in blocks]
            if difficulties and any(difficulties):
                features["avg_difficulty"] = np.mean(difficulties)
                features["difficulty_trend"] = np.polyfit(range(len(difficulties)), difficulties, 1)[0]
            
            logger.info(f"Extracted {len(features)} block features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract block features: {e}")
            return {}
    
    def extract_network_features(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from network data."""
        try:
            features = {}
            
            # Peer features
            peers = network_data.get("peers", [])
            features["peer_count"] = len(peers)
            
            if peers:
                # Geographic features
                countries = [peer.get("country", "unknown") for peer in peers]
                features["unique_countries"] = len(set(countries))
                features["most_common_country"] = max(set(countries), key=countries.count)
                
                # Connection features
                connection_types = [peer.get("connection_type", "unknown") for peer in peers]
                features["unique_connection_types"] = len(set(connection_types))
                
                # Latency features
                latencies = [peer.get("latency", 0) for peer in peers if peer.get("latency")]
                if latencies:
                    features["avg_latency"] = np.mean(latencies)
                    features["min_latency"] = np.min(latencies)
                    features["max_latency"] = np.max(latencies)
                    features["latency_std"] = np.std(latencies)
            
            # Network health features
            features["network_hashrate"] = network_data.get("hashrate", 0)
            features["network_difficulty"] = network_data.get("difficulty", 0)
            features["active_nodes"] = network_data.get("active_nodes", 0)
            features["sync_status"] = network_data.get("sync_status", "unknown")
            
            logger.info(f"Extracted {len(features)} network features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract network features: {e}")
            return {}
    
    def extract_time_series_features(self, data: List[float], window_size: int = None) -> Dict[str, Any]:
        """Extract time series features."""
        try:
            if not data or len(data) < 2:
                return {}
            
            window_size = window_size or self.config.window_size
            features = {}
            
            # Basic statistics
            features["mean"] = np.mean(data)
            features["std"] = np.std(data)
            features["min"] = np.min(data)
            features["max"] = np.max(data)
            features["median"] = np.median(data)
            features["skewness"] = self._calculate_skewness(data)
            features["kurtosis"] = self._calculate_kurtosis(data)
            
            # Trend features
            if len(data) > 1:
                features["trend"] = np.polyfit(range(len(data)), data, 1)[0]
                features["trend_strength"] = abs(features["trend"])
            
            # Rolling window features
            if len(data) >= window_size:
                rolling_mean = pd.Series(data).rolling(window=window_size).mean()
                rolling_std = pd.Series(data).rolling(window=window_size).std()
                
                features["rolling_mean_avg"] = rolling_mean.mean()
                features["rolling_std_avg"] = rolling_std.mean()
                features["rolling_mean_trend"] = np.polyfit(range(len(rolling_mean)), rolling_mean, 1)[0]
            
            # Autocorrelation features
            if len(data) > 10:
                features["autocorr_lag1"] = self._calculate_autocorrelation(data, lag=1)
                features["autocorr_lag5"] = self._calculate_autocorrelation(data, lag=5)
            
            logger.info(f"Extracted {len(features)} time series features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract time series features: {e}")
            return {}
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness."""
        if not NUMPY_AVAILABLE or len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis."""
        if not NUMPY_AVAILABLE or len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean([(x - mean) ** 4 for x in data]) / (std ** 4) - 3
    
    def _calculate_autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation."""
        if len(data) <= lag:
            return 0.0
        
        mean = np.mean(data)
        numerator = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(len(data) - lag))
        denominator = sum((x - mean) ** 2 for x in data)
        
        return numerator / denominator if denominator != 0 else 0.0


class FeatureScaler:
    """Scales and normalizes features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scalers: Dict[str, Any] = {}
    
    def fit_transform(self, features: Dict[str, Any], feature_name: str = "default") -> Dict[str, Any]:
        """Fit scaler and transform features."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available. Returning unscaled features.")
                return features
            
            # Convert features to array
            feature_values = list(features.values())
            feature_array = np.array(feature_values).reshape(-1, 1)
            
            # Create and fit scaler
            if self.config.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif self.config.scaling_method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            scaled_array = scaler.fit_transform(feature_array)
            
            # Store scaler
            self.scalers[feature_name] = scaler
            
            # Convert back to dict
            scaled_features = dict(zip(features.keys(), scaled_array.flatten()))
            
            logger.info(f"Scaled {len(features)} features using {self.config.scaling_method}")
            return scaled_features
            
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            return features
    
    def transform(self, features: Dict[str, Any], feature_name: str = "default") -> Dict[str, Any]:
        """Transform features using fitted scaler."""
        try:
            if feature_name not in self.scalers:
                logger.warning(f"No scaler found for {feature_name}. Returning unscaled features.")
                return features
            
            scaler = self.scalers[feature_name]
            feature_values = list(features.values())
            feature_array = np.array(feature_values).reshape(-1, 1)
            
            scaled_array = scaler.transform(feature_array)
            scaled_features = dict(zip(features.keys(), scaled_array.flatten()))
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Failed to transform features: {e}")
            return features


class FeatureSelector:
    """Selects relevant features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.selectors: Dict[str, Any] = {}
    
    def select_features(self, features: Dict[str, Any], target: Optional[List[float]] = None) -> Dict[str, Any]:
        """Select relevant features."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available. Returning all features.")
                return features
            
            feature_names = list(features.keys())
            feature_values = np.array(list(features.values())).reshape(1, -1)
            
            if self.config.feature_selection_method == "mutual_info" and target:
                # Mutual information-based selection
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.n_features_to_select)
                selected_indices = selector.fit(feature_values, target).get_support(indices=True)
            else:
                # Variance-based selection
                selector = SelectKBest(k=self.config.n_features_to_select)
                selected_indices = selector.fit(feature_values).get_support(indices=True)
            
            # Select features
            selected_features = {
                feature_names[i]: features[feature_names[i]]
                for i in selected_indices
            }
            
            logger.info(f"Selected {len(selected_features)} features from {len(features)}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Failed to select features: {e}")
            return features


class BlockchainFeaturePipeline:
    """Complete feature engineering pipeline for blockchain data."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.scaler = FeatureScaler(config)
        self.selector = FeatureSelector(config)
        self.feature_history: deque = deque(maxlen=1000)
    
    async def process_blockchain_data(
        self,
        transactions: List[Dict[str, Any]] = None,
        blocks: List[Dict[str, Any]] = None,
        network_data: Dict[str, Any] = None,
        target: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Process blockchain data through the complete pipeline."""
        try:
            logger.info("Starting blockchain feature processing pipeline")
            
            # Extract features
            all_features = {}
            
            if transactions:
                tx_features = self.extractor.extract_transaction_features(transactions)
                all_features.update(tx_features)
            
            if blocks:
                block_features = self.extractor.extract_block_features(blocks)
                all_features.update(block_features)
            
            if network_data:
                network_features = self.extractor.extract_network_features(network_data)
                all_features.update(network_features)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(all_features)
            
            # Select features
            selected_features = self.selector.select_features(scaled_features, target)
            
            # Apply dimensionality reduction if enabled
            if self.config.enable_pca and len(selected_features) > self.config.pca_components:
                selected_features = self._apply_pca(selected_features)
            
            # Store in history
            self.feature_history.append({
                "timestamp": time.time(),
                "features": selected_features,
                "feature_count": len(selected_features)
            })
            
            logger.info(f"Pipeline completed. Generated {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            logger.error(f"Feature processing pipeline failed: {e}")
            raise ClientError(f"Feature processing failed: {e}")
    
    def _apply_pca(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PCA dimensionality reduction."""
        try:
            if not SKLEARN_AVAILABLE:
                return features
            
            feature_values = np.array(list(features.values())).reshape(1, -1)
            pca = PCA(n_components=self.config.pca_components)
            reduced_features = pca.fit_transform(feature_values)
            
            # Create new feature names
            reduced_features_dict = {
                f"pca_component_{i}": reduced_features[0][i]
                for i in range(self.config.pca_components)
            }
            
            logger.info(f"Applied PCA. Reduced to {len(reduced_features_dict)} components")
            return reduced_features_dict
            
        except Exception as e:
            logger.error(f"PCA application failed: {e}")
            return features
    
    def get_feature_history(self) -> List[Dict[str, Any]]:
        """Get feature processing history."""
        return list(self.feature_history)
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature processing statistics."""
        if not self.feature_history:
            return {}
        
        feature_counts = [entry["feature_count"] for entry in self.feature_history]
        
        return {
            "total_processing_runs": len(self.feature_history),
            "avg_features_per_run": np.mean(feature_counts),
            "min_features_per_run": np.min(feature_counts),
            "max_features_per_run": np.max(feature_counts),
            "last_processing_time": self.feature_history[-1]["timestamp"]
        }


__all__ = [
    "FeatureConfig",
    "FeatureExtractor",
    "FeatureScaler",
    "FeatureSelector",
    "BlockchainFeaturePipeline",
]
