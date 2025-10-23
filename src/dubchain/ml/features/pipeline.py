"""
ML Feature Engineering Pipeline for Blockchain Data

This module provides comprehensive feature engineering for blockchain data including:
- Transaction feature extraction
- Network topology features
- Time-series feature engineering
- Graph-based feature extraction
- Feature scaling and normalization
- Feature selection and dimensionality reduction
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

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ...errors import BridgeError, ClientError
from ...logging import get_logger
from ..universal import UniversalTransaction, ChainType, TokenType

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    enable_transaction_features: bool = True
    enable_network_features: bool = True
    enable_temporal_features: bool = True
    enable_graph_features: bool = True
    enable_statistical_features: bool = True
    feature_scaling: str = "standard"  # standard, minmax, robust
    feature_selection: bool = True
    n_features_select: int = 100
    enable_dimensionality_reduction: bool = True
    n_components_pca: int = 50
    enable_feature_caching: bool = True
    cache_duration: float = 3600.0  # seconds


@dataclass
class TransactionFeatures:
    """Transaction-level features."""
    tx_id: str
    amount: float
    fee: float
    gas_price: Optional[float] = None
    gas_limit: Optional[int] = None
    nonce: Optional[int] = None
    block_height: Optional[int] = None
    confirmations: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # Derived features
    amount_log: float = 0.0
    fee_ratio: float = 0.0
    gas_efficiency: float = 0.0
    urgency_score: float = 0.0


@dataclass
class NetworkFeatures:
    """Network-level features."""
    node_id: str
    degree: int = 0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank: float = 0.0
    local_clustering: float = 0.0
    global_clustering: float = 0.0


@dataclass
class TemporalFeatures:
    """Time-series features."""
    timestamp: float
    hour_of_day: int = 0
    day_of_week: int = 0
    month: int = 0
    is_weekend: bool = False
    is_holiday: bool = False
    
    # Rolling statistics
    rolling_mean_1h: float = 0.0
    rolling_std_1h: float = 0.0
    rolling_mean_24h: float = 0.0
    rolling_std_24h: float = 0.0
    
    # Trend features
    trend_slope: float = 0.0
    trend_intercept: float = 0.0
    trend_r2: float = 0.0


@dataclass
class GraphFeatures:
    """Graph-based features."""
    node_id: str
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0
    weighted_degree: float = 0.0
    triangle_count: int = 0
    square_count: int = 0
    
    # Community features
    community_id: Optional[int] = None
    community_size: int = 0
    modularity: float = 0.0
    
    # Path features
    avg_path_length: float = 0.0
    diameter: int = 0
    radius: int = 0


@dataclass
class StatisticalFeatures:
    """Statistical features."""
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    var: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    min: float = 0.0
    max: float = 0.0
    q25: float = 0.0
    q75: float = 0.0
    iqr: float = 0.0


class TransactionFeatureExtractor:
    """Extracts features from transactions."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_features(self, transaction: UniversalTransaction) -> TransactionFeatures:
        """Extract transaction features."""
        features = TransactionFeatures(
            tx_id=transaction.tx_hash,
            amount=float(transaction.value),
            fee=0.0,  # UniversalTransaction doesn't have fee field
            gas_price=float(transaction.gas_price) if transaction.gas_price else None,
            gas_limit=transaction.gas_used,  # Use gas_used as gas_limit
            nonce=0,  # UniversalTransaction doesn't have nonce
            block_height=transaction.block_number,
            confirmations=1,  # Default confirmations
            timestamp=transaction.timestamp
        )
        
        # Calculate derived features
        features.amount_log = np.log10(max(features.amount, 1))
        features.fee_ratio = features.fee / max(features.amount, 1)
        
        if features.gas_price and features.gas_limit:
            features.gas_efficiency = features.gas_price / features.gas_limit
        
        # Urgency score based on gas price and confirmations
        if features.gas_price:
            features.urgency_score = features.gas_price / (features.confirmations + 1)
        
        return features


class NetworkFeatureExtractor:
    """Extracts network topology features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.graph_cache: Dict[str, nx.Graph] = {}
    
    def extract_features(self, node_id: str, transactions: List[UniversalTransaction]) -> NetworkFeatures:
        """Extract network features for a node."""
        if not NETWORKX_AVAILABLE:
            return NetworkFeatures(node_id=node_id)
        
        # Build transaction graph
        graph = self._build_transaction_graph(transactions)
        
        if node_id not in graph.nodes:
            return NetworkFeatures(node_id=node_id)
        
        features = NetworkFeatures(node_id=node_id)
        
        try:
            # Basic centrality measures
            features.degree = graph.degree(node_id)
            features.betweenness_centrality = nx.betweenness_centrality(graph)[node_id]
            features.closeness_centrality = nx.closeness_centrality(graph)[node_id]
            features.eigenvector_centrality = nx.eigenvector_centrality(graph)[node_id]
            features.clustering_coefficient = nx.clustering(graph, node_id)
            features.pagerank = nx.pagerank(graph)[node_id]
            
            # Clustering features
            features.local_clustering = nx.clustering(graph, node_id)
            features.global_clustering = nx.average_clustering(graph)
            
        except Exception as e:
            logger.error(f"Failed to extract network features: {e}")
        
        return features
    
    def _build_transaction_graph(self, transactions: List[UniversalTransaction]) -> nx.Graph:
        """Build transaction graph."""
        graph = nx.Graph()
        
        for tx in transactions:
            from_addr = tx.from_address
            to_addr = tx.to_address
            
            if from_addr not in graph.nodes:
                graph.add_node(from_addr)
            if to_addr not in graph.nodes:
                graph.add_node(to_addr)
            
            if not graph.has_edge(from_addr, to_addr):
                graph.add_edge(from_addr, to_addr, weight=0)
            
            # Update edge weight
            graph[from_addr][to_addr]['weight'] += tx.value
        
        return graph


class TemporalFeatureExtractor:
    """Extracts temporal features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.holidays = self._load_holidays()
    
    def extract_features(self, timestamp: float, 
                        historical_data: List[Dict[str, Any]]) -> TemporalFeatures:
        """Extract temporal features."""
        dt = datetime.fromtimestamp(timestamp)
        
        features = TemporalFeatures(timestamp=timestamp)
        
        # Basic temporal features
        features.hour_of_day = dt.hour
        features.day_of_week = dt.weekday()
        features.month = dt.month
        features.is_weekend = dt.weekday() >= 5
        features.is_holiday = self._is_holiday(dt)
        
        # Rolling statistics
        if historical_data:
            features.rolling_mean_1h = self._calculate_rolling_mean(historical_data, 3600)
            features.rolling_std_1h = self._calculate_rolling_std(historical_data, 3600)
            features.rolling_mean_24h = self._calculate_rolling_mean(historical_data, 86400)
            features.rolling_std_24h = self._calculate_rolling_std(historical_data, 86400)
            
            # Trend analysis
            trend_features = self._calculate_trend(historical_data)
            features.trend_slope = trend_features['slope']
            features.trend_intercept = trend_features['intercept']
            features.trend_r2 = trend_features['r2']
        
        return features
    
    def _load_holidays(self) -> List[Tuple[int, int]]:
        """Load holiday dates."""
        # Simplified holiday list
        return [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Check if date is a holiday."""
        return (dt.month, dt.day) in self.holidays
    
    def _calculate_rolling_mean(self, data: List[Dict[str, Any]], window: int) -> float:
        """Calculate rolling mean."""
        if not data:
            return 0.0
        
        current_time = time.time()
        window_data = [d for d in data if current_time - d['timestamp'] <= window]
        
        if not window_data:
            return 0.0
        
        return np.mean([d['value'] for d in window_data])
    
    def _calculate_rolling_std(self, data: List[Dict[str, Any]], window: int) -> float:
        """Calculate rolling standard deviation."""
        if not data:
            return 0.0
        
        current_time = time.time()
        window_data = [d for d in data if current_time - d['timestamp'] <= window]
        
        if not window_data:
            return 0.0
        
        return np.std([d['value'] for d in window_data])
    
    def _calculate_trend(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend features."""
        if len(data) < 2:
            return {'slope': 0.0, 'intercept': 0.0, 'r2': 0.0}
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        x = np.array([d['timestamp'] for d in sorted_data])
        y = np.array([d['value'] for d in sorted_data])
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Calculate R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {'slope': slope, 'intercept': intercept, 'r2': r2}


class GraphFeatureExtractor:
    """Extracts graph-based features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_features(self, node_id: str, graph: nx.Graph) -> GraphFeatures:
        """Extract graph features."""
        if not NETWORKX_AVAILABLE or node_id not in graph.nodes:
            return GraphFeatures(node_id=node_id)
        
        features = GraphFeatures(node_id=node_id)
        
        try:
            # Degree features
            features.in_degree = graph.in_degree(node_id) if graph.is_directed() else graph.degree(node_id)
            features.out_degree = graph.out_degree(node_id) if graph.is_directed() else graph.degree(node_id)
            features.total_degree = features.in_degree + features.out_degree
            
            # Weighted degree
            if graph.is_directed():
                features.weighted_degree = sum(graph[node_id][neighbor]['weight'] 
                                             for neighbor in graph.neighbors(node_id))
            else:
                features.weighted_degree = sum(graph[node_id][neighbor]['weight'] 
                                             for neighbor in graph.neighbors(node_id))
            
            # Triangle and square counts
            features.triangle_count = len(list(nx.enumerate_all_cliques(graph.subgraph([node_id] + list(graph.neighbors(node_id))))))
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(graph)
                for i, community in enumerate(communities):
                    if node_id in community:
                        features.community_id = i
                        features.community_size = len(community)
                        break
                
                features.modularity = nx.community.modularity(graph, communities)
            except:
                pass
            
            # Path features
            try:
                features.avg_path_length = nx.average_shortest_path_length(graph)
                features.diameter = nx.diameter(graph)
                features.radius = nx.radius(graph)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Failed to extract graph features: {e}")
        
        return features


class StatisticalFeatureExtractor:
    """Extracts statistical features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_features(self, data: List[float]) -> StatisticalFeatures:
        """Extract statistical features."""
        if not data:
            return StatisticalFeatures()
        
        features = StatisticalFeatures()
        
        try:
            features.mean = np.mean(data)
            features.median = np.median(data)
            features.std = np.std(data)
            features.var = np.var(data)
            features.skewness = self._calculate_skewness(data)
            features.kurtosis = self._calculate_kurtosis(data)
            features.min = np.min(data)
            features.max = np.max(data)
            features.q25 = np.percentile(data, 25)
            features.q75 = np.percentile(data, 75)
            features.iqr = features.q75 - features.q25
            
        except Exception as e:
            logger.error(f"Failed to extract statistical features: {e}")
        
        return features
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean([(x - mean) ** 4 for x in data]) / (std ** 4) - 3
        return kurtosis


class FeaturePipeline:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.transaction_extractor = TransactionFeatureExtractor(config)
        self.network_extractor = NetworkFeatureExtractor(config)
        self.temporal_extractor = TemporalFeatureExtractor(config)
        self.graph_extractor = GraphFeatureExtractor(config)
        self.statistical_extractor = StatisticalFeatureExtractor(config)
        
        # Feature scaling
        self.scalers = {}
        self.feature_cache = {}
        
        if SKLEARN_AVAILABLE:
            if config.feature_scaling == "standard":
                self.scaler = StandardScaler()
            elif config.feature_scaling == "minmax":
                self.scaler = MinMaxScaler()
            elif config.feature_scaling == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
    
    def extract_all_features(self, transactions: List[UniversalTransaction],
                           historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Extract all features from transactions."""
        features = {}
        
        try:
            # Transaction features
            if self.config.enable_transaction_features:
                tx_features = []
                for tx in transactions:
                    tx_feat = self.transaction_extractor.extract_features(tx)
                    tx_features.append(tx_feat)
                features['transaction'] = tx_features
            
            # Network features
            if self.config.enable_network_features:
                network_features = []
                for tx in transactions:
                    net_feat = self.network_extractor.extract_features(
                        tx.from_address, transactions
                    )
                    network_features.append(net_feat)
                features['network'] = network_features
            
            # Temporal features
            if self.config.enable_temporal_features:
                temporal_features = []
                for tx in transactions:
                    temp_feat = self.temporal_extractor.extract_features(
                        tx.created_at, historical_data or []
                    )
                    temporal_features.append(temp_feat)
                features['temporal'] = temporal_features
            
            # Graph features
            if self.config.enable_graph_features and NETWORKX_AVAILABLE:
                graph = self.network_extractor._build_transaction_graph(transactions)
                graph_features = []
                for tx in transactions:
                    graph_feat = self.graph_extractor.extract_features(
                        tx.from_address.address, graph
                    )
                    graph_features.append(graph_feat)
                features['graph'] = graph_features
            
            # Statistical features
            if self.config.enable_statistical_features:
                amounts = [tx.amount for tx in transactions]
                stat_feat = self.statistical_extractor.extract_features(amounts)
                features['statistical'] = stat_feat
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            raise BridgeError(f"Feature extraction failed: {e}")
        
        return features
    
    def scale_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Scale features using configured scaler."""
        if not SKLEARN_AVAILABLE:
            return features
        
        try:
            # Convert features to numpy array
            feature_matrix = self._features_to_matrix(features)
            
            # Scale features
            scaled_matrix = self.scaler.fit_transform(feature_matrix)
            
            # Convert back to feature format
            scaled_features = self._matrix_to_features(scaled_matrix, features)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            return features
    
    def select_features(self, features: Dict[str, Any], labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Select best features using statistical tests."""
        if not self.config.feature_selection or not SKLEARN_AVAILABLE:
            return features
        
        try:
            # Convert features to matrix
            feature_matrix = self._features_to_matrix(features)
            
            if labels is None:
                # Use variance-based selection
                selector = SelectKBest(k=self.config.n_features_select)
                selected_matrix = selector.fit_transform(feature_matrix, np.zeros(feature_matrix.shape[0]))
            else:
                # Use supervised selection
                selector = SelectKBest(score_func=f_classif, k=self.config.n_features_select)
                selected_matrix = selector.fit_transform(feature_matrix, labels)
            
            # Convert back to feature format
            selected_features = self._matrix_to_features(selected_matrix, features)
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Failed to select features: {e}")
            return features
    
    def reduce_dimensions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce feature dimensions using PCA."""
        if not self.config.enable_dimensionality_reduction or not SKLEARN_AVAILABLE:
            return features
        
        try:
            # Convert features to matrix
            feature_matrix = self._features_to_matrix(features)
            
            # Apply PCA
            pca = PCA(n_components=self.config.n_components_pca)
            reduced_matrix = pca.fit_transform(feature_matrix)
            
            # Convert back to feature format
            reduced_features = self._matrix_to_features(reduced_matrix, features)
            
            return reduced_features
            
        except Exception as e:
            logger.error(f"Failed to reduce dimensions: {e}")
            return features
    
    def _features_to_matrix(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to numpy matrix."""
        # This is a simplified implementation
        # Real implementation would handle different feature types properly
        
        matrix_data = []
        
        if 'transaction' in features:
            for tx_feat in features['transaction']:
                row = [
                    tx_feat.amount,
                    tx_feat.fee,
                    tx_feat.amount_log,
                    tx_feat.fee_ratio,
                    tx_feat.gas_efficiency,
                    tx_feat.urgency_score
                ]
                matrix_data.append(row)
        
        if 'network' in features:
            for net_feat in features['network']:
                row = [
                    net_feat.degree,
                    net_feat.betweenness_centrality,
                    net_feat.closeness_centrality,
                    net_feat.eigenvector_centrality,
                    net_feat.clustering_coefficient,
                    net_feat.pagerank
                ]
                matrix_data.append(row)
        
        return np.array(matrix_data)
    
    def _matrix_to_features(self, matrix: np.ndarray, original_features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert matrix back to feature format."""
        # This is a simplified implementation
        # Real implementation would reconstruct the original feature structure
        
        return {
            'matrix': matrix,
            'original_features': original_features
        }
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            'amount', 'fee', 'amount_log', 'fee_ratio', 'gas_efficiency', 'urgency_score',
            'degree', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality',
            'clustering_coefficient', 'pagerank'
        ]
