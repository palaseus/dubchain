"""
GNN-based Network Topology Optimization

This module provides Graph Neural Network-based network topology optimization including:
- Graph Neural Network models for peer selection
- Network topology optimization algorithms
- Peer prioritization using GNN embeddings
- Network partition prediction
- Automatic topology rebalancing
- Network health scoring with ML
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
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Fallback classes
    class Data:
        def __init__(self, x=None, edge_index=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class Batch:
        def __init__(self, batch=None, **kwargs):
            self.batch = batch
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ...errors import BridgeError, ClientError
from ...logging import get_logger
from ..features import FeaturePipeline, FeatureConfig, NetworkFeatures, GraphFeatures

logger = get_logger(__name__)


@dataclass
class TopologyConfig:
    """Configuration for topology optimization."""
    enable_gnn_optimization: bool = True
    gnn_model_type: str = "GCN"  # GCN, GAT, SAGE, GIN
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    enable_peer_prioritization: bool = True
    enable_partition_prediction: bool = True
    enable_topology_rebalancing: bool = True
    rebalancing_threshold: float = 0.7
    health_scoring_enabled: bool = True


@dataclass
class PeerNode:
    """Peer node representation."""
    node_id: str
    address: str
    features: List[float]
    embedding: Optional[List[float]] = None
    priority_score: float = 0.0
    health_score: float = 0.0
    is_active: bool = True
    last_seen: float = field(default_factory=time.time)
    connection_count: int = 0
    latency: float = 0.0
    bandwidth: float = 0.0


@dataclass
class NetworkTopology:
    """Network topology representation."""
    nodes: Dict[str, PeerNode]
    edges: List[Tuple[str, str]]
    adjacency_matrix: Optional[np.ndarray] = None
    graph: Optional[nx.Graph] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class TopologyOptimizationResult:
    """Result of topology optimization."""
    optimized_topology: NetworkTopology
    improvement_score: float
    removed_edges: List[Tuple[str, str]]
    added_edges: List[Tuple[str, str]]
    peer_priorities: Dict[str, float]
    partition_risk: float
    health_scores: Dict[str, float]


class GNNModel(nn.Module):
    """Graph Neural Network model for topology optimization."""
    
    def __init__(self, config: TopologyConfig, input_dim: int):
        super(GNNModel, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        
        if config.gnn_model_type == "GCN":
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif config.gnn_model_type == "GAT":
            self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=8, dropout=dropout))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout))
        elif config.gnn_model_type == "SAGE":
            self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        elif config.gnn_model_type == "GIN":
            self.gnn_layers.append(GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )))
        
        # Output layers
        self.priority_head = nn.Linear(hidden_dim, 1)
        self.health_head = nn.Linear(hidden_dim, 1)
        self.partition_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GNN."""
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output heads
        priority_scores = torch.sigmoid(self.priority_head(x))
        health_scores = torch.sigmoid(self.health_head(x))
        partition_scores = torch.sigmoid(self.partition_head(x))
        
        return {
            'embeddings': x,
            'priority_scores': priority_scores,
            'health_scores': health_scores,
            'partition_scores': partition_scores
        }


class TopologyDataset(Dataset):
    """Dataset for topology optimization."""
    
    def __init__(self, topologies: List[NetworkTopology], labels: Optional[List[Dict[str, Any]]] = None):
        self.topologies = topologies
        self.labels = labels or []
    
    def __len__(self):
        return len(self.topologies)
    
    def __getitem__(self, idx):
        topology = self.topologies[idx]
        
        # Convert to PyTorch Geometric format
        if TORCH_GEOMETRIC_AVAILABLE:
            data = self._topology_to_pyg_data(topology)
        else:
            data = None
        
        if self.labels and idx < len(self.labels):
            label = self.labels[idx]
        else:
            label = {}
        
        return data, label
    
    def _topology_to_pyg_data(self, topology: NetworkTopology) -> Data:
        """Convert topology to PyTorch Geometric data."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            return None
        
        # Node features
        node_features = []
        node_ids = list(topology.nodes.keys())
        
        for node_id in node_ids:
            node = topology.nodes[node_id]
            features = node.features + [node.priority_score, node.health_score, node.latency, node.bandwidth]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices
        edge_indices = []
        for edge in topology.edges:
            src_idx = node_ids.index(edge[0])
            dst_idx = node_ids.index(edge[1])
            edge_indices.append([src_idx, dst_idx])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)


class TopologyOptimizer:
    """Main topology optimizer using GNN."""
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.model: Optional[GNNModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.training_history = []
        
    def initialize_model(self, input_dim: int) -> None:
        """Initialize the GNN model."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available, using simplified model")
            return
        
        self.model = GNNModel(self.config, input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        logger.info(f"GNN model initialized with {input_dim} input dimensions")
    
    def train_model(self, dataset: TopologyDataset) -> None:
        """Train the GNN model."""
        if not self.model or not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("Model not initialized or PyTorch Geometric not available")
            return
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0.0
            
            for batch_data, batch_labels in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
                
                # Calculate loss
                loss = self._calculate_loss(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.training_history.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, Any]) -> torch.Tensor:
        """Calculate loss for training."""
        loss = 0.0
        
        # Priority score loss
        if 'priority_scores' in labels:
            priority_loss = F.mse_loss(outputs['priority_scores'], labels['priority_scores'])
            loss += priority_loss
        
        # Health score loss
        if 'health_scores' in labels:
            health_loss = F.mse_loss(outputs['health_scores'], labels['health_scores'])
            loss += health_loss
        
        # Partition score loss
        if 'partition_scores' in labels:
            partition_loss = F.mse_loss(outputs['partition_scores'], labels['partition_scores'])
            loss += partition_loss
        
        return loss
    
    def optimize_topology(self, topology: NetworkTopology) -> TopologyOptimizationResult:
        """Optimize network topology using GNN."""
        if not self.model or not TORCH_GEOMETRIC_AVAILABLE:
            return self._fallback_optimization(topology)
        
        self.model.eval()
        
        with torch.no_grad():
            # Convert topology to PyTorch Geometric format
            data = self._topology_to_pyg_data(topology)
            
            # Get model predictions
            outputs = self.model(data.x, data.edge_index)
            
            # Extract predictions
            priority_scores = outputs['priority_scores'].squeeze().numpy()
            health_scores = outputs['health_scores'].squeeze().numpy()
            partition_scores = outputs['partition_scores'].squeeze().numpy()
            
            # Create optimized topology
            optimized_topology = self._create_optimized_topology(
                topology, priority_scores, health_scores, partition_scores
            )
            
            # Calculate improvement
            improvement_score = self._calculate_improvement(topology, optimized_topology)
            
            # Identify changes
            removed_edges, added_edges = self._identify_changes(topology, optimized_topology)
            
            # Calculate peer priorities
            peer_priorities = dict(zip(list(topology.nodes.keys()), priority_scores))
            
            # Calculate partition risk
            partition_risk = np.mean(partition_scores)
            
            # Calculate health scores
            health_scores_dict = dict(zip(list(topology.nodes.keys()), health_scores))
            
            return TopologyOptimizationResult(
                optimized_topology=optimized_topology,
                improvement_score=improvement_score,
                removed_edges=removed_edges,
                added_edges=added_edges,
                peer_priorities=peer_priorities,
                partition_risk=partition_risk,
                health_scores=health_scores_dict
            )
    
    def _fallback_optimization(self, topology: NetworkTopology) -> TopologyOptimizationResult:
        """Fallback optimization without GNN."""
        # Simple heuristic-based optimization
        optimized_topology = NetworkTopology(
            nodes=topology.nodes.copy(),
            edges=topology.edges.copy()
        )
        
        # Remove low-priority edges
        removed_edges = []
        for edge in topology.edges:
            if self._should_remove_edge(edge, topology):
                removed_edges.append(edge)
                optimized_topology.edges.remove(edge)
        
        # Add high-priority edges
        added_edges = []
        for node_id in topology.nodes:
            if self._should_add_edge(node_id, optimized_topology):
                # Find best connection
                best_connection = self._find_best_connection(node_id, optimized_topology)
                if best_connection:
                    added_edges.append((node_id, best_connection))
                    optimized_topology.edges.append((node_id, best_connection))
        
        # Calculate peer priorities (simplified)
        peer_priorities = {}
        for node_id, node in topology.nodes.items():
            peer_priorities[node_id] = node.connection_count / max(len(topology.nodes), 1)
        
        return TopologyOptimizationResult(
            optimized_topology=optimized_topology,
            improvement_score=0.5,  # Placeholder
            removed_edges=removed_edges,
            added_edges=added_edges,
            peer_priorities=peer_priorities,
            partition_risk=0.3,  # Placeholder
            health_scores={node_id: 0.8 for node_id in topology.nodes.keys()}
        )
    
    def _topology_to_pyg_data(self, topology: NetworkTopology) -> Data:
        """Convert topology to PyTorch Geometric data."""
        # Node features
        node_features = []
        node_ids = list(topology.nodes.keys())
        
        for node_id in node_ids:
            node = topology.nodes[node_id]
            features = node.features + [node.priority_score, node.health_score, node.latency, node.bandwidth]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices
        edge_indices = []
        for edge in topology.edges:
            src_idx = node_ids.index(edge[0])
            dst_idx = node_ids.index(edge[1])
            edge_indices.append([src_idx, dst_idx])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def _create_optimized_topology(self, original_topology: NetworkTopology,
                                 priority_scores: np.ndarray, health_scores: np.ndarray,
                                 partition_scores: np.ndarray) -> NetworkTopology:
        """Create optimized topology based on GNN predictions."""
        optimized_topology = NetworkTopology(
            nodes=original_topology.nodes.copy(),
            edges=[]
        )
        
        # Update node scores
        node_ids = list(original_topology.nodes.keys())
        for i, node_id in enumerate(node_ids):
            optimized_topology.nodes[node_id].priority_score = priority_scores[i]
            optimized_topology.nodes[node_id].health_score = health_scores[i]
        
        # Rebuild edges based on predictions
        for i, src_node in enumerate(node_ids):
            for j, dst_node in enumerate(node_ids):
                if i != j:
                    # Calculate edge score based on node scores
                    edge_score = (priority_scores[i] + priority_scores[j]) / 2
                    
                    # Add edge if score is above threshold
                    if edge_score > self.config.rebalancing_threshold:
                        optimized_topology.edges.append((src_node, dst_node))
        
        return optimized_topology
    
    def _calculate_improvement(self, original: NetworkTopology, optimized: NetworkTopology) -> float:
        """Calculate improvement score."""
        # Simple improvement metric based on connectivity
        original_connectivity = len(original.edges) / max(len(original.nodes) * (len(original.nodes) - 1), 1)
        optimized_connectivity = len(optimized.edges) / max(len(optimized.nodes) * (len(optimized.nodes) - 1), 1)
        
        return optimized_connectivity - original_connectivity
    
    def _identify_changes(self, original: NetworkTopology, optimized: NetworkTopology) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Identify changes between original and optimized topologies."""
        original_edges = set(original.edges)
        optimized_edges = set(optimized.edges)
        
        removed_edges = list(original_edges - optimized_edges)
        added_edges = list(optimized_edges - original_edges)
        
        return removed_edges, added_edges
    
    def _should_remove_edge(self, edge: Tuple[str, str], topology: NetworkTopology) -> bool:
        """Determine if an edge should be removed."""
        # Simple heuristic: remove edges between low-priority nodes
        src_node = topology.nodes[edge[0]]
        dst_node = topology.nodes[edge[1]]
        
        return (src_node.priority_score < 0.3 and dst_node.priority_score < 0.3)
    
    def _should_add_edge(self, node_id: str, topology: NetworkTopology) -> bool:
        """Determine if an edge should be added."""
        # Simple heuristic: add edges for high-priority nodes with few connections
        node = topology.nodes[node_id]
        return node.priority_score > 0.7 and node.connection_count < 3
    
    def _find_best_connection(self, node_id: str, topology: NetworkTopology) -> Optional[str]:
        """Find the best node to connect to."""
        current_node = topology.nodes[node_id]
        best_connection = None
        best_score = 0.0
        
        for other_id, other_node in topology.nodes.items():
            if other_id != node_id and (node_id, other_id) not in topology.edges:
                # Calculate connection score
                score = (current_node.priority_score + other_node.priority_score) / 2
                
                if score > best_score:
                    best_score = score
                    best_connection = other_id
        
        return best_connection
    
    def predict_partition_risk(self, topology: NetworkTopology) -> float:
        """Predict network partition risk."""
        if not NETWORKX_AVAILABLE:
            return 0.5  # Placeholder
        
        try:
            # Build networkx graph
            graph = nx.Graph()
            graph.add_nodes_from(topology.nodes.keys())
            graph.add_edges_from(topology.edges)
            
            # Calculate connectivity metrics
            if len(graph.nodes) < 2:
                return 0.0
            
            # Check if graph is connected
            if not nx.is_connected(graph):
                return 1.0  # Already partitioned
            
            # Calculate minimum cut
            min_cut = nx.minimum_edge_cut(graph)
            
            # Calculate partition risk based on minimum cut
            partition_risk = len(min_cut) / max(len(graph.edges), 1)
            
            return min(partition_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to predict partition risk: {e}")
            return 0.5
    
    def calculate_network_health(self, topology: NetworkTopology) -> Dict[str, float]:
        """Calculate network health scores."""
        health_scores = {}
        
        for node_id, node in topology.nodes.items():
            # Calculate health score based on multiple factors
            health_score = 0.0
            
            # Connection health
            connection_health = min(node.connection_count / 5, 1.0)
            health_score += connection_health * 0.3
            
            # Latency health
            latency_health = max(0, 1.0 - node.latency / 1000)  # Assume 1000ms is max
            health_score += latency_health * 0.2
            
            # Bandwidth health
            bandwidth_health = min(node.bandwidth / 1000, 1.0)  # Assume 1000 Mbps is max
            health_score += bandwidth_health * 0.2
            
            # Activity health
            time_since_seen = time.time() - node.last_seen
            activity_health = max(0, 1.0 - time_since_seen / 3600)  # 1 hour timeout
            health_score += activity_health * 0.3
            
            health_scores[node_id] = min(health_score, 1.0)
        
        return health_scores
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "model_initialized": self.model is not None,
            "training_epochs": len(self.training_history),
            "last_training_loss": self.training_history[-1] if self.training_history else 0.0,
            "gnn_model_type": self.config.gnn_model_type,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "learning_rate": self.config.learning_rate,
            "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
            "networkx_available": NETWORKX_AVAILABLE
        }
