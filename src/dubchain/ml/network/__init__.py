"""
ML Network Topology Optimization Module

This module provides Graph Neural Network-based network topology optimization including:
- Graph Neural Network models for peer selection
- Network topology optimization algorithms
- Peer prioritization using GNN embeddings
- Network partition prediction
- Automatic topology rebalancing
- Network health scoring with ML
"""

from .topology import (
    TopologyOptimizer,
    TopologyConfig,
    GNNModel,
    TopologyDataset,
    PeerNode,
    NetworkTopology,
    TopologyOptimizationResult,
)

__all__ = [
    "TopologyOptimizer",
    "TopologyConfig",
    "GNNModel",
    "TopologyDataset",
    "PeerNode",
    "NetworkTopology",
    "TopologyOptimizationResult",
]