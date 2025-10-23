"""
GodChain Advanced P2P Networking System.

This package provides sophisticated peer-to-peer networking capabilities including
gossip protocols, peer discovery, connection management, and network optimization.
"""

import logging

logger = logging.getLogger(__name__)
from .connection_manager import (
    ConnectionConfig,
    ConnectionManager,
    ConnectionPool,
    ConnectionHealthMonitor,
    ConnectionLoadBalancer,
    Connection,
    ConnectionMetrics,
    ConnectionState,
    ConnectionType,
    ConnectionPriority,
    ConnectionStrategy,
)
from .discovery import DiscoveryConfig, DiscoveryMethod, PeerDiscovery
from .fault_tolerance import FaultTolerance, FaultToleranceConfig
from .gossip import GossipConfig, GossipMessage, GossipProtocol, MessageType
from .message_router import MessageRouter, RouteInfo, RoutingStrategy
from .network_topology import (
    NetworkTopologyManager,
    TopologyDiscovery,
    TopologyOptimizer,
    NetworkTopology,
    NetworkNode,
    NetworkLink,
    TopologyMetrics,
    TopologyType,
    NodeRole,
    ConnectionQuality,
)
from .peer import (
    PeerConfig,
    PeerStatus,
    PeerConnectionStatus,
    PeerNode,
    Peer,
    PeerInfo,
    PeerManager,
)
from .performance import NetworkPerformance, PerformanceConfig, PerformanceMonitor
from .security import NetworkSecurity, SecurityConfig

__all__ = [
    # Peer management
    "PeerNode",
    "Peer",
    "PeerInfo",
    "PeerManager",
    "PeerConfig",
    "PeerStatus",
    "PeerConnectionStatus",
    # Gossip protocol
    "GossipProtocol",
    "GossipMessage",
    "MessageType",
    "GossipConfig",
    # Peer discovery
    "PeerDiscovery",
    "DiscoveryMethod",
    "DiscoveryConfig",
    # Connection management
    "ConnectionManager",
    "ConnectionPool",
    "ConnectionHealthMonitor",
    "ConnectionLoadBalancer",
    "Connection",
    "ConnectionConfig",
    "ConnectionMetrics",
    "ConnectionState",
    "ConnectionType",
    "ConnectionPriority",
    "ConnectionStrategy",
    # Message routing
    "MessageRouter",
    "RoutingStrategy",
    "RouteInfo",
    # Network topology
    "NetworkTopologyManager",
    "TopologyDiscovery",
    "TopologyOptimizer",
    "NetworkTopology",
    "NetworkNode",
    "NetworkLink",
    "TopologyMetrics",
    "TopologyType",
    "NodeRole",
    "ConnectionQuality",
    # Security
    "NetworkSecurity",
    "SecurityConfig",
    # Performance
    "NetworkPerformance",
    "PerformanceConfig",
    "PerformanceMonitor",
    # Fault tolerance
    "FaultTolerance",
    "FaultToleranceConfig",
]
