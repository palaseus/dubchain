"""
GodChain Advanced P2P Networking System.

This package provides sophisticated peer-to-peer networking capabilities including
gossip protocols, peer discovery, connection management, and network optimization.
"""

from .connection_manager import (
    ConnectionConfig,
    ConnectionManager,
    ConnectionPool,
    ConnectionStrategy,
)
from .discovery import DiscoveryConfig, DiscoveryMethod, PeerDiscovery
from .fault_tolerance import FaultTolerance, FaultToleranceConfig
from .gossip import GossipConfig, GossipMessage, GossipProtocol, MessageType
from .message_router import MessageRouter, RouteInfo, RoutingStrategy
from .network_topology import NetworkTopology, TopologyConfig, TopologyManager
from .peer import ConnectionType, Peer, PeerInfo, PeerStatus
from .performance import NetworkPerformance, PerformanceConfig, PerformanceMonitor
from .security import NetworkSecurity, SecurityConfig

__all__ = [
    # Peer management
    "Peer",
    "PeerInfo",
    "PeerStatus",
    "ConnectionType",
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
    "ConnectionConfig",
    "ConnectionStrategy",
    # Message routing
    "MessageRouter",
    "RoutingStrategy",
    "RouteInfo",
    # Network topology
    "NetworkTopology",
    "TopologyManager",
    "TopologyConfig",
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
