"""
Fault tolerance mechanisms for GodChain P2P network.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .peer import Peer, PeerInfo


class FaultType(Enum):
    """Types of network faults."""

    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    MESSAGE_LOSS = "message_loss"
    CONNECTION_TIMEOUT = "connection_timeout"


@dataclass
class FaultEvent:
    """Fault event data."""

    fault_type: FaultType
    peer_id: str
    timestamp: float
    severity: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance."""

    max_faulty_peers: int = 3
    heartbeat_interval: float = 30.0
    timeout_threshold: float = 60.0
    enable_byzantine_detection: bool = True
    enable_auto_recovery: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ByzantineDetector:
    """Byzantine fault detection system."""

    def __init__(self, config: FaultToleranceConfig):
        """Initialize Byzantine detector."""
        self.config = config
        self.suspicious_peers: Dict[str, List[FaultEvent]] = {}
        self.byzantine_peers: Set[str] = set()

    def detect_byzantine_behavior(
        self, peer_id: str, behavior_data: Dict[str, Any]
    ) -> bool:
        """Detect Byzantine behavior."""
        # Simple detection logic for demo
        if behavior_data.get("inconsistent_messages", 0) > 5:
            return True

        if behavior_data.get("malicious_actions", 0) > 0:
            return True

        return False

    def is_byzantine_peer(self, peer_id: str) -> bool:
        """Check if peer is Byzantine."""
        return peer_id in self.byzantine_peers


class NetworkPartitionDetector:
    """Network partition detection system."""

    def __init__(self, config: FaultToleranceConfig):
        """Initialize partition detector."""
        self.config = config
        self.partitions: List[Set[str]] = []
        self.last_connectivity_check: Dict[str, float] = {}

    def detect_partition(self, peer_connections: Dict[str, Set[str]]) -> List[Set[str]]:
        """Detect network partitions."""
        visited = set()
        partitions = []

        for peer_id in peer_connections:
            if peer_id not in visited:
                partition = self._dfs_partition(peer_id, peer_connections, visited)
                if partition:
                    partitions.append(partition)

        self.partitions = partitions
        return partitions

    def _dfs_partition(
        self, peer_id: str, connections: Dict[str, Set[str]], visited: Set[str]
    ) -> Set[str]:
        """DFS to find connected components."""
        if peer_id in visited:
            return set()

        visited.add(peer_id)
        partition = {peer_id}

        for connected_peer in connections.get(peer_id, set()):
            partition.update(self._dfs_partition(connected_peer, connections, visited))

        return partition


class AutoRecovery:
    """Automatic recovery system."""

    def __init__(self, config: FaultToleranceConfig):
        """Initialize auto recovery."""
        self.config = config
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3

    def attempt_recovery(self, peer_id: str, fault_type: FaultType) -> bool:
        """Attempt to recover from fault."""
        if peer_id not in self.recovery_attempts:
            self.recovery_attempts[peer_id] = 0

        if self.recovery_attempts[peer_id] >= self.max_recovery_attempts:
            return False

        self.recovery_attempts[peer_id] += 1

        # Simple recovery logic for demo
        if fault_type == FaultType.CONNECTION_TIMEOUT:
            return True  # Assume reconnection works

        return False

    def reset_recovery_attempts(self, peer_id: str) -> None:
        """Reset recovery attempts for peer."""
        if peer_id in self.recovery_attempts:
            del self.recovery_attempts[peer_id]


class FaultTolerance:
    """Fault tolerance manager."""

    def __init__(self, config: FaultToleranceConfig):
        """Initialize fault tolerance."""
        self.config = config
        self.byzantine_detector = ByzantineDetector(config)
        self.partition_detector = NetworkPartitionDetector(config)
        self.auto_recovery = AutoRecovery(config)
        self.fault_history: List[FaultEvent] = []

    def handle_fault(self, fault_event: FaultEvent) -> bool:
        """Handle fault event."""
        self.fault_history.append(fault_event)

        if fault_event.fault_type == FaultType.BYZANTINE_BEHAVIOR:
            self.byzantine_detector.byzantine_peers.add(fault_event.peer_id)
            return False

        if self.config.enable_auto_recovery:
            return self.auto_recovery.attempt_recovery(
                fault_event.peer_id, fault_event.fault_type
            )

        return False

    def detect_network_partitions(
        self, peer_connections: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """Detect network partitions."""
        return self.partition_detector.detect_partition(peer_connections)

    def is_peer_faulty(self, peer_id: str) -> bool:
        """Check if peer is faulty."""
        return self.byzantine_detector.is_byzantine_peer(peer_id)
