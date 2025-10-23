"""
Shard network management for DubChain.

This module provides network management for shards including:
- Shard topology management
- Shard discovery
- Shard routing
"""

import logging

logger = logging.getLogger(__name__)
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .shard_types import ShardId, ShardState


@dataclass
class ShardTopology:
    """Topology of shard network."""

    shard_connections: Dict[ShardId, List[ShardId]] = field(default_factory=dict)
    connection_weights: Dict[Tuple[ShardId, ShardId], float] = field(
        default_factory=dict
    )

    def add_connection(
        self, shard1: ShardId, shard2: ShardId, weight: float = 1.0
    ) -> None:
        """Add connection between shards."""
        if shard1 not in self.shard_connections:
            self.shard_connections[shard1] = []
        if shard2 not in self.shard_connections:
            self.shard_connections[shard2] = []

        if shard2 not in self.shard_connections[shard1]:
            self.shard_connections[shard1].append(shard2)
        if shard1 not in self.shard_connections[shard2]:
            self.shard_connections[shard2].append(shard1)

        self.connection_weights[(shard1, shard2)] = weight
        self.connection_weights[(shard2, shard1)] = weight

    def remove_connection(self, shard1: ShardId, shard2: ShardId) -> None:
        """Remove connection between shards."""
        if (
            shard1 in self.shard_connections
            and shard2 in self.shard_connections[shard1]
        ):
            self.shard_connections[shard1].remove(shard2)
        if (
            shard2 in self.shard_connections
            and shard1 in self.shard_connections[shard2]
        ):
            self.shard_connections[shard2].remove(shard1)

        # Remove weights
        if (shard1, shard2) in self.connection_weights:
            del self.connection_weights[(shard1, shard2)]
        if (shard2, shard1) in self.connection_weights:
            del self.connection_weights[(shard2, shard1)]

    def get_connections(self, shard_id: ShardId) -> List[ShardId]:
        """Get connections for a shard."""
        return self.shard_connections.get(shard_id, [])

    def get_connection_weight(self, shard1: ShardId, shard2: ShardId) -> float:
        """Get connection weight between shards."""
        return self.connection_weights.get((shard1, shard2), 0.0)


@dataclass
class ShardDiscovery:
    """Discovers shards in the network."""

    known_shards: Set[ShardId] = field(default_factory=set)
    discovery_interval: float = 30.0
    last_discovery: float = field(default_factory=time.time)

    def discover_shards(self, shard_states: Dict[ShardId, ShardState]) -> Set[ShardId]:
        """Discover available shards."""
        discovered = set(shard_states.keys())
        self.known_shards.update(discovered)
        self.last_discovery = time.time()
        return discovered

    def is_shard_known(self, shard_id: ShardId) -> bool:
        """Check if shard is known."""
        return shard_id in self.known_shards


@dataclass
class ShardRouting:
    """Routes messages between shards."""

    routing_table: Dict[ShardId, Dict[ShardId, List[ShardId]]] = field(
        default_factory=dict
    )
    routing_metrics: Dict[str, int] = field(default_factory=dict)

    def add_route(self, source: ShardId, target: ShardId, path: List[ShardId]) -> None:
        """Add route between shards."""
        if source not in self.routing_table:
            self.routing_table[source] = {}
        self.routing_table[source][target] = path

    def find_route(self, source: ShardId, target: ShardId) -> List[ShardId]:
        """Find route between shards."""
        if source == target:
            return [source]

        if source in self.routing_table and target in self.routing_table[source]:
            return self.routing_table[source][target]

        return []  # No route found

    def update_metrics(self, route_used: bool) -> None:
        """Update routing metrics."""
        if route_used:
            self.routing_metrics["successful_routes"] = (
                self.routing_metrics.get("successful_routes", 0) + 1
            )
        else:
            self.routing_metrics["failed_routes"] = (
                self.routing_metrics.get("failed_routes", 0) + 1
            )


class ShardNetwork:
    """Manages shard network topology and communication."""

    def __init__(self):
        """Initialize shard network."""
        self.topology = ShardTopology()
        self.discovery = ShardDiscovery()
        self.routing = ShardRouting()
        self.network_metrics = {"connections_established": 0, "connections_lost": 0}

    def add_shard(self, shard_id: ShardId) -> None:
        """Add shard to network."""
        self.discovery.known_shards.add(shard_id)

    def remove_shard(self, shard_id: ShardId) -> None:
        """Remove shard from network."""
        self.discovery.known_shards.discard(shard_id)

        # Remove all connections involving this shard
        if shard_id in self.topology.shard_connections:
            connected_shards = self.topology.shard_connections[shard_id].copy()
            for connected_shard in connected_shards:
                self.topology.remove_connection(shard_id, connected_shard)

    def establish_connection(
        self, shard1: ShardId, shard2: ShardId, weight: float = 1.0
    ) -> bool:
        """Establish connection between shards."""
        if (
            shard1 not in self.discovery.known_shards
            or shard2 not in self.discovery.known_shards
        ):
            return False

        self.topology.add_connection(shard1, shard2, weight)
        self.network_metrics["connections_established"] += 1

        # Update routing table
        self.routing.add_route(shard1, shard2, [shard1, shard2])
        self.routing.add_route(shard2, shard1, [shard2, shard1])

        return True

    def break_connection(self, shard1: ShardId, shard2: ShardId) -> bool:
        """Break connection between shards."""
        self.topology.remove_connection(shard1, shard2)
        self.network_metrics["connections_lost"] += 1

        # Remove from routing table
        if (
            shard1 in self.routing.routing_table
            and shard2 in self.routing.routing_table[shard1]
        ):
            del self.routing.routing_table[shard1][shard2]
        if (
            shard2 in self.routing.routing_table
            and shard1 in self.routing.routing_table[shard2]
        ):
            del self.routing.routing_table[shard2][shard1]

        return True

    def discover_network(self, shard_states: Dict[ShardId, ShardState]) -> Set[ShardId]:
        """Discover network topology."""
        return self.discovery.discover_shards(shard_states)

    def get_route(self, source: ShardId, target: ShardId) -> List[ShardId]:
        """Get route between shards."""
        route = self.routing.find_route(source, target)
        self.routing.update_metrics(len(route) > 0)
        return route

    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics."""
        return {
            "known_shards": len(self.discovery.known_shards),
            "total_connections": len(self.topology.connection_weights) // 2,
            "routing_table_size": sum(
                len(routes) for routes in self.routing.routing_table.values()
            ),
            "network_metrics": self.network_metrics,
            "routing_metrics": self.routing.routing_metrics,
        }
