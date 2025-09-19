"""
Advanced message routing for GodChain P2P network.

This module provides sophisticated message routing including intelligent
routing strategies, load balancing, and network optimization.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .peer import Peer, PeerInfo


class RoutingStrategy(Enum):
    """Message routing strategies."""

    SHORTEST_PATH = "shortest_path"
    LOAD_BALANCED = "load_balanced"
    LATENCY_OPTIMIZED = "latency_optimized"
    RANDOM = "random"
    FLOODING = "flooding"
    GEOMETRIC = "geometric"


@dataclass
class RouteInfo:
    """Information about a message route."""

    route_id: str
    source_peer: str
    target_peer: str
    intermediate_peers: List[str] = field(default_factory=list)
    total_hops: int = 0
    estimated_latency: float = 0.0
    route_quality: float = 1.0
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_used: int = field(default_factory=lambda: int(time.time()))
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate route info."""
        if not self.route_id:
            raise ValueError("Route ID cannot be empty")

        if not self.source_peer:
            raise ValueError("Source peer cannot be empty")

        if not self.target_peer:
            raise ValueError("Target peer cannot be empty")

        if self.total_hops < 0:
            raise ValueError("Total hops cannot be negative")

        if self.estimated_latency < 0:
            raise ValueError("Estimated latency cannot be negative")

        if not 0 <= self.route_quality <= 1:
            raise ValueError("Route quality must be between 0 and 1")

    def update_usage(self, success: bool = True) -> None:
        """Update route usage statistics."""
        self.usage_count += 1
        import time

        self.last_used = int(time.time())
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def record_success(self) -> None:
        """Record successful route usage."""
        self.update_usage(success=True)

    def record_failure(self) -> None:
        """Record failed route usage."""
        self.update_usage(success=False)

    def get_success_rate(self) -> float:
        """Get route success rate."""
        total_attempts = self.success_count + self.failure_count
        if total_attempts == 0:
            return 0.0
        return self.success_count / total_attempts

    @property
    def success_rate(self) -> float:
        """Get route success rate as property."""
        return self.get_success_rate()

    def get_age(self) -> int:
        """Get route age in seconds."""
        return int(time.time()) - self.created_at

    def get_idle_time(self) -> int:
        """Get route idle time in seconds."""
        return int(time.time()) - self.last_used

    def is_healthy(self, max_idle_time: int = 3600) -> bool:
        """Check if route is healthy."""
        return (
            self.get_success_rate() > 0.5
            and self.get_idle_time() < max_idle_time
            and self.route_quality > 0.3
        )


class RoutingTable:
    """Routing table for managing routes."""

    def __init__(self, max_routes: int = 1000):
        """Initialize routing table."""
        self._routes: Dict[str, RouteInfo] = {}
        self._peer_routes: Dict[str, List[RouteInfo]] = {}
        self._max_routes = max_routes

    def add_route(self, route: RouteInfo) -> None:
        """Add route to table."""
        if len(self._routes) >= self._max_routes:
            # Remove oldest route
            oldest_route_id = min(
                self._routes.keys(), key=lambda k: self._routes[k].last_used
            )
            self.remove_route(oldest_route_id)

        self._routes[route.route_id] = route

        # Update peer routes index
        if route.target_peer not in self._peer_routes:
            self._peer_routes[route.target_peer] = []
        self._peer_routes[route.target_peer].append(route)

    def remove_route(self, route_id: str) -> bool:
        """Remove route from table."""
        if route_id not in self._routes:
            return False

        route = self._routes[route_id]
        del self._routes[route_id]

        # Update peer routes index
        if route.target_peer in self._peer_routes:
            self._peer_routes[route.target_peer] = [
                r
                for r in self._peer_routes[route.target_peer]
                if r.route_id != route_id
            ]

        return True

    def get_route(self, route_id: str) -> Optional[RouteInfo]:
        """Get route by ID."""
        return self._routes.get(route_id)

    def get_routes_for_peer(self, peer_id: str) -> List[RouteInfo]:
        """Get all routes involving a peer (as source or target)."""
        routes = []
        for route in self._routes.values():
            if route.source_peer == peer_id or route.target_peer == peer_id:
                routes.append(route)
        return routes

    def cleanup_old_routes(self, max_age: int) -> None:
        """Clean up old routes."""
        current_time = int(time.time())
        routes_to_remove = []

        for route_id, route in self._routes.items():
            if current_time - route.created_at > max_age:
                routes_to_remove.append(route_id)

        for route_id in routes_to_remove:
            self.remove_route(route_id)

    @property
    def routes(self) -> Dict[str, RouteInfo]:
        """Get all routes."""
        return self._routes.copy()


class LoadBalancer:
    """Load balancer for distributing messages across peers."""

    def __init__(self):
        """Initialize load balancer."""
        self._peer_loads: Dict[str, float] = {}
        self._peer_capacities: Dict[str, float] = {}
        self._load_history: Dict[str, List[Tuple[int, float]]] = {}
        self._max_history = 100
        self._max_load = 100.0

    def update_peer_load(self, peer_id: str, load: float) -> None:
        """Update peer load."""
        self._peer_loads[peer_id] = load

        # Update load history
        if peer_id not in self._load_history:
            self._load_history[peer_id] = []

        self._load_history[peer_id].append((int(time.time()), load))

        # Trim history
        if len(self._load_history[peer_id]) > self._max_history:
            self._load_history[peer_id] = self._load_history[peer_id][
                -self._max_history :
            ]

    def set_peer_capacity(self, peer_id: str, capacity: float) -> None:
        """Set peer capacity."""
        self._peer_capacities[peer_id] = max(0.0, capacity)

    def get_peer_load(self, peer_id: str) -> float:
        """Get peer load."""
        return self._peer_loads.get(peer_id, 0.0)

    def get_least_loaded_peer(self, available_peers: List[str]) -> Optional[str]:
        """Get least loaded peer from available peers."""
        if not available_peers:
            return None

        least_loaded = None
        min_load = float("inf")

        for peer_id in available_peers:
            load = self.get_peer_load(peer_id)
            if load < min_load:
                min_load = load
                least_loaded = peer_id

        return least_loaded

    def select_least_loaded_peer(self, available_peers: List[str]) -> Optional[str]:
        """Select least loaded peer from available peers."""
        return self.get_least_loaded_peer(available_peers)

    def is_peer_overloaded(self, peer_id: str, threshold: float = 80.0) -> bool:
        """Check if peer is overloaded."""
        load = self.get_peer_load(peer_id)
        return load > threshold

    def get_load_distribution(self) -> Dict[str, float]:
        """Get load distribution across all peers."""
        return self._peer_loads.copy()

    def reset_loads(self) -> None:
        """Reset all peer loads."""
        self._peer_loads.clear()
        self._load_history.clear()


class LatencyTracker:
    """Tracks latency between peers."""

    def __init__(self, max_samples: int = 100):
        """Initialize latency tracker."""
        self._latencies: Dict[str, List[float]] = {}
        self._max_samples = max_samples
        self._last_measurement: Dict[str, int] = {}

    def record_latency(self, peer_id: str, latency_ms: float) -> None:
        """Record latency measurement for a peer."""
        if peer_id not in self._latencies:
            self._latencies[peer_id] = []

        self._latencies[peer_id].append(latency_ms)
        self._last_measurement[peer_id] = int(time.time())

        # Trim samples
        if len(self._latencies[peer_id]) > self._max_samples:
            self._latencies[peer_id] = self._latencies[peer_id][-self._max_samples :]

    def get_average_latency(self, peer_id: str) -> Optional[float]:
        """Get average latency for a peer."""
        if peer_id not in self._latencies or not self._latencies[peer_id]:
            return 0.0

        return sum(self._latencies[peer_id]) / len(self._latencies[peer_id])

    def get_latest_latency(self, peer_id: str) -> Optional[float]:
        """Get latest latency measurement for a peer."""
        if peer_id not in self._latencies or not self._latencies[peer_id]:
            return None

        return self._latencies[peer_id][-1]

    def get_best_latency_peer(self, available_peers: List[str]) -> Optional[str]:
        """Get peer with best (lowest) latency."""
        best_peer = None
        best_latency = float("inf")

        for peer_id in available_peers:
            latency = self.get_average_latency(peer_id)
            if latency is not None and latency < best_latency:
                best_latency = latency
                best_peer = peer_id

        return best_peer

    def get_lowest_latency_peer(self, available_peers: List[str]) -> Optional[str]:
        """Get peer with lowest latency."""
        return self.get_best_latency_peer(available_peers)

    def get_latency_stats(self, peer_id: str) -> Dict[str, float]:
        """Get latency statistics for a peer."""
        if peer_id not in self._latencies or not self._latencies[peer_id]:
            return {}

        latencies = self._latencies[peer_id]
        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": sum(latencies) / len(latencies),
            "count": len(latencies),
        }

    def cleanup_old_measurements(self, max_age: int) -> None:
        """Clean up old measurements."""
        current_time = int(time.time())
        peers_to_clean = []

        for peer_id, last_time in self._last_measurement.items():
            if current_time - last_time > max_age:
                peers_to_clean.append(peer_id)

        for peer_id in peers_to_clean:
            if peer_id in self._latencies:
                del self._latencies[peer_id]
            if peer_id in self._last_measurement:
                del self._last_measurement[peer_id]


class NetworkTopology:
    """Network topology management for message routing."""

    def __init__(self):
        """Initialize network topology."""
        self.connections: Dict[str, Set[str]] = {}
        self._connections: Dict[str, Set[str]] = {}  # Alias for compatibility
        self.peer_info: Dict[str, Dict[str, Any]] = {}
        self._peer_info: Dict[str, Dict[str, Any]] = {}  # Alias for compatibility
        self.topology_metrics: Dict[str, Any] = {}
        self.connection_latencies: Dict[Tuple[str, str], float] = {}

    def add_peer(self, peer_id_or_info, info: Dict[str, Any] = None) -> None:
        """Add peer to topology."""
        # Handle both peer_id string and peer_info object
        if isinstance(peer_id_or_info, str):
            peer_id = peer_id_or_info
        else:
            # Assume it's a peer_info object
            peer_info_obj = peer_id_or_info
            peer_id = getattr(peer_info_obj, "peer_id", str(peer_info_obj))
            info = peer_info_obj

        if peer_id not in self.connections:
            self.connections[peer_id] = set()
        if peer_id not in self._connections:
            self._connections[peer_id] = set()

        # Always add to _peer_info for compatibility
        if info:
            self.peer_info[peer_id] = info
            self._peer_info[peer_id] = info
        else:
            self._peer_info[peer_id] = {}

    def remove_peer(self, peer_id: str) -> None:
        """Remove peer from topology."""
        if peer_id in self.connections:
            # Remove connections to this peer
            for other_peer in self.connections[peer_id]:
                if other_peer in self.connections:
                    self.connections[other_peer].discard(peer_id)

            del self.connections[peer_id]

        if peer_id in self._connections:
            # Remove connections to this peer
            for other_peer in self._connections[peer_id]:
                if other_peer in self._connections:
                    self._connections[other_peer].discard(peer_id)

            del self._connections[peer_id]

        if peer_id in self.peer_info:
            del self.peer_info[peer_id]

        if peer_id in self._peer_info:
            del self._peer_info[peer_id]

    def add_connection(
        self, peer1_id: str, peer2_id: str, latency: float = 0.0
    ) -> None:
        """Add connection between peers."""
        if peer1_id not in self.connections:
            self.connections[peer1_id] = set()
        if peer2_id not in self.connections:
            self.connections[peer2_id] = set()
        if peer1_id not in self._connections:
            self._connections[peer1_id] = set()
        if peer2_id not in self._connections:
            self._connections[peer2_id] = set()

        self.connections[peer1_id].add(peer2_id)
        self.connections[peer2_id].add(peer1_id)
        self._connections[peer1_id].add(peer2_id)
        self._connections[peer2_id].add(peer1_id)

        # Store latency for the connection
        self.connection_latencies[(peer1_id, peer2_id)] = latency
        self.connection_latencies[(peer2_id, peer1_id)] = latency

    def remove_connection(self, peer1_id: str, peer2_id: str) -> None:
        """Remove connection between peers."""
        if peer1_id in self.connections:
            self.connections[peer1_id].discard(peer2_id)
        if peer2_id in self.connections:
            self.connections[peer2_id].discard(peer1_id)

    def get_connected_peers(self, peer_id: str) -> Set[str]:
        """Get peers connected to a peer."""
        return self.connections.get(peer_id, set()).copy()

    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two peers."""
        if source not in self.connections or target not in self.connections:
            return None

        if source == target:
            return [source]

        # BFS for shortest path
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            for neighbor in self.connections[current]:
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_connection_latency(self, peer1_id: str, peer2_id: str) -> Optional[float]:
        """Get latency between two peers."""
        latency = self.connection_latencies.get((peer1_id, peer2_id))
        if latency is None:
            return float("inf")
        return latency

    def get_topology_stats(self) -> Dict[str, Any]:
        """Get topology statistics."""
        total_peers = len(self.connections)
        total_connections = (
            sum(len(connections) for connections in self.connections.values()) // 2
        )

        return {
            "total_peers": total_peers,
            "total_connections": total_connections,
            "average_connections": total_connections / total_peers
            if total_peers > 0
            else 0,
        }


class MessageRouter:
    """Advanced message router with intelligent routing strategies."""

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.LOAD_BALANCED):
        """Initialize message router."""
        self.strategy = strategy
        self.routes: Dict[str, RouteInfo] = {}
        self.peer_connections: Dict[str, Peer] = {}
        self._peers = self.peer_connections  # Alias for compatibility
        self.network_topology = NetworkTopology()
        self.topology = self.network_topology  # Alias for compatibility
        self.routing_table = RoutingTable()
        self.load_balancer = LoadBalancer()
        self.latency_tracker = LatencyTracker()
        self.message_handlers: Dict[str, Callable] = {}
        self.routing_callbacks: List[Callable] = []
        self._lock = asyncio.Lock()

    def add_peer(self, peer: Peer) -> None:
        """Add peer to router."""
        # Handle both get_peer_id() method and peer_id attribute
        if hasattr(peer, "get_peer_id") and callable(peer.get_peer_id):
            peer_id = peer.get_peer_id()
        else:
            peer_id = getattr(peer, "peer_id", str(peer))

        self.peer_connections[peer_id] = peer

        # Initialize network topology
        self.network_topology.add_peer(peer_id)

        # Initialize routing table
        if peer_id not in self.routing_table._routes:
            pass  # RoutingTable handles this internally

    def remove_peer(self, peer_id: str) -> None:
        """Remove peer from router."""
        if peer_id in self.peer_connections:
            del self.peer_connections[peer_id]

        self.network_topology.remove_peer(peer_id)

        # Remove from routing table
        routes_to_remove = []
        for route_id, route in self.routing_table._routes.items():
            if route.source_peer == peer_id or route.target_peer == peer_id:
                routes_to_remove.append(route_id)

        for route_id in routes_to_remove:
            self.routing_table.remove_route(route_id)

        # Remove routes involving this peer
        routes_to_remove = []
        for route_id, route in self.routes.items():
            if (
                peer_id == route.source_peer
                or peer_id == route.target_peer
                or peer_id in route.intermediate_peers
            ):
                routes_to_remove.append(route_id)

        for route_id in routes_to_remove:
            del self.routes[route_id]

    def add_connection(self, peer1_id: str, peer2_id: str) -> None:
        """Add connection between peers."""
        self.network_topology.add_connection(peer1_id, peer2_id)

        # Update routing table
        self._update_routing_table()

    def remove_connection(self, peer1_id: str, peer2_id: str) -> None:
        """Remove connection between peers."""
        self.network_topology.remove_connection(peer1_id, peer2_id)

        # Update routing table
        self._update_routing_table()

    async def route_message(
        self, source_peer: str, target_peer: str, message: Any, max_hops: int = 10
    ) -> bool:
        """Route message from source to target peer."""
        if source_peer not in self.peer_connections:
            return False

        if target_peer not in self.peer_connections:
            return False

        # Find route
        route = await self._find_route(source_peer, target_peer, max_hops)
        if not route:
            return False

        # Execute route
        success = await self._execute_route(route, message)

        # Update route statistics
        if success:
            route.record_success()
        else:
            route.record_failure()

        return success

    async def broadcast_message(
        self, source_peer: str, message: Any, exclude_peers: Optional[List[str]] = None
    ) -> int:
        """Broadcast message to all peers."""
        exclude_peers = exclude_peers or []
        target_peers = [
            peer_id
            for peer_id in self.peer_connections.keys()
            if peer_id != source_peer and peer_id not in exclude_peers
        ]

        success_count = 0
        tasks = []

        for target_peer in target_peers:
            task = asyncio.create_task(
                self.route_message(source_peer, target_peer, message)
            )
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)

        return success_count

    async def _find_route(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> Optional[RouteInfo]:
        """Find route from source to target peer."""
        if self.strategy == RoutingStrategy.SHORTEST_PATH:
            return await self._find_shortest_path(source_peer, target_peer, max_hops)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._find_load_balanced_route(
                source_peer, target_peer, max_hops
            )
        elif self.strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return await self._find_latency_optimized_route(
                source_peer, target_peer, max_hops
            )
        elif self.strategy == RoutingStrategy.RANDOM:
            return await self._find_random_route(source_peer, target_peer, max_hops)
        elif self.strategy == RoutingStrategy.FLOODING:
            return await self._find_flooding_route(source_peer, target_peer, max_hops)
        else:
            return await self._find_shortest_path(source_peer, target_peer, max_hops)

    async def _find_shortest_path(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> Optional[RouteInfo]:
        """Find shortest path route."""
        if source_peer == target_peer:
            return RouteInfo(
                route_id=f"direct_{source_peer}_{target_peer}",
                source_peer=source_peer,
                target_peer=target_peer,
                total_hops=0,
                estimated_latency=0.0,
            )

        # Use BFS to find shortest path
        queue = [(source_peer, [source_peer])]
        visited = {source_peer}

        while queue:
            current_peer, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            if current_peer == target_peer:
                # Found target
                route_id = f"route_{source_peer}_{target_peer}_{int(time.time())}"
                return RouteInfo(
                    route_id=route_id,
                    source_peer=source_peer,
                    target_peer=target_peer,
                    intermediate_peers=path[1:-1],
                    total_hops=len(path) - 1,
                    estimated_latency=len(path) * 10.0,  # Estimate 10ms per hop
                )

            # Explore neighbors
            neighbors = self.network_topology.connections.get(current_peer, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    async def _find_load_balanced_route(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> Optional[RouteInfo]:
        """Find load-balanced route."""
        # Find all possible routes
        routes = await self._find_all_routes(source_peer, target_peer, max_hops)
        if not routes:
            return None

        # Select route with least load
        best_route = None
        best_load = float("inf")

        for route in routes:
            # Calculate route load based on usage
            load = route.usage_count
            if load < best_load:
                best_load = load
                best_route = route

        return best_route

    async def _find_latency_optimized_route(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> Optional[RouteInfo]:
        """Find latency-optimized route."""
        # Find all possible routes
        routes = await self._find_all_routes(source_peer, target_peer, max_hops)
        if not routes:
            return None

        # Select route with lowest estimated latency
        best_route = None
        best_latency = float("inf")

        for route in routes:
            if route.estimated_latency < best_latency:
                best_latency = route.estimated_latency
                best_route = route

        return best_route

    async def _find_random_route(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> Optional[RouteInfo]:
        """Find random route."""
        routes = await self._find_all_routes(source_peer, target_peer, max_hops)
        if not routes:
            return None

        return random.choice(routes)

    async def _find_flooding_route(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> Optional[RouteInfo]:
        """Find flooding route (direct broadcast)."""
        return RouteInfo(
            route_id=f"flood_{source_peer}_{target_peer}_{int(time.time())}",
            source_peer=source_peer,
            target_peer=target_peer,
            total_hops=1,
            estimated_latency=5.0,
        )

    async def _find_all_routes(
        self, source_peer: str, target_peer: str, max_hops: int
    ) -> List[RouteInfo]:
        """Find all possible routes."""
        routes = []

        if source_peer == target_peer:
            return [
                RouteInfo(
                    route_id=f"direct_{source_peer}_{target_peer}",
                    source_peer=source_peer,
                    target_peer=target_peer,
                    total_hops=0,
                    estimated_latency=0.0,
                )
            ]

        # Use DFS to find all routes
        def dfs(current_peer: str, path: List[str], visited: Set[str]):
            if len(path) > max_hops:
                return

            if current_peer == target_peer:
                route_id = f"route_{source_peer}_{target_peer}_{len(routes)}"
                route = RouteInfo(
                    route_id=route_id,
                    source_peer=source_peer,
                    target_peer=target_peer,
                    intermediate_peers=path[1:-1],
                    total_hops=len(path) - 1,
                    estimated_latency=len(path) * 10.0,
                )
                routes.append(route)
                return

            neighbors = self.network_topology.connections.get(current_peer, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, path + [neighbor], visited)
                    visited.remove(neighbor)

        dfs(source_peer, [source_peer], {source_peer})
        return routes

    async def _execute_route(self, route: RouteInfo, message: Any) -> bool:
        """Execute message routing along the route."""
        try:
            if route.total_hops == 0:
                # Direct route
                return await self._send_direct_message(
                    route.source_peer, route.target_peer, message
                )
            else:
                # Multi-hop route
                return await self._send_multi_hop_message(route, message)
        except Exception:
            return False

    async def _send_direct_message(
        self, source_peer: str, target_peer: str, message: Any
    ) -> bool:
        """Send direct message between peers."""
        if source_peer not in self.peer_connections:
            return False

        source_peer_obj = self.peer_connections[source_peer]

        # Create message
        import json

        message_data = {
            "type": "routed_message",
            "source": source_peer,
            "target": target_peer,
            "message": message,
            "timestamp": int(time.time()),
        }

        message_bytes = json.dumps(message_data).encode("utf-8")
        return await source_peer_obj.send_message(message_bytes)

    async def _send_multi_hop_message(self, route: RouteInfo, message: Any) -> bool:
        """Send multi-hop message."""
        # For simplicity, send directly to target
        # In a real implementation, this would route through intermediate peers
        return await self._send_direct_message(
            route.source_peer, route.target_peer, message
        )

    def _update_routing_table(self) -> None:
        """Update routing table based on network topology."""
        # Simple implementation - in a real system, this would use
        # algorithms like Dijkstra's or Bellman-Ford
        for source_peer in self.network_topology.connections:
            # For each destination, find next hop
            for target_peer in self.network_topology.connections:
                if source_peer != target_peer:
                    next_hop = self._find_next_hop(source_peer, target_peer)
                    if next_hop:
                        # Create route info
                        route_id = f"{source_peer}->{target_peer}"
                        route = RouteInfo(
                            route_id=route_id,
                            source_peer=source_peer,
                            target_peer=target_peer,
                            intermediate_peers=[next_hop]
                            if next_hop != target_peer
                            else [],
                            total_hops=1 if next_hop == target_peer else 2,
                        )
                        self.routing_table.add_route(route)

    def _find_next_hop(self, source_peer: str, target_peer: str) -> Optional[str]:
        """Find next hop for routing."""
        if source_peer == target_peer:
            return None

        # Check if direct connection exists
        if target_peer in self.network_topology.connections.get(source_peer, set()):
            return target_peer

        # Find shortest path
        queue = [(source_peer, [source_peer])]
        visited = {source_peer}

        while queue:
            current_peer, path = queue.pop(0)

            if current_peer == target_peer:
                return path[1] if len(path) > 1 else None

            neighbors = self.network_topology.connections.get(current_peer, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_route_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_routes = len(self.routes)
        healthy_routes = sum(1 for route in self.routes.values() if route.is_healthy())

        return {
            "strategy": self.strategy.value,
            "total_routes": total_routes,
            "healthy_routes": healthy_routes,
            "peers_count": len(self.peer_connections),
            "topology_connections": sum(
                len(neighbors)
                for neighbors in self.network_topology.connections.values()
            )
            // 2,
        }

    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """Set routing strategy."""
        self.strategy = strategy

    def register_peer(self, peer: Peer) -> None:
        """Register a peer."""
        self.add_peer(peer)

    def unregister_peer(self, peer_id: str) -> None:
        """Unregister a peer."""
        self.remove_peer(peer_id)

    def update_route_metrics(
        self, route_id: str, success: bool = True, latency: float = 0.0
    ) -> None:
        """Update route metrics."""
        route = self.routing_table.get_route(route_id)
        if route:
            if success:
                route.record_success()
            else:
                route.record_failure()

            if latency > 0:
                self.latency_tracker.record_latency(route.target_peer, latency)

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        stats = self.get_route_stats()
        # Add expected keys for compatibility
        stats["total_peers"] = stats.get("peers_count", 0)
        stats["average_latency"] = 0.0  # Placeholder
        stats["load_distribution"] = self.load_balancer.get_load_distribution()
        return stats

    def __str__(self) -> str:
        """String representation."""
        return f"MessageRouter(strategy={self.strategy.value}, routes={len(self.routes)}, peers={len(self.peer_connections)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"MessageRouter(strategy={self.strategy.value}, routes={len(self.routes)}, "
            f"peers={len(self.peer_connections)})"
        )


# Import asyncio at the top level
import asyncio
