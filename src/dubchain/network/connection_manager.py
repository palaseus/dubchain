"""
Advanced connection management for GodChain P2P network.

This module provides sophisticated connection management including connection
pooling, load balancing, and automatic reconnection mechanisms.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .peer import ConnectionType, Peer, PeerInfo, PeerStatus


class ConnectionStrategy(Enum):
    """Connection management strategies."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LATENCY_BASED = "latency_based"
    LOAD_BALANCED = "load_balanced"
    GEOGRAPHIC = "geographic"


@dataclass
class ConnectionConfig:
    """Configuration for connection management."""

    max_connections: int = 50
    min_connections: int = 5
    connection_timeout: float = 10.0
    keepalive_interval: float = 30.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 5
    connection_strategy: ConnectionStrategy = ConnectionStrategy.LOAD_BALANCED
    enable_compression: bool = True
    enable_encryption: bool = True
    max_message_size: int = 1024 * 1024  # 1MB
    connection_pool_size: int = 100
    health_check_interval: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")

        if self.min_connections < 0:
            raise ValueError("Min connections cannot be negative")

        if self.min_connections > self.max_connections:
            raise ValueError("Min connections cannot be greater than max connections")

        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")

        if self.keepalive_interval <= 0:
            raise ValueError("Keepalive interval must be positive")

        if self.reconnect_interval <= 0:
            raise ValueError("Reconnect interval must be positive")

        if self.max_reconnect_attempts < 0:
            raise ValueError("Max reconnect attempts cannot be negative")

        if self.max_message_size <= 0:
            raise ValueError("Max message size must be positive")

        if self.connection_pool_size <= 0:
            raise ValueError("Connection pool size must be positive")

        if self.health_check_interval <= 0:
            raise ValueError("Health check interval must be positive")


class ConnectionPool:
    """Connection pool for managing peer connections."""

    def __init__(self, config: ConnectionConfig):
        """Initialize connection pool."""
        self.config = config
        self.connections: Dict[str, Peer] = {}
        self.connection_queue: List[PeerInfo] = []
        self.failed_connections: Dict[str, int] = {}
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def add_peer(self, peer_info: PeerInfo) -> None:
        """Add peer to connection queue."""
        async with self._lock:
            if peer_info.peer_id not in self.connection_queue:
                self.connection_queue.append(peer_info)

    async def remove_peer(self, peer_id: str) -> None:
        """Remove peer from pool."""
        async with self._lock:
            # Remove from connections
            if peer_id in self.connections:
                peer = self.connections[peer_id]
                await peer.disconnect()
                del self.connections[peer_id]

            # Remove from queue
            self.connection_queue = [
                p for p in self.connection_queue if p.peer_id != peer_id
            ]

            # Remove from failed connections
            if peer_id in self.failed_connections:
                del self.failed_connections[peer_id]

            # Remove stats
            if peer_id in self.connection_stats:
                del self.connection_stats[peer_id]

    async def get_connection(self, peer_id: str) -> Optional[Peer]:
        """Get connection to specific peer."""
        async with self._lock:
            return self.connections.get(peer_id)

    async def get_available_connections(self) -> List[Peer]:
        """Get all available connections."""
        async with self._lock:
            return [peer for peer in self.connections.values() if peer.is_connected()]

    async def get_connection_count(self) -> int:
        """Get current connection count."""
        async with self._lock:
            return len(self.connections)

    async def can_add_connection(self) -> bool:
        """Check if we can add more connections."""
        async with self._lock:
            return len(self.connections) < self.config.max_connections

    async def needs_more_connections(self) -> bool:
        """Check if we need more connections."""
        async with self._lock:
            return len(self.connections) < self.config.min_connections

    async def get_connection_stats(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Get connection statistics."""
        async with self._lock:
            return self.connection_stats.get(peer_id)

    async def update_connection_stats(
        self, peer_id: str, stats: Dict[str, Any]
    ) -> None:
        """Update connection statistics."""
        async with self._lock:
            if peer_id not in self.connection_stats:
                self.connection_stats[peer_id] = {}

            self.connection_stats[peer_id].update(stats)
            self.connection_stats[peer_id]["last_updated"] = time.time()


class ConnectionManager:
    """Advanced connection manager for P2P network."""

    def __init__(
        self, config: ConnectionConfig, node_id: str, private_key: Optional[Any] = None
    ):
        """Initialize connection manager."""
        self.config = config
        self.node_id = node_id
        self.private_key = private_key
        self.connection_pool = ConnectionPool(config)
        self.connection_callbacks: List[Callable] = []
        self.disconnection_callbacks: List[Callable] = []
        self.health_check_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None
        self.running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start connection manager."""
        if self.running:
            return

        self.running = True

        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        # Start keepalive task
        self.keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop(self) -> None:
        """Stop connection manager."""
        if not self.running:
            return

        self.running = False

        # Cancel tasks
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass

        # Disconnect all peers
        await self.disconnect_all()

    def add_connection_callback(self, callback: Callable) -> None:
        """Add connection callback."""
        self.connection_callbacks.append(callback)

    def add_disconnection_callback(self, callback: Callable) -> None:
        """Add disconnection callback."""
        self.disconnection_callbacks.append(callback)

    async def connect_to_peer(self, peer_info: PeerInfo) -> Optional[Peer]:
        """Connect to a peer."""
        # Check if we can add more connections
        if not await self.connection_pool.can_add_connection():
            return None

        # Check if already connected
        existing_peer = await self.connection_pool.get_connection(peer_info.peer_id)
        if existing_peer and existing_peer.is_connected():
            return existing_peer

        try:
            # Create peer
            peer = Peer(peer_info, self.private_key)

            # Add callbacks
            peer.add_connection_callback(self._on_peer_connected)
            peer.add_disconnection_callback(self._on_peer_disconnected)

            # Connect
            success = await peer.connect(timeout=self.config.connection_timeout)
            if not success:
                await self._record_failed_connection(peer_info.peer_id)
                return None

            # Add to connection pool
            async with self._lock:
                self.connection_pool.connections[peer_info.peer_id] = peer

            # Initialize stats
            await self.connection_pool.update_connection_stats(
                peer_info.peer_id,
                {
                    "connected_at": time.time(),
                    "messages_sent": 0,
                    "messages_received": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "last_activity": time.time(),
                },
            )

            return peer

        except Exception:
            await self._record_failed_connection(peer_info.peer_id)
            return None

    async def disconnect_peer(self, peer_id: str) -> None:
        """Disconnect from a peer."""
        await self.connection_pool.remove_peer(peer_id)

    async def disconnect_all(self) -> None:
        """Disconnect from all peers."""
        async with self._lock:
            for peer in list(self.connection_pool.connections.values()):
                await peer.disconnect()

            self.connection_pool.connections.clear()
            self.connection_pool.connection_queue.clear()
            self.connection_pool.failed_connections.clear()
            self.connection_pool.connection_stats.clear()

    async def send_message(self, peer_id: str, message: bytes) -> bool:
        """Send message to specific peer."""
        peer = await self.connection_pool.get_connection(peer_id)
        if not peer or not peer.is_connected():
            return False

        try:
            success = await peer.send_message(message)
            if success:
                # Update stats
                stats = await self.connection_pool.get_connection_stats(peer_id)
                if stats:
                    stats["messages_sent"] += 1
                    stats["bytes_sent"] += len(message)
                    stats["last_activity"] = time.time()
                    await self.connection_pool.update_connection_stats(peer_id, stats)

            return success

        except Exception:
            return False

    async def broadcast_message(
        self, message: bytes, exclude_peers: Optional[List[str]] = None
    ) -> int:
        """Broadcast message to all connected peers."""
        exclude_peers = exclude_peers or []
        connected_peers = await self.connection_pool.get_available_connections()

        success_count = 0
        tasks = []

        for peer in connected_peers:
            if peer.get_peer_id() not in exclude_peers:
                task = asyncio.create_task(
                    self.send_message(peer.get_peer_id(), message)
                )
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)

        return success_count

    async def get_peer_for_message(self, message_type: str) -> Optional[Peer]:
        """Get best peer for sending a message."""
        connected_peers = await self.connection_pool.get_available_connections()
        if not connected_peers:
            return None

        if self.config.connection_strategy == ConnectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(connected_peers)
        elif self.config.connection_strategy == ConnectionStrategy.RANDOM:
            return random.choice(connected_peers)
        elif self.config.connection_strategy == ConnectionStrategy.LATENCY_BASED:
            return self._select_lowest_latency(connected_peers)
        elif self.config.connection_strategy == ConnectionStrategy.LOAD_BALANCED:
            return self._select_least_loaded(connected_peers)
        else:
            return random.choice(connected_peers)

    async def maintain_connections(self, available_peers: List[PeerInfo]) -> None:
        """Maintain optimal number of connections."""
        current_count = await self.connection_pool.get_connection_count()

        # Add connections if needed
        if current_count < self.config.min_connections:
            needed = self.config.min_connections - current_count
            await self._add_connections(available_peers, needed)

        # Remove excess connections
        elif current_count > self.config.max_connections:
            excess = current_count - self.config.max_connections
            await self._remove_excess_connections(excess)

    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Check health of all connections
                await self._check_connection_health()

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _keepalive_loop(self) -> None:
        """Keepalive loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.keepalive_interval)

                # Send keepalive to all connections
                await self._send_keepalive()

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _check_connection_health(self) -> None:
        """Check health of all connections."""
        connected_peers = await self.connection_pool.get_available_connections()

        for peer in connected_peers:
            try:
                # Ping peer
                latency = await peer.ping()

                if latency is None:
                    # Connection is unhealthy, disconnect
                    await self.disconnect_peer(peer.get_peer_id())
                else:
                    # Update stats
                    await self.connection_pool.update_connection_stats(
                        peer.get_peer_id(),
                        {"latency": latency, "last_health_check": time.time()},
                    )

            except Exception:
                # Connection is unhealthy, disconnect
                await self.disconnect_peer(peer.get_peer_id())

    async def _send_keepalive(self) -> None:
        """Send keepalive to all connections."""
        connected_peers = await self.connection_pool.get_available_connections()

        for peer in connected_peers:
            try:
                keepalive_message = {
                    "type": "keepalive",
                    "node_id": self.node_id,
                    "timestamp": int(time.time()),
                }

                import json

                message_bytes = json.dumps(keepalive_message).encode("utf-8")
                await self.send_message(peer.get_peer_id(), message_bytes)

            except Exception:
                # Connection failed, will be handled by health check
                pass

    async def _add_connections(
        self, available_peers: List[PeerInfo], count: int
    ) -> None:
        """Add new connections."""
        # Filter out already connected peers
        connected_peer_ids = set(
            peer.get_peer_id()
            for peer in await self.connection_pool.get_available_connections()
        )

        available_peers = [
            peer for peer in available_peers if peer.peer_id not in connected_peer_ids
        ]

        # Select peers to connect to
        selected_peers = self._select_peers_for_connection(available_peers, count)

        # Connect to selected peers
        for peer_info in selected_peers:
            await self.connect_to_peer(peer_info)

    async def _remove_excess_connections(self, count: int) -> None:
        """Remove excess connections."""
        connected_peers = await self.connection_pool.get_available_connections()

        # Sort by priority (least important first)
        sorted_peers = self._sort_peers_by_priority(connected_peers)

        # Remove excess connections
        for peer in sorted_peers[:count]:
            await self.disconnect_peer(peer.get_peer_id())

    def _select_peers_for_connection(
        self, available_peers: List[PeerInfo], count: int
    ) -> List[PeerInfo]:
        """Select peers for connection."""
        if len(available_peers) <= count:
            return available_peers

        # Filter out peers with too many failed connections
        filtered_peers = [
            peer
            for peer in available_peers
            if self.connection_pool.failed_connections.get(peer.peer_id, 0)
            < self.config.max_reconnect_attempts
        ]

        if len(filtered_peers) <= count:
            return filtered_peers

        # Select based on strategy
        if self.config.connection_strategy == ConnectionStrategy.RANDOM:
            return random.sample(filtered_peers, count)
        else:
            # Default to random selection
            return random.sample(filtered_peers, count)

    def _sort_peers_by_priority(self, peers: List[Peer]) -> List[Peer]:
        """Sort peers by priority for removal."""

        # Sort by connection type (remove outbound before inbound)
        # and then by activity (remove least active first)
        def priority_key(peer):
            peer_info = peer.get_info()
            type_priority = (
                0 if peer_info.connection_type == ConnectionType.INBOUND else 1
            )
            activity_priority = peer_info.get_idle_time()
            return (type_priority, activity_priority)

        return sorted(peers, key=priority_key)

    def _select_round_robin(self, peers: List[Peer]) -> Peer:
        """Select peer using round-robin strategy."""
        # Simple round-robin implementation
        if not hasattr(self, "_round_robin_index"):
            self._round_robin_index = 0

        peer = peers[self._round_robin_index % len(peers)]
        self._round_robin_index += 1
        return peer

    def _select_lowest_latency(self, peers: List[Peer]) -> Peer:
        """Select peer with lowest latency."""
        best_peer = None
        best_latency = float("inf")

        for peer in peers:
            peer_info = peer.get_info()
            if peer_info.latency is not None and peer_info.latency < best_latency:
                best_latency = peer_info.latency
                best_peer = peer

        return best_peer or peers[0]

    def _select_least_loaded(self, peers: List[Peer]) -> Peer:
        """Select least loaded peer."""
        best_peer = None
        best_load = float("inf")

        for peer in peers:
            peer_info = peer.get_info()
            # Simple load calculation based on messages sent
            load = peer_info.messages_sent
            if load < best_load:
                best_load = load
                best_peer = peer

        return best_peer or peers[0]

    async def _record_failed_connection(self, peer_id: str) -> None:
        """Record failed connection attempt."""
        async with self._lock:
            self.connection_pool.failed_connections[peer_id] = (
                self.connection_pool.failed_connections.get(peer_id, 0) + 1
            )

    async def _on_peer_connected(self, peer: Peer) -> None:
        """Handle peer connection."""
        # Reset failed connection count
        async with self._lock:
            if peer.get_peer_id() in self.connection_pool.failed_connections:
                del self.connection_pool.failed_connections[peer.get_peer_id()]

        # Notify callbacks
        for callback in self.connection_callbacks:
            try:
                await callback(peer)
            except Exception:
                pass

    async def _on_peer_disconnected(self, peer: Peer) -> None:
        """Handle peer disconnection."""
        # Notify callbacks
        for callback in self.disconnection_callbacks:
            try:
                await callback(peer)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        return {
            "node_id": self.node_id,
            "connections_count": len(self.connection_pool.connections),
            "connection_queue_count": len(self.connection_pool.connection_queue),
            "failed_connections_count": len(self.connection_pool.failed_connections),
            "running": self.running,
            "config": {
                "max_connections": self.config.max_connections,
                "min_connections": self.config.min_connections,
                "connection_timeout": self.config.connection_timeout,
                "connection_strategy": self.config.connection_strategy.value,
            },
        }

    def __str__(self) -> str:
        """String representation."""
        return f"ConnectionManager(node_id={self.node_id}, connections={len(self.connection_pool.connections)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"ConnectionManager(node_id={self.node_id}, connections={len(self.connection_pool.connections)}, "
            f"running={self.running})"
        )
