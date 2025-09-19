"""
Advanced peer discovery system for GodChain.

This module provides sophisticated peer discovery mechanisms including
DNS-based discovery, bootstrap nodes, and peer exchange protocols.
"""

import asyncio
import random
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..crypto.signatures import PublicKey
from .peer import ConnectionType, Peer, PeerInfo


class DiscoveryMethod(Enum):
    """Peer discovery methods."""

    BOOTSTRAP = "bootstrap"
    DNS = "dns"
    PEER_EXCHANGE = "peer_exchange"
    MULTICAST = "multicast"
    MANUAL = "manual"
    DHT = "dht"  # Distributed Hash Table


@dataclass
class DiscoveryConfig:
    """Configuration for peer discovery."""

    bootstrap_nodes: List[str] = field(default_factory=list)
    dns_seeds: List[str] = field(default_factory=list)
    multicast_address: str = "224.0.0.1"
    multicast_port: int = 12345
    discovery_interval: float = 30.0
    max_peers: int = 100
    min_peers: int = 5
    peer_timeout: float = 10.0
    enable_dns_discovery: bool = True
    enable_multicast: bool = True
    enable_peer_exchange: bool = True
    peer_exchange_interval: float = 60.0
    max_peer_exchange_peers: int = 20
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.discovery_interval <= 0:
            raise ValueError("Discovery interval must be positive")

        if self.max_peers <= 0:
            raise ValueError("Max peers must be positive")

        if self.min_peers < 0:
            raise ValueError("Min peers cannot be negative")

        if self.min_peers > self.max_peers:
            raise ValueError("Min peers cannot be greater than max peers")

        if self.peer_timeout <= 0:
            raise ValueError("Peer timeout must be positive")

        if self.peer_exchange_interval <= 0:
            raise ValueError("Peer exchange interval must be positive")

        if self.max_peer_exchange_peers <= 0:
            raise ValueError("Max peer exchange peers must be positive")


class PeerDiscovery:
    """Advanced peer discovery implementation."""

    def __init__(
        self, config: DiscoveryConfig, node_id: str, private_key: Optional[Any] = None
    ):
        """Initialize peer discovery."""
        self.config = config
        self.node_id = node_id
        self.private_key = private_key
        self.discovered_peers: Dict[str, PeerInfo] = {}
        self.connected_peers: Dict[str, Peer] = {}
        self.bootstrap_peers: List[PeerInfo] = []
        self.discovery_callbacks: List[Callable] = []
        self.discovery_task: Optional[asyncio.Task] = None
        self.peer_exchange_task: Optional[asyncio.Task] = None
        self.running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start peer discovery."""
        if self.running:
            return

        self.running = True

        # Initialize bootstrap peers
        await self._initialize_bootstrap_peers()

        # Start discovery task
        self.discovery_task = asyncio.create_task(self._discovery_loop())

        # Start peer exchange task
        if self.config.enable_peer_exchange:
            self.peer_exchange_task = asyncio.create_task(self._peer_exchange_loop())

    async def stop(self) -> None:
        """Stop peer discovery."""
        if not self.running:
            return

        self.running = False

        # Cancel tasks
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass

        if self.peer_exchange_task:
            self.peer_exchange_task.cancel()
            try:
                await self.peer_exchange_task
            except asyncio.CancelledError:
                pass

    def add_discovery_callback(self, callback: Callable) -> None:
        """Add discovery callback."""
        self.discovery_callbacks.append(callback)

    async def discover_peers(self, method: DiscoveryMethod) -> List[PeerInfo]:
        """Discover peers using specified method."""
        if method == DiscoveryMethod.BOOTSTRAP:
            return await self._discover_bootstrap_peers()
        elif method == DiscoveryMethod.DNS:
            return await self._discover_dns_peers()
        elif method == DiscoveryMethod.MULTICAST:
            return await self._discover_multicast_peers()
        elif method == DiscoveryMethod.PEER_EXCHANGE:
            return await self._discover_peer_exchange_peers()
        else:
            return []

    async def add_peer(self, peer_info: PeerInfo) -> None:
        """Add discovered peer."""
        async with self._lock:
            if peer_info.peer_id not in self.discovered_peers:
                self.discovered_peers[peer_info.peer_id] = peer_info

                # Notify callbacks
                for callback in self.discovery_callbacks:
                    try:
                        await callback(peer_info)
                    except Exception:
                        pass

    async def remove_peer(self, peer_id: str) -> None:
        """Remove peer."""
        async with self._lock:
            if peer_id in self.discovered_peers:
                del self.discovered_peers[peer_id]

            if peer_id in self.connected_peers:
                del self.connected_peers[peer_id]

    async def get_peers(
        self,
        count: Optional[int] = None,
        connection_type: Optional[ConnectionType] = None,
    ) -> List[PeerInfo]:
        """Get discovered peers."""
        async with self._lock:
            peers = list(self.discovered_peers.values())

            # Filter by connection type
            if connection_type:
                peers = [
                    peer for peer in peers if peer.connection_type == connection_type
                ]

            # Limit count
            if count:
                peers = peers[:count]

            return peers

    async def get_connected_peers(self) -> List[Peer]:
        """Get connected peers."""
        async with self._lock:
            return list(self.connected_peers.values())

    async def connect_to_peer(self, peer_info: PeerInfo) -> Optional[Peer]:
        """Connect to a discovered peer."""
        try:
            # Create peer
            peer = Peer(peer_info, self.private_key)

            # Connect
            success = await peer.connect(timeout=self.config.peer_timeout)
            if not success:
                return None

            # Store connected peer
            async with self._lock:
                self.connected_peers[peer_info.peer_id] = peer

            return peer

        except Exception:
            return None

    async def _discovery_loop(self) -> None:
        """Main discovery loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.discovery_interval)

                # Check if we need more peers
                if len(self.connected_peers) < self.config.min_peers:
                    await self._perform_discovery()

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _peer_exchange_loop(self) -> None:
        """Peer exchange loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.peer_exchange_interval)

                if len(self.connected_peers) > 0:
                    await self._exchange_peers()

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _perform_discovery(self) -> None:
        """Perform peer discovery using multiple methods."""
        discovery_methods = []

        if self.config.bootstrap_nodes:
            discovery_methods.append(DiscoveryMethod.BOOTSTRAP)

        if self.config.enable_dns_discovery and self.config.dns_seeds:
            discovery_methods.append(DiscoveryMethod.DNS)

        if self.config.enable_multicast:
            discovery_methods.append(DiscoveryMethod.MULTICAST)

        if self.config.enable_peer_exchange and len(self.connected_peers) > 0:
            discovery_methods.append(DiscoveryMethod.PEER_EXCHANGE)

        # Try discovery methods
        for method in discovery_methods:
            try:
                peers = await self.discover_peers(method)

                # Add discovered peers
                for peer_info in peers:
                    await self.add_peer(peer_info)

                # Try to connect to new peers
                await self._connect_to_new_peers()

                # Stop if we have enough peers
                if len(self.connected_peers) >= self.config.min_peers:
                    break

            except Exception:
                continue

    async def _connect_to_new_peers(self) -> None:
        """Connect to newly discovered peers."""
        current_connected = set(self.connected_peers.keys())

        for peer_info in self.discovered_peers.values():
            if (
                peer_info.peer_id not in current_connected
                and len(self.connected_peers) < self.config.max_peers
            ):
                peer = await self.connect_to_peer(peer_info)
                if peer:
                    current_connected.add(peer_info.peer_id)

    async def _initialize_bootstrap_peers(self) -> None:
        """Initialize bootstrap peers."""
        for bootstrap_node in self.config.bootstrap_nodes:
            try:
                # Parse bootstrap node (format: "address:port" or "address:port:peer_id")
                parts = bootstrap_node.split(":")
                if len(parts) >= 2:
                    address = parts[0]
                    port = int(parts[1])
                    peer_id = (
                        parts[2] if len(parts) > 2 else f"bootstrap_{address}_{port}"
                    )

                    # Create peer info
                    peer_info = PeerInfo(
                        peer_id=peer_id,
                        public_key=PublicKey.generate(),  # Placeholder
                        address=address,
                        port=port,
                        connection_type=ConnectionType.SEED,
                    )

                    self.bootstrap_peers.append(peer_info)

            except Exception:
                continue

    async def _discover_bootstrap_peers(self) -> List[PeerInfo]:
        """Discover peers from bootstrap nodes."""
        discovered_peers = []

        for bootstrap_peer in self.bootstrap_peers:
            try:
                # Try to connect to bootstrap peer
                peer = await self.connect_to_peer(bootstrap_peer)
                if peer:
                    # Request peer list
                    peer_list = await self._request_peer_list(peer)
                    discovered_peers.extend(peer_list)

                    # Disconnect from bootstrap peer
                    await peer.disconnect()

            except Exception:
                continue

        return discovered_peers

    async def _discover_dns_peers(self) -> List[PeerInfo]:
        """Discover peers using DNS seeds."""
        discovered_peers = []

        for dns_seed in self.config.dns_seeds:
            try:
                # Resolve DNS seed
                addresses = await self._resolve_dns_seed(dns_seed)

                for address in addresses:
                    # Create peer info for each address
                    peer_info = PeerInfo(
                        peer_id=f"dns_{address}",
                        public_key=PublicKey.generate(),  # Placeholder
                        address=address,
                        port=8333,  # Default port
                        connection_type=ConnectionType.SEED,
                    )

                    discovered_peers.append(peer_info)

            except Exception:
                continue

        return discovered_peers

    async def _discover_multicast_peers(self) -> List[PeerInfo]:
        """Discover peers using multicast."""
        discovered_peers = []

        try:
            # Create multicast socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind to multicast address
            sock.bind(("", self.config.multicast_port))

            # Join multicast group
            mreq = socket.inet_aton(self.config.multicast_address) + socket.inet_aton(
                "0.0.0.0"
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            # Set timeout
            sock.settimeout(5.0)

            # Send discovery request
            discovery_request = {
                "type": "discovery_request",
                "node_id": self.node_id,
                "timestamp": int(time.time()),
            }

            import json

            message = json.dumps(discovery_request).encode("utf-8")
            sock.sendto(
                message, (self.config.multicast_address, self.config.multicast_port)
            )

            # Listen for responses
            start_time = time.time()
            while time.time() - start_time < 5.0:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode("utf-8"))

                    if response.get("type") == "discovery_response":
                        peer_info = PeerInfo(
                            peer_id=response.get("peer_id", f"multicast_{addr[0]}"),
                            public_key=PublicKey.generate(),  # Placeholder
                            address=addr[0],
                            port=response.get("port", 8333),
                            connection_type=ConnectionType.OUTBOUND,
                        )

                        discovered_peers.append(peer_info)

                except socket.timeout:
                    break
                except Exception:
                    continue

            sock.close()

        except Exception:
            pass

        return discovered_peers

    async def _discover_peer_exchange_peers(self) -> List[PeerInfo]:
        """Discover peers through peer exchange."""
        discovered_peers = []

        # Select random connected peers for peer exchange
        connected_peers = list(self.connected_peers.values())
        if not connected_peers:
            return discovered_peers

        selected_peers = random.sample(connected_peers, min(3, len(connected_peers)))

        for peer in selected_peers:
            try:
                # Request peer list
                peer_list = await self._request_peer_list(peer)
                discovered_peers.extend(peer_list)

            except Exception:
                continue

        return discovered_peers

    async def _exchange_peers(self) -> None:
        """Exchange peer information with connected peers."""
        if len(self.connected_peers) == 0:
            return

        # Select random peer for exchange
        peer = random.choice(list(self.connected_peers.values()))

        try:
            # Send our peer list
            await self._send_peer_list(peer)

        except Exception:
            pass

    async def _request_peer_list(self, peer: Peer) -> List[PeerInfo]:
        """Request peer list from a peer."""
        try:
            request = {
                "type": "peer_list_request",
                "node_id": self.node_id,
                "timestamp": int(time.time()),
            }

            import json

            message = json.dumps(request).encode("utf-8")
            success = await peer.send_message(message)

            if success:
                # Wait for response (simplified)
                await asyncio.sleep(1.0)

                # Return empty list for now (would parse actual response)
                return []

            return []

        except Exception:
            return []

    async def _send_peer_list(self, peer: Peer) -> None:
        """Send peer list to a peer."""
        try:
            # Get our peer list
            peer_list = await self.get_peers(count=self.config.max_peer_exchange_peers)

            # Convert to serializable format
            peer_data = []
            for peer_info in peer_list:
                peer_data.append(
                    {
                        "peer_id": peer_info.peer_id,
                        "address": peer_info.address,
                        "port": peer_info.port,
                        "connection_type": peer_info.connection_type.value,
                        "last_seen": peer_info.last_seen,
                    }
                )

            response = {
                "type": "peer_list_response",
                "node_id": self.node_id,
                "peers": peer_data,
                "timestamp": int(time.time()),
            }

            import json

            message = json.dumps(response).encode("utf-8")
            await peer.send_message(message)

        except Exception:
            pass

    async def _resolve_dns_seed(self, dns_seed: str) -> List[str]:
        """Resolve DNS seed to IP addresses."""
        try:
            # Use asyncio to resolve DNS
            loop = asyncio.get_event_loop()
            result = await loop.getaddrinfo(dns_seed, None, family=socket.AF_INET)

            addresses = []
            for addr_info in result:
                addresses.append(addr_info[4][0])

            return addresses

        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            "node_id": self.node_id,
            "discovered_peers_count": len(self.discovered_peers),
            "connected_peers_count": len(self.connected_peers),
            "bootstrap_peers_count": len(self.bootstrap_peers),
            "running": self.running,
            "config": {
                "discovery_interval": self.config.discovery_interval,
                "max_peers": self.config.max_peers,
                "min_peers": self.config.min_peers,
                "peer_timeout": self.config.peer_timeout,
                "enable_dns_discovery": self.config.enable_dns_discovery,
                "enable_multicast": self.config.enable_multicast,
                "enable_peer_exchange": self.config.enable_peer_exchange,
            },
        }

    def __str__(self) -> str:
        """String representation."""
        return f"PeerDiscovery(node_id={self.node_id}, discovered={len(self.discovered_peers)}, connected={len(self.connected_peers)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"PeerDiscovery(node_id={self.node_id}, discovered={len(self.discovered_peers)}, "
            f"connected={len(self.connected_peers)}, running={self.running})"
        )
