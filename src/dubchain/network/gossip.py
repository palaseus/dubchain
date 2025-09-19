"""
Advanced gossip protocol implementation for GodChain.

This module provides a sophisticated gossip protocol for efficient message
propagation across the P2P network with anti-entropy and epidemic algorithms.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import Signature
from .peer import Peer, PeerInfo


class MessageType(Enum):
    """Types of gossip messages."""

    BLOCK = "block"
    TRANSACTION = "transaction"
    PEER_INFO = "peer_info"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    HEARTBEAT = "heartbeat"
    ANNOUNCEMENT = "announcement"
    QUERY = "query"
    RESPONSE = "response"
    CUSTOM = "custom"


@dataclass
class GossipMessage:
    """Gossip protocol message."""

    message_id: str
    message_type: MessageType
    sender_id: str
    content: Any
    timestamp: int = field(default_factory=lambda: int(time.time()))
    ttl: int = 3600  # Time to live in seconds
    hop_count: int = 0
    max_hops: int = 10
    signature: Optional[Signature] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate message."""
        if not self.message_id:
            raise ValueError("Message ID cannot be empty")

        if not self.sender_id:
            raise ValueError("Sender ID cannot be empty")

        if self.ttl <= 0:
            raise ValueError("TTL must be positive")

        if self.max_hops <= 0:
            raise ValueError("Max hops must be positive")

        if self.hop_count < 0:
            raise ValueError("Hop count cannot be negative")

    def is_expired(self) -> bool:
        """Check if message is expired."""
        return int(time.time()) - self.timestamp > self.ttl

    def can_hop(self) -> bool:
        """Check if message can hop further."""
        return self.hop_count < self.max_hops and not self.is_expired()

    def increment_hop(self) -> None:
        """Increment hop count."""
        if self.can_hop():
            self.hop_count += 1

    def get_age(self) -> int:
        """Get message age in seconds."""
        return int(time.time()) - self.timestamp

    def get_remaining_ttl(self) -> int:
        """Get remaining TTL in seconds."""
        return max(0, self.ttl - self.get_age())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "hop_count": self.hop_count,
            "max_hops": self.max_hops,
            "signature": self.signature.to_hex() if self.signature else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GossipMessage":
        """Create from dictionary."""
        signature = None
        if data.get("signature"):
            signature = Signature.from_hex(data["signature"])

        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            content=data["content"],
            timestamp=data.get("timestamp", int(time.time())),
            ttl=data.get("ttl", 3600),
            hop_count=data.get("hop_count", 0),
            max_hops=data.get("max_hops", 10),
            signature=signature,
            metadata=data.get("metadata", {}),
        )


@dataclass
class GossipConfig:
    """Configuration for gossip protocol."""

    fanout: int = 3  # Number of peers to gossip to
    interval: float = 1.0  # Gossip interval in seconds
    max_messages: int = 1000  # Maximum messages to store
    message_ttl: int = 3600  # Default message TTL
    max_hops: int = 10  # Maximum hops for messages
    anti_entropy_interval: float = 60.0  # Anti-entropy interval
    push_pull_ratio: float = 0.5  # Ratio of push vs pull operations
    duplicate_detection_window: int = 300  # Duplicate detection window in seconds
    enable_compression: bool = True  # Enable message compression
    enable_encryption: bool = True  # Enable message encryption
    max_message_size: int = 1024 * 1024  # Maximum message size (1MB)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.fanout <= 0:
            raise ValueError("Fanout must be positive")

        if self.interval <= 0:
            raise ValueError("Interval must be positive")

        if self.max_messages <= 0:
            raise ValueError("Max messages must be positive")

        if self.message_ttl <= 0:
            raise ValueError("Message TTL must be positive")

        if self.max_hops <= 0:
            raise ValueError("Max hops must be positive")

        if not 0 <= self.push_pull_ratio <= 1:
            raise ValueError("Push-pull ratio must be between 0 and 1")

        if self.duplicate_detection_window <= 0:
            raise ValueError("Duplicate detection window must be positive")

        if self.max_message_size <= 0:
            raise ValueError("Max message size must be positive")


class GossipProtocol:
    """Advanced gossip protocol implementation."""

    def __init__(self, config: GossipConfig, node_id: str):
        """Initialize gossip protocol."""
        self.config = config
        self.node_id = node_id
        self.peers: Dict[str, Peer] = {}
        self.messages: Dict[str, GossipMessage] = {}
        self.message_history: List[str] = []
        self.duplicate_cache: Dict[str, int] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.gossip_task: Optional[asyncio.Task] = None
        self.anti_entropy_task: Optional[asyncio.Task] = None
        self.running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start gossip protocol."""
        if self.running:
            return

        self.running = True

        # Start gossip task
        self.gossip_task = asyncio.create_task(self._gossip_loop())

        # Start anti-entropy task
        self.anti_entropy_task = asyncio.create_task(self._anti_entropy_loop())

    async def stop(self) -> None:
        """Stop gossip protocol."""
        if not self.running:
            return

        self.running = False

        # Cancel tasks
        if self.gossip_task:
            self.gossip_task.cancel()
            try:
                await self.gossip_task
            except asyncio.CancelledError:
                pass

        if self.anti_entropy_task:
            self.anti_entropy_task.cancel()
            try:
                await self.anti_entropy_task
            except asyncio.CancelledError:
                pass

    def add_peer(self, peer: Peer) -> None:
        """Add peer to gossip network."""
        self.peers[peer.get_peer_id()] = peer

    def remove_peer(self, peer_id: str) -> None:
        """Remove peer from gossip network."""
        if peer_id in self.peers:
            del self.peers[peer_id]

    def add_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Add message handler."""
        self.message_handlers[message_type] = handler

    async def broadcast_message(
        self,
        message_type: MessageType,
        content: Any,
        ttl: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> str:
        """Broadcast message to all peers."""
        message_id = self._generate_message_id()

        message = GossipMessage(
            message_id=message_id,
            message_type=message_type,
            sender_id=self.node_id,
            content=content,
            ttl=ttl or self.config.message_ttl,
            max_hops=max_hops or self.config.max_hops,
        )

        # Store message
        await self._store_message(message)

        # Broadcast to peers
        await self._broadcast_to_peers(message)

        return message_id

    async def send_message_to_peer(
        self,
        peer_id: str,
        message_type: MessageType,
        content: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Send message to specific peer."""
        if peer_id not in self.peers:
            return False

        message_id = self._generate_message_id()

        message = GossipMessage(
            message_id=message_id,
            message_type=message_type,
            sender_id=self.node_id,
            content=content,
            ttl=ttl or self.config.message_ttl,
        )

        # Store message
        await self._store_message(message)

        # Send to peer
        peer = self.peers[peer_id]
        return await self._send_message_to_peer(peer, message)

    async def handle_incoming_message(
        self, peer: Peer, message_data: Dict[str, Any]
    ) -> None:
        """Handle incoming gossip message."""
        try:
            message = GossipMessage.from_dict(message_data)

            # Check if message is expired
            if message.is_expired():
                return

            # Check for duplicates
            if await self._is_duplicate(message.message_id):
                return

            # Store message
            await self._store_message(message)

            # Handle message content
            if message.message_type in self.message_handlers:
                await self.message_handlers[message.message_type](peer, message)

            # Forward message if it can hop
            if message.can_hop():
                message.increment_hop()
                await self._forward_message(message, exclude_peer=peer.get_peer_id())

        except Exception:
            pass  # Ignore malformed messages

    async def get_peer_messages(self, peer_id: str) -> List[GossipMessage]:
        """Get messages that peer might not have."""
        if peer_id not in self.peers:
            return []

        # Simple implementation: return recent messages
        # In a real implementation, this would use vector clocks or similar
        recent_messages = []
        current_time = int(time.time())

        for message in self.messages.values():
            if (
                message.sender_id != peer_id and current_time - message.timestamp < 300
            ):  # Last 5 minutes
                recent_messages.append(message)

        return recent_messages[:50]  # Limit to 50 messages

    async def sync_with_peer(self, peer_id: str) -> bool:
        """Perform anti-entropy sync with peer."""
        if peer_id not in self.peers:
            return False

        peer = self.peers[peer_id]

        try:
            # Request sync
            sync_message = {
                "type": "sync_request",
                "node_id": self.node_id,
                "timestamp": int(time.time()),
            }

            import json

            message_bytes = json.dumps(sync_message).encode("utf-8")
            success = await peer.send_message(message_bytes)

            return success

        except Exception:
            return False

    async def _gossip_loop(self) -> None:
        """Main gossip loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.interval)

                if not self.peers:
                    continue

                # Select random peers for gossip
                selected_peers = self._select_peers_for_gossip()

                # Gossip messages to selected peers
                for peer in selected_peers:
                    await self._gossip_to_peer(peer)

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _anti_entropy_loop(self) -> None:
        """Anti-entropy loop for consistency."""
        while self.running:
            try:
                await asyncio.sleep(self.config.anti_entropy_interval)

                if not self.peers:
                    continue

                # Select random peer for anti-entropy
                peer_id = random.choice(list(self.peers.keys()))
                await self.sync_with_peer(peer_id)

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    def _select_peers_for_gossip(self) -> List[Peer]:
        """Select peers for gossip."""
        available_peers = [peer for peer in self.peers.values() if peer.is_connected()]

        if not available_peers:
            return []

        # Select random peers up to fanout
        fanout = min(self.config.fanout, len(available_peers))
        return random.sample(available_peers, fanout)

    async def _gossip_to_peer(self, peer: Peer) -> None:
        """Gossip messages to a specific peer."""
        try:
            # Get messages to send
            messages_to_send = await self.get_peer_messages(peer.get_peer_id())

            if not messages_to_send:
                return

            # Send messages
            for message in messages_to_send[:10]:  # Limit to 10 messages
                await self._send_message_to_peer(peer, message)

        except Exception:
            pass

    async def _broadcast_to_peers(self, message: GossipMessage) -> None:
        """Broadcast message to all peers."""
        tasks = []
        for peer in self.peers.values():
            if peer.is_connected():
                task = asyncio.create_task(self._send_message_to_peer(peer, message))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _forward_message(
        self, message: GossipMessage, exclude_peer: Optional[str] = None
    ) -> None:
        """Forward message to other peers."""
        selected_peers = self._select_peers_for_gossip()

        # Remove excluded peer
        if exclude_peer:
            selected_peers = [
                peer for peer in selected_peers if peer.get_peer_id() != exclude_peer
            ]

        # Send to selected peers
        for peer in selected_peers:
            await self._send_message_to_peer(peer, message)

    async def _send_message_to_peer(self, peer: Peer, message: GossipMessage) -> bool:
        """Send message to specific peer."""
        try:
            gossip_message = {"type": "gossip", "message": message.to_dict()}

            import json

            message_bytes = json.dumps(gossip_message).encode("utf-8")

            return await peer.send_message(message_bytes)

        except Exception:
            return False

    async def _store_message(self, message: GossipMessage) -> None:
        """Store message in local cache."""
        async with self._lock:
            # Add to messages
            self.messages[message.message_id] = message

            # Add to history
            self.message_history.append(message.message_id)

            # Add to duplicate cache
            self.duplicate_cache[message.message_id] = int(time.time())

            # Cleanup old messages
            await self._cleanup_messages()

    async def _is_duplicate(self, message_id: str) -> bool:
        """Check if message is duplicate."""
        current_time = int(time.time())

        # Check in messages
        if message_id in self.messages:
            return True

        # Check in duplicate cache
        if message_id in self.duplicate_cache:
            cache_time = self.duplicate_cache[message_id]
            if current_time - cache_time < self.config.duplicate_detection_window:
                return True

        return False

    async def _cleanup_messages(self) -> None:
        """Cleanup old messages."""
        current_time = int(time.time())

        # Remove expired messages
        expired_messages = []
        for message_id, message in self.messages.items():
            if message.is_expired():
                expired_messages.append(message_id)

        for message_id in expired_messages:
            del self.messages[message_id]

        # Limit message count
        if len(self.messages) > self.config.max_messages:
            # Remove oldest messages
            sorted_messages = sorted(
                self.messages.items(), key=lambda x: x[1].timestamp
            )

            messages_to_remove = len(self.messages) - self.config.max_messages
            for message_id, _ in sorted_messages[:messages_to_remove]:
                del self.messages[message_id]

        # Cleanup duplicate cache
        expired_duplicates = []
        for message_id, timestamp in self.duplicate_cache.items():
            if current_time - timestamp > self.config.duplicate_detection_window:
                expired_duplicates.append(message_id)

        for message_id in expired_duplicates:
            del self.duplicate_cache[message_id]

        # Cleanup message history
        if len(self.message_history) > self.config.max_messages * 2:
            self.message_history = self.message_history[-self.config.max_messages :]

    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_data = random.getrandbits(64)
        data = f"{self.node_id}{timestamp}{random_data}".encode("utf-8")
        return SHA256Hasher.hash(data).to_hex()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get gossip protocol statistics."""
        return {
            "node_id": self.node_id,
            "peers_count": len(self.peers),
            "messages_count": len(self.messages),
            "message_history_count": len(self.message_history),
            "duplicate_cache_count": len(self.duplicate_cache),
            "running": self.running,
            "config": {
                "fanout": self.config.fanout,
                "interval": self.config.interval,
                "max_messages": self.config.max_messages,
                "message_ttl": self.config.message_ttl,
                "max_hops": self.config.max_hops,
            },
        }

    def __str__(self) -> str:
        """String representation."""
        return f"GossipProtocol(node_id={self.node_id}, peers={len(self.peers)}, messages={len(self.messages)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"GossipProtocol(node_id={self.node_id}, peers={len(self.peers)}, "
            f"messages={len(self.messages)}, running={self.running})"
        )
