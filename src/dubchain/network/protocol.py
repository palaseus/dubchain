"""
Network protocol implementation for DubChain.

This module provides the core networking protocol for peer-to-peer communication.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature


class MessageType(Enum):
    """Types of network messages."""

    HANDSHAKE = "handshake"
    PING = "ping"
    PONG = "pong"
    BLOCK = "block"
    TRANSACTION = "transaction"
    REQUEST_BLOCKS = "request_blocks"
    REQUEST_TRANSACTIONS = "request_transactions"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    PEER_DISCOVERY = "peer_discovery"
    CONSENSUS_MESSAGE = "consensus_message"
    ERROR = "error"


@dataclass
class NetworkMessage:
    """A network message."""

    message_type: MessageType
    data: Dict[str, Any]
    timestamp: int = 0
    sender_id: str = ""
    signature: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time())

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        message_dict = {
            "type": self.message_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "signature": self.signature,
        }
        return json.dumps(message_dict).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "NetworkMessage":
        """Deserialize message from bytes."""
        message_dict = json.loads(data.decode())
        return cls(
            message_type=MessageType(message_dict["type"]),
            data=message_dict["data"],
            timestamp=message_dict["timestamp"],
            sender_id=message_dict["sender_id"],
            signature=message_dict.get("signature"),
        )

    def sign(self, private_key: PrivateKey) -> "NetworkMessage":
        """Sign the message."""
        # Create hash of message data for signing
        message_hash = SHA256Hasher.hash(self.to_bytes())
        signature = private_key.sign(message_hash)
        self.signature = signature.to_hex()
        return self

    def verify(self, public_key: PublicKey) -> bool:
        """Verify the message signature."""
        if not self.signature:
            return False

        try:
            message_hash = SHA256Hasher.hash(self.to_bytes())
            signature = Signature.from_hex(self.signature, message_hash.value)
            return public_key.verify(signature, message_hash)
        except Exception:
            return False


class NetworkProtocol:
    """Core network protocol implementation."""

    def __init__(self, private_key: PrivateKey):
        self.private_key = private_key
        self.public_key = private_key.get_public_key()
        self.peer_id = self._generate_peer_id()
        self.message_handlers: Dict[MessageType, callable] = {}
        self.connected_peers: Dict[str, Any] = {}

    def _generate_peer_id(self) -> str:
        """Generate unique peer ID."""
        peer_id_hash = SHA256Hasher.hash(self.public_key.to_bytes())
        return peer_id_hash.value[:16].hex()

    def register_handler(self, message_type: MessageType, handler: callable) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler

    def create_message(
        self, message_type: MessageType, data: Dict[str, Any]
    ) -> NetworkMessage:
        """Create a new network message."""
        return NetworkMessage(
            message_type=message_type, data=data, sender_id=self.peer_id
        )

    def send_message(self, peer_id: str, message: NetworkMessage) -> bool:
        """Send a message to a peer."""
        try:
            if peer_id not in self.connected_peers:
                return False

            # Sign the message
            message.sign(self.private_key)

            # In a real implementation, would send through network
            # For now, just simulate success
            return True
        except Exception:
            return False

    def receive_message(self, message_bytes: bytes) -> Optional[NetworkMessage]:
        """Receive and process a message."""
        try:
            message = NetworkMessage.from_bytes(message_bytes)

            # Verify signature if present
            if message.signature and message.sender_id in self.connected_peers:
                peer_public_key = self.connected_peers[message.sender_id]
                if not message.verify(peer_public_key):
                    return None

            # Handle message
            if message.message_type in self.message_handlers:
                self.message_handlers[message.message_type](message)

            return message
        except Exception:
            return None

    def create_handshake_message(self) -> NetworkMessage:
        """Create a handshake message."""
        return self.create_message(
            MessageType.HANDSHAKE,
            {
                "peer_id": self.peer_id,
                "public_key": self.public_key.to_hex(),
                "version": "1.0.0",
                "capabilities": ["block_sync", "transaction_relay"],
            },
        )

    def create_ping_message(self) -> NetworkMessage:
        """Create a ping message."""
        return self.create_message(
            MessageType.PING,
            {
                "timestamp": int(time.time()),
                "nonce": SHA256Hasher.hash(f"ping_{time.time()}".encode()).to_hex(),
            },
        )

    def create_pong_message(self, ping_nonce: str) -> NetworkMessage:
        """Create a pong message."""
        return self.create_message(
            MessageType.PONG, {"ping_nonce": ping_nonce, "timestamp": int(time.time())}
        )

    def create_block_message(self, block_data: Dict[str, Any]) -> NetworkMessage:
        """Create a block message."""
        return self.create_message(MessageType.BLOCK, block_data)

    def create_transaction_message(
        self, transaction_data: Dict[str, Any]
    ) -> NetworkMessage:
        """Create a transaction message."""
        return self.create_message(MessageType.TRANSACTION, transaction_data)

    def create_sync_request_message(
        self, start_height: int, end_height: int
    ) -> NetworkMessage:
        """Create a sync request message."""
        return self.create_message(
            MessageType.SYNC_REQUEST,
            {
                "start_height": start_height,
                "end_height": end_height,
                "request_id": SHA256Hasher.hash(
                    f"sync_{start_height}_{end_height}".encode()
                ).to_hex(),
            },
        )

    def create_sync_response_message(
        self, blocks: List[Dict[str, Any]], request_id: str
    ) -> NetworkMessage:
        """Create a sync response message."""
        return self.create_message(
            MessageType.SYNC_RESPONSE, {"blocks": blocks, "request_id": request_id}
        )

    def create_error_message(
        self, error_code: str, error_message: str
    ) -> NetworkMessage:
        """Create an error message."""
        return self.create_message(
            MessageType.ERROR,
            {"error_code": error_code, "error_message": error_message},
        )


class MessageRouter:
    """Routes messages to appropriate handlers."""

    def __init__(self):
        self.routes: Dict[MessageType, List[callable]] = {}

    def register_route(self, message_type: MessageType, handler: callable) -> None:
        """Register a route for a message type."""
        if message_type not in self.routes:
            self.routes[message_type] = []
        self.routes[message_type].append(handler)

    def route_message(self, message: NetworkMessage) -> bool:
        """Route a message to its handlers."""
        if message.message_type not in self.routes:
            return False

        success = True
        for handler in self.routes[message.message_type]:
            try:
                handler(message)
            except Exception:
                success = False

        return success


class NetworkManager:
    """Manages network operations and peer connections."""

    def __init__(self, private_key: PrivateKey):
        self.protocol = NetworkProtocol(private_key)
        self.router = MessageRouter()
        self.peers: Dict[str, Any] = {}
        self.message_queue: List[NetworkMessage] = []

    def add_peer(self, peer_id: str, peer_info: Any) -> None:
        """Add a peer to the network."""
        self.peers[peer_id] = peer_info
        self.protocol.connected_peers[peer_id] = peer_info.get("public_key")

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the network."""
        if peer_id in self.peers:
            del self.peers[peer_id]
        if peer_id in self.protocol.connected_peers:
            del self.protocol.connected_peers[peer_id]

    def send_message_to_peer(self, peer_id: str, message: NetworkMessage) -> bool:
        """Send a message to a specific peer."""
        return self.protocol.send_message(peer_id, message)

    def broadcast_message(self, message: NetworkMessage) -> int:
        """Broadcast a message to all peers."""
        sent_count = 0
        for peer_id in self.peers:
            if self.protocol.send_message(peer_id, message):
                sent_count += 1
        return sent_count

    def process_message(self, message_bytes: bytes) -> Optional[NetworkMessage]:
        """Process an incoming message."""
        message = self.protocol.receive_message(message_bytes)
        if message:
            self.router.route_message(message)
        return message

    def get_peer_count(self) -> int:
        """Get the number of connected peers."""
        return len(self.peers)

    def get_peer_list(self) -> List[str]:
        """Get list of peer IDs."""
        return list(self.peers.keys())
