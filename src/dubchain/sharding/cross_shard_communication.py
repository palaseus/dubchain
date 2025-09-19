"""
Cross-shard communication system for DubChain.

This module provides sophisticated cross-shard communication including:
- Cross-shard messaging protocols
- Shard routing and discovery
- Message relay and validation
- Cross-shard transaction coordination
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .shard_types import CrossShardTransaction, ShardId, ShardMetrics, ShardState


class MessageType(Enum):
    """Types of cross-shard messages."""

    TRANSACTION = "transaction"
    STATE_UPDATE = "state_update"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    HEARTBEAT = "heartbeat"


@dataclass
class CrossShardMessage:
    """Cross-shard message structure."""

    message_id: str
    message_type: MessageType
    source_shard: ShardId
    target_shard: ShardId
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    nonce: int = 0
    ttl: int = 300  # Time to live in seconds

    def calculate_hash(self) -> str:
        """Calculate message hash."""
        data_string = f"{self.message_id}{self.message_type.value}{self.source_shard.value}{self.target_shard.value}{self.timestamp}{self.nonce}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def is_expired(self) -> bool:
        """Check if message is expired."""
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_shard": self.source_shard.value,
            "target_shard": self.target_shard.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "nonce": self.nonce,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossShardMessage":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            source_shard=ShardId(data["source_shard"]),
            target_shard=ShardId(data["target_shard"]),
            payload=data["payload"],
            timestamp=data["timestamp"],
            signature=data.get("signature"),
            nonce=data["nonce"],
            ttl=data["ttl"],
        )


@dataclass
class MessageRelay:
    """Relays messages between shards."""

    relay_nodes: Dict[ShardId, List[str]] = field(default_factory=dict)
    message_queue: List[CrossShardMessage] = field(default_factory=list)
    processed_messages: Set[str] = field(default_factory=set)
    relay_metrics: Dict[str, int] = field(default_factory=dict)

    def add_relay_node(self, shard_id: ShardId, node_id: str) -> None:
        """Add relay node for a shard."""
        if shard_id not in self.relay_nodes:
            self.relay_nodes[shard_id] = []

        if node_id not in self.relay_nodes[shard_id]:
            self.relay_nodes[shard_id].append(node_id)

    def remove_relay_node(self, shard_id: ShardId, node_id: str) -> None:
        """Remove relay node for a shard."""
        if shard_id in self.relay_nodes and node_id in self.relay_nodes[shard_id]:
            self.relay_nodes[shard_id].remove(node_id)

    def queue_message(self, message: CrossShardMessage) -> bool:
        """Queue message for relay."""
        if message.message_id in self.processed_messages:
            return False

        if message.is_expired():
            return False

        self.message_queue.append(message)
        return True

    def process_messages(self) -> List[CrossShardMessage]:
        """Process queued messages."""
        processed = []
        remaining = []

        for message in self.message_queue:
            if message.is_expired():
                continue

            if message.message_id in self.processed_messages:
                continue

            # Check if we have relay nodes for target shard
            if (
                message.target_shard in self.relay_nodes
                and self.relay_nodes[message.target_shard]
            ):
                processed.append(message)
                self.processed_messages.add(message.message_id)
                self.relay_metrics["messages_relayed"] = (
                    self.relay_metrics.get("messages_relayed", 0) + 1
                )
            else:
                remaining.append(message)

        self.message_queue = remaining
        return processed

    def get_relay_metrics(self) -> Dict[str, Any]:
        """Get relay metrics."""
        return {
            "queued_messages": len(self.message_queue),
            "processed_messages": len(self.processed_messages),
            "relay_nodes": {
                shard_id.value: len(nodes)
                for shard_id, nodes in self.relay_nodes.items()
            },
            "metrics": self.relay_metrics,
        }


@dataclass
class ShardRouter:
    """Routes messages between shards."""

    routing_table: Dict[ShardId, List[ShardId]] = field(default_factory=dict)
    connection_matrix: Dict[Tuple[ShardId, ShardId], float] = field(
        default_factory=dict
    )
    routing_metrics: Dict[str, int] = field(default_factory=dict)

    def add_route(
        self,
        source_shard: ShardId,
        target_shard: ShardId,
        intermediate_shards: List[ShardId],
    ) -> None:
        """Add route between shards."""
        if source_shard not in self.routing_table:
            self.routing_table[source_shard] = []

        # Add direct route
        if target_shard not in self.routing_table[source_shard]:
            self.routing_table[source_shard].append(target_shard)

        # Add intermediate routes
        for intermediate in intermediate_shards:
            if intermediate not in self.routing_table[source_shard]:
                self.routing_table[source_shard].append(intermediate)

    def find_route(self, source_shard: ShardId, target_shard: ShardId) -> List[ShardId]:
        """Find route between shards."""
        if source_shard == target_shard:
            return [source_shard]

        # Direct route
        if (
            source_shard in self.routing_table
            and target_shard in self.routing_table[source_shard]
        ):
            return [source_shard, target_shard]

        # Find shortest path using BFS
        queue = [(source_shard, [source_shard])]
        visited = {source_shard}

        while queue:
            current_shard, path = queue.pop(0)

            if current_shard in self.routing_table:
                for next_shard in self.routing_table[current_shard]:
                    if next_shard == target_shard:
                        return path + [target_shard]

                    if next_shard not in visited:
                        visited.add(next_shard)
                        queue.append((next_shard, path + [next_shard]))

        return []  # No route found

    def update_connection_quality(
        self, source_shard: ShardId, target_shard: ShardId, quality: float
    ) -> None:
        """Update connection quality between shards."""
        self.connection_matrix[(source_shard, target_shard)] = quality

    def get_connection_quality(
        self, source_shard: ShardId, target_shard: ShardId
    ) -> float:
        """Get connection quality between shards."""
        return self.connection_matrix.get((source_shard, target_shard), 0.0)

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        return {
            "routing_table_size": len(self.routing_table),
            "connection_count": len(self.connection_matrix),
            "metrics": self.routing_metrics,
        }


@dataclass
class CrossShardValidator:
    """Validates cross-shard transactions and messages."""

    validation_rules: Dict[str, Any] = field(default_factory=dict)
    validation_cache: Dict[str, bool] = field(default_factory=dict)
    validation_metrics: Dict[str, int] = field(default_factory=dict)

    def validate_transaction(
        self,
        transaction: CrossShardTransaction,
        source_shard: ShardState,
        target_shard: ShardState,
    ) -> bool:
        """Validate cross-shard transaction."""
        # Check if transaction is already validated
        tx_hash = transaction.calculate_hash()
        if tx_hash in self.validation_cache:
            return self.validation_cache[tx_hash]

        # Basic validation rules
        if transaction.amount <= 0:
            self.validation_cache[tx_hash] = False
            return False

        if transaction.gas_limit <= 0:
            self.validation_cache[tx_hash] = False
            return False

        if transaction.source_shard == transaction.target_shard:
            self.validation_cache[tx_hash] = False
            return False

        # Check shard states
        if (
            source_shard.status.value != "active"
            or target_shard.status.value != "active"
        ):
            self.validation_cache[tx_hash] = False
            return False

        # Validate transaction data
        if not self._validate_transaction_data(transaction):
            self.validation_cache[tx_hash] = False
            return False

        self.validation_cache[tx_hash] = True
        self.validation_metrics["validated_transactions"] = (
            self.validation_metrics.get("validated_transactions", 0) + 1
        )
        return True

    def validate_message(self, message: CrossShardMessage) -> bool:
        """Validate cross-shard message."""
        # Check message expiration
        if message.is_expired():
            return False

        # Validate message structure
        if not message.message_id or not message.payload:
            return False

        # Validate message hash
        expected_hash = message.calculate_hash()
        if message.signature and not self._verify_signature(message, expected_hash):
            return False

        self.validation_metrics["validated_messages"] = (
            self.validation_metrics.get("validated_messages", 0) + 1
        )
        return True

    def _validate_transaction_data(self, transaction: CrossShardTransaction) -> bool:
        """Validate transaction data."""
        # Check required fields
        if not all(
            [transaction.sender, transaction.receiver, transaction.transaction_id]
        ):
            return False

        # Check data size
        if len(transaction.data) > 1024 * 1024:  # 1MB limit
            return False

        return True

    def _verify_signature(self, message: CrossShardMessage, expected_hash: str) -> bool:
        """Verify message signature."""
        # Simplified signature verification
        # In practice, this would use cryptographic verification
        return message.signature is not None and len(message.signature) > 0

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            "cache_size": len(self.validation_cache),
            "metrics": self.validation_metrics,
        }


class CrossShardMessaging:
    """Main cross-shard messaging system."""

    def __init__(self):
        """Initialize cross-shard messaging system."""
        self.relay = MessageRelay()
        self.router = ShardRouter()
        self.validator = CrossShardValidator()
        self.message_handlers: Dict[MessageType, callable] = {}
        self.active_connections: Dict[Tuple[ShardId, ShardId], bool] = {}

    def register_message_handler(
        self, message_type: MessageType, handler: callable
    ) -> None:
        """Register message handler for specific message type."""
        self.message_handlers[message_type] = handler

    def send_message(self, message: CrossShardMessage) -> bool:
        """Send cross-shard message."""
        # Validate message
        if not self.validator.validate_message(message):
            return False

        # Find route
        route = self.router.find_route(message.source_shard, message.target_shard)
        if not route:
            return False

        # Queue message for relay
        return self.relay.queue_message(message)

    def process_cross_shard_transaction(
        self,
        transaction: CrossShardTransaction,
        source_shard: ShardState,
        target_shard: ShardState,
    ) -> bool:
        """Process cross-shard transaction."""
        # Validate transaction
        if not self.validator.validate_transaction(
            transaction, source_shard, target_shard
        ):
            return False

        # Create cross-shard message
        message = CrossShardMessage(
            message_id=f"tx_{transaction.transaction_id}",
            message_type=MessageType.TRANSACTION,
            source_shard=transaction.source_shard,
            target_shard=transaction.target_shard,
            payload=transaction.to_dict(),
        )

        # Send message
        return self.send_message(message)

    def handle_message(self, message: CrossShardMessage) -> bool:
        """Handle incoming cross-shard message."""
        if message.message_type not in self.message_handlers:
            return False

        handler = self.message_handlers[message.message_type]
        return handler(message)

    def process_messages(self) -> List[CrossShardMessage]:
        """Process all queued messages."""
        processed_messages = self.relay.process_messages()

        for message in processed_messages:
            self.handle_message(message)

        return processed_messages

    def establish_connection(self, shard1: ShardId, shard2: ShardId) -> bool:
        """Establish connection between shards."""
        self.active_connections[(shard1, shard2)] = True
        self.active_connections[(shard2, shard1)] = True

        # Add route
        self.router.add_route(shard1, shard2, [])

        return True

    def break_connection(self, shard1: ShardId, shard2: ShardId) -> bool:
        """Break connection between shards."""
        if (shard1, shard2) in self.active_connections:
            del self.active_connections[(shard1, shard2)]
        if (shard2, shard1) in self.active_connections:
            del self.active_connections[(shard2, shard1)]

        return True

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            "relay_metrics": self.relay.get_relay_metrics(),
            "routing_metrics": self.router.get_routing_metrics(),
            "validation_metrics": self.validator.get_validation_metrics(),
            "active_connections": len(self.active_connections),
            "registered_handlers": len(self.message_handlers),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relay": {
                "relay_nodes": {
                    shard_id.value: nodes
                    for shard_id, nodes in self.relay.relay_nodes.items()
                },
                "message_queue": [msg.to_dict() for msg in self.relay.message_queue],
                "processed_messages": list(self.relay.processed_messages),
                "relay_metrics": self.relay.relay_metrics,
            },
            "router": {
                "routing_table": {
                    shard_id.value: [s.value for s in targets]
                    for shard_id, targets in self.router.routing_table.items()
                },
                "connection_matrix": {
                    f"{s1.value}_{s2.value}": quality
                    for (s1, s2), quality in self.router.connection_matrix.items()
                },
                "routing_metrics": self.router.routing_metrics,
            },
            "validator": {
                "validation_rules": self.validator.validation_rules,
                "validation_cache": self.validator.validation_cache,
                "validation_metrics": self.validator.validation_metrics,
            },
            "active_connections": {
                f"{s1.value}_{s2.value}": active
                for (s1, s2), active in self.active_connections.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossShardMessaging":
        """Create from dictionary."""
        messaging = cls()

        # Restore relay
        relay_data = data["relay"]
        messaging.relay.relay_nodes = {
            ShardId(int(k)): v for k, v in relay_data["relay_nodes"].items()
        }
        messaging.relay.message_queue = [
            CrossShardMessage.from_dict(msg_data)
            for msg_data in relay_data["message_queue"]
        ]
        messaging.relay.processed_messages = set(relay_data["processed_messages"])
        messaging.relay.relay_metrics = relay_data["relay_metrics"]

        # Restore router
        router_data = data["router"]
        messaging.router.routing_table = {
            ShardId(int(k)): [ShardId(int(s)) for s in v]
            for k, v in router_data["routing_table"].items()
        }
        messaging.router.connection_matrix = {
            (ShardId(int(s1)), ShardId(int(s2))): quality
            for (s1, s2), quality in router_data["connection_matrix"].items()
        }
        messaging.router.routing_metrics = router_data["routing_metrics"]

        # Restore validator
        validator_data = data["validator"]
        messaging.validator.validation_rules = validator_data["validation_rules"]
        messaging.validator.validation_cache = validator_data["validation_cache"]
        messaging.validator.validation_metrics = validator_data["validation_metrics"]

        # Restore connections
        messaging.active_connections = {
            (ShardId(int(s1)), ShardId(int(s2))): active
            for (s1, s2), active in data["active_connections"].items()
        }

        return messaging
