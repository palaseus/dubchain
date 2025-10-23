"""
Tests for sharding cross-shard communication module.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.dubchain.sharding.cross_shard_communication import (
    CrossShardMessage,
    CrossShardMessaging,
    CrossShardValidator,
    MessageRelay,
    MessageType,
    ShardRouter,
)
from src.dubchain.sharding.shard_types import (
    CrossShardTransaction,
    ShardId,
    ShardState,
    ShardStatus,
    ShardType,
)


class TestCrossShardMessage:
    """Test CrossShardMessage class."""

    def test_cross_shard_message_creation(self):
        """Test creating a cross-shard message."""
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100, "recipient": "alice"},
            timestamp=time.time(),
        )

        assert message.message_id == "msg_001"
        assert message.message_type == MessageType.TRANSACTION
        assert message.source_shard == ShardId.SHARD_1
        assert message.target_shard == ShardId.SHARD_2
        assert message.payload == {"amount": 100, "recipient": "alice"}
        assert message.timestamp > 0
        assert message.signature is None
        assert message.nonce == 0
        assert message.ttl == 300

    def test_cross_shard_message_calculate_hash(self):
        """Test message hash calculation."""
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        hash1 = message.calculate_hash()
        hash2 = message.calculate_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert isinstance(hash1, str)

    def test_cross_shard_message_is_expired(self):
        """Test message expiration check."""
        # Create message with very short TTL
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
            ttl=0.1,  # 100ms TTL
        )

        # Should not be expired immediately
        assert message.is_expired() is False

        # Wait for expiration
        time.sleep(0.2)
        assert message.is_expired() is True

    def test_cross_shard_message_to_dict(self):
        """Test message to dictionary conversion."""
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=1234567890.0,
            signature="0x1234",
            nonce=5,
            ttl=300,
        )

        data = message.to_dict()

        assert data["message_id"] == "msg_001"
        assert data["message_type"] == "transaction"
        assert data["source_shard"] == 1
        assert data["target_shard"] == 2
        assert data["payload"] == {"amount": 100}
        assert data["timestamp"] == 1234567890.0
        assert data["signature"] == "0x1234"
        assert data["nonce"] == 5
        assert data["ttl"] == 300

    def test_cross_shard_message_from_dict(self):
        """Test message creation from dictionary."""
        data = {
            "message_id": "msg_001",
            "message_type": "transaction",
            "source_shard": 1,
            "target_shard": 2,
            "payload": {"amount": 100},
            "timestamp": 1234567890.0,
            "signature": "0x1234",
            "nonce": 5,
            "ttl": 300,
        }

        message = CrossShardMessage.from_dict(data)

        assert message.message_id == "msg_001"
        assert message.message_type == MessageType.TRANSACTION
        assert message.source_shard == ShardId.SHARD_1
        assert message.target_shard == ShardId.SHARD_2
        assert message.payload == {"amount": 100}
        assert message.timestamp == 1234567890.0
        assert message.signature == "0x1234"
        assert message.nonce == 5
        assert message.ttl == 300


class TestMessageRelay:
    """Test MessageRelay class."""

    def test_message_relay_initialization(self):
        """Test message relay initialization."""
        relay = MessageRelay()

        assert relay.relay_nodes == {}
        assert relay.message_queue == []
        assert relay.processed_messages == set()
        assert relay.relay_metrics == {}

    def test_add_relay_node(self):
        """Test adding relay node."""
        relay = MessageRelay()

        relay.add_relay_node(ShardId.SHARD_1, "node_001")
        relay.add_relay_node(ShardId.SHARD_1, "node_002")
        relay.add_relay_node(ShardId.SHARD_2, "node_003")

        assert ShardId.SHARD_1 in relay.relay_nodes
        assert ShardId.SHARD_2 in relay.relay_nodes
        assert "node_001" in relay.relay_nodes[ShardId.SHARD_1]
        assert "node_002" in relay.relay_nodes[ShardId.SHARD_1]
        assert "node_003" in relay.relay_nodes[ShardId.SHARD_2]
        assert len(relay.relay_nodes[ShardId.SHARD_1]) == 2
        assert len(relay.relay_nodes[ShardId.SHARD_2]) == 1

    def test_add_duplicate_relay_node(self):
        """Test adding duplicate relay node."""
        relay = MessageRelay()

        relay.add_relay_node(ShardId.SHARD_1, "node_001")
        relay.add_relay_node(ShardId.SHARD_1, "node_001")  # Duplicate

        assert len(relay.relay_nodes[ShardId.SHARD_1]) == 1
        assert "node_001" in relay.relay_nodes[ShardId.SHARD_1]

    def test_remove_relay_node(self):
        """Test removing relay node."""
        relay = MessageRelay()

        relay.add_relay_node(ShardId.SHARD_1, "node_001")
        relay.add_relay_node(ShardId.SHARD_1, "node_002")

        relay.remove_relay_node(ShardId.SHARD_1, "node_001")

        assert "node_001" not in relay.relay_nodes[ShardId.SHARD_1]
        assert "node_002" in relay.relay_nodes[ShardId.SHARD_1]
        assert len(relay.relay_nodes[ShardId.SHARD_1]) == 1

    def test_remove_nonexistent_relay_node(self):
        """Test removing non-existent relay node."""
        relay = MessageRelay()

        relay.add_relay_node(ShardId.SHARD_1, "node_001")
        relay.remove_relay_node(ShardId.SHARD_1, "node_002")  # Non-existent

        assert len(relay.relay_nodes[ShardId.SHARD_1]) == 1
        assert "node_001" in relay.relay_nodes[ShardId.SHARD_1]

    def test_queue_message(self):
        """Test queuing a message."""
        relay = MessageRelay()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = relay.queue_message(message)
        assert result is True
        assert len(relay.message_queue) == 1
        assert relay.message_queue[0] == message

    def test_queue_duplicate_message(self):
        """Test queuing duplicate message."""
        relay = MessageRelay()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        # Queue first time
        result1 = relay.queue_message(message)
        assert result1 is True

        # Queue second time (duplicate) - should still work as the message is not processed yet
        result2 = relay.queue_message(message)
        assert result2 is True  # The actual implementation allows this
        assert len(relay.message_queue) == 2

    def test_queue_expired_message(self):
        """Test queuing expired message."""
        relay = MessageRelay()

        # Create expired message
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time() - 400,  # 400 seconds ago
            ttl=300,  # 300 second TTL
        )

        result = relay.queue_message(message)
        assert result is False
        assert len(relay.message_queue) == 0

    def test_process_messages(self):
        """Test processing messages."""
        relay = MessageRelay()

        # Add relay node for target shard
        relay.add_relay_node(ShardId.SHARD_2, "node_001")

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        relay.queue_message(message)

        processed = relay.process_messages()

        assert len(processed) == 1
        assert processed[0] == message
        assert "msg_001" in relay.processed_messages
        assert len(relay.message_queue) == 0
        assert relay.relay_metrics["messages_relayed"] == 1

    def test_process_messages_no_relay_node(self):
        """Test processing messages with no relay node."""
        relay = MessageRelay()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        relay.queue_message(message)

        processed = relay.process_messages()

        assert len(processed) == 0
        assert len(relay.message_queue) == 1
        assert "msg_001" not in relay.processed_messages

    def test_get_relay_metrics(self):
        """Test getting relay metrics."""
        relay = MessageRelay()

        relay.add_relay_node(ShardId.SHARD_1, "node_001")
        relay.add_relay_node(ShardId.SHARD_2, "node_002")

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        relay.queue_message(message)
        relay.processed_messages.add("msg_002")
        relay.relay_metrics["messages_relayed"] = 5

        metrics = relay.get_relay_metrics()

        assert metrics["queued_messages"] == 1
        assert metrics["processed_messages"] == 1
        assert metrics["relay_nodes"][1] == 1  # ShardId.SHARD_1 = 1
        assert metrics["relay_nodes"][2] == 1  # ShardId.SHARD_2 = 2
        assert metrics["metrics"]["messages_relayed"] == 5


class TestShardRouter:
    """Test ShardRouter class."""

    def test_shard_router_initialization(self):
        """Test shard router initialization."""
        router = ShardRouter()

        assert router.routing_table == {}
        assert router.connection_matrix == {}
        assert router.routing_metrics == {}

    def test_add_route(self):
        """Test adding a route."""
        router = ShardRouter()

        router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])
        router.add_route(ShardId.SHARD_1, ShardId.SHARD_3, [ShardId.SHARD_2])

        assert ShardId.SHARD_1 in router.routing_table
        assert ShardId.SHARD_2 in router.routing_table[ShardId.SHARD_1]
        assert ShardId.SHARD_3 in router.routing_table[ShardId.SHARD_1]
        assert (
            ShardId.SHARD_2 in router.routing_table[ShardId.SHARD_1]
        )  # Intermediate shard

    def test_find_route_direct(self):
        """Test finding direct route."""
        router = ShardRouter()

        router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])

        route = router.find_route(ShardId.SHARD_1, ShardId.SHARD_2)
        assert route == [ShardId.SHARD_1, ShardId.SHARD_2]

    def test_find_route_same_shard(self):
        """Test finding route to same shard."""
        router = ShardRouter()

        route = router.find_route(ShardId.SHARD_1, ShardId.SHARD_1)
        assert route == [ShardId.SHARD_1]

    def test_find_route_no_route(self):
        """Test finding route when none exists."""
        router = ShardRouter()

        route = router.find_route(ShardId.SHARD_1, ShardId.SHARD_2)
        assert route == []

    def test_find_route_multi_hop(self):
        """Test finding multi-hop route."""
        router = ShardRouter()

        # Create multi-hop route: SHARD_1 -> SHARD_2 -> SHARD_3
        router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])
        router.add_route(ShardId.SHARD_2, ShardId.SHARD_3, [])

        route = router.find_route(ShardId.SHARD_1, ShardId.SHARD_3)
        assert route == [ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3]

    def test_update_connection_quality(self):
        """Test updating connection quality."""
        router = ShardRouter()

        router.update_connection_quality(ShardId.SHARD_1, ShardId.SHARD_2, 0.8)

        quality = router.get_connection_quality(ShardId.SHARD_1, ShardId.SHARD_2)
        assert quality == 0.8

    def test_get_connection_quality_default(self):
        """Test getting connection quality for non-existent connection."""
        router = ShardRouter()

        quality = router.get_connection_quality(ShardId.SHARD_1, ShardId.SHARD_2)
        assert quality == 0.0

    def test_get_routing_metrics(self):
        """Test getting routing metrics."""
        router = ShardRouter()

        router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])
        router.update_connection_quality(ShardId.SHARD_1, ShardId.SHARD_2, 0.8)
        router.routing_metrics["routes_used"] = 10

        metrics = router.get_routing_metrics()

        assert metrics["routing_table_size"] == 1
        assert metrics["connection_count"] == 1
        assert metrics["metrics"]["routes_used"] == 10


class TestCrossShardValidator:
    """Test CrossShardValidator class."""

    def test_cross_shard_validator_initialization(self):
        """Test cross-shard validator initialization."""
        validator = CrossShardValidator()

        assert validator.validation_rules == {}
        assert validator.validation_cache == {}
        assert validator.validation_metrics == {}

    def test_validate_transaction_valid(self):
        """Test validating valid transaction."""
        validator = CrossShardValidator()

        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"hello",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice", "bob"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        target_shard = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["charlie", "dave"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        result = validator.validate_transaction(transaction, source_shard, target_shard)
        assert result is True
        assert validator.validation_metrics["validated_transactions"] == 1

    def test_validate_transaction_invalid_amount(self):
        """Test validating transaction with invalid amount."""
        validator = CrossShardValidator()

        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            amount=-100,  # Invalid negative amount
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        target_shard = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["bob"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        result = validator.validate_transaction(transaction, source_shard, target_shard)
        assert result is False

    def test_validate_transaction_same_shard(self):
        """Test validating transaction with same source and target shard."""
        validator = CrossShardValidator()

        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_1,  # Same shard
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        result = validator.validate_transaction(transaction, source_shard, source_shard)
        assert result is False

    def test_validate_transaction_inactive_shard(self):
        """Test validating transaction with inactive shard."""
        validator = CrossShardValidator()

        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.INACTIVE,  # Inactive shard
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        target_shard = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["bob"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        result = validator.validate_transaction(transaction, source_shard, target_shard)
        assert result is False

    def test_validate_message_valid(self):
        """Test validating valid message."""
        validator = CrossShardValidator()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
            signature="0x1234",
        )

        result = validator.validate_message(message)
        assert result is True
        assert validator.validation_metrics["validated_messages"] == 1

    def test_validate_message_expired(self):
        """Test validating expired message."""
        validator = CrossShardValidator()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time() - 400,  # 400 seconds ago
            ttl=300,  # 300 second TTL
        )

        result = validator.validate_message(message)
        assert result is False

    def test_validate_message_invalid_structure(self):
        """Test validating message with invalid structure."""
        validator = CrossShardValidator()

        message = CrossShardMessage(
            message_id="",  # Empty message ID
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={},  # Empty payload
            timestamp=time.time(),
        )

        result = validator.validate_message(message)
        assert result is False

    def test_validation_cache(self):
        """Test validation caching."""
        validator = CrossShardValidator()

        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        target_shard = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["bob"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        # First validation
        result1 = validator.validate_transaction(
            transaction, source_shard, target_shard
        )
        assert result1 is True

        # Second validation (should use cache)
        result2 = validator.validate_transaction(
            transaction, source_shard, target_shard
        )
        assert result2 is True

        # Should only have one validation in metrics (cached)
        assert validator.validation_metrics["validated_transactions"] == 1
        assert len(validator.validation_cache) == 1

    def test_get_validation_metrics(self):
        """Test getting validation metrics."""
        validator = CrossShardValidator()

        validator.validation_cache["hash1"] = True
        validator.validation_cache["hash2"] = False
        validator.validation_metrics["validated_transactions"] = 5
        validator.validation_metrics["validated_messages"] = 3

        metrics = validator.get_validation_metrics()

        assert metrics["cache_size"] == 2
        assert metrics["metrics"]["validated_transactions"] == 5
        assert metrics["metrics"]["validated_messages"] == 3


class TestCrossShardMessaging:
    """Test CrossShardMessaging class."""

    def test_cross_shard_messaging_initialization(self):
        """Test cross-shard messaging initialization."""
        messaging = CrossShardMessaging()

        assert messaging.relay is not None
        assert messaging.router is not None
        assert messaging.validator is not None
        assert messaging.message_handlers == {}
        assert messaging.active_connections == {}

    def test_register_message_handler(self):
        """Test registering message handler."""
        messaging = CrossShardMessaging()

        def handler(message):
            return True

        messaging.register_message_handler(MessageType.TRANSACTION, handler)

        assert MessageType.TRANSACTION in messaging.message_handlers
        assert messaging.message_handlers[MessageType.TRANSACTION] == handler

    def test_send_message_valid(self):
        """Test sending valid message."""
        messaging = CrossShardMessaging()

        # Add route
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.send_message(message)
        assert result is True
        assert len(messaging.relay.message_queue) == 1

    def test_send_message_invalid(self):
        """Test sending invalid message."""
        messaging = CrossShardMessaging()

        # Create invalid message (empty message ID)
        message = CrossShardMessage(
            message_id="",  # Invalid
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.send_message(message)
        assert result is False
        assert len(messaging.relay.message_queue) == 0

    def test_send_message_no_route(self):
        """Test sending message with no route."""
        messaging = CrossShardMessaging()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.send_message(message)
        assert result is False
        assert len(messaging.relay.message_queue) == 0

    def test_process_cross_shard_transaction(self):
        """Test processing cross-shard transaction."""
        messaging = CrossShardMessaging()

        # Add route
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])

        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        target_shard = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["bob"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        result = messaging.process_cross_shard_transaction(
            transaction, source_shard, target_shard
        )
        assert result is True
        assert len(messaging.relay.message_queue) == 1

    def test_handle_message(self):
        """Test handling message."""
        messaging = CrossShardMessaging()

        def handler(message):
            return message.message_id == "msg_001"

        messaging.register_message_handler(MessageType.TRANSACTION, handler)

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.handle_message(message)
        assert result is True

    def test_handle_message_no_handler(self):
        """Test handling message with no handler."""
        messaging = CrossShardMessaging()

        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.handle_message(message)
        assert result is False

    def test_process_messages(self):
        """Test processing messages."""
        messaging = CrossShardMessaging()

        # Add relay node
        messaging.relay.add_relay_node(ShardId.SHARD_2, "node_001")

        # Add route
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])

        # Register handler
        def handler(message):
            return True

        messaging.register_message_handler(MessageType.TRANSACTION, handler)

        # Send message
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        messaging.send_message(message)

        # Process messages
        processed = messaging.process_messages()

        assert len(processed) == 1
        assert processed[0] == message

    def test_establish_connection(self):
        """Test establishing connection."""
        messaging = CrossShardMessaging()

        result = messaging.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert result is True
        assert (ShardId.SHARD_1, ShardId.SHARD_2) in messaging.active_connections
        assert (ShardId.SHARD_2, ShardId.SHARD_1) in messaging.active_connections
        assert messaging.active_connections[(ShardId.SHARD_1, ShardId.SHARD_2)] is True
        assert messaging.active_connections[(ShardId.SHARD_2, ShardId.SHARD_1)] is True

    def test_break_connection(self):
        """Test breaking connection."""
        messaging = CrossShardMessaging()

        # Establish connection first
        messaging.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        # Break connection
        result = messaging.break_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert result is True
        assert (ShardId.SHARD_1, ShardId.SHARD_2) not in messaging.active_connections
        assert (ShardId.SHARD_2, ShardId.SHARD_1) not in messaging.active_connections

    def test_get_system_metrics(self):
        """Test getting system metrics."""
        messaging = CrossShardMessaging()

        # Add some data
        messaging.relay.add_relay_node(ShardId.SHARD_1, "node_001")
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])
        messaging.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        def handler(message):
            return True

        messaging.register_message_handler(MessageType.TRANSACTION, handler)

        metrics = messaging.get_system_metrics()

        assert "relay_metrics" in metrics
        assert "routing_metrics" in metrics
        assert "validation_metrics" in metrics
        assert metrics["active_connections"] == 2  # Bidirectional
        assert metrics["registered_handlers"] == 1

    def test_to_dict(self):
        """Test converting to dictionary."""
        messaging = CrossShardMessaging()

        # Add some data
        messaging.relay.add_relay_node(ShardId.SHARD_1, "node_001")
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])
        messaging.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        data = messaging.to_dict()

        assert "relay" in data
        assert "router" in data
        assert "validator" in data
        assert "active_connections" in data
        assert data["relay"]["relay_nodes"][1] == ["node_001"]
        assert data["router"]["routing_table"][1] == [2]
        assert data["active_connections"]["1_2"] is True

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "relay": {
                "relay_nodes": {1: ["node_001"], 2: ["node_002"]},
                "message_queue": [],
                "processed_messages": ["msg_001"],
                "relay_metrics": {"messages_relayed": 5},
            },
            "router": {
                "routing_table": {1: [2], 2: [3]},
                "connection_matrix": {("1", "2"): 0.8, ("2", "3"): 0.9},
                "routing_metrics": {"routes_used": 10},
            },
            "validator": {
                "validation_rules": {"rule1": "value1"},
                "validation_cache": {"hash1": True},
                "validation_metrics": {"validated_transactions": 3},
            },
            "active_connections": {("1", "2"): True, ("2", "1"): True},
        }

        messaging = CrossShardMessaging.from_dict(data)

        assert len(messaging.relay.relay_nodes[ShardId.SHARD_1]) == 1
        assert len(messaging.relay.relay_nodes[ShardId.SHARD_2]) == 1
        assert "msg_001" in messaging.relay.processed_messages
        assert messaging.relay.relay_metrics["messages_relayed"] == 5

        assert ShardId.SHARD_2 in messaging.router.routing_table[ShardId.SHARD_1]
        assert (
            messaging.router.connection_matrix[(ShardId.SHARD_1, ShardId.SHARD_2)]
            == 0.8
        )

        assert messaging.validator.validation_rules["rule1"] == "value1"
        assert messaging.validator.validation_cache["hash1"] is True

        assert (ShardId.SHARD_1, ShardId.SHARD_2) in messaging.active_connections
        assert (ShardId.SHARD_2, ShardId.SHARD_1) in messaging.active_connections


class TestCrossShardIntegration:
    """Test cross-shard communication integration."""

    def test_full_message_flow(self):
        """Test complete message flow."""
        messaging = CrossShardMessaging()

        # Setup
        messaging.relay.add_relay_node(ShardId.SHARD_2, "node_001")
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])

        # Register handler
        received_messages = []

        def handler(message):
            received_messages.append(message)
            return True

        messaging.register_message_handler(MessageType.TRANSACTION, handler)

        # Send message
        message = CrossShardMessage(
            message_id="msg_001",
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.send_message(message)
        assert result is True

        # Process messages
        processed = messaging.process_messages()

        assert len(processed) == 1
        assert len(received_messages) == 1
        assert received_messages[0] == message

    def test_full_transaction_flow(self):
        """Test complete transaction flow."""
        messaging = CrossShardMessaging()

        # Setup
        messaging.relay.add_relay_node(ShardId.SHARD_2, "node_001")
        messaging.router.add_route(ShardId.SHARD_1, ShardId.SHARD_2, [])

        # Create transaction
        transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        target_shard = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["bob"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        # Process transaction
        result = messaging.process_cross_shard_transaction(
            transaction, source_shard, target_shard
        )
        assert result is True

        # Process messages
        processed = messaging.process_messages()
        assert len(processed) == 1

        # Verify message content
        processed_message = processed[0]
        assert processed_message.message_type == MessageType.TRANSACTION
        assert processed_message.payload["transaction_id"] == "tx_001"
        assert processed_message.payload["amount"] == 100

    def test_error_handling(self):
        """Test error handling in messaging system."""
        messaging = CrossShardMessaging()

        # Test sending invalid message
        invalid_message = CrossShardMessage(
            message_id="",  # Invalid
            message_type=MessageType.TRANSACTION,
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_2,
            payload={"amount": 100},
            timestamp=time.time(),
        )

        result = messaging.send_message(invalid_message)
        assert result is False

        # Test processing invalid transaction
        invalid_transaction = CrossShardTransaction(
            transaction_id="tx_001",
            source_shard=ShardId.SHARD_1,
            target_shard=ShardId.SHARD_1,  # Same shard - invalid
            amount=100,
            sender="alice",
            receiver="bob",
            data=b"",
            gas_limit=21000,
            gas_price=20,
            timestamp=time.time(),
        )

        source_shard = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["alice"],
            current_epoch=1,
            last_block_number=100,
            last_block_hash="0xabc123",
            state_root="0xdef456",
        )

        result = messaging.process_cross_shard_transaction(
            invalid_transaction, source_shard, source_shard
        )
        assert result is False
