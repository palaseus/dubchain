"""
Unit tests for network gossip module.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from dubchain.network.gossip import (
    GossipConfig,
    GossipMessage,
    GossipProtocol,
    MessageType,
)


class TestMessageType:
    """Test MessageType enum."""

    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.BLOCK.value == "block"
        assert MessageType.TRANSACTION.value == "transaction"
        assert MessageType.PEER_INFO.value == "peer_info"
        assert MessageType.SYNC_REQUEST.value == "sync_request"
        assert MessageType.SYNC_RESPONSE.value == "sync_response"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.ANNOUNCEMENT.value == "announcement"
        assert MessageType.QUERY.value == "query"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.CUSTOM.value == "custom"


class TestGossipMessage:
    """Test GossipMessage class."""

    def test_gossip_message_creation(self):
        """Test GossipMessage creation."""
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
        )

        assert message.message_id == "test_id"
        assert message.message_type == MessageType.BLOCK
        assert message.sender_id == "sender_1"
        assert message.content == {"block": "data"}
        assert message.ttl == 3600
        assert message.hop_count == 0
        assert message.max_hops == 10
        assert message.signature is None
        assert message.metadata == {}

    def test_gossip_message_creation_with_custom_values(self):
        """Test GossipMessage creation with custom values."""
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.TRANSACTION,
            sender_id="sender_1",
            content={"tx": "data"},
            ttl=1800,
            hop_count=2,
            max_hops=5,
            metadata={"priority": "high"},
        )

        assert message.message_id == "test_id"
        assert message.message_type == MessageType.TRANSACTION
        assert message.sender_id == "sender_1"
        assert message.content == {"tx": "data"}
        assert message.ttl == 1800
        assert message.hop_count == 2
        assert message.max_hops == 5
        assert message.metadata == {"priority": "high"}

    def test_gossip_message_validation_empty_message_id(self):
        """Test GossipMessage validation with empty message ID."""
        with pytest.raises(ValueError, match="Message ID cannot be empty"):
            GossipMessage(
                message_id="",
                message_type=MessageType.BLOCK,
                sender_id="sender_1",
                content={"block": "data"},
            )

    def test_gossip_message_validation_empty_sender_id(self):
        """Test GossipMessage validation with empty sender ID."""
        with pytest.raises(ValueError, match="Sender ID cannot be empty"):
            GossipMessage(
                message_id="test_id",
                message_type=MessageType.BLOCK,
                sender_id="",
                content={"block": "data"},
            )

    def test_gossip_message_validation_invalid_ttl(self):
        """Test GossipMessage validation with invalid TTL."""
        with pytest.raises(ValueError, match="TTL must be positive"):
            GossipMessage(
                message_id="test_id",
                message_type=MessageType.BLOCK,
                sender_id="sender_1",
                content={"block": "data"},
                ttl=0,
            )

    def test_gossip_message_validation_invalid_max_hops(self):
        """Test GossipMessage validation with invalid max hops."""
        with pytest.raises(ValueError, match="Max hops must be positive"):
            GossipMessage(
                message_id="test_id",
                message_type=MessageType.BLOCK,
                sender_id="sender_1",
                content={"block": "data"},
                max_hops=0,
            )

    def test_gossip_message_validation_negative_hop_count(self):
        """Test GossipMessage validation with negative hop count."""
        with pytest.raises(ValueError, match="Hop count cannot be negative"):
            GossipMessage(
                message_id="test_id",
                message_type=MessageType.BLOCK,
                sender_id="sender_1",
                content={"block": "data"},
                hop_count=-1,
            )

    @patch("time.time")
    def test_gossip_message_is_expired(self, mock_time):
        """Test GossipMessage is_expired method."""
        mock_time.return_value = 1000

        # Message with TTL of 100 seconds, created at time 900
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            timestamp=900,
            ttl=100,
        )

        # At time 1000, message should not be expired (age = 100, TTL = 100)
        assert not message.is_expired()

        # At time 1001, message should be expired (age = 101, TTL = 100)
        mock_time.return_value = 1001
        assert message.is_expired()

    def test_gossip_message_can_hop(self):
        """Test GossipMessage can_hop method."""
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            hop_count=5,
            max_hops=10,
        )

        assert message.can_hop()

        # Set hop count to max_hops
        message.hop_count = 10
        assert not message.can_hop()

    @patch("time.time")
    def test_gossip_message_can_hop_expired(self, mock_time):
        """Test GossipMessage can_hop method with expired message."""
        mock_time.return_value = 1000

        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            timestamp=900,
            ttl=50,  # Expired
        )

        assert not message.can_hop()

    def test_gossip_message_increment_hop(self):
        """Test GossipMessage increment_hop method."""
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            hop_count=5,
            max_hops=10,
        )

        message.increment_hop()
        assert message.hop_count == 6

        # Try to increment when at max hops
        message.hop_count = 10
        message.increment_hop()
        assert message.hop_count == 10  # Should not increment

    @patch("time.time")
    def test_gossip_message_get_age(self, mock_time):
        """Test GossipMessage get_age method."""
        mock_time.return_value = 1000

        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            timestamp=900,
        )

        assert message.get_age() == 100

    @patch("time.time")
    def test_gossip_message_get_remaining_ttl(self, mock_time):
        """Test GossipMessage get_remaining_ttl method."""
        mock_time.return_value = 1000

        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            timestamp=900,
            ttl=200,
        )

        assert message.get_remaining_ttl() == 100

    def test_gossip_message_get_remaining_ttl_expired(self):
        """Test GossipMessage get_remaining_ttl method with expired message."""
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            timestamp=900,
            ttl=50,
        )

        with patch("time.time", return_value=1000):
            assert message.get_remaining_ttl() == 0

    def test_gossip_message_to_dict(self):
        """Test GossipMessage to_dict method."""
        message = GossipMessage(
            message_id="test_id",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data"},
            ttl=3600,
            hop_count=2,
            max_hops=10,
            metadata={"priority": "high"},
        )

        data = message.to_dict()

        assert data["message_id"] == "test_id"
        assert data["message_type"] == "block"
        assert data["sender_id"] == "sender_1"
        assert data["content"] == {"block": "data"}
        assert data["ttl"] == 3600
        assert data["hop_count"] == 2
        assert data["max_hops"] == 10
        assert data["signature"] is None
        assert data["metadata"] == {"priority": "high"}

    def test_gossip_message_from_dict(self):
        """Test GossipMessage from_dict method."""
        data = {
            "message_id": "test_id",
            "message_type": "block",
            "sender_id": "sender_1",
            "content": {"block": "data"},
            "timestamp": 1000,
            "ttl": 3600,
            "hop_count": 2,
            "max_hops": 10,
            "signature": None,
            "metadata": {"priority": "high"},
        }

        message = GossipMessage.from_dict(data)

        assert message.message_id == "test_id"
        assert message.message_type == MessageType.BLOCK
        assert message.sender_id == "sender_1"
        assert message.content == {"block": "data"}
        assert message.timestamp == 1000
        assert message.ttl == 3600
        assert message.hop_count == 2
        assert message.max_hops == 10
        assert message.signature is None
        assert message.metadata == {"priority": "high"}


class TestGossipConfig:
    """Test GossipConfig class."""

    def test_gossip_config_creation(self):
        """Test GossipConfig creation."""
        config = GossipConfig()

        assert config.fanout == 3
        assert config.interval == 1.0
        assert config.max_messages == 1000
        assert config.message_ttl == 3600
        assert config.max_hops == 10
        assert config.anti_entropy_interval == 60.0
        assert config.push_pull_ratio == 0.5
        assert config.duplicate_detection_window == 300
        assert config.enable_compression is True
        assert config.enable_encryption is True
        assert config.max_message_size == 1024 * 1024
        assert config.metadata == {}

    def test_gossip_config_custom_values(self):
        """Test GossipConfig with custom values."""
        config = GossipConfig(
            fanout=5,
            interval=2.0,
            max_messages=2000,
            message_ttl=7200,
            max_hops=15,
            anti_entropy_interval=120.0,
            push_pull_ratio=0.7,
            duplicate_detection_window=600,
            enable_compression=False,
            enable_encryption=False,
            max_message_size=2048 * 1024,
            metadata={"version": "2.0"},
        )

        assert config.fanout == 5
        assert config.interval == 2.0
        assert config.max_messages == 2000
        assert config.message_ttl == 7200
        assert config.max_hops == 15
        assert config.anti_entropy_interval == 120.0
        assert config.push_pull_ratio == 0.7
        assert config.duplicate_detection_window == 600
        assert config.enable_compression is False
        assert config.enable_encryption is False
        assert config.max_message_size == 2048 * 1024
        assert config.metadata == {"version": "2.0"}

    def test_gossip_config_validation_invalid_fanout(self):
        """Test GossipConfig validation with invalid fanout."""
        with pytest.raises(ValueError, match="Fanout must be positive"):
            GossipConfig(fanout=0)

    def test_gossip_config_validation_invalid_interval(self):
        """Test GossipConfig validation with invalid interval."""
        with pytest.raises(ValueError, match="Interval must be positive"):
            GossipConfig(interval=0)

    def test_gossip_config_validation_invalid_max_messages(self):
        """Test GossipConfig validation with invalid max messages."""
        with pytest.raises(ValueError, match="Max messages must be positive"):
            GossipConfig(max_messages=0)

    def test_gossip_config_validation_invalid_message_ttl(self):
        """Test GossipConfig validation with invalid message TTL."""
        with pytest.raises(ValueError, match="Message TTL must be positive"):
            GossipConfig(message_ttl=0)

    def test_gossip_config_validation_invalid_max_hops(self):
        """Test GossipConfig validation with invalid max hops."""
        with pytest.raises(ValueError, match="Max hops must be positive"):
            GossipConfig(max_hops=0)

    def test_gossip_config_validation_invalid_push_pull_ratio(self):
        """Test GossipConfig validation with invalid push-pull ratio."""
        with pytest.raises(ValueError, match="Push-pull ratio must be between 0 and 1"):
            GossipConfig(push_pull_ratio=1.5)

    def test_gossip_config_validation_invalid_duplicate_detection_window(self):
        """Test GossipConfig validation with invalid duplicate detection window."""
        with pytest.raises(
            ValueError, match="Duplicate detection window must be positive"
        ):
            GossipConfig(duplicate_detection_window=0)

    def test_gossip_config_validation_invalid_max_message_size(self):
        """Test GossipConfig validation with invalid max message size."""
        with pytest.raises(ValueError, match="Max message size must be positive"):
            GossipConfig(max_message_size=0)


class TestGossipProtocol:
    """Test GossipProtocol class."""

    @pytest.fixture
    def gossip_config(self):
        """Create a gossip configuration."""
        return GossipConfig(
            fanout=3, interval=1.0, max_messages=100, message_ttl=3600, max_hops=10
        )

    @pytest.fixture
    def gossip_protocol(self, gossip_config):
        """Create a gossip protocol instance."""
        return GossipProtocol(gossip_config, "test_node")

    def test_gossip_protocol_creation(self, gossip_protocol, gossip_config):
        """Test GossipProtocol creation."""
        protocol = gossip_protocol

        assert protocol.config == gossip_config
        assert protocol.node_id == "test_node"
        assert protocol.peers == {}
        assert protocol.messages == {}
        assert protocol.message_history == []
        assert protocol.duplicate_cache == {}
        assert protocol.message_handlers == {}
        assert protocol.gossip_task is None
        assert protocol.anti_entropy_task is None
        assert protocol.running is False

    def test_gossip_protocol_add_peer(self, gossip_protocol):
        """Test adding a peer to gossip protocol."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"

        gossip_protocol.add_peer(mock_peer)

        assert "peer_1" in gossip_protocol.peers
        assert gossip_protocol.peers["peer_1"] == mock_peer

    def test_gossip_protocol_remove_peer(self, gossip_protocol):
        """Test removing a peer from gossip protocol."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"

        gossip_protocol.add_peer(mock_peer)
        assert "peer_1" in gossip_protocol.peers

        gossip_protocol.remove_peer("peer_1")
        assert "peer_1" not in gossip_protocol.peers

    def test_gossip_protocol_add_message_handler(self, gossip_protocol):
        """Test adding a message handler."""
        handler = Mock()

        gossip_protocol.add_message_handler(MessageType.BLOCK, handler)

        assert MessageType.BLOCK in gossip_protocol.message_handlers
        assert gossip_protocol.message_handlers[MessageType.BLOCK] == handler

    @pytest.mark.asyncio
    async def test_gossip_protocol_broadcast_message(self, gossip_protocol):
        """Test broadcasting a message."""
        content = {"block": "data"}

        with patch.object(
            gossip_protocol, "_store_message"
        ) as mock_store, patch.object(
            gossip_protocol, "_broadcast_to_peers"
        ) as mock_broadcast:
            message_id = await gossip_protocol.broadcast_message(
                MessageType.BLOCK, content
            )

            assert message_id is not None
            mock_store.assert_called_once()
            mock_broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_protocol_send_message_to_peer(self, gossip_protocol):
        """Test sending message to specific peer."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"
        gossip_protocol.add_peer(mock_peer)

        content = {"tx": "data"}

        with patch.object(
            gossip_protocol, "_store_message"
        ) as mock_store, patch.object(
            gossip_protocol, "_send_message_to_peer"
        ) as mock_send:
            mock_send.return_value = True

            result = await gossip_protocol.send_message_to_peer(
                "peer_1", MessageType.TRANSACTION, content
            )

            assert result is True
            mock_store.assert_called_once()
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_protocol_send_message_to_nonexistent_peer(
        self, gossip_protocol
    ):
        """Test sending message to nonexistent peer."""
        content = {"tx": "data"}

        result = await gossip_protocol.send_message_to_peer(
            "nonexistent_peer", MessageType.TRANSACTION, content
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_gossip_protocol_handle_incoming_message(self, gossip_protocol):
        """Test handling incoming message."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"

        message_data = {
            "message_id": "test_id",
            "message_type": "block",
            "sender_id": "sender_1",
            "content": {"block": "data"},
            "timestamp": int(time.time()),
            "ttl": 3600,
            "hop_count": 0,
            "max_hops": 10,
            "signature": None,
            "metadata": {},
        }

        with patch.object(
            gossip_protocol, "_is_duplicate", return_value=False
        ), patch.object(gossip_protocol, "_store_message") as mock_store, patch.object(
            gossip_protocol, "_forward_message"
        ) as mock_forward:
            await gossip_protocol.handle_incoming_message(mock_peer, message_data)

            mock_store.assert_called_once()
            mock_forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_protocol_handle_incoming_message_duplicate(
        self, gossip_protocol
    ):
        """Test handling duplicate incoming message."""
        mock_peer = Mock()

        message_data = {
            "message_id": "test_id",
            "message_type": "block",
            "sender_id": "sender_1",
            "content": {"block": "data"},
            "timestamp": int(time.time()),
            "ttl": 3600,
            "hop_count": 0,
            "max_hops": 10,
            "signature": None,
            "metadata": {},
        }

        with patch.object(
            gossip_protocol, "_is_duplicate", return_value=True
        ), patch.object(gossip_protocol, "_store_message") as mock_store:
            await gossip_protocol.handle_incoming_message(mock_peer, message_data)

            mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_gossip_protocol_handle_incoming_message_expired(
        self, gossip_protocol
    ):
        """Test handling expired incoming message."""
        mock_peer = Mock()

        message_data = {
            "message_id": "test_id",
            "message_type": "block",
            "sender_id": "sender_1",
            "content": {"block": "data"},
            "timestamp": int(time.time()) - 4000,  # Expired
            "ttl": 3600,
            "hop_count": 0,
            "max_hops": 10,
            "signature": None,
            "metadata": {},
        }

        with patch.object(gossip_protocol, "_store_message") as mock_store:
            await gossip_protocol.handle_incoming_message(mock_peer, message_data)

            mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_gossip_protocol_get_peer_messages(self, gossip_protocol):
        """Test getting messages for a peer."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"
        gossip_protocol.add_peer(mock_peer)

        # Add some messages
        message1 = GossipMessage(
            message_id="msg_1",
            message_type=MessageType.BLOCK,
            sender_id="sender_1",
            content={"block": "data1"},
            timestamp=int(time.time()) - 100,
        )
        message2 = GossipMessage(
            message_id="msg_2",
            message_type=MessageType.TRANSACTION,
            sender_id="peer_1",  # Same as peer_1, should be excluded
            content={"tx": "data2"},
            timestamp=int(time.time()) - 200,
        )

        gossip_protocol.messages["msg_1"] = message1
        gossip_protocol.messages["msg_2"] = message2

        messages = await gossip_protocol.get_peer_messages("peer_1")

        # Should only return message1 (not from peer_1 and recent)
        assert len(messages) == 1
        assert messages[0].message_id == "msg_1"

    @pytest.mark.asyncio
    async def test_gossip_protocol_sync_with_peer(self, gossip_protocol):
        """Test syncing with a peer."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"
        mock_peer.send_message = AsyncMock(return_value=True)
        gossip_protocol.add_peer(mock_peer)

        result = await gossip_protocol.sync_with_peer("peer_1")

        assert result is True
        mock_peer.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_protocol_sync_with_nonexistent_peer(self, gossip_protocol):
        """Test syncing with nonexistent peer."""
        result = await gossip_protocol.sync_with_peer("nonexistent_peer")

        assert result is False

    def test_gossip_protocol_select_peers_for_gossip(self, gossip_protocol):
        """Test selecting peers for gossip."""
        # Add some peers
        mock_peer1 = Mock()
        mock_peer1.get_peer_id.return_value = "peer_1"
        mock_peer1.is_connected.return_value = True

        mock_peer2 = Mock()
        mock_peer2.get_peer_id.return_value = "peer_2"
        mock_peer2.is_connected.return_value = True

        mock_peer3 = Mock()
        mock_peer3.get_peer_id.return_value = "peer_3"
        mock_peer3.is_connected.return_value = False  # Not connected

        gossip_protocol.add_peer(mock_peer1)
        gossip_protocol.add_peer(mock_peer2)
        gossip_protocol.add_peer(mock_peer3)

        selected_peers = gossip_protocol._select_peers_for_gossip()

        # Should only select connected peers, up to fanout (3)
        assert len(selected_peers) == 2
        assert all(peer.is_connected() for peer in selected_peers)

    def test_gossip_protocol_select_peers_for_gossip_no_peers(self, gossip_protocol):
        """Test selecting peers for gossip when no peers exist."""
        selected_peers = gossip_protocol._select_peers_for_gossip()

        assert selected_peers == []

    @pytest.mark.asyncio
    async def test_gossip_protocol_gossip_to_peer(self, gossip_protocol):
        """Test gossiping to a specific peer."""
        mock_peer = Mock()
        mock_peer.get_peer_id.return_value = "peer_1"
        gossip_protocol.add_peer(mock_peer)

        with patch.object(
            gossip_protocol, "get_peer_messages"
        ) as mock_get_messages, patch.object(
            gossip_protocol, "_send_message_to_peer"
        ) as mock_send:
            mock_message = Mock()
            mock_get_messages.return_value = [mock_message]

            await gossip_protocol._gossip_to_peer(mock_peer)

            mock_get_messages.assert_called_once_with("peer_1")
            mock_send.assert_called_once_with(mock_peer, mock_message)

    @pytest.mark.asyncio
    async def test_gossip_protocol_broadcast_to_peers(self, gossip_protocol):
        """Test broadcasting to all peers."""
        mock_peer1 = Mock()
        mock_peer1.get_peer_id.return_value = "peer_1"
        mock_peer1.is_connected.return_value = True

        mock_peer2 = Mock()
        mock_peer2.get_peer_id.return_value = "peer_2"
        mock_peer2.is_connected.return_value = True

        gossip_protocol.add_peer(mock_peer1)
        gossip_protocol.add_peer(mock_peer2)

        mock_message = Mock()

        with patch.object(gossip_protocol, "_send_message_to_peer") as mock_send:
            await gossip_protocol._broadcast_to_peers(mock_message)

            assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_gossip_protocol_forward_message(self, gossip_protocol):
        """Test forwarding a message."""
        mock_peer1 = Mock()
        mock_peer1.get_peer_id.return_value = "peer_1"
        mock_peer1.is_connected.return_value = True

        mock_peer2 = Mock()
        mock_peer2.get_peer_id.return_value = "peer_2"
        mock_peer2.is_connected.return_value = True

        gossip_protocol.add_peer(mock_peer1)
        gossip_protocol.add_peer(mock_peer2)

        mock_message = Mock()

        with patch.object(
            gossip_protocol, "_select_peers_for_gossip"
        ) as mock_select, patch.object(
            gossip_protocol, "_send_message_to_peer"
        ) as mock_send:
            mock_select.return_value = [mock_peer1, mock_peer2]

            await gossip_protocol._forward_message(mock_message, exclude_peer="peer_1")

            # Should only send to peer_2 (peer_1 excluded)
            mock_send.assert_called_once_with(mock_peer2, mock_message)

    @pytest.mark.asyncio
    async def test_gossip_protocol_send_message_to_peer(self, gossip_protocol):
        """Test sending message to a specific peer."""
        mock_peer = Mock()
        mock_peer.send_message = AsyncMock(return_value=True)

        mock_message = Mock()
        mock_message.to_dict.return_value = {
            "message_id": "test_id",
            "message_type": "block",
            "sender_id": "sender_1",
            "content": {"block": "data"},
        }

        result = await gossip_protocol._send_message_to_peer(mock_peer, mock_message)

        assert result is True
        mock_peer.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_protocol_send_message_to_peer_error(self, gossip_protocol):
        """Test sending message to peer with error."""
        mock_peer = Mock()
        mock_peer.send_message.side_effect = Exception("Send failed")

        mock_message = Mock()
        mock_message.to_dict.return_value = {}

        result = await gossip_protocol._send_message_to_peer(mock_peer, mock_message)

        assert result is False

    @pytest.mark.asyncio
    async def test_gossip_protocol_store_message(self, gossip_protocol):
        """Test storing a message."""
        mock_message = Mock()
        mock_message.message_id = "test_id"

        with patch.object(gossip_protocol, "_cleanup_messages") as mock_cleanup:
            await gossip_protocol._store_message(mock_message)

            assert "test_id" in gossip_protocol.messages
            assert gossip_protocol.messages["test_id"] == mock_message
            assert "test_id" in gossip_protocol.message_history
            assert "test_id" in gossip_protocol.duplicate_cache
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_protocol_is_duplicate(self, gossip_protocol):
        """Test checking if message is duplicate."""
        # Test message in messages dict
        mock_message = Mock()
        mock_message.message_id = "test_id"
        gossip_protocol.messages["test_id"] = mock_message

        result = await gossip_protocol._is_duplicate("test_id")
        assert result is True

        # Test message in duplicate cache
        gossip_protocol.messages.clear()
        gossip_protocol.duplicate_cache["test_id"] = int(time.time())

        result = await gossip_protocol._is_duplicate("test_id")
        assert result is True

        # Test non-duplicate
        result = await gossip_protocol._is_duplicate("new_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_gossip_protocol_cleanup_messages(self, gossip_protocol):
        """Test cleaning up old messages."""
        # Add some messages
        current_time = int(time.time())

        # Expired message
        expired_message = Mock()
        expired_message.is_expired.return_value = True
        expired_message.timestamp = current_time - 4000

        # Recent message
        recent_message = Mock()
        recent_message.is_expired.return_value = False
        recent_message.timestamp = current_time - 100

        gossip_protocol.messages["expired"] = expired_message
        gossip_protocol.messages["recent"] = recent_message

        # Add to duplicate cache
        gossip_protocol.duplicate_cache["expired"] = current_time - 400
        gossip_protocol.duplicate_cache["recent"] = current_time - 100

        # Add to message history
        gossip_protocol.message_history = ["expired", "recent", "old1", "old2"]

        await gossip_protocol._cleanup_messages()

        # Expired message should be removed
        assert "expired" not in gossip_protocol.messages
        assert "recent" in gossip_protocol.messages

        # Expired duplicate should be removed
        assert "expired" not in gossip_protocol.duplicate_cache
        assert "recent" in gossip_protocol.duplicate_cache

    def test_gossip_protocol_generate_message_id(self, gossip_protocol):
        """Test generating message ID."""
        message_id = gossip_protocol._generate_message_id()

        assert isinstance(message_id, str)
        assert len(message_id) == 16  # Should be truncated to 16 chars

    def test_gossip_protocol_get_stats(self, gossip_protocol):
        """Test getting gossip protocol statistics."""
        # Add some peers and messages
        mock_peer = Mock()
        gossip_protocol.add_peer(mock_peer)

        mock_message = Mock()
        gossip_protocol.messages["msg_1"] = mock_message

        stats = gossip_protocol.get_stats()

        assert stats["node_id"] == "test_node"
        assert stats["peers_count"] == 1
        assert stats["messages_count"] == 1
        assert stats["message_history_count"] == 0
        assert stats["duplicate_cache_count"] == 0
        assert stats["running"] is False
        assert "config" in stats

    def test_gossip_protocol_str_representation(self, gossip_protocol):
        """Test string representation of gossip protocol."""
        str_repr = str(gossip_protocol)

        assert "GossipProtocol" in str_repr
        assert "test_node" in str_repr
        assert "peers=0" in str_repr
        assert "messages=0" in str_repr

    def test_gossip_protocol_repr_representation(self, gossip_protocol):
        """Test detailed representation of gossip protocol."""
        repr_str = repr(gossip_protocol)

        assert "GossipProtocol" in repr_str
        assert "test_node" in repr_str
        assert "peers=0" in repr_str
        assert "messages=0" in repr_str
        assert "running=False" in repr_str
