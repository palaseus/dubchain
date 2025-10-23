"""
Unit tests for peer functionality.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio

# Network peer tests - fixed async task management issues
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.crypto.signatures import PrivateKey, PublicKey
from dubchain.network.peer import ConnectionType, Peer, PeerInfo, PeerStatus


class TestPeerInfo:
    """Test the PeerInfo class."""

    @pytest.fixture
    def mock_public_key(self):
        """Create a mock public key."""
        mock_key = Mock(spec=PublicKey)
        mock_key.to_hex = Mock(
            return_value="02" + "0" * 64
        )  # 33 bytes compressed public key
        return mock_key

    def test_peer_info_creation(self, mock_public_key):
        """Test creating peer information."""
        peer_info = PeerInfo(
            peer_id="peer_123",
            public_key=mock_public_key,
            address="192.168.1.100",
            port=8080,
            connection_type=ConnectionType.OUTBOUND,
            version="1.0.0",
            capabilities=["block_sync", "transaction_relay"],
        )

        assert peer_info.peer_id == "peer_123"
        assert peer_info.public_key == mock_public_key
        assert peer_info.address == "192.168.1.100"
        assert peer_info.port == 8080
        assert peer_info.connection_type == ConnectionType.OUTBOUND
        assert peer_info.status == PeerStatus.DISCONNECTED
        assert peer_info.version == "1.0.0"
        assert "block_sync" in peer_info.capabilities
        assert peer_info.last_seen > 0
        assert peer_info.connection_count == 0

    def test_peer_info_validation(self, mock_public_key):
        """Test peer information validation."""
        # Valid peer info
        peer_info = PeerInfo(
            peer_id="peer_123",
            public_key=mock_public_key,
            address="192.168.1.100",
            port=8080,
            connection_type=ConnectionType.OUTBOUND,
        )

        assert peer_info.peer_id == "peer_123"

        # Invalid peer info - empty peer ID
        with pytest.raises(ValueError, match="Peer ID cannot be empty"):
            PeerInfo(
                peer_id="",
                public_key=mock_public_key,
                address="192.168.1.100",
                port=8080,
                connection_type=ConnectionType.OUTBOUND,
            )

        # Invalid peer info - empty address
        with pytest.raises(ValueError, match="Address cannot be empty"):
            PeerInfo(
                peer_id="peer_123",
                public_key=mock_public_key,
                address="",
                port=8080,
                connection_type=ConnectionType.OUTBOUND,
            )

        # Invalid peer info - invalid port
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            PeerInfo(
                peer_id="peer_123",
                public_key=mock_public_key,
                address="192.168.1.100",
                port=0,
                connection_type=ConnectionType.OUTBOUND,
            )

    def test_peer_info_methods(self, mock_public_key):
        """Test peer information methods."""
        peer_info = PeerInfo(
            peer_id="peer_123",
            public_key=mock_public_key,
            address="192.168.1.100",
            port=8080,
            connection_type=ConnectionType.OUTBOUND,
        )

        initial_last_seen = peer_info.last_seen

        # Test update_last_seen
        time.sleep(0.01)
        peer_info.update_last_seen()
        assert peer_info.last_seen >= initial_last_seen

        # Test increment_connection_count
        initial_count = peer_info.connection_count
        peer_info.increment_connection_count()
        assert peer_info.connection_count == initial_count + 1

        # Test record_successful_connection
        initial_successful = peer_info.successful_connections
        peer_info.record_successful_connection()
        assert peer_info.successful_connections == initial_successful + 1

        # Test record_failed_connection
        initial_failed = peer_info.failed_connections
        peer_info.record_failed_connection()
        assert peer_info.failed_connections == initial_failed + 1

    def test_peer_info_serialization(self, mock_public_key):
        """Test peer information serialization."""
        peer_info = PeerInfo(
            peer_id="peer_123",
            public_key=mock_public_key,
            address="192.168.1.100",
            port=8080,
            connection_type=ConnectionType.OUTBOUND,
            version="1.0.0",
            capabilities=["block_sync", "transaction_relay"],
        )

        # Test to_dict
        data = peer_info.to_dict()
        assert isinstance(data, dict)
        assert data["peer_id"] == "peer_123"
        assert data["address"] == "192.168.1.100"
        assert data["port"] == 8080
        assert data["connection_type"] == "outbound"
        assert data["version"] == "1.0.0"
        assert "block_sync" in data["capabilities"]

        # Test from_dict
        with patch(
            "dubchain.crypto.signatures.PublicKey.from_hex",
            return_value=mock_public_key,
        ):
            deserialized = PeerInfo.from_dict(data)
            assert deserialized.peer_id == peer_info.peer_id
            assert deserialized.address == peer_info.address
            assert deserialized.port == peer_info.port
            assert deserialized.connection_type == peer_info.connection_type


class TestPeer:
    """Test the Peer class."""

    @pytest.fixture
    def mock_public_key(self):
        """Create a mock public key."""
        mock_key = Mock(spec=PublicKey)
        mock_key.to_hex = Mock(
            return_value="02" + "0" * 64
        )  # 33 bytes compressed public key
        return mock_key

    @pytest.fixture
    def peer_info(self, mock_public_key):
        """Create peer information."""
        return PeerInfo(
            peer_id="peer_123",
            public_key=mock_public_key,
            address="192.168.1.100",
            port=8080,
            connection_type=ConnectionType.OUTBOUND,
            version="1.0.0",
            capabilities=["block_sync", "transaction_relay"],
        )

    @pytest.fixture
    def mock_private_key(self):
        """Create a mock private key."""
        mock_key = Mock(spec=PrivateKey)
        mock_signature = Mock()
        mock_signature.to_hex = Mock(return_value="mock_signature_hex")
        mock_key.sign = Mock(return_value=mock_signature)
        return mock_key

    @pytest.fixture
    def peer(self, peer_info, mock_private_key):
        """Create a peer instance."""
        return Peer(peer_info, mock_private_key)

    def test_peer_creation(self, peer, peer_info):
        """Test creating a peer instance."""
        assert peer is not None
        assert peer.peer_info == peer_info
        assert peer.status == PeerStatus.DISCONNECTED
        assert peer.connection_count == 0
        assert peer.last_activity == 0
        assert peer.is_connected() is False

    @pytest.mark.asyncio
    async def test_peer_connection(self, peer):
        """Test peer connection."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer
            result = await peer.connect()

            assert result is True
            assert peer.status == PeerStatus.CONNECTED
            assert peer.info.connection_count == 1

            # Disconnect to clean up
            await peer.disconnect()

        assert peer.is_connected() is False

    @pytest.mark.asyncio
    async def test_peer_disconnection(self, peer):
        """Test peer disconnection."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect then disconnect
            await peer.connect()
            await peer.disconnect()

            assert peer.status == PeerStatus.DISCONNECTED
            assert peer.connection_count == 0
            assert peer.is_connected() is False

    @pytest.mark.asyncio
    async def test_peer_send_message(self, peer):
        """Test sending a message to peer."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer first
            await peer.connect()

            # Send message
            message = b'{"type": "ping", "data": "hello"}'
            result = await peer.send_message(message)

            assert result is True
            assert peer.peer_info.messages_sent > 0

    @pytest.mark.asyncio
    async def test_peer_send_message_not_connected(self, peer):
        """Test sending message to disconnected peer."""
        message = b'{"type": "ping", "data": "hello"}'
        result = await peer.send_message(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_peer_receive_message(self, peer):
        """Test receiving a message from peer."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer first
            await peer.connect()

            # Receive message
            message = {"type": "pong", "data": "world"}
            peer.receive_message(message)

            assert peer.peer_info.messages_received > 0

    def test_peer_receive_message_not_connected(self, peer):
        """Test receiving message from disconnected peer."""
        message = {"type": "pong", "data": "world"}
        peer.receive_message(message)

        # Should still increment message count even when not connected
        assert peer.peer_info.messages_received > 0

    @pytest.mark.asyncio
    async def test_peer_heartbeat(self, peer):
        """Test peer heartbeat."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer first
            await peer.connect()

            # Send heartbeat
            result = peer.heartbeat()

            assert result is True

    def test_peer_heartbeat_not_connected(self, peer):
        """Test heartbeat from disconnected peer."""
        result = peer.heartbeat()

        assert result is False

    @pytest.mark.asyncio
    async def test_peer_is_alive(self, peer):
        """Test checking if peer is alive."""
        # Initially not alive (not connected)
        assert not peer.is_alive()

        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect and should be alive
            await peer.connect()
            assert peer.is_alive()

            # Disconnect and should not be alive
            await peer.disconnect()
            assert not peer.is_alive()

    @pytest.mark.asyncio
    async def test_peer_get_latency(self, peer):
        """Test getting peer latency."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer first
            await peer.connect()

            # Set some latency
            peer.info.update_latency(50.0)

            # Get latency
            latency = peer.get_latency()

            assert latency is not None
            assert latency >= 0

    def test_peer_get_latency_not_connected(self, peer):
        """Test getting latency from disconnected peer."""
        latency = peer.get_latency()

        assert latency is None

    def test_peer_update_capabilities(self, peer):
        """Test updating peer capabilities."""
        new_capabilities = ["block_sync", "transaction_relay", "consensus"]

        result = peer.update_capabilities(new_capabilities)

        assert result is True
        assert peer.peer_info.capabilities == new_capabilities

    def test_peer_has_capability(self, peer):
        """Test checking if peer has a capability."""
        # Check existing capability
        assert peer.has_capability("block_sync")

        # Check non-existing capability
        assert not peer.has_capability("mining")

    def test_peer_info_add_bytes_sent(self, peer):
        """Test adding bytes sent."""
        initial_bytes = peer.info.bytes_sent
        peer.info.add_bytes_sent(1000)
        
        assert peer.info.bytes_sent == initial_bytes + 1000
        assert peer.info.last_seen > 0

    def test_peer_info_add_bytes_sent_negative(self, peer):
        """Test adding negative bytes sent (should be ignored)."""
        initial_bytes = peer.info.bytes_sent
        peer.info.add_bytes_sent(-100)
        
        assert peer.info.bytes_sent == initial_bytes

    def test_peer_info_add_bytes_received(self, peer):
        """Test adding bytes received."""
        initial_bytes = peer.info.bytes_received
        peer.info.add_bytes_received(2000)
        
        assert peer.info.bytes_received == initial_bytes + 2000
        assert peer.info.last_seen > 0

    def test_peer_info_add_bytes_received_negative(self, peer):
        """Test adding negative bytes received (should be ignored)."""
        initial_bytes = peer.info.bytes_received
        peer.info.add_bytes_received(-200)
        
        assert peer.info.bytes_received == initial_bytes

    def test_peer_info_increment_messages_sent(self, peer):
        """Test incrementing messages sent."""
        initial_count = peer.info.messages_sent
        peer.info.increment_messages_sent()
        
        assert peer.info.messages_sent == initial_count + 1
        assert peer.info.last_seen > 0

    def test_peer_info_increment_messages_received(self, peer):
        """Test incrementing messages received."""
        initial_count = peer.info.messages_received
        peer.info.increment_messages_received()
        
        assert peer.info.messages_received == initial_count + 1
        assert peer.info.last_seen > 0

    def test_peer_info_update_latency_positive(self, peer):
        """Test updating latency with positive value."""
        peer.info.update_latency(50.0)
        
        assert peer.info.latency == 50.0
        assert peer.info.last_seen > 0

    def test_peer_info_update_latency_negative(self, peer):
        """Test updating latency with negative value (should be ignored)."""
        initial_latency = peer.info.latency
        peer.info.update_latency(-10.0)
        
        assert peer.info.latency == initial_latency

    def test_peer_info_add_capability_new(self, peer):
        """Test adding new capability."""
        peer.info.add_capability("new_capability")
        
        assert "new_capability" in peer.info.capabilities
        assert peer.info.last_seen > 0

    def test_peer_info_add_capability_duplicate(self, peer):
        """Test adding duplicate capability (should not add)."""
        peer.info.add_capability("block_sync")
        initial_count = len(peer.info.capabilities)
        
        peer.info.add_capability("block_sync")
        
        assert len(peer.info.capabilities) == initial_count

    def test_peer_info_add_capability_empty(self, peer):
        """Test adding empty capability (should be ignored)."""
        initial_count = len(peer.info.capabilities)
        peer.info.add_capability("")
        
        assert len(peer.info.capabilities) == initial_count

    def test_peer_info_get_connection_success_rate_no_attempts(self, peer):
        """Test getting connection success rate with no attempts."""
        rate = peer.info.get_connection_success_rate()
        
        assert rate == 0.0

    def test_peer_info_get_connection_success_rate_successful(self, peer):
        """Test getting connection success rate with successful connections."""
        peer.info.successful_connections = 8
        peer.info.failed_connections = 2
        
        rate = peer.info.get_connection_success_rate()
        
        assert rate == 0.8

    def test_peer_info_get_connection_success_rate_all_failed(self, peer):
        """Test getting connection success rate with all failed connections."""
        peer.info.successful_connections = 0
        peer.info.failed_connections = 5
        
        rate = peer.info.get_connection_success_rate()
        
        assert rate == 0.0

    def test_peer_info_get_connection_success_rate_all_successful(self, peer):
        """Test getting connection success rate with all successful connections."""
        peer.info.successful_connections = 10
        peer.info.failed_connections = 0
        
        rate = peer.info.get_connection_success_rate()
        
        assert rate == 1.0

    def test_peer_get_info(self, peer):
        """Test getting peer information."""
        info = peer.get_info()

        assert info is not None
        assert info.peer_id == peer.peer_info.peer_id
        assert info.address == peer.peer_info.address
        assert info.status == peer.status

    @pytest.mark.asyncio
    async def test_peer_reset_connection(self, peer):
        """Test resetting peer connection."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer first
            await peer.connect()
            assert peer.status == PeerStatus.CONNECTED

            # Reset connection
            peer.reset_connection()

            assert peer.status == PeerStatus.DISCONNECTED
            assert peer.connection_count == 0

    def test_peer_ban(self, peer):
        """Test banning a peer."""
        # Ban peer
        result = peer.ban("spam")

        assert result is True
        assert peer.status == PeerStatus.BANNED

    def test_peer_unban(self, peer):
        """Test unbanning a peer."""
        # Ban peer first
        peer.ban("spam")
        assert peer.status == PeerStatus.BANNED

        # Unban peer
        result = peer.unban()

        assert result is True
        assert peer.status == PeerStatus.DISCONNECTED

    def test_peer_is_banned(self, peer):
        """Test checking if peer is banned."""
        # Initially not banned
        assert not peer.is_banned()

        # Ban peer
        peer.ban("spam")
        assert peer.is_banned()

        # Unban peer
        peer.unban()
        assert not peer.is_banned()

    def test_peer_get_statistics(self, peer):
        """Test getting peer statistics."""
        # Send some messages
        peer.receive_message({"type": "pong"})

        # Get statistics
        stats = peer.get_statistics()

        assert stats is not None
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "connection_count" in stats
        assert "last_activity" in stats
        assert stats["messages_received"] >= 1

    def test_peer_cleanup(self, peer):
        """Test cleaning up peer resources."""
        # Cleanup
        result = peer.cleanup()

        assert result is True
        assert peer.status == PeerStatus.DISCONNECTED
        assert peer.connection_count == 0

    @pytest.mark.asyncio
    async def test_peer_authenticate(self, peer):
        """Test peer authentication."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect peer first
            await peer.connect()

            # Authenticate
            result = await peer.authenticate()

            assert result is True
            assert peer.status == PeerStatus.AUTHENTICATING

    @pytest.mark.asyncio
    async def test_peer_sync(self, peer):
        """Test peer synchronization."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect and authenticate peer first
            await peer.connect()
            await peer.authenticate()

            # Sync
            result = peer.sync()

            assert result is True
            assert peer.status == PeerStatus.SYNCING

    @pytest.mark.asyncio
    async def test_peer_ready(self, peer):
        """Test peer ready state."""
        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect, authenticate, and sync peer first
            await peer.connect()
            await peer.authenticate()
            peer.sync()

            # Mark as ready
            result = peer.ready()

            assert result is True
            assert peer.status == PeerStatus.READY

    def test_peer_error_handling(self, peer):
        """Test peer error handling."""
        # Set error state
        result = peer.set_error("Connection failed")

        assert result is True
        assert peer.status == PeerStatus.ERROR

    @pytest.mark.asyncio
    async def test_peer_status_transitions(self, peer):
        """Test peer status transitions."""
        # Start disconnected
        assert peer.status == PeerStatus.DISCONNECTED

        # Mock the network connection
        mock_reader = Mock()
        mock_writer = AsyncMock()
        mock_writer.write = Mock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # Connect
            await peer.connect()
            assert peer.status == PeerStatus.CONNECTED

            # Authenticate
            await peer.authenticate()
            assert peer.status == PeerStatus.AUTHENTICATING

            # Sync
            peer.sync()
            assert peer.status == PeerStatus.SYNCING

            # Ready
            peer.ready()
            assert peer.status == PeerStatus.READY

            # Disconnect
            await peer.disconnect()
            assert peer.status == PeerStatus.DISCONNECTED
