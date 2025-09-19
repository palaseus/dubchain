"""Basic tests for network peer module."""

import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.network.peer import (
    PeerStatus,
    ConnectionType,
    PeerInfo,
)
from src.dubchain.crypto.signatures import PublicKey


class TestPeerStatus:
    """Test PeerStatus enum."""

    def test_peer_status_values(self):
        """Test peer status enum values."""
        assert PeerStatus.DISCONNECTED.value == "disconnected"
        assert PeerStatus.CONNECTING.value == "connecting"
        assert PeerStatus.CONNECTED.value == "connected"
        assert PeerStatus.AUTHENTICATING.value == "authenticating"
        assert PeerStatus.AUTHENTICATED.value == "authenticated"
        assert PeerStatus.SYNCING.value == "syncing"
        assert PeerStatus.READY.value == "ready"
        assert PeerStatus.ERROR.value == "error"
        assert PeerStatus.BANNED.value == "banned"


class TestConnectionType:
    """Test ConnectionType enum."""

    def test_connection_type_values(self):
        """Test connection type enum values."""
        assert ConnectionType.INBOUND.value == "inbound"
        assert ConnectionType.OUTBOUND.value == "outbound"
        assert ConnectionType.RELAY.value == "relay"
        assert ConnectionType.SEED.value == "seed"


class TestPeerInfo:
    """Test PeerInfo dataclass."""

    def test_init(self):
        """Test PeerInfo initialization."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        assert peer_info.peer_id == "test_peer"
        assert peer_info.public_key == public_key
        assert peer_info.address == "127.0.0.1"
        assert peer_info.port == 8080
        assert peer_info.connection_type == ConnectionType.INBOUND
        assert peer_info.status == PeerStatus.DISCONNECTED

    def test_init_validation_empty_peer_id(self):
        """Test PeerInfo validation with empty peer ID."""
        public_key = Mock(spec=PublicKey)
        
        with pytest.raises(ValueError, match="Peer ID cannot be empty"):
            PeerInfo(
                peer_id="",
                public_key=public_key,
                address="127.0.0.1",
                port=8080,
                connection_type=ConnectionType.INBOUND
            )

    def test_init_validation_invalid_port(self):
        """Test PeerInfo validation with invalid port."""
        public_key = Mock(spec=PublicKey)
        
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            PeerInfo(
                peer_id="test_peer",
                public_key=public_key,
                address="127.0.0.1",
                port=0,
                connection_type=ConnectionType.INBOUND
            )

    def test_update_last_seen(self):
        """Test updating last seen timestamp."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        original_time = peer_info.last_seen
        
        # Mock time.time to return a different value
        with patch('time.time', return_value=original_time + 1):
            peer_info.update_last_seen()
            assert peer_info.last_seen == original_time + 1

    def test_record_successful_connection(self):
        """Test recording successful connection."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        original_count = peer_info.successful_connections
        peer_info.record_successful_connection()
        
        assert peer_info.successful_connections == original_count + 1

    def test_record_failed_connection(self):
        """Test recording failed connection."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        original_count = peer_info.failed_connections
        peer_info.record_failed_connection()
        
        assert peer_info.failed_connections == original_count + 1

    def test_add_capability(self):
        """Test adding capability."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        peer_info.add_capability("block_sync")
        assert "block_sync" in peer_info.capabilities

    def test_has_capability(self):
        """Test checking capability."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        assert peer_info.has_capability("block_sync") is False
        peer_info.add_capability("block_sync")
        assert peer_info.has_capability("block_sync") is True

    def test_get_connection_success_rate(self):
        """Test getting connection success rate."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND,
            successful_connections=10,
            failed_connections=2
        )
        
        rate = peer_info.get_connection_success_rate()
        assert rate == 10 / (10 + 2)  # 0.833...

    def test_is_healthy(self):
        """Test checking if peer is healthy."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND,
            status=PeerStatus.CONNECTED,
            successful_connections=10,
            failed_connections=5
        )
        
        # Should be healthy with good connection rate and connected status
        assert peer_info.is_healthy() is True
        
        # Set status to disconnected
        peer_info.status = PeerStatus.DISCONNECTED
        assert peer_info.is_healthy() is False

    def test_to_dict(self):
        """Test converting to dictionary."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        peer_info.add_capability("block_sync")
        
        data = peer_info.to_dict()
        
        assert data["peer_id"] == "test_peer"
        assert data["address"] == "127.0.0.1"
        assert data["port"] == 8080
        assert data["connection_type"] == "inbound"
        assert data["status"] == "disconnected"
        assert "block_sync" in data["capabilities"]

    def test_str_repr(self):
        """Test string representation."""
        public_key = Mock(spec=PublicKey)
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=public_key,
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        
        str_repr = str(peer_info)
        assert "test_peer" in str_repr
        assert "127.0.0.1" in str_repr
        assert "8080" in str_repr
