"""
Unit tests for network discovery module.
"""

import asyncio
import json
import socket
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from dubchain.crypto.signatures import ECDSASigner, PublicKey
from dubchain.network.discovery import DiscoveryConfig, DiscoveryMethod, PeerDiscovery
from dubchain.network.peer import ConnectionType, PeerInfo


class TestDiscoveryMethod:
    """Test DiscoveryMethod enum."""

    def test_discovery_method_values(self):
        """Test DiscoveryMethod enum values."""
        assert DiscoveryMethod.BOOTSTRAP.value == "bootstrap"
        assert DiscoveryMethod.DNS.value == "dns"
        assert DiscoveryMethod.PEER_EXCHANGE.value == "peer_exchange"
        assert DiscoveryMethod.MULTICAST.value == "multicast"
        assert DiscoveryMethod.MANUAL.value == "manual"
        assert DiscoveryMethod.DHT.value == "dht"


class TestDiscoveryConfig:
    """Test DiscoveryConfig class."""

    def test_discovery_config_creation(self):
        """Test DiscoveryConfig creation."""
        config = DiscoveryConfig()

        assert config.bootstrap_nodes == []
        assert config.dns_seeds == []
        assert config.multicast_address == "224.0.0.1"
        assert config.multicast_port == 12345
        assert config.discovery_interval == 30.0
        assert config.max_peers == 100
        assert config.min_peers == 5
        assert config.peer_timeout == 10.0
        assert config.enable_dns_discovery is True
        assert config.enable_multicast is True
        assert config.enable_peer_exchange is True
        assert config.peer_exchange_interval == 60.0
        assert config.max_peer_exchange_peers == 20
        assert config.metadata == {}

    def test_discovery_config_custom_values(self):
        """Test DiscoveryConfig with custom values."""
        config = DiscoveryConfig(
            bootstrap_nodes=["node1", "node2"],
            dns_seeds=["seed1.example.com"],
            multicast_address="224.0.0.2",
            multicast_port=54321,
            discovery_interval=60.0,
            max_peers=200,
            min_peers=10,
            peer_timeout=20.0,
            enable_dns_discovery=False,
            enable_multicast=False,
            enable_peer_exchange=False,
            peer_exchange_interval=120.0,
            max_peer_exchange_peers=50,
            metadata={"version": "1.0"},
        )

        assert config.bootstrap_nodes == ["node1", "node2"]
        assert config.dns_seeds == ["seed1.example.com"]
        assert config.multicast_address == "224.0.0.2"
        assert config.multicast_port == 54321
        assert config.discovery_interval == 60.0
        assert config.max_peers == 200
        assert config.min_peers == 10
        assert config.peer_timeout == 20.0
        assert config.enable_dns_discovery is False
        assert config.enable_multicast is False
        assert config.enable_peer_exchange is False
        assert config.peer_exchange_interval == 120.0
        assert config.max_peer_exchange_peers == 50
        assert config.metadata == {"version": "1.0"}

    def test_discovery_config_validation(self):
        """Test DiscoveryConfig validation."""
        # Test invalid discovery_interval
        with pytest.raises(ValueError):
            DiscoveryConfig(discovery_interval=0)

        # Test invalid max_peers
        with pytest.raises(ValueError):
            DiscoveryConfig(max_peers=0)

        # Test invalid min_peers
        with pytest.raises(ValueError):
            DiscoveryConfig(min_peers=-1)

        # Test invalid peer_timeout
        with pytest.raises(ValueError):
            DiscoveryConfig(peer_timeout=0)


class TestPeerDiscovery:
    """Test PeerDiscovery class."""

    @pytest.fixture
    def discovery_config(self):
        """Create a discovery config for testing."""
        return DiscoveryConfig(
            bootstrap_nodes=["127.0.0.1:8333", "127.0.0.1:8334"],
            dns_seeds=["seed.example.com"],
            discovery_interval=1.0,
            max_peers=10,
            min_peers=2,
        )

    @pytest.fixture
    def peer_discovery(self, discovery_config):
        """Create a PeerDiscovery instance for testing."""
        return PeerDiscovery(discovery_config, "test_node_id")

    @pytest.fixture
    def mock_peer_info(self):
        """Create a mock PeerInfo for testing."""
        _, public_key = ECDSASigner.generate_keypair()
        return PeerInfo(
            peer_id="test_peer_1",
            public_key=public_key,
            address="127.0.0.1",
            port=8333,
            connection_type=ConnectionType.OUTBOUND,
        )

    def test_peer_discovery_creation(self, discovery_config):
        """Test PeerDiscovery creation."""
        discovery = PeerDiscovery(discovery_config, "test_node_id")

        assert discovery.config == discovery_config
        assert discovery.node_id == "test_node_id"
        assert discovery.discovered_peers == {}
        assert discovery.connected_peers == {}
        assert discovery.bootstrap_peers == []
        assert discovery.discovery_callbacks == []
        assert discovery.running is False

    def test_peer_discovery_add_discovery_callback(self, peer_discovery):
        """Test adding discovery callback."""
        callback = Mock()
        peer_discovery.add_discovery_callback(callback)

        assert callback in peer_discovery.discovery_callbacks

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_peers_bootstrap(self, peer_discovery):
        """Test discovering peers via bootstrap method."""
        with patch.object(
            peer_discovery, "_discover_bootstrap_peers", return_value=[]
        ) as mock_discover:
            result = await peer_discovery.discover_peers(DiscoveryMethod.BOOTSTRAP)
            mock_discover.assert_called_once()
            assert result == []

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_peers_dns(self, peer_discovery):
        """Test discovering peers via DNS method."""
        with patch.object(
            peer_discovery, "_discover_dns_peers", return_value=[]
        ) as mock_discover:
            result = await peer_discovery.discover_peers(DiscoveryMethod.DNS)
            mock_discover.assert_called_once()
            assert result == []

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_peers_multicast(self, peer_discovery):
        """Test discovering peers via multicast method."""
        with patch.object(
            peer_discovery, "_discover_multicast_peers", return_value=[]
        ) as mock_discover:
            result = await peer_discovery.discover_peers(DiscoveryMethod.MULTICAST)
            mock_discover.assert_called_once()
            assert result == []

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_peers_peer_exchange(self, peer_discovery):
        """Test discovering peers via peer exchange method."""
        with patch.object(
            peer_discovery, "_discover_peer_exchange_peers", return_value=[]
        ) as mock_discover:
            result = await peer_discovery.discover_peers(DiscoveryMethod.PEER_EXCHANGE)
            mock_discover.assert_called_once()
            assert result == []

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_peers_unknown_method(self, peer_discovery):
        """Test discovering peers with unknown method."""
        result = await peer_discovery.discover_peers(DiscoveryMethod.MANUAL)
        assert result == []

    @pytest.mark.asyncio
    async def test_peer_discovery_add_peer(self, peer_discovery, mock_peer_info):
        """Test adding a peer."""
        callback = Mock()
        callback.return_value = None
        peer_discovery.add_discovery_callback(callback)

        await peer_discovery.add_peer(mock_peer_info)

        assert mock_peer_info.peer_id in peer_discovery.discovered_peers
        assert peer_discovery.discovered_peers[mock_peer_info.peer_id] == mock_peer_info
        callback.assert_called_once_with(mock_peer_info)

    @pytest.mark.asyncio
    async def test_peer_discovery_add_peer_duplicate(
        self, peer_discovery, mock_peer_info
    ):
        """Test adding a duplicate peer."""
        await peer_discovery.add_peer(mock_peer_info)
        await peer_discovery.add_peer(mock_peer_info)  # Should not add again

        assert len(peer_discovery.discovered_peers) == 1

    @pytest.mark.asyncio
    async def test_peer_discovery_remove_peer(self, peer_discovery, mock_peer_info):
        """Test removing a peer."""
        await peer_discovery.add_peer(mock_peer_info)
        assert mock_peer_info.peer_id in peer_discovery.discovered_peers

        await peer_discovery.remove_peer(mock_peer_info.peer_id)
        assert mock_peer_info.peer_id not in peer_discovery.discovered_peers

    @pytest.mark.asyncio
    async def test_peer_discovery_get_peers(self, peer_discovery, mock_peer_info):
        """Test getting peers."""
        await peer_discovery.add_peer(mock_peer_info)

        peers = await peer_discovery.get_peers()
        assert len(peers) == 1
        assert peers[0] == mock_peer_info

    @pytest.mark.asyncio
    async def test_peer_discovery_get_peers_with_count(self, peer_discovery):
        """Test getting peers with count limit."""
        # Add multiple peers
        for i in range(5):
            _, public_key = ECDSASigner.generate_keypair()
            peer_info = PeerInfo(
                peer_id=f"peer_{i}",
                public_key=public_key,
                address="127.0.0.1",
                port=8333 + i,
                connection_type=ConnectionType.OUTBOUND,
            )
            await peer_discovery.add_peer(peer_info)

        peers = await peer_discovery.get_peers(count=3)
        assert len(peers) == 3

    @pytest.mark.asyncio
    async def test_peer_discovery_get_peers_with_connection_type(self, peer_discovery):
        """Test getting peers filtered by connection type."""
        # Add peers with different connection types
        _, outbound_public_key = ECDSASigner.generate_keypair()
        _, inbound_public_key = ECDSASigner.generate_keypair()
        outbound_peer = PeerInfo(
            peer_id="outbound_peer",
            public_key=outbound_public_key,
            address="127.0.0.1",
            port=8333,
            connection_type=ConnectionType.OUTBOUND,
        )
        inbound_peer = PeerInfo(
            peer_id="inbound_peer",
            public_key=inbound_public_key,
            address="127.0.0.1",
            port=8334,
            connection_type=ConnectionType.INBOUND,
        )

        await peer_discovery.add_peer(outbound_peer)
        await peer_discovery.add_peer(inbound_peer)

        outbound_peers = await peer_discovery.get_peers(
            connection_type=ConnectionType.OUTBOUND
        )
        assert len(outbound_peers) == 1
        assert outbound_peers[0].peer_id == "outbound_peer"

    @pytest.mark.asyncio
    async def test_peer_discovery_get_connected_peers(self, peer_discovery):
        """Test getting connected peers."""
        # Add a connected peer
        mock_peer = Mock()
        peer_discovery.connected_peers["peer_1"] = mock_peer

        connected_peers = await peer_discovery.get_connected_peers()
        assert len(connected_peers) == 1
        assert connected_peers[0] == mock_peer

    @pytest.mark.asyncio
    async def test_peer_discovery_connect_to_peer_success(
        self, peer_discovery, mock_peer_info
    ):
        """Test successful connection to a peer."""
        with patch("dubchain.network.discovery.Peer") as mock_peer_class:
            mock_peer = Mock()
            mock_peer.connect = AsyncMock(return_value=True)
            mock_peer_class.return_value = mock_peer

            result = await peer_discovery.connect_to_peer(mock_peer_info)

            assert result == mock_peer
            assert mock_peer_info.peer_id in peer_discovery.connected_peers
            mock_peer.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_peer_discovery_connect_to_peer_failure(
        self, peer_discovery, mock_peer_info
    ):
        """Test failed connection to a peer."""
        with patch("dubchain.network.discovery.Peer") as mock_peer_class:
            mock_peer = Mock()
            mock_peer.connect = AsyncMock(return_value=False)
            mock_peer_class.return_value = mock_peer

            result = await peer_discovery.connect_to_peer(mock_peer_info)

            assert result is None
            assert mock_peer_info.peer_id not in peer_discovery.connected_peers

    @pytest.mark.asyncio
    async def test_peer_discovery_connect_to_peer_exception(
        self, peer_discovery, mock_peer_info
    ):
        """Test connection to peer with exception."""
        with patch(
            "dubchain.network.discovery.Peer",
            side_effect=Exception("Connection failed"),
        ):
            result = await peer_discovery.connect_to_peer(mock_peer_info)
            assert result is None

    def test_peer_discovery_get_stats(self, peer_discovery):
        """Test getting discovery stats."""
        stats = peer_discovery.get_stats()

        assert isinstance(stats, dict)
        assert stats["node_id"] == "test_node_id"
        assert stats["discovered_peers_count"] == 0
        assert stats["connected_peers_count"] == 0
        assert stats["bootstrap_peers_count"] == 0
        assert stats["running"] is False
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_peer_discovery_start_stop(self, peer_discovery):
        """Test starting and stopping peer discovery."""
        with patch.object(
            peer_discovery, "_initialize_bootstrap_peers", return_value=None
        ) as mock_init:
            with patch.object(
                peer_discovery, "_discovery_loop", return_value=None
            ) as mock_discovery:
                with patch.object(
                    peer_discovery, "_peer_exchange_loop", return_value=None
                ) as mock_exchange:
                    # Start discovery
                    await peer_discovery.start()

                    assert peer_discovery.running is True
                    mock_init.assert_called_once()
                    assert peer_discovery.discovery_task is not None
                    assert peer_discovery.peer_exchange_task is not None

                    # Stop discovery
                    await peer_discovery.stop()

                    assert peer_discovery.running is False

    @pytest.mark.asyncio
    async def test_peer_discovery_start_already_running(self, peer_discovery):
        """Test starting discovery when already running."""
        peer_discovery.running = True

        with patch.object(peer_discovery, "_initialize_bootstrap_peers") as mock_init:
            await peer_discovery.start()
            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_peer_discovery_stop_not_running(self, peer_discovery):
        """Test stopping discovery when not running."""
        peer_discovery.running = False

        await peer_discovery.stop()
        assert peer_discovery.running is False

    @pytest.mark.asyncio
    async def test_peer_discovery_initialize_bootstrap_peers(self, peer_discovery):
        """Test initializing bootstrap peers."""
        with patch("dubchain.network.discovery.PublicKey") as mock_public_key:
            mock_public_key.generate.return_value = Mock()
            await peer_discovery._initialize_bootstrap_peers()

            assert len(peer_discovery.bootstrap_peers) == 2
            assert peer_discovery.bootstrap_peers[0].address == "127.0.0.1"
            assert peer_discovery.bootstrap_peers[0].port == 8333
            assert peer_discovery.bootstrap_peers[1].port == 8334

    @pytest.mark.asyncio
    async def test_peer_discovery_initialize_bootstrap_peers_invalid_format(
        self, discovery_config
    ):
        """Test initializing bootstrap peers with invalid format."""
        discovery_config.bootstrap_nodes = ["invalid_format"]
        discovery = PeerDiscovery(discovery_config, "test_node_id")

        await discovery._initialize_bootstrap_peers()
        assert len(discovery.bootstrap_peers) == 0

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_bootstrap_peers(self, peer_discovery):
        """Test discovering peers from bootstrap nodes."""
        # Initialize bootstrap peers first
        with patch("dubchain.network.discovery.PublicKey") as mock_public_key:
            mock_public_key.generate.return_value = Mock()
            await peer_discovery._initialize_bootstrap_peers()

        with patch.object(
            peer_discovery, "connect_to_peer", return_value=Mock()
        ) as mock_connect:
            with patch.object(
                peer_discovery, "_request_peer_list", return_value=[]
            ) as mock_request:
                mock_peer = Mock()
                mock_peer.disconnect = AsyncMock()
                mock_connect.return_value = mock_peer

                result = await peer_discovery._discover_bootstrap_peers()

                assert result == []
                mock_connect.assert_called()
                mock_request.assert_called()

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_dns_peers(self, peer_discovery):
        """Test discovering peers using DNS seeds."""
        with patch("dubchain.network.discovery.PublicKey") as mock_public_key:
            mock_public_key.generate.return_value = Mock()
            with patch.object(
                peer_discovery,
                "_resolve_dns_seed",
                return_value=["192.168.1.1", "192.168.1.2"],
            ) as mock_resolve:
                result = await peer_discovery._discover_dns_peers()

                assert len(result) == 2
                assert result[0].address == "192.168.1.1"
                assert result[1].address == "192.168.1.2"
                mock_resolve.assert_called_once_with("seed.example.com")

    @pytest.mark.asyncio
    async def test_peer_discovery_resolve_dns_seed(self, peer_discovery):
        """Test resolving DNS seed to IP addresses."""
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(
                return_value=[
                    (None, None, None, None, ("192.168.1.1", 0)),
                    (None, None, None, None, ("192.168.1.2", 0)),
                ]
            )

            result = await peer_discovery._resolve_dns_seed("example.com")

            assert result == ["192.168.1.1", "192.168.1.2"]

    @pytest.mark.asyncio
    async def test_peer_discovery_resolve_dns_seed_exception(self, peer_discovery):
        """Test DNS resolution with exception."""
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(
                side_effect=Exception("DNS resolution failed")
            )

            result = await peer_discovery._resolve_dns_seed("example.com")

            assert result == []

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_multicast_peers(self, peer_discovery):
        """Test discovering peers using multicast."""
        with patch("dubchain.network.discovery.PublicKey") as mock_public_key:
            mock_public_key.generate.return_value = Mock()
            with patch("socket.socket") as mock_socket:
                mock_sock = Mock()
                mock_socket.return_value = mock_sock
                mock_sock.recvfrom.side_effect = [
                    (
                        json.dumps(
                            {
                                "type": "discovery_response",
                                "peer_id": "multicast_peer",
                                "port": 8333,
                            }
                        ).encode("utf-8"),
                        ("192.168.1.1", 12345),
                    ),
                    socket.timeout(),
                ]

                result = await peer_discovery._discover_multicast_peers()

                assert len(result) == 1
                assert result[0].peer_id == "multicast_peer"
                assert result[0].address == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_peer_discovery_discover_peer_exchange_peers(self, peer_discovery):
        """Test discovering peers through peer exchange."""
        # Add a connected peer
        mock_peer = Mock()
        peer_discovery.connected_peers["peer_1"] = mock_peer

        with patch.object(
            peer_discovery, "_request_peer_list", return_value=[]
        ) as mock_request:
            result = await peer_discovery._discover_peer_exchange_peers()

            assert result == []
            mock_request.assert_called_once_with(mock_peer)

    @pytest.mark.asyncio
    async def test_peer_discovery_exchange_peers(self, peer_discovery):
        """Test exchanging peer information with connected peers."""
        # Add a connected peer
        mock_peer = Mock()
        peer_discovery.connected_peers["peer_1"] = mock_peer

        with patch.object(peer_discovery, "_send_peer_list") as mock_send:
            await peer_discovery._exchange_peers()
            mock_send.assert_called_once_with(mock_peer)

    @pytest.mark.asyncio
    async def test_peer_discovery_exchange_peers_no_peers(self, peer_discovery):
        """Test peer exchange with no connected peers."""
        with patch.object(peer_discovery, "_send_peer_list") as mock_send:
            await peer_discovery._exchange_peers()
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_peer_discovery_request_peer_list(self, peer_discovery):
        """Test requesting peer list from a peer."""
        mock_peer = Mock()
        mock_peer.send_message = AsyncMock(return_value=True)

        with patch("asyncio.sleep"):
            result = await peer_discovery._request_peer_list(mock_peer)

            assert result == []
            mock_peer.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_peer_discovery_send_peer_list(self, peer_discovery, mock_peer_info):
        """Test sending peer list to a peer."""
        await peer_discovery.add_peer(mock_peer_info)

        mock_peer = Mock()
        mock_peer.send_message = AsyncMock()

        await peer_discovery._send_peer_list(mock_peer)

        mock_peer.send_message.assert_called_once()
        # Verify the message contains peer data
        call_args = mock_peer.send_message.call_args[0][0]
        message_data = json.loads(call_args.decode("utf-8"))
        assert message_data["type"] == "peer_list_response"
        assert len(message_data["peers"]) == 1

    @pytest.mark.asyncio
    async def test_peer_discovery_perform_discovery(self, peer_discovery):
        """Test performing peer discovery using multiple methods."""
        with patch.object(
            peer_discovery, "discover_peers", return_value=[]
        ) as mock_discover:
            with patch.object(peer_discovery, "_connect_to_new_peers") as mock_connect:
                await peer_discovery._perform_discovery()

                # Should try multiple discovery methods
                assert mock_discover.call_count >= 1
                mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_peer_discovery_connect_to_new_peers(
        self, peer_discovery, mock_peer_info
    ):
        """Test connecting to newly discovered peers."""
        await peer_discovery.add_peer(mock_peer_info)

        with patch.object(
            peer_discovery, "connect_to_peer", return_value=Mock()
        ) as mock_connect:
            await peer_discovery._connect_to_new_peers()
            mock_connect.assert_called_once_with(mock_peer_info)

    @pytest.mark.asyncio
    async def test_peer_discovery_connect_to_new_peers_max_peers_reached(
        self, peer_discovery, mock_peer_info
    ):
        """Test connecting to new peers when max peers reached."""
        # Fill up connected peers to max
        for i in range(peer_discovery.config.max_peers):
            mock_peer = Mock()
            peer_discovery.connected_peers[f"peer_{i}"] = mock_peer

        # Add a discovered peer
        await peer_discovery.add_peer(mock_peer_info)

        with patch.object(peer_discovery, "connect_to_peer") as mock_connect:
            await peer_discovery._connect_to_new_peers()
            mock_connect.assert_not_called()

    def test_peer_discovery_str_repr(self, peer_discovery):
        """Test string representation of PeerDiscovery."""
        str_repr = str(peer_discovery)
        assert "PeerDiscovery" in str_repr
        assert "test_node_id" in str_repr

        repr_str = repr(peer_discovery)
        assert "PeerDiscovery" in repr_str
        assert "test_node_id" in repr_str
        assert "running=False" in repr_str
