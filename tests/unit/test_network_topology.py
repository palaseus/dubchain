"""
Unit tests for Network Topology implementation.
"""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.network.network_topology import (
    NetworkTopology,
    TopologyConfig,
    TopologyType,
)
from dubchain.network.peer import ConnectionType, PeerInfo, PeerStatus
from dubchain.network.performance import (
    MetricType,
    NetworkPerformance,
    PerformanceConfig,
)
from dubchain.network.security import NetworkSecurity, SecurityConfig, SecurityLevel


class TestTopologyConfig:
    """Test the TopologyConfig class."""

    def test_topology_config_creation(self):
        """Test creating a topology config."""
        config = TopologyConfig()

        assert config.topology_type == TopologyType.MESH
        assert config.max_peers == 100
        assert config.min_peers == 10
        assert config.connection_timeout == 30.0
        assert config.heartbeat_interval == 10.0
        assert config.reconnect_attempts == 3
        assert config.reconnect_delay == 5.0
        assert config.enable_peer_discovery is True
        assert config.enable_peer_filtering is True
        assert config.trusted_peers == []
        assert config.blocked_peers == []

    def test_topology_config_custom_values(self):
        """Test creating a topology config with custom values."""
        config = TopologyConfig(
            topology_type=TopologyType.STAR,
            max_peers=50,
            min_peers=5,
            connection_timeout=60.0,
            heartbeat_interval=20.0,
            reconnect_attempts=5,
            reconnect_delay=10.0,
            enable_peer_discovery=False,
            enable_peer_filtering=False,
            trusted_peers=["peer1", "peer2"],
            blocked_peers=["peer3", "peer4"],
        )

        assert config.topology_type == TopologyType.STAR
        assert config.max_peers == 50
        assert config.min_peers == 5
        assert config.connection_timeout == 60.0
        assert config.heartbeat_interval == 20.0
        assert config.reconnect_attempts == 5
        assert config.reconnect_delay == 10.0
        assert config.enable_peer_discovery is False
        assert config.enable_peer_filtering is False
        assert config.trusted_peers == ["peer1", "peer2"]
        assert config.blocked_peers == ["peer3", "peer4"]


class TestNetworkTopology:
    """Test the NetworkTopology class."""

    @pytest.fixture
    def topology_config(self):
        """Fixture for topology config."""
        return TopologyConfig()

    @pytest.fixture
    def network_topology(self, topology_config):
        """Fixture for NetworkTopology instance."""
        return NetworkTopology(topology_config)

    @pytest.fixture
    def mock_peers(self):
        """Fixture for mock peers."""
        peers = []
        for i in range(5):
            mock_public_key = Mock()
            mock_public_key.to_hex.return_value = f"pubkey_{i}"
            peer = PeerInfo(
                peer_id=f"peer_{i}",
                address=f"192.168.1.{i+1}",
                port=8000 + i,
                public_key=mock_public_key,
                status=PeerStatus.CONNECTED,
                connection_type=ConnectionType.OUTBOUND,
                last_seen=time.time(),
                latency=10.0 + i,
                metadata={"bandwidth": 1000 - i * 100},
            )
            peers.append(peer)
        return peers

    def test_network_topology_creation(self, network_topology):
        """Test creating a NetworkTopology instance."""
        assert network_topology.config is not None
        assert network_topology.peers == {}
        assert network_topology.connections == {}
        assert network_topology.topology_graph is not None
        assert network_topology.peer_discovery is not None
        assert network_topology.performance_monitor is not None
        assert network_topology.security_manager is not None
        assert network_topology.is_initialized is False

    def test_initialize(self, network_topology):
        """Test initializing the network topology."""
        result = network_topology.initialize()
        assert result is True
        assert network_topology.is_initialized is True

    def test_shutdown(self, network_topology):
        """Test shutting down the network topology."""
        network_topology.initialize()
        result = network_topology.shutdown()
        assert result is True
        assert network_topology.is_initialized is False

    def test_add_peer(self, network_topology, mock_peers):
        """Test adding a peer to the topology."""
        network_topology.initialize()
        peer = mock_peers[0]

        result = network_topology.add_peer(peer)
        assert result is True
        assert peer.peer_id in network_topology.peers
        assert network_topology.peers[peer.peer_id] == peer

    def test_remove_peer(self, network_topology, mock_peers):
        """Test removing a peer from the topology."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)

        result = network_topology.remove_peer(peer.peer_id)
        assert result is True
        assert peer.peer_id not in network_topology.peers

    def test_get_peer(self, network_topology, mock_peers):
        """Test getting a peer by ID."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)

        retrieved_peer = network_topology.get_peer(peer.peer_id)
        assert retrieved_peer == peer

        # Test non-existent peer
        retrieved_peer = network_topology.get_peer("non_existent")
        assert retrieved_peer is None

    def test_get_all_peers(self, network_topology, mock_peers):
        """Test getting all peers."""
        network_topology.initialize()

        # Add some peers
        for peer in mock_peers[:3]:
            network_topology.add_peer(peer)

        all_peers = network_topology.get_all_peers()
        assert len(all_peers) == 3
        assert all(
            peer.peer_id in [p.peer_id for p in all_peers] for peer in mock_peers[:3]
        )

    def test_get_connected_peers(self, network_topology, mock_peers):
        """Test getting connected peers."""
        network_topology.initialize()

        # Add peers with different statuses
        connected_peer = mock_peers[0]
        connected_peer.status = PeerStatus.CONNECTED
        network_topology.add_peer(connected_peer)

        disconnected_peer = mock_peers[1]
        disconnected_peer.status = PeerStatus.DISCONNECTED
        network_topology.add_peer(disconnected_peer)

        connected_peers = network_topology.get_connected_peers()
        assert len(connected_peers) == 1
        assert connected_peers[0].peer_id == connected_peer.peer_id

    def test_connect_peer(self, network_topology, mock_peers):
        """Test connecting to a peer."""
        network_topology.initialize()
        peer = mock_peers[0]

        result = network_topology.connect_peer(peer)
        assert result is True
        assert peer.peer_id in network_topology.connections
        assert network_topology.connections[peer.peer_id] == peer

    def test_disconnect_peer(self, network_topology, mock_peers):
        """Test disconnecting from a peer."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.connect_peer(peer)

        result = network_topology.disconnect_peer(peer.peer_id)
        assert result is True
        assert peer.peer_id not in network_topology.connections

    def test_update_peer_status(self, network_topology, mock_peers):
        """Test updating peer status."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)

        result = network_topology.update_peer_status(
            peer.peer_id, PeerStatus.DISCONNECTED
        )
        assert result is True
        assert network_topology.peers[peer.peer_id].status == PeerStatus.DISCONNECTED

    def test_get_peer_count(self, network_topology, mock_peers):
        """Test getting peer count."""
        network_topology.initialize()

        # Add some peers
        for peer in mock_peers[:3]:
            network_topology.add_peer(peer)

        count = network_topology.get_peer_count()
        assert count == 3

    def test_get_connection_count(self, network_topology, mock_peers):
        """Test getting connection count."""
        network_topology.initialize()

        # Connect to some peers
        for peer in mock_peers[:2]:
            network_topology.connect_peer(peer)

        count = network_topology.get_connection_count()
        assert count == 2

    def test_is_peer_connected(self, network_topology, mock_peers):
        """Test checking if a peer is connected."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)
        network_topology.connect_peer(peer)

        # Test connected peer
        assert network_topology.is_peer_connected(peer.peer_id) is True

        # Test disconnected peer
        network_topology.disconnect_peer(peer.peer_id)
        assert network_topology.is_peer_connected(peer.peer_id) is False

        # Test non-existent peer
        assert network_topology.is_peer_connected("non_existent") is False

    def test_get_peer_latency(self, network_topology, mock_peers):
        """Test getting peer latency."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)

        latency = network_topology.get_peer_latency(peer.peer_id)
        assert latency == peer.latency

        # Test non-existent peer
        latency = network_topology.get_peer_latency("non_existent")
        assert latency is None

    def test_get_peer_bandwidth(self, network_topology, mock_peers):
        """Test getting peer bandwidth."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)

        bandwidth = network_topology.get_peer_bandwidth(peer.peer_id)
        assert bandwidth == peer.metadata["bandwidth"]

        # Test non-existent peer
        bandwidth = network_topology.get_peer_bandwidth("non_existent")
        assert bandwidth is None

    def test_discover_peers(self, network_topology):
        """Test discovering peers."""
        network_topology.initialize()

        # Test peer discovery (returns empty list by default)
        peers = network_topology.discover_peers()
        assert isinstance(peers, list)

    def test_filter_peer(self, network_topology, mock_peers):
        """Test filtering peers."""
        network_topology.initialize()
        peer = mock_peers[0]

        # Test allowed peer
        result = network_topology.filter_peer(peer.peer_id)
        assert result is True

        # Test blocked peer
        network_topology.config.blocked_peers = [peer.peer_id]
        result = network_topology.filter_peer(peer.peer_id)
        assert result is False

    def test_get_topology_info(self, network_topology, mock_peers):
        """Test getting topology information."""
        network_topology.initialize()

        # Add some peers
        for peer in mock_peers[:3]:
            network_topology.add_peer(peer)

        info = network_topology.get_topology_info()
        assert info is not None
        assert "topology_type" in info
        assert "peer_count" in info
        assert "connection_count" in info
        assert info["peer_count"] == 3

    def test_optimize_topology(self, network_topology, mock_peers):
        """Test optimizing the topology."""
        network_topology.initialize()

        # Add some peers
        for peer in mock_peers:
            network_topology.add_peer(peer)

        result = network_topology.optimize_topology()
        assert result is True

    def test_handle_peer_failure(self, network_topology, mock_peers):
        """Test handling peer failure."""
        network_topology.initialize()
        peer = mock_peers[0]
        network_topology.add_peer(peer)

        result = network_topology.handle_peer_failure(peer.peer_id)
        assert result is True
        assert network_topology.peers[peer.peer_id].status == PeerStatus.ERROR

    def test_reconnect_failed_peers(self, network_topology, mock_peers):
        """Test reconnecting to failed peers."""
        network_topology.initialize()

        # Add a disconnected peer
        peer = mock_peers[0]
        peer.status = PeerStatus.DISCONNECTED
        network_topology.add_peer(peer)

        result = network_topology.reconnect_failed_peers()
        assert result is True

    def test_get_network_metrics(self, network_topology, mock_peers):
        """Test getting network metrics."""
        network_topology.initialize()

        # Add some peers
        for peer in mock_peers[:3]:
            network_topology.add_peer(peer)

        metrics = network_topology.get_network_metrics()
        assert metrics is not None
        assert "peer_count" in metrics
        assert "connection_count" in metrics
        assert "topology_type" in metrics
        assert metrics["peer_count"] == 3

    def test_validate_topology(self, network_topology, mock_peers):
        """Test validating the topology."""
        network_topology.initialize()

        # Test empty topology
        result = network_topology.validate_topology()
        assert result is True  # Empty topology is valid

        # Add some peers
        for peer in mock_peers[:3]:
            network_topology.add_peer(peer)

        result = network_topology.validate_topology()
        assert result is True  # Should pass with peers
