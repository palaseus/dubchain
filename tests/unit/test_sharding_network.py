"""Tests for sharding network module."""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.sharding.shard_network import (
    ShardDiscovery,
    ShardNetwork,
    ShardRouting,
    ShardTopology,
)
from dubchain.sharding.shard_types import ShardId, ShardState, ShardStatus, ShardType


class TestShardTopology:
    """Test ShardTopology functionality."""

    @pytest.fixture
    def topology(self):
        """Fixture for shard topology."""
        return ShardTopology()

    def test_shard_topology_creation(self, topology):
        """Test creating shard topology."""
        assert topology.shard_connections == {}
        assert topology.connection_weights == {}

    def test_add_connection(self, topology):
        """Test adding connection between shards."""
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2, 1.5)

        assert ShardId.SHARD_1 in topology.shard_connections
        assert ShardId.SHARD_2 in topology.shard_connections
        assert ShardId.SHARD_2 in topology.shard_connections[ShardId.SHARD_1]
        assert ShardId.SHARD_1 in topology.shard_connections[ShardId.SHARD_2]
        assert topology.connection_weights[(ShardId.SHARD_1, ShardId.SHARD_2)] == 1.5
        assert topology.connection_weights[(ShardId.SHARD_2, ShardId.SHARD_1)] == 1.5

    def test_add_connection_default_weight(self, topology):
        """Test adding connection with default weight."""
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert topology.connection_weights[(ShardId.SHARD_1, ShardId.SHARD_2)] == 1.0
        assert topology.connection_weights[(ShardId.SHARD_2, ShardId.SHARD_1)] == 1.0

    def test_add_duplicate_connection(self, topology):
        """Test adding duplicate connection."""
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2, 1.0)
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2, 2.0)

        # Should not add duplicate connections
        assert len(topology.shard_connections[ShardId.SHARD_1]) == 1
        assert len(topology.shard_connections[ShardId.SHARD_2]) == 1
        # Weight should be updated
        assert topology.connection_weights[(ShardId.SHARD_1, ShardId.SHARD_2)] == 2.0

    def test_remove_connection(self, topology):
        """Test removing connection between shards."""
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2, 1.5)
        topology.remove_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert ShardId.SHARD_2 not in topology.shard_connections[ShardId.SHARD_1]
        assert ShardId.SHARD_1 not in topology.shard_connections[ShardId.SHARD_2]
        assert (ShardId.SHARD_1, ShardId.SHARD_2) not in topology.connection_weights
        assert (ShardId.SHARD_2, ShardId.SHARD_1) not in topology.connection_weights

    def test_remove_nonexistent_connection(self, topology):
        """Test removing nonexistent connection."""
        # Should not raise error
        topology.remove_connection(ShardId.SHARD_1, ShardId.SHARD_2)

    def test_get_connections(self, topology):
        """Test getting connections for a shard."""
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2)
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_3)

        connections = topology.get_connections(ShardId.SHARD_1)
        assert ShardId.SHARD_2 in connections
        assert ShardId.SHARD_3 in connections
        assert len(connections) == 2

    def test_get_connections_nonexistent_shard(self, topology):
        """Test getting connections for nonexistent shard."""
        connections = topology.get_connections(ShardId.SHARD_1)
        assert connections == []

    def test_get_connection_weight(self, topology):
        """Test getting connection weight."""
        topology.add_connection(ShardId.SHARD_1, ShardId.SHARD_2, 2.5)

        weight = topology.get_connection_weight(ShardId.SHARD_1, ShardId.SHARD_2)
        assert weight == 2.5

        # Test reverse direction
        weight = topology.get_connection_weight(ShardId.SHARD_2, ShardId.SHARD_1)
        assert weight == 2.5

    def test_get_connection_weight_nonexistent(self, topology):
        """Test getting weight for nonexistent connection."""
        weight = topology.get_connection_weight(ShardId.SHARD_1, ShardId.SHARD_2)
        assert weight == 0.0


class TestShardDiscovery:
    """Test ShardDiscovery functionality."""

    @pytest.fixture
    def discovery(self):
        """Fixture for shard discovery."""
        return ShardDiscovery()

    def test_shard_discovery_creation(self, discovery):
        """Test creating shard discovery."""
        assert discovery.known_shards == set()
        assert discovery.discovery_interval == 30.0
        assert discovery.last_discovery > 0

    def test_discover_shards(self, discovery):
        """Test discovering shards."""
        shard_states = {
            ShardId.SHARD_1: ShardState(
                ShardId.SHARD_1, ShardStatus.ACTIVE, ShardType.EXECUTION
            ),
            ShardId.SHARD_2: ShardState(
                ShardId.SHARD_2, ShardStatus.ACTIVE, ShardType.EXECUTION
            ),
            ShardId.SHARD_3: ShardState(
                ShardId.SHARD_3, ShardStatus.INACTIVE, ShardType.EXECUTION
            ),
        }

        discovered = discovery.discover_shards(shard_states)

        assert discovered == {ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3}
        assert discovery.known_shards == {
            ShardId.SHARD_1,
            ShardId.SHARD_2,
            ShardId.SHARD_3,
        }
        assert discovery.last_discovery > 0

    def test_discover_shards_empty(self, discovery):
        """Test discovering shards with empty state."""
        discovered = discovery.discover_shards({})

        assert discovered == set()
        assert discovery.known_shards == set()

    def test_discover_shards_updates_known(self, discovery):
        """Test that discovery updates known shards."""
        # Initial discovery
        shard_states1 = {
            ShardId.SHARD_1: ShardState(
                ShardId.SHARD_1, ShardStatus.ACTIVE, ShardType.EXECUTION
            )
        }
        discovery.discover_shards(shard_states1)

        # Second discovery with additional shards
        shard_states2 = {
            ShardId.SHARD_1: ShardState(
                ShardId.SHARD_1, ShardStatus.ACTIVE, ShardType.EXECUTION
            ),
            ShardId.SHARD_2: ShardState(
                ShardId.SHARD_2, ShardStatus.ACTIVE, ShardType.EXECUTION
            ),
        }
        discovered = discovery.discover_shards(shard_states2)

        assert discovered == {ShardId.SHARD_1, ShardId.SHARD_2}
        assert discovery.known_shards == {ShardId.SHARD_1, ShardId.SHARD_2}

    def test_is_shard_known(self, discovery):
        """Test checking if shard is known."""
        discovery.known_shards.add(ShardId.SHARD_1)

        assert discovery.is_shard_known(ShardId.SHARD_1) == True
        assert discovery.is_shard_known(ShardId.SHARD_2) == False


class TestShardRouting:
    """Test ShardRouting functionality."""

    @pytest.fixture
    def routing(self):
        """Fixture for shard routing."""
        return ShardRouting()

    def test_shard_routing_creation(self, routing):
        """Test creating shard routing."""
        assert routing.routing_table == {}
        assert routing.routing_metrics == {}

    def test_add_route(self, routing):
        """Test adding route between shards."""
        path = [ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3]
        routing.add_route(ShardId.SHARD_1, ShardId.SHARD_3, path)

        assert ShardId.SHARD_1 in routing.routing_table
        assert ShardId.SHARD_3 in routing.routing_table[ShardId.SHARD_1]
        assert routing.routing_table[ShardId.SHARD_1][ShardId.SHARD_3] == path

    def test_add_multiple_routes(self, routing):
        """Test adding multiple routes from same source."""
        path1 = [ShardId.SHARD_1, ShardId.SHARD_2]
        path2 = [ShardId.SHARD_1, ShardId.SHARD_3]

        routing.add_route(ShardId.SHARD_1, ShardId.SHARD_2, path1)
        routing.add_route(ShardId.SHARD_1, ShardId.SHARD_3, path2)

        assert len(routing.routing_table[ShardId.SHARD_1]) == 2
        assert routing.routing_table[ShardId.SHARD_1][ShardId.SHARD_2] == path1
        assert routing.routing_table[ShardId.SHARD_1][ShardId.SHARD_3] == path2

    def test_find_route_same_shard(self, routing):
        """Test finding route to same shard."""
        route = routing.find_route(ShardId.SHARD_1, ShardId.SHARD_1)
        assert route == [ShardId.SHARD_1]

    def test_find_route_existing(self, routing):
        """Test finding existing route."""
        path = [ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3]
        routing.add_route(ShardId.SHARD_1, ShardId.SHARD_3, path)

        route = routing.find_route(ShardId.SHARD_1, ShardId.SHARD_3)
        assert route == path

    def test_find_route_nonexistent(self, routing):
        """Test finding nonexistent route."""
        route = routing.find_route(ShardId.SHARD_1, ShardId.SHARD_2)
        assert route == []

    def test_update_metrics_successful(self, routing):
        """Test updating metrics for successful route."""
        routing.update_metrics(True)

        assert routing.routing_metrics["successful_routes"] == 1
        assert "failed_routes" not in routing.routing_metrics

    def test_update_metrics_failed(self, routing):
        """Test updating metrics for failed route."""
        routing.update_metrics(False)

        assert routing.routing_metrics["failed_routes"] == 1
        assert "successful_routes" not in routing.routing_metrics

    def test_update_metrics_multiple(self, routing):
        """Test updating metrics multiple times."""
        routing.update_metrics(True)
        routing.update_metrics(True)
        routing.update_metrics(False)

        assert routing.routing_metrics["successful_routes"] == 2
        assert routing.routing_metrics["failed_routes"] == 1


class TestShardNetwork:
    """Test ShardNetwork functionality."""

    @pytest.fixture
    def network(self):
        """Fixture for shard network."""
        return ShardNetwork()

    def test_shard_network_creation(self, network):
        """Test creating shard network."""
        assert isinstance(network.topology, ShardTopology)
        assert isinstance(network.discovery, ShardDiscovery)
        assert isinstance(network.routing, ShardRouting)
        assert network.network_metrics == {
            "connections_established": 0,
            "connections_lost": 0,
        }

    def test_add_shard(self, network):
        """Test adding shard to network."""
        network.add_shard(ShardId.SHARD_1)

        assert ShardId.SHARD_1 in network.discovery.known_shards

    def test_remove_shard(self, network):
        """Test removing shard from network."""
        # Add shard and establish connections
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        # Remove shard
        network.remove_shard(ShardId.SHARD_1)

        assert ShardId.SHARD_1 not in network.discovery.known_shards
        # Connections should be removed (but shard entry may still exist with empty connections)
        assert len(network.topology.get_connections(ShardId.SHARD_1)) == 0

    def test_establish_connection_success(self, network):
        """Test establishing connection between known shards."""
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)

        result = network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2, 1.5)

        assert result == True
        assert network.network_metrics["connections_established"] == 1
        assert ShardId.SHARD_2 in network.topology.get_connections(ShardId.SHARD_1)
        assert ShardId.SHARD_1 in network.topology.get_connections(ShardId.SHARD_2)
        # Check routing table
        assert network.routing.find_route(ShardId.SHARD_1, ShardId.SHARD_2) == [
            ShardId.SHARD_1,
            ShardId.SHARD_2,
        ]

    def test_establish_connection_unknown_shard(self, network):
        """Test establishing connection with unknown shard."""
        network.add_shard(ShardId.SHARD_1)

        result = network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert result == False
        assert network.network_metrics["connections_established"] == 0

    def test_establish_connection_both_unknown(self, network):
        """Test establishing connection with both shards unknown."""
        result = network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert result == False
        assert network.network_metrics["connections_established"] == 0

    def test_break_connection(self, network):
        """Test breaking connection between shards."""
        # Establish connection first
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        result = network.break_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        assert result == True
        assert network.network_metrics["connections_lost"] == 1
        assert ShardId.SHARD_2 not in network.topology.get_connections(ShardId.SHARD_1)
        # Check routing table is updated
        assert network.routing.find_route(ShardId.SHARD_1, ShardId.SHARD_2) == []

    def test_discover_network(self, network):
        """Test discovering network topology."""
        shard_states = {
            ShardId.SHARD_1: ShardState(
                ShardId.SHARD_1, ShardStatus.ACTIVE, ShardType.EXECUTION
            ),
            ShardId.SHARD_2: ShardState(
                ShardId.SHARD_2, ShardStatus.ACTIVE, ShardType.EXECUTION
            ),
        }

        discovered = network.discover_network(shard_states)

        assert discovered == {ShardId.SHARD_1, ShardId.SHARD_2}
        assert network.discovery.known_shards == {ShardId.SHARD_1, ShardId.SHARD_2}

    def test_get_route_direct(self, network):
        """Test getting direct route between shards."""
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        route = network.get_route(ShardId.SHARD_1, ShardId.SHARD_2)

        assert route == [ShardId.SHARD_1, ShardId.SHARD_2]
        assert network.routing.routing_metrics["successful_routes"] == 1

    def test_get_route_nonexistent(self, network):
        """Test getting route between unconnected shards."""
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)

        route = network.get_route(ShardId.SHARD_1, ShardId.SHARD_2)

        assert route == []
        assert network.routing.routing_metrics["failed_routes"] == 1

    def test_get_route_same_shard(self, network):
        """Test getting route to same shard."""
        network.add_shard(ShardId.SHARD_1)

        route = network.get_route(ShardId.SHARD_1, ShardId.SHARD_1)

        assert route == [ShardId.SHARD_1]
        assert network.routing.routing_metrics["successful_routes"] == 1

    def test_get_network_metrics(self, network):
        """Test getting network metrics."""
        # Setup network
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)
        network.get_route(ShardId.SHARD_1, ShardId.SHARD_2)

        metrics = network.get_network_metrics()

        assert metrics["known_shards"] == 2
        assert metrics["total_connections"] == 1
        assert metrics["routing_table_size"] == 2  # Two routes (bidirectional)
        assert "network_metrics" in metrics
        assert "routing_metrics" in metrics
        assert metrics["network_metrics"]["connections_established"] == 1
        assert metrics["routing_metrics"]["successful_routes"] == 1

    def test_complex_network_topology(self, network):
        """Test complex network topology with multiple shards."""
        # Add multiple shards
        for shard_id in [
            ShardId.SHARD_1,
            ShardId.SHARD_2,
            ShardId.SHARD_3,
            ShardId.SHARD_4,
        ]:
            network.add_shard(shard_id)

        # Create mesh topology
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)
        network.establish_connection(ShardId.SHARD_2, ShardId.SHARD_3)
        network.establish_connection(ShardId.SHARD_3, ShardId.SHARD_4)
        network.establish_connection(ShardId.SHARD_4, ShardId.SHARD_1)

        # Test direct routes only (multi-hop routing not implemented)
        route1 = network.get_route(ShardId.SHARD_1, ShardId.SHARD_2)
        assert route1 == [ShardId.SHARD_1, ShardId.SHARD_2]

        route2 = network.get_route(ShardId.SHARD_1, ShardId.SHARD_4)
        assert route2 == [ShardId.SHARD_1, ShardId.SHARD_4]

        # Test non-direct route (should fail)
        route3 = network.get_route(ShardId.SHARD_1, ShardId.SHARD_3)
        assert route3 == []  # No direct connection

        # Test metrics
        metrics = network.get_network_metrics()
        assert metrics["known_shards"] == 4
        assert metrics["total_connections"] == 4
        assert metrics["network_metrics"]["connections_established"] == 4

    def test_network_failure_recovery(self, network):
        """Test network failure and recovery."""
        # Setup network
        network.add_shard(ShardId.SHARD_1)
        network.add_shard(ShardId.SHARD_2)
        network.add_shard(ShardId.SHARD_3)
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)
        network.establish_connection(ShardId.SHARD_2, ShardId.SHARD_3)

        # Break connection
        network.break_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        # Route should fail
        route = network.get_route(ShardId.SHARD_1, ShardId.SHARD_3)
        assert route == []

        # Re-establish connection
        network.establish_connection(ShardId.SHARD_1, ShardId.SHARD_2)

        # Direct route should work again
        route = network.get_route(ShardId.SHARD_1, ShardId.SHARD_2)
        assert route == [ShardId.SHARD_1, ShardId.SHARD_2]

        # Multi-hop route should still fail (not implemented)
        route = network.get_route(ShardId.SHARD_1, ShardId.SHARD_3)
        assert route == []

        # Check metrics
        metrics = network.get_network_metrics()
        assert (
            metrics["network_metrics"]["connections_established"] == 3
        )  # 2 initial + 1 re-established
        assert metrics["network_metrics"]["connections_lost"] == 1
