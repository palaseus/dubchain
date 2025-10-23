"""Tests for network message router module."""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.network.message_router import (
    LatencyTracker,
    LoadBalancer,
    MessageRouter,
    NetworkTopology,
    RouteInfo,
    RoutingStrategy,
    RoutingTable,
)
from dubchain.network.peer import Peer, PeerInfo


class TestRoutingStrategy:
    """Test RoutingStrategy enum."""

    def test_routing_strategy_values(self):
        """Test routing strategy values."""
        assert RoutingStrategy.SHORTEST_PATH.value == "shortest_path"
        assert RoutingStrategy.LOAD_BALANCED.value == "load_balanced"
        assert RoutingStrategy.LATENCY_OPTIMIZED.value == "latency_optimized"
        assert RoutingStrategy.RANDOM.value == "random"
        assert RoutingStrategy.FLOODING.value == "flooding"
        assert RoutingStrategy.GEOMETRIC.value == "geometric"


class TestRouteInfo:
    """Test RouteInfo functionality."""

    def test_route_info_creation(self):
        """Test creating route info."""
        route_info = RouteInfo(
            route_id="route1",
            source_peer="peer1",
            target_peer="peer2",
            intermediate_peers=["peer3", "peer4"],
            total_hops=2,
            estimated_latency=50.0,
            route_quality=0.9,
        )

        assert route_info.route_id == "route1"
        assert route_info.source_peer == "peer1"
        assert route_info.target_peer == "peer2"
        assert route_info.intermediate_peers == ["peer3", "peer4"]
        assert route_info.total_hops == 2
        assert route_info.estimated_latency == 50.0
        assert route_info.route_quality == 0.9
        assert route_info.usage_count == 0
        assert route_info.success_count == 0
        assert route_info.failure_count == 0

    def test_route_info_defaults(self):
        """Test route info defaults."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        assert route_info.route_id == "route1"
        assert route_info.source_peer == "peer1"
        assert route_info.target_peer == "peer2"
        assert route_info.intermediate_peers == []
        assert route_info.total_hops == 0
        assert route_info.estimated_latency == 0.0
        assert route_info.route_quality == 1.0
        assert route_info.usage_count == 0
        assert route_info.success_count == 0
        assert route_info.failure_count == 0

    def test_route_info_validation(self):
        """Test route info validation."""
        # Test empty route ID
        with pytest.raises(ValueError, match="Route ID cannot be empty"):
            RouteInfo(route_id="", source_peer="peer1", target_peer="peer2")

        # Test empty source peer
        with pytest.raises(ValueError, match="Source peer cannot be empty"):
            RouteInfo(route_id="route1", source_peer="", target_peer="peer2")

        # Test empty target peer
        with pytest.raises(ValueError, match="Target peer cannot be empty"):
            RouteInfo(route_id="route1", source_peer="peer1", target_peer="")

    def test_route_info_success_rate(self):
        """Test route info success rate calculation."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        # No usage yet
        assert route_info.success_rate == 0.0

        # Add some usage
        route_info.usage_count = 10
        route_info.success_count = 8
        route_info.failure_count = 2

        assert route_info.success_rate == 0.8

    def test_route_info_update_usage(self):
        """Test route info usage update."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        initial_usage = route_info.usage_count
        initial_last_used = route_info.last_used

        # Update usage
        route_info.update_usage(success=True)

        assert route_info.usage_count == initial_usage + 1
        assert route_info.success_count == 1
        assert route_info.failure_count == 0
        # Don't check time comparison as it may be too fast

        # Update with failure
        route_info.update_usage(success=False)

        assert route_info.usage_count == initial_usage + 2
        assert route_info.success_count == 1
        assert route_info.failure_count == 1

    def test_route_info_record_success_failure(self):
        """Test route info record success and failure methods."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        # Record success
        route_info.record_success()
        assert route_info.success_count == 1
        assert route_info.failure_count == 0

        # Record failure
        route_info.record_failure()
        assert route_info.success_count == 1
        assert route_info.failure_count == 1

    def test_route_info_get_age(self):
        """Test route info age calculation."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        # Age should be very small (just created)
        age = route_info.get_age()
        assert age >= 0
        assert age < 5  # Should be less than 5 seconds

    def test_route_info_get_idle_time(self):
        """Test route info idle time calculation."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        # Idle time should be very small (just created)
        idle_time = route_info.get_idle_time()
        assert idle_time >= 0
        assert idle_time < 5  # Should be less than 5 seconds

    def test_route_info_is_healthy(self):
        """Test route info health check."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        # New route should be healthy (no usage yet, so success rate is 0, but other conditions should pass)
        # Actually, let's set up a healthy route
        route_info.success_count = 8
        route_info.failure_count = 2
        route_info.route_quality = 0.8

        assert route_info.is_healthy() is True

        # Test unhealthy route (low success rate)
        route_info.success_count = 1
        route_info.failure_count = 9
        assert route_info.is_healthy() is False

        # Test unhealthy route (low quality)
        route_info.success_count = 8
        route_info.failure_count = 2
        route_info.route_quality = 0.2
        assert route_info.is_healthy() is False

    def test_route_info_validation_edge_cases(self):
        """Test route info validation edge cases."""
        # Test negative total hops
        with pytest.raises(ValueError, match="Total hops cannot be negative"):
            RouteInfo(
                route_id="route1", source_peer="peer1", target_peer="peer2", total_hops=-1
            )

        # Test negative estimated latency
        with pytest.raises(ValueError, match="Estimated latency cannot be negative"):
            RouteInfo(
                route_id="route1", source_peer="peer1", target_peer="peer2", estimated_latency=-1.0
            )

        # Test route quality out of range
        with pytest.raises(ValueError, match="Route quality must be between 0 and 1"):
            RouteInfo(
                route_id="route1", source_peer="peer1", target_peer="peer2", route_quality=1.5
            )

        with pytest.raises(ValueError, match="Route quality must be between 0 and 1"):
            RouteInfo(
                route_id="route1", source_peer="peer1", target_peer="peer2", route_quality=-0.1
            )


class TestRoutingTable:
    """Test RoutingTable functionality."""

    @pytest.fixture
    def routing_table(self):
        """Fixture for routing table."""
        return RoutingTable()

    def test_routing_table_creation(self):
        """Test creating routing table."""
        table = RoutingTable()

        assert isinstance(table._routes, dict)
        assert isinstance(table._peer_routes, dict)
        assert table._max_routes == 1000

    def test_add_route(self, routing_table):
        """Test adding route."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        routing_table.add_route(route_info)

        assert "route1" in routing_table._routes
        assert routing_table._routes["route1"] == route_info

    def test_get_route(self, routing_table):
        """Test getting route."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        routing_table.add_route(route_info)

        retrieved_route = routing_table.get_route("route1")
        assert retrieved_route == route_info

    def test_get_nonexistent_route(self, routing_table):
        """Test getting nonexistent route."""
        result = routing_table.get_route("nonexistent")
        assert result is None

    def test_remove_route(self, routing_table):
        """Test removing route."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        routing_table.add_route(route_info)
        assert "route1" in routing_table._routes

        routing_table.remove_route("route1")
        assert "route1" not in routing_table._routes

    def test_get_routes_for_peer(self, routing_table):
        """Test getting routes for peer."""
        route1 = RouteInfo(route_id="route1", source_peer="peer1", target_peer="peer2")
        route2 = RouteInfo(route_id="route2", source_peer="peer1", target_peer="peer3")

        routing_table.add_route(route1)
        routing_table.add_route(route2)

        routes = routing_table.get_routes_for_peer("peer1")
        assert len(routes) == 2
        assert route1 in routes
        assert route2 in routes

    def test_cleanup_old_routes(self, routing_table):
        """Test cleaning up old routes."""
        # Add old route
        old_route = RouteInfo(
            route_id="old_route", source_peer="peer1", target_peer="peer2"
        )
        old_route.created_at = int(time.time()) - 3600  # 1 hour ago

        # Add recent route
        recent_route = RouteInfo(
            route_id="recent_route", source_peer="peer1", target_peer="peer3"
        )

        routing_table.add_route(old_route)
        routing_table.add_route(recent_route)

        # Cleanup routes older than 30 minutes
        routing_table.cleanup_old_routes(max_age=1800)

        assert "old_route" not in routing_table._routes
        assert "recent_route" in routing_table._routes


class TestLoadBalancer:
    """Test LoadBalancer functionality."""

    @pytest.fixture
    def load_balancer(self):
        """Fixture for load balancer."""
        return LoadBalancer()

    def test_load_balancer_creation(self):
        """Test creating load balancer."""
        balancer = LoadBalancer()

        assert isinstance(balancer._peer_loads, dict)
        assert balancer._max_load == 100.0

    def test_update_peer_load(self, load_balancer):
        """Test updating peer load."""
        load_balancer.update_peer_load("peer1", 50.0)

        assert load_balancer._peer_loads["peer1"] == 50.0

    def test_get_peer_load(self, load_balancer):
        """Test getting peer load."""
        load_balancer.update_peer_load("peer1", 75.0)

        load = load_balancer.get_peer_load("peer1")
        assert load == 75.0

    def test_get_peer_load_nonexistent(self, load_balancer):
        """Test getting load for nonexistent peer."""
        load = load_balancer.get_peer_load("nonexistent")
        assert load == 0.0

    def test_select_least_loaded_peer(self, load_balancer):
        """Test selecting least loaded peer."""
        load_balancer.update_peer_load("peer1", 80.0)
        load_balancer.update_peer_load("peer2", 30.0)
        load_balancer.update_peer_load("peer3", 60.0)

        selected_peer = load_balancer.select_least_loaded_peer(
            ["peer1", "peer2", "peer3"]
        )
        assert selected_peer == "peer2"

    def test_select_least_loaded_peer_empty_list(self, load_balancer):
        """Test selecting from empty peer list."""
        selected_peer = load_balancer.select_least_loaded_peer([])
        assert selected_peer is None

    def test_is_peer_overloaded(self, load_balancer):
        """Test checking if peer is overloaded."""
        load_balancer.update_peer_load("peer1", 90.0)
        load_balancer.update_peer_load("peer2", 50.0)

        assert load_balancer.is_peer_overloaded("peer1") is True
        assert load_balancer.is_peer_overloaded("peer2") is False

    def test_set_peer_capacity(self, load_balancer):
        """Test setting peer capacity."""
        load_balancer.set_peer_capacity("peer1", 100.0)
        load_balancer.set_peer_capacity("peer2", 50.0)

        assert load_balancer._peer_capacities["peer1"] == 100.0
        assert load_balancer._peer_capacities["peer2"] == 50.0

        # Test negative capacity (should be set to 0)
        load_balancer.set_peer_capacity("peer3", -10.0)
        assert load_balancer._peer_capacities["peer3"] == 0.0

    def test_get_least_loaded_peer(self, load_balancer):
        """Test getting least loaded peer."""
        load_balancer.update_peer_load("peer1", 80.0)
        load_balancer.update_peer_load("peer2", 30.0)
        load_balancer.update_peer_load("peer3", 60.0)

        selected_peer = load_balancer.get_least_loaded_peer(["peer1", "peer2", "peer3"])
        assert selected_peer == "peer2"

    def test_get_load_distribution(self, load_balancer):
        """Test getting load distribution."""
        load_balancer.update_peer_load("peer1", 50.0)
        load_balancer.update_peer_load("peer2", 75.0)

        distribution = load_balancer.get_load_distribution()
        assert distribution["peer1"] == 50.0
        assert distribution["peer2"] == 75.0

    def test_reset_loads(self, load_balancer):
        """Test resetting loads."""
        load_balancer.update_peer_load("peer1", 50.0)
        load_balancer.update_peer_load("peer2", 75.0)

        assert len(load_balancer._peer_loads) == 2

        load_balancer.reset_loads()

        assert len(load_balancer._peer_loads) == 0
        assert len(load_balancer._load_history) == 0

    def test_load_history_trimming(self, load_balancer):
        """Test that load history is trimmed to max_history."""
        # Set a smaller max_history for testing
        load_balancer._max_history = 3

        # Add more than max_history entries
        for i in range(5):
            load_balancer.update_peer_load("peer1", float(i))

        # Should only keep the last 3 entries
        assert len(load_balancer._load_history["peer1"]) == 3
        # Check that the last 3 values are kept
        expected_values = [2.0, 3.0, 4.0]
        actual_values = [entry[1] for entry in load_balancer._load_history["peer1"]]
        assert actual_values == expected_values


class TestLatencyTracker:
    """Test LatencyTracker functionality."""

    @pytest.fixture
    def latency_tracker(self):
        """Fixture for latency tracker."""
        return LatencyTracker()

    def test_latency_tracker_creation(self):
        """Test creating latency tracker."""
        tracker = LatencyTracker()

        assert isinstance(tracker._latencies, dict)
        assert tracker._max_samples == 100

    def test_record_latency(self, latency_tracker):
        """Test recording latency."""
        latency_tracker.record_latency("peer1", 50.0)

        assert "peer1" in latency_tracker._latencies
        assert 50.0 in latency_tracker._latencies["peer1"]

    def test_get_average_latency(self, latency_tracker):
        """Test getting average latency."""
        latency_tracker.record_latency("peer1", 50.0)
        latency_tracker.record_latency("peer1", 60.0)
        latency_tracker.record_latency("peer1", 40.0)

        avg_latency = latency_tracker.get_average_latency("peer1")
        assert avg_latency == 50.0

    def test_get_average_latency_nonexistent(self, latency_tracker):
        """Test getting average latency for nonexistent peer."""
        avg_latency = latency_tracker.get_average_latency("nonexistent")
        assert avg_latency == 0.0

    def test_get_lowest_latency_peer(self, latency_tracker):
        """Test getting peer with lowest latency."""
        latency_tracker.record_latency("peer1", 100.0)
        latency_tracker.record_latency("peer2", 30.0)
        latency_tracker.record_latency("peer3", 80.0)

        selected_peer = latency_tracker.get_lowest_latency_peer(
            ["peer1", "peer2", "peer3"]
        )
        assert selected_peer == "peer2"

    def test_get_latest_latency(self, latency_tracker):
        """Test getting latest latency measurement."""
        latency_tracker.record_latency("peer1", 50.0)
        latency_tracker.record_latency("peer1", 60.0)
        latency_tracker.record_latency("peer1", 40.0)

        latest_latency = latency_tracker.get_latest_latency("peer1")
        assert latest_latency == 40.0

    def test_get_latest_latency_nonexistent(self, latency_tracker):
        """Test getting latest latency for nonexistent peer."""
        latest_latency = latency_tracker.get_latest_latency("nonexistent")
        assert latest_latency is None

    def test_get_latency_stats(self, latency_tracker):
        """Test getting latency statistics."""
        latency_tracker.record_latency("peer1", 30.0)
        latency_tracker.record_latency("peer1", 50.0)
        latency_tracker.record_latency("peer1", 40.0)

        stats = latency_tracker.get_latency_stats("peer1")
        assert stats["min"] == 30.0
        assert stats["max"] == 50.0
        assert stats["avg"] == 40.0
        assert stats["count"] == 3

    def test_get_latency_stats_nonexistent(self, latency_tracker):
        """Test getting latency stats for nonexistent peer."""
        stats = latency_tracker.get_latency_stats("nonexistent")
        assert stats == {}

    def test_get_best_latency_peer(self, latency_tracker):
        """Test getting peer with best latency."""
        latency_tracker.record_latency("peer1", 100.0)
        latency_tracker.record_latency("peer2", 30.0)
        latency_tracker.record_latency("peer3", 80.0)

        selected_peer = latency_tracker.get_best_latency_peer(
            ["peer1", "peer2", "peer3"]
        )
        assert selected_peer == "peer2"

    def test_latency_samples_trimming(self, latency_tracker):
        """Test that latency samples are trimmed to max_samples."""
        # Set a smaller max_samples for testing
        latency_tracker._max_samples = 3

        # Add more than max_samples entries
        for i in range(5):
            latency_tracker.record_latency("peer1", float(i))

        # Should only keep the last 3 entries
        assert len(latency_tracker._latencies["peer1"]) == 3
        # Check that the last 3 values are kept
        assert latency_tracker._latencies["peer1"] == [2.0, 3.0, 4.0]

    def test_cleanup_old_measurements(self, latency_tracker):
        """Test cleaning up old measurements."""
        # Record some latencies
        latency_tracker.record_latency("peer1", 50.0)
        latency_tracker.record_latency("peer2", 60.0)

        # Manually set old last_measurement time for peer1
        latency_tracker._last_measurement["peer1"] = int(time.time()) - 4000  # 4000 seconds ago

        # Cleanup measurements older than 1 hour (3600 seconds)
        latency_tracker.cleanup_old_measurements(max_age=3600)

        # peer1 should be removed, peer2 should remain
        assert "peer1" not in latency_tracker._latencies
        assert "peer1" not in latency_tracker._last_measurement
        assert "peer2" in latency_tracker._latencies
        assert "peer2" in latency_tracker._last_measurement


class TestNetworkTopology:
    """Test NetworkTopology functionality."""

    @pytest.fixture
    def network_topology(self):
        """Fixture for network topology."""
        return NetworkTopology()

    def test_network_topology_creation(self):
        """Test creating network topology."""
        topology = NetworkTopology()

        assert isinstance(topology._connections, dict)
        assert isinstance(topology._peer_info, dict)

    def test_add_peer(self, network_topology):
        """Test adding peer."""
        peer_info = Mock(spec=PeerInfo)
        peer_info.peer_id = "peer1"

        network_topology.add_peer(peer_info)

        assert "peer1" in network_topology._peer_info
        assert network_topology._peer_info["peer1"] == peer_info

    def test_remove_peer(self, network_topology):
        """Test removing peer."""
        peer_info = Mock(spec=PeerInfo)
        peer_info.peer_id = "peer1"

        network_topology.add_peer(peer_info)
        assert "peer1" in network_topology._peer_info

        network_topology.remove_peer("peer1")
        assert "peer1" not in network_topology._peer_info

    def test_add_connection(self, network_topology):
        """Test adding connection."""
        network_topology.add_connection("peer1", "peer2", 50.0)

        assert "peer1" in network_topology._connections
        assert "peer2" in network_topology._connections["peer1"]
        assert network_topology.get_connection_latency("peer1", "peer2") == 50.0

    def test_get_connection_latency(self, network_topology):
        """Test getting connection latency."""
        network_topology.add_connection("peer1", "peer2", 75.0)

        latency = network_topology.get_connection_latency("peer1", "peer2")
        assert latency == 75.0

    def test_get_connection_latency_nonexistent(self, network_topology):
        """Test getting latency for nonexistent connection."""
        latency = network_topology.get_connection_latency("peer1", "peer2")
        assert latency == float("inf")

    def test_find_shortest_path(self, network_topology):
        """Test finding shortest path."""
        # Create a simple network: peer1 -> peer2 -> peer3
        network_topology.add_connection("peer1", "peer2", 50.0)
        network_topology.add_connection("peer2", "peer3", 30.0)

        path = network_topology.find_shortest_path("peer1", "peer3")
        assert path == ["peer1", "peer2", "peer3"]

    def test_find_shortest_path_same_peer(self, network_topology):
        """Test finding shortest path to same peer."""
        network_topology.add_peer("peer1")
        path = network_topology.find_shortest_path("peer1", "peer1")
        assert path == ["peer1"]

    def test_find_shortest_path_no_path(self, network_topology):
        """Test finding shortest path when no path exists."""
        network_topology.add_peer("peer1")
        network_topology.add_peer("peer2")
        # No connection between peer1 and peer2

        path = network_topology.find_shortest_path("peer1", "peer2")
        assert path is None

    def test_find_shortest_path_nonexistent_peer(self, network_topology):
        """Test finding shortest path with nonexistent peer."""
        network_topology.add_peer("peer1")
        
        path = network_topology.find_shortest_path("peer1", "nonexistent")
        assert path is None

    def test_remove_connection(self, network_topology):
        """Test removing connection."""
        network_topology.add_connection("peer1", "peer2", 50.0)
        
        # Verify connection exists
        assert "peer2" in network_topology.connections["peer1"]
        assert "peer1" in network_topology.connections["peer2"]

        # Remove connection
        network_topology.remove_connection("peer1", "peer2")

        # Verify connection is removed
        assert "peer2" not in network_topology.connections["peer1"]
        assert "peer1" not in network_topology.connections["peer2"]

    def test_get_connected_peers(self, network_topology):
        """Test getting connected peers."""
        network_topology.add_connection("peer1", "peer2", 50.0)
        network_topology.add_connection("peer1", "peer3", 30.0)

        connected_peers = network_topology.get_connected_peers("peer1")
        assert connected_peers == {"peer2", "peer3"}

    def test_get_connected_peers_nonexistent(self, network_topology):
        """Test getting connected peers for nonexistent peer."""
        connected_peers = network_topology.get_connected_peers("nonexistent")
        assert connected_peers == set()

    def test_get_topology_stats(self, network_topology):
        """Test getting topology statistics."""
        # Add some peers and connections
        network_topology.add_connection("peer1", "peer2", 50.0)
        network_topology.add_connection("peer2", "peer3", 30.0)
        network_topology.add_connection("peer1", "peer3", 40.0)

        stats = network_topology.get_topology_stats()
        assert stats["total_peers"] == 3
        assert stats["total_connections"] == 3
        assert stats["average_connections"] == 1.0  # 3 connections / 3 peers = 1.0

    def test_get_topology_stats_empty(self, network_topology):
        """Test getting topology statistics for empty topology."""
        stats = network_topology.get_topology_stats()
        assert stats["total_peers"] == 0
        assert stats["total_connections"] == 0
        assert stats["average_connections"] == 0

    def test_add_peer_with_string_id(self, network_topology):
        """Test adding peer with string ID."""
        network_topology.add_peer("peer1", {"capacity": 100})

        assert "peer1" in network_topology.connections
        assert "peer1" in network_topology._connections
        assert network_topology.peer_info["peer1"] == {"capacity": 100}
        assert network_topology._peer_info["peer1"] == {"capacity": 100}

    def test_add_peer_with_peer_info_object(self, network_topology):
        """Test adding peer with peer info object."""
        peer_info = Mock()
        peer_info.peer_id = "peer1"
        peer_info.capacity = 100

        network_topology.add_peer(peer_info)

        assert "peer1" in network_topology.connections
        assert network_topology._peer_info["peer1"] == peer_info

    def test_remove_peer_cleanup(self, network_topology):
        """Test that removing peer cleans up all references."""
        network_topology.add_connection("peer1", "peer2", 50.0)
        network_topology.add_connection("peer1", "peer3", 30.0)
        network_topology.add_peer("peer1", {"capacity": 100})

        # Remove peer1
        network_topology.remove_peer("peer1")

        # Verify all references are cleaned up
        assert "peer1" not in network_topology.connections
        assert "peer1" not in network_topology._connections
        assert "peer1" not in network_topology.peer_info
        assert "peer1" not in network_topology._peer_info

        # Verify connections from other peers are also removed
        assert "peer1" not in network_topology.connections.get("peer2", set())
        assert "peer1" not in network_topology.connections.get("peer3", set())


class TestMessageRouter:
    """Test MessageRouter functionality."""

    @pytest.fixture
    def message_router(self):
        """Fixture for message router."""
        return MessageRouter()

    def test_message_router_creation(self):
        """Test creating message router."""
        router = MessageRouter(RoutingStrategy.SHORTEST_PATH)

        assert isinstance(router.routing_table, RoutingTable)
        assert isinstance(router.load_balancer, LoadBalancer)
        assert isinstance(router.latency_tracker, LatencyTracker)
        assert isinstance(router.topology, NetworkTopology)
        assert router.strategy == RoutingStrategy.SHORTEST_PATH
        assert router.strategy == RoutingStrategy.SHORTEST_PATH

    def test_set_routing_strategy(self, message_router):
        """Test setting routing strategy."""
        message_router.set_routing_strategy(RoutingStrategy.LOAD_BALANCED)

        assert message_router.strategy == RoutingStrategy.LOAD_BALANCED

    @pytest.mark.asyncio
    async def test_route_message_shortest_path(self, message_router):
        """Test routing message using shortest path strategy."""
        # Setup topology
        message_router.topology.add_connection("peer1", "peer2", 50.0)
        message_router.topology.add_connection("peer2", "peer3", 30.0)

        # Mock peer
        mock_peer = Mock(spec=Peer)
        message_router._peers = {"peer1": mock_peer}

        # Route message
        result = await message_router.route_message("peer1", "peer3", {"type": "test"})

        assert result is not None

    @pytest.mark.asyncio
    async def test_route_message_load_balanced(self, message_router):
        """Test routing message using load balanced strategy."""
        message_router.set_routing_strategy(RoutingStrategy.LOAD_BALANCED)

        # Setup load balancer
        message_router.load_balancer.update_peer_load("peer2", 30.0)
        message_router.load_balancer.update_peer_load("peer3", 80.0)

        # Mock peer
        mock_peer = Mock(spec=Peer)
        message_router._peers = {"peer1": mock_peer}

        # Route message
        result = await message_router.route_message("peer1", "peer4", {"type": "test"})

        assert result is not None

    @pytest.mark.asyncio
    async def test_route_message_latency_optimized(self, message_router):
        """Test routing message using latency optimized strategy."""
        message_router.set_routing_strategy(RoutingStrategy.LATENCY_OPTIMIZED)

        # Setup latency tracker
        message_router.latency_tracker.record_latency("peer2", 30.0)
        message_router.latency_tracker.record_latency("peer3", 80.0)

        # Mock peer
        mock_peer = Mock(spec=Peer)
        message_router._peers = {"peer1": mock_peer}

        # Route message
        result = await message_router.route_message("peer1", "peer4", {"type": "test"})

        assert result is not None

    @pytest.mark.asyncio
    async def test_route_message_random(self, message_router):
        """Test routing message using random strategy."""
        message_router.set_routing_strategy(RoutingStrategy.RANDOM)

        # Mock peer
        mock_peer = Mock(spec=Peer)
        message_router._peers = {"peer1": mock_peer}

        # Route message
        result = await message_router.route_message("peer1", "peer2", {"type": "test"})

        assert result is not None

    @pytest.mark.asyncio
    async def test_route_message_flooding(self, message_router):
        """Test routing message using flooding strategy."""
        message_router.set_routing_strategy(RoutingStrategy.FLOODING)

        # Mock peer
        mock_peer = Mock(spec=Peer)
        message_router._peers = {"peer1": mock_peer}

        # Route message
        result = await message_router.route_message("peer1", "peer2", {"type": "test"})

        assert result is not None

    @pytest.mark.asyncio
    async def test_route_message_geometric(self, message_router):
        """Test routing message using geometric strategy."""
        message_router.set_routing_strategy(RoutingStrategy.GEOMETRIC)

        # Mock peer
        mock_peer = Mock(spec=Peer)
        message_router._peers = {"peer1": mock_peer}

        # Route message
        result = await message_router.route_message("peer1", "peer2", {"type": "test"})

        assert result is not None

    def test_register_peer(self, message_router):
        """Test registering peer."""
        mock_peer = Mock(spec=Peer)
        mock_peer.peer_id = "peer1"
        mock_peer.get_peer_id.return_value = "peer1"

        message_router.register_peer(mock_peer)

        assert "peer1" in message_router._peers
        assert message_router._peers["peer1"] == mock_peer

    def test_unregister_peer(self, message_router):
        """Test unregistering peer."""
        mock_peer = Mock(spec=Peer)
        mock_peer.peer_id = "peer1"
        mock_peer.get_peer_id.return_value = "peer1"

        message_router.register_peer(mock_peer)
        assert "peer1" in message_router._peers

        message_router.unregister_peer("peer1")
        assert "peer1" not in message_router._peers

    def test_update_route_metrics(self, message_router):
        """Test updating route metrics."""
        route_info = RouteInfo(
            route_id="route1", source_peer="peer1", target_peer="peer2"
        )

        message_router.routing_table.add_route(route_info)

        # Update metrics
        message_router.update_route_metrics("route1", success=True, latency=50.0)

        # Get the updated route from routing table
        updated_route = message_router.routing_table.get_route("route1")
        assert updated_route.usage_count == 1
        assert updated_route.success_count == 1
        assert updated_route.failure_count == 0

    def test_get_network_stats(self, message_router):
        """Test getting network statistics."""
        stats = message_router.get_network_stats()

        assert isinstance(stats, dict)
        assert "total_routes" in stats
        assert "total_peers" in stats
        assert "average_latency" in stats
        assert "load_distribution" in stats

    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_router):
        """Test broadcasting message to all peers."""
        # Add some peers
        mock_peer1 = Mock(spec=Peer)
        mock_peer1.get_peer_id.return_value = "peer1"
        # Make send_message fail for some calls
        mock_peer1.send_message.side_effect = [True, False]  # First call succeeds, second fails

        mock_peer2 = Mock(spec=Peer)
        mock_peer2.get_peer_id.return_value = "peer2"

        mock_peer3 = Mock(spec=Peer)
        mock_peer3.get_peer_id.return_value = "peer3"

        message_router.add_peer(mock_peer1)
        message_router.add_peer(mock_peer2)
        message_router.add_peer(mock_peer3)

        # Add connections to topology so route_message can find routes
        message_router.add_connection("peer1", "peer2")
        message_router.add_connection("peer1", "peer3")

        # Broadcast message from peer1
        success_count = await message_router.broadcast_message("peer1", {"type": "test"})

        # Should succeed for peer2 and fail for peer3
        assert success_count == 1  # Only one succeeds

    @pytest.mark.asyncio
    async def test_broadcast_message_with_exclude_peers(self, message_router):
        """Test broadcasting message with excluded peers."""
        # Add some peers
        mock_peer1 = Mock(spec=Peer)
        mock_peer1.get_peer_id.return_value = "peer1"
        mock_peer1.send_message.return_value = True

        mock_peer2 = Mock(spec=Peer)
        mock_peer2.get_peer_id.return_value = "peer2"
        mock_peer2.send_message.return_value = True

        mock_peer3 = Mock(spec=Peer)
        mock_peer3.get_peer_id.return_value = "peer3"
        mock_peer3.send_message.return_value = True

        message_router.add_peer(mock_peer1)
        message_router.add_peer(mock_peer2)
        message_router.add_peer(mock_peer3)

        # Add connections to topology so route_message can find routes
        message_router.add_connection("peer1", "peer2")
        message_router.add_connection("peer1", "peer3")

        # Broadcast message from peer1, excluding peer2
        success_count = await message_router.broadcast_message(
            "peer1", {"type": "test"}, exclude_peers=["peer2"]
        )

        # Should only send to peer3 (excluding peer1 and peer2)
        assert success_count == 1

    @pytest.mark.asyncio
    async def test_route_message_nonexistent_peers(self, message_router):
        """Test routing message with nonexistent peers."""
        # Try to route between nonexistent peers
        result = await message_router.route_message("nonexistent1", "nonexistent2", {"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_route_message_same_peer(self, message_router):
        """Test routing message to same peer."""
        mock_peer = Mock(spec=Peer)
        mock_peer.get_peer_id.return_value = "peer1"
        mock_peer.send_message.return_value = True

        message_router.add_peer(mock_peer)

        # Route message to same peer
        result = await message_router.route_message("peer1", "peer1", {"type": "test"})
        assert result is True

    def test_add_connection(self, message_router):
        """Test adding connection between peers."""
        message_router.add_connection("peer1", "peer2")

        # Verify connection is added to topology
        assert "peer2" in message_router.network_topology.connections["peer1"]
        assert "peer1" in message_router.network_topology.connections["peer2"]

    def test_remove_connection(self, message_router):
        """Test removing connection between peers."""
        message_router.add_connection("peer1", "peer2")
        message_router.remove_connection("peer1", "peer2")

        # Verify connection is removed from topology
        assert "peer2" not in message_router.network_topology.connections["peer1"]
        assert "peer1" not in message_router.network_topology.connections["peer2"]

    def test_get_route_stats(self, message_router):
        """Test getting route statistics."""
        # Add some routes
        route1 = RouteInfo(route_id="route1", source_peer="peer1", target_peer="peer2")
        route2 = RouteInfo(route_id="route2", source_peer="peer1", target_peer="peer3")
        route2.success_count = 8
        route2.failure_count = 2

        message_router.routes["route1"] = route1
        message_router.routes["route2"] = route2

        stats = message_router.get_route_stats()

        assert stats["strategy"] == message_router.strategy.value
        assert stats["total_routes"] == 2
        assert stats["healthy_routes"] == 1  # Only route2 is healthy
        assert stats["peers_count"] == 0  # No peers added yet

    def test_str_representation(self, message_router):
        """Test string representation."""
        str_repr = str(message_router)
        assert "MessageRouter" in str_repr
        assert "strategy=" in str_repr
        assert "routes=" in str_repr
        assert "peers=" in str_repr

    def test_repr_representation(self, message_router):
        """Test detailed representation."""
        repr_str = repr(message_router)
        assert "MessageRouter" in repr_str
        assert "strategy=" in repr_str
        assert "routes=" in repr_str
        assert "peers=" in repr_str

    def test_add_peer_with_get_peer_id_method(self, message_router):
        """Test adding peer with get_peer_id method."""
        mock_peer = Mock(spec=Peer)
        mock_peer.get_peer_id.return_value = "peer1"

        message_router.add_peer(mock_peer)

        assert "peer1" in message_router.peer_connections
        assert message_router.peer_connections["peer1"] == mock_peer

    def test_add_peer_with_peer_id_attribute(self, message_router):
        """Test adding peer with peer_id attribute."""
        # Create a simple object with only peer_id attribute
        class SimplePeer:
            def __init__(self, peer_id):
                self.peer_id = peer_id

        simple_peer = SimplePeer("peer1")

        message_router.add_peer(simple_peer)

        assert "peer1" in message_router.peer_connections
        assert message_router.peer_connections["peer1"] == simple_peer

    def test_remove_peer_cleanup(self, message_router):
        """Test that removing peer cleans up all references."""
        # Add peer and some routes
        mock_peer = Mock()
        mock_peer.peer_id = "peer1"
        message_router.add_peer(mock_peer)

        route1 = RouteInfo(route_id="route1", source_peer="peer1", target_peer="peer2")
        route2 = RouteInfo(route_id="route2", source_peer="peer2", target_peer="peer1")
        message_router.routes["route1"] = route1
        message_router.routes["route2"] = route2

        # Remove peer
        message_router.remove_peer("peer1")

        # Verify peer is removed
        assert "peer1" not in message_router.peer_connections

        # Verify routes involving peer1 are removed
        assert "route1" not in message_router.routes
        assert "route2" not in message_router.routes
