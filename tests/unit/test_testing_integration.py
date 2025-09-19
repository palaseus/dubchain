"""Unit tests for dubchain.testing.integration module."""

import os
import tempfile
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import pytest

from dubchain.testing.integration import (
    DatabaseManager,
    NetworkManager,
    NodeManager,
    ClusterManager,
    IntegrationTestCase,
    IntegrationTestSuite,
    IntegrationTestRunner,
)
from dubchain.testing.base import ExecutionConfig, ExecutionType, ExecutionStatus


class TestDatabaseManager:
    """Test DatabaseManager functionality."""

    def test_database_manager_creation_sqlite(self):
        """Test creating SQLite database manager."""
        db_manager = DatabaseManager("sqlite")
        
        assert db_manager.db_type == "sqlite"
        assert db_manager.db_path.endswith(".db")
        assert db_manager.connection is not None
        assert db_manager.cursor is not None

    def test_database_manager_creation_with_path(self):
        """Test creating database manager with custom path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_manager = DatabaseManager("sqlite", db_path)
            
            assert db_manager.db_path == db_path
            assert os.path.exists(db_path)
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_database_manager_unsupported_type(self):
        """Test creating database manager with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported database type: mysql"):
            DatabaseManager("mysql")

    def test_execute_sql_with_params(self):
        """Test executing SQL with parameters."""
        db_manager = DatabaseManager("sqlite")
        
        # Create a test table
        db_manager.execute_sql("CREATE TABLE test (id INTEGER, name TEXT)")
        
        # Insert data with parameters
        db_manager.execute_sql("INSERT INTO test (id, name) VALUES (?, ?)", (1, "test"))
        db_manager.commit()
        
        # Verify data
        result = db_manager.fetch_one("SELECT * FROM test WHERE id = ?", (1,))
        assert result == (1, "test")

    def test_execute_sql_without_params(self):
        """Test executing SQL without parameters."""
        db_manager = DatabaseManager("sqlite")
        
        # Create a test table
        db_manager.execute_sql("CREATE TABLE test (id INTEGER)")
        
        # Insert data without parameters
        db_manager.execute_sql("INSERT INTO test (id) VALUES (1)")
        db_manager.commit()
        
        # Verify data
        result = db_manager.fetch_one("SELECT * FROM test")
        assert result == (1,)

    def test_fetch_all(self):
        """Test fetching all results."""
        db_manager = DatabaseManager("sqlite")
        
        # Create and populate test table
        db_manager.execute_sql("CREATE TABLE test (id INTEGER, name TEXT)")
        db_manager.execute_sql("INSERT INTO test (id, name) VALUES (?, ?)", (1, "test1"))
        db_manager.execute_sql("INSERT INTO test (id, name) VALUES (?, ?)", (2, "test2"))
        db_manager.commit()
        
        # Fetch all results
        results = db_manager.fetch_all("SELECT * FROM test ORDER BY id")
        assert len(results) == 2
        assert results[0] == (1, "test1")
        assert results[1] == (2, "test2")

    def test_fetch_one(self):
        """Test fetching one result."""
        db_manager = DatabaseManager("sqlite")
        
        # Create and populate test table
        db_manager.execute_sql("CREATE TABLE test (id INTEGER, name TEXT)")
        db_manager.execute_sql("INSERT INTO test (id, name) VALUES (?, ?)", (1, "test1"))
        db_manager.execute_sql("INSERT INTO test (id, name) VALUES (?, ?)", (2, "test2"))
        db_manager.commit()
        
        # Fetch one result
        result = db_manager.fetch_one("SELECT * FROM test WHERE id = ?", (1,))
        assert result == (1, "test1")

    def test_commit_rollback(self):
        """Test commit and rollback functionality."""
        db_manager = DatabaseManager("sqlite")
        
        # Create test table
        db_manager.execute_sql("CREATE TABLE test (id INTEGER)")
        
        # Insert and commit
        db_manager.execute_sql("INSERT INTO test (id) VALUES (1)")
        db_manager.commit()
        
        # Verify data is committed
        result = db_manager.fetch_one("SELECT * FROM test")
        assert result == (1,)
        
        # Insert and rollback
        db_manager.execute_sql("INSERT INTO test (id) VALUES (2)")
        db_manager.rollback()
        
        # Verify rollback worked
        results = db_manager.fetch_all("SELECT * FROM test")
        assert len(results) == 1
        assert results[0] == (1,)

    def test_create_table(self):
        """Test creating table with column definitions."""
        db_manager = DatabaseManager("sqlite")
        
        columns = {"id": "INTEGER", "name": "TEXT", "value": "REAL"}
        db_manager.create_table("test_table", columns)
        
        # Verify table was created
        result = db_manager.fetch_one("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
        assert result is not None

    def test_insert_data(self):
        """Test inserting data using insert_data method."""
        db_manager = DatabaseManager("sqlite")
        
        # Create table
        db_manager.create_table("test_table", {"id": "INTEGER", "name": "TEXT"})
        
        # Insert data
        data = {"id": 1, "name": "test"}
        db_manager.insert_data("test_table", data)
        
        # Verify data
        result = db_manager.fetch_one("SELECT * FROM test_table")
        assert result == (1, "test")

    def test_cleanup(self):
        """Test database cleanup."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_manager = DatabaseManager("sqlite", db_path)
            assert os.path.exists(db_path)
            
            db_manager.cleanup()
            
            # Verify cleanup
            assert db_manager.connection is None
            assert db_manager.cursor is None
            assert not os.path.exists(db_path)
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestNetworkManager:
    """Test NetworkManager functionality."""

    @pytest.fixture
    def network_manager(self):
        """Fixture for network manager."""
        return NetworkManager("test_network")

    @pytest.fixture
    def mock_node1(self):
        """Fixture for mock node 1."""
        node = Mock(spec=NodeManager)
        node.node_id = "node1"
        node.network = None
        node.add_peer = Mock()
        node.remove_peer = Mock()
        return node

    @pytest.fixture
    def mock_node2(self):
        """Fixture for mock node 2."""
        node = Mock(spec=NodeManager)
        node.node_id = "node2"
        node.network = None
        node.add_peer = Mock()
        node.remove_peer = Mock()
        return node

    def test_network_manager_creation(self, network_manager):
        """Test creating network manager."""
        assert network_manager.network_name == "test_network"
        assert network_manager.nodes == []
        assert network_manager.connections == []
        assert network_manager.ports == []

    def test_add_node(self, network_manager, mock_node1):
        """Test adding node to network."""
        network_manager.add_node(mock_node1)
        
        assert mock_node1 in network_manager.nodes
        assert mock_node1.network == network_manager

    def test_remove_node(self, network_manager, mock_node1):
        """Test removing node from network."""
        network_manager.add_node(mock_node1)
        network_manager.remove_node(mock_node1)
        
        assert mock_node1 not in network_manager.nodes
        assert mock_node1.network is None

    def test_remove_nonexistent_node(self, network_manager, mock_node1):
        """Test removing nonexistent node."""
        # Should not raise exception
        network_manager.remove_node(mock_node1)
        assert mock_node1 not in network_manager.nodes

    def test_connect_nodes(self, network_manager, mock_node1, mock_node2):
        """Test connecting two nodes."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        
        network_manager.connect_nodes(mock_node1, mock_node2)
        
        assert (mock_node1, mock_node2) in network_manager.connections
        mock_node1.add_peer.assert_called_once_with(mock_node2)
        mock_node2.add_peer.assert_called_once_with(mock_node1)

    def test_connect_nodes_duplicate(self, network_manager, mock_node1, mock_node2):
        """Test connecting already connected nodes."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        
        # Connect first time
        network_manager.connect_nodes(mock_node1, mock_node2)
        
        # Connect second time (should not duplicate)
        network_manager.connect_nodes(mock_node1, mock_node2)
        
        # Should only have one connection
        assert network_manager.connections.count((mock_node1, mock_node2)) == 1

    def test_disconnect_nodes(self, network_manager, mock_node1, mock_node2):
        """Test disconnecting two nodes."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        network_manager.connect_nodes(mock_node1, mock_node2)
        
        network_manager.disconnect_nodes(mock_node1, mock_node2)
        
        assert (mock_node1, mock_node2) not in network_manager.connections
        mock_node1.remove_peer.assert_called_once_with(mock_node2)
        mock_node2.remove_peer.assert_called_once_with(mock_node1)

    def test_disconnect_nonexistent_connection(self, network_manager, mock_node1, mock_node2):
        """Test disconnecting nonexistent connection."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        
        # Should not raise exception
        network_manager.disconnect_nodes(mock_node1, mock_node2)
        assert (mock_node1, mock_node2) not in network_manager.connections

    def test_get_available_port(self, network_manager):
        """Test getting available port."""
        port = network_manager.get_available_port()
        
        assert isinstance(port, int)
        assert port > 0
        assert port in network_manager.ports

    def test_get_multiple_available_ports(self, network_manager):
        """Test getting multiple available ports."""
        ports = []
        for _ in range(3):
            port = network_manager.get_available_port()
            ports.append(port)
        
        # All ports should be different
        assert len(set(ports)) == 3
        assert all(port in network_manager.ports for port in ports)

    def test_broadcast_message(self, network_manager, mock_node1, mock_node2):
        """Test broadcasting message to all nodes."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        
        message = {"type": "test", "data": "hello"}
        network_manager.broadcast_message(message, mock_node1)
        
        # Only node2 should receive the message (sender is excluded)
        mock_node2.receive_message.assert_called_once_with(message)
        mock_node1.receive_message.assert_not_called()

    def test_broadcast_message_no_sender(self, network_manager, mock_node1, mock_node2):
        """Test broadcasting message with no sender."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        
        message = {"type": "test", "data": "hello"}
        network_manager.broadcast_message(message)
        
        # Both nodes should receive the message
        mock_node1.receive_message.assert_called_once_with(message)
        mock_node2.receive_message.assert_called_once_with(message)

    def test_cleanup(self, network_manager, mock_node1, mock_node2):
        """Test network cleanup."""
        network_manager.add_node(mock_node1)
        network_manager.add_node(mock_node2)
        network_manager.connect_nodes(mock_node1, mock_node2)
        network_manager.get_available_port()
        
        network_manager.cleanup()
        
        # Verify cleanup
        assert network_manager.nodes == []
        assert network_manager.connections == []
        assert network_manager.ports == []
        mock_node1.cleanup.assert_called_once()
        mock_node2.cleanup.assert_called_once()


class TestNodeManager:
    """Test NodeManager functionality."""

    @pytest.fixture
    def node_manager(self):
        """Fixture for node manager."""
        return NodeManager("test_node", 8080)

    def test_node_manager_creation(self, node_manager):
        """Test creating node manager."""
        assert node_manager.node_id == "test_node"
        assert node_manager.port == 8080
        assert node_manager.network is None
        assert node_manager.peers == []
        assert node_manager.messages == []
        assert node_manager.state == {}
        assert node_manager.running is False
        assert node_manager._message_handlers == {}

    def test_start_stop(self, node_manager):
        """Test starting and stopping node."""
        # Initially stopped
        assert not node_manager.running
        assert "status" not in node_manager.state
        
        # Start node
        node_manager.start()
        assert node_manager.running
        assert node_manager.state["status"] == "running"
        assert "start_time" in node_manager.state
        
        # Stop node
        node_manager.stop()
        assert not node_manager.running
        assert node_manager.state["status"] == "stopped"
        assert "stop_time" in node_manager.state

    def test_add_peer(self, node_manager):
        """Test adding peer node."""
        peer = Mock(spec=NodeManager)
        peer.node_id = "peer1"
        
        node_manager.add_peer(peer)
        assert peer in node_manager.peers

    def test_add_peer_duplicate(self, node_manager):
        """Test adding duplicate peer."""
        peer = Mock(spec=NodeManager)
        peer.node_id = "peer1"
        
        node_manager.add_peer(peer)
        node_manager.add_peer(peer)  # Add again
        
        # Should only have one instance
        assert node_manager.peers.count(peer) == 1

    def test_remove_peer(self, node_manager):
        """Test removing peer node."""
        peer = Mock(spec=NodeManager)
        peer.node_id = "peer1"
        
        node_manager.add_peer(peer)
        node_manager.remove_peer(peer)
        
        assert peer not in node_manager.peers

    def test_remove_nonexistent_peer(self, node_manager):
        """Test removing nonexistent peer."""
        peer = Mock(spec=NodeManager)
        peer.node_id = "peer1"
        
        # Should not raise exception
        node_manager.remove_peer(peer)
        assert peer not in node_manager.peers

    def test_send_message_to_target(self, node_manager):
        """Test sending message to specific target."""
        target = Mock(spec=NodeManager)
        message = {"type": "test", "data": "hello"}
        
        node_manager.send_message(message, target)
        target.receive_message.assert_called_once_with(message)

    def test_send_message_broadcast(self, node_manager):
        """Test broadcasting message to all peers."""
        peer1 = Mock(spec=NodeManager)
        peer2 = Mock(spec=NodeManager)
        node_manager.add_peer(peer1)
        node_manager.add_peer(peer2)
        
        message = {"type": "test", "data": "hello"}
        node_manager.send_message(message)
        
        peer1.receive_message.assert_called_once_with(message)
        peer2.receive_message.assert_called_once_with(message)

    def test_receive_message(self, node_manager):
        """Test receiving message."""
        message = Mock()
        message.type = "test"
        message.sender = "sender1"
        
        node_manager.receive_message(message)
        
        assert len(node_manager.messages) == 1
        msg_data = node_manager.messages[0]
        assert msg_data["message"] == message
        assert msg_data["sender"] == "sender1"
        assert "timestamp" in msg_data

    def test_receive_message_with_handler(self, node_manager):
        """Test receiving message with registered handler."""
        message = Mock()
        message.type = "test"
        message.sender = "sender1"
        
        handler = Mock()
        node_manager.register_message_handler("test", handler)
        
        node_manager.receive_message(message)
        
        handler.assert_called_once_with(message)

    def test_receive_message_handler_exception(self, node_manager):
        """Test receiving message with handler that raises exception."""
        message = Mock()
        message.type = "test"
        message.sender = "sender1"
        
        handler = Mock(side_effect=Exception("Handler error"))
        node_manager.register_message_handler("test", handler)
        
        # Should not raise exception, just log error
        node_manager.receive_message(message)
        
        # Message should still be stored
        assert len(node_manager.messages) == 1

    def test_register_message_handler(self, node_manager):
        """Test registering message handler."""
        handler = Mock()
        node_manager.register_message_handler("test_type", handler)
        
        assert node_manager._message_handlers["test_type"] == handler

    def test_get_messages_all(self, node_manager):
        """Test getting all messages."""
        message1 = Mock()
        message1.type = "type1"
        message2 = Mock()
        message2.type = "type2"
        
        node_manager.receive_message(message1)
        node_manager.receive_message(message2)
        
        messages = node_manager.get_messages()
        assert len(messages) == 2

    def test_get_messages_by_type(self, node_manager):
        """Test getting messages by type."""
        message1 = Mock()
        message1.type = "type1"
        message2 = Mock()
        message2.type = "type2"
        
        node_manager.receive_message(message1)
        node_manager.receive_message(message2)
        
        messages = node_manager.get_messages("type1")
        assert len(messages) == 1
        assert messages[0]["message"] == message1

    def test_clear_messages(self, node_manager):
        """Test clearing messages."""
        message = Mock()
        message.type = "test"
        node_manager.receive_message(message)
        
        assert len(node_manager.messages) == 1
        node_manager.clear_messages()
        assert len(node_manager.messages) == 0

    def test_cleanup(self, node_manager):
        """Test node cleanup."""
        # Setup node state
        peer = Mock(spec=NodeManager)
        node_manager.add_peer(peer)
        node_manager.start()
        message = Mock()
        message.type = "test"
        node_manager.receive_message(message)
        node_manager.register_message_handler("test", Mock())
        
        node_manager.cleanup()
        
        # Verify cleanup
        assert not node_manager.running
        assert node_manager.peers == []
        assert node_manager.messages == []
        assert node_manager.state == {}
        assert node_manager._message_handlers == {}


class TestClusterManager:
    """Test ClusterManager functionality."""

    @pytest.fixture
    def cluster_manager(self):
        """Fixture for cluster manager."""
        return ClusterManager("test_cluster")

    def test_cluster_manager_creation(self, cluster_manager):
        """Test creating cluster manager."""
        assert cluster_manager.cluster_name == "test_cluster"
        assert cluster_manager.nodes == []
        assert cluster_manager.network is not None
        assert cluster_manager.network.network_name == "test_cluster_network"
        assert cluster_manager.coordinator is None

    def test_add_node(self, cluster_manager):
        """Test adding node to cluster."""
        node = cluster_manager.add_node("node1")
        
        assert node.node_id == "node1"
        assert node in cluster_manager.nodes
        assert node in cluster_manager.network.nodes
        assert node.network == cluster_manager.network

    def test_add_node_with_port(self, cluster_manager):
        """Test adding node with specific port."""
        node = cluster_manager.add_node("node1", 8080)
        
        assert node.node_id == "node1"
        assert node.port == 8080

    def test_remove_node(self, cluster_manager):
        """Test removing node from cluster."""
        node = cluster_manager.add_node("node1")
        cluster_manager.remove_node(node)
        
        assert node not in cluster_manager.nodes
        assert node not in cluster_manager.network.nodes

    def test_remove_nonexistent_node(self, cluster_manager):
        """Test removing nonexistent node."""
        node = Mock(spec=NodeManager)
        node.node_id = "nonexistent"
        
        # Should not raise exception
        cluster_manager.remove_node(node)
        assert node not in cluster_manager.nodes

    def test_set_coordinator(self, cluster_manager):
        """Test setting cluster coordinator."""
        node = cluster_manager.add_node("node1")
        cluster_manager.set_coordinator(node)
        
        assert cluster_manager.coordinator == node

    def test_start_all_nodes(self, cluster_manager):
        """Test starting all nodes in cluster."""
        node1 = cluster_manager.add_node("node1")
        node2 = cluster_manager.add_node("node2")
        
        cluster_manager.start_all_nodes()
        
        assert node1.running
        assert node2.running

    def test_stop_all_nodes(self, cluster_manager):
        """Test stopping all nodes in cluster."""
        node1 = cluster_manager.add_node("node1")
        node2 = cluster_manager.add_node("node2")
        cluster_manager.start_all_nodes()
        
        cluster_manager.stop_all_nodes()
        
        assert not node1.running
        assert not node2.running

    def test_connect_all_nodes(self, cluster_manager):
        """Test connecting all nodes in cluster."""
        node1 = cluster_manager.add_node("node1")
        node2 = cluster_manager.add_node("node2")
        node3 = cluster_manager.add_node("node3")
        
        cluster_manager.connect_all_nodes()
        
        # Each node should be connected to all others
        assert len(cluster_manager.network.connections) == 3  # 3 choose 2 = 3 connections

    def test_broadcast_to_cluster(self, cluster_manager):
        """Test broadcasting message to cluster."""
        node1 = cluster_manager.add_node("node1")
        node2 = cluster_manager.add_node("node2")
        
        message = {"type": "test", "data": "hello"}
        cluster_manager.broadcast_to_cluster(message, node1)
        
        # Only node2 should receive the message
        assert len(node2.messages) == 1
        assert node2.messages[0]["message"] == message

    def test_get_cluster_state(self, cluster_manager):
        """Test getting cluster state."""
        node1 = cluster_manager.add_node("node1", 8080)
        node2 = cluster_manager.add_node("node2", 8081)
        cluster_manager.set_coordinator(node1)
        cluster_manager.start_all_nodes()
        
        state = cluster_manager.get_cluster_state()
        
        assert state["cluster_name"] == "test_cluster"
        assert state["nodes_count"] == 2
        assert state["connections_count"] == 0  # No connections yet
        assert state["coordinator"] == "node1"
        assert len(state["nodes"]) == 2
        
        # Check node details
        node_states = {node["node_id"]: node for node in state["nodes"]}
        assert node_states["node1"]["port"] == 8080
        assert node_states["node1"]["running"] is True
        assert node_states["node2"]["port"] == 8081
        assert node_states["node2"]["running"] is True

    def test_cleanup(self, cluster_manager):
        """Test cluster cleanup."""
        node1 = cluster_manager.add_node("node1")
        node2 = cluster_manager.add_node("node2")
        cluster_manager.set_coordinator(node1)
        cluster_manager.start_all_nodes()
        
        cluster_manager.cleanup()
        
        # Verify cleanup
        assert cluster_manager.nodes == []
        assert cluster_manager.coordinator is None
        assert not node1.running
        assert not node2.running


class ConcreteIntegrationTestCase(IntegrationTestCase):
    """Concrete implementation of IntegrationTestCase for testing."""
    
    def run_test(self) -> None:
        """Run the actual test."""
        pass


class TestIntegrationTestCase:
    """Test IntegrationTestCase functionality."""

    @pytest.fixture
    def integration_test_case(self):
        """Fixture for integration test case."""
        return ConcreteIntegrationTestCase("test_integration")

    def test_integration_test_case_creation(self, integration_test_case):
        """Test creating integration test case."""
        assert integration_test_case.name == "test_integration"
        assert integration_test_case.database is None
        assert integration_test_case.network is None
        assert integration_test_case.cluster is None
        assert integration_test_case.external_services == {}
        assert integration_test_case.test_data == {}

    def test_setup_teardown(self, integration_test_case):
        """Test setup and teardown."""
        integration_test_case.setup()
        
        assert integration_test_case.database is not None
        assert integration_test_case.network is not None
        
        integration_test_case.teardown()
        
        assert integration_test_case.database is None
        assert integration_test_case.network is None

    def test_create_test_cluster(self, integration_test_case):
        """Test creating test cluster."""
        cluster = integration_test_case.create_test_cluster(3)
        
        assert cluster is not None
        assert len(cluster.nodes) == 3
        assert all(node.running for node in cluster.nodes)
        assert integration_test_case.cluster == cluster

    def test_add_get_external_service(self, integration_test_case):
        """Test adding and getting external service."""
        service = Mock()
        integration_test_case.add_external_service("test_service", service)
        
        retrieved_service = integration_test_case.get_external_service("test_service")
        assert retrieved_service == service

    def test_get_nonexistent_external_service(self, integration_test_case):
        """Test getting nonexistent external service."""
        with pytest.raises(KeyError, match="External service 'nonexistent' not found"):
            integration_test_case.get_external_service("nonexistent")

    def test_set_get_test_data(self, integration_test_case):
        """Test setting and getting test data."""
        integration_test_case.set_test_data("key1", "value1")
        integration_test_case.set_test_data("key2", 42)
        
        assert integration_test_case.get_test_data("key1") == "value1"
        assert integration_test_case.get_test_data("key2") == 42
        assert integration_test_case.get_test_data("nonexistent", "default") == "default"

    def test_wait_for_condition_success(self, integration_test_case):
        """Test waiting for condition that succeeds."""
        condition_met = False
        
        def condition():
            nonlocal condition_met
            return condition_met
        
        # Start a thread to set condition after a short delay
        def set_condition():
            time.sleep(0.1)
            nonlocal condition_met
            condition_met = True
        
        thread = threading.Thread(target=set_condition)
        thread.start()
        
        result = integration_test_case.wait_for_condition(condition, timeout=1.0)
        
        assert result is True
        thread.join()

    def test_wait_for_condition_timeout(self, integration_test_case):
        """Test waiting for condition that times out."""
        def condition():
            return False
        
        result = integration_test_case.wait_for_condition(condition, timeout=0.1)
        assert result is False

    def test_assert_cluster_consensus_success(self, integration_test_case):
        """Test asserting cluster consensus that succeeds."""
        cluster = integration_test_case.create_test_cluster(2)
        
        # Set consensus value on all nodes
        for node in cluster.nodes:
            node.state["consensus_value"] = "agreed_value"
        
        # Should not raise exception
        integration_test_case.assert_cluster_consensus("agreed_value", timeout=1.0)

    def test_assert_cluster_consensus_failure(self, integration_test_case):
        """Test asserting cluster consensus that fails."""
        cluster = integration_test_case.create_test_cluster(2)
        
        # Set different consensus values
        cluster.nodes[0].state["consensus_value"] = "value1"
        cluster.nodes[1].state["consensus_value"] = "value2"
        
        with pytest.raises(AssertionError, match="Cluster did not reach consensus on agreed_value"):
            integration_test_case.assert_cluster_consensus("agreed_value", timeout=0.1)

    def test_assert_cluster_consensus_no_cluster(self, integration_test_case):
        """Test asserting cluster consensus with no cluster."""
        with pytest.raises(AssertionError, match="No test cluster available"):
            integration_test_case.assert_cluster_consensus("value")

    def test_assert_message_delivery_success(self, integration_test_case):
        """Test asserting message delivery that succeeds."""
        cluster = integration_test_case.create_test_cluster(2)
        sender = cluster.nodes[0]
        receiver = cluster.nodes[1]
        message = {"type": "test", "data": "hello"}
        
        # Send message
        sender.send_message(message, receiver)
        
        # Should not raise exception
        integration_test_case.assert_message_delivery(sender, receiver, message, timeout=1.0)

    def test_assert_message_delivery_failure(self, integration_test_case):
        """Test asserting message delivery that fails."""
        cluster = integration_test_case.create_test_cluster(2)
        sender = cluster.nodes[0]
        receiver = cluster.nodes[1]
        message = {"type": "test", "data": "hello"}
        
        # Don't send message
        
        with pytest.raises(AssertionError, match="Message .* was not delivered"):
            integration_test_case.assert_message_delivery(sender, receiver, message, timeout=0.1)


class TestIntegrationTestSuite:
    """Test IntegrationTestSuite functionality."""

    @pytest.fixture
    def integration_test_suite(self):
        """Fixture for integration test suite."""
        return IntegrationTestSuite("test_suite")

    @pytest.fixture
    def mock_test_case(self):
        """Fixture for mock test case."""
        test_case = Mock(spec=IntegrationTestCase)
        test_case.name = "test_case"
        test_case.environment = None
        test_case.run = Mock()
        return test_case

    def test_integration_test_suite_creation(self, integration_test_suite):
        """Test creating integration test suite."""
        assert integration_test_suite.name == "test_suite"
        assert integration_test_suite.config.test_type == ExecutionType.INTEGRATION
        assert integration_test_suite.test_environments == {}

    def test_add_test_default_environment(self, integration_test_suite, mock_test_case):
        """Test adding test with default environment."""
        integration_test_suite.add_test(mock_test_case)
        
        assert mock_test_case in integration_test_suite.tests
        assert "default" in integration_test_suite.test_environments
        assert mock_test_case.environment is not None

    def test_add_test_custom_environment(self, integration_test_suite, mock_test_case):
        """Test adding test with custom environment."""
        integration_test_suite.add_test(mock_test_case, "custom_env")
        
        assert mock_test_case in integration_test_suite.tests
        assert "custom_env" in integration_test_suite.test_environments
        assert mock_test_case.environment is not None

    def test_run_by_environment_nonexistent(self, integration_test_suite):
        """Test running tests by nonexistent environment."""
        results = integration_test_suite.run_by_environment("nonexistent")
        assert results == []

    def test_run_by_environment_success(self, integration_test_suite, mock_test_case):
        """Test running tests by environment successfully."""
        integration_test_suite.add_test(mock_test_case, "test_env")
        
        mock_result = Mock()
        mock_test_case.run.return_value = mock_result
        
        results = integration_test_suite.run_by_environment("test_env")
        
        assert len(results) == 1
        assert results[0] == mock_result
        mock_test_case.run.assert_called_once()

    def test_run_by_environment_with_exception(self, integration_test_suite, mock_test_case):
        """Test running tests by environment with exception."""
        integration_test_suite.add_test(mock_test_case, "test_env")
        
        mock_test_case.run.side_effect = Exception("Test error")
        
        results = integration_test_suite.run_by_environment("test_env")
        
        assert len(results) == 1
        result = results[0]
        assert result.test_name == "test_case"
        assert result.test_type == ExecutionType.INTEGRATION
        assert result.status == ExecutionStatus.ERROR
        assert result.error_message == "Test error"


class TestIntegrationTestRunner:
    """Test IntegrationTestRunner functionality."""

    @pytest.fixture
    def integration_test_runner(self):
        """Fixture for integration test runner."""
        return IntegrationTestRunner()

    def test_integration_test_runner_creation(self, integration_test_runner):
        """Test creating integration test runner."""
        assert integration_test_runner.config.test_type == ExecutionType.INTEGRATION
        assert integration_test_runner.parallel_execution is False
        assert integration_test_runner.environment_isolation is True

    def test_run_with_environment_isolation(self, integration_test_runner):
        """Test running with environment isolation."""
        # Mock the run_all method
        mock_results = [Mock(), Mock()]
        integration_test_runner.run_all = Mock(return_value=mock_results)
        
        results = integration_test_runner.run_with_environment_isolation()
        
        assert results == mock_results
        integration_test_runner.run_all.assert_called_once()

    def test_run_with_parallel_execution(self, integration_test_runner):
        """Test running with parallel execution."""
        # Mock the run_all method
        mock_results = [Mock(), Mock()]
        integration_test_runner.run_all = Mock(return_value=mock_results)
        
        results = integration_test_runner.run_with_parallel_execution(max_workers=2)
        
        assert results == mock_results
        integration_test_runner.run_all.assert_called_once()
