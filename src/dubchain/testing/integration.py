"""Integration testing infrastructure for DubChain.

This module provides integration testing capabilities for testing
components working together in realistic environments.
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..logging import get_logger
from .base import (
    AsyncTestCase,
    BaseTestCase,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionResult,
    ExecutionStatus,
    ExecutionType,
    FixtureManager,
    RunnerManager,
    SuiteManager,
)


class DatabaseManager:
    """Test database for integration testing."""

    def __init__(self, db_type: str = "sqlite", db_path: str = None):
        self.db_type = db_type
        self.db_path = db_path or tempfile.mktemp(suffix=".db")
        self.connection = None
        self.cursor = None
        self._setup_database()

    def _setup_database(self) -> None:
        """Setup test database."""
        if self.db_type == "sqlite":
            import sqlite3

            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def execute_sql(self, sql: str, params: tuple = None) -> Any:
        """Execute SQL statement."""
        if params:
            return self.cursor.execute(sql, params)
        else:
            return self.cursor.execute(sql)

    def fetch_all(self, sql: str, params: tuple = None) -> List[tuple]:
        """Fetch all results."""
        self.execute_sql(sql, params)
        return self.cursor.fetchall()

    def fetch_one(self, sql: str, params: tuple = None) -> tuple:
        """Fetch one result."""
        self.execute_sql(sql, params)
        return self.cursor.fetchone()

    def commit(self) -> None:
        """Commit transaction."""
        self.connection.commit()

    def rollback(self) -> None:
        """Rollback transaction."""
        self.connection.rollback()

    def create_table(self, table_name: str, columns: Dict[str, str]) -> None:
        """Create test table."""
        column_defs = [f"{name} {type_def}" for name, type_def in columns.items()]
        sql = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
        self.execute_sql(sql)
        self.commit()

    def insert_data(self, table_name: str, data: Dict[str, Any]) -> None:
        """Insert test data."""
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(["?" for _ in columns])
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        self.execute_sql(sql, tuple(values))
        self.commit()

    def cleanup(self) -> None:
        """Cleanup test database."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)


class NetworkManager:
    """Test network for integration testing."""

    def __init__(self, network_name: str = "test_network"):
        self.network_name = network_name
        self.nodes: List[NodeManager] = []
        self.connections: List[tuple] = []
        self.ports: List[int] = []
        self._lock = threading.RLock()

    def add_node(self, node: "NodeManager") -> None:
        """Add node to network."""
        with self._lock:
            self.nodes.append(node)
            node.network = self

    def remove_node(self, node: "NodeManager") -> None:
        """Remove node from network."""
        with self._lock:
            if node in self.nodes:
                self.nodes.remove(node)
                node.network = None

    def connect_nodes(self, node1: "NodeManager", node2: "NodeManager") -> None:
        """Connect two nodes."""
        with self._lock:
            connection = (node1, node2)
            if connection not in self.connections:
                self.connections.append(connection)
                node1.add_peer(node2)
                node2.add_peer(node1)

    def disconnect_nodes(self, node1: "NodeManager", node2: "NodeManager") -> None:
        """Disconnect two nodes."""
        with self._lock:
            connection = (node1, node2)
            if connection in self.connections:
                self.connections.remove(connection)
                node1.remove_peer(node2)
                node2.remove_peer(node1)

    def get_available_port(self) -> int:
        """Get available port for testing."""
        with self._lock:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            port = sock.getsockname()[1]
            sock.close()

            if port not in self.ports:
                self.ports.append(port)

            return port

    def broadcast_message(self, message: Any, sender: "NodeManager" = None) -> None:
        """Broadcast message to all nodes."""
        with self._lock:
            for node in self.nodes:
                if node != sender:
                    node.receive_message(message)

    def cleanup(self) -> None:
        """Cleanup test network."""
        with self._lock:
            for node in self.nodes:
                node.cleanup()
            self.nodes.clear()
            self.connections.clear()
            self.ports.clear()


class NodeManager:
    """Test node for integration testing."""

    def __init__(self, node_id: str, port: int = None):
        self.node_id = node_id
        self.port = port
        self.network: Optional[NetworkManager] = None
        self.peers: List["NodeManager"] = []
        self.messages: List[Any] = []
        self.state: Dict[str, Any] = {}
        self.running = False
        self._lock = threading.RLock()
        self._message_handlers: Dict[str, Callable] = {}

    def start(self) -> None:
        """Start test node."""
        with self._lock:
            self.running = True
            self.state["status"] = "running"
            self.state["start_time"] = time.time()

    def stop(self) -> None:
        """Stop test node."""
        with self._lock:
            self.running = False
            self.state["status"] = "stopped"
            self.state["stop_time"] = time.time()

    def add_peer(self, peer: "NodeManager") -> None:
        """Add peer node."""
        with self._lock:
            if peer not in self.peers:
                self.peers.append(peer)

    def remove_peer(self, peer: "NodeManager") -> None:
        """Remove peer node."""
        with self._lock:
            if peer in self.peers:
                self.peers.remove(peer)

    def send_message(self, message: Any, target: "NodeManager" = None) -> None:
        """Send message to target node or broadcast."""
        with self._lock:
            if target:
                target.receive_message(message)
            else:
                # Broadcast to all peers
                for peer in self.peers:
                    peer.receive_message(message)

    def receive_message(self, message: Any) -> None:
        """Receive message from another node."""
        with self._lock:
            self.messages.append(
                {
                    "message": message,
                    "timestamp": time.time(),
                    "sender": getattr(message, "sender", "unknown"),
                }
            )

            # Handle message if handler exists
            message_type = getattr(message, "type", "default")
            if message_type in self._message_handlers:
                try:
                    self._message_handlers[message_type](message)
                except Exception as e:
                    logging.error(f"Error handling message {message_type}: {e}")

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler."""
        with self._lock:
            self._message_handlers[message_type] = handler

    def get_messages(self, message_type: str = None) -> List[Any]:
        """Get received messages."""
        with self._lock:
            if message_type:
                return [
                    msg
                    for msg in self.messages
                    if getattr(msg["message"], "type", "default") == message_type
                ]
            return self.messages.copy()

    def clear_messages(self) -> None:
        """Clear received messages."""
        with self._lock:
            self.messages.clear()

    def cleanup(self) -> None:
        """Cleanup test node."""
        with self._lock:
            self.stop()
            self.peers.clear()
            self.messages.clear()
            self.state.clear()
            self._message_handlers.clear()


class ClusterManager:
    """Test cluster for multi-node integration testing."""

    def __init__(self, cluster_name: str = "test_cluster"):
        self.cluster_name = cluster_name
        self.nodes: List[NodeManager] = []
        self.network = NetworkManager(f"{cluster_name}_network")
        self.coordinator: Optional[NodeManager] = None
        self._lock = threading.RLock()

    def add_node(self, node_id: str, port: int = None) -> NodeManager:
        """Add node to cluster."""
        with self._lock:
            if port is None:
                port = self.network.get_available_port()

            node = NodeManager(node_id, port)
            self.nodes.append(node)
            self.network.add_node(node)

            return node

    def remove_node(self, node: NodeManager) -> None:
        """Remove node from cluster."""
        with self._lock:
            if node in self.nodes:
                self.nodes.remove(node)
                self.network.remove_node(node)

    def set_coordinator(self, node: NodeManager) -> None:
        """Set cluster coordinator."""
        with self._lock:
            self.coordinator = node

    def start_all_nodes(self) -> None:
        """Start all nodes in cluster."""
        with self._lock:
            for node in self.nodes:
                node.start()

    def stop_all_nodes(self) -> None:
        """Stop all nodes in cluster."""
        with self._lock:
            for node in self.nodes:
                node.stop()

    def connect_all_nodes(self) -> None:
        """Connect all nodes in cluster."""
        with self._lock:
            for i, node1 in enumerate(self.nodes):
                for node2 in self.nodes[i + 1 :]:
                    self.network.connect_nodes(node1, node2)

    def broadcast_to_cluster(self, message: Any, sender: NodeManager = None) -> None:
        """Broadcast message to entire cluster."""
        self.network.broadcast_message(message, sender)

    def get_cluster_state(self) -> Dict[str, Any]:
        """Get cluster state."""
        with self._lock:
            return {
                "cluster_name": self.cluster_name,
                "nodes_count": len(self.nodes),
                "connections_count": len(self.network.connections),
                "coordinator": self.coordinator.node_id if self.coordinator else None,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "port": node.port,
                        "running": node.running,
                        "peers_count": len(node.peers),
                        "messages_count": len(node.messages),
                    }
                    for node in self.nodes
                ],
            }

    def cleanup(self) -> None:
        """Cleanup test cluster."""
        with self._lock:
            self.stop_all_nodes()
            self.network.cleanup()
            self.nodes.clear()
            self.coordinator = None


class IntegrationTestCase(BaseTestCase):
    """Integration test case for testing component interactions."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.database: Optional[DatabaseManager] = None
        self.network: Optional[NetworkManager] = None
        self.cluster: Optional[ClusterManager] = None
        self.external_services: Dict[str, Any] = {}
        self.test_data: Dict[str, Any] = {}

    def setup(self) -> None:
        """Setup integration test case."""
        super().setup()
        self._setup_database()
        self._setup_network()
        self._setup_external_services()
        self._setup_test_data()

    def teardown(self) -> None:
        """Teardown integration test case."""
        self._cleanup_test_data()
        self._cleanup_external_services()
        self._cleanup_network()
        self._cleanup_database()
        super().teardown()

    def _setup_database(self) -> None:
        """Setup test database."""
        self.database = DatabaseManager()

    def _cleanup_database(self) -> None:
        """Cleanup test database."""
        if self.database:
            self.database.cleanup()
            self.database = None

    def _setup_network(self) -> None:
        """Setup test network."""
        self.network = NetworkManager()

    def _cleanup_network(self) -> None:
        """Cleanup test network."""
        if self.network:
            self.network.cleanup()
            self.network = None

    def _setup_external_services(self) -> None:
        """Setup external services."""
        pass

    def _cleanup_external_services(self) -> None:
        """Cleanup external services."""
        for service_name, service in self.external_services.items():
            if hasattr(service, "cleanup"):
                service.cleanup()
        self.external_services.clear()

    def _setup_test_data(self) -> None:
        """Setup test data."""
        pass

    def _cleanup_test_data(self) -> None:
        """Cleanup test data."""
        self.test_data.clear()

    def create_test_cluster(self, node_count: int = 3) -> ClusterManager:
        """Create test cluster with specified number of nodes."""
        cluster = ClusterManager(f"test_cluster_{self.name}")

        for i in range(node_count):
            node_id = f"node_{i}"
            node = cluster.add_node(node_id)
            node.start()

        cluster.connect_all_nodes()
        self.cluster = cluster
        return cluster

    def add_external_service(self, name: str, service: Any) -> None:
        """Add external service."""
        self.external_services[name] = service

    def get_external_service(self, name: str) -> Any:
        """Get external service."""
        if name not in self.external_services:
            raise KeyError(f"External service '{name}' not found")
        return self.external_services[name]

    def set_test_data(self, key: str, value: Any) -> None:
        """Set test data."""
        self.test_data[key] = value

    def get_test_data(self, key: str, default: Any = None) -> Any:
        """Get test data."""
        return self.test_data.get(key, default)

    def wait_for_condition(
        self, condition: Callable, timeout: float = 10.0, interval: float = 0.1
    ) -> bool:
        """Wait for condition to be true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            time.sleep(interval)
        return False

    def assert_cluster_consensus(
        self, expected_value: Any, timeout: float = 10.0
    ) -> None:
        """Assert cluster reaches consensus on value."""
        if not self.cluster:
            raise AssertionError("No test cluster available")

        def check_consensus():
            for node in self.cluster.nodes:
                node_value = node.state.get("consensus_value")
                if node_value != expected_value:
                    return False
            return True

        if not self.wait_for_condition(check_consensus, timeout):
            raise AssertionError(f"Cluster did not reach consensus on {expected_value}")

    def assert_message_delivery(
        self,
        sender: NodeManager,
        receiver: NodeManager,
        message: Any,
        timeout: float = 5.0,
    ) -> None:
        """Assert message was delivered."""

        def check_message():
            messages = receiver.get_messages()
            return any(msg["message"] == message for msg in messages)

        if not self.wait_for_condition(check_message, timeout):
            raise AssertionError(
                f"Message {message} was not delivered from {sender.node_id} to {receiver.node_id}"
            )


class IntegrationTestSuite(SuiteManager):
    """Integration test suite for organizing integration tests."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config.test_type = ExecutionType.INTEGRATION
        self.test_environments: Dict[str, ExecutionEnvironment] = {}

    def add_test(self, test: IntegrationTestCase, environment: str = "default") -> None:
        """Add integration test to suite with environment."""
        super().add_test(test)

        if environment not in self.test_environments:
            self.test_environments[environment] = ExecutionEnvironment(
                environment, self.config
            )

        test.environment = self.test_environments[environment]

    def run_by_environment(
        self, environment: str, config: ExecutionConfig = None
    ) -> List[ExecutionResult]:
        """Run tests by environment."""
        if environment not in self.test_environments:
            return []

        env = self.test_environments[environment]
        env_tests = [test for test in self.tests if test.environment == env]

        config = config or self.config
        results = []

        for test in env_tests:
            try:
                result = test.run(config)
                results.append(result)
            except Exception as e:
                error_result = ExecutionResult(
                    test_name=test.name,
                    test_type=ExecutionType.INTEGRATION,
                    status=ExecutionStatus.ERROR,
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0,
                    error_message=str(e),
                    error_traceback=str(e),
                )
                results.append(error_result)

        return results


class IntegrationTestRunner(RunnerManager):
    """Integration test runner for executing integration test suites."""

    def __init__(self, config: ExecutionConfig = None):
        super().__init__(config)
        self.config.test_type = ExecutionType.INTEGRATION
        self.parallel_execution = False
        self.environment_isolation = True

    def run_with_environment_isolation(self) -> List[ExecutionResult]:
        """Run tests with environment isolation."""
        # This would implement proper environment isolation
        # For now, we'll just run normally
        return self.run_all()

    def run_with_parallel_execution(
        self, max_workers: int = 4
    ) -> List[ExecutionResult]:
        """Run tests with parallel execution."""
        # This would implement parallel test execution
        # For now, we'll just run normally
        return self.run_all()
