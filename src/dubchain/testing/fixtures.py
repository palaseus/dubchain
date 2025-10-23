"""Test fixtures for DubChain.

This module provides pre-built test fixtures for common testing scenarios.
"""

import logging

logger = logging.getLogger(__name__)
import os
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..logging import get_logger
from .base import ExecutionConfig, ExecutionEnvironment, ExecutionType, FixtureManager
from .integration import ClusterManager, DatabaseManager, NetworkManager, NodeManager


class FixturesManager:
    """Base test fixtures class."""

    def __init__(self):
        self.fixtures: Dict[str, FixtureManager] = {}
        self.logger = get_logger("test_fixtures")

    def add_fixture(self, name: str, fixture: FixtureManager) -> None:
        """Add test fixture."""
        self.fixtures[name] = fixture

    def get_fixture(self, name: str) -> FixtureManager:
        """Get test fixture."""
        if name not in self.fixtures:
            raise KeyError(f"Fixture '{name}' not found")
        return self.fixtures[name]

    def setup_all(self) -> None:
        """Setup all fixtures."""
        for fixture in self.fixtures.values():
            fixture.setup()

    def teardown_all(self) -> None:
        """Teardown all fixtures."""
        for fixture in self.fixtures.values():
            fixture.teardown()


class DatabaseFixtures(FixturesManager):
    """Database test fixtures."""

    def __init__(self):
        super().__init__()
        self._setup_database_fixtures()

    def _setup_database_fixtures(self) -> None:
        """Setup database fixtures."""

        # SQLite test database
        def setup_sqlite_db():
            db = DatabaseManager("sqlite")
            return {"database": db}

        def teardown_sqlite_db(data):
            if "database" in data:
                data["database"].cleanup()

        sqlite_fixture = FixtureManager(
            "sqlite_database", setup_sqlite_db, teardown_sqlite_db
        )
        self.add_fixture("sqlite_database", sqlite_fixture)

        # In-memory test database
        def setup_memory_db():
            db = DatabaseManager("sqlite", ":memory:")
            return {"database": db}

        def teardown_memory_db(data):
            if "database" in data:
                data["database"].cleanup()

        memory_fixture = FixtureManager(
            "memory_database", setup_memory_db, teardown_memory_db
        )
        self.add_fixture("memory_database", memory_fixture)

    def create_database_fixture(self, name: str) -> FixtureManager:
        """Create a database fixture."""

        def setup_func():
            from .integration import DatabaseManager

            db = DatabaseManager("sqlite")
            return {"database": db}

        def teardown_func(data):
            if "database" in data:
                data["database"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_memory_database_fixture(self, name: str) -> FixtureManager:
        """Create a memory database fixture."""

        def setup_func():
            from .integration import DatabaseManager

            db = DatabaseManager("sqlite", ":memory:")
            return {"database": db}

        def teardown_func(data):
            if "database" in data:
                data["database"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def get_sqlite_database(self) -> DatabaseManager:
        """Get SQLite test database."""
        fixture = self.get_fixture("sqlite_database")
        return fixture.get("database")

    def get_memory_database(self) -> DatabaseManager:
        """Get in-memory test database."""
        fixture = self.get_fixture("memory_database")
        return fixture.get("database")


class NetworkFixtures(FixturesManager):
    """Network test fixtures."""

    def __init__(self):
        super().__init__()
        self._setup_network_fixtures()

    def _setup_network_fixtures(self) -> None:
        """Setup network fixtures."""

        # Local test network
        def setup_local_network():
            network = NetworkManager("local_test_network")
            return {"network": network}

        def teardown_local_network(data):
            if "network" in data:
                data["network"].cleanup()

        local_network_fixture = FixtureManager(
            "local_network", setup_local_network, teardown_local_network
        )
        self.add_fixture("local_network", local_network_fixture)

        # Multi-node test network
        def setup_multi_node_network():
            network = NetworkManager("multi_node_network")
            # Add some test nodes
            for i in range(3):
                node = NodeManager(f"node_{i}")
                network.add_node(node)
            return {"network": network}

        def teardown_multi_node_network(data):
            if "network" in data:
                data["network"].cleanup()

        multi_node_fixture = FixtureManager(
            "multi_node_network", setup_multi_node_network, teardown_multi_node_network
        )
        self.add_fixture("multi_node_network", multi_node_fixture)

    def create_network_fixture(self, name: str) -> FixtureManager:
        """Create a network fixture."""

        def setup_func():
            from .integration import NetworkManager

            network = NetworkManager(name)
            return {"network": network}

        def teardown_func(data):
            if "network" in data:
                data["network"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_local_network_fixture(self, name: str) -> FixtureManager:
        """Create a local network fixture."""

        def setup_func():
            from .integration import NetworkManager

            network = NetworkManager(name, network_type="local")
            return {"network": network}

        def teardown_func(data):
            if "network" in data:
                data["network"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def get_local_network(self) -> NetworkManager:
        """Get local test network."""
        fixture = self.get_fixture("local_network")
        return fixture.get("network")

    def get_multi_node_network(self) -> NetworkManager:
        """Get multi-node test network."""
        fixture = self.get_fixture("multi_node_network")
        return fixture.get("network")


class NodeFixtures(FixturesManager):
    """Node test fixtures."""

    def __init__(self):
        super().__init__()
        self._setup_node_fixtures()

    def _setup_node_fixtures(self) -> None:
        """Setup node fixtures."""

        # Single test node
        def setup_single_node():
            node = NodeManager("test_node", 8080)
            return {"node": node}

        def teardown_single_node(data):
            if "node" in data:
                data["node"].cleanup()

        single_node_fixture = FixtureManager(
            "single_node", setup_single_node, teardown_single_node
        )
        self.add_fixture("single_node", single_node_fixture)

        # Node cluster
        def setup_node_cluster():
            cluster = ClusterManager("test_cluster")
            for i in range(3):
                cluster.add_node(f"node_{i}")
            return {"cluster": cluster}

        def teardown_node_cluster(data):
            if "cluster" in data:
                data["cluster"].cleanup()

        node_cluster_fixture = FixtureManager(
            "node_cluster", setup_node_cluster, teardown_node_cluster
        )
        self.add_fixture("node_cluster", node_cluster_fixture)

    def create_node_fixture(self, name: str) -> FixtureManager:
        """Create a node fixture."""

        def setup_func():
            from .integration import NodeManager

            node = NodeManager(name, port=8080)
            return {"node": node}

        def teardown_func(data):
            if "node" in data:
                data["node"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_validator_node_fixture(self, name: str) -> FixtureManager:
        """Create a validator node fixture."""

        def setup_func():
            from .integration import NodeManager

            node = NodeManager(name, port=8080, node_type="validator")
            return {"node": node}

        def teardown_func(data):
            if "node" in data:
                data["node"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def get_single_node(self) -> NodeManager:
        """Get single test node."""
        fixture = self.get_fixture("single_node")
        return fixture.get("node")

    def get_node_cluster(self) -> ClusterManager:
        """Get node cluster."""
        fixture = self.get_fixture("node_cluster")
        return fixture.get("cluster")


class BlockchainFixtures(FixturesManager):
    """Blockchain test fixtures."""

    def __init__(self):
        super().__init__()
        self._setup_blockchain_fixtures()

    def _setup_blockchain_fixtures(self) -> None:
        """Setup blockchain fixtures."""

        # Test blockchain data
        def setup_blockchain_data():
            return {
                "genesis_block": {
                    "id": "genesis_block",
                    "previous_hash": "0",
                    "merkle_root": "genesis_merkle_root",
                    "timestamp": time.time(),
                    "nonce": 0,
                    "transactions": [],
                },
                "test_transactions": [
                    {
                        "id": "tx_1",
                        "from_address": "alice",
                        "to_address": "bob",
                        "amount": 100,
                        "timestamp": time.time(),
                    },
                    {
                        "id": "tx_2",
                        "from_address": "bob",
                        "to_address": "charlie",
                        "amount": 50,
                        "timestamp": time.time(),
                    },
                ],
                "test_blocks": [
                    {
                        "id": "block_1",
                        "previous_hash": "genesis_block",
                        "merkle_root": "block_1_merkle_root",
                        "timestamp": time.time(),
                        "nonce": 12345,
                        "transactions": ["tx_1"],
                    },
                    {
                        "id": "block_2",
                        "previous_hash": "block_1",
                        "merkle_root": "block_2_merkle_root",
                        "timestamp": time.time(),
                        "nonce": 67890,
                        "transactions": ["tx_2"],
                    },
                ],
            }

        def teardown_blockchain_data(data):
            data.clear()

        blockchain_data_fixture = FixtureManager(
            "blockchain_data", setup_blockchain_data, teardown_blockchain_data
        )
        self.add_fixture("blockchain_data", blockchain_data_fixture)

        # Test wallet data
        def setup_wallet_data():
            return {
                "test_wallets": [
                    {
                        "address": "alice",
                        "private_key": "alice_private_key",
                        "public_key": "alice_public_key",
                        "balance": 1000,
                    },
                    {
                        "address": "bob",
                        "private_key": "bob_private_key",
                        "public_key": "bob_public_key",
                        "balance": 500,
                    },
                    {
                        "address": "charlie",
                        "private_key": "charlie_private_key",
                        "public_key": "charlie_public_key",
                        "balance": 250,
                    },
                ]
            }

        def teardown_wallet_data(data):
            data.clear()

        wallet_data_fixture = FixtureManager(
            "wallet_data", setup_wallet_data, teardown_wallet_data
        )
        self.add_fixture("wallet_data", wallet_data_fixture)

        # Test smart contract data
        def setup_smart_contract_data():
            return {
                "test_contracts": [
                    {
                        "id": "contract_1",
                        "name": "SimpleToken",
                        "bytecode": "0x608060405234801561001057600080fd5b50",
                        "abi": [
                            {
                                "name": "transfer",
                                "type": "function",
                                "inputs": [
                                    {"name": "to", "type": "address"},
                                    {"name": "amount", "type": "uint256"},
                                ],
                                "outputs": [{"name": "", "type": "bool"}],
                            }
                        ],
                        "deployed_address": "0x1234567890123456789012345678901234567890",
                    }
                ]
            }

        def teardown_smart_contract_data(data):
            data.clear()

        smart_contract_fixture = FixtureManager(
            "smart_contract_data",
            setup_smart_contract_data,
            teardown_smart_contract_data,
        )
        self.add_fixture("smart_contract_data", smart_contract_fixture)

    def get_blockchain_data(self) -> Dict[str, Any]:
        """Get blockchain test data."""
        fixture = self.get_fixture("blockchain_data")
        return fixture.data

    def get_wallet_data(self) -> Dict[str, Any]:
        """Get wallet test data."""
        fixture = self.get_fixture("wallet_data")
        return fixture.data

    def get_smart_contract_data(self) -> Dict[str, Any]:
        """Get smart contract test data."""
        fixture = self.get_fixture("smart_contract_data")
        return fixture.data

    def create_blockchain_fixture(self, name: str) -> FixtureManager:
        """Create a blockchain fixture."""

        def setup_func():
            from ..testing.base import ExecutionEnvironment

            env = ExecutionEnvironment()
            return {"environment": env}

        def teardown_func(data):
            if "environment" in data:
                data["environment"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_genesis_blockchain_fixture(self, name: str) -> FixtureManager:
        """Create a genesis blockchain fixture."""

        def setup_func():
            from ..testing.base import ExecutionEnvironment

            env = ExecutionEnvironment(has_genesis=True)
            return {"environment": env}

        def teardown_func(data):
            if "environment" in data:
                data["environment"].cleanup()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture


class ClusterFixtures(FixturesManager):
    """Cluster testing fixtures."""

    def __init__(self):
        """Initialize cluster fixtures."""
        super().__init__()
        self.clusters = {}

    def create_cluster_fixture(self, name: str, node_count: int = 3) -> FixtureManager:
        """Create a cluster fixture."""

        def setup_func():
            from .integration import ClusterManager

            cluster = ClusterManager(node_count=node_count)
            return {"cluster": cluster}

        def teardown_func(data):
            if "cluster" in data:
                data["cluster"].stop()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_small_cluster_fixture(self, name: str) -> Any:
        """Create a small cluster fixture."""
        return self.create_cluster_fixture(name, node_count=2)

    def _setup_cluster_fixtures(self) -> None:
        """Setup cluster fixtures."""
        pass

    def _teardown_cluster_fixtures(self) -> None:
        """Teardown cluster fixtures."""
        for cluster in self.clusters.values():
            if hasattr(cluster, "stop"):
                cluster.stop()


class ConsensusFixtures(FixturesManager):
    """Consensus testing fixtures."""

    def __init__(self):
        """Initialize consensus fixtures."""
        super().__init__()
        self.consensus_engines = {}

    def create_consensus_fixture(
        self, name: str, consensus_type: str = "pow"
    ) -> FixtureManager:
        """Create a consensus fixture."""

        def setup_func():
            from ..core.consensus import ConsensusConfig, ConsensusEngine

            config = ConsensusConfig(consensus_type=consensus_type)
            engine = ConsensusEngine(config)
            return {"engine": engine}

        def teardown_func(data):
            if "engine" in data and hasattr(data["engine"], "stop"):
                data["engine"].stop()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_pow_consensus_fixture(self, name: str) -> Any:
        """Create a PoW consensus fixture."""
        return self.create_consensus_fixture(name, "pow")

    def _setup_consensus_fixtures(self) -> None:
        """Setup consensus fixtures."""
        pass

    def _teardown_consensus_fixtures(self) -> None:
        """Teardown consensus fixtures."""
        for engine in self.consensus_engines.values():
            if hasattr(engine, "stop"):
                engine.stop()


class StorageFixtures(FixturesManager):
    """Storage testing fixtures."""

    def __init__(self):
        """Initialize storage fixtures."""
        super().__init__()
        self.storage_backends = {}

    def create_storage_fixture(
        self, name: str, storage_type: str = "memory"
    ) -> FixtureManager:
        """Create a storage fixture."""

        def setup_func():
            from ..storage.database import DatabaseConfig, SQLiteBackend

            if storage_type == "memory":
                config = DatabaseConfig(database_path=":memory:")
            else:
                import tempfile

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
                config = DatabaseConfig(database_path=temp_file.name)
                temp_file.close()

            backend = SQLiteBackend(config)
            return {"backend": backend}

        def teardown_func(data):
            if "backend" in data and hasattr(data["backend"], "close"):
                data["backend"].close()

        fixture = FixtureManager(
            name=name, setup_func=setup_func, teardown_func=teardown_func
        )
        self.add_fixture(name, fixture)
        return fixture

    def create_memory_storage_fixture(self, name: str) -> Any:
        """Create a memory storage fixture."""
        return self.create_storage_fixture(name, "memory")

    def create_file_storage_fixture(self, name: str) -> Any:
        """Create a file storage fixture."""
        return self.create_storage_fixture(name, "file")

    def _setup_storage_fixtures(self) -> None:
        """Setup storage fixtures."""
        pass

    def _teardown_storage_fixtures(self) -> None:
        """Teardown storage fixtures."""
        for backend in self.storage_backends.values():
            if hasattr(backend, "close"):
                backend.close()


class EnvironmentFixturesManager:
    """Test environment fixtures."""

    def __init__(self):
        self.environments: Dict[str, ExecutionEnvironment] = {}
        self.logger = get_logger("test_environment_fixtures")

    def create_development_environment(self) -> ExecutionEnvironment:
        """Create development test environment."""
        config = ExecutionConfig()
        config.test_type = ExecutionType.UNIT
        config.verbose = True
        config.collect_coverage = True

        environment = ExecutionEnvironment("development", config)

        # Add development-specific setup hooks
        def setup_dev_environment():
            self.logger.info("Setting up development environment")

        def teardown_dev_environment():
            self.logger.info("Tearing down development environment")

        environment.setup_hooks.append(setup_dev_environment)
        environment.teardown_hooks.append(teardown_dev_environment)

        self.environments["development"] = environment
        return environment

    def create_production_environment(self) -> ExecutionEnvironment:
        """Create production test environment."""
        config = ExecutionConfig()
        config.test_type = ExecutionType.INTEGRATION
        config.verbose = False
        config.collect_coverage = False
        config.collect_metrics = True

        environment = ExecutionEnvironment("production", config)

        # Add production-specific setup hooks
        def setup_prod_environment():
            self.logger.info("Setting up production environment")

        def teardown_prod_environment():
            self.logger.info("Tearing down production environment")

        environment.setup_hooks.append(setup_prod_environment)
        environment.teardown_hooks.append(teardown_prod_environment)

        self.environments["production"] = environment
        return environment

    def create_ci_environment(self) -> ExecutionEnvironment:
        """Create CI test environment."""
        config = ExecutionConfig()
        config.test_type = ExecutionType.UNIT
        config.verbose = False
        config.collect_coverage = True
        config.parallel = True
        config.max_workers = 4

        environment = ExecutionEnvironment("ci", config)

        # Add CI-specific setup hooks
        def setup_ci_environment():
            self.logger.info("Setting up CI environment")

        def teardown_ci_environment():
            self.logger.info("Tearing down CI environment")

        environment.setup_hooks.append(setup_ci_environment)
        environment.teardown_hooks.append(teardown_ci_environment)

        self.environments["ci"] = environment
        return environment

    def get_environment(self, name: str) -> ExecutionEnvironment:
        """Get test environment by name."""
        if name not in self.environments:
            raise KeyError(f"Environment '{name}' not found")
        return self.environments[name]

    def get_all_environments(self) -> Dict[str, ExecutionEnvironment]:
        """Get all test environments."""
        return self.environments.copy()
