"""
Comprehensive tests for bridge manager module.

This module tests the bridge management system including:
- Chain management functionality
- Asset management across chains
- Bridge validator management
- Cross-chain transaction processing
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.bridge.bridge_manager import AssetManager, BridgeManager, ChainManager
from dubchain.bridge.bridge_types import (
    AssetType,
    BridgeAsset,
    BridgeConfig,
    BridgeMetrics,
    BridgeStatus,
    BridgeType,
    BridgeValidator,
    ChainType,
    CrossChainTransaction,
)


class TestChainManager:
    """Test ChainManager functionality."""

    def test_chain_manager_creation(self):
        """Test creating a chain manager."""
        manager = ChainManager()

        assert manager.supported_chains == {}
        assert manager.chain_connections == {}
        assert manager.chain_metrics == {}

    def test_add_chain(self):
        """Test adding a new blockchain network."""
        manager = ChainManager()
        chain_info = {
            "name": "DubChain Mainnet",
            "type": ChainType.MAINNET,
            "rpc_url": "https://rpc.dubchain.com",
            "chain_id": 1,
        }

        result = manager.add_chain("dubchain_mainnet", chain_info)

        assert result is True
        assert "dubchain_mainnet" in manager.supported_chains
        assert manager.supported_chains["dubchain_mainnet"] == chain_info
        assert manager.chain_connections["dubchain_mainnet"] is True
        assert "dubchain_mainnet" in manager.chain_metrics

    def test_add_chain_duplicate(self):
        """Test adding duplicate chain."""
        manager = ChainManager()
        chain_info = {"name": "DubChain Mainnet"}

        # Add first time
        result1 = manager.add_chain("dubchain_mainnet", chain_info)
        assert result1 is True

        # Add second time
        result2 = manager.add_chain("dubchain_mainnet", chain_info)
        assert result2 is False

    def test_remove_chain(self):
        """Test removing a blockchain network."""
        manager = ChainManager()
        chain_info = {"name": "DubChain Mainnet"}

        # Add chain first
        manager.add_chain("dubchain_mainnet", chain_info)

        # Remove chain
        result = manager.remove_chain("dubchain_mainnet")

        assert result is True
        assert "dubchain_mainnet" not in manager.supported_chains
        assert "dubchain_mainnet" not in manager.chain_connections
        assert "dubchain_mainnet" not in manager.chain_metrics

    def test_remove_chain_not_found(self):
        """Test removing non-existent chain."""
        manager = ChainManager()

        result = manager.remove_chain("nonexistent_chain")
        assert result is False

    def test_update_chain_status(self):
        """Test updating chain connection status."""
        manager = ChainManager()
        chain_info = {"name": "DubChain Mainnet"}

        # Add chain
        manager.add_chain("dubchain_mainnet", chain_info)

        # Update status
        manager.update_chain_status("dubchain_mainnet", False)
        assert manager.chain_connections["dubchain_mainnet"] is False

        manager.update_chain_status("dubchain_mainnet", True)
        assert manager.chain_connections["dubchain_mainnet"] is True

    def test_update_chain_status_not_found(self):
        """Test updating status for non-existent chain."""
        manager = ChainManager()

        # Should not raise exception
        manager.update_chain_status("nonexistent_chain", False)

    def test_get_chain_info(self):
        """Test getting chain information."""
        manager = ChainManager()
        chain_info = {"name": "DubChain Mainnet", "type": "mainnet"}

        # Add chain
        manager.add_chain("dubchain_mainnet", chain_info)

        # Get info
        retrieved_info = manager.get_chain_info("dubchain_mainnet")
        assert retrieved_info == chain_info

        # Get non-existent chain
        non_existent = manager.get_chain_info("nonexistent_chain")
        assert non_existent is None

    def test_get_connected_chains(self):
        """Test getting list of connected chains."""
        manager = ChainManager()

        # Add chains with different statuses
        manager.add_chain("chain1", {"name": "Chain 1"})
        manager.add_chain("chain2", {"name": "Chain 2"})
        manager.add_chain("chain3", {"name": "Chain 3"})

        # Update statuses
        manager.update_chain_status("chain1", True)
        manager.update_chain_status("chain2", False)
        manager.update_chain_status("chain3", True)

        connected = manager.get_connected_chains()
        assert "chain1" in connected
        assert "chain2" not in connected
        assert "chain3" in connected
        assert len(connected) == 2

    def test_update_chain_metrics(self):
        """Test updating chain metrics."""
        manager = ChainManager()
        chain_info = {"name": "DubChain Mainnet"}

        # Add chain
        manager.add_chain("dubchain_mainnet", chain_info)

        # Update metrics
        manager.update_chain_metrics("dubchain_mainnet", transactions=10, volume=1000)

        metrics = manager.chain_metrics["dubchain_mainnet"]
        assert metrics["transactions"] == 10
        assert metrics["volume"] == 1000
        assert "last_activity" in metrics

        # Update again
        manager.update_chain_metrics("dubchain_mainnet", transactions=5, volume=500)

        metrics = manager.chain_metrics["dubchain_mainnet"]
        assert metrics["transactions"] == 15  # Accumulated
        assert metrics["volume"] == 1500  # Accumulated

    def test_update_chain_metrics_not_found(self):
        """Test updating metrics for non-existent chain."""
        manager = ChainManager()

        # Should not raise exception
        manager.update_chain_metrics("nonexistent_chain", transactions=10, volume=1000)


class TestAssetManager:
    """Test AssetManager functionality."""

    def test_asset_manager_creation(self):
        """Test creating an asset manager."""
        manager = AssetManager()

        assert manager.registered_assets == {}
        assert manager.asset_mappings == {}
        assert manager.asset_balances == {}

    def test_register_asset(self):
        """Test registering a new asset."""
        manager = AssetManager()
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        result = manager.register_asset(asset)

        assert result is True
        assert "DUB" in manager.registered_assets
        assert manager.registered_assets["DUB"] == asset
        assert "DUB" in manager.asset_mappings
        assert "DUB" in manager.asset_balances

    def test_register_asset_duplicate(self):
        """Test registering duplicate asset."""
        manager = AssetManager()
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        # Register first time
        result1 = manager.register_asset(asset)
        assert result1 is True

        # Register second time
        result2 = manager.register_asset(asset)
        assert result2 is False

    def test_add_asset_to_chain(self):
        """Test adding asset to another chain."""
        manager = AssetManager()
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        # Register asset first
        manager.register_asset(asset)

        # Add to another chain
        result = manager.add_asset_to_chain(
            "DUB", "ethereum_mainnet", "0xabcdef1234567890"
        )

        assert result is True
        assert "ethereum_mainnet" in manager.asset_mappings["DUB"]
        assert manager.asset_mappings["DUB"]["ethereum_mainnet"] == "0xabcdef1234567890"
        assert "ethereum_mainnet" in manager.asset_balances["DUB"]
        assert manager.asset_balances["DUB"]["ethereum_mainnet"] == 0

    def test_add_asset_to_chain_not_registered(self):
        """Test adding non-registered asset to chain."""
        manager = AssetManager()

        result = manager.add_asset_to_chain("NONEXISTENT", "ethereum_mainnet", "0x123")
        assert result is False

    def test_get_asset_balance(self):
        """Test getting asset balance on specific chain."""
        manager = AssetManager()
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        # Register asset
        manager.register_asset(asset)

        # Get balance
        balance = manager.get_asset_balance("DUB", "dubchain_mainnet")
        assert balance == 0

        # Get balance for non-existent asset
        balance = manager.get_asset_balance("NONEXISTENT", "dubchain_mainnet")
        assert balance == 0

        # Get balance for non-existent chain
        balance = manager.get_asset_balance("DUB", "nonexistent_chain")
        assert balance == 0

    def test_update_asset_balance(self):
        """Test updating asset balance on specific chain."""
        manager = AssetManager()
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        # Register asset
        manager.register_asset(asset)

        # Update balance
        manager.update_asset_balance("DUB", "dubchain_mainnet", 1000)

        balance = manager.get_asset_balance("DUB", "dubchain_mainnet")
        assert balance == 1000

        # Update balance for non-existent asset
        manager.update_asset_balance("NONEXISTENT", "dubchain_mainnet", 500)
        balance = manager.get_asset_balance("NONEXISTENT", "dubchain_mainnet")
        assert balance == 500

    def test_get_asset_info(self):
        """Test getting asset information."""
        manager = AssetManager()
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        # Register asset
        manager.register_asset(asset)

        # Get info
        retrieved_asset = manager.get_asset_info("DUB")
        assert retrieved_asset == asset

        # Get non-existent asset
        non_existent = manager.get_asset_info("NONEXISTENT")
        assert non_existent is None

    def test_get_chain_assets(self):
        """Test getting all assets on specific chain."""
        manager = AssetManager()

        # Create assets for different chains
        asset1 = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        asset2 = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        # Register assets
        manager.register_asset(asset1)
        manager.register_asset(asset2)

        # Add asset1 to ethereum chain
        manager.add_asset_to_chain("DUB", "ethereum_mainnet", "0x1111111111111111")

        # Get assets for dubchain_mainnet
        dubchain_assets = manager.get_chain_assets("dubchain_mainnet")
        assert len(dubchain_assets) == 1
        assert dubchain_assets[0].asset_id == "DUB"

        # Get assets for ethereum_mainnet
        ethereum_assets = manager.get_chain_assets("ethereum_mainnet")
        assert len(ethereum_assets) == 2  # ETH and DUB

        # Get assets for non-existent chain
        nonexistent_assets = manager.get_chain_assets("nonexistent_chain")
        assert len(nonexistent_assets) == 0


class TestBridgeManager:
    """Test BridgeManager functionality."""

    @pytest.fixture
    def bridge_config(self):
        """Fixture for bridge configuration."""
        return BridgeConfig(
            bridge_type=BridgeType.LOCK_AND_MINT,
            min_transfer_amount=1,
            max_transfer_amount=1000000,
            timeout_blocks=100,
            security_threshold=0.67,
        )

    @pytest.fixture
    def bridge_manager(self, bridge_config):
        """Fixture for bridge manager."""
        return BridgeManager(bridge_config)

    def test_bridge_manager_creation(self, bridge_config):
        """Test creating a bridge manager."""
        manager = BridgeManager(bridge_config)

        assert manager.config == bridge_config
        assert manager.status == BridgeStatus.ACTIVE
        assert isinstance(manager.chain_manager, ChainManager)
        assert isinstance(manager.asset_manager, AssetManager)
        assert manager.validators == {}
        assert manager.transactions == {}
        assert isinstance(manager.metrics, BridgeMetrics)
        assert manager.created_at > 0

    def test_add_chain(self, bridge_manager):
        """Test adding a new blockchain network."""
        chain_info = {
            "name": "DubChain Mainnet",
            "type": ChainType.MAINNET,
            "rpc_url": "https://rpc.dubchain.com",
            "chain_id": 1,
        }

        result = bridge_manager.add_chain("dubchain_mainnet", chain_info)

        assert result is True
        assert bridge_manager.metrics.supported_chains == 1

    def test_register_asset(self, bridge_manager):
        """Test registering a new asset."""
        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        result = bridge_manager.register_asset(asset)

        assert result is True
        assert bridge_manager.metrics.supported_assets == 1

    def test_add_validator(self, bridge_manager):
        """Test adding bridge validator."""
        validator = BridgeValidator(
            validator_id="validator_1",
            chain_id="dubchain_mainnet",
            public_key="0x1234567890abcdef",
            stake_amount=1000,
            is_active=True,
        )

        result = bridge_manager.add_validator(validator)

        assert result is True
        assert "validator_1" in bridge_manager.validators
        assert bridge_manager.metrics.active_validators == 1

    def test_add_validator_duplicate(self, bridge_manager):
        """Test adding duplicate validator."""
        validator = BridgeValidator(
            validator_id="validator_1",
            chain_id="dubchain_mainnet",
            public_key="0x1234567890abcdef",
            stake_amount=1000,
            is_active=True,
        )

        # Add first time
        result1 = bridge_manager.add_validator(validator)
        assert result1 is True

        # Add second time
        result2 = bridge_manager.add_validator(validator)
        assert result2 is False

    def test_remove_validator(self, bridge_manager):
        """Test removing bridge validator."""
        validator = BridgeValidator(
            validator_id="validator_1",
            chain_id="dubchain_mainnet",
            public_key="0x1234567890abcdef",
            stake_amount=1000,
            is_active=True,
        )

        # Add validator first
        bridge_manager.add_validator(validator)

        # Remove validator
        result = bridge_manager.remove_validator("validator_1")

        assert result is True
        assert "validator_1" not in bridge_manager.validators
        assert bridge_manager.metrics.active_validators == 0

    def test_remove_validator_not_found(self, bridge_manager):
        """Test removing non-existent validator."""
        result = bridge_manager.remove_validator("nonexistent_validator")
        assert result is False

    def test_create_cross_chain_transaction(self, bridge_manager):
        """Test creating a cross-chain transaction."""
        # Add chains
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        # Register assets
        source_asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        target_asset = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        bridge_manager.register_asset(source_asset)
        bridge_manager.register_asset(target_asset)

        # Create transaction
        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,
        )

        assert transaction is not None
        assert transaction.source_chain == "dubchain_mainnet"
        assert transaction.target_chain == "ethereum_mainnet"
        assert transaction.source_asset == "DUB"
        assert transaction.target_asset == "ETH"
        assert transaction.sender == "alice"
        assert transaction.receiver == "bob"
        assert transaction.amount == 1000
        assert transaction.transaction_id in bridge_manager.transactions

    def test_create_cross_chain_transaction_invalid_chain(self, bridge_manager):
        """Test creating transaction with invalid chain."""
        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="nonexistent_chain",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,
        )

        assert transaction is None

    def test_create_cross_chain_transaction_invalid_asset(self, bridge_manager):
        """Test creating transaction with invalid asset."""
        # Add chains
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="NONEXISTENT",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,
        )

        assert transaction is None

    def test_create_cross_chain_transaction_invalid_amount(self, bridge_manager):
        """Test creating transaction with invalid amount."""
        # Add chains
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        # Register assets
        source_asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        target_asset = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        bridge_manager.register_asset(source_asset)
        bridge_manager.register_asset(target_asset)

        # Test amount too small
        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=0,  # Below minimum
        )

        assert transaction is None

        # Test amount too large
        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=2000000,  # Above maximum
        )

        assert transaction is None

    def test_process_transaction_lock_and_mint(self, bridge_manager):
        """Test processing lock and mint transaction."""
        # Setup chains and assets
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        source_asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        target_asset = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        bridge_manager.register_asset(source_asset)
        bridge_manager.register_asset(target_asset)

        # Set initial balance
        bridge_manager.asset_manager.update_asset_balance(
            "DUB", "dubchain_mainnet", 10000
        )

        # Create transaction
        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,
        )

        # Process transaction
        result = bridge_manager.process_transaction(transaction.transaction_id)

        assert result is True
        assert transaction.status == "completed"
        assert transaction.completed_at is not None
        assert transaction.completion_hash is not None

        # Check balances
        source_balance = bridge_manager.asset_manager.get_asset_balance(
            "DUB", "dubchain_mainnet"
        )
        target_balance = bridge_manager.asset_manager.get_asset_balance(
            "ETH", "ethereum_mainnet"
        )

        assert source_balance == 9000  # 10000 - 1000
        assert target_balance == 1000  # 0 + 1000

        # Check metrics
        assert bridge_manager.metrics.successful_transactions == 1
        assert bridge_manager.metrics.total_volume == 1000

    def test_process_transaction_insufficient_balance(self, bridge_manager):
        """Test processing transaction with insufficient balance."""
        # Setup chains and assets
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        source_asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        target_asset = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        bridge_manager.register_asset(source_asset)
        bridge_manager.register_asset(target_asset)

        # Set insufficient balance
        bridge_manager.asset_manager.update_asset_balance(
            "DUB", "dubchain_mainnet", 500
        )

        # Create transaction
        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,  # More than available balance
        )

        # Process transaction
        result = bridge_manager.process_transaction(transaction.transaction_id)

        assert result is False
        assert transaction.status == "failed"
        assert bridge_manager.metrics.failed_transactions == 1

    def test_process_transaction_not_found(self, bridge_manager):
        """Test processing non-existent transaction."""
        result = bridge_manager.process_transaction("nonexistent_transaction_id")
        assert result is False

    def test_get_transaction(self, bridge_manager):
        """Test getting transaction by ID."""
        # Create a transaction
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        source_asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        target_asset = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        bridge_manager.register_asset(source_asset)
        bridge_manager.register_asset(target_asset)

        transaction = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,
        )

        # Get transaction
        retrieved_transaction = bridge_manager.get_transaction(
            transaction.transaction_id
        )
        assert retrieved_transaction == transaction

        # Get non-existent transaction
        non_existent = bridge_manager.get_transaction("nonexistent_id")
        assert non_existent is None

    def test_get_transactions_by_status(self, bridge_manager):
        """Test getting transactions by status."""
        # Create transactions with different statuses
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        source_asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        target_asset = BridgeAsset(
            asset_id="ETH",
            name="Ethereum",
            symbol="ETH",
            chain_id="ethereum_mainnet",
            contract_address="0xabcdef1234567890",
            asset_type=AssetType.NATIVE,
            decimals=18,
        )

        bridge_manager.register_asset(source_asset)
        bridge_manager.register_asset(target_asset)

        # Create transactions
        transaction1 = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="alice",
            receiver="bob",
            amount=1000,
        )

        transaction2 = bridge_manager.create_cross_chain_transaction(
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            sender="charlie",
            receiver="david",
            amount=2000,
        )

        # Set different statuses
        transaction1.status = "completed"
        transaction2.status = "pending"

        # Get transactions by status
        completed_transactions = bridge_manager.get_transactions_by_status("completed")
        pending_transactions = bridge_manager.get_transactions_by_status("pending")
        failed_transactions = bridge_manager.get_transactions_by_status("failed")

        assert len(completed_transactions) == 1
        assert len(pending_transactions) == 1
        assert len(failed_transactions) == 0
        assert completed_transactions[0].transaction_id == transaction1.transaction_id
        assert pending_transactions[0].transaction_id == transaction2.transaction_id

    def test_get_bridge_metrics(self, bridge_manager):
        """Test getting bridge metrics."""
        # Add some data
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        bridge_manager.register_asset(asset)

        validator = BridgeValidator(
            validator_id="validator_1",
            chain_id="dubchain_mainnet",
            public_key="0x1234567890abcdef",
            stake_amount=1000,
            is_active=True,
        )

        bridge_manager.add_validator(validator)

        # Get metrics
        metrics = bridge_manager.get_bridge_metrics()

        assert metrics.total_transactions == 0
        assert metrics.supported_chains == 2
        assert metrics.supported_assets == 1
        assert metrics.active_validators == 1
        assert metrics.last_updated > 0

    def test_get_bridge_status(self, bridge_manager):
        """Test getting comprehensive bridge status."""
        # Add some data
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})
        bridge_manager.add_chain("ethereum_mainnet", {"name": "Ethereum"})

        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        bridge_manager.register_asset(asset)

        validator = BridgeValidator(
            validator_id="validator_1",
            chain_id="dubchain_mainnet",
            public_key="0x1234567890abcdef",
            stake_amount=1000,
            is_active=True,
        )

        bridge_manager.add_validator(validator)

        # Get status
        status = bridge_manager.get_bridge_status()

        assert status["status"] == BridgeStatus.ACTIVE.value
        assert status["bridge_type"] == BridgeType.LOCK_AND_MINT.value
        assert len(status["supported_chains"]) == 2
        assert len(status["supported_assets"]) == 1
        assert status["active_validators"] == 1
        assert status["total_transactions"] == 0
        assert "metrics" in status

    def test_to_dict(self, bridge_manager):
        """Test converting to dictionary."""
        # Add some data
        bridge_manager.add_chain("dubchain_mainnet", {"name": "DubChain"})

        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        bridge_manager.register_asset(asset)

        # Convert to dict
        data = bridge_manager.to_dict()

        assert "config" in data
        assert "status" in data
        assert "chain_manager" in data
        assert "asset_manager" in data
        assert "validators" in data
        assert "transactions" in data
        assert "metrics" in data
        assert "created_at" in data

    def test_from_dict(self, bridge_config):
        """Test creating from dictionary."""
        # Create manager with data
        manager = BridgeManager(bridge_config)
        manager.add_chain("dubchain_mainnet", {"name": "DubChain"})

        asset = BridgeAsset(
            asset_id="DUB",
            name="DubChain Token",
            symbol="DUB",
            chain_id="dubchain_mainnet",
            contract_address="0x1234567890abcdef",
            asset_type=AssetType.ERC20,
            decimals=18,
        )

        manager.register_asset(asset)

        # Convert to dict and back
        data = manager.to_dict()
        restored_manager = BridgeManager.from_dict(data)

        assert restored_manager.config.bridge_type == manager.config.bridge_type
        assert restored_manager.status == manager.status
        assert len(restored_manager.chain_manager.supported_chains) == 1
        assert len(restored_manager.asset_manager.registered_assets) == 1
        assert restored_manager.created_at == manager.created_at
