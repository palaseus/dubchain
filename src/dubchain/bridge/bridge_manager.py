"""
Bridge management system for DubChain.

This module provides comprehensive bridge management including:
- Multi-chain bridge coordination
- Asset management across chains
- Bridge validator management
- Cross-chain transaction processing
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .bridge_types import (
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


@dataclass
class ChainManager:
    """Manages connected blockchain networks."""

    supported_chains: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    chain_connections: Dict[str, bool] = field(default_factory=dict)
    chain_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_chain(self, chain_id: str, chain_info: Dict[str, Any]) -> bool:
        """Add a new blockchain network."""
        if chain_id in self.supported_chains:
            return False

        self.supported_chains[chain_id] = chain_info
        self.chain_connections[chain_id] = True
        self.chain_metrics[chain_id] = {
            "transactions": 0,
            "volume": 0,
            "last_activity": time.time(),
        }
        return True

    def remove_chain(self, chain_id: str) -> bool:
        """Remove a blockchain network."""
        if chain_id not in self.supported_chains:
            return False

        del self.supported_chains[chain_id]
        del self.chain_connections[chain_id]
        del self.chain_metrics[chain_id]
        return True

    def update_chain_status(self, chain_id: str, is_connected: bool) -> None:
        """Update chain connection status."""
        if chain_id in self.chain_connections:
            self.chain_connections[chain_id] = is_connected

    def get_chain_info(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get chain information."""
        return self.supported_chains.get(chain_id)

    def get_connected_chains(self) -> List[str]:
        """Get list of connected chains."""
        return [
            chain_id
            for chain_id, connected in self.chain_connections.items()
            if connected
        ]

    def update_chain_metrics(
        self, chain_id: str, transactions: int = 0, volume: int = 0
    ) -> None:
        """Update chain metrics."""
        if chain_id in self.chain_metrics:
            self.chain_metrics[chain_id]["transactions"] += transactions
            self.chain_metrics[chain_id]["volume"] += volume
            self.chain_metrics[chain_id]["last_activity"] = time.time()


@dataclass
class AssetManager:
    """Manages assets across chains."""

    registered_assets: Dict[str, BridgeAsset] = field(default_factory=dict)
    asset_mappings: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # asset_id -> {chain_id: contract_address}
    asset_balances: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # asset_id -> {chain_id: balance}

    def register_asset(self, asset: BridgeAsset) -> bool:
        """Register a new asset."""
        if asset.asset_id in self.registered_assets:
            return False

        self.registered_assets[asset.asset_id] = asset
        self.asset_mappings[asset.asset_id] = {
            asset.chain_id: asset.contract_address or ""
        }
        self.asset_balances[asset.asset_id] = {asset.chain_id: 0}
        return True

    def add_asset_to_chain(
        self, asset_id: str, chain_id: str, contract_address: str
    ) -> bool:
        """Add asset to another chain."""
        if asset_id not in self.registered_assets:
            return False

        if asset_id not in self.asset_mappings:
            self.asset_mappings[asset_id] = {}

        self.asset_mappings[asset_id][chain_id] = contract_address

        if asset_id not in self.asset_balances:
            self.asset_balances[asset_id] = {}

        if chain_id not in self.asset_balances[asset_id]:
            self.asset_balances[asset_id][chain_id] = 0

        return True

    def get_asset_balance(self, asset_id: str, chain_id: str) -> int:
        """Get asset balance on a specific chain."""
        return self.asset_balances.get(asset_id, {}).get(chain_id, 0)

    def update_asset_balance(self, asset_id: str, chain_id: str, balance: int) -> None:
        """Update asset balance on a specific chain."""
        if asset_id not in self.asset_balances:
            self.asset_balances[asset_id] = {}
        self.asset_balances[asset_id][chain_id] = balance

    def get_asset_info(self, asset_id: str) -> Optional[BridgeAsset]:
        """Get asset information."""
        return self.registered_assets.get(asset_id)

    def get_chain_assets(self, chain_id: str) -> List[BridgeAsset]:
        """Get all assets on a specific chain."""
        assets = []
        for asset_id, asset in self.registered_assets.items():
            if chain_id in self.asset_mappings.get(asset_id, {}):
                assets.append(asset)
        return assets


class BridgeManager:
    """Main bridge management system."""

    def __init__(self, config: BridgeConfig):
        """Initialize bridge manager."""
        self.config = config
        self.status = BridgeStatus.ACTIVE
        self.chain_manager = ChainManager()
        self.asset_manager = AssetManager()
        self.validators: Dict[str, BridgeValidator] = {}
        self.transactions: Dict[str, CrossChainTransaction] = {}
        self.metrics = BridgeMetrics()
        self.created_at = time.time()

    def add_chain(self, chain_id: str, chain_info: Dict[str, Any]) -> bool:
        """Add a new blockchain network."""
        success = self.chain_manager.add_chain(chain_id, chain_info)
        if success:
            self.metrics.supported_chains = len(
                self.chain_manager.get_connected_chains()
            )
        return success

    def register_asset(self, asset: BridgeAsset) -> bool:
        """Register a new asset."""
        success = self.asset_manager.register_asset(asset)
        if success:
            self.metrics.supported_assets = len(self.asset_manager.registered_assets)
        return success

    def add_validator(self, validator: BridgeValidator) -> bool:
        """Add bridge validator."""
        if validator.validator_id in self.validators:
            return False

        self.validators[validator.validator_id] = validator
        self.metrics.active_validators = len(
            [v for v in self.validators.values() if v.is_active]
        )
        return True

    def remove_validator(self, validator_id: str) -> bool:
        """Remove bridge validator."""
        if validator_id not in self.validators:
            return False

        del self.validators[validator_id]
        self.metrics.active_validators = len(
            [v for v in self.validators.values() if v.is_active]
        )
        return True

    def create_cross_chain_transaction(
        self,
        source_chain: str,
        target_chain: str,
        source_asset: str,
        target_asset: str,
        sender: str,
        receiver: str,
        amount: int,
    ) -> Optional[CrossChainTransaction]:
        """Create a new cross-chain transaction."""
        # Validate chains
        if source_chain not in self.chain_manager.get_connected_chains():
            return None
        if target_chain not in self.chain_manager.get_connected_chains():
            return None

        # Validate assets
        if source_asset not in self.asset_manager.registered_assets:
            return None
        if target_asset not in self.asset_manager.registered_assets:
            return None

        # Validate amount
        if (
            amount < self.config.min_transfer_amount
            or amount > self.config.max_transfer_amount
        ):
            return None

        # Create transaction
        transaction_id = self._generate_transaction_id()
        transaction = CrossChainTransaction(
            transaction_id=transaction_id,
            source_chain=source_chain,
            target_chain=target_chain,
            source_asset=source_asset,
            target_asset=target_asset,
            sender=sender,
            receiver=receiver,
            amount=amount,
            bridge_type=self.config.bridge_type,
        )

        self.transactions[transaction_id] = transaction
        return transaction

    def process_transaction(self, transaction_id: str) -> bool:
        """Process a cross-chain transaction."""
        if transaction_id not in self.transactions:
            return False

        transaction = self.transactions[transaction_id]

        # Check if transaction is expired
        if transaction.is_expired(self.config.timeout_blocks):
            transaction.status = "failed"
            self.metrics.failed_transactions += 1
            return False

        # Validate transaction
        if not self._validate_transaction(transaction):
            transaction.status = "failed"
            self.metrics.failed_transactions += 1
            return False

        # Process based on bridge type
        if self.config.bridge_type == BridgeType.LOCK_AND_MINT:
            return self._process_lock_and_mint(transaction)
        elif self.config.bridge_type == BridgeType.BURN_AND_MINT:
            return self._process_burn_and_mint(transaction)
        elif self.config.bridge_type == BridgeType.ATOMIC_SWAP:
            return self._process_atomic_swap(transaction)
        else:
            return False

    def _validate_transaction(self, transaction: CrossChainTransaction) -> bool:
        """Validate cross-chain transaction."""
        # Check asset balances
        source_balance = self.asset_manager.get_asset_balance(
            transaction.source_asset, transaction.source_chain
        )
        if source_balance < transaction.amount:
            return False

        # Check if chains are connected
        if not self.chain_manager.chain_connections.get(
            transaction.source_chain, False
        ):
            return False
        if not self.chain_manager.chain_connections.get(
            transaction.target_chain, False
        ):
            return False

        return True

    def _process_lock_and_mint(self, transaction: CrossChainTransaction) -> bool:
        """Process lock and mint transaction."""
        # Lock assets on source chain
        source_balance = self.asset_manager.get_asset_balance(
            transaction.source_asset, transaction.source_chain
        )
        self.asset_manager.update_asset_balance(
            transaction.source_asset,
            transaction.source_chain,
            source_balance - transaction.amount,
        )

        # Mint assets on target chain
        target_balance = self.asset_manager.get_asset_balance(
            transaction.target_asset, transaction.target_chain
        )
        self.asset_manager.update_asset_balance(
            transaction.target_asset,
            transaction.target_chain,
            target_balance + transaction.amount,
        )

        # Update transaction status
        transaction.status = "completed"
        transaction.completed_at = time.time()
        transaction.completion_hash = transaction.calculate_hash()

        # Update metrics
        self.metrics.successful_transactions += 1
        self.metrics.total_volume += transaction.amount

        return True

    def _process_burn_and_mint(self, transaction: CrossChainTransaction) -> bool:
        """Process burn and mint transaction."""
        # Burn assets on source chain
        source_balance = self.asset_manager.get_asset_balance(
            transaction.source_asset, transaction.source_chain
        )
        self.asset_manager.update_asset_balance(
            transaction.source_asset,
            transaction.source_chain,
            source_balance - transaction.amount,
        )

        # Mint assets on target chain
        target_balance = self.asset_manager.get_asset_balance(
            transaction.target_asset, transaction.target_chain
        )
        self.asset_manager.update_asset_balance(
            transaction.target_asset,
            transaction.target_chain,
            target_balance + transaction.amount,
        )

        # Update transaction status
        transaction.status = "completed"
        transaction.completed_at = time.time()
        transaction.completion_hash = transaction.calculate_hash()

        # Update metrics
        self.metrics.successful_transactions += 1
        self.metrics.total_volume += transaction.amount

        return True

    def _process_atomic_swap(self, transaction: CrossChainTransaction) -> bool:
        """Process atomic swap transaction."""
        # This would involve more complex logic for atomic swaps
        # For now, we'll use a simplified version

        # Lock assets on both chains
        source_balance = self.asset_manager.get_asset_balance(
            transaction.source_asset, transaction.source_chain
        )
        self.asset_manager.update_asset_balance(
            transaction.source_asset,
            transaction.source_chain,
            source_balance - transaction.amount,
        )

        target_balance = self.asset_manager.get_asset_balance(
            transaction.target_asset, transaction.target_chain
        )
        self.asset_manager.update_asset_balance(
            transaction.target_asset,
            transaction.target_chain,
            target_balance + transaction.amount,
        )

        # Update transaction status
        transaction.status = "completed"
        transaction.completed_at = time.time()
        transaction.completion_hash = transaction.calculate_hash()

        # Update metrics
        self.metrics.successful_transactions += 1
        self.metrics.total_volume += transaction.amount

        return True

    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        timestamp = str(int(time.time() * 1000))
        random_data = str(hash(str(time.time())))[:8]
        return f"bridge_tx_{timestamp}_{random_data}"

    def get_transaction(self, transaction_id: str) -> Optional[CrossChainTransaction]:
        """Get transaction by ID."""
        return self.transactions.get(transaction_id)

    def get_transactions_by_status(self, status: str) -> List[CrossChainTransaction]:
        """Get transactions by status."""
        return [tx for tx in self.transactions.values() if tx.status == status]

    def get_bridge_metrics(self) -> BridgeMetrics:
        """Get bridge metrics."""
        self.metrics.total_transactions = len(self.transactions)
        self.metrics.supported_chains = len(self.chain_manager.get_connected_chains())
        self.metrics.supported_assets = len(self.asset_manager.registered_assets)
        self.metrics.active_validators = len(
            [v for v in self.validators.values() if v.is_active]
        )
        self.metrics.last_updated = time.time()
        return self.metrics

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get comprehensive bridge status."""
        return {
            "status": self.status.value,
            "bridge_type": self.config.bridge_type.value,
            "supported_chains": self.chain_manager.get_connected_chains(),
            "supported_assets": list(self.asset_manager.registered_assets.keys()),
            "active_validators": len(
                [v for v in self.validators.values() if v.is_active]
            ),
            "total_transactions": len(self.transactions),
            "pending_transactions": len(self.get_transactions_by_status("pending")),
            "completed_transactions": len(self.get_transactions_by_status("completed")),
            "failed_transactions": len(self.get_transactions_by_status("failed")),
            "metrics": self.get_bridge_metrics(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "chain_manager": {
                "supported_chains": self.chain_manager.supported_chains,
                "chain_connections": self.chain_manager.chain_connections,
                "chain_metrics": self.chain_manager.chain_metrics,
            },
            "asset_manager": {
                "registered_assets": {
                    k: v.to_dict()
                    for k, v in self.asset_manager.registered_assets.items()
                },
                "asset_mappings": self.asset_manager.asset_mappings,
                "asset_balances": self.asset_manager.asset_balances,
            },
            "validators": {k: v.to_dict() for k, v in self.validators.items()},
            "transactions": {k: v.to_dict() for k, v in self.transactions.items()},
            "metrics": {
                "total_transactions": self.metrics.total_transactions,
                "successful_transactions": self.metrics.successful_transactions,
                "failed_transactions": self.metrics.failed_transactions,
                "total_volume": self.metrics.total_volume,
                "average_transaction_time": self.metrics.average_transaction_time,
                "active_validators": self.metrics.active_validators,
                "supported_chains": self.metrics.supported_chains,
                "supported_assets": self.metrics.supported_assets,
                "last_updated": self.metrics.last_updated,
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeManager":
        """Create from dictionary."""
        config = BridgeConfig.from_dict(data["config"])
        manager = cls(config)

        # Restore status
        manager.status = BridgeStatus(data["status"])

        # Restore chain manager
        chain_data = data["chain_manager"]
        manager.chain_manager.supported_chains = chain_data["supported_chains"]
        manager.chain_manager.chain_connections = chain_data["chain_connections"]
        manager.chain_manager.chain_metrics = chain_data["chain_metrics"]

        # Restore asset manager
        asset_data = data["asset_manager"]
        manager.asset_manager.registered_assets = {
            k: BridgeAsset.from_dict(v)
            for k, v in asset_data["registered_assets"].items()
        }
        manager.asset_manager.asset_mappings = asset_data["asset_mappings"]
        manager.asset_manager.asset_balances = asset_data["asset_balances"]

        # Restore validators
        manager.validators = {
            k: BridgeValidator.from_dict(v) for k, v in data["validators"].items()
        }

        # Restore transactions
        manager.transactions = {
            k: CrossChainTransaction.from_dict(v)
            for k, v in data["transactions"].items()
        }

        # Restore metrics
        metrics_data = data["metrics"]
        manager.metrics = BridgeMetrics(
            total_transactions=metrics_data["total_transactions"],
            successful_transactions=metrics_data["successful_transactions"],
            failed_transactions=metrics_data["failed_transactions"],
            total_volume=metrics_data["total_volume"],
            average_transaction_time=metrics_data["average_transaction_time"],
            active_validators=metrics_data["active_validators"],
            supported_chains=metrics_data["supported_chains"],
            supported_assets=metrics_data["supported_assets"],
            last_updated=metrics_data["last_updated"],
        )

        manager.created_at = data["created_at"]

        return manager
