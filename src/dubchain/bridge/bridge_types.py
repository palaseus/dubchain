"""
Cross-chain bridge types and data structures for DubChain.

This module defines the core types used in the cross-chain bridge system.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Union


class BridgeType(Enum):
    """Types of cross-chain bridges."""

    LOCK_AND_MINT = "lock_and_mint"
    BURN_AND_MINT = "burn_and_mint"
    ATOMIC_SWAP = "atomic_swap"
    RELAY_CHAIN = "relay_chain"
    SIDECHAIN = "sidechain"


class BridgeStatus(Enum):
    """Bridge status states."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"
    ERROR = "error"


class ChainType(Enum):
    """Types of blockchain networks."""

    MAINNET = "mainnet"
    TESTNET = "testnet"
    SIDECHAIN = "sidechain"
    LAYER2 = "layer2"
    PRIVATE = "private"


class AssetType(Enum):
    """Types of assets."""

    NATIVE = "native"
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    CUSTOM = "custom"


@dataclass
class BridgeConfig:
    """Configuration for cross-chain bridge."""

    bridge_type: BridgeType = BridgeType.LOCK_AND_MINT
    supported_chains: List[str] = field(default_factory=list)
    supported_assets: List[str] = field(default_factory=list)
    min_transfer_amount: int = 1
    max_transfer_amount: int = 1000000000
    transfer_fee_percentage: float = 0.001  # 0.1%
    confirmation_blocks: int = 12
    timeout_blocks: int = 100
    security_threshold: float = 0.67  # 67% of validators must confirm
    enable_atomic_swaps: bool = True
    enable_cross_chain_messaging: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bridge_type": self.bridge_type.value,
            "supported_chains": self.supported_chains,
            "supported_assets": self.supported_assets,
            "min_transfer_amount": self.min_transfer_amount,
            "max_transfer_amount": self.max_transfer_amount,
            "transfer_fee_percentage": self.transfer_fee_percentage,
            "confirmation_blocks": self.confirmation_blocks,
            "timeout_blocks": self.timeout_blocks,
            "security_threshold": self.security_threshold,
            "enable_atomic_swaps": self.enable_atomic_swaps,
            "enable_cross_chain_messaging": self.enable_cross_chain_messaging,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeConfig":
        """Create from dictionary."""
        return cls(
            bridge_type=BridgeType(data.get("bridge_type", "lock_and_mint")),
            supported_chains=data.get("supported_chains", []),
            supported_assets=data.get("supported_assets", []),
            min_transfer_amount=data.get("min_transfer_amount", 1),
            max_transfer_amount=data.get("max_transfer_amount", 1000000000),
            transfer_fee_percentage=data.get("transfer_fee_percentage", 0.001),
            confirmation_blocks=data.get("confirmation_blocks", 12),
            timeout_blocks=data.get("timeout_blocks", 100),
            security_threshold=data.get("security_threshold", 0.67),
            enable_atomic_swaps=data.get("enable_atomic_swaps", True),
            enable_cross_chain_messaging=data.get("enable_cross_chain_messaging", True),
        )


@dataclass
class BridgeMetrics:
    """Metrics for bridge performance."""

    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    total_volume: int = 0
    average_transaction_time: float = 0.0
    active_validators: int = 0
    supported_chains: int = 0
    supported_assets: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_transactions == 0:
            return 0.0
        return self.successful_transactions / self.total_transactions

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


@dataclass
class CrossChainTransaction:
    """Cross-chain transaction data."""

    transaction_id: str
    source_chain: str
    target_chain: str
    source_asset: str
    target_asset: str
    sender: str
    receiver: str
    amount: int
    bridge_type: BridgeType
    status: str = "pending"  # pending, confirmed, completed, failed
    created_at: float = field(default_factory=time.time)
    confirmed_at: Optional[float] = None
    completed_at: Optional[float] = None
    gas_used: int = 0
    fees: int = 0
    confirmation_hash: Optional[str] = None
    completion_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_hash(self) -> str:
        """Calculate transaction hash."""
        data_string = f"{self.transaction_id}{self.source_chain}{self.target_chain}{self.amount}{self.created_at}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def is_expired(self, timeout_blocks: int = 100) -> bool:
        """Check if transaction is expired."""
        return (
            time.time() - self.created_at > timeout_blocks * 2
        )  # Assuming 2 seconds per block

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "source_asset": self.source_asset,
            "target_asset": self.target_asset,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "bridge_type": self.bridge_type.value,
            "status": self.status,
            "created_at": self.created_at,
            "confirmed_at": self.confirmed_at,
            "completed_at": self.completed_at,
            "gas_used": self.gas_used,
            "fees": self.fees,
            "confirmation_hash": self.confirmation_hash,
            "completion_hash": self.completion_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossChainTransaction":
        """Create from dictionary."""
        return cls(
            transaction_id=data["transaction_id"],
            source_chain=data["source_chain"],
            target_chain=data["target_chain"],
            source_asset=data["source_asset"],
            target_asset=data["target_asset"],
            sender=data["sender"],
            receiver=data["receiver"],
            amount=data["amount"],
            bridge_type=BridgeType(data["bridge_type"]),
            status=data["status"],
            created_at=data["created_at"],
            confirmed_at=data.get("confirmed_at"),
            completed_at=data.get("completed_at"),
            gas_used=data.get("gas_used", 0),
            fees=data.get("fees", 0),
            confirmation_hash=data.get("confirmation_hash"),
            completion_hash=data.get("completion_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BridgeAsset:
    """Asset information for bridge."""

    asset_id: str
    asset_type: AssetType
    chain_id: str
    contract_address: Optional[str] = None
    symbol: str = ""
    name: str = ""
    decimals: int = 18
    total_supply: int = 0
    bridge_address: Optional[str] = None
    is_bridged: bool = False
    original_chain: Optional[str] = None
    original_asset: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "chain_id": self.chain_id,
            "contract_address": self.contract_address,
            "symbol": self.symbol,
            "name": self.name,
            "decimals": self.decimals,
            "total_supply": self.total_supply,
            "bridge_address": self.bridge_address,
            "is_bridged": self.is_bridged,
            "original_chain": self.original_chain,
            "original_asset": self.original_asset,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeAsset":
        """Create from dictionary."""
        return cls(
            asset_id=data["asset_id"],
            asset_type=AssetType(data["asset_type"]),
            chain_id=data["chain_id"],
            contract_address=data.get("contract_address"),
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            decimals=data.get("decimals", 18),
            total_supply=data.get("total_supply", 0),
            bridge_address=data.get("bridge_address"),
            is_bridged=data.get("is_bridged", False),
            original_chain=data.get("original_chain"),
            original_asset=data.get("original_asset"),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BridgeValidator:
    """Bridge validator information."""

    validator_id: str
    chain_id: str
    public_key: str
    stake_amount: int = 0
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    validation_count: int = 0
    slashing_count: int = 0
    rewards_earned: int = 0
    created_at: float = field(default_factory=time.time)

    def update_heartbeat(self) -> None:
        """Update validator heartbeat."""
        self.last_heartbeat = time.time()

    def is_online(self, timeout: float = 60.0) -> bool:
        """Check if validator is online."""
        return time.time() - self.last_heartbeat < timeout

    def add_validation(self) -> None:
        """Add validation count."""
        self.validation_count += 1

    def add_slashing(self, amount: int) -> None:
        """Add slashing penalty."""
        self.slashing_count += 1
        self.stake_amount = max(0, self.stake_amount - amount)

    def add_rewards(self, amount: int) -> None:
        """Add rewards."""
        self.rewards_earned += amount

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validator_id": self.validator_id,
            "chain_id": self.chain_id,
            "public_key": self.public_key,
            "stake_amount": self.stake_amount,
            "is_active": self.is_active,
            "last_heartbeat": self.last_heartbeat,
            "validation_count": self.validation_count,
            "slashing_count": self.slashing_count,
            "rewards_earned": self.rewards_earned,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeValidator":
        """Create from dictionary."""
        return cls(
            validator_id=data["validator_id"],
            chain_id=data["chain_id"],
            public_key=data["public_key"],
            stake_amount=data.get("stake_amount", 0),
            is_active=data.get("is_active", True),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            validation_count=data.get("validation_count", 0),
            slashing_count=data.get("slashing_count", 0),
            rewards_earned=data.get("rewards_earned", 0),
            created_at=data.get("created_at", time.time()),
        )
