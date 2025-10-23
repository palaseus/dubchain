"""
Sharding types and data structures for DubChain.

This module defines the core types used in the sharding system.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Union


class ShardId(IntEnum):
    """Shard identifier."""

    BEACON_CHAIN = 0
    SHARD_1 = 1
    SHARD_2 = 2
    SHARD_3 = 3
    SHARD_4 = 4
    SHARD_5 = 5
    SHARD_6 = 6
    SHARD_7 = 7
    SHARD_8 = 8
    SHARD_9 = 9
    SHARD_10 = 10
    SHARD_11 = 11
    SHARD_12 = 12
    SHARD_13 = 13
    SHARD_14 = 14
    SHARD_15 = 15
    SHARD_16 = 16
    SHARD_17 = 17
    SHARD_18 = 18
    SHARD_19 = 19
    SHARD_20 = 20
    SHARD_21 = 21
    SHARD_22 = 22
    SHARD_23 = 23
    SHARD_24 = 24
    SHARD_25 = 25
    SHARD_26 = 26
    SHARD_27 = 27
    SHARD_28 = 28
    SHARD_29 = 29
    SHARD_30 = 30
    SHARD_31 = 31
    SHARD_32 = 32


class ShardStatus(Enum):
    """Shard status states."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCING = "syncing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ShardType(Enum):
    """Types of shards."""

    BEACON = "beacon"
    EXECUTION = "execution"
    CONSENSUS = "consensus"
    STORAGE = "storage"


@dataclass
class ShardConfig:
    """Configuration for shard management."""

    max_shards: int = 64
    min_validators_per_shard: int = 64
    max_validators_per_shard: int = 256
    shard_epoch_length: int = 64  # blocks
    cross_shard_delay: int = 4  # epochs
    state_sync_interval: int = 32  # blocks
    rebalance_threshold: float = 0.1  # 10% imbalance triggers rebalancing
    enable_dynamic_sharding: bool = True
    shard_consensus_type: str = "proof_of_stake"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_shards": self.max_shards,
            "min_validators_per_shard": self.min_validators_per_shard,
            "max_validators_per_shard": self.max_validators_per_shard,
            "shard_epoch_length": self.shard_epoch_length,
            "cross_shard_delay": self.cross_shard_delay,
            "state_sync_interval": self.state_sync_interval,
            "rebalance_threshold": self.rebalance_threshold,
            "enable_dynamic_sharding": self.enable_dynamic_sharding,
            "shard_consensus_type": self.shard_consensus_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardConfig":
        """Create from dictionary."""
        return cls(
            max_shards=data.get("max_shards", 64),
            min_validators_per_shard=data.get("min_validators_per_shard", 64),
            max_validators_per_shard=data.get("max_validators_per_shard", 256),
            shard_epoch_length=data.get("shard_epoch_length", 64),
            cross_shard_delay=data.get("cross_shard_delay", 4),
            state_sync_interval=data.get("state_sync_interval", 32),
            rebalance_threshold=data.get("rebalance_threshold", 0.1),
            enable_dynamic_sharding=data.get("enable_dynamic_sharding", True),
            shard_consensus_type=data.get("shard_consensus_type", "proof_of_stake"),
        )


@dataclass
class ShardMetrics:
    """Metrics for shard performance."""

    shard_id: ShardId
    total_blocks: int = 0
    successful_blocks: int = 0
    failed_blocks: int = 0
    average_block_time: float = 0.0
    average_gas_used: float = 0.0
    validator_count: int = 0
    active_validators: int = 0
    cross_shard_transactions: int = 0
    state_sync_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_blocks == 0:
            return 0.0
        return self.successful_blocks / self.total_blocks

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

    def get_success_rate(self) -> float:
        """Get success rate."""
        return self.success_rate

    def get_validator_utilization(self) -> float:
        """Get validator utilization rate."""
        if self.validator_count == 0:
            return 0.0
        return self.active_validators / self.validator_count


@dataclass
class CrossShardTransaction:
    """Cross-shard transaction data."""

    transaction_id: str
    source_shard: ShardId
    target_shard: ShardId
    sender: str
    receiver: str
    amount: int
    gas_limit: int
    gas_price: int
    data: bytes
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending, confirmed, failed
    confirmation_epoch: Optional[int] = None
    cross_shard_proof: Optional[str] = None

    def calculate_hash(self) -> str:
        """Calculate transaction hash."""
        data_string = f"{self.transaction_id}{self.source_shard}{self.target_shard}{self.sender}{self.receiver}{self.amount}{self.timestamp}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "source_shard": self.source_shard.value,
            "target_shard": self.target_shard.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "gas_limit": self.gas_limit,
            "gas_price": self.gas_price,
            "data": self.data.hex() if isinstance(self.data, bytes) else self.data,
            "timestamp": self.timestamp,
            "status": self.status,
            "confirmation_epoch": self.confirmation_epoch,
            "cross_shard_proof": self.cross_shard_proof,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossShardTransaction":
        """Create from dictionary."""
        return cls(
            transaction_id=data["transaction_id"],
            source_shard=ShardId(data["source_shard"]),
            target_shard=ShardId(data["target_shard"]),
            sender=data["sender"],
            receiver=data["receiver"],
            amount=data["amount"],
            gas_limit=data["gas_limit"],
            gas_price=data["gas_price"],
            data=bytes.fromhex(data["data"])
            if isinstance(data["data"], str)
            else data["data"],
            timestamp=data["timestamp"],
            status=data["status"],
            confirmation_epoch=data.get("confirmation_epoch"),
            cross_shard_proof=data.get("cross_shard_proof"),
        )


@dataclass
class ShardState:
    """State of a shard."""

    shard_id: ShardId
    status: ShardStatus
    shard_type: ShardType
    validator_set: List[str] = field(default_factory=list)
    proposer: Optional[str] = None
    current_epoch: int = 0
    last_block_number: int = 0
    last_block_hash: str = ""
    state_root: str = ""
    cross_shard_queues: Dict[ShardId, List[CrossShardTransaction]] = field(
        default_factory=dict
    )
    metrics: ShardMetrics = field(
        default_factory=lambda: ShardMetrics(ShardId.BEACON_CHAIN)
    )
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def update_metrics(self, success: bool, block_time: float, gas_used: int) -> None:
        """Update shard metrics."""
        self.metrics.total_blocks += 1
        if success:
            self.metrics.successful_blocks += 1
        else:
            self.metrics.failed_blocks += 1

        # Update average block time
        total_time = self.metrics.average_block_time * (self.metrics.total_blocks - 1)
        self.metrics.average_block_time = (
            total_time + block_time
        ) / self.metrics.total_blocks

        # Update average gas used
        total_gas = self.metrics.average_gas_used * (self.metrics.total_blocks - 1)
        self.metrics.average_gas_used = (
            total_gas + gas_used
        ) / self.metrics.total_blocks

        self.metrics.last_updated = time.time()
        self.last_updated = time.time()

    def add_cross_shard_transaction(self, transaction: CrossShardTransaction) -> None:
        """Add cross-shard transaction to queue."""
        if transaction.target_shard not in self.cross_shard_queues:
            self.cross_shard_queues[transaction.target_shard] = []

        self.cross_shard_queues[transaction.target_shard].append(transaction)
        self.metrics.cross_shard_transactions += 1

    def get_cross_shard_transactions(
        self, target_shard: ShardId
    ) -> List[CrossShardTransaction]:
        """Get cross-shard transactions for target shard."""
        return self.cross_shard_queues.get(target_shard, [])

    def clear_cross_shard_transactions(self, target_shard: ShardId) -> None:
        """Clear cross-shard transactions for target shard."""
        if target_shard in self.cross_shard_queues:
            del self.cross_shard_queues[target_shard]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shard_id": self.shard_id.value,
            "status": self.status.value,
            "shard_type": self.shard_type.value,
            "validator_set": self.validator_set,
            "proposer": self.proposer,
            "current_epoch": self.current_epoch,
            "last_block_number": self.last_block_number,
            "last_block_hash": self.last_block_hash,
            "state_root": self.state_root,
            "cross_shard_queues": {
                shard_id.value: [tx.to_dict() for tx in transactions]
                for shard_id, transactions in self.cross_shard_queues.items()
            },
            "metrics": {
                "shard_id": self.metrics.shard_id.value,
                "total_blocks": self.metrics.total_blocks,
                "successful_blocks": self.metrics.successful_blocks,
                "failed_blocks": self.metrics.failed_blocks,
                "average_block_time": self.metrics.average_block_time,
                "average_gas_used": self.metrics.average_gas_used,
                "validator_count": self.metrics.validator_count,
                "active_validators": self.metrics.active_validators,
                "cross_shard_transactions": self.metrics.cross_shard_transactions,
                "state_sync_count": self.metrics.state_sync_count,
                "last_updated": self.metrics.last_updated,
            },
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardState":
        """Create from dictionary."""
        shard_state = cls(
            shard_id=ShardId(data["shard_id"]),
            status=ShardStatus(data["status"]),
            shard_type=ShardType(data["shard_type"]),
            validator_set=data["validator_set"],
            proposer=data.get("proposer"),
            current_epoch=data["current_epoch"],
            last_block_number=data["last_block_number"],
            last_block_hash=data["last_block_hash"],
            state_root=data["state_root"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
        )

        # Restore cross-shard queues
        for shard_id_str, transactions_data in data["cross_shard_queues"].items():
            shard_id = ShardId(int(shard_id_str))
            shard_state.cross_shard_queues[shard_id] = [
                CrossShardTransaction.from_dict(tx_data)
                for tx_data in transactions_data
            ]

        # Restore metrics
        metrics_data = data["metrics"]
        shard_state.metrics = ShardMetrics(
            shard_id=ShardId(metrics_data["shard_id"]),
            total_blocks=metrics_data["total_blocks"],
            successful_blocks=metrics_data["successful_blocks"],
            failed_blocks=metrics_data["failed_blocks"],
            average_block_time=metrics_data["average_block_time"],
            average_gas_used=metrics_data["average_gas_used"],
            validator_count=metrics_data["validator_count"],
            active_validators=metrics_data["active_validators"],
            cross_shard_transactions=metrics_data["cross_shard_transactions"],
            state_sync_count=metrics_data["state_sync_count"],
            last_updated=metrics_data["last_updated"],
        )

        return shard_state
