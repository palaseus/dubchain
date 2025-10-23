"""
Shard management system for DubChain.

This module provides comprehensive shard management including:
- Shard creation and allocation
- Validator assignment and rebalancing
- Shard coordination and monitoring
- Dynamic shard management
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..consensus.validator import Validator, ValidatorInfo, ValidatorSet
from .shard_types import (
    CrossShardTransaction,
    ShardConfig,
    ShardId,
    ShardMetrics,
    ShardState,
    ShardStatus,
    ShardType,
)


@dataclass
class ShardAllocator:
    """Allocates validators to shards."""

    allocation_strategy: str = "random"  # random, round_robin, weighted
    rebalance_threshold: float = 0.1
    last_rebalance: float = field(default_factory=time.time)

    def allocate_validators(
        self, validators: List[ValidatorInfo], shard_count: int
    ) -> Dict[ShardId, List[str]]:
        """Allocate validators to shards."""
        if self.allocation_strategy == "random":
            return self._random_allocation(validators, shard_count)
        elif self.allocation_strategy == "round_robin":
            return self._round_robin_allocation(validators, shard_count)
        elif self.allocation_strategy == "weighted":
            return self._weighted_allocation(validators, shard_count)
        else:
            return self._random_allocation(validators, shard_count)

    def _random_allocation(
        self, validators: List[ValidatorInfo], shard_count: int
    ) -> Dict[ShardId, List[str]]:
        """Random allocation of validators to shards."""
        allocation = {ShardId(i + 1): [] for i in range(shard_count)}

        # Shuffle validators for random distribution
        shuffled_validators = validators.copy()
        random.shuffle(shuffled_validators)

        # Distribute validators across shards
        for i, validator in enumerate(shuffled_validators):
            shard_id = ShardId((i % shard_count) + 1)
            allocation[shard_id].append(validator.validator_id)

        return allocation

    def _round_robin_allocation(
        self, validators: List[ValidatorInfo], shard_count: int
    ) -> Dict[ShardId, List[str]]:
        """Round-robin allocation of validators to shards."""
        allocation = {ShardId(i + 1): [] for i in range(shard_count)}

        for i, validator in enumerate(validators):
            shard_id = ShardId((i % shard_count) + 1)
            allocation[shard_id].append(validator.validator_id)

        return allocation

    def _weighted_allocation(
        self, validators: List[ValidatorInfo], shard_count: int
    ) -> Dict[ShardId, List[str]]:
        """Weighted allocation based on validator stake."""
        allocation = {ShardId(i + 1): [] for i in range(shard_count)}

        # Sort validators by stake (descending)
        sorted_validators = sorted(
            validators, key=lambda v: v.total_stake, reverse=True
        )

        # Distribute high-stake validators evenly across shards
        for i, validator in enumerate(sorted_validators):
            shard_id = ShardId((i % shard_count) + 1)
            allocation[shard_id].append(validator.validator_id)

        return allocation


@dataclass
class ShardBalancer:
    """Balances validators across shards."""

    balance_threshold: float = 0.1  # 10% imbalance threshold
    rebalance_interval: float = 3600.0  # 1 hour
    last_rebalance: float = field(default_factory=time.time)

    def should_rebalance(self, shard_states: Dict[ShardId, ShardState]) -> bool:
        """Check if shards need rebalancing."""
        if time.time() - self.last_rebalance < self.rebalance_interval:
            return False

        validator_counts = [len(state.validator_set) for state in shard_states.values()]
        if not validator_counts or len(validator_counts) < 2:
            return False

        min_count = min(validator_counts)
        max_count = max(validator_counts)

        if min_count == 0:
            return True

        imbalance = (max_count - min_count) / min_count
        return imbalance > self.balance_threshold

    def rebalance_shards(
        self,
        shard_states: Dict[ShardId, ShardState],
        all_validators: List[ValidatorInfo],
    ) -> Dict[ShardId, List[str]]:
        """Rebalance validators across shards."""
        # Collect all validators from all shards
        all_validator_ids = []
        for state in shard_states.values():
            all_validator_ids.extend(state.validator_set)

        # Create validator info objects
        validator_map = {v.validator_id: v for v in all_validators}
        validators = [
            validator_map[vid] for vid in all_validator_ids if vid in validator_map
        ]

        # Reallocate using round-robin for better balance
        allocator = ShardAllocator(allocation_strategy="round_robin")
        new_allocation = allocator.allocate_validators(validators, len(shard_states))

        self.last_rebalance = time.time()
        return new_allocation


@dataclass
class ShardCoordinator:
    """Coordinates operations across shards."""

    cross_shard_delay: int = 4  # epochs
    state_sync_interval: int = 32  # blocks
    last_state_sync: float = field(default_factory=time.time)
    coordination_events: List[Dict[str, Any]] = field(default_factory=list)

    def coordinate_cross_shard_transaction(
        self,
        transaction: CrossShardTransaction,
        source_shard: ShardState,
        target_shard: ShardState,
    ) -> bool:
        """Coordinate cross-shard transaction."""
        # Add transaction to source shard's queue
        source_shard.add_cross_shard_transaction(transaction)

        # Record coordination event
        event = {
            "type": "cross_shard_transaction",
            "transaction_id": transaction.transaction_id,
            "source_shard": transaction.source_shard.value,
            "target_shard": transaction.target_shard.value,
            "timestamp": time.time(),
        }
        self.coordination_events.append(event)

        return True

    def should_sync_state(self) -> bool:
        """Check if state sync is needed."""
        return time.time() - self.last_state_sync >= self.state_sync_interval

    def sync_shard_states(self, shard_states: Dict[ShardId, ShardState]) -> None:
        """Synchronize state across shards."""
        # Update cross-shard transaction queues
        for shard_id, state in shard_states.items():
            for target_shard_id, transactions in state.cross_shard_queues.items():
                if target_shard_id in shard_states:
                    target_state = shard_states[target_shard_id]
                    # Process cross-shard transactions
                    for transaction in transactions:
                        if transaction.status == "pending":
                            transaction.status = "confirmed"
                            transaction.confirmation_epoch = state.current_epoch

        self.last_state_sync = time.time()

        # Record sync event
        event = {
            "type": "state_sync",
            "timestamp": time.time(),
            "shard_count": len(shard_states),
        }
        self.coordination_events.append(event)


class ShardManager:
    """Manages all shards in the network."""

    def __init__(self, config: ShardConfig):
        """Initialize shard manager."""
        self.config = config
        self.shards: Dict[ShardId, ShardState] = {}
        self._shards: Dict[ShardId, ShardState] = {}  # Alias for compatibility
        self._validators: Dict[str, List[str]] = {}  # For compatibility
        self._running: bool = False
        self.allocator = ShardAllocator()
        self.balancer = ShardBalancer()
        self.coordinator = ShardCoordinator(
            cross_shard_delay=config.cross_shard_delay,
            state_sync_interval=config.state_sync_interval,
        )
        self.global_validator_set: List[ValidatorInfo] = []
        self.current_epoch = 0

    def create_shard(
        self,
        shard_type: ShardType = ShardType.EXECUTION,
        validators: Optional[List[str]] = None,
    ) -> ShardState:
        """Create a new shard."""
        # Find next available shard ID
        shard_id = None
        for i in range(1, self.config.max_shards + 1):
            candidate_id = ShardId(i)
            if candidate_id not in self.shards:
                shard_id = candidate_id
                break

        if shard_id is None:
            raise ValueError("Maximum number of shards reached")

        shard_state = ShardState(
            shard_id=shard_id,
            status=ShardStatus.ACTIVE,
            shard_type=shard_type,
            metrics=ShardMetrics(shard_id=shard_id),
        )

        if validators:
            shard_state.validator_set = validators.copy()
            shard_state.metrics.validator_count = len(validators)
            shard_state.metrics.active_validators = len(validators)

        self.shards[shard_id] = shard_state
        self._shards[shard_id] = shard_state  # Update alias
        return shard_state

    def add_validator_to_shard(self, shard_id: ShardId, validator_id: str) -> bool:
        """Add validator to a specific shard."""
        if shard_id not in self.shards:
            return False

        shard_state = self.shards[shard_id]
        if validator_id not in shard_state.validator_set:
            shard_state.validator_set.append(validator_id)
            shard_state.metrics.validator_count = len(shard_state.validator_set)
            shard_state.metrics.active_validators = len(shard_state.validator_set)
            return True

        return False

    def remove_validator_from_shard(self, shard_id: ShardId, validator_id: str) -> bool:
        """Remove validator from a specific shard."""
        if shard_id not in self.shards:
            return False

        shard_state = self.shards[shard_id]
        if validator_id in shard_state.validator_set:
            shard_state.validator_set.remove(validator_id)
            shard_state.metrics.validator_count = len(shard_state.validator_set)
            shard_state.metrics.active_validators = len(shard_state.validator_set)
            return True

        return False

    def allocate_validators_to_shards(self, validators: List[ValidatorInfo]) -> None:
        """Allocate validators to shards."""
        self.global_validator_set = validators

        if not self.shards:
            return

        # Allocate validators to existing shards
        allocation = self.allocator.allocate_validators(validators, len(self.shards))

        for shard_id, validator_ids in allocation.items():
            if shard_id in self.shards:
                self.shards[shard_id].validator_set = validator_ids
                self.shards[shard_id].metrics.validator_count = len(validator_ids)
                self.shards[shard_id].metrics.active_validators = len(validator_ids)

    def rebalance_shards(self) -> Dict[str, Any]:
        """Rebalance validators across shards."""
        if not self.balancer.should_rebalance(self.shards):
            return {"rebalanced_shards": [], "moved_validators": []}

        new_allocation = self.balancer.rebalance_shards(
            self.shards, self.global_validator_set
        )

        # Track moved validators
        moved_validators = []
        rebalanced_shards = []

        # Update shard validator sets
        for shard_id, validator_ids in new_allocation.items():
            if shard_id in self.shards:
                old_validators = set(self.shards[shard_id].validator_set)
                new_validators = set(validator_ids)

                # Track moved validators
                moved = old_validators.symmetric_difference(new_validators)
                if moved:
                    moved_validators.extend(list(moved))
                    rebalanced_shards.append(shard_id.value)

                self.shards[shard_id].validator_set = validator_ids
                self.shards[shard_id].metrics.validator_count = len(validator_ids)
                self.shards[shard_id].metrics.active_validators = len(validator_ids)

        return {
            "rebalanced_shards": rebalanced_shards,
            "moved_validators": moved_validators,
        }

    def process_cross_shard_transaction(
        self, transaction: CrossShardTransaction
    ) -> bool:
        """Process cross-shard transaction."""
        if (
            transaction.source_shard not in self.shards
            or transaction.target_shard not in self.shards
        ):
            return False

        source_shard = self.shards[transaction.source_shard]
        target_shard = self.shards[transaction.target_shard]

        return self.coordinator.coordinate_cross_shard_transaction(
            transaction, source_shard, target_shard
        )

    def sync_shard_states(self) -> None:
        """Synchronize state across all shards."""
        if self.coordinator.should_sync_state():
            self.coordinator.sync_shard_states(self.shards)

    def get_shard_by_validator(self, validator_id: str) -> Optional[ShardId]:
        """Get shard ID for a validator."""
        for shard_id, shard_state in self.shards.items():
            if validator_id in shard_state.validator_set:
                return shard_id
        return None

    def get_shard_metrics(self, shard_id: ShardId) -> Optional[ShardMetrics]:
        """Get metrics for a specific shard."""
        if shard_id in self.shards:
            return self.shards[shard_id].metrics
        return None

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global shard metrics."""
        total_blocks = sum(shard.metrics.total_blocks for shard in self.shards.values())
        total_validators = sum(
            shard.metrics.validator_count for shard in self.shards.values()
        )
        total_cross_shard_txs = sum(
            shard.metrics.cross_shard_transactions for shard in self.shards.values()
        )

        return {
            "total_shards": len(self.shards),
            "active_shards": len(
                [s for s in self.shards.values() if s.status == ShardStatus.ACTIVE]
            ),
            "total_blocks": total_blocks,
            "total_validators": total_validators,
            "total_cross_shard_transactions": total_cross_shard_txs,
            "current_epoch": self.current_epoch,
            "last_rebalance": self.balancer.last_rebalance,
            "last_state_sync": self.coordinator.last_state_sync,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "shards": {
                shard_id.value: shard_state.to_dict()
                for shard_id, shard_state in self.shards.items()
            },
            "global_validator_set": [v.to_dict() for v in self.global_validator_set],
            "current_epoch": self.current_epoch,
            "allocator": {
                "allocation_strategy": self.allocator.allocation_strategy,
                "rebalance_threshold": self.allocator.rebalance_threshold,
                "last_rebalance": self.allocator.last_rebalance,
            },
            "balancer": {
                "balance_threshold": self.balancer.balance_threshold,
                "rebalance_interval": self.balancer.rebalance_interval,
                "last_rebalance": self.balancer.last_rebalance,
            },
            "coordinator": {
                "cross_shard_delay": self.coordinator.cross_shard_delay,
                "state_sync_interval": self.coordinator.state_sync_interval,
                "last_state_sync": self.coordinator.last_state_sync,
                "coordination_events": self.coordinator.coordination_events,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardManager":
        """Create from dictionary."""
        config = ShardConfig.from_dict(data["config"])
        manager = cls(config)

        # Restore shards
        for shard_id_str, shard_data in data["shards"].items():
            shard_id = ShardId(int(shard_id_str))
            manager.shards[shard_id] = ShardState.from_dict(shard_data)

        # Restore global validator set
        manager.global_validator_set = [
            ValidatorInfo.from_dict(v_data) for v_data in data["global_validator_set"]
        ]
        manager.current_epoch = data["current_epoch"]

        # Restore components
        allocator_data = data["allocator"]
        manager.allocator = ShardAllocator(
            allocation_strategy=allocator_data["allocation_strategy"],
            rebalance_threshold=allocator_data["rebalance_threshold"],
        )
        manager.allocator.last_rebalance = allocator_data["last_rebalance"]

        balancer_data = data["balancer"]
        manager.balancer = ShardBalancer(
            balance_threshold=balancer_data["balance_threshold"],
            rebalance_interval=balancer_data["rebalance_interval"],
        )
        manager.balancer.last_rebalance = balancer_data["last_rebalance"]

        coordinator_data = data["coordinator"]
        manager.coordinator = ShardCoordinator(
            cross_shard_delay=coordinator_data["cross_shard_delay"],
            state_sync_interval=coordinator_data["state_sync_interval"],
        )
        manager.coordinator.last_state_sync = coordinator_data["last_state_sync"]
        manager.coordinator.coordination_events = coordinator_data[
            "coordination_events"
        ]

        return manager

    def get_shard(self, shard_id: ShardId) -> Optional[ShardState]:
        """Get shard by ID."""
        return self.shards.get(shard_id)

    def list_shards(self) -> List[ShardState]:
        """List all shards."""
        return list(self.shards.values())

    def add_validator(self, shard_id: ShardId, validator_id: str) -> bool:
        """Add validator to shard."""
        return self.add_validator_to_shard(shard_id, validator_id)

    def remove_validator(self, shard_id: ShardId, validator_id: str) -> bool:
        """Remove validator from shard."""
        return self.remove_validator_from_shard(shard_id, validator_id)

    def update_shard_status(self, shard_id: ShardId, status: ShardStatus) -> bool:
        """Update shard status."""
        if shard_id in self.shards:
            self.shards[shard_id].status = status
            return True
        return False

    def get_active_shards(self) -> List[ShardState]:
        """Get active shards."""
        return [
            shard
            for shard in self.shards.values()
            if shard.status == ShardStatus.ACTIVE
        ]

    def validate_shard(self, shard_id: ShardId) -> bool:
        """Validate shard."""
        if shard_id not in self.shards:
            return False

        shard = self.shards[shard_id]
        return (
            shard.shard_id is not None
            and shard.status is not None
            and shard.shard_type is not None
        )

    def start(self) -> bool:
        """Start shard manager."""
        self._running = True
        return True

    def stop(self) -> bool:
        """Stop shard manager."""
        self._running = False
        return True

    def get_shard_info(self, shard_id: ShardId) -> Optional[ShardState]:
        """Get shard information."""
        return self.shards.get(shard_id)

    def assign_validators_to_shard(
        self, shard_id: ShardId, validators: List[ValidatorInfo]
    ) -> bool:
        """Assign validators to a shard."""
        if shard_id not in self.shards:
            return False

        shard_state = self.shards[shard_id]
        validator_ids = [v.validator_id for v in validators]
        shard_state.validator_set = validator_ids
        shard_state.metrics.validator_count = len(validator_ids)
        shard_state.metrics.active_validators = len(validator_ids)
        return True

    def get_shard_validators(self, shard_id: ShardId) -> List[str]:
        """Get validators for a shard."""
        if shard_id in self.shards:
            return self.shards[shard_id].validator_set
        return []

    def get_all_shards(self) -> List[ShardState]:
        """Get all shards."""
        return list(self.shards.values())

    def remove_shard(self, shard_id: ShardId) -> bool:
        """Remove a shard."""
        if shard_id in self.shards:
            del self.shards[shard_id]
            if shard_id in self._shards:
                del self._shards[shard_id]
            return True
        return False

    def get_shard_statistics(self) -> Dict[str, Any]:
        """Get shard statistics."""
        total_shards = len(self.shards)
        active_shards = len(
            [s for s in self.shards.values() if s.status == ShardStatus.ACTIVE]
        )
        total_validators = sum(len(s.validator_set) for s in self.shards.values())

        avg_validators_per_shard = (
            total_validators / total_shards if total_shards > 0 else 0
        )

        return {
            "total_shards": total_shards,
            "active_shards": active_shards,
            "total_validators": total_validators,
            "average_validators_per_shard": avg_validators_per_shard,
        }

    @property
    def shard_states(self) -> Dict[ShardId, ShardState]:
        """Get shard states."""
        return self.shards
