"""
Shard state management for DubChain.

This module provides state management for shards including:
- State synchronization
- State validation
- State snapshots
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .shard_types import ShardId, ShardState


@dataclass
class StateSnapshot:
    """Snapshot of shard state."""

    shard_id: ShardId
    state_root: str
    block_number: int
    timestamp: float = field(default_factory=time.time)
    validator_set: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_hash(self) -> str:
        """Calculate snapshot hash."""
        data_string = (
            f"{self.shard_id.value}{self.state_root}{self.block_number}{self.timestamp}"
        )
        return hashlib.sha256(data_string.encode()).hexdigest()


@dataclass
class StateValidator:
    """Validates shard state."""

    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def validate_state(self, shard_state: ShardState) -> bool:
        """Validate shard state."""
        # Basic validation rules
        if not shard_state.shard_id:
            return False

        if not shard_state.validator_set:
            return False

        return True


class StateSync:
    """Synchronizes state across shards."""

    def __init__(self):
        """Initialize state sync."""
        self.sync_interval = 32  # blocks
        self.last_sync = time.time()
        self.sync_metrics = {"syncs_performed": 0, "sync_failures": 0}

    def should_sync(self) -> bool:
        """Check if sync is needed."""
        return time.time() - self.last_sync >= self.sync_interval

    def sync_states(self, shard_states: Dict[ShardId, ShardState]) -> bool:
        """Synchronize states across shards."""
        try:
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

            self.last_sync = time.time()
            self.sync_metrics["syncs_performed"] += 1
            return True
        except Exception:
            self.sync_metrics["sync_failures"] += 1
            return False


class ShardStateManager:
    """Manages state for all shards."""

    def __init__(self):
        """Initialize shard state manager."""
        self.shard_states: Dict[ShardId, ShardState] = {}
        self.state_validator = StateValidator()
        self.state_sync = StateSync()
        self.snapshots: List[StateSnapshot] = []

    def add_shard_state(self, shard_state: ShardState) -> None:
        """Add shard state."""
        self.shard_states[shard_state.shard_id] = shard_state

    def get_shard_state(self, shard_id: ShardId) -> Optional[ShardState]:
        """Get shard state."""
        return self.shard_states.get(shard_id)

    def validate_all_states(self) -> Dict[ShardId, bool]:
        """Validate all shard states."""
        results = {}
        for shard_id, state in self.shard_states.items():
            results[shard_id] = self.state_validator.validate_state(state)
        return results

    def sync_all_states(self) -> bool:
        """Synchronize all shard states."""
        if self.state_sync.should_sync():
            return self.state_sync.sync_states(self.shard_states)
        return True

    def create_snapshot(self, shard_id: ShardId) -> Optional[StateSnapshot]:
        """Create state snapshot."""
        if shard_id not in self.shard_states:
            return None

        state = self.shard_states[shard_id]
        snapshot = StateSnapshot(
            shard_id=shard_id,
            state_root=state.state_root,
            block_number=state.last_block_number,
            validator_set=state.validator_set.copy(),
        )

        self.snapshots.append(snapshot)
        return snapshot

    def get_metrics(self) -> Dict[str, Any]:
        """Get state management metrics."""
        return {
            "total_shards": len(self.shard_states),
            "total_snapshots": len(self.snapshots),
            "sync_metrics": self.state_sync.sync_metrics,
            "last_sync": self.state_sync.last_sync,
        }
