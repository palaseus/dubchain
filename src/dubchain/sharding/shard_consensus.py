"""
Shard consensus implementation for DubChain.

This module provides consensus mechanisms for individual shards.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..consensus.validator import Validator, ValidatorInfo
from .shard_types import ShardId, ShardMetrics, ShardState


@dataclass
class ShardValidator:
    """Validator for a specific shard."""

    validator_id: str
    shard_id: ShardId
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)

    def update_heartbeat(self) -> None:
        """Update validator heartbeat."""
        self.last_heartbeat = time.time()

    def is_online(self, timeout: float = 30.0) -> bool:
        """Check if validator is online."""
        return time.time() - self.last_heartbeat < timeout


@dataclass
class ShardProposer:
    """Block proposer for a shard."""

    shard_id: ShardId
    current_proposer: Optional[str] = None
    proposer_rotation: List[str] = field(default_factory=list)
    current_index: int = 0

    def get_next_proposer(self) -> Optional[str]:
        """Get next proposer in rotation."""
        if not self.proposer_rotation:
            return None

        proposer = self.proposer_rotation[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.proposer_rotation)
        return proposer

    def update_rotation(self, validators: List[str]) -> None:
        """Update proposer rotation."""
        self.proposer_rotation = validators.copy()
        self.current_index = 0


@dataclass
class ShardCommittee:
    """Committee of validators for a shard."""

    shard_id: ShardId
    validators: List[ShardValidator] = field(default_factory=list)
    proposer: ShardProposer = field(
        default_factory=lambda: ShardProposer(ShardId.BEACON_CHAIN)
    )

    def add_validator(self, validator: ShardValidator) -> None:
        """Add validator to committee."""
        if validator not in self.validators:
            self.validators.append(validator)
            self.proposer.update_rotation([v.validator_id for v in self.validators])

    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator from committee."""
        for i, validator in enumerate(self.validators):
            if validator.validator_id == validator_id:
                del self.validators[i]
                self.proposer.update_rotation([v.validator_id for v in self.validators])
                return True
        return False

    def get_active_validators(self) -> List[ShardValidator]:
        """Get active validators."""
        return [v for v in self.validators if v.is_active and v.is_online()]


class ShardConsensus:
    """Consensus mechanism for individual shards."""

    def __init__(self, shard_id: ShardId):
        """Initialize shard consensus."""
        self.shard_id = shard_id
        self.committee = ShardCommittee(shard_id)
        self.metrics = ShardMetrics(shard_id=shard_id)
        self.current_epoch = 0

    def add_validator(self, validator_id: str) -> None:
        """Add validator to shard consensus."""
        shard_validator = ShardValidator(
            validator_id=validator_id, shard_id=self.shard_id
        )
        self.committee.add_validator(shard_validator)
        self.metrics.validator_count = len(self.committee.validators)
        self.metrics.active_validators = len(self.committee.get_active_validators())

    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator from shard consensus."""
        success = self.committee.remove_validator(validator_id)
        if success:
            self.metrics.validator_count = len(self.committee.validators)
            self.metrics.active_validators = len(self.committee.get_active_validators())
        return success

    def get_proposer(self) -> Optional[str]:
        """Get current proposer."""
        return self.committee.proposer.get_next_proposer()

    def update_metrics(self, success: bool, block_time: float, gas_used: int) -> None:
        """Update consensus metrics."""
        self.metrics.total_blocks += 1
        if success:
            self.metrics.successful_blocks += 1
        else:
            self.metrics.failed_blocks += 1

        # Update averages
        total_time = self.metrics.average_block_time * (self.metrics.total_blocks - 1)
        self.metrics.average_block_time = (
            total_time + block_time
        ) / self.metrics.total_blocks

        total_gas = self.metrics.average_gas_used * (self.metrics.total_blocks - 1)
        self.metrics.average_gas_used = (
            total_gas + gas_used
        ) / self.metrics.total_blocks

        self.metrics.last_updated = time.time()

    def get_metrics(self) -> ShardMetrics:
        """Get shard metrics."""
        return self.metrics
