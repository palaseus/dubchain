"""
Validator management system for DubChain consensus.

This module provides comprehensive validator management including:
- Validator registration and management
- Staking and delegation
- Slashing and rewards
- Validator set management
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..crypto.signatures import PrivateKey, PublicKey, Signature
from .consensus_types import (
    ConsensusMetrics,
    StakingInfo,
    ValidatorRole,
    ValidatorStatus,
    VotingPower,
)


@dataclass
class ValidatorInfo:
    """Information about a validator."""

    validator_id: str
    public_key: PublicKey
    status: ValidatorStatus = ValidatorStatus.INACTIVE
    role: ValidatorRole = ValidatorRole.VALIDATOR
    total_stake: int = 0
    self_stake: int = 0
    delegated_stake: int = 0
    voting_power: int = 0
    commission_rate: float = 0.1  # 10% commission
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    slashing_count: int = 0
    total_rewards: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_voting_power(self) -> None:
        """Update voting power based on stake."""
        self.voting_power = self.total_stake
        self.last_active = time.time()

    def add_stake(self, amount: int, is_self_stake: bool = False) -> None:
        """Add stake to validator."""
        self.total_stake += amount
        if is_self_stake:
            self.self_stake += amount
        else:
            self.delegated_stake += amount
        self.update_voting_power()

    def remove_stake(self, amount: int, is_self_stake: bool = False) -> None:
        """Remove stake from validator."""
        self.total_stake = max(0, self.total_stake - amount)
        if is_self_stake:
            self.self_stake = max(0, self.self_stake - amount)
        else:
            self.delegated_stake = max(0, self.delegated_stake - amount)
        self.update_voting_power()

    def slash(self, percentage: float) -> int:
        """Slash validator stake."""
        slashed_amount = int(self.total_stake * percentage)
        self.total_stake = max(0, self.total_stake - slashed_amount)
        self.slashing_count += 1
        self.update_voting_power()
        return slashed_amount

    def add_rewards(self, amount: int) -> None:
        """Add rewards to validator."""
        self.total_rewards += amount

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validator_id": self.validator_id,
            "public_key": self.public_key.to_hex(),
            "status": self.status.value,
            "role": self.role.value,
            "total_stake": self.total_stake,
            "self_stake": self.self_stake,
            "delegated_stake": self.delegated_stake,
            "voting_power": self.voting_power,
            "commission_rate": self.commission_rate,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "slashing_count": self.slashing_count,
            "total_rewards": self.total_rewards,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidatorInfo":
        """Create from dictionary."""
        return cls(
            validator_id=data["validator_id"],
            public_key=PublicKey.from_hex(data["public_key"]),
            status=ValidatorStatus(data["status"]),
            role=ValidatorRole(data["role"]),
            total_stake=data["total_stake"],
            self_stake=data["self_stake"],
            delegated_stake=data["delegated_stake"],
            voting_power=data["voting_power"],
            commission_rate=data["commission_rate"],
            created_at=data["created_at"],
            last_active=data["last_active"],
            slashing_count=data["slashing_count"],
            total_rewards=data["total_rewards"],
            metadata=data.get("metadata", {}),
        )


class Validator:
    """Individual validator implementation."""

    def __init__(
        self, validator_id: str, private_key: PrivateKey, commission_rate: float = 0.1
    ):
        """Initialize validator."""
        self.validator_id = validator_id
        self.private_key = private_key
        self.public_key = private_key.get_public_key()
        self.commission_rate = commission_rate
        self.info = ValidatorInfo(
            validator_id=validator_id,
            public_key=self.public_key,
            commission_rate=commission_rate,
        )
        self.is_active = False
        self.last_heartbeat = time.time()

    def activate(self) -> None:
        """Activate validator."""
        self.is_active = True
        self.info.status = ValidatorStatus.ACTIVE
        self.last_heartbeat = time.time()

    def deactivate(self) -> None:
        """Deactivate validator."""
        self.is_active = False
        self.info.status = ValidatorStatus.INACTIVE

    def jail(self) -> None:
        """Jail validator for misbehavior."""
        self.is_active = False
        self.info.status = ValidatorStatus.JAILED

    def unjail(self) -> None:
        """Unjail validator."""
        self.info.status = ValidatorStatus.ACTIVE
        self.is_active = True

    def sign_message(self, message: bytes) -> Signature:
        """Sign a message."""
        return self.private_key.sign(message)

    def verify_signature(self, message: bytes, signature: Signature) -> bool:
        """Verify a signature."""
        return self.public_key.verify(signature, message)

    def update_heartbeat(self) -> None:
        """Update validator heartbeat."""
        self.last_heartbeat = time.time()
        self.info.last_active = time.time()

    def is_online(self, timeout: float = 30.0) -> bool:
        """Check if validator is online."""
        return time.time() - self.last_heartbeat < timeout


class ValidatorSet:
    """Manages the set of active validators."""

    def __init__(self, max_validators: int = 100):
        """Initialize validator set."""
        self.max_validators = max_validators
        self.validators: Dict[str, ValidatorInfo] = {}
        self.active_validators: Set[str] = set()
        self.proposer_rotation: List[str] = []
        self.current_proposer_index = 0

    def add_validator(self, validator_info: ValidatorInfo) -> bool:
        """Add validator to set."""
        if len(self.validators) >= self.max_validators:
            return False

        self.validators[validator_info.validator_id] = validator_info
        if validator_info.status == ValidatorStatus.ACTIVE:
            self.active_validators.add(validator_info.validator_id)
            self._update_proposer_rotation()

        return True

    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator from set."""
        if validator_id not in self.validators:
            return False

        del self.validators[validator_id]
        self.active_validators.discard(validator_id)
        self._update_proposer_rotation()
        return True

    def update_validator_status(
        self, validator_id: str, status: ValidatorStatus
    ) -> bool:
        """Update validator status."""
        if validator_id not in self.validators:
            return False

        validator = self.validators[validator_id]
        validator.status = status

        if status == ValidatorStatus.ACTIVE:
            self.active_validators.add(validator_id)
        else:
            self.active_validators.discard(validator_id)

        self._update_proposer_rotation()
        return True

    def get_next_proposer(self) -> Optional[str]:
        """Get next proposer in rotation."""
        if not self.proposer_rotation:
            return None

        proposer = self.proposer_rotation[self.current_proposer_index]
        self.current_proposer_index = (self.current_proposer_index + 1) % len(
            self.proposer_rotation
        )
        return proposer

    def get_validator_by_power(self, limit: int = None) -> List[ValidatorInfo]:
        """Get validators sorted by voting power."""
        validators = list(self.validators.values())
        validators.sort(key=lambda v: v.voting_power, reverse=True)

        if limit:
            validators = validators[:limit]

        return validators

    def get_total_voting_power(self) -> int:
        """Get total voting power of all validators."""
        return sum(v.voting_power for v in self.validators.values())

    def _update_proposer_rotation(self) -> None:
        """Update proposer rotation based on voting power."""
        active_validators = [
            v
            for v in self.validators.values()
            if v.validator_id in self.active_validators
        ]
        active_validators.sort(key=lambda v: v.voting_power, reverse=True)

        self.proposer_rotation = [v.validator_id for v in active_validators]
        self.current_proposer_index = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_validators": self.max_validators,
            "validators": {k: v.to_dict() for k, v in self.validators.items()},
            "active_validators": list(self.active_validators),
            "proposer_rotation": self.proposer_rotation,
            "current_proposer_index": self.current_proposer_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidatorSet":
        """Create from dictionary."""
        validator_set = cls(max_validators=data["max_validators"])

        for validator_data in data["validators"].values():
            validator_info = ValidatorInfo.from_dict(validator_data)
            validator_set.validators[validator_info.validator_id] = validator_info

        validator_set.active_validators = set(data["active_validators"])
        validator_set.proposer_rotation = data["proposer_rotation"]
        validator_set.current_proposer_index = data["current_proposer_index"]

        return validator_set


class ValidatorManager:
    """Manages validators and their operations."""

    def __init__(self, validator_set: ValidatorSet):
        """Initialize validator manager."""
        self.validator_set = validator_set
        self.staking_pools: Dict[str, List[StakingInfo]] = {}
        self.slashing_events: List[Dict[str, Any]] = []
        self.reward_pool = 0

    def register_validator(self, validator: Validator, initial_stake: int = 0) -> bool:
        """Register a new validator."""
        validator_info = validator.info
        validator_info.add_stake(initial_stake, is_self_stake=True)

        success = self.validator_set.add_validator(validator_info)
        if success:
            validator.activate()
            # Update validator set status after activation
            self.validator_set.update_validator_status(
                validator.validator_id, ValidatorStatus.ACTIVE
            )
            self.staking_pools[validator.validator_id] = []

        return success

    def stake(self, validator_id: str, delegator_id: str, amount: int) -> bool:
        """Stake tokens to a validator."""
        if validator_id not in self.validator_set.validators:
            return False

        validator = self.validator_set.validators[validator_id]
        validator.add_stake(amount, is_self_stake=False)

        staking_info = StakingInfo(
            validator_id=validator_id, delegator_id=delegator_id, amount=amount
        )

        if validator_id not in self.staking_pools:
            self.staking_pools[validator_id] = []

        self.staking_pools[validator_id].append(staking_info)
        return True

    def unstake(self, validator_id: str, delegator_id: str, amount: int) -> bool:
        """Unstake tokens from a validator."""
        if validator_id not in self.staking_pools:
            return False

        # Find and remove staking info
        for i, staking_info in enumerate(self.staking_pools[validator_id]):
            if (
                staking_info.validator_id == validator_id
                and staking_info.delegator_id == delegator_id
            ):
                if staking_info.amount >= amount:
                    staking_info.amount -= amount
                    if staking_info.amount == 0:
                        del self.staking_pools[validator_id][i]

                    validator = self.validator_set.validators[validator_id]
                    validator.remove_stake(amount, is_self_stake=False)
                    return True

        return False

    def slash_validator(self, validator_id: str, percentage: float, reason: str) -> int:
        """Slash a validator for misbehavior."""
        if validator_id not in self.validator_set.validators:
            return 0

        validator = self.validator_set.validators[validator_id]
        slashed_amount = validator.slash(percentage)

        # Record slashing event
        slashing_event = {
            "validator_id": validator_id,
            "percentage": percentage,
            "amount": slashed_amount,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.slashing_events.append(slashing_event)

        # Jail validator if slashed too much
        if validator.slashing_count >= 3:
            validator.status = ValidatorStatus.JAILED
            self.validator_set.update_validator_status(
                validator_id, ValidatorStatus.JAILED
            )

        return slashed_amount

    def distribute_rewards(self, total_rewards: int) -> Dict[str, int]:
        """Distribute rewards to validators."""
        if not self.validator_set.active_validators:
            return {}

        total_power = self.validator_set.get_total_voting_power()
        if total_power == 0:
            return {}

        rewards = {}
        for validator_id in self.validator_set.active_validators:
            validator = self.validator_set.validators[validator_id]
            validator_reward = int(
                (validator.voting_power / total_power) * total_rewards
            )

            # Apply commission
            commission = int(validator_reward * validator.commission_rate)
            delegator_reward = validator_reward - commission

            validator.add_rewards(commission)
            rewards[validator_id] = delegator_reward

        return rewards

    def get_validator_metrics(self) -> ConsensusMetrics:
        """Get validator metrics."""
        active_count = len(self.validator_set.active_validators)
        total_count = len(self.validator_set.validators)

        return ConsensusMetrics(
            validator_count=total_count,
            active_validators=active_count,
            last_updated=time.time(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validator_set": self.validator_set.to_dict(),
            "staking_pools": {
                k: [
                    {
                        "validator_id": staking.validator_id,
                        "delegator_id": staking.delegator_id,
                        "amount": staking.amount,
                        "timestamp": staking.timestamp,
                        "unbonding_time": staking.unbonding_time,
                        "reward_rate": staking.reward_rate,
                        "slashing_penalty": staking.slashing_penalty,
                    }
                    for staking in v
                ]
                for k, v in self.staking_pools.items()
            },
            "slashing_events": self.slashing_events,
            "reward_pool": self.reward_pool,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidatorManager":
        """Create from dictionary."""
        validator_set = ValidatorSet.from_dict(data["validator_set"])
        manager = cls(validator_set)

        # Restore staking pools
        for validator_id, staking_list in data["staking_pools"].items():
            manager.staking_pools[validator_id] = [
                StakingInfo(**staking) for staking in staking_list
            ]

        manager.slashing_events = data["slashing_events"]
        manager.reward_pool = data["reward_pool"]

        return manager
