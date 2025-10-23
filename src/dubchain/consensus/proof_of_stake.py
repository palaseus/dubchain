"""
Proof of Stake (PoS) consensus implementation for DubChain.

This module implements a sophisticated Proof of Stake consensus mechanism with:
- Validator selection based on stake
- Slashing for misbehavior
- Reward distribution
- Staking pools and delegation
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature
from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    StakingInfo,
)
from .validator import Validator, ValidatorInfo, ValidatorManager, ValidatorSet


@dataclass
class StakingPool:
    """Manages staking pool for a validator."""

    validator_id: str
    total_stake: int = 0
    delegators: Dict[str, int] = field(default_factory=dict)  # delegator_id -> amount
    rewards: int = 0
    slashing_penalty: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_reward_distribution: float = field(default_factory=time.time)

    def add_delegation(self, delegator_id: str, amount: int) -> None:
        """Add delegation to pool."""
        if delegator_id in self.delegators:
            self.delegators[delegator_id] += amount
        else:
            self.delegators[delegator_id] = amount
        self.total_stake += amount

    def remove_delegation(self, delegator_id: str, amount: int) -> bool:
        """Remove delegation from pool."""
        if delegator_id not in self.delegators:
            return False

        if self.delegators[delegator_id] < amount:
            return False

        self.delegators[delegator_id] -= amount
        self.total_stake -= amount

        if self.delegators[delegator_id] == 0:
            del self.delegators[delegator_id]

        return True

    def calculate_delegator_rewards(
        self, total_rewards: int, commission_rate: float
    ) -> Dict[str, int]:
        """Calculate rewards for each delegator."""
        if self.total_stake == 0:
            return {}

        # Calculate validator commission
        validator_commission = int(total_rewards * commission_rate)
        delegator_rewards_total = total_rewards - validator_commission

        rewards = {}
        for delegator_id, stake in self.delegators.items():
            delegator_share = stake / self.total_stake
            rewards[delegator_id] = int(delegator_share * delegator_rewards_total)

        return rewards


@dataclass
class RewardCalculator:
    """Calculates rewards for validators and delegators."""

    base_reward_rate: float = 0.1  # 10% annual
    inflation_rate: float = 0.02  # 2% annual inflation
    block_time: float = 2.0  # seconds per block
    total_supply: int = 1000000000  # 1 billion tokens

    def calculate_block_reward(self, block_number: int) -> int:
        """Calculate reward for a block."""
        # Simple linear reward calculation
        # In practice, this would be more sophisticated
        annual_blocks = 365 * 24 * 3600 / self.block_time
        annual_reward = int(self.total_supply * self.base_reward_rate)
        block_reward = annual_reward / annual_blocks

        # Apply inflation
        inflation_factor = 1 + (self.inflation_rate * block_number / annual_blocks)
        return int(block_reward * inflation_factor)

    def calculate_validator_reward(
        self, validator_stake: int, total_stake: int, block_reward: int
    ) -> int:
        """Calculate reward for a specific validator."""
        if total_stake == 0:
            return 0

        stake_ratio = validator_stake / total_stake
        return int(block_reward * stake_ratio)


class ProofOfStake:
    """Proof of Stake consensus implementation."""

    def __init__(self, config: ConsensusConfig):
        """Initialize Proof of Stake consensus."""
        self.config = config
        self.validator_set = ValidatorSet(max_validators=config.max_validators)
        self.validator_manager = ValidatorManager(self.validator_set)
        self.staking_pools: Dict[str, StakingPool] = {}
        self.reward_calculator = RewardCalculator(
            base_reward_rate=config.reward_rate, block_time=config.block_time
        )
        self.metrics = ConsensusMetrics(consensus_type=ConsensusType.PROOF_OF_STAKE)
        self.current_epoch = 0
        self.epoch_start_time = time.time()
        self.last_block_time = time.time()

    def register_validator(self, validator: Validator, initial_stake: int = 0) -> bool:
        """Register a new validator."""
        try:
            if initial_stake < self.config.min_stake:
                return False

            success = self.validator_manager.register_validator(validator, initial_stake)
            if success:
                # Create staking pool
                self.staking_pools[validator.validator_id] = StakingPool(
                    validator_id=validator.validator_id, total_stake=initial_stake
                )

            return success
        except Exception as e:
            logger.error(f"Error registering validator: {e}")
            return False

    def add_validator(self, address: str, stake_amount: int) -> bool:
        """Add a validator to the system."""
        try:
            if address in self.validator_set.validators:
                return False
            if stake_amount < self.config.min_stake:
                return False
            
            # Create a validator
            validator = Validator(
                validator_id=address,
                address=address,
                public_key=None,  # Will be set later
                voting_power=stake_amount,
                is_active=True
            )
            
            # Register the validator
            success = self.register_validator(validator, stake_amount)
            return success
        except Exception as e:
            logger.error(f"Error adding validator: {e}")
            return False

    def stake_to_validator(
        self, validator_id: str, delegator_id: str, amount: int
    ) -> bool:
        """Stake tokens to a validator."""
        if validator_id not in self.staking_pools:
            return False

        success = self.validator_manager.stake(validator_id, delegator_id, amount)
        if success:
            self.staking_pools[validator_id].add_delegation(delegator_id, amount)

        return success

    def unstake_from_validator(
        self, validator_id: str, delegator_id: str, amount: int
    ) -> bool:
        """Unstake tokens from a validator."""
        if validator_id not in self.staking_pools:
            return False

        success = self.validator_manager.unstake(validator_id, delegator_id, amount)
        if success:
            self.staking_pools[validator_id].remove_delegation(delegator_id, amount)

        return success

    def select_proposer(self, block_number: int) -> Optional[str]:
        """Select proposer for next block based on stake."""
        try:
            active_validators = list(self.validator_set.active_validators)
            if not active_validators:
                return None

            # Calculate total voting power
            total_power = sum(
                self.validator_set.validators[vid].voting_power for vid in active_validators
            )

            if total_power == 0:
                return None

            # Weighted random selection based on stake
            weights = [
                self.validator_set.validators[vid].voting_power / total_power
                for vid in active_validators
            ]

            # Use block number as seed for deterministic selection
            random.seed(block_number)
            selected_index = random.choices(range(len(active_validators)), weights=weights)[
                0
            ]
            random.seed()  # Reset seed

            return active_validators[selected_index]
        except Exception as e:
            logger.error(f"Error selecting proposer: {e}")
            return None

    def validate_block_proposal(
        self, proposer_id: str, block_data: Dict[str, Any]
    ) -> bool:
        """Validate a block proposal."""
        if proposer_id not in self.validator_set.active_validators:
            return False

        # Check if proposer has minimum stake
        proposer = self.validator_set.validators[proposer_id]
        if proposer.total_stake < self.config.min_stake:
            return False

        # Validate block structure and content
        required_fields = ["block_number", "timestamp", "transactions", "previous_hash"]
        for field in required_fields:
            if field not in block_data:
                return False

        # Check timestamp is reasonable
        current_time = time.time()
        block_timestamp = block_data["timestamp"]
        if abs(current_time - block_timestamp) > 300:  # 5 minutes tolerance
            return False

        return True

    def finalize_block(
        self, block_data: Dict[str, Any], proposer_id: str
    ) -> ConsensusResult:
        """Finalize a block through consensus."""
        start_time = time.time()

        # Validate block proposal
        if not self.validate_block_proposal(proposer_id, block_data):
            return ConsensusResult(
                success=False,
                error_message="Invalid block proposal",
                consensus_type=ConsensusType.PROOF_OF_STAKE,
            )

        # Simulate block finalization (in practice, this would involve more validators)
        block_hash = self._calculate_block_hash(block_data)

        # Update metrics
        block_time = time.time() - start_time
        self.metrics.total_blocks += 1
        self.metrics.successful_blocks += 1
        self.metrics.last_updated = time.time()

        # Update proposer activity
        if proposer_id in self.validator_set.validators:
            self.validator_set.validators[proposer_id].last_active = time.time()

        # Distribute rewards
        self._distribute_block_rewards(block_data.get("block_number", 0))

        return ConsensusResult(
            success=True,
            block_hash=block_hash,
            validator_id=proposer_id,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            timestamp=time.time(),
            gas_used=block_data.get("gas_used", 0),
        )

    def slash_validator(
        self, validator_id: str, reason: str, evidence: Optional[Dict[str, Any]] = None
    ) -> int:
        """Slash a validator for misbehavior."""
        if validator_id not in self.validator_set.validators:
            return 0

        # Calculate slashing amount
        validator = self.validator_set.validators[validator_id]
        slashed_amount = self.validator_manager.slash_validator(
            validator_id, self.config.slashing_percentage, reason
        )

        # Update staking pool
        if validator_id in self.staking_pools:
            self.staking_pools[
                validator_id
            ].slashing_penalty += self.config.slashing_percentage

        return slashed_amount

    def _calculate_block_hash(self, block_data: Dict[str, Any]) -> str:
        """Calculate hash of block data."""
        # Create deterministic block hash
        block_string = json.dumps(block_data, sort_keys=True)
        return SHA256Hasher.hash(block_string.encode()).to_hex()

    def _distribute_block_rewards(self, block_number: int) -> None:
        """Distribute rewards for a block."""
        block_reward = self.reward_calculator.calculate_block_reward(block_number)

        if block_reward == 0:
            return

        # Distribute rewards to validators
        validator_rewards = self.validator_manager.distribute_rewards(block_reward)

        # Update staking pools
        for validator_id, reward in validator_rewards.items():
            if validator_id in self.staking_pools:
                self.staking_pools[validator_id].rewards += reward
                self.staking_pools[validator_id].last_reward_distribution = time.time()

    def get_validator_info(self, validator_id: str) -> Optional[ValidatorInfo]:
        """Get information about a validator."""
        return self.validator_set.validators.get(validator_id)

    def get_staking_pool_info(self, validator_id: str) -> Optional[StakingPool]:
        """Get staking pool information."""
        return self.staking_pools.get(validator_id)

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        self.metrics.validator_count = len(self.validator_set.validators)
        self.metrics.active_validators = len(self.validator_set.active_validators)
        return self.metrics

    def get_top_validators(self, limit: int = 10) -> List[ValidatorInfo]:
        """Get top validators by stake."""
        return self.validator_set.get_validator_by_power(limit)

    def is_validator_active(self, validator_id: str) -> bool:
        """Check if validator is active."""
        return validator_id in self.validator_set.active_validators

    def get_total_stake(self) -> int:
        """Get total stake in the network."""
        return sum(pool.total_stake for pool in self.staking_pools.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "validator_set": self.validator_set.to_dict(),
            "staking_pools": {
                k: {
                    "validator_id": v.validator_id,
                    "total_stake": v.total_stake,
                    "delegators": v.delegators,
                    "rewards": v.rewards,
                    "slashing_penalty": v.slashing_penalty,
                    "created_at": v.created_at,
                    "last_reward_distribution": v.last_reward_distribution,
                }
                for k, v in self.staking_pools.items()
            },
            "metrics": {
                "total_blocks": self.metrics.total_blocks,
                "successful_blocks": self.metrics.successful_blocks,
                "failed_blocks": self.metrics.failed_blocks,
                "average_block_time": self.metrics.average_block_time,
                "average_gas_used": self.metrics.average_gas_used,
                "validator_count": self.metrics.validator_count,
                "active_validators": self.metrics.active_validators,
                "consensus_type": self.metrics.consensus_type.value,
                "last_updated": self.metrics.last_updated,
            },
            "current_epoch": self.current_epoch,
            "epoch_start_time": self.epoch_start_time,
            "last_block_time": self.last_block_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofOfStake":
        """Create from dictionary."""
        config = ConsensusConfig.from_dict(data["config"])
        pos = cls(config)

        # Restore validator set
        pos.validator_set = ValidatorSet.from_dict(data["validator_set"])
        pos.validator_manager = ValidatorManager(pos.validator_set)

        # Restore staking pools
        for validator_id, pool_data in data["staking_pools"].items():
            pos.staking_pools[validator_id] = StakingPool(
                validator_id=pool_data["validator_id"],
                total_stake=pool_data["total_stake"],
                delegators=pool_data["delegators"],
                rewards=pool_data["rewards"],
                slashing_penalty=pool_data["slashing_penalty"],
                created_at=pool_data["created_at"],
                last_reward_distribution=pool_data["last_reward_distribution"],
            )

        # Restore metrics
        metrics_data = data["metrics"]
        pos.metrics = ConsensusMetrics(
            total_blocks=metrics_data["total_blocks"],
            successful_blocks=metrics_data["successful_blocks"],
            failed_blocks=metrics_data["failed_blocks"],
            average_block_time=metrics_data["average_block_time"],
            average_gas_used=metrics_data["average_gas_used"],
            validator_count=metrics_data["validator_count"],
            active_validators=metrics_data["active_validators"],
            consensus_type=ConsensusType(metrics_data["consensus_type"]),
            last_updated=metrics_data["last_updated"],
        )

        pos.current_epoch = data["current_epoch"]
        pos.epoch_start_time = data["epoch_start_time"]
        pos.last_block_time = data["last_block_time"]

        return pos
