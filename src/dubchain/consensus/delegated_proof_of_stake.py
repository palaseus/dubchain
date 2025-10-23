"""
Delegated Proof of Stake (DPoS) consensus implementation for DubChain.

This module implements DPoS consensus with:
- Delegate election and voting
- Block production by elected delegates
- Voting power management
- Delegate rotation and rewards
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
    DelegateInfo,
    VotingPower,
)
from .validator import Validator, ValidatorInfo, ValidatorSet


@dataclass
class ElectionManager:
    """Manages delegate elections in DPoS."""

    election_interval: int = 86400  # 24 hours
    delegate_count: int = 21  # Number of delegates
    voting_period: int = 3600  # 1 hour voting period
    last_election: float = field(default_factory=time.time)
    current_delegates: List[str] = field(default_factory=list)
    next_election_time: float = field(default_factory=lambda: time.time() + 86400)

    def is_election_time(self) -> bool:
        """Check if it's time for an election."""
        return time.time() >= self.next_election_time

    def schedule_next_election(self) -> None:
        """Schedule the next election."""
        self.next_election_time = time.time() + self.election_interval

    def get_voting_deadline(self) -> float:
        """Get voting deadline for current election."""
        return self.next_election_time - self.voting_period


class DelegatedProofOfStake:
    """Delegated Proof of Stake consensus implementation."""

    def __init__(self, config: ConsensusConfig):
        """Initialize DPoS consensus."""
        self.config = config
        self.validators: Dict[str, ValidatorInfo] = {}
        self.delegates: Dict[str, ValidatorInfo] = {}
        self.voting_power: Dict[str, VotingPower] = {}
        self.delegations: Dict[str, List[DelegateInfo]] = {}  # voter -> delegates
        self.election_manager = ElectionManager()
        self.metrics = ConsensusMetrics(
            consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE
        )

        # Block production
        self.current_producer_index = 0
        self.block_production_schedule: List[str] = []
        self.last_block_time = time.time()
        self.block_interval = config.block_time

        # Rewards
        self.delegate_rewards: Dict[str, int] = {}
        self.voter_rewards: Dict[str, int] = {}

    def register_delegate(self, validator: Validator, initial_stake: int = 0) -> bool:
        """Register a validator as a delegate candidate."""
        if len(self.validators) >= self.config.max_validators:
            return False

        validator_info = ValidatorInfo(
            validator_id=validator.validator_id,
            public_key=validator.public_key,
            total_stake=initial_stake,
            self_stake=initial_stake,
        )

        self.validators[validator.validator_id] = validator_info
        self.voting_power[validator.validator_id] = VotingPower(
            validator_id=validator.validator_id,
            total_power=initial_stake,
            self_stake=initial_stake,
            delegated_stake=0,
        )

        return True

    def vote_for_delegate(self, voter_id: str, delegate_id: str, amount: int) -> bool:
        """Vote for a delegate with voting power."""
        if delegate_id not in self.validators:
            return False

        if voter_id not in self.delegations:
            self.delegations[voter_id] = []

        # Check if already voted for this delegate
        for delegation in self.delegations[voter_id]:
            if delegation.delegate_id == delegate_id:
                delegation.amount += amount
                break
        else:
            # New delegation
            delegation = DelegateInfo(
                delegate_id=delegate_id, voter_id=voter_id, amount=amount
            )
            self.delegations[voter_id].append(delegation)

        # Update voting power
        if delegate_id in self.voting_power:
            self.voting_power[delegate_id].delegated_stake += amount
            self.voting_power[delegate_id].total_power += amount

        return True

    def unvote_delegate(self, voter_id: str, delegate_id: str, amount: int) -> bool:
        """Remove votes from a delegate."""
        if voter_id not in self.delegations:
            return False

        for delegation in self.delegations[voter_id]:
            if delegation.delegate_id == delegate_id:
                if delegation.amount >= amount:
                    delegation.amount -= amount
                    if delegation.amount == 0:
                        self.delegations[voter_id].remove(delegation)

                    # Update voting power
                    if delegate_id in self.voting_power:
                        self.voting_power[delegate_id].delegated_stake -= amount
                        self.voting_power[delegate_id].total_power -= amount

                    return True

        return False

    def conduct_election(self) -> List[str]:
        """Conduct delegate election."""
        if not self.validators:
            return []

        # Calculate total voting power for each delegate
        delegate_scores = []
        for validator_id, validator in self.validators.items():
            if validator_id in self.voting_power:
                voting_power = self.voting_power[validator_id]
                score = voting_power.total_power
                delegate_scores.append((validator_id, score))

        # Sort by voting power (descending)
        delegate_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top delegates
        elected_delegates = [
            delegate_id
            for delegate_id, _ in delegate_scores[
                : self.election_manager.delegate_count
            ]
        ]

        # Update delegates
        self.delegates.clear()
        for delegate_id in elected_delegates:
            if delegate_id in self.validators:
                self.delegates[delegate_id] = self.validators[delegate_id]

        # Update election manager
        self.election_manager.current_delegates = elected_delegates
        self.election_manager.last_election = time.time()
        self.election_manager.schedule_next_election()

        # Create block production schedule
        self._create_production_schedule()

        return elected_delegates

    def _create_production_schedule(self) -> None:
        """Create block production schedule for delegates."""
        self.block_production_schedule = self.election_manager.current_delegates.copy()
        random.shuffle(self.block_production_schedule)  # Randomize order
        self.current_producer_index = 0

    def get_current_producer(self) -> Optional[str]:
        """Get current block producer."""
        if not self.block_production_schedule:
            return None

        return self.block_production_schedule[self.current_producer_index]

    def advance_producer(self) -> None:
        """Advance to next block producer."""
        if not self.block_production_schedule:
            return

        self.current_producer_index = (self.current_producer_index + 1) % len(
            self.block_production_schedule
        )
        self.last_block_time = time.time()

    def is_production_time(self) -> bool:
        """Check if it's time to produce a block."""
        return time.time() - self.last_block_time >= self.block_interval

    def produce_block(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """Produce a block through DPoS consensus."""
        current_producer = self.get_current_producer()
        if not current_producer:
            return ConsensusResult(
                success=False,
                error_message="No active producer",
                consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            )

        # Check if it's the producer's turn
        if not self.is_production_time():
            return ConsensusResult(
                success=False,
                error_message="Not production time",
                consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            )

        # Validate block
        if not self._validate_block(block_data, current_producer):
            return ConsensusResult(
                success=False,
                error_message="Invalid block",
                consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            )

        # Create block hash
        block_hash = self._calculate_block_hash(block_data)

        # Update metrics
        self.metrics.total_blocks += 1
        self.metrics.successful_blocks += 1
        self.metrics.last_updated = time.time()

        # Advance to next producer
        self.advance_producer()

        # Check if election is needed
        if self.election_manager.is_election_time():
            self.conduct_election()

        return ConsensusResult(
            success=True,
            block_hash=block_hash,
            validator_id=current_producer,
            consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            timestamp=time.time(),
            gas_used=block_data.get("gas_used", 0),
        )

    def _validate_block(self, block_data: Dict[str, Any], producer_id: str) -> bool:
        """Validate block produced by delegate."""
        # Check required fields
        required_fields = ["block_number", "timestamp", "transactions", "previous_hash"]
        for field in required_fields:
            if field not in block_data:
                return False

        # Check timestamp
        current_time = time.time()
        block_timestamp = block_data["timestamp"]
        if abs(current_time - block_timestamp) > 300:  # 5 minutes tolerance
            return False

        # Check if producer is valid delegate
        if producer_id not in self.delegates:
            return False

        return True

    def _calculate_block_hash(self, block_data: Dict[str, Any]) -> str:
        """Calculate hash of block data."""
        block_string = json.dumps(block_data, sort_keys=True)
        return SHA256Hasher.hash(block_string.encode()).to_hex()

    def distribute_rewards(self, total_rewards: int) -> Dict[str, int]:
        """Distribute rewards to delegates and voters."""
        if not self.delegates:
            return {}

        # Distribute rewards to delegates
        delegate_rewards = {}
        for delegate_id in self.delegates:
            # Each delegate gets equal share
            delegate_reward = total_rewards // len(self.delegates)
            delegate_rewards[delegate_id] = delegate_reward
            self.delegate_rewards[delegate_id] = (
                delegate_rewards.get(delegate_id, 0) + delegate_reward
            )

        # Distribute rewards to voters based on their stake
        voter_rewards = {}
        for voter_id, delegations in self.delegations.items():
            total_voter_stake = sum(d.amount for d in delegations)
            if total_voter_stake == 0:
                continue

            voter_reward = 0
            for delegation in delegations:
                if delegation.delegate_id in delegate_rewards:
                    # Voter gets proportional share of delegate's reward
                    delegate_reward = delegate_rewards[delegation.delegate_id]
                    voter_share = (
                        delegation.amount
                        / self.voting_power[delegation.delegate_id].total_power
                    )
                    voter_reward += int(delegate_reward * voter_share)

            if voter_reward > 0:
                voter_rewards[voter_id] = voter_reward
                self.voter_rewards[voter_id] = (
                    self.voter_rewards.get(voter_id, 0) + voter_reward
                )

        return {**delegate_rewards, **voter_rewards}

    def get_delegate_rankings(self) -> List[Tuple[str, int]]:
        """Get delegate rankings by voting power."""
        rankings = []
        for validator_id, validator in self.validators.items():
            if validator_id in self.voting_power:
                voting_power = self.voting_power[validator_id]
                rankings.append((validator_id, voting_power.total_power))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_voting_statistics(self) -> Dict[str, Any]:
        """Get voting statistics."""
        total_voters = len(self.delegations)
        total_delegates = len(self.validators)
        active_delegates = len(self.delegates)

        # Calculate total voting power
        total_voting_power = sum(vp.total_power for vp in self.voting_power.values())

        return {
            "total_voters": total_voters,
            "total_delegates": total_delegates,
            "active_delegates": active_delegates,
            "total_voting_power": total_voting_power,
            "current_producer": self.get_current_producer(),
            "next_election": self.election_manager.next_election_time,
            "election_interval": self.election_manager.election_interval,
            "block_interval": self.block_interval,
        }

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        self.metrics.validator_count = len(self.validators)
        self.metrics.active_validators = len(self.delegates)
        return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "validators": {k: v.to_dict() for k, v in self.validators.items()},
            "delegates": {k: v.to_dict() for k, v in self.delegates.items()},
            "voting_power": {
                k: {
                    "validator_id": v.validator_id,
                    "total_power": v.total_power,
                    "self_stake": v.self_stake,
                    "delegated_stake": v.delegated_stake,
                    "voting_weight": v.voting_weight,
                    "last_updated": v.last_updated,
                }
                for k, v in self.voting_power.items()
            },
            "delegations": {
                k: [d.__dict__ for d in v] for k, v in self.delegations.items()
            },
            "election_manager": {
                "election_interval": self.election_manager.election_interval,
                "delegate_count": self.election_manager.delegate_count,
                "voting_period": self.election_manager.voting_period,
                "last_election": self.election_manager.last_election,
                "current_delegates": self.election_manager.current_delegates,
                "next_election_time": self.election_manager.next_election_time,
            },
            "current_producer_index": self.current_producer_index,
            "block_production_schedule": self.block_production_schedule,
            "last_block_time": self.last_block_time,
            "block_interval": self.block_interval,
            "delegate_rewards": self.delegate_rewards,
            "voter_rewards": self.voter_rewards,
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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegatedProofOfStake":
        """Create from dictionary."""
        config = ConsensusConfig.from_dict(data["config"])
        dpos = cls(config)

        # Restore validators
        for validator_data in data["validators"].values():
            validator_info = ValidatorInfo.from_dict(validator_data)
            dpos.validators[validator_info.validator_id] = validator_info

        # Restore delegates
        for delegate_data in data["delegates"].values():
            delegate_info = ValidatorInfo.from_dict(delegate_data)
            dpos.delegates[delegate_info.validator_id] = delegate_info

        # Restore voting power
        for validator_id, vp_data in data["voting_power"].items():
            dpos.voting_power[validator_id] = VotingPower(
                validator_id=vp_data["validator_id"],
                total_power=vp_data["total_power"],
                self_stake=vp_data["self_stake"],
                delegated_stake=vp_data["delegated_stake"],
                voting_weight=vp_data["voting_weight"],
                last_updated=vp_data["last_updated"],
            )

        # Restore delegations
        for voter_id, delegation_list in data["delegations"].items():
            dpos.delegations[voter_id] = [DelegateInfo(**d) for d in delegation_list]

        # Restore election manager
        em_data = data["election_manager"]
        dpos.election_manager = ElectionManager(
            election_interval=em_data["election_interval"],
            delegate_count=em_data["delegate_count"],
            voting_period=em_data["voting_period"],
        )
        dpos.election_manager.last_election = em_data["last_election"]
        dpos.election_manager.current_delegates = em_data["current_delegates"]
        dpos.election_manager.next_election_time = em_data["next_election_time"]

        # Restore other state
        dpos.current_producer_index = data["current_producer_index"]
        dpos.block_production_schedule = data["block_production_schedule"]
        dpos.last_block_time = data["last_block_time"]
        dpos.block_interval = data["block_interval"]
        dpos.delegate_rewards = data["delegate_rewards"]
        dpos.voter_rewards = data["voter_rewards"]

        # Restore metrics
        metrics_data = data["metrics"]
        dpos.metrics = ConsensusMetrics(
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

        return dpos
