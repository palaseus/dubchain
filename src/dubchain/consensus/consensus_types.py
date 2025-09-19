"""
Consensus types and enums for DubChain.

This module defines the core types used across all consensus mechanisms.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union


class ConsensusType(Enum):
    """Types of consensus mechanisms."""

    PROOF_OF_WORK = "proof_of_work"
    PROOF_OF_STAKE = "proof_of_stake"
    DELEGATED_PROOF_OF_STAKE = "delegated_proof_of_stake"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "pbft"
    HYBRID = "hybrid"


class ValidatorStatus(Enum):
    """Validator status states."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    JAILED = "jailed"
    UNBONDING = "unbonding"
    BONDED = "bonded"
    SLASHED = "slashed"


class ValidatorRole(Enum):
    """Validator roles in consensus."""

    PROPOSER = "proposer"
    VALIDATOR = "validator"
    OBSERVER = "observer"
    DELEGATOR = "delegator"


class PBFTPhase(Enum):
    """PBFT consensus phases."""

    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    REPLY = "reply"


@dataclass
class ConsensusResult:
    """Result of consensus operation."""

    success: bool
    block_hash: Optional[str] = None
    validator_id: Optional[str] = None
    consensus_type: Optional[ConsensusType] = None
    timestamp: float = field(default_factory=time.time)
    gas_used: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusMetrics:
    """Consensus performance metrics."""

    total_blocks: int = 0
    successful_blocks: int = 0
    failed_blocks: int = 0
    average_block_time: float = 0.0
    average_gas_used: float = 0.0
    validator_count: int = 0
    active_validators: int = 0
    consensus_type: ConsensusType = ConsensusType.PROOF_OF_WORK
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate consensus success rate."""
        if self.total_blocks == 0:
            return 0.0
        return self.successful_blocks / self.total_blocks

    @property
    def failure_rate(self) -> float:
        """Calculate consensus failure rate."""
        return 1.0 - self.success_rate


@dataclass
class StakingInfo:
    """Information about staking operations."""

    validator_id: str
    delegator_id: str
    amount: int
    timestamp: float = field(default_factory=time.time)
    unbonding_time: Optional[float] = None
    reward_rate: float = 0.0
    slashing_penalty: float = 0.0


@dataclass
class VotingPower:
    """Voting power information."""

    validator_id: str
    total_power: int
    self_stake: int
    delegated_stake: int
    voting_weight: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class DelegateInfo:
    """Delegate information for DPoS."""

    delegate_id: str
    voter_id: str
    amount: int
    timestamp: float = field(default_factory=time.time)
    is_active: bool = True


@dataclass
class PBFTMessage:
    """PBFT consensus message."""

    message_type: PBFTPhase
    view_number: int
    sequence_number: int
    block_hash: str
    validator_id: str
    signature: str
    timestamp: float = field(default_factory=time.time)
    payload: Optional[Dict[str, Any]] = None


@dataclass
class ConsensusConfig:
    """Configuration for consensus mechanisms."""

    consensus_type: ConsensusType = ConsensusType.PROOF_OF_STAKE
    block_time: float = 2.0  # seconds
    max_validators: int = 100
    min_stake: int = 1000000  # minimum stake amount
    unbonding_period: int = 86400 * 7  # 7 days in seconds
    slashing_percentage: float = 0.05  # 5% slashing
    reward_rate: float = 0.1  # 10% annual reward
    max_gas_per_block: int = 10000000
    enable_hybrid: bool = False
    hybrid_switch_threshold: float = 0.8  # 80% failure rate triggers switch
    pbft_fault_tolerance: int = 1  # f = (n-1)/3 where n is total validators

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consensus_type": self.consensus_type.value,
            "block_time": self.block_time,
            "max_validators": self.max_validators,
            "min_stake": self.min_stake,
            "unbonding_period": self.unbonding_period,
            "slashing_percentage": self.slashing_percentage,
            "reward_rate": self.reward_rate,
            "max_gas_per_block": self.max_gas_per_block,
            "enable_hybrid": self.enable_hybrid,
            "hybrid_switch_threshold": self.hybrid_switch_threshold,
            "pbft_fault_tolerance": self.pbft_fault_tolerance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusConfig":
        """Create from dictionary."""
        return cls(
            consensus_type=ConsensusType(data.get("consensus_type", "proof_of_stake")),
            block_time=data.get("block_time", 2.0),
            max_validators=data.get("max_validators", 100),
            min_stake=data.get("min_stake", 1000000),
            unbonding_period=data.get("unbonding_period", 86400 * 7),
            slashing_percentage=data.get("slashing_percentage", 0.05),
            reward_rate=data.get("reward_rate", 0.1),
            max_gas_per_block=data.get("max_gas_per_block", 10000000),
            enable_hybrid=data.get("enable_hybrid", False),
            hybrid_switch_threshold=data.get("hybrid_switch_threshold", 0.8),
            pbft_fault_tolerance=data.get("pbft_fault_tolerance", 1),
        )


@dataclass
class ConsensusState:
    """Current state of consensus mechanism."""

    current_consensus: ConsensusType
    active_validators: List[str] = field(default_factory=list)
    current_proposer: Optional[str] = None
    current_view: int = 0
    last_block_time: float = field(default_factory=time.time)
    metrics: ConsensusMetrics = field(default_factory=ConsensusMetrics)
    config: ConsensusConfig = field(default_factory=ConsensusConfig)

    def update_metrics(self, success: bool, block_time: float, gas_used: int) -> None:
        """Update consensus metrics."""
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

    def should_switch_consensus(self) -> bool:
        """Check if consensus should be switched."""
        if not self.config.enable_hybrid:
            return False

        return self.metrics.failure_rate >= self.config.hybrid_switch_threshold
