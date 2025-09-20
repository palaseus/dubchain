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
    # New consensus mechanisms
    PROOF_OF_AUTHORITY = "proof_of_authority"
    PROOF_OF_HISTORY = "proof_of_history"
    PROOF_OF_SPACE_TIME = "proof_of_space_time"
    HOTSTUFF = "hotstuff"


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


class HotStuffPhase(Enum):
    """HotStuff consensus phases."""

    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDE = "decide"


class PoAStatus(Enum):
    """Proof-of-Authority validator status."""

    AUTHORITY = "authority"
    CANDIDATE = "candidate"
    REVOKED = "revoked"


class PoHStatus(Enum):
    """Proof-of-History status."""

    GENERATING = "generating"
    VERIFIED = "verified"
    INVALID = "invalid"


class PoSpaceStatus(Enum):
    """Proof-of-Space/Time status."""

    PLOTTING = "plotting"
    FARMING = "farming"
    CHALLENGING = "challenging"
    PROVING = "proving"


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
class HotStuffMessage:
    """HotStuff consensus message."""

    message_type: HotStuffPhase
    view_number: int
    block_hash: str
    parent_hash: str
    validator_id: str
    signature: str
    timestamp: float = field(default_factory=time.time)
    payload: Optional[Dict[str, Any]] = None


@dataclass
class PoAAuthority:
    """Proof-of-Authority authority information."""

    authority_id: str
    public_key: str
    status: PoAStatus
    reputation_score: float = 100.0
    blocks_proposed: int = 0
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True


@dataclass
class PoHEntry:
    """Proof-of-History entry."""

    entry_id: str
    timestamp: float
    hash: str
    previous_hash: str
    data: bytes
    proof: str
    validator_id: str


@dataclass
class PoSpacePlot:
    """Proof-of-Space plot information."""

    plot_id: str
    farmer_id: str
    size_bytes: int
    plot_hash: str
    created_at: float = field(default_factory=time.time)
    last_challenge: float = field(default_factory=time.time)
    challenges_won: int = 0
    is_active: bool = True


@dataclass
class PoSpaceChallenge:
    """Proof-of-Space challenge."""

    challenge_id: str
    challenge_data: bytes
    difficulty: int
    timestamp: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 minutes


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
    
    # Proof-of-Authority specific parameters
    poa_authority_set: List[str] = field(default_factory=list)  # List of authority IDs
    poa_reputation_threshold: float = 50.0  # Minimum reputation to propose blocks
    poa_slashing_threshold: float = 0.1  # 10% reputation loss for misbehavior
    poa_rotation_period: int = 86400 * 30  # 30 days authority rotation
    
    # Proof-of-History specific parameters
    poh_clock_frequency: float = 1.0  # Hz - frequency of PoH generation
    poh_verification_window: int = 100  # Number of entries to verify
    poh_max_skew: float = 0.1  # Maximum time skew allowed (seconds)
    poh_leader_rotation: int = 10  # Number of PoH entries per leader
    
    # Proof-of-Space/Time specific parameters
    pospace_min_plot_size: int = 1024 * 1024 * 100  # 100MB minimum plot size
    pospace_challenge_interval: int = 30  # seconds between challenges
    pospace_difficulty_adjustment: float = 0.1  # 10% difficulty adjustment
    pospace_max_plot_age: int = 86400 * 365  # 1 year maximum plot age
    
    # HotStuff specific parameters
    hotstuff_view_timeout: float = 5.0  # seconds before view change
    hotstuff_max_view_changes: int = 3  # Maximum view changes before fallback
    hotstuff_leader_rotation: int = 1  # Blocks per leader
    hotstuff_safety_threshold: float = 0.67  # 2/3 threshold for safety

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
            # PoA parameters
            "poa_authority_set": self.poa_authority_set,
            "poa_reputation_threshold": self.poa_reputation_threshold,
            "poa_slashing_threshold": self.poa_slashing_threshold,
            "poa_rotation_period": self.poa_rotation_period,
            # PoH parameters
            "poh_clock_frequency": self.poh_clock_frequency,
            "poh_verification_window": self.poh_verification_window,
            "poh_max_skew": self.poh_max_skew,
            "poh_leader_rotation": self.poh_leader_rotation,
            # PoSpace parameters
            "pospace_min_plot_size": self.pospace_min_plot_size,
            "pospace_challenge_interval": self.pospace_challenge_interval,
            "pospace_difficulty_adjustment": self.pospace_difficulty_adjustment,
            "pospace_max_plot_age": self.pospace_max_plot_age,
            # HotStuff parameters
            "hotstuff_view_timeout": self.hotstuff_view_timeout,
            "hotstuff_max_view_changes": self.hotstuff_max_view_changes,
            "hotstuff_leader_rotation": self.hotstuff_leader_rotation,
            "hotstuff_safety_threshold": self.hotstuff_safety_threshold,
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
            # PoA parameters
            poa_authority_set=data.get("poa_authority_set", []),
            poa_reputation_threshold=data.get("poa_reputation_threshold", 50.0),
            poa_slashing_threshold=data.get("poa_slashing_threshold", 0.1),
            poa_rotation_period=data.get("poa_rotation_period", 86400 * 30),
            # PoH parameters
            poh_clock_frequency=data.get("poh_clock_frequency", 1.0),
            poh_verification_window=data.get("poh_verification_window", 100),
            poh_max_skew=data.get("poh_max_skew", 0.1),
            poh_leader_rotation=data.get("poh_leader_rotation", 10),
            # PoSpace parameters
            pospace_min_plot_size=data.get("pospace_min_plot_size", 1024 * 1024 * 100),
            pospace_challenge_interval=data.get("pospace_challenge_interval", 30),
            pospace_difficulty_adjustment=data.get("pospace_difficulty_adjustment", 0.1),
            pospace_max_plot_age=data.get("pospace_max_plot_age", 86400 * 365),
            # HotStuff parameters
            hotstuff_view_timeout=data.get("hotstuff_view_timeout", 5.0),
            hotstuff_max_view_changes=data.get("hotstuff_max_view_changes", 3),
            hotstuff_leader_rotation=data.get("hotstuff_leader_rotation", 1),
            hotstuff_safety_threshold=data.get("hotstuff_safety_threshold", 0.67),
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
