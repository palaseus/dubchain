"""
Core governance types and data structures.

This module defines the fundamental types and data structures used throughout
the governance system, including proposals, votes, and governance state.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey
from ..errors.exceptions import ValidationError, GovernanceError


class ProposalStatus(Enum):
    """Status of a governance proposal."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    QUEUED = "queued"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class ProposalType(Enum):
    """Type of governance proposal."""
    
    PARAMETER_CHANGE = "parameter_change"
    TREASURY_SPENDING = "treasury_spending"
    UPGRADE = "upgrade"
    EMERGENCY = "emergency"
    CUSTOM = "custom"


class VoteChoice(Enum):
    """Vote choices for governance proposals."""
    
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"


@dataclass
class VotingPower:
    """Represents voting power for a voter."""
    
    voter_address: str
    power: int
    token_balance: int
    delegated_power: int = 0
    delegation_chain_length: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def total_power(self) -> int:
        """Get total voting power including delegations."""
        return self.power + self.delegated_power
    
    def is_delegated(self) -> bool:
        """Check if this voting power includes delegations."""
        return self.delegated_power > 0
    
    def __post_init__(self):
        """Validate voting power after initialization."""
        if self.power < 0:
            raise ValidationError("Voting power cannot be negative")


@dataclass
class Vote:
    """A governance vote."""
    
    proposal_id: str
    voter_address: str
    choice: VoteChoice
    voting_power: VotingPower
    signature: str
    timestamp: float = field(default_factory=time.time)
    block_height: Optional[int] = None
    transaction_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate vote after initialization."""
        if self.voting_power.total_power() <= 0:
            raise ValidationError("Voting power must be positive")
        
        if not self.signature:
            raise ValidationError("Vote must be signed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vote to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "voter_address": self.voter_address,
            "choice": self.choice.value,
            "voting_power": {
                "voter_address": self.voting_power.voter_address,
                "power": self.voting_power.power,
                "token_balance": self.voting_power.token_balance,
                "delegated_power": self.voting_power.delegated_power,
                "delegation_chain_length": self.voting_power.delegation_chain_length,
                "timestamp": self.voting_power.timestamp,
            },
            "signature": self.signature,
            "timestamp": self.timestamp,
            "block_height": self.block_height,
            "transaction_hash": self.transaction_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vote":
        """Create vote from dictionary."""
        voting_power_data = data["voting_power"]
        voting_power = VotingPower(
            voter_address=voting_power_data["voter_address"],
            power=voting_power_data["power"],
            token_balance=voting_power_data["token_balance"],
            delegated_power=voting_power_data["delegated_power"],
            delegation_chain_length=voting_power_data["delegation_chain_length"],
            timestamp=voting_power_data["timestamp"],
        )
        
        return cls(
            proposal_id=data["proposal_id"],
            voter_address=data["voter_address"],
            choice=VoteChoice(data["choice"]),
            voting_power=voting_power,
            signature=data["signature"],
            timestamp=data["timestamp"],
            block_height=data.get("block_height"),
            transaction_hash=data.get("transaction_hash"),
        )


@dataclass
class Proposal:
    """A governance proposal."""
    
    proposal_id: str = field(default_factory=lambda: str(uuid4()))
    proposer_address: str = ""
    title: str = ""
    description: str = ""
    proposal_type: ProposalType = ProposalType.CUSTOM
    status: ProposalStatus = ProposalStatus.DRAFT
    
    # Voting parameters
    voting_strategy: str = "token_weighted"
    start_block: Optional[int] = None
    end_block: Optional[int] = None
    quorum_threshold: int = 1000  # Minimum voting power required
    approval_threshold: float = 0.5  # Minimum approval percentage
    
    # Execution parameters
    execution_delay: int = 0  # Blocks to wait before execution
    execution_data: Optional[Dict[str, Any]] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    
    # Votes
    votes: List[Vote] = field(default_factory=list)
    
    # Security and validation
    signature: Optional[str] = None
    block_height: Optional[int] = None
    transaction_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate proposal after initialization."""
        if not self.proposer_address:
            raise ValidationError("Proposal must have a proposer")
        
        if not self.title:
            raise ValidationError("Proposal must have a title")
        
        if self.quorum_threshold <= 0:
            raise ValidationError("Quorum threshold must be positive")
        
        if not 0 <= self.approval_threshold <= 1:
            raise ValidationError("Approval threshold must be between 0 and 1")
    
    def add_vote(self, vote: Vote) -> None:
        """Add a vote to this proposal."""
        if vote.proposal_id != self.proposal_id:
            raise ValidationError("Vote proposal ID does not match")
        
        # Check for duplicate votes
        for existing_vote in self.votes:
            if existing_vote.voter_address == vote.voter_address:
                raise ValidationError("Voter has already voted on this proposal")
        
        self.votes.append(vote)
        self.updated_at = time.time()
    
    def get_vote_summary(self) -> Dict[str, Any]:
        """Get summary of votes for this proposal."""
        total_power = 0
        for_power = 0
        against_power = 0
        abstain_power = 0
        
        for vote in self.votes:
            power = vote.voting_power.total_power()
            total_power += power
            
            if vote.choice == VoteChoice.FOR:
                for_power += power
            elif vote.choice == VoteChoice.AGAINST:
                against_power += power
            elif vote.choice == VoteChoice.ABSTAIN:
                abstain_power += power
        
        return {
            "total_voting_power": total_power,
            "for_power": for_power,
            "against_power": against_power,
            "abstain_power": abstain_power,
            "total_votes": len(self.votes),
            "quorum_met": total_power >= self.quorum_threshold,
            "approval_percentage": for_power / max(total_power, 1),
            "approved": (
                total_power >= self.quorum_threshold and 
                for_power / max(total_power, 1) >= self.approval_threshold
            ),
        }
    
    def can_execute(self) -> bool:
        """Check if proposal can be executed."""
        if self.status != ProposalStatus.QUEUED:
            return False
        
        summary = self.get_vote_summary()
        return summary["approved"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "proposer_address": self.proposer_address,
            "title": self.title,
            "description": self.description,
            "proposal_type": self.proposal_type.value,
            "status": self.status.value,
            "voting_strategy": self.voting_strategy,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "quorum_threshold": self.quorum_threshold,
            "approval_threshold": self.approval_threshold,
            "execution_delay": self.execution_delay,
            "execution_data": self.execution_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "executed_at": self.executed_at,
            "votes": [vote.to_dict() for vote in self.votes],
            "signature": self.signature,
            "block_height": self.block_height,
            "transaction_hash": self.transaction_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proposal":
        """Create proposal from dictionary."""
        proposal = cls(
            proposal_id=data["proposal_id"],
            proposer_address=data["proposer_address"],
            title=data["title"],
            description=data["description"],
            proposal_type=ProposalType(data["proposal_type"]),
            status=ProposalStatus(data["status"]),
            voting_strategy=data["voting_strategy"],
            start_block=data.get("start_block"),
            end_block=data.get("end_block"),
            quorum_threshold=data["quorum_threshold"],
            approval_threshold=data["approval_threshold"],
            execution_delay=data.get("execution_delay", 0),
            execution_data=data.get("execution_data"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            executed_at=data.get("executed_at"),
            signature=data.get("signature"),
            block_height=data.get("block_height"),
            transaction_hash=data.get("transaction_hash"),
        )
        
        # Add votes
        for vote_data in data.get("votes", []):
            vote = Vote.from_dict(vote_data)
            proposal.votes.append(vote)
        
        return proposal


@dataclass
class GovernanceConfig:
    """Configuration for the governance system."""
    
    # Voting parameters
    default_quorum_threshold: int = 1000
    default_approval_threshold: float = 0.5
    default_voting_period: int = 1000  # blocks
    default_execution_delay: int = 100  # blocks
    
    # Security parameters
    max_proposal_description_length: int = 10000
    min_proposal_title_length: int = 10
    max_voting_strategies: int = 10
    
    # Emergency parameters
    emergency_threshold: float = 0.8  # 80% for emergency proposals
    emergency_execution_delay: int = 10  # blocks
    
    # Delegation parameters
    max_delegation_chain_length: int = 10
    delegation_cooldown_period: int = 100  # blocks
    
    # Treasury parameters
    max_treasury_spending_per_proposal: int = 1000000
    treasury_multisig_threshold: int = 3
    
    # Upgrade parameters
    upgrade_timelock_period: int = 1000  # blocks
    emergency_upgrade_threshold: float = 0.9  # 90% for emergency upgrades
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.default_quorum_threshold <= 0:
            raise ValidationError("Default quorum threshold must be positive")
        
        if not 0 <= self.default_approval_threshold <= 1:
            raise ValidationError("Default approval threshold must be between 0 and 1")
        
        if self.default_voting_period <= 0:
            raise ValidationError("Default voting period must be positive")
        
        if self.max_delegation_chain_length <= 0:
            raise ValidationError("Max delegation chain length must be positive")


@dataclass
class GovernanceState:
    """Current state of the governance system."""
    
    config: GovernanceConfig
    proposals: Dict[str, Proposal] = field(default_factory=dict)
    active_proposals: Set[str] = field(default_factory=set)
    queued_proposals: Set[str] = field(default_factory=set)
    
    # Voting power snapshots
    voting_power_snapshots: Dict[int, Dict[str, VotingPower]] = field(default_factory=dict)
    
    # Delegation state
    delegations: Dict[str, str] = field(default_factory=dict)  # delegatee -> delegator
    delegation_chains: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution state
    execution_queue: List[str] = field(default_factory=list)
    timelock_queue: Dict[str, int] = field(default_factory=dict)
    
    # Emergency state
    emergency_paused: bool = False
    emergency_pause_reason: Optional[str] = None
    emergency_pause_block: Optional[int] = None
    
    # Metrics
    total_proposals: int = 0
    executed_proposals: int = 0
    failed_proposals: int = 0
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get a proposal by ID."""
        return self.proposals.get(proposal_id)
    
    def add_proposal(self, proposal: Proposal) -> None:
        """Add a proposal to the state."""
        self.proposals[proposal.proposal_id] = proposal
        self.total_proposals += 1
        
        if proposal.status == ProposalStatus.ACTIVE:
            self.active_proposals.add(proposal.proposal_id)
        elif proposal.status == ProposalStatus.QUEUED:
            self.queued_proposals.add(proposal.proposal_id)
    
    def update_proposal_status(self, proposal_id: str, status: ProposalStatus) -> None:
        """Update proposal status."""
        if proposal_id not in self.proposals:
            raise ValidationError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        old_status = proposal.status
        proposal.status = status
        proposal.updated_at = time.time()
        
        # Update sets
        self.active_proposals.discard(proposal_id)
        self.queued_proposals.discard(proposal_id)
        
        if status == ProposalStatus.ACTIVE:
            self.active_proposals.add(proposal_id)
        elif status == ProposalStatus.QUEUED:
            self.queued_proposals.add(proposal_id)
        elif status == ProposalStatus.EXECUTED:
            self.executed_proposals += 1
        elif status == ProposalStatus.FAILED:
            self.failed_proposals += 1


class GovernanceEngine:
    """Main governance engine for DubChain."""
    
    def __init__(self, config: GovernanceConfig):
        """Initialize governance engine."""
        config.validate()
        self.config = config
        self.state = GovernanceState(config)
        
        # Initialize components (will be injected)
        self.voting_strategy = None
        self.delegation_manager = None
        self.execution_engine = None
        self.security_manager = None
        self.treasury_manager = None
        self.upgrade_manager = None
        self.observability = None
    
    def create_proposal(
        self,
        proposer_address: str,
        title: str,
        description: str,
        proposal_type: ProposalType,
        voting_strategy: str = "token_weighted",
        quorum_threshold: Optional[int] = None,
        approval_threshold: Optional[float] = None,
        execution_delay: Optional[int] = None,
        execution_data: Optional[Dict[str, Any]] = None,
    ) -> Proposal:
        """Create a new governance proposal."""
        if self.state.emergency_paused:
            raise GovernanceError("Governance is paused due to emergency")
        
        # Validate inputs
        if len(title) < self.config.min_proposal_title_length:
            raise ValidationError(f"Title must be at least {self.config.min_proposal_title_length} characters")
        
        if len(description) > self.config.max_proposal_description_length:
            raise ValidationError(f"Description must be at most {self.config.max_proposal_description_length} characters")
        
        # Create proposal
        proposal = Proposal(
            proposer_address=proposer_address,
            title=title,
            description=description,
            proposal_type=proposal_type,
            voting_strategy=voting_strategy,
            quorum_threshold=quorum_threshold or self.config.default_quorum_threshold,
            approval_threshold=approval_threshold or self.config.default_approval_threshold,
            execution_delay=execution_delay or self.config.default_execution_delay,
            execution_data=execution_data,
        )
        
        # Add to state
        self.state.add_proposal(proposal)
        
        return proposal
    
    def cast_vote(
        self,
        proposal_id: str,
        voter_address: str,
        choice: VoteChoice,
        voting_power: VotingPower,
        signature: str,
    ) -> Vote:
        """Cast a vote on a proposal."""
        if self.state.emergency_paused:
            raise GovernanceError("Governance is paused due to emergency")
        
        proposal = self.state.get_proposal(proposal_id)
        if not proposal:
            raise ValidationError(f"Proposal {proposal_id} not found")
        
        if proposal.status != ProposalStatus.ACTIVE:
            raise ValidationError("Can only vote on active proposals")
        
        # Create vote
        vote = Vote(
            proposal_id=proposal_id,
            voter_address=voter_address,
            choice=choice,
            voting_power=voting_power,
            signature=signature,
        )
        
        # Add vote to proposal
        proposal.add_vote(vote)
        
        return vote
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get a proposal by ID."""
        return self.state.get_proposal(proposal_id)
    
    def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals."""
        return [self.state.proposals[pid] for pid in self.state.active_proposals]
    
    def get_queued_proposals(self) -> List[Proposal]:
        """Get all queued proposals."""
        return [self.state.proposals[pid] for pid in self.state.queued_proposals]
    
    def emergency_pause(self, reason: str, block_height: int) -> None:
        """Pause governance due to emergency."""
        self.state.emergency_paused = True
        self.state.emergency_pause_reason = reason
        self.state.emergency_pause_block = block_height
    
    def emergency_resume(self) -> None:
        """Resume governance after emergency."""
        self.state.emergency_paused = False
        self.state.emergency_pause_reason = None
        self.state.emergency_pause_block = None
