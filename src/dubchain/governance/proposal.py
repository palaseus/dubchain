"""
Governance proposal system for DubChain.

This module provides a comprehensive governance system for managing
blockchain parameters, protocol upgrades, and community decisions.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey


class ProposalType(Enum):
    """Types of governance proposals."""

    PARAMETER_CHANGE = "parameter_change"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    CONSENSUS_CHANGE = "consensus_change"
    TREASURY_ALLOCATION = "treasury_allocation"
    VALIDATOR_MANAGEMENT = "validator_management"
    EMERGENCY_PAUSE = "emergency_pause"
    CUSTOM = "custom"


class ProposalStatus(Enum):
    """Status of a governance proposal."""

    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class VoteType(Enum):
    """Types of votes."""

    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class ProposalParameter:
    """A parameter in a governance proposal."""

    name: str
    current_value: Any
    proposed_value: Any
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "description": self.description,
        }


@dataclass
class ProposalVote:
    """A vote on a governance proposal."""

    voter_address: str
    vote_type: VoteType
    voting_power: int
    timestamp: int = 0
    signature: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "voter_address": self.voter_address,
            "vote_type": self.vote_type.value,
            "voting_power": self.voting_power,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }


@dataclass
class GovernanceProposal:
    """A governance proposal."""

    proposal_id: str
    proposer_address: str
    proposal_type: ProposalType
    title: str
    description: str
    parameters: List[ProposalParameter] = field(default_factory=list)
    votes: List[ProposalVote] = field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: int = 0
    voting_start: int = 0
    voting_end: int = 0
    execution_time: int = 0
    quorum_threshold: float = 0.5
    approval_threshold: float = 0.5
    min_voting_power: int = 1000
    max_voting_power: int = 1000000
    execution_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = int(time.time())

    def add_parameter(self, parameter: ProposalParameter) -> None:
        """Add a parameter to the proposal."""
        self.parameters.append(parameter)

    def add_vote(self, vote: ProposalVote) -> bool:
        """Add a vote to the proposal."""
        # Check if voter already voted
        for existing_vote in self.votes:
            if existing_vote.voter_address == vote.voter_address:
                return False

        # Check voting period
        current_time = int(time.time())
        if current_time < self.voting_start or current_time > self.voting_end:
            return False

        # Check voting power
        if vote.voting_power < self.min_voting_power:
            return False

        self.votes.append(vote)
        return True

    def get_vote_summary(self) -> Dict[str, Any]:
        """Get summary of votes."""
        yes_votes = sum(
            v.voting_power for v in self.votes if v.vote_type == VoteType.YES
        )
        no_votes = sum(v.voting_power for v in self.votes if v.vote_type == VoteType.NO)
        abstain_votes = sum(
            v.voting_power for v in self.votes if v.vote_type == VoteType.ABSTAIN
        )

        total_votes = yes_votes + no_votes + abstain_votes

        return {
            "total_votes": total_votes,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "abstain_votes": abstain_votes,
            "voter_count": len(self.votes),
            "quorum_met": total_votes
            >= (self.quorum_threshold * self.max_voting_power),
            "approval_met": yes_votes >= (self.approval_threshold * total_votes)
            if total_votes > 0
            else False,
        }

    def update_status(self) -> None:
        """Update proposal status based on votes and timing."""
        current_time = int(time.time())

        if self.status in [
            ProposalStatus.PASSED,
            ProposalStatus.REJECTED,
            ProposalStatus.EXECUTED,
            ProposalStatus.CANCELLED,
        ]:
            return

        # Check if voting period has ended
        if current_time > self.voting_end:
            vote_summary = self.get_vote_summary()

            if vote_summary["quorum_met"] and vote_summary["approval_met"]:
                self.status = ProposalStatus.PASSED
            else:
                self.status = ProposalStatus.REJECTED
        elif current_time >= self.voting_start:
            self.status = ProposalStatus.ACTIVE

    def can_be_executed(self) -> bool:
        """Check if proposal can be executed."""
        return (
            self.status == ProposalStatus.PASSED
            and int(time.time()) >= self.execution_time
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "proposer_address": self.proposer_address,
            "proposal_type": self.proposal_type.value,
            "title": self.title,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "votes": [v.to_dict() for v in self.votes],
            "status": self.status.value,
            "created_at": self.created_at,
            "voting_start": self.voting_start,
            "voting_end": self.voting_end,
            "execution_time": self.execution_time,
            "quorum_threshold": self.quorum_threshold,
            "approval_threshold": self.approval_threshold,
            "min_voting_power": self.min_voting_power,
            "max_voting_power": self.max_voting_power,
            "execution_data": self.execution_data,
            "vote_summary": self.get_vote_summary(),
        }


class GovernanceManager:
    """Manages governance proposals and voting."""

    def __init__(self):
        self.proposals: Dict[str, GovernanceProposal] = {}
        self.voting_power: Dict[str, int] = {}
        self.validators: Set[str] = set()

    def create_proposal(
        self,
        proposer_address: str,
        proposal_type: ProposalType,
        title: str,
        description: str,
        voting_duration: int = 7 * 24 * 3600,  # 7 days
        execution_delay: int = 24 * 3600,  # 1 day
    ) -> str:
        """Create a new governance proposal."""
        proposal_id = SHA256Hasher.hash(
            f"{proposer_address}_{title}_{int(time.time())}".encode()
        ).to_hex()

        current_time = int(time.time())
        proposal = GovernanceProposal(
            proposal_id=proposal_id,
            proposer_address=proposer_address,
            proposal_type=proposal_type,
            title=title,
            description=description,
            voting_start=current_time,
            voting_end=current_time + voting_duration,
            execution_time=current_time + voting_duration + execution_delay,
        )

        self.proposals[proposal_id] = proposal
        return proposal_id

    def add_parameter_to_proposal(
        self,
        proposal_id: str,
        name: str,
        current_value: Any,
        proposed_value: Any,
        description: str = "",
    ) -> bool:
        """Add a parameter to a proposal."""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]
        if proposal.status != ProposalStatus.DRAFT:
            return False

        parameter = ProposalParameter(
            name=name,
            current_value=current_value,
            proposed_value=proposed_value,
            description=description,
        )

        proposal.add_parameter(parameter)
        return True

    def vote_on_proposal(
        self,
        proposal_id: str,
        voter_address: str,
        vote_type: VoteType,
        private_key: PrivateKey,
    ) -> bool:
        """Vote on a proposal."""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        # Check if voter has voting power
        voting_power = self.voting_power.get(voter_address, 0)
        if voting_power == 0:
            return False

        # Create vote
        vote = ProposalVote(
            voter_address=voter_address, vote_type=vote_type, voting_power=voting_power
        )

        # Sign the vote
        vote_data = f"{proposal_id}_{voter_address}_{vote_type.value}_{voting_power}_{vote.timestamp}"
        vote_hash = SHA256Hasher.hash(vote_data.encode())
        signature = private_key.sign(vote_hash)
        vote.signature = signature.to_hex()

        # Add vote to proposal
        success = proposal.add_vote(vote)
        if success:
            proposal.update_status()

        return success

    def get_proposal(self, proposal_id: str) -> Optional[GovernanceProposal]:
        """Get a proposal by ID."""
        return self.proposals.get(proposal_id)

    def get_all_proposals(self) -> List[GovernanceProposal]:
        """Get all proposals."""
        return list(self.proposals.values())

    def get_active_proposals(self) -> List[GovernanceProposal]:
        """Get active proposals."""
        return [p for p in self.proposals.values() if p.status == ProposalStatus.ACTIVE]

    def get_passed_proposals(self) -> List[GovernanceProposal]:
        """Get passed proposals."""
        return [p for p in self.proposals.values() if p.status == ProposalStatus.PASSED]

    def set_voting_power(self, address: str, power: int) -> None:
        """Set voting power for an address."""
        self.voting_power[address] = power

    def add_validator(self, address: str) -> None:
        """Add a validator address."""
        self.validators.add(address)

    def remove_validator(self, address: str) -> None:
        """Remove a validator address."""
        self.validators.discard(address)

    def is_validator(self, address: str) -> bool:
        """Check if address is a validator."""
        return address in self.validators

    def execute_proposal(self, proposal_id: str) -> bool:
        """Execute a passed proposal."""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        if not proposal.can_be_executed():
            return False

        # In a real implementation, would execute the proposal changes
        # For now, just mark as executed
        proposal.status = ProposalStatus.EXECUTED
        return True

    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        total_proposals = len(self.proposals)
        active_proposals = len(self.get_active_proposals())
        passed_proposals = len(self.get_passed_proposals())

        total_voters = len(self.voting_power)
        total_validators = len(self.validators)

        return {
            "total_proposals": total_proposals,
            "active_proposals": active_proposals,
            "passed_proposals": passed_proposals,
            "total_voters": total_voters,
            "total_validators": total_validators,
            "total_voting_power": sum(self.voting_power.values()),
        }
