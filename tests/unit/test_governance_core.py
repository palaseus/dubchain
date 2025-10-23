"""
Unit tests for governance core functionality.

This module tests the core governance types, proposal lifecycle,
voting mechanisms, and basic governance operations.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
from unittest.mock import Mock, patch

from dubchain.governance.core import (
    GovernanceEngine,
    GovernanceConfig,
    GovernanceState,
    Proposal,
    ProposalStatus,
    ProposalType,
    Vote,
    VoteChoice,
    VotingPower,
)
from dubchain.errors.exceptions import ValidationError


class TestVotingPower:
    """Test VotingPower class."""
    
    def test_voting_power_creation(self):
        """Test creating voting power."""
        power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000,
            delegated_power=500
        )
        
        assert power.voter_address == "0x123"
        assert power.power == 1000
        assert power.token_balance == 1000
        assert power.delegated_power == 500
        assert power.total_power() == 1500
        assert power.is_delegated() is True
    
    def test_voting_power_no_delegation(self):
        """Test voting power without delegation."""
        power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        assert power.delegated_power == 0
        assert power.total_power() == 1000
        assert power.is_delegated() is False
    
    def test_voting_power_negative_power(self):
        """Test that negative power raises validation error."""
        with pytest.raises(ValidationError):
            VotingPower(
                voter_address="0x123",
                power=-100,
                token_balance=1000
            )


class TestVote:
    """Test Vote class."""
    
    def test_vote_creation(self):
        """Test creating a vote."""
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        assert vote.proposal_id == "prop_123"
        assert vote.voter_address == "0x123"
        assert vote.choice == VoteChoice.FOR
        assert vote.voting_power == voting_power
        assert vote.signature == "0xabc123"
    
    def test_vote_validation_zero_power(self):
        """Test that vote with zero power raises validation error."""
        voting_power = VotingPower(
            voter_address="0x123",
            power=0,
            token_balance=0
        )

        with pytest.raises(ValidationError):
            Vote(
                proposal_id="prop_123",
                voter_address="0x123",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature="0xabc123"
            )
    
    def test_vote_validation_no_signature(self):
        """Test that vote without signature raises validation error."""
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )

        with pytest.raises(ValidationError):
            Vote(
                proposal_id="prop_123",
                voter_address="0x123",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=""
            )
    
    def test_vote_serialization(self):
        """Test vote serialization and deserialization."""
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000,
            delegated_power=500
        )
        
        original_vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123",
            block_height=100,
            transaction_hash="0xtx123"
        )
        
        # Serialize
        vote_dict = original_vote.to_dict()
        
        # Deserialize
        deserialized_vote = Vote.from_dict(vote_dict)
        
        assert deserialized_vote.proposal_id == original_vote.proposal_id
        assert deserialized_vote.voter_address == original_vote.voter_address
        assert deserialized_vote.choice == original_vote.choice
        assert deserialized_vote.voting_power.voter_address == original_vote.voting_power.voter_address
        assert deserialized_vote.voting_power.power == original_vote.voting_power.power
        assert deserialized_vote.signature == original_vote.signature
        assert deserialized_vote.block_height == original_vote.block_height
        assert deserialized_vote.transaction_hash == original_vote.transaction_hash


class TestProposal:
    """Test Proposal class."""
    
    def test_proposal_creation(self):
        """Test creating a proposal."""
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            quorum_threshold=1000,
            approval_threshold=0.5
        )
        
        assert proposal.proposer_address == "0x123"
        assert proposal.title == "Test Proposal"
        assert proposal.description == "This is a test proposal"
        assert proposal.proposal_type == ProposalType.PARAMETER_CHANGE
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.quorum_threshold == 1000
        assert proposal.approval_threshold == 0.5
        assert len(proposal.votes) == 0
    
    def test_proposal_validation_no_proposer(self):
        """Test that proposal without proposer raises validation error."""
        with pytest.raises(ValidationError):
            Proposal(
                proposer_address="",
                title="Test Proposal",
                description="This is a test proposal"
            )
    
    def test_proposal_validation_no_title(self):
        """Test that proposal without title raises validation error."""
        with pytest.raises(ValidationError):
            Proposal(
                proposer_address="0x123",
                title="",
                description="This is a test proposal"
            )
    
    def test_proposal_validation_negative_quorum(self):
        """Test that proposal with negative quorum raises validation error."""
        with pytest.raises(ValidationError):
            Proposal(
                proposer_address="0x123",
                title="Test Proposal",
                description="This is a test proposal",
                quorum_threshold=-100
            )
    
    def test_proposal_validation_invalid_approval_threshold(self):
        """Test that proposal with invalid approval threshold raises validation error."""
        with pytest.raises(ValidationError):
            Proposal(
                proposer_address="0x123",
                title="Test Proposal",
                description="This is a test proposal",
                approval_threshold=1.5
            )
    
    def test_proposal_add_vote(self):
        """Test adding votes to a proposal."""
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal.add_vote(vote)
        
        assert len(proposal.votes) == 1
        assert proposal.votes[0] == vote
    
    def test_proposal_duplicate_vote(self):
        """Test that duplicate votes raise validation error."""
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        
        vote1 = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        vote2 = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.AGAINST,
            voting_power=voting_power,
            signature="0xdef456"
        )
        
        proposal.add_vote(vote1)
        
        with pytest.raises(ValidationError):
            proposal.add_vote(vote2)
    
    def test_proposal_vote_summary(self):
        """Test proposal vote summary calculation."""
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            quorum_threshold=2000,
            approval_threshold=0.5
        )
        
        # Add FOR votes
        for i in range(3):
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=1000,
                token_balance=1000
            )
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter_address=f"0x{i}",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            proposal.add_vote(vote)
        
        # Add AGAINST vote
        voting_power = VotingPower(
            voter_address="0x3",
            power=500,
            token_balance=500
        )
        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x3",
            choice=VoteChoice.AGAINST,
            voting_power=voting_power,
            signature="0x3"
        )
        proposal.add_vote(vote)
        
        summary = proposal.get_vote_summary()
        
        assert summary["total_voting_power"] == 3500
        assert summary["for_power"] == 3000
        assert summary["against_power"] == 500
        assert summary["abstain_power"] == 0
        assert summary["total_votes"] == 4
        assert summary["quorum_met"] is True
        assert summary["approval_percentage"] == 3000 / 3500
        assert summary["approved"] is True
    
    def test_proposal_can_execute(self):
        """Test proposal execution eligibility."""
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            status=ProposalStatus.QUEUED,
            quorum_threshold=1000,
            approval_threshold=0.5
        )
        
        # Add enough FOR votes to meet quorum and approval threshold
        voting_power = VotingPower(
            voter_address="0x456",
            power=1500,
            token_balance=1500
        )
        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        proposal.add_vote(vote)
        
        assert proposal.can_execute() is True
    
    def test_proposal_serialization(self):
        """Test proposal serialization and deserialization."""
        original_proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            status=ProposalStatus.ACTIVE,
            voting_strategy="token_weighted",
            start_block=100,
            end_block=200,
            quorum_threshold=1000,
            approval_threshold=0.5,
            execution_delay=50,
            execution_data={"param": "value"},
            signature="0xproposal123",
            block_height=100,
            transaction_hash="0xtx123"
        )
        
        # Add a vote
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        vote = Vote(
            proposal_id=original_proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        original_proposal.add_vote(vote)
        
        # Serialize
        proposal_dict = original_proposal.to_dict()
        
        # Deserialize
        deserialized_proposal = Proposal.from_dict(proposal_dict)
        
        assert deserialized_proposal.proposal_id == original_proposal.proposal_id
        assert deserialized_proposal.proposer_address == original_proposal.proposer_address
        assert deserialized_proposal.title == original_proposal.title
        assert deserialized_proposal.description == original_proposal.description
        assert deserialized_proposal.proposal_type == original_proposal.proposal_type
        assert deserialized_proposal.status == original_proposal.status
        assert deserialized_proposal.voting_strategy == original_proposal.voting_strategy
        assert deserialized_proposal.start_block == original_proposal.start_block
        assert deserialized_proposal.end_block == original_proposal.end_block
        assert deserialized_proposal.quorum_threshold == original_proposal.quorum_threshold
        assert deserialized_proposal.approval_threshold == original_proposal.approval_threshold
        assert deserialized_proposal.execution_delay == original_proposal.execution_delay
        assert deserialized_proposal.execution_data == original_proposal.execution_data
        assert deserialized_proposal.signature == original_proposal.signature
        assert deserialized_proposal.block_height == original_proposal.block_height
        assert deserialized_proposal.transaction_hash == original_proposal.transaction_hash
        assert len(deserialized_proposal.votes) == 1


class TestGovernanceConfig:
    """Test GovernanceConfig class."""
    
    def test_governance_config_creation(self):
        """Test creating governance configuration."""
        config = GovernanceConfig(
            default_quorum_threshold=2000,
            default_approval_threshold=0.6,
            default_voting_period=2000,
            default_execution_delay=200,
            max_proposal_description_length=5000,
            min_proposal_title_length=20
        )
        
        assert config.default_quorum_threshold == 2000
        assert config.default_approval_threshold == 0.6
        assert config.default_voting_period == 2000
        assert config.default_execution_delay == 200
        assert config.max_proposal_description_length == 5000
        assert config.min_proposal_title_length == 20
    
    def test_governance_config_validation(self):
        """Test governance configuration validation."""
        # Test negative quorum threshold
        with pytest.raises(ValidationError):
            GovernanceConfig(default_quorum_threshold=-100)
        
        # Test invalid approval threshold
        with pytest.raises(ValidationError):
            GovernanceConfig(default_approval_threshold=1.5)
        
        # Test negative voting period
        with pytest.raises(ValidationError):
            GovernanceConfig(default_voting_period=-100)
        
        # Test negative delegation chain length
        with pytest.raises(ValidationError):
            GovernanceConfig(max_delegation_chain_length=-1)


class TestGovernanceState:
    """Test GovernanceState class."""
    
    def test_governance_state_creation(self):
        """Test creating governance state."""
        config = GovernanceConfig()
        state = GovernanceState(config)
        
        assert state.config == config
        assert len(state.proposals) == 0
        assert len(state.active_proposals) == 0
        assert len(state.queued_proposals) == 0
        assert state.total_proposals == 0
        assert state.executed_proposals == 0
        assert state.failed_proposals == 0
    
    def test_governance_state_add_proposal(self):
        """Test adding proposals to governance state."""
        config = GovernanceConfig()
        state = GovernanceState(config)
        
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            status=ProposalStatus.ACTIVE
        )
        
        state.add_proposal(proposal)
        
        assert len(state.proposals) == 1
        assert proposal.proposal_id in state.proposals
        assert proposal.proposal_id in state.active_proposals
        assert state.total_proposals == 1
    
    def test_governance_state_update_proposal_status(self):
        """Test updating proposal status."""
        config = GovernanceConfig()
        state = GovernanceState(config)
        
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            status=ProposalStatus.ACTIVE
        )
        
        state.add_proposal(proposal)
        
        # Update to QUEUED
        state.update_proposal_status(proposal.proposal_id, ProposalStatus.QUEUED)
        
        assert proposal.status == ProposalStatus.QUEUED
        assert proposal.proposal_id not in state.active_proposals
        assert proposal.proposal_id in state.queued_proposals
        
        # Update to EXECUTED
        state.update_proposal_status(proposal.proposal_id, ProposalStatus.EXECUTED)
        
        assert proposal.status == ProposalStatus.EXECUTED
        assert proposal.proposal_id not in state.queued_proposals
        assert state.executed_proposals == 1


class TestGovernanceEngine:
    """Test GovernanceEngine class."""
    
    def test_governance_engine_creation(self):
        """Test creating governance engine."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        assert engine.config == config
        assert isinstance(engine.state, GovernanceState)
    
    def test_create_proposal(self):
        """Test creating a proposal through governance engine."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        proposal = engine.create_proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        assert proposal.proposer_address == "0x123"
        assert proposal.title == "Test Proposal"
        assert proposal.description == "This is a test proposal"
        assert proposal.proposal_type == ProposalType.PARAMETER_CHANGE
        assert proposal.quorum_threshold == config.default_quorum_threshold
        assert proposal.approval_threshold == config.default_approval_threshold
        assert proposal.execution_delay == config.default_execution_delay
        
        # Check that proposal was added to state
        assert proposal.proposal_id in engine.state.proposals
    
    def test_create_proposal_emergency_paused(self):
        """Test that proposal creation fails when governance is paused."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        # Pause governance
        engine.emergency_pause("Test emergency", 100)
        
        with pytest.raises(Exception):  # GovernanceError
            engine.create_proposal(
                proposer_address="0x123",
                title="Test Proposal",
                description="This is a test proposal",
                proposal_type=ProposalType.PARAMETER_CHANGE
            )
    
    def test_create_proposal_validation_errors(self):
        """Test proposal creation validation errors."""
        config = GovernanceConfig(min_proposal_title_length=20)
        engine = GovernanceEngine(config)
        
        # Test title too short
        with pytest.raises(ValidationError):
            engine.create_proposal(
                proposer_address="0x123",
                title="Short",
                description="This is a test proposal",
                proposal_type=ProposalType.PARAMETER_CHANGE
            )
        
        # Test description too long
        with pytest.raises(ValidationError):
            engine.create_proposal(
                proposer_address="0x123",
                title="This is a sufficiently long title",
                description="x" * (config.max_proposal_description_length + 1),
                proposal_type=ProposalType.PARAMETER_CHANGE
            )
    
    def test_cast_vote(self):
        """Test casting a vote through governance engine."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        # Create proposal
        proposal = engine.create_proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        # Activate proposal
        engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        # Create voting power
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        
        # Cast vote
        vote = engine.cast_vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        assert vote.proposal_id == proposal.proposal_id
        assert vote.voter_address == "0x456"
        assert vote.choice == VoteChoice.FOR
        assert vote.voting_power == voting_power
        assert vote.signature == "0xabc123"
        
        # Check that vote was added to proposal
        assert len(proposal.votes) == 1
        assert proposal.votes[0] == vote
    
    def test_cast_vote_invalid_proposal(self):
        """Test casting vote on invalid proposal."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        
        with pytest.raises(ValidationError):
            engine.cast_vote(
                proposal_id="nonexistent",
                voter_address="0x456",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature="0xabc123"
            )
    
    def test_cast_vote_inactive_proposal(self):
        """Test casting vote on inactive proposal."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        # Create proposal but don't activate it
        proposal = engine.create_proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="This is a test proposal",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        
        with pytest.raises(ValidationError):
            engine.cast_vote(
                proposal_id=proposal.proposal_id,
                voter_address="0x456",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature="0xabc123"
            )
    
    def test_emergency_pause_resume(self):
        """Test emergency pause and resume functionality."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        # Initially not paused
        assert not engine.state.emergency_paused
        
        # Pause governance
        engine.emergency_pause("Test emergency", 100)
        
        assert engine.state.emergency_paused
        assert engine.state.emergency_pause_reason == "Test emergency"
        assert engine.state.emergency_pause_block == 100
        
        # Resume governance
        engine.emergency_resume()
        
        assert not engine.state.emergency_paused
        assert engine.state.emergency_pause_reason is None
        assert engine.state.emergency_pause_block is None
