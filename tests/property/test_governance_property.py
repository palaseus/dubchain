"""
Property-based tests for governance system using Hypothesis.

This module tests governance invariants and properties using property-based
testing to ensure correctness under various conditions.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, Bundle
from unittest.mock import Mock
import time
import math

from dubchain.governance.core import (
    GovernanceEngine,
    GovernanceConfig,
    Proposal,
    ProposalStatus,
    ProposalType,
    Vote,
    VoteChoice,
    VotingPower,
)
from dubchain.governance.strategies import (
    TokenWeightedStrategy,
    QuadraticVotingStrategy,
    ConvictionVotingStrategy,
    SnapshotVotingStrategy,
)
from dubchain.governance.delegation import DelegationManager, Delegation
from dubchain.governance.security import SecurityManager


class TestGovernanceProperties:
    """Property-based tests for governance system."""
    
    @given(
        st.integers(min_value=1, max_value=1000000),
        st.integers(min_value=0, max_value=1000000)
    )
    def test_voting_power_total_always_positive(self, power, delegated_power):
        """Test that total voting power is always positive or zero."""
        voting_power = VotingPower(
            voter_address="0x123",
            power=power,
            token_balance=power,
            delegated_power=delegated_power
        )
        
        assert voting_power.total_power() >= 0
    
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100),
        st.integers(min_value=1, max_value=1000000)
    )
    def test_proposal_creation_validation(self, title, description, quorum_threshold):
        """Test that proposal creation works with various inputs."""
        # The Proposal class is a dataclass without validation, so we just test creation
        proposal = Proposal(
            proposer_address="0x123",
            title=title,
            description=description,
            quorum_threshold=quorum_threshold
        )
        assert proposal.title == title
        assert proposal.description == description
        assert proposal.quorum_threshold == quorum_threshold
    
    @given(
        st.integers(min_value=1, max_value=1000000),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_proposal_approval_threshold_validation(self, quorum_threshold, approval_threshold):
        """Test that approval threshold validation works correctly."""
        if not (0 <= approval_threshold <= 1):
            with pytest.raises(ValueError):
                Proposal(
                    proposer_address="0x123",
                    title="Test Proposal",
                    description="Test description",
                    quorum_threshold=quorum_threshold,
                    approval_threshold=approval_threshold
                )
        else:
            proposal = Proposal(
                proposer_address="0x123",
                title="Test Proposal",
                description="Test description",
                quorum_threshold=quorum_threshold,
                approval_threshold=approval_threshold
            )
            assert proposal.approval_threshold == approval_threshold
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from([VoteChoice.FOR, VoteChoice.AGAINST, VoteChoice.ABSTAIN]),
                st.integers(min_value=1, max_value=10000)
            ),
            min_size=1,
            max_size=100
        )
    )
    def test_proposal_vote_summary_invariants(self, vote_data):
        """Test that vote summary maintains invariants."""
        proposal = Proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="Test description",
            quorum_threshold=1000,
            approval_threshold=0.5
        )
        
        # Add votes
        for i, (choice, power) in enumerate(vote_data):
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=power,
                token_balance=power
            )
            
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter_address=f"0x{i}",
                choice=choice,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            
            proposal.add_vote(vote)
        
        summary = proposal.get_vote_summary()
        
        # Invariants
        assert summary["total_voting_power"] >= 0
        assert summary["for_power"] >= 0
        assert summary["against_power"] >= 0
        assert summary["abstain_power"] >= 0
        assert summary["total_votes"] == len(vote_data)
        
        # Power conservation
        total_power = summary["for_power"] + summary["against_power"] + summary["abstain_power"]
        assert total_power == summary["total_voting_power"]
        
        # Approval percentage bounds
        assert 0 <= summary["approval_percentage"] <= 1
        
        # Quorum and approval logic
        if summary["total_voting_power"] >= proposal.quorum_threshold:
            assert summary["quorum_met"] is True
            if summary["approval_percentage"] >= proposal.approval_threshold:
                assert summary["approved"] is True
            else:
                assert summary["approved"] is False
        else:
            assert summary["quorum_met"] is False
            assert summary["approved"] is False
    
    @given(
        st.integers(min_value=1, max_value=1000000),
        st.integers(min_value=0, max_value=1000000)
    )
    def test_token_weighted_strategy_properties(self, token_balance, delegated_power):
        """Test token-weighted strategy properties."""
        strategy = TokenWeightedStrategy()
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=token_balance,
            delegated_power=delegated_power
        )
        
        # Properties
        assert power.power >= 0
        assert power.token_balance == token_balance
        assert power.delegated_power == delegated_power
        assert power.total_power() == power.power + delegated_power
        
        # For token-weighted: power should equal token_balance (if above minimum)
        if token_balance >= strategy.min_token_balance:
            assert power.power == token_balance
        else:
            assert power.power == 0
    
    @given(
        st.integers(min_value=1, max_value=1000000),
        st.integers(min_value=0, max_value=1000000)
    )
    def test_quadratic_voting_strategy_properties(self, token_balance, delegated_power):
        """Test quadratic voting strategy properties."""
        strategy = QuadraticVotingStrategy()
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=token_balance,
            delegated_power=delegated_power
        )
        
        # Properties
        assert power.power >= 0
        assert power.token_balance == token_balance
        assert power.delegated_power == delegated_power
        assert power.total_power() == power.power + delegated_power
        
        # For quadratic voting: power should be sqrt(token_balance) (if above minimum)
        if token_balance >= strategy.min_token_balance:
            expected_power = int(math.sqrt(token_balance))
            assert power.power == expected_power
        else:
            assert power.power == 0
    
    @given(
        st.integers(min_value=1, max_value=1000),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_conviction_voting_strategy_properties(self, token_balance, conviction):
        """Test conviction voting strategy properties."""
        strategy = ConvictionVotingStrategy()
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=token_balance,
            delegated_power=0,
            conviction=conviction
        )
        
        # Properties
        assert power.power >= 0
        assert power.token_balance == token_balance
        
        # For conviction voting: power should be token_balance * (1 + conviction)
        expected_power = int(token_balance * (1 + conviction))
        assert power.power == expected_power
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.integers(min_value=0, max_value=1000000),
            min_size=1,
            max_size=100
        )
    )
    def test_snapshot_voting_merkle_tree_properties(self, balances):
        """Test snapshot voting Merkle tree properties."""
        strategy = SnapshotVotingStrategy()
        
        # Create snapshot
        merkle_root = strategy.create_snapshot(1000, balances)
        
        # Properties
        assert merkle_root is not None
        assert len(merkle_root) > 0  # Should be a valid hash
        
        # Test Merkle proof for each address
        for address, balance in balances.items():
            proof = strategy.generate_merkle_proof(1000, address)
            
            assert proof is not None
            assert proof["address"] == address
            assert proof["balance"] == balance
            assert proof["merkle_root"] == merkle_root
            
            # Verify proof
            assert strategy.verify_merkle_proof(proof) is True
    
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=10),
                st.integers(min_value=1, max_value=10000)
            ).filter(lambda x: x[0] != x[1]),  # Filter out self-delegation
            min_size=1,
            max_size=50
        )
    )
    def test_delegation_manager_properties(self, delegation_data):
        """Test delegation manager properties."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        created_delegations = []
        
        for delegator, delegatee, power in delegation_data:
            try:
                delegation = manager.create_delegation(
                    delegator_address=delegator,
                    delegatee_address=delegatee,
                    delegation_power=power
                )
                created_delegations.append(delegation)
            except ValueError:
                # Skip invalid delegations (e.g., self-delegation)
                continue
        
        # Properties
        total_delegated_power = sum(d.delegation_power for d in created_delegations)
        assert total_delegated_power >= 0
        
        # Check that all created delegations are valid
        for delegation in created_delegations:
            assert delegation.is_valid()
            assert delegation.delegation_power > 0
            assert delegation.delegator_address != delegation.delegatee_address
        
        # Check delegation statistics
        stats = manager.get_delegation_statistics()
        assert stats["total_delegations"] == len(created_delegations)
        assert stats["active_delegations"] == len(created_delegations)
        assert stats["total_delegated_power"] == total_delegated_power
    
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.sampled_from(["low", "medium", "high", "critical"]),
                st.text(min_size=1, max_size=100)
            ),
            min_size=1,
            max_size=100
        )
    )
    def test_security_manager_properties(self, alert_data):
        """Test security manager properties."""
        manager = SecurityManager()
        
        # Add alerts
        for alert_id, severity, description in alert_data:
            alert = manager.analyze_vote(
                vote=Mock(),
                proposal=Mock(),
                context={}
            )
            # Note: This is a simplified test since we can't easily create valid votes
        
        # Properties
        stats = manager.get_security_statistics()
        assert stats["total_alerts"] >= 0
        assert stats["blocked_addresses"] >= 0
        assert stats["suspicious_addresses"] >= 0
        assert stats["active_detectors"] == 4  # Should always have 4 detectors


class GovernanceStateMachine(RuleBasedStateMachine):
    """State machine for testing governance system properties."""
    
    def __init__(self):
        super().__init__()
        self.config = GovernanceConfig(
            default_quorum_threshold=1000,
            default_approval_threshold=0.5,
            default_voting_period=100,
            default_execution_delay=10
        )
        self.engine = GovernanceEngine(self.config)
        self.engine.voting_strategy = TokenWeightedStrategy()
        self.engine.delegation_manager = DelegationManager(self.config)
        self.engine.security_manager = SecurityManager()
        
        self.proposals = Bundle("proposals")
        self.voters = Bundle("voters")
    
    @rule(target=Bundle("proposals"), proposer=st.text(min_size=1, max_size=10))
    def create_proposal(self, proposer):
        """Create a new proposal."""
        try:
            proposal = self.engine.create_proposal(
                proposer_address=proposer,
                title="Test Proposal",
                description="Test description",
                proposal_type=ProposalType.PARAMETER_CHANGE
            )
            return proposal
        except Exception:
            return None
    
    @rule(proposal=Bundle("proposals"))
    def activate_proposal(self, proposal):
        """Activate a proposal."""
        if proposal is not None:
            self.engine.state.update_proposal_status(
                proposal.proposal_id, ProposalStatus.ACTIVE
            )
    
    @rule(
        target=Bundle("voters"),
        voter=st.text(min_size=1, max_size=10),
        power=st.integers(min_value=1, max_value=10000)
    )
    def create_voter(self, voter, power):
        """Create a voter with voting power."""
        return (voter, power)
    
    @rule(
        proposal=Bundle("proposals"),
        voter=Bundle("voters"),
        choice=st.sampled_from([VoteChoice.FOR, VoteChoice.AGAINST, VoteChoice.ABSTAIN])
    )
    def cast_vote(self, proposal, voter, choice):
        """Cast a vote on a proposal."""
        if proposal is not None and voter is not None:
            voter_address, power = voter
            
            voting_power = VotingPower(
                voter_address=voter_address,
                power=power,
                token_balance=power
            )
            
            try:
                self.engine.cast_vote(
                    proposal_id=proposal.proposal_id,
                    voter_address=voter_address,
                    choice=choice,
                    voting_power=voting_power,
                    signature=f"0x{voter_address}"
                )
            except Exception:
                pass  # Ignore validation errors
    
    @rule(proposal=Bundle("proposals"))
    def queue_proposal(self, proposal):
        """Queue a proposal for execution."""
        if proposal is not None:
            self.engine.state.update_proposal_status(
                proposal.proposal_id, ProposalStatus.QUEUED
            )
    
    @rule(proposal=Bundle("proposals"))
    def execute_proposal(self, proposal):
        """Execute a proposal."""
        if proposal is not None and hasattr(self.engine, 'execution_engine'):
            try:
                self.engine.execution_engine.execute_proposal(
                    proposal, current_block=100
                )
            except Exception:
                pass  # Ignore execution errors
    
    @invariant()
    def proposal_count_never_negative(self):
        """Proposal count should never be negative."""
        assert self.engine.state.total_proposals >= 0
        assert self.engine.state.executed_proposals >= 0
        assert self.engine.state.failed_proposals >= 0
    
    @invariant()
    def proposal_status_consistency(self):
        """Proposal status should be consistent."""
        for proposal in self.engine.state.proposals.values():
            if proposal.status == ProposalStatus.ACTIVE:
                assert proposal.proposal_id in self.engine.state.active_proposals
            elif proposal.status == ProposalStatus.QUEUED:
                assert proposal.proposal_id in self.engine.state.queued_proposals
    
    @invariant()
    def vote_power_conservation(self):
        """Voting power should be conserved."""
        for proposal in self.engine.state.proposals.values():
            summary = proposal.get_vote_summary()
            total_power = (
                summary["for_power"] + 
                summary["against_power"] + 
                summary["abstain_power"]
            )
            assert total_power == summary["total_voting_power"]
    
    @invariant()
    def approval_logic_consistency(self):
        """Approval logic should be consistent."""
        for proposal in self.engine.state.proposals.values():
            summary = proposal.get_vote_summary()
            
            if summary["quorum_met"] and summary["approval_percentage"] >= proposal.approval_threshold:
                assert summary["approved"] is True
            else:
                assert summary["approved"] is False


# Register the state machine test
TestGovernanceStateMachine = GovernanceStateMachine.TestCase
