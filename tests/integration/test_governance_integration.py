"""
Integration tests for governance system.

This module tests the integration between different governance components
including proposal lifecycle, voting, delegation, execution, and security.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
from unittest.mock import Mock, patch

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
from dubchain.governance.strategies import StrategyFactory
from dubchain.governance.delegation import DelegationManager
from dubchain.governance.execution import ExecutionEngine
from dubchain.governance.security import SecurityManager
from dubchain.governance.treasury import TreasuryManager, TreasuryOperationType, TreasuryStatus
from dubchain.governance.observability import GovernanceEvents, EventType
from dubchain.errors.exceptions import ValidationError


class TestGovernanceIntegration:
    """Test integrated governance system functionality."""
    
    @pytest.fixture
    def governance_config(self):
        """Create governance configuration for tests."""
        return GovernanceConfig(
            default_quorum_threshold=1000,
            default_approval_threshold=0.5,
            default_voting_period=100,
            default_execution_delay=10,
            max_proposal_description_length=1000,
            min_proposal_title_length=10
        )
    
    @pytest.fixture
    def governance_engine(self, governance_config):
        """Create governance engine with all components."""
        engine = GovernanceEngine(governance_config)
        
        # Initialize components
        engine.voting_strategy = StrategyFactory.create_strategy("token_weighted")
        engine.delegation_manager = DelegationManager(governance_config)
        engine.execution_engine = ExecutionEngine(engine.state)
        engine.security_manager = SecurityManager()
        engine.treasury_manager = TreasuryManager()
        engine.observability = GovernanceEvents()
        
        return engine
    
    def test_complete_proposal_lifecycle(self, governance_engine):
        """Test complete proposal lifecycle from creation to execution."""
        # Create proposal
        proposal = governance_engine.create_proposal(
            proposer_address="0x123",
            title="Test Parameter Change Proposal",
            description="This proposal changes a system parameter",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            execution_data={"parameter_name": "block_size", "new_value": 2000000}
        )
        
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.proposal_id in governance_engine.state.proposals
        
        # Activate proposal
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        # Create voters with sufficient voting power
        voters = []
        for i in range(5):
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=500,  # Total: 2500, above quorum of 1000
                token_balance=500
            )
            voters.append(voting_power)
        
        # Cast votes (4 FOR, 1 AGAINST)
        for i, (voter, choice) in enumerate(zip(voters, [VoteChoice.FOR] * 4 + [VoteChoice.AGAINST])):
            vote = governance_engine.cast_vote(
                proposal_id=proposal.proposal_id,
                voter_address=voter.voter_address,
                choice=choice,
                voting_power=voter,
                signature=f"0x{i}"
            )
            assert vote.proposal_id == proposal.proposal_id
        
        # Check vote summary
        summary = proposal.get_vote_summary()
        assert summary["total_voting_power"] == 2500
        assert summary["for_power"] == 2000
        assert summary["against_power"] == 500
        assert summary["quorum_met"] is True
        assert summary["approved"] is True
        
        # Queue proposal for execution
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.QUEUED)
        
        # Execute proposal
        result = governance_engine.execution_engine.execute_proposal(
            proposal, current_block=100
        )
        
        assert result.is_successful()
        assert proposal.status == ProposalStatus.EXECUTED
    
    def test_delegation_integration(self, governance_engine):
        """Test delegation integration with voting."""
        # Create delegation
        delegation = governance_engine.delegation_manager.create_delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        assert delegation.delegator_address == "0x123"
        assert delegation.delegatee_address == "0x456"
        
        # Create proposal
        proposal = governance_engine.create_proposal(
            proposer_address="0x789",
            title="Test Delegation Proposal",
            description="This proposal tests delegation",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        # Create voting power with delegation
        voting_power = VotingPower(
            voter_address="0x456",  # Delegatee
            power=500,  # Own power
            token_balance=500,
            delegated_power=1000  # Delegated power
        )
        
        # Cast vote with delegated power
        vote = governance_engine.cast_vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0x456"
        )
        
        assert vote.voting_power.total_power() == 1500
        
        # Check that delegation is tracked
        delegated_power = governance_engine.delegation_manager.get_delegated_power("0x456", 100)
        assert delegated_power == 1000
    
    def test_security_integration(self, governance_engine):
        """Test security integration with voting."""
        # Create proposal
        proposal = governance_engine.create_proposal(
            proposer_address="0x123",
            title="Test Security Proposal",
            description="This proposal tests security",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        # Create suspicious voting power (flash loan pattern)
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000000,  # Very high power
            token_balance=1000000
        )
        
        # Add suspicious transaction to trigger vote buying detection
        governance_engine.security_manager.add_suspicious_transaction(
            "0x456",
            {
                "timestamp": time.time() - 1800,  # 30 minutes ago
                "amount": 1000000,
                "from": "0x999",
                "to": "0x456"
            }
        )
        
        # Analyze vote for security threats
        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0x456"
        )
        
        context = {"recent_votes": []}
        alerts = governance_engine.security_manager.analyze_vote(vote, proposal, context)
        
        # Should detect vote buying
        assert len(alerts) > 0
        vote_buying_alerts = [alert for alert in alerts if alert.alert_type == "vote_buying"]
        assert len(vote_buying_alerts) > 0
    
    def test_treasury_integration(self, governance_engine):
        """Test treasury integration with governance."""
        # Add treasury balance
        governance_engine.treasury_manager.add_treasury_balance(
            token_address="0xTOKEN",
            amount=1000000,
            token_symbol="TOKEN"
        )
        
        # Create treasury proposal
        treasury_proposal = governance_engine.treasury_manager.create_treasury_proposal(
            proposer_address="0x123",
            operation_type=TreasuryOperationType.SPENDING,
            recipient_address="0x456",
            amount=100000,
            token_address="0xTOKEN",
            description="Test treasury spending",
            justification="Testing treasury functionality"
        )
        
        assert treasury_proposal.amount == 100000
        assert treasury_proposal.recipient_address == "0x456"
        
        # Add multisig signers
        governance_engine.treasury_manager.add_multisig_signer("0x789")
        governance_engine.treasury_manager.add_multisig_signer("0xabc")
        governance_engine.treasury_manager.add_multisig_signer("0xdef")
        
        # Approve treasury proposal
        governance_engine.treasury_manager.approve_treasury_proposal(
            treasury_proposal.proposal_id,
            "0x789",
            "0xsig1"
        )
        
        governance_engine.treasury_manager.approve_treasury_proposal(
            treasury_proposal.proposal_id,
            "0xabc",
            "0xsig2"
        )
        
        governance_engine.treasury_manager.approve_treasury_proposal(
            treasury_proposal.proposal_id,
            "0xdef",
            "0xsig3"
        )
        
        assert treasury_proposal.is_multisig_approved()
        
        # Execute treasury proposal
        success = governance_engine.treasury_manager.execute_treasury_proposal(
            treasury_proposal.proposal_id,
            "0x789"
        )
        
        assert success
        assert treasury_proposal.status == TreasuryStatus.EXECUTED
        
        # Check treasury balance decreased
        balance = governance_engine.treasury_manager.get_treasury_balance("0xTOKEN")
        assert balance == 900000  # 1000000 - 100000
    
    def test_observability_integration(self, governance_engine):
        """Test observability integration with governance events."""
        # Create proposal
        proposal = governance_engine.create_proposal(
            proposer_address="0x123",
            title="Test Observability Proposal",
            description="This proposal tests observability",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        # Emit proposal created event
        event = governance_engine.observability.emit_event(
            event_type=EventType.PROPOSAL_CREATED,
            proposal_id=proposal.proposal_id,
            voter_address="0x123",
            metadata={"title": proposal.title}
        )
        
        assert event.event_type == EventType.PROPOSAL_CREATED
        assert event.proposal_id == proposal.proposal_id
        
        # Activate proposal
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        # Emit proposal activated event
        event = governance_engine.observability.emit_event(
            event_type=EventType.PROPOSAL_ACTIVATED,
            proposal_id=proposal.proposal_id
        )
        
        assert event.event_type == EventType.PROPOSAL_ACTIVATED
        
        # Cast vote
        voting_power = VotingPower(
            voter_address="0x456",
            power=1000,
            token_balance=1000
        )
        
        vote = governance_engine.cast_vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0x456"
        )
        
        # Emit vote cast event
        event = governance_engine.observability.emit_event(
            event_type=EventType.VOTE_CAST,
            proposal_id=proposal.proposal_id,
            voter_address="0x456",
            metadata={"choice": vote.choice.value}
        )
        
        assert event.event_type == EventType.VOTE_CAST
        assert event.voter_address == "0x456"
        
        # Check audit trail
        audit_trail = governance_engine.observability.get_audit_trail()
        assert len(audit_trail.events) == 3
        
        # Check proposal events
        proposal_events = audit_trail.get_proposal_events(proposal.proposal_id)
        assert len(proposal_events) == 3
        
        # Check voter events
        voter_events = audit_trail.get_voter_events("0x456")
        assert len(voter_events) == 1
    
    def test_emergency_pause_integration(self, governance_engine):
        """Test emergency pause integration."""
        # Create proposal
        proposal = governance_engine.create_proposal(
            proposer_address="0x123",
            title="Test Emergency Proposal",
            description="This proposal tests emergency functionality",
            proposal_type=ProposalType.EMERGENCY
        )
        
        # Pause governance
        governance_engine.emergency_pause("Security threat detected", 100)
        
        assert governance_engine.state.emergency_paused
        
        # Try to create new proposal (should fail)
        with pytest.raises(Exception):  # GovernanceError
            governance_engine.create_proposal(
                proposer_address="0x456",
                title="Test Proposal During Pause",
                description="This should fail",
                proposal_type=ProposalType.PARAMETER_CHANGE
            )
        
        # Try to cast vote (should fail)
        voting_power = VotingPower(
            voter_address="0x789",
            power=1000,
            token_balance=1000
        )
        
        with pytest.raises(Exception):  # GovernanceError
            governance_engine.cast_vote(
                proposal_id=proposal.proposal_id,
                voter_address="0x789",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature="0x789"
            )
        
        # Resume governance
        governance_engine.emergency_resume()
        
        assert not governance_engine.state.emergency_paused
        
        # Now should be able to create proposal
        new_proposal = governance_engine.create_proposal(
            proposer_address="0x456",
            title="Test Proposal After Resume",
            description="This should work now",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        assert new_proposal.proposer_address == "0x456"
    
    def test_multiple_voting_strategies_integration(self, governance_config):
        """Test integration with multiple voting strategies."""
        # Test token-weighted strategy
        engine1 = GovernanceEngine(governance_config)
        engine1.voting_strategy = StrategyFactory.create_strategy("token_weighted")
        
        proposal1 = engine1.create_proposal(
            proposer_address="0x123",
            title="Token Weighted Proposal",
            description="Test token weighted voting",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        # Test quadratic voting strategy
        engine2 = GovernanceEngine(governance_config)
        engine2.voting_strategy = StrategyFactory.create_strategy("quadratic_voting")
        
        proposal2 = engine2.create_proposal(
            proposer_address="0x123",
            title="Quadratic Voting Proposal",
            description="Test quadratic voting",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        # Create same voting power for both
        voting_power = VotingPower(
            voter_address="0x456",
            power=10000,  # 10000 tokens
            token_balance=10000
        )
        
        # Token-weighted: power = 10000
        power1 = engine1.voting_strategy.calculate_voting_power(
            "0x456", 10000, 0
        )
        assert power1.power == 10000
        
        # Quadratic: power = sqrt(10000) = 100
        power2 = engine2.voting_strategy.calculate_voting_power(
            "0x456", 10000, 0
        )
        assert power2.power == 100
    
    def test_delegation_chain_integration(self, governance_engine):
        """Test delegation chain integration."""
        # Create delegation chain: A -> B -> C
        delegation1 = governance_engine.delegation_manager.create_delegation(
            delegator_address="0xA",
            delegatee_address="0xB",
            delegation_power=1000
        )
        
        delegation2 = governance_engine.delegation_manager.create_delegation(
            delegator_address="0xB",
            delegatee_address="0xC",
            delegation_power=800  # Some decay
        )
        
        # Get delegation chain
        chain = governance_engine.delegation_manager.get_delegation_chain("0xA", "0xC")
        assert chain is not None
        assert chain.get_chain_length() == 3
        
        # Test that circular delegation is prevented
        with pytest.raises(ValidationError):
            governance_engine.delegation_manager.create_delegation(
                delegator_address="0xC",
                delegatee_address="0xA",  # Would create cycle
                delegation_power=500
            )
    
    def test_timelock_integration(self, governance_engine):
        """Test timelock integration with execution."""
        # Create proposal
        proposal = governance_engine.create_proposal(
            proposer_address="0x123",
            title="Test Timelock Proposal",
            description="This proposal tests timelock",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            execution_delay=50,  # 50 blocks delay
            execution_data={"parameter_name": "block_size", "new_value": 2000000}
        )
        
        # Activate and approve proposal
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        # Cast enough votes to approve
        for i in range(3):
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=500,
                token_balance=500
            )
            
            governance_engine.cast_vote(
                proposal_id=proposal.proposal_id,
                voter_address=f"0x{i}",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
        
        # Queue proposal
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.QUEUED)
        
        # Try to execute immediately (should fail due to timelock)
        with pytest.raises(ValidationError):
            governance_engine.execution_engine.execute_proposal(
                proposal, current_block=25  # Less than 50 block delay
            )
        
        # Queue proposal for execution
        entry = governance_engine.execution_engine.queue_proposal_for_execution(
            proposal, current_block=100
        )
        
        assert entry.execution_block == 150  # 100 + 50 delay
        
        # Check timelock status
        status = governance_engine.execution_engine.timelock_manager.get_timelock_status(
            proposal.proposal_id, current_block=100
        )
        
        assert status["blocks_remaining"] == 50
        assert not status["can_execute"]
        
        # Process timelock queue after delay
        results = governance_engine.execution_engine.process_timelock_queue(150)
        
        assert len(results) == 1
        assert results[0].is_successful()
        assert proposal.status == ProposalStatus.EXECUTED
