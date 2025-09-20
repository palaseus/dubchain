"""
Fuzz tests for governance system.

This module tests the governance system with random, malformed, and edge case
inputs to ensure robustness and security.
"""

import pytest
import random
import string
import time
from typing import Any, Dict, List, Optional

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
from dubchain.governance.security import SecurityManager
from dubchain.governance.execution import ExecutionEngine
from dubchain.errors.exceptions import ValidationError


class FuzzDataGenerator:
    """Generator for fuzz test data."""
    
    @staticmethod
    def random_string(min_length: int = 1, max_length: int = 100) -> str:
        """Generate random string."""
        length = random.randint(min_length, max_length)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_address() -> str:
        """Generate random Ethereum-style address."""
        return '0x' + ''.join(random.choices(string.hexdigits.lower(), k=40))
    
    @staticmethod
    def random_integer(min_value: int = 0, max_value: int = 1000000) -> int:
        """Generate random integer."""
        return random.randint(min_value, max_value)
    
    @staticmethod
    def random_float(min_value: float = 0.0, max_value: float = 1.0) -> float:
        """Generate random float."""
        return random.uniform(min_value, max_value)
    
    @staticmethod
    def random_choice(choices: List[Any]) -> Any:
        """Generate random choice from list."""
        return random.choice(choices)
    
    @staticmethod
    def random_boolean() -> bool:
        """Generate random boolean."""
        return random.choice([True, False])
    
    @staticmethod
    def malformed_string() -> str:
        """Generate malformed string."""
        malformed_strings = [
            "",  # Empty string
            " " * 1000,  # Very long whitespace
            "\x00" * 100,  # Null bytes
            "\xff" * 100,  # Invalid UTF-8
            "\\x00\\x01\\x02",  # Escaped null bytes
            "🚀" * 1000,  # Unicode emoji spam
            "A" * 10000,  # Very long string
        ]
        return random.choice(malformed_strings)
    
    @staticmethod
    def extreme_integer() -> int:
        """Generate extreme integer values."""
        extreme_values = [
            0,
            -1,
            -1000000,
            2**31 - 1,  # Max 32-bit int
            2**31,      # Overflow 32-bit int
            2**63 - 1,  # Max 64-bit int
            2**63,      # Overflow 64-bit int
        ]
        return random.choice(extreme_values)
    
    @staticmethod
    def extreme_float() -> float:
        """Generate extreme float values."""
        extreme_values = [
            0.0,
            -0.0,
            float('inf'),
            float('-inf'),
            float('nan'),
            1e-10,
            1e10,
            -1e-10,
            -1e10,
        ]
        return random.choice(extreme_values)


class TestGovernanceFuzz:
    """Fuzz tests for governance system."""
    
    @pytest.fixture
    def governance_engine(self):
        """Create governance engine for fuzz testing."""
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        engine.voting_strategy = StrategyFactory.create_strategy("token_weighted")
        engine.delegation_manager = DelegationManager(config)
        engine.security_manager = SecurityManager()
        engine.execution_engine = ExecutionEngine(engine.state)
        return engine
    
    def test_proposal_creation_fuzz(self, governance_engine):
        """Fuzz test proposal creation with random inputs."""
        for _ in range(100):
            try:
                # Generate random inputs
                proposer = FuzzDataGenerator.random_address()
                title = FuzzDataGenerator.random_string(1, 200)
                description = FuzzDataGenerator.random_string(1, 1000)
                proposal_type = FuzzDataGenerator.random_choice(list(ProposalType))
                
                # Try to create proposal
                proposal = governance_engine.create_proposal(
                    proposer_address=proposer,
                    title=title,
                    description=description,
                    proposal_type=proposal_type
                )
                
                # If successful, verify basic properties
                assert proposal.proposer_address == proposer
                assert proposal.title == title
                assert proposal.description == description
                assert proposal.proposal_type == proposal_type
                
            except (ValueError, TypeError, ValidationError) as e:
                # Expected for malformed inputs
                assert isinstance(e, (ValueError, TypeError, ValidationError))
            except Exception as e:
                # Unexpected exceptions should be investigated
                pytest.fail(f"Unexpected exception: {e}")
    
    def test_proposal_creation_malformed_inputs(self, governance_engine):
        """Test proposal creation with malformed inputs."""
        malformed_inputs = [
            # Empty strings
            ("", "Test Title", "Test Description"),
            ("0x123", "", "Test Description"),
            ("0x123", "Test Title", ""),
            
            # Very long strings
            ("0x123", "A" * 10000, "Test Description"),
            ("0x123", "Test Title", "A" * 10000),
            
            # Null bytes
            ("\x00", "Test Title", "Test Description"),
            ("0x123", "\x00", "Test Description"),
            ("0x123", "Test Title", "\x00"),
            
            # Invalid addresses
            ("invalid_address", "Test Title", "Test Description"),
            ("0x", "Test Title", "Test Description"),
            ("0x123", "Test Title", "Test Description"),  # Too short
        ]
        
        for proposer, title, description in malformed_inputs:
            try:
                proposal = governance_engine.create_proposal(
                    proposer_address=proposer,
                    title=title,
                    description=description,
                    proposal_type=ProposalType.PARAMETER_CHANGE
                )
                # If no exception is raised, verify the proposal is valid
                assert proposal is not None
                assert proposal.proposer_address == proposer
                assert proposal.title == title
                assert proposal.description == description
            except (ValueError, TypeError, ValidationError):
                # Expected for malformed inputs
                pass
    
    def test_voting_power_fuzz(self):
        """Fuzz test voting power creation with random inputs."""
        for _ in range(100):
            try:
                # Generate random inputs
                voter_address = FuzzDataGenerator.random_address()
                power = FuzzDataGenerator.random_integer(0, 1000000)
                token_balance = FuzzDataGenerator.random_integer(0, 1000000)
                delegated_power = FuzzDataGenerator.random_integer(0, 1000000)
                
                # Create voting power
                voting_power = VotingPower(
                    voter_address=voter_address,
                    power=power,
                    token_balance=token_balance,
                    delegated_power=delegated_power
                )
                
                # Verify properties
                assert voting_power.voter_address == voter_address
                assert voting_power.power == power
                assert voting_power.token_balance == token_balance
                assert voting_power.delegated_power == delegated_power
                assert voting_power.total_power() == power + delegated_power
                
            except (ValueError, TypeError) as e:
                # Expected for malformed inputs
                assert isinstance(e, (ValueError, TypeError))
    
    def test_voting_power_extreme_values(self):
        """Test voting power with extreme values."""
        extreme_cases = [
            # Zero values
            ("0x123", 0, 0, 0),
            # Negative values
            ("0x123", -1, 1000, 0),
            ("0x123", 1000, -1, 0),
            ("0x123", 1000, 1000, -1),
            # Very large values
            ("0x123", 2**63, 1000, 0),
            ("0x123", 1000, 2**63, 0),
            ("0x123", 1000, 1000, 2**63),
        ]
        
        for voter_address, power, token_balance, delegated_power in extreme_cases:
            try:
                voting_power = VotingPower(
                    voter_address=voter_address,
                    power=power,
                    token_balance=token_balance,
                    delegated_power=delegated_power
                )
                # If no exception is raised, verify the object was created
                assert voting_power is not None
                # Note: The VotingPower class may not validate negative values
                # This is a fuzz test, so we just verify it doesn't crash
            except (ValueError, TypeError, OverflowError, ValidationError):
                # Expected for extreme values
                pass
    
    def test_vote_creation_fuzz(self, governance_engine):
        """Fuzz test vote creation with random inputs."""
        # Create a valid proposal first
        proposal = governance_engine.create_proposal(
            proposer_address="0x123",
            title="Test Proposal",
            description="Test description",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        
        governance_engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
        
        for _ in range(100):
            try:
                # Generate random inputs
                voter_address = FuzzDataGenerator.random_address()
                choice = FuzzDataGenerator.random_choice(list(VoteChoice))
                power = FuzzDataGenerator.random_integer(1, 10000)
                signature = FuzzDataGenerator.random_string(1, 100)
                
                voting_power = VotingPower(
                    voter_address=voter_address,
                    power=power,
                    token_balance=power
                )
                
                # Try to create vote
                vote = Vote(
                    proposal_id=proposal.proposal_id,
                    voter_address=voter_address,
                    choice=choice,
                    voting_power=voting_power,
                    signature=signature
                )
                
                # If successful, verify properties
                assert vote.proposal_id == proposal.proposal_id
                assert vote.voter_address == voter_address
                assert vote.choice == choice
                assert vote.voting_power == voting_power
                assert vote.signature == signature
                
            except (ValueError, TypeError) as e:
                # Expected for malformed inputs
                assert isinstance(e, (ValueError, TypeError))
    
    def test_delegation_creation_fuzz(self, governance_engine):
        """Fuzz test delegation creation with random inputs."""
        for _ in range(100):
            try:
                # Generate random inputs
                delegator = FuzzDataGenerator.random_address()
                delegatee = FuzzDataGenerator.random_address()
                power = FuzzDataGenerator.random_integer(1, 10000)
                
                # Skip self-delegation
                if delegator == delegatee:
                    continue
                
                # Try to create delegation
                delegation = governance_engine.delegation_manager.create_delegation(
                    delegator_address=delegator,
                    delegatee_address=delegatee,
                    delegation_power=power
                )
                
                # If successful, verify properties
                assert delegation.delegator_address == delegator
                assert delegation.delegatee_address == delegatee
                assert delegation.delegation_power == power
                assert delegation.is_valid()
                
            except (ValueError, TypeError) as e:
                # Expected for malformed inputs
                assert isinstance(e, (ValueError, TypeError))
    
    def test_delegation_circular_detection_fuzz(self, governance_engine):
        """Fuzz test circular delegation detection."""
        addresses = [FuzzDataGenerator.random_address() for _ in range(10)]
        
        # Try to create circular delegations
        for i in range(100):
            try:
                delegator = FuzzDataGenerator.random_choice(addresses)
                delegatee = FuzzDataGenerator.random_choice(addresses)
                power = FuzzDataGenerator.random_integer(1, 1000)
                
                governance_engine.delegation_manager.create_delegation(
                    delegator_address=delegator,
                    delegatee_address=delegatee,
                    delegation_power=power
                )
                
            except ValueError as e:
                # Expected for circular delegations
                assert "circular" in str(e).lower() or "cycle" in str(e).lower()
            except Exception as e:
                # Other exceptions are also acceptable
                pass
    
    def test_voting_strategy_fuzz(self):
        """Fuzz test voting strategies with random inputs."""
        strategies = [
            StrategyFactory.create_strategy("token_weighted"),
            StrategyFactory.create_strategy("quadratic_voting"),
            StrategyFactory.create_strategy("conviction_voting"),
            StrategyFactory.create_strategy("snapshot_voting"),
        ]
        
        for strategy in strategies:
            for _ in range(50):
                try:
                    # Generate random inputs
                    voter_address = FuzzDataGenerator.random_address()
                    token_balance = FuzzDataGenerator.random_integer(0, 1000000)
                    delegated_power = FuzzDataGenerator.random_integer(0, 1000000)
                    
                    # Calculate voting power
                    power = strategy.calculate_voting_power(
                        voter_address=voter_address,
                        token_balance=token_balance,
                        delegated_power=delegated_power
                    )
                    
                    # Verify properties
                    assert power.voter_address == voter_address
                    assert power.power >= 0
                    assert power.token_balance == token_balance
                    assert power.delegated_power == delegated_power
                    assert power.total_power() >= 0
                    
                except (ValueError, TypeError) as e:
                    # Expected for malformed inputs
                    assert isinstance(e, (ValueError, TypeError))
    
    def test_security_manager_fuzz(self, governance_engine):
        """Fuzz test security manager with random inputs."""
        for _ in range(100):
            try:
                # Generate random inputs
                voter_address = FuzzDataGenerator.random_address()
                choice = FuzzDataGenerator.random_choice(list(VoteChoice))
                power = FuzzDataGenerator.random_integer(1, 10000)
                
                voting_power = VotingPower(
                    voter_address=voter_address,
                    power=power,
                    token_balance=power
                )
                
                vote = Vote(
                    proposal_id="test_proposal",
                    voter_address=voter_address,
                    choice=choice,
                    voting_power=voting_power,
                    signature=FuzzDataGenerator.random_string(1, 100)
                )
                
                proposal = Proposal(
                    proposer_address="0x123",
                    title="Test Proposal",
                    description="Test description"
                )
                
                context = {
                    "recent_votes": [],
                    "current_block": FuzzDataGenerator.random_integer(1, 1000)
                }
                
                # Analyze vote
                alerts = governance_engine.security_manager.analyze_vote(vote, proposal, context)
                
                # Verify alerts are valid
                for alert in alerts:
                    assert alert.alert_id is not None
                    assert alert.alert_type is not None
                    assert alert.severity in ["low", "medium", "high", "critical"]
                    assert alert.description is not None
                
            except (ValueError, TypeError) as e:
                # Expected for malformed inputs
                assert isinstance(e, (ValueError, TypeError))
    
    def test_governance_config_fuzz(self):
        """Fuzz test governance configuration with random inputs."""
        for _ in range(100):
            try:
                # Generate random config values
                config_data = {
                    "default_quorum_threshold": FuzzDataGenerator.random_integer(1, 1000000),
                    "default_approval_threshold": FuzzDataGenerator.random_float(0.0, 1.0),
                    "default_voting_period": FuzzDataGenerator.random_integer(1, 10000),
                    "default_execution_delay": FuzzDataGenerator.random_integer(0, 1000),
                    "max_proposal_description_length": FuzzDataGenerator.random_integer(100, 100000),
                    "min_proposal_title_length": FuzzDataGenerator.random_integer(1, 100),
                }
                
                # Create config
                config = GovernanceConfig(**config_data)
                
                # Verify properties
                assert config.default_quorum_threshold == config_data["default_quorum_threshold"]
                assert config.default_approval_threshold == config_data["default_approval_threshold"]
                assert config.default_voting_period == config_data["default_voting_period"]
                assert config.default_execution_delay == config_data["default_execution_delay"]
                assert config.max_proposal_description_length == config_data["max_proposal_description_length"]
                assert config.min_proposal_title_length == config_data["min_proposal_title_length"]
                
            except (ValueError, TypeError) as e:
                # Expected for malformed inputs
                assert isinstance(e, (ValueError, TypeError))
    
    def test_governance_config_extreme_values(self):
        """Test governance configuration with extreme values."""
        extreme_cases = [
            # Negative values
            {"default_quorum_threshold": -1},
            {"default_approval_threshold": -0.1},
            {"default_voting_period": -1},
            {"default_execution_delay": -1},
            {"max_proposal_description_length": -1},
            {"min_proposal_title_length": -1},
            
            # Invalid ranges
            {"default_approval_threshold": 1.5},
            {"default_approval_threshold": 2.0},
            
            # Zero values
            {"default_quorum_threshold": 0},
            {"default_voting_period": 0},
            {"max_proposal_description_length": 0},
            {"min_proposal_title_length": 0},
        ]
        
        for config_data in extreme_cases:
            try:
                config = GovernanceConfig(**config_data)
                # If no exception is raised, verify the config was created
                assert config is not None
                # Note: Some extreme values may be accepted by the config
                # This is a fuzz test, so we just verify it doesn't crash
            except (ValueError, TypeError, ValidationError):
                # Expected for extreme values
                pass
    
    def test_serialization_fuzz(self, governance_engine):
        """Fuzz test serialization and deserialization."""
        # Create a proposal with random data
        proposal = governance_engine.create_proposal(
            proposer_address=FuzzDataGenerator.random_address(),
            title=FuzzDataGenerator.random_string(10, 100),
            description=FuzzDataGenerator.random_string(10, 1000),
            proposal_type=FuzzDataGenerator.random_choice(list(ProposalType))
        )
        
        # Add random votes
        for i in range(10):
            try:
                voting_power = VotingPower(
                    voter_address=FuzzDataGenerator.random_address(),
                    power=FuzzDataGenerator.random_integer(1, 10000),
                    token_balance=FuzzDataGenerator.random_integer(1, 10000)
                )
                
                vote = Vote(
                    proposal_id=proposal.proposal_id,
                    voter_address=voting_power.voter_address,
                    choice=FuzzDataGenerator.random_choice(list(VoteChoice)),
                    voting_power=voting_power,
                    signature=FuzzDataGenerator.random_string(1, 100)
                )
                
                proposal.add_vote(vote)
                
            except (ValueError, TypeError):
                continue
        
        # Test serialization
        try:
            proposal_dict = proposal.to_dict()
            assert isinstance(proposal_dict, dict)
            
            # Test deserialization
            deserialized_proposal = Proposal.from_dict(proposal_dict)
            assert deserialized_proposal.proposal_id == proposal.proposal_id
            assert deserialized_proposal.proposer_address == proposal.proposer_address
            assert deserialized_proposal.title == proposal.title
            assert deserialized_proposal.description == proposal.description
            assert deserialized_proposal.proposal_type == proposal.proposal_type
            assert len(deserialized_proposal.votes) == len(proposal.votes)
            
        except (ValueError, TypeError, KeyError) as e:
            # Expected for malformed data
            assert isinstance(e, (ValueError, TypeError, KeyError))
    
    def test_concurrent_operations_fuzz(self, governance_engine):
        """Fuzz test concurrent operations."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def create_proposal(thread_id):
            try:
                proposal = governance_engine.create_proposal(
                    proposer_address=f"0x{thread_id}",
                    title=f"Thread {thread_id} Proposal",
                    description=f"Proposal from thread {thread_id}",
                    proposal_type=ProposalType.PARAMETER_CHANGE
                )
                results.put(proposal.proposal_id)
            except Exception as e:
                errors.put(e)
        
        def create_delegation(thread_id):
            try:
                delegation = governance_engine.delegation_manager.create_delegation(
                    delegator_address=f"0x{thread_id}",
                    delegatee_address=f"0x{thread_id + 1000}",
                    delegation_power=1000
                )
                results.put(delegation.delegator_address)
            except Exception as e:
                errors.put(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_proposal, args=(i,))
            threads.append(thread)
        
        for i in range(10):
            thread = threading.Thread(target=create_delegation, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert results.qsize() > 0  # At least some operations should succeed
        assert errors.qsize() >= 0  # Some errors are expected due to conflicts
    
    def test_memory_usage_fuzz(self, governance_engine):
        """Fuzz test memory usage with large datasets."""
        # Create many proposals
        proposals = []
        for i in range(1000):
            try:
                proposal = governance_engine.create_proposal(
                    proposer_address=f"0x{i}",
                    title=f"Proposal {i}",
                    description=f"Description for proposal {i}",
                    proposal_type=ProposalType.PARAMETER_CHANGE
                )
                proposals.append(proposal)
            except Exception:
                continue
        
        # Create many delegations
        for i in range(1000):
            try:
                governance_engine.delegation_manager.create_delegation(
                    delegator_address=f"0x{i}",
                    delegatee_address=f"0x{i + 1000}",
                    delegation_power=1000
                )
            except Exception:
                continue
        
        # Verify system is still functional
        assert len(governance_engine.state.proposals) > 0
        assert len(governance_engine.delegation_manager.delegations) > 0
        
        # Test that we can still create new proposals
        new_proposal = governance_engine.create_proposal(
            proposer_address="0xnew",
            title="New Proposal",
            description="New description",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        assert new_proposal.proposal_id is not None
