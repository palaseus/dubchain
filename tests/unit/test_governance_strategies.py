"""
Unit tests for governance voting strategies.

This module tests all voting strategies including token-weighted,
quadratic voting, conviction voting, and snapshot-based voting.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import math
from unittest.mock import Mock, patch

from dubchain.governance.strategies import (
    VotingStrategy,
    TokenWeightedStrategy,
    QuadraticVotingStrategy,
    ConvictionVotingStrategy,
    ConvictionVotingConfig,
    SnapshotVotingStrategy,
    SnapshotVotingConfig,
    StrategyFactory,
)
from dubchain.governance.core import Vote, VoteChoice, VotingPower, Proposal, ProposalType
from dubchain.errors.exceptions import ValidationError


class TestTokenWeightedStrategy:
    """Test TokenWeightedStrategy class."""
    
    def test_token_weighted_strategy_creation(self):
        """Test creating token-weighted strategy."""
        strategy = TokenWeightedStrategy()
        
        assert strategy.name == "TokenWeightedStrategy"
        assert strategy.min_token_balance == 1
        assert strategy.max_voting_power is None
    
    def test_token_weighted_strategy_with_config(self):
        """Test creating token-weighted strategy with configuration."""
        config = {
            "min_token_balance": 100,
            "max_voting_power": 10000
        }
        strategy = TokenWeightedStrategy(config)
        
        assert strategy.min_token_balance == 100
        assert strategy.max_voting_power == 10000
    
    def test_calculate_voting_power_sufficient_balance(self):
        """Test calculating voting power with sufficient token balance."""
        strategy = TokenWeightedStrategy()
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=1000,
            delegated_power=500
        )
        
        assert power.voter_address == "0x123"
        assert power.power == 1000
        assert power.token_balance == 1000
        assert power.delegated_power == 500
        assert power.total_power() == 1500
    
    def test_calculate_voting_power_insufficient_balance(self):
        """Test calculating voting power with insufficient token balance."""
        config = {"min_token_balance": 100}
        strategy = TokenWeightedStrategy(config)
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=50,
            delegated_power=0
        )
        
        assert power.power == 0
        assert power.token_balance == 50
    
    def test_calculate_voting_power_with_max_limit(self):
        """Test calculating voting power with maximum limit."""
        config = {"max_voting_power": 5000}
        strategy = TokenWeightedStrategy(config)
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=10000,
            delegated_power=0
        )
        
        assert power.power == 5000  # Capped at max_voting_power
        assert power.token_balance == 10000
    
    def test_validate_vote_valid(self):
        """Test validating a valid vote."""
        strategy = TokenWeightedStrategy()
        
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
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        assert strategy.validate_vote(vote, proposal) is True
    
    def test_validate_vote_insufficient_balance(self):
        """Test validating vote with insufficient balance."""
        config = {"min_token_balance": 100}
        strategy = TokenWeightedStrategy(config)
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=50,
            token_balance=50
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        assert strategy.validate_vote(vote, proposal) is False
    
    def test_calculate_proposal_result(self):
        """Test calculating proposal result."""
        strategy = TokenWeightedStrategy()
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal",
            quorum_threshold=2000,
            approval_threshold=0.5
        )
        
        # Create votes
        votes = []
        for i, (choice, power) in enumerate([(VoteChoice.FOR, 1000), (VoteChoice.FOR, 800), (VoteChoice.AGAINST, 500)]):
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
            votes.append(vote)
        
        result = strategy.calculate_proposal_result(proposal, votes)
        
        assert result["strategy"] == "token_weighted"
        assert result["total_voting_power"] == 2300
        assert result["for_power"] == 1800
        assert result["against_power"] == 500
        assert result["abstain_power"] == 0
        assert result["total_votes"] == 3
        assert result["quorum_met"] is True
        assert result["approval_percentage"] == 1800 / 2300
        assert result["approved"] is True


class TestQuadraticVotingStrategy:
    """Test QuadraticVotingStrategy class."""
    
    def test_quadratic_voting_strategy_creation(self):
        """Test creating quadratic voting strategy."""
        strategy = QuadraticVotingStrategy()
        
        assert strategy.name == "QuadraticVotingStrategy"
        assert strategy.min_token_balance == 1
        assert strategy.max_votes_per_proposal == 100
        assert strategy.vote_cost_multiplier == 1.0
    
    def test_calculate_voting_power_quadratic(self):
        """Test calculating voting power using quadratic formula."""
        strategy = QuadraticVotingStrategy()
        
        # Test with perfect square
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=10000,  # sqrt(10000) = 100
            delegated_power=0
        )
        
        assert power.power == 100
        assert power.token_balance == 10000
        
        # Test with non-perfect square
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=15000,  # sqrt(15000) ≈ 122
            delegated_power=0
        )
        
        assert power.power == int(math.sqrt(15000))
    
    def test_calculate_voting_power_insufficient_balance(self):
        """Test calculating voting power with insufficient balance."""
        config = {"min_token_balance": 100}
        strategy = QuadraticVotingStrategy(config)
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=50,
            delegated_power=0
        )
        
        assert power.power == 0
        assert power.token_balance == 50
    
    def test_validate_vote_with_max_limit(self):
        """Test validating vote with maximum votes per proposal limit."""
        config = {"max_votes_per_proposal": 50}
        strategy = QuadraticVotingStrategy(config)
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=100,  # Exceeds max_votes_per_proposal
            token_balance=10000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        assert strategy.validate_vote(vote, proposal) is False
    
    def test_calculate_proposal_result_quadratic(self):
        """Test calculating proposal result with quadratic voting."""
        strategy = QuadraticVotingStrategy()
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal",
            quorum_threshold=100,
            approval_threshold=0.5
        )
        
        # Create votes with quadratic power
        votes = []
        for i, (choice, token_balance) in enumerate([(VoteChoice.FOR, 10000), (VoteChoice.FOR, 40000), (VoteChoice.AGAINST, 2500)]):
            power = int(math.sqrt(token_balance))
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=power,
                token_balance=token_balance
            )
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter_address=f"0x{i}",
                choice=choice,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            votes.append(vote)
        
        result = strategy.calculate_proposal_result(proposal, votes)
        
        assert result["strategy"] == "quadratic_voting"
        assert result["total_voting_power"] == 100 + 200 + 50  # sqrt(10000) + sqrt(40000) + sqrt(2500)
        assert result["for_power"] == 300
        assert result["against_power"] == 50
        assert result["abstain_power"] == 0
        assert result["total_votes"] == 3
        assert result["quorum_met"] is True
        assert result["approval_percentage"] == 300 / 350
        assert result["approved"] is True


class TestConvictionVotingStrategy:
    """Test ConvictionVotingStrategy class."""
    
    def test_conviction_voting_strategy_creation(self):
        """Test creating conviction voting strategy."""
        strategy = ConvictionVotingStrategy()
        
        assert strategy.name == "ConvictionVotingStrategy"
        assert isinstance(strategy.conviction_config, ConvictionVotingConfig)
        assert strategy.conviction_config.max_conviction == 1.0
        assert strategy.conviction_config.conviction_growth_rate == 0.1
        assert strategy.conviction_config.conviction_decay_rate == 0.05
    
    def test_calculate_voting_power_with_conviction(self):
        """Test calculating voting power with conviction."""
        strategy = ConvictionVotingStrategy()
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=1000,
            delegated_power=0,
            conviction=0.5
        )
        
        # Base power * (1 + conviction) = 1000 * 1.5 = 1500
        assert power.power == 1500
        assert power.token_balance == 1000
    
    def test_update_conviction_growth(self):
        """Test conviction growth when actively voting."""
        strategy = ConvictionVotingStrategy()
        
        # Initial conviction
        conviction = strategy.update_conviction("0x123", "prop_123", 100, has_voted=True)
        assert conviction == 0.1  # Initial growth
        
        # Second vote
        conviction = strategy.update_conviction("0x123", "prop_123", 101, has_voted=True)
        assert conviction == 0.2  # 0.1 + 0.1
        
        # Third vote
        conviction = strategy.update_conviction("0x123", "prop_123", 102, has_voted=True)
        assert conviction == 0.3  # 0.2 + 0.1
    
    def test_update_conviction_decay(self):
        """Test conviction decay when not voting."""
        strategy = ConvictionVotingStrategy()
        
        # Build up some conviction first
        strategy.update_conviction("0x123", "prop_123", 100, has_voted=True)
        strategy.update_conviction("0x123", "prop_123", 101, has_voted=True)
        conviction = strategy.update_conviction("0x123", "prop_123", 102, has_voted=True)
        assert conviction == 0.3
        
        # Now decay
        conviction = strategy.update_conviction("0x123", "prop_123", 103, has_voted=False)
        assert conviction == 0.25  # 0.3 - 0.05
        
        conviction = strategy.update_conviction("0x123", "prop_123", 104, has_voted=False)
        assert conviction == 0.2  # 0.25 - 0.05
    
    def test_update_conviction_max_limit(self):
        """Test conviction growth with maximum limit."""
        strategy = ConvictionVotingStrategy()
        
        # Build up to maximum conviction
        for i in range(15):  # More than needed to reach max
            conviction = strategy.update_conviction("0x123", "prop_123", 100 + i, has_voted=True)
        
        assert conviction == 1.0  # Should be capped at max_conviction
    
    def test_validate_vote_conviction_threshold(self):
        """Test vote validation with conviction threshold."""
        strategy = ConvictionVotingStrategy()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        # Build up conviction using the actual proposal ID
        strategy.update_conviction("0x123", proposal.proposal_id, 100, has_voted=True)
        strategy.update_conviction("0x123", proposal.proposal_id, 101, has_voted=True)
        
        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        # Should be valid with sufficient conviction
        assert strategy.validate_vote(vote, proposal) is True
        
        # Test with insufficient conviction
        strategy.conviction_snapshots["0x123"][proposal.proposal_id] = 0.05  # Below threshold
        assert strategy.validate_vote(vote, proposal) is False
    
    def test_calculate_proposal_result_conviction(self):
        """Test calculating proposal result with conviction voting."""
        strategy = ConvictionVotingStrategy()
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal",
            quorum_threshold=1000,
            approval_threshold=0.5
        )
        
        # Create votes with conviction
        votes = []
        for i, (choice, base_power, conviction) in enumerate([(VoteChoice.FOR, 1000, 0.5), (VoteChoice.FOR, 500, 0.3), (VoteChoice.AGAINST, 800, 0.2)]):
            power = int(base_power * (1 + conviction))
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=power,
                token_balance=base_power
            )
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter_address=f"0x{i}",
                choice=choice,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            votes.append(vote)
        
        result = strategy.calculate_proposal_result(proposal, votes)
        
        assert result["strategy"] == "conviction_voting"
        assert result["total_voting_power"] == 1500 + 650 + 960  # 1000*1.5 + 500*1.3 + 800*1.2
        assert result["for_power"] == 1500 + 650
        assert result["against_power"] == 960
        assert result["abstain_power"] == 0
        assert result["total_votes"] == 3
        assert result["quorum_met"] is True
        assert result["approved"] is True


class TestSnapshotVotingStrategy:
    """Test SnapshotVotingStrategy class."""
    
    def test_snapshot_voting_strategy_creation(self):
        """Test creating snapshot voting strategy."""
        strategy = SnapshotVotingStrategy()
        
        assert strategy.name == "SnapshotVotingStrategy"
        assert isinstance(strategy.snapshot_config, SnapshotVotingConfig)
        assert strategy.snapshot_config.snapshot_block_offset == 100
        assert strategy.snapshot_config.merkle_tree_depth == 20
    
    def test_create_snapshot(self):
        """Test creating a voting power snapshot."""
        strategy = SnapshotVotingStrategy()
        
        balances = {
            "0x123": 1000,
            "0x456": 2000,
            "0x789": 500
        }
        
        merkle_root = strategy.create_snapshot(1000, balances)
        
        assert merkle_root is not None
        assert 1000 in strategy.snapshots
        assert strategy.snapshots[1000] == balances
        assert 1000 in strategy.merkle_trees
    
    def test_create_merkle_tree_empty(self):
        """Test creating Merkle tree with empty data."""
        strategy = SnapshotVotingStrategy()
        
        merkle_root = strategy._create_merkle_tree({})
        
        assert merkle_root is not None
        assert merkle_root == strategy._create_merkle_tree({})  # Deterministic
    
    def test_create_merkle_tree_single_item(self):
        """Test creating Merkle tree with single item."""
        strategy = SnapshotVotingStrategy()
        
        data = {"0x123": 1000}
        merkle_root = strategy._create_merkle_tree(data)
        
        assert merkle_root is not None
        assert merkle_root == strategy._create_merkle_tree(data)  # Deterministic
    
    def test_create_merkle_tree_multiple_items(self):
        """Test creating Merkle tree with multiple items."""
        strategy = SnapshotVotingStrategy()
        
        data = {"0x123": 1000, "0x456": 2000, "0x789": 500}
        merkle_root = strategy._create_merkle_tree(data)
        
        assert merkle_root is not None
        assert merkle_root == strategy._create_merkle_tree(data)  # Deterministic
    
    def test_generate_merkle_proof(self):
        """Test generating Merkle proof for an address."""
        strategy = SnapshotVotingStrategy()
        
        balances = {
            "0x123": 1000,
            "0x456": 2000,
            "0x789": 500
        }
        
        strategy.create_snapshot(1000, balances)
        
        proof = strategy.generate_merkle_proof(1000, "0x456")
        
        assert proof is not None
        assert proof["block_height"] == 1000
        assert proof["address"] == "0x456"
        assert proof["balance"] == 2000
        assert proof["merkle_root"] == strategy.merkle_trees[1000]
        assert len(proof["proof_path"]) > 0
        assert proof["leaf_index"] >= 0
    
    def test_generate_merkle_proof_nonexistent_address(self):
        """Test generating Merkle proof for nonexistent address."""
        strategy = SnapshotVotingStrategy()
        
        balances = {"0x123": 1000}
        strategy.create_snapshot(1000, balances)
        
        proof = strategy.generate_merkle_proof(1000, "0x456")
        
        assert proof is None
    
    def test_verify_merkle_proof(self):
        """Test verifying a Merkle proof."""
        strategy = SnapshotVotingStrategy()
        
        balances = {
            "0x123": 1000,
            "0x456": 2000,
            "0x789": 500
        }
        
        strategy.create_snapshot(1000, balances)
        
        proof = strategy.generate_merkle_proof(1000, "0x456")
        assert proof is not None
        
        # Verify the proof
        assert strategy.verify_merkle_proof(proof) is True
        
        # Tamper with the proof
        proof["balance"] = 3000
        assert strategy.verify_merkle_proof(proof) is False
    
    def test_calculate_voting_power_with_snapshot(self):
        """Test calculating voting power using snapshot."""
        strategy = SnapshotVotingStrategy()
        
        # Create snapshot
        balances = {"0x123": 1000, "0x456": 2000}
        strategy.create_snapshot(1000, balances)
        
        # Calculate power using snapshot
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=1500,  # Current balance
            delegated_power=0,
            snapshot_block=1000
        )
        
        assert power.power == 1000  # Should use snapshot balance
        assert power.token_balance == 1500  # Current balance preserved
    
    def test_calculate_voting_power_without_snapshot(self):
        """Test calculating voting power without snapshot."""
        strategy = SnapshotVotingStrategy()
        
        power = strategy.calculate_voting_power(
            voter_address="0x123",
            token_balance=1500,
            delegated_power=0
        )
        
        assert power.power == 1500  # Should use current balance
        assert power.token_balance == 1500
    
    def test_create_vote_merkle_tree(self):
        """Test creating Merkle tree for proposal votes."""
        strategy = SnapshotVotingStrategy()
        
        # Create mock votes
        votes = []
        for i, (choice, power) in enumerate([(VoteChoice.FOR, 1000), (VoteChoice.AGAINST, 500)]):
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=power,
                token_balance=power
            )
            vote = Vote(
                proposal_id="prop_123",
                voter_address=f"0x{i}",
                choice=choice,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            votes.append(vote)
        
        merkle_root = strategy.create_vote_merkle_tree("prop_123", votes)
        
        assert merkle_root is not None
        assert "votes_prop_123" in strategy.merkle_trees
    
    def test_generate_vote_proof(self):
        """Test generating Merkle proof for a vote."""
        strategy = SnapshotVotingStrategy()
        
        # Create vote
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
        
        # Create Merkle tree
        strategy.create_vote_merkle_tree("prop_123", [vote])
        
        # Generate proof
        proof = strategy.generate_vote_proof("prop_123", [vote], vote)
        
        assert proof is not None
        assert proof["address"] == "0x123:for:1000"
        assert proof["balance"] == 1000
        assert "merkle_root" in proof
        assert "proof_path" in proof


class TestStrategyFactory:
    """Test StrategyFactory class."""
    
    def test_create_token_weighted_strategy(self):
        """Test creating token-weighted strategy through factory."""
        strategy = StrategyFactory.create_strategy("token_weighted")
        
        assert isinstance(strategy, TokenWeightedStrategy)
        assert strategy.name == "TokenWeightedStrategy"
    
    def test_create_quadratic_voting_strategy(self):
        """Test creating quadratic voting strategy through factory."""
        strategy = StrategyFactory.create_strategy("quadratic_voting")
        
        assert isinstance(strategy, QuadraticVotingStrategy)
        assert strategy.name == "QuadraticVotingStrategy"
    
    def test_create_conviction_voting_strategy(self):
        """Test creating conviction voting strategy through factory."""
        strategy = StrategyFactory.create_strategy("conviction_voting")
        
        assert isinstance(strategy, ConvictionVotingStrategy)
        assert strategy.name == "ConvictionVotingStrategy"
    
    def test_create_snapshot_voting_strategy(self):
        """Test creating snapshot voting strategy through factory."""
        strategy = StrategyFactory.create_strategy("snapshot_voting")
        
        assert isinstance(strategy, SnapshotVotingStrategy)
        assert strategy.name == "SnapshotVotingStrategy"
    
    def test_create_strategy_with_config(self):
        """Test creating strategy with configuration."""
        config = {"min_token_balance": 100}
        strategy = StrategyFactory.create_strategy("token_weighted", config)
        
        assert isinstance(strategy, TokenWeightedStrategy)
        assert strategy.min_token_balance == 100
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with pytest.raises(ValidationError):
            StrategyFactory.create_strategy("unknown_strategy")
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        strategies = StrategyFactory.get_available_strategies()
        
        assert "token_weighted" in strategies
        assert "quadratic_voting" in strategies
        assert "conviction_voting" in strategies
        assert "snapshot_voting" in strategies
        assert len(strategies) == 4
    
    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        class CustomStrategy(VotingStrategy):
            def calculate_voting_power(self, voter_address, token_balance, delegated_power=0, **kwargs):
                return VotingPower(voter_address, token_balance * 2, token_balance, delegated_power)
            
            def validate_vote(self, vote, proposal):
                return True
            
            def calculate_proposal_result(self, proposal, votes):
                return {"strategy": "custom"}
        
        # Register custom strategy
        StrategyFactory.register_strategy("custom", CustomStrategy)
        
        # Create strategy
        strategy = StrategyFactory.create_strategy("custom")
        assert isinstance(strategy, CustomStrategy)
        
        # Check it's in available strategies
        strategies = StrategyFactory.get_available_strategies()
        assert "custom" in strategies
    
    def test_register_invalid_strategy(self):
        """Test registering invalid strategy raises error."""
        class InvalidStrategy:
            pass
        
        with pytest.raises(ValidationError):
            StrategyFactory.register_strategy("invalid", InvalidStrategy)
