"""
Unit tests for delegated proof of stake consensus.
"""

import time
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    DelegateInfo,
    VotingPower,
)
from dubchain.consensus.delegated_proof_of_stake import (
    DelegatedProofOfStake,
    ElectionManager,
)
from dubchain.consensus.validator import Validator, ValidatorInfo
from dubchain.crypto.signatures import PrivateKey


class TestElectionManager:
    """Test ElectionManager class."""

    def test_election_manager_creation(self):
        """Test creating election manager."""
        manager = ElectionManager()
        assert manager.election_interval == 86400
        assert manager.delegate_count == 21
        assert manager.voting_period == 3600
        assert isinstance(manager.last_election, float)
        assert len(manager.current_delegates) == 0
        assert isinstance(manager.next_election_time, float)

    def test_is_election_time_future(self):
        """Test election time check when election is in future."""
        manager = ElectionManager()
        manager.next_election_time = time.time() + 3600  # 1 hour from now
        assert manager.is_election_time() is False

    def test_is_election_time_now(self):
        """Test election time check when election is now."""
        manager = ElectionManager()
        manager.next_election_time = time.time() - 1  # 1 second ago
        assert manager.is_election_time() is True

    def test_schedule_next_election(self):
        """Test scheduling next election."""
        manager = ElectionManager()
        current_time = time.time()
        manager.schedule_next_election()

        expected_time = current_time + manager.election_interval
        assert abs(manager.next_election_time - expected_time) < 1.0

    def test_get_voting_deadline(self):
        """Test getting voting deadline."""
        manager = ElectionManager()
        manager.next_election_time = time.time() + 86400
        deadline = manager.get_voting_deadline()

        expected_deadline = manager.next_election_time - manager.voting_period
        assert abs(deadline - expected_deadline) < 1.0


class TestDelegatedProofOfStake:
    """Test DelegatedProofOfStake class."""

    @pytest.fixture
    def consensus_config(self):
        """Create consensus config."""
        return ConsensusConfig(
            consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            block_time=10,
            max_validators=100,
            min_stake=1000000,
        )

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator."""
        private_key = PrivateKey.generate()
        validator = Mock(spec=Validator)
        validator.validator_id = "validator_1"
        validator.public_key = private_key.get_public_key()
        return validator

    @pytest.fixture
    def dpos_consensus(self, consensus_config):
        """Create DPoS consensus instance."""
        return DelegatedProofOfStake(consensus_config)

    def test_dpos_consensus_creation(self, dpos_consensus, consensus_config):
        """Test creating DPoS consensus."""
        assert dpos_consensus.config == consensus_config
        assert len(dpos_consensus.validators) == 0
        assert len(dpos_consensus.delegates) == 0
        assert len(dpos_consensus.voting_power) == 0
        assert len(dpos_consensus.delegations) == 0
        assert isinstance(dpos_consensus.election_manager, ElectionManager)
        assert isinstance(dpos_consensus.metrics, ConsensusMetrics)
        assert dpos_consensus.current_producer_index == 0
        assert len(dpos_consensus.block_production_schedule) == 0
        assert isinstance(dpos_consensus.last_block_time, float)
        assert dpos_consensus.block_interval == consensus_config.block_time
        assert len(dpos_consensus.delegate_rewards) == 0
        assert len(dpos_consensus.voter_rewards) == 0

    def test_register_delegate_success(self, dpos_consensus, mock_validator):
        """Test successful delegate registration."""
        initial_stake = 2000000
        result = dpos_consensus.register_delegate(mock_validator, initial_stake)

        assert result is True
        assert mock_validator.validator_id in dpos_consensus.validators
        assert mock_validator.validator_id in dpos_consensus.voting_power

        validator_info = dpos_consensus.validators[mock_validator.validator_id]
        assert validator_info.validator_id == mock_validator.validator_id
        assert validator_info.public_key == mock_validator.public_key
        assert validator_info.total_stake == initial_stake
        assert validator_info.self_stake == initial_stake

        voting_power = dpos_consensus.voting_power[mock_validator.validator_id]
        assert voting_power.validator_id == mock_validator.validator_id
        assert voting_power.total_power == initial_stake
        assert voting_power.self_stake == initial_stake
        assert voting_power.delegated_stake == 0

    def test_register_delegate_max_validators(self, dpos_consensus, mock_validator):
        """Test delegate registration when max validators reached."""
        # Set max validators to 1
        dpos_consensus.config.max_validators = 1

        # Register first validator
        result1 = dpos_consensus.register_delegate(mock_validator, 1000000)
        assert result1 is True

        # Try to register second validator
        mock_validator2 = Mock(spec=Validator)
        mock_validator2.validator_id = "validator_2"
        mock_validator2.public_key = PrivateKey.generate().get_public_key()

        result2 = dpos_consensus.register_delegate(mock_validator2, 1000000)
        assert result2 is False
        assert mock_validator2.validator_id not in dpos_consensus.validators

    def test_vote_for_delegate_success(self, dpos_consensus, mock_validator):
        """Test successful voting for delegate."""
        # Register delegate first
        dpos_consensus.register_delegate(mock_validator, 1000000)

        voter_id = "voter_1"
        delegate_id = mock_validator.validator_id
        amount = 500000

        result = dpos_consensus.vote_for_delegate(voter_id, delegate_id, amount)

        assert result is True
        assert voter_id in dpos_consensus.delegations
        assert len(dpos_consensus.delegations[voter_id]) == 1

        delegation = dpos_consensus.delegations[voter_id][0]
        assert delegation.delegate_id == delegate_id
        assert delegation.voter_id == voter_id
        assert delegation.amount == amount

        # Check voting power update
        voting_power = dpos_consensus.voting_power[delegate_id]
        assert voting_power.delegated_stake == amount
        assert voting_power.total_power == 1000000 + amount

    def test_vote_for_delegate_invalid_delegate(self, dpos_consensus):
        """Test voting for non-existent delegate."""
        voter_id = "voter_1"
        delegate_id = "non_existent_delegate"
        amount = 500000

        result = dpos_consensus.vote_for_delegate(voter_id, delegate_id, amount)
        assert result is False
        assert voter_id not in dpos_consensus.delegations

    def test_vote_for_delegate_multiple_votes(self, dpos_consensus, mock_validator):
        """Test voting for same delegate multiple times."""
        # Register delegate first
        dpos_consensus.register_delegate(mock_validator, 1000000)

        voter_id = "voter_1"
        delegate_id = mock_validator.validator_id

        # First vote
        result1 = dpos_consensus.vote_for_delegate(voter_id, delegate_id, 500000)
        assert result1 is True

        # Second vote for same delegate
        result2 = dpos_consensus.vote_for_delegate(voter_id, delegate_id, 300000)
        assert result2 is True

        # Should have only one delegation with combined amount
        assert len(dpos_consensus.delegations[voter_id]) == 1
        delegation = dpos_consensus.delegations[voter_id][0]
        assert delegation.amount == 800000

        # Check voting power
        voting_power = dpos_consensus.voting_power[delegate_id]
        assert voting_power.delegated_stake == 800000

    def test_unvote_delegate_success(self, dpos_consensus, mock_validator):
        """Test successful unvoting from delegate."""
        # Register delegate and vote
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.vote_for_delegate("voter_1", mock_validator.validator_id, 500000)

        # Unvote part of the amount
        result = dpos_consensus.unvote_delegate(
            "voter_1", mock_validator.validator_id, 200000
        )

        assert result is True
        delegation = dpos_consensus.delegations["voter_1"][0]
        assert delegation.amount == 300000

        # Check voting power
        voting_power = dpos_consensus.voting_power[mock_validator.validator_id]
        assert voting_power.delegated_stake == 300000

    def test_unvote_delegate_complete(self, dpos_consensus, mock_validator):
        """Test unvoting all votes from delegate."""
        # Register delegate and vote
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.vote_for_delegate("voter_1", mock_validator.validator_id, 500000)

        # Unvote all
        result = dpos_consensus.unvote_delegate(
            "voter_1", mock_validator.validator_id, 500000
        )

        assert result is True
        assert len(dpos_consensus.delegations["voter_1"]) == 0

        # Check voting power
        voting_power = dpos_consensus.voting_power[mock_validator.validator_id]
        assert voting_power.delegated_stake == 0

    def test_unvote_delegate_invalid_voter(self, dpos_consensus, mock_validator):
        """Test unvoting with invalid voter."""
        dpos_consensus.register_delegate(mock_validator, 1000000)

        result = dpos_consensus.unvote_delegate(
            "non_existent_voter", mock_validator.validator_id, 100000
        )
        assert result is False

    def test_unvote_delegate_insufficient_amount(self, dpos_consensus, mock_validator):
        """Test unvoting more than available."""
        # Register delegate and vote
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.vote_for_delegate("voter_1", mock_validator.validator_id, 500000)

        # Try to unvote more than available
        result = dpos_consensus.unvote_delegate(
            "voter_1", mock_validator.validator_id, 600000
        )
        assert result is False

        # Amount should remain unchanged
        delegation = dpos_consensus.delegations["voter_1"][0]
        assert delegation.amount == 500000

    def test_conduct_election_no_validators(self, dpos_consensus):
        """Test conducting election with no validators."""
        result = dpos_consensus.conduct_election()
        assert result == []

    def test_conduct_election_success(self, dpos_consensus, mock_validator):
        """Test successful election."""
        # Register multiple validators with different stakes
        dpos_consensus.register_delegate(mock_validator, 2000000)

        mock_validator2 = Mock(spec=Validator)
        mock_validator2.validator_id = "validator_2"
        mock_validator2.public_key = PrivateKey.generate().get_public_key()
        dpos_consensus.register_delegate(mock_validator2, 1500000)

        mock_validator3 = Mock(spec=Validator)
        mock_validator3.validator_id = "validator_3"
        mock_validator3.public_key = PrivateKey.generate().get_public_key()
        dpos_consensus.register_delegate(mock_validator3, 1000000)

        # Vote for validators to give them different voting power
        dpos_consensus.vote_for_delegate("voter_1", "validator_1", 1000000)
        dpos_consensus.vote_for_delegate("voter_2", "validator_2", 500000)

        # Conduct election
        elected_delegates = dpos_consensus.conduct_election()

        # Should elect top delegates by voting power
        assert len(elected_delegates) <= dpos_consensus.election_manager.delegate_count
        assert "validator_1" in elected_delegates  # Highest voting power
        assert "validator_2" in elected_delegates  # Second highest

        # Check that delegates are updated
        assert len(dpos_consensus.delegates) == len(elected_delegates)
        for delegate_id in elected_delegates:
            assert delegate_id in dpos_consensus.delegates

        # Check election manager state
        assert dpos_consensus.election_manager.current_delegates == elected_delegates
        assert dpos_consensus.election_manager.last_election > 0
        assert dpos_consensus.election_manager.next_election_time > time.time()

    def test_create_production_schedule(self, dpos_consensus, mock_validator):
        """Test creating block production schedule."""
        # Register delegates and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        # Should have production schedule
        assert len(dpos_consensus.block_production_schedule) > 0
        assert dpos_consensus.current_producer_index == 0

    def test_get_current_producer_no_schedule(self, dpos_consensus):
        """Test getting current producer with no schedule."""
        producer = dpos_consensus.get_current_producer()
        assert producer is None

    def test_get_current_producer_with_schedule(self, dpos_consensus, mock_validator):
        """Test getting current producer with schedule."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        producer = dpos_consensus.get_current_producer()
        assert producer is not None
        assert producer in dpos_consensus.block_production_schedule

    def test_advance_producer(self, dpos_consensus, mock_validator):
        """Test advancing to next producer."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        original_index = dpos_consensus.current_producer_index
        original_time = dpos_consensus.last_block_time

        dpos_consensus.advance_producer()

        # Should advance index
        expected_index = (original_index + 1) % len(
            dpos_consensus.block_production_schedule
        )
        assert dpos_consensus.current_producer_index == expected_index

        # Should update last block time
        assert dpos_consensus.last_block_time > original_time

    def test_is_production_time(self, dpos_consensus):
        """Test production time check."""
        # Initially last_block_time is set to current time, so not production time
        assert dpos_consensus.is_production_time() is False

        # Set last block time to allow production
        dpos_consensus.last_block_time = time.time() - 20  # 20 seconds ago

        # Should be production time now
        assert dpos_consensus.is_production_time() is True

    def test_produce_block_no_producer(self, dpos_consensus):
        """Test producing block with no active producer."""
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x0",
        }

        result = dpos_consensus.produce_block(block_data)

        assert result.success is False
        assert result.error_message == "No active producer"
        assert result.consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE

    def test_produce_block_not_production_time(self, dpos_consensus, mock_validator):
        """Test producing block when not production time."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        # Set last block time to now (not enough time passed)
        dpos_consensus.last_block_time = time.time()

        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x0",
        }

        result = dpos_consensus.produce_block(block_data)

        assert result.success is False
        assert result.error_message == "Not production time"

    def test_produce_block_invalid_block(self, dpos_consensus, mock_validator):
        """Test producing invalid block."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        # Set last block time to allow production
        dpos_consensus.last_block_time = time.time() - 20

        # Invalid block data (missing required fields)
        block_data = {
            "block_number": 1,
            "timestamp": time.time()
            # Missing 'transactions' and 'previous_hash'
        }

        result = dpos_consensus.produce_block(block_data)

        assert result.success is False
        assert result.error_message == "Invalid block"

    def test_produce_block_success(self, dpos_consensus, mock_validator):
        """Test successful block production."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        # Set last block time to allow production
        dpos_consensus.last_block_time = time.time() - 20

        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x0",
            "gas_used": 100000,
        }

        result = dpos_consensus.produce_block(block_data)

        assert result.success is True
        assert result.block_hash is not None
        assert result.validator_id == mock_validator.validator_id
        assert result.consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE
        assert result.gas_used == 100000

        # Check metrics update
        assert dpos_consensus.metrics.total_blocks == 1
        assert dpos_consensus.metrics.successful_blocks == 1

    def test_validate_block_success(self, dpos_consensus, mock_validator):
        """Test successful block validation."""
        # Register delegate
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x0",
        }

        result = dpos_consensus._validate_block(block_data, mock_validator.validator_id)
        assert result is True

    def test_validate_block_missing_fields(self, dpos_consensus, mock_validator):
        """Test block validation with missing fields."""
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        # Missing required fields
        block_data = {
            "block_number": 1,
            "timestamp": time.time()
            # Missing 'transactions' and 'previous_hash'
        }

        result = dpos_consensus._validate_block(block_data, mock_validator.validator_id)
        assert result is False

    def test_validate_block_invalid_timestamp(self, dpos_consensus, mock_validator):
        """Test block validation with invalid timestamp."""
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        block_data = {
            "block_number": 1,
            "timestamp": time.time() - 400,  # Too old (> 5 minutes)
            "transactions": [],
            "previous_hash": "0x0",
        }

        result = dpos_consensus._validate_block(block_data, mock_validator.validator_id)
        assert result is False

    def test_validate_block_invalid_producer(self, dpos_consensus, mock_validator):
        """Test block validation with invalid producer."""
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x0",
        }

        # Use invalid producer ID
        result = dpos_consensus._validate_block(block_data, "invalid_producer")
        assert result is False

    def test_calculate_block_hash(self, dpos_consensus):
        """Test block hash calculation."""
        block_data = {
            "block_number": 1,
            "timestamp": 1234567890,
            "transactions": ["tx1", "tx2"],
            "previous_hash": "0x0",
        }

        hash1 = dpos_consensus._calculate_block_hash(block_data)
        hash2 = dpos_consensus._calculate_block_hash(block_data)

        # Same data should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length

    def test_distribute_rewards_no_delegates(self, dpos_consensus):
        """Test reward distribution with no delegates."""
        result = dpos_consensus.distribute_rewards(1000000)
        assert result == {}

    def test_distribute_rewards_success(self, dpos_consensus, mock_validator):
        """Test successful reward distribution."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        # Vote for delegate
        dpos_consensus.vote_for_delegate("voter_1", mock_validator.validator_id, 500000)

        total_rewards = 1000000
        result = dpos_consensus.distribute_rewards(total_rewards)

        # Should distribute rewards
        assert len(result) > 0
        assert mock_validator.validator_id in result  # Delegate reward
        assert "voter_1" in result  # Voter reward

        # Check delegate rewards
        assert mock_validator.validator_id in dpos_consensus.delegate_rewards
        assert dpos_consensus.delegate_rewards[mock_validator.validator_id] > 0

        # Check voter rewards
        assert "voter_1" in dpos_consensus.voter_rewards
        assert dpos_consensus.voter_rewards["voter_1"] > 0

    def test_get_delegate_rankings(self, dpos_consensus, mock_validator):
        """Test getting delegate rankings."""
        # Register multiple validators with different stakes
        dpos_consensus.register_delegate(mock_validator, 2000000)

        mock_validator2 = Mock(spec=Validator)
        mock_validator2.validator_id = "validator_2"
        mock_validator2.public_key = PrivateKey.generate().get_public_key()
        dpos_consensus.register_delegate(mock_validator2, 1500000)

        # Vote to change voting power
        dpos_consensus.vote_for_delegate("voter_1", "validator_2", 1000000)

        rankings = dpos_consensus.get_delegate_rankings()

        assert len(rankings) == 2
        # Should be sorted by voting power (descending)
        assert rankings[0][1] >= rankings[1][1]
        assert rankings[0][0] == "validator_2"  # Higher voting power
        assert rankings[1][0] == "validator_1"

    def test_get_voting_statistics(self, dpos_consensus, mock_validator):
        """Test getting voting statistics."""
        # Register delegate and vote
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()
        dpos_consensus.vote_for_delegate("voter_1", mock_validator.validator_id, 500000)

        stats = dpos_consensus.get_voting_statistics()

        assert stats["total_voters"] == 1
        assert stats["total_delegates"] == 1
        assert stats["active_delegates"] == 1
        assert stats["total_voting_power"] == 1500000  # 1000000 + 500000
        assert stats["current_producer"] == mock_validator.validator_id
        assert stats["next_election"] > 0
        assert stats["election_interval"] == 86400
        assert stats["block_interval"] == 10

    def test_get_consensus_metrics(self, dpos_consensus, mock_validator):
        """Test getting consensus metrics."""
        # Register delegate and conduct election
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()

        metrics = dpos_consensus.get_consensus_metrics()

        assert isinstance(metrics, ConsensusMetrics)
        assert metrics.validator_count == 1
        assert metrics.active_validators == 1
        assert metrics.consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE

    def test_to_dict(self, dpos_consensus, mock_validator):
        """Test converting to dictionary."""
        # Register delegate and vote
        dpos_consensus.register_delegate(mock_validator, 1000000)
        dpos_consensus.conduct_election()
        dpos_consensus.vote_for_delegate("voter_1", mock_validator.validator_id, 500000)

        data = dpos_consensus.to_dict()

        assert "config" in data
        assert "validators" in data
        assert "delegates" in data
        assert "voting_power" in data
        assert "delegations" in data
        assert "election_manager" in data
        assert "current_producer_index" in data
        assert "block_production_schedule" in data
        assert "last_block_time" in data
        assert "block_interval" in data
        assert "delegate_rewards" in data
        assert "voter_rewards" in data
        assert "metrics" in data

        # Check specific values
        assert len(data["validators"]) == 1
        assert len(data["delegates"]) == 1
        assert len(data["delegations"]) == 1
        assert data["block_interval"] == 10

    def test_from_dict(self, consensus_config):
        """Test creating from dictionary."""
        # Create original DPoS instance
        original = DelegatedProofOfStake(consensus_config)

        # Convert to dict and back
        data = original.to_dict()
        restored = DelegatedProofOfStake.from_dict(data)

        # Check that key properties are restored
        assert restored.config.consensus_type == original.config.consensus_type
        assert restored.config.block_time == original.config.block_time
        assert restored.config.max_validators == original.config.max_validators
        assert restored.config.min_stake == original.config.min_stake
        assert len(restored.validators) == len(original.validators)
        assert len(restored.delegates) == len(original.delegates)
        assert len(restored.voting_power) == len(original.voting_power)
        assert len(restored.delegations) == len(original.delegations)
        assert restored.block_interval == original.block_interval
        assert (
            restored.election_manager.election_interval
            == original.election_manager.election_interval
        )
        assert (
            restored.election_manager.delegate_count
            == original.election_manager.delegate_count
        )
