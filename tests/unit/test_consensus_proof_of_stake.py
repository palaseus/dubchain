"""
Unit tests for Proof of Stake consensus implementation.
"""

import time
from unittest.mock import Mock, patch

import pytest

from dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    StakingInfo,
    ValidatorRole,
    ValidatorStatus,
)
from dubchain.consensus.proof_of_stake import (
    ProofOfStake,
    RewardCalculator,
    StakingPool,
)
from dubchain.consensus.validator import ValidatorInfo
from dubchain.core.block import Block, BlockHeader
from dubchain.core.transaction import Transaction, TransactionType
from dubchain.crypto.hashing import SHA256Hasher
from dubchain.crypto.signatures import PrivateKey, PublicKey


class TestStakingPool:
    """Test the StakingPool class."""

    def test_staking_pool_creation(self):
        """Test creating a staking pool."""
        pool = StakingPool(validator_id="validator_1")

        assert pool.validator_id == "validator_1"
        assert pool.total_stake == 0
        assert pool.delegators == {}
        assert pool.rewards == 0
        assert pool.slashing_penalty == 0.0
        assert pool.created_at > 0
        assert pool.last_reward_distribution > 0

    def test_add_delegation(self):
        """Test adding delegation to pool."""
        pool = StakingPool(validator_id="validator_1")

        # Add first delegation
        pool.add_delegation("delegator_1", 1000)
        assert pool.total_stake == 1000
        assert pool.delegators["delegator_1"] == 1000

        # Add more to same delegator
        pool.add_delegation("delegator_1", 500)
        assert pool.total_stake == 1500
        assert pool.delegators["delegator_1"] == 1500

        # Add new delegator
        pool.add_delegation("delegator_2", 2000)
        assert pool.total_stake == 3500
        assert pool.delegators["delegator_2"] == 2000

    def test_remove_delegation(self):
        """Test removing delegation from pool."""
        pool = StakingPool(validator_id="validator_1")
        pool.add_delegation("delegator_1", 1000)
        pool.add_delegation("delegator_2", 2000)

        # Remove partial delegation
        result = pool.remove_delegation("delegator_1", 500)
        assert result is True
        assert pool.total_stake == 2500
        assert pool.delegators["delegator_1"] == 500

        # Remove remaining delegation
        result = pool.remove_delegation("delegator_1", 500)
        assert result is True
        assert pool.total_stake == 2000
        assert "delegator_1" not in pool.delegators

        # Try to remove from non-existent delegator
        result = pool.remove_delegation("delegator_3", 100)
        assert result is False

        # Try to remove more than available
        result = pool.remove_delegation("delegator_2", 3000)
        assert result is False

    def test_calculate_delegator_rewards(self):
        """Test calculating delegator rewards."""
        pool = StakingPool(validator_id="validator_1")
        pool.add_delegation("delegator_1", 1000)
        pool.add_delegation("delegator_2", 2000)

        rewards = pool.calculate_delegator_rewards(1000, 0.1)  # 10% commission

        # Validator gets 10% commission (100), delegators share 900
        # delegator_1: 1000/3000 * 900 = 300
        # delegator_2: 2000/3000 * 900 = 600
        assert rewards["delegator_1"] == 300
        assert rewards["delegator_2"] == 600

        # Test with no stake
        empty_pool = StakingPool(validator_id="validator_2")
        rewards = empty_pool.calculate_delegator_rewards(1000, 0.1)
        assert rewards == {}


class TestRewardCalculator:
    """Test the RewardCalculator class."""

    def test_reward_calculator_creation(self):
        """Test creating a reward calculator."""
        calculator = RewardCalculator()

        assert calculator.base_reward_rate == 0.1
        assert calculator.inflation_rate == 0.02
        assert calculator.block_time == 2.0
        assert calculator.total_supply == 1000000000

    def test_calculate_block_reward(self):
        """Test calculating block reward."""
        calculator = RewardCalculator()

        # Test first block
        reward = calculator.calculate_block_reward(0)
        assert reward > 0

        # Test later block (should be higher due to inflation)
        reward_later = calculator.calculate_block_reward(
            100000
        )  # Use larger block number for inflation
        assert reward_later >= reward  # Should be same or higher due to inflation

    def test_calculate_validator_reward(self):
        """Test calculating validator reward."""
        calculator = RewardCalculator()

        # Test with different stake amounts
        reward1 = calculator.calculate_validator_reward(1000, 10000, 100)
        reward2 = calculator.calculate_validator_reward(2000, 10000, 100)

        assert reward2 > reward1  # More stake = more reward
        assert reward1 > 0
        assert reward2 > 0

    def test_calculate_validator_reward(self):
        """Test calculating validator reward."""
        calculator = RewardCalculator()

        # Test delegation reward calculation
        reward = calculator.calculate_validator_reward(1000, 10000, 100)
        assert reward > 0
        assert reward < 100  # Should be less than total


class TestProofOfStake:
    """Test the ProofOfStake class."""

    @pytest.fixture
    def pos_consensus(self):
        """Fixture for ProofOfStake instance."""
        config = ConsensusConfig()
        return ProofOfStake(config)

    @pytest.fixture
    def mock_validators(self):
        """Fixture for mock validators."""
        from dubchain.consensus.validator import Validator

        validators = []
        for i in range(5):
            private_key = PrivateKey.generate()
            validator = Validator(
                validator_id=f"validator_{i}",
                private_key=private_key,
                commission_rate=0.1,
            )
            validators.append(validator)
        return validators

    def test_proof_of_stake_creation(self, pos_consensus):
        """Test creating a ProofOfStake instance."""
        assert pos_consensus.validator_set is not None
        assert pos_consensus.staking_pools == {}
        assert pos_consensus.reward_calculator is not None
        assert pos_consensus.validator_manager is not None
        assert pos_consensus.metrics is not None

    def test_register_validator(self, pos_consensus, mock_validators):
        """Test registering a validator."""
        validator = mock_validators[0]

        result = pos_consensus.register_validator(
            validator, 1000000
        )  # Use min_stake amount
        assert result is True
        assert validator.validator_id in pos_consensus.validator_set.validators
        assert validator.validator_id in pos_consensus.staking_pools

    def test_stake_tokens(self, pos_consensus, mock_validators):
        """Test staking tokens."""
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)

        # Test delegation
        result = pos_consensus.stake_to_validator(
            validator.validator_id, "delegator_1", 500000
        )
        assert result is True
        assert validator.validator_id in pos_consensus.staking_pools

    def test_unstake_tokens(self, pos_consensus, mock_validators):
        """Test unstaking tokens."""
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)
        pos_consensus.stake_to_validator(validator.validator_id, "delegator_1", 500000)

        # Test undelegation
        result = pos_consensus.unstake_from_validator(
            validator.validator_id, "delegator_1", 300000
        )
        assert result is True

    def test_select_validator(self, pos_consensus, mock_validators):
        """Test selecting a validator for block production."""
        # Register validators with different stakes
        for i, validator in enumerate(mock_validators):
            pos_consensus.register_validator(validator, 1000000 + i * 100000)
            # Manually update validator set status after activation
            pos_consensus.validator_set.update_validator_status(
                validator.validator_id, ValidatorStatus.ACTIVE
            )

        # Test validator selection
        selected_validator_id = pos_consensus.select_proposer(0)  # block_number=0
        assert selected_validator_id is not None
        assert selected_validator_id in [v.validator_id for v in mock_validators]

        # Test with different block numbers (should give different results due to randomness)
        validators_selected = set()
        for i in range(10):
            validator_id = pos_consensus.select_proposer(i)
            validators_selected.add(validator_id)

        # Should have some variety in selection
        assert len(validators_selected) > 1

    def test_validate_block(self, pos_consensus, mock_validators):
        """Test validating a block."""
        # Register validator
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)
        # Manually update validator set status after activation
        pos_consensus.validator_set.update_validator_status(
            validator.validator_id, ValidatorStatus.ACTIVE
        )

        # Test block proposal validation
        block_data = {
            "block_number": 1,
            "timestamp": int(time.time()),
            "transactions": [],
            "previous_hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        }
        result = pos_consensus.validate_block_proposal(
            validator.validator_id, block_data
        )
        assert result is True

    def test_distribute_rewards(self, pos_consensus, mock_validators):
        """Test distributing rewards."""
        # Register validators and stake tokens
        for validator in mock_validators[:3]:
            pos_consensus.register_validator(validator, 1000000)
            pos_consensus.stake_to_validator(
                validator.validator_id, "delegator_1", 500000
            )

        # Test reward distribution
        block_data = {
            "block_number": 1,
            "timestamp": int(time.time()),
            "transactions": [],
            "previous_hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        }
        result = pos_consensus.finalize_block(
            block_data, mock_validators[0].validator_id
        )

        assert result is not None
        assert result.consensus_type == ConsensusType.PROOF_OF_STAKE

    def test_slash_validator(self, pos_consensus, mock_validators):
        """Test slashing a validator."""
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)

        # Test slashing
        result = pos_consensus.slash_validator(validator.validator_id, "double_signing")
        assert result > 0  # Should return the slashed amount

    def test_get_validator_info(self, pos_consensus, mock_validators):
        """Test getting validator information."""
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)

        # Test getting validator info
        info = pos_consensus.get_validator_info(validator.validator_id)
        assert info is not None
        assert info.validator_id == validator.validator_id

        # Test getting non-existent validator
        info = pos_consensus.get_validator_info("non_existent")
        assert info is None

    def test_get_consensus_metrics(self, pos_consensus, mock_validators):
        """Test getting consensus metrics."""
        # Register some validators
        for validator in mock_validators[:3]:
            pos_consensus.register_validator(validator, 1000000)

        # Test getting metrics
        metrics = pos_consensus.get_consensus_metrics()
        assert metrics is not None
        assert metrics.consensus_type == ConsensusType.PROOF_OF_STAKE

    def test_get_staking_pool_info(self, pos_consensus, mock_validators):
        """Test getting staking pool information."""
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)

        # Test getting pool info
        pool = pos_consensus.get_staking_pool_info(validator.validator_id)
        assert pool is not None
        assert pool.validator_id == validator.validator_id

        # Test getting non-existent pool
        pool = pos_consensus.get_staking_pool_info("non_existent")
        assert pool is None

    def test_get_total_stake(self, pos_consensus, mock_validators):
        """Test getting total stake."""
        # Register validators
        for validator in mock_validators[:3]:
            pos_consensus.register_validator(validator, 1000000)

        # Test total stake
        total_stake = pos_consensus.get_total_stake()
        assert total_stake >= 3000000  # At least 3 validators with 1000000 each

    def test_is_validator_active(self, pos_consensus, mock_validators):
        """Test checking if validator is active."""
        validator = mock_validators[0]
        pos_consensus.register_validator(validator, 1000000)
        # Manually update validator set status after activation
        pos_consensus.validator_set.update_validator_status(
            validator.validator_id, ValidatorStatus.ACTIVE
        )

        # Test active validator
        assert pos_consensus.is_validator_active(validator.validator_id) is True

        # Test non-existent validator
        assert pos_consensus.is_validator_active("non_existent") is False
