"""
Unit tests for consensus validator module.
"""

import logging

logger = logging.getLogger(__name__)
import json
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.consensus.consensus_types import (
    ConsensusMetrics,
    StakingInfo,
    ValidatorRole,
    ValidatorStatus,
)
from dubchain.consensus.validator import (
    Validator,
    ValidatorInfo,
    ValidatorManager,
    ValidatorSet,
)
from dubchain.crypto.signatures import PrivateKey, PublicKey, Signature


class TestValidatorInfo:
    """Test ValidatorInfo class."""

    @pytest.fixture
    def mock_public_key(self):
        """Fixture for mock public key."""
        mock_key = Mock(spec=PublicKey)
        mock_key.to_hex.return_value = "mock_public_key_hex"
        return mock_key

    @pytest.fixture
    def validator_info(self, mock_public_key):
        """Fixture for validator info."""
        return ValidatorInfo(
            validator_id="validator_1",
            public_key=mock_public_key,
            status=ValidatorStatus.ACTIVE,
            role=ValidatorRole.VALIDATOR,
            total_stake=1000,
            self_stake=500,
            delegated_stake=500,
            voting_power=1000,
            commission_rate=0.1,
            created_at=1234567890.0,
            last_active=1234567890.0,
            slashing_count=0,
            total_rewards=100,
            metadata={"key": "value"},
        )

    def test_validator_info_creation(self, validator_info):
        """Test creating validator info."""
        assert validator_info.validator_id == "validator_1"
        assert validator_info.status == ValidatorStatus.ACTIVE
        assert validator_info.role == ValidatorRole.VALIDATOR
        assert validator_info.total_stake == 1000
        assert validator_info.self_stake == 500
        assert validator_info.delegated_stake == 500
        assert validator_info.voting_power == 1000
        assert validator_info.commission_rate == 0.1
        assert validator_info.slashing_count == 0
        assert validator_info.total_rewards == 100
        assert validator_info.metadata == {"key": "value"}

    def test_validator_info_defaults(self, mock_public_key):
        """Test validator info with default values."""
        validator_info = ValidatorInfo(
            validator_id="validator_1", public_key=mock_public_key
        )
        assert validator_info.status == ValidatorStatus.INACTIVE
        assert validator_info.role == ValidatorRole.VALIDATOR
        assert validator_info.total_stake == 0
        assert validator_info.self_stake == 0
        assert validator_info.delegated_stake == 0
        assert validator_info.voting_power == 0
        assert validator_info.commission_rate == 0.1
        assert validator_info.slashing_count == 0
        assert validator_info.total_rewards == 0
        assert validator_info.metadata == {}

    @patch("time.time")
    def test_update_voting_power(self, mock_time, validator_info):
        """Test updating voting power."""
        mock_time.return_value = 1234567891.0
        validator_info.update_voting_power()
        assert validator_info.voting_power == 1000
        assert validator_info.last_active == 1234567891.0

    def test_add_stake_self_stake(self, validator_info):
        """Test adding self stake."""
        validator_info.add_stake(200, is_self_stake=True)
        assert validator_info.total_stake == 1200
        assert validator_info.self_stake == 700
        assert validator_info.delegated_stake == 500
        assert validator_info.voting_power == 1200

    def test_add_stake_delegated_stake(self, validator_info):
        """Test adding delegated stake."""
        validator_info.add_stake(200, is_self_stake=False)
        assert validator_info.total_stake == 1200
        assert validator_info.self_stake == 500
        assert validator_info.delegated_stake == 700
        assert validator_info.voting_power == 1200

    def test_remove_stake_self_stake(self, validator_info):
        """Test removing self stake."""
        validator_info.remove_stake(100, is_self_stake=True)
        assert validator_info.total_stake == 900
        assert validator_info.self_stake == 400
        assert validator_info.delegated_stake == 500
        assert validator_info.voting_power == 900

    def test_remove_stake_delegated_stake(self, validator_info):
        """Test removing delegated stake."""
        validator_info.remove_stake(100, is_self_stake=False)
        assert validator_info.total_stake == 900
        assert validator_info.self_stake == 500
        assert validator_info.delegated_stake == 400
        assert validator_info.voting_power == 900

    def test_remove_stake_negative_amount(self, validator_info):
        """Test removing stake that would result in negative amount."""
        validator_info.remove_stake(1500, is_self_stake=True)
        assert validator_info.total_stake == 0
        assert validator_info.self_stake == 0
        assert validator_info.delegated_stake == 500
        assert validator_info.voting_power == 0

    def test_slash_validator(self, validator_info):
        """Test slashing validator."""
        slashed_amount = validator_info.slash(0.1)  # 10% slash
        assert slashed_amount == 100
        assert validator_info.total_stake == 900
        assert validator_info.slashing_count == 1
        assert validator_info.voting_power == 900

    def test_slash_validator_high_percentage(self, validator_info):
        """Test slashing validator with high percentage."""
        slashed_amount = validator_info.slash(0.5)  # 50% slash
        assert slashed_amount == 500
        assert validator_info.total_stake == 500
        assert validator_info.slashing_count == 1
        assert validator_info.voting_power == 500

    def test_slash_validator_100_percent(self, validator_info):
        """Test slashing validator with 100% percentage."""
        slashed_amount = validator_info.slash(1.0)  # 100% slash
        assert slashed_amount == 1000
        assert validator_info.total_stake == 0
        assert validator_info.slashing_count == 1
        assert validator_info.voting_power == 0

    def test_add_rewards(self, validator_info):
        """Test adding rewards."""
        validator_info.add_rewards(50)
        assert validator_info.total_rewards == 150

    def test_to_dict(self, validator_info, mock_public_key):
        """Test converting to dictionary."""
        result = validator_info.to_dict()
        expected = {
            "validator_id": "validator_1",
            "public_key": "mock_public_key_hex",
            "status": "active",
            "role": "validator",
            "total_stake": 1000,
            "self_stake": 500,
            "delegated_stake": 500,
            "voting_power": 1000,
            "commission_rate": 0.1,
            "created_at": 1234567890.0,
            "last_active": 1234567890.0,
            "slashing_count": 0,
            "total_rewards": 100,
            "metadata": {"key": "value"},
        }
        assert result == expected

    def test_from_dict(self, mock_public_key):
        """Test creating from dictionary."""
        data = {
            "validator_id": "validator_1",
            "public_key": "mock_public_key_hex",
            "status": "active",
            "role": "validator",
            "total_stake": 1000,
            "self_stake": 500,
            "delegated_stake": 500,
            "voting_power": 1000,
            "commission_rate": 0.1,
            "created_at": 1234567890.0,
            "last_active": 1234567890.0,
            "slashing_count": 0,
            "total_rewards": 100,
            "metadata": {"key": "value"},
        }

        with patch.object(PublicKey, "from_hex", return_value=mock_public_key):
            validator_info = ValidatorInfo.from_dict(data)
            assert validator_info.validator_id == "validator_1"
            assert validator_info.status == ValidatorStatus.ACTIVE
            assert validator_info.role == ValidatorRole.VALIDATOR
            assert validator_info.total_stake == 1000
            assert validator_info.self_stake == 500
            assert validator_info.delegated_stake == 500
            assert validator_info.voting_power == 1000
            assert validator_info.commission_rate == 0.1
            assert validator_info.slashing_count == 0
            assert validator_info.total_rewards == 100
            assert validator_info.metadata == {"key": "value"}

    def test_from_dict_default_metadata(self, mock_public_key):
        """Test creating from dictionary with default metadata."""
        data = {
            "validator_id": "validator_1",
            "public_key": "mock_public_key_hex",
            "status": "active",
            "role": "validator",
            "total_stake": 1000,
            "self_stake": 500,
            "delegated_stake": 500,
            "voting_power": 1000,
            "commission_rate": 0.1,
            "created_at": 1234567890.0,
            "last_active": 1234567890.0,
            "slashing_count": 0,
            "total_rewards": 100,
        }

        with patch.object(PublicKey, "from_hex", return_value=mock_public_key):
            validator_info = ValidatorInfo.from_dict(data)
            assert validator_info.metadata == {}


class TestValidator:
    """Test Validator class."""

    @pytest.fixture
    def mock_private_key(self):
        """Fixture for mock private key."""
        mock_key = Mock(spec=PrivateKey)
        mock_public_key = Mock(spec=PublicKey)
        mock_key.get_public_key.return_value = mock_public_key
        return mock_key

    @pytest.fixture
    def validator(self, mock_private_key):
        """Fixture for validator."""
        return Validator("validator_1", mock_private_key, 0.15)

    def test_validator_creation(self, validator, mock_private_key):
        """Test creating validator."""
        assert validator.validator_id == "validator_1"
        assert validator.private_key == mock_private_key
        assert validator.commission_rate == 0.15
        assert validator.is_active is False
        assert validator.info.validator_id == "validator_1"
        assert validator.info.commission_rate == 0.15

    def test_validator_default_commission(self, mock_private_key):
        """Test validator with default commission rate."""
        validator = Validator("validator_1", mock_private_key)
        assert validator.commission_rate == 0.1

    @patch("time.time")
    def test_activate(self, mock_time, validator):
        """Test activating validator."""
        mock_time.return_value = 1234567891.0
        validator.activate()
        assert validator.is_active is True
        assert validator.info.status == ValidatorStatus.ACTIVE
        assert validator.last_heartbeat == 1234567891.0

    def test_deactivate(self, validator):
        """Test deactivating validator."""
        validator.activate()
        validator.deactivate()
        assert validator.is_active is False
        assert validator.info.status == ValidatorStatus.INACTIVE

    def test_jail(self, validator):
        """Test jailing validator."""
        validator.activate()
        validator.jail()
        assert validator.is_active is False
        assert validator.info.status == ValidatorStatus.JAILED

    def test_unjail(self, validator):
        """Test unjailing validator."""
        validator.jail()
        validator.unjail()
        assert validator.is_active is True
        assert validator.info.status == ValidatorStatus.ACTIVE

    def test_sign_message(self, validator, mock_private_key):
        """Test signing message."""
        mock_signature = Mock(spec=Signature)
        mock_private_key.sign.return_value = mock_signature
        message = b"test message"
        result = validator.sign_message(message)
        assert result == mock_signature
        mock_private_key.sign.assert_called_once_with(message)

    def test_verify_signature(self, validator):
        """Test verifying signature."""
        mock_signature = Mock(spec=Signature)
        message = b"test message"
        validator.public_key.verify.return_value = True
        result = validator.verify_signature(message, mock_signature)
        assert result is True
        validator.public_key.verify.assert_called_once_with(mock_signature, message)

    @patch("time.time")
    def test_update_heartbeat(self, mock_time, validator):
        """Test updating heartbeat."""
        mock_time.return_value = 1234567891.0
        validator.update_heartbeat()
        assert validator.last_heartbeat == 1234567891.0
        assert validator.info.last_active == 1234567891.0

    @patch("time.time")
    def test_is_online_within_timeout(self, mock_time, validator):
        """Test checking if validator is online within timeout."""
        mock_time.return_value = 1234567890.0
        validator.last_heartbeat = 1234567860.0  # 30 seconds ago
        result = validator.is_online(60.0)  # 60 second timeout
        assert result is True

    @patch("time.time")
    def test_is_online_exceeds_timeout(self, mock_time, validator):
        """Test checking if validator is online exceeds timeout."""
        mock_time.return_value = 1234567890.0
        validator.last_heartbeat = 1234567860.0  # 30 seconds ago
        result = validator.is_online(20.0)  # 20 second timeout
        assert result is False

    @patch("time.time")
    def test_is_online_default_timeout(self, mock_time, validator):
        """Test checking if validator is online with default timeout."""
        mock_time.return_value = 1234567890.0
        validator.last_heartbeat = 1234567861.0  # 29 seconds ago (within 30 second timeout)
        result = validator.is_online()  # Default 30 second timeout
        assert result is True


class TestValidatorSet:
    """Test ValidatorSet class."""

    @pytest.fixture
    def validator_set(self):
        """Fixture for validator set."""
        return ValidatorSet(max_validators=5)

    @pytest.fixture
    def mock_validator_info(self):
        """Fixture for mock validator info."""
        mock_public_key = Mock(spec=PublicKey)
        return ValidatorInfo(
            validator_id="validator_1",
            public_key=mock_public_key,
            status=ValidatorStatus.ACTIVE,
            voting_power=1000,
        )

    def test_validator_set_creation(self, validator_set):
        """Test creating validator set."""
        assert validator_set.max_validators == 5
        assert validator_set.validators == {}
        assert validator_set.active_validators == set()
        assert validator_set.proposer_rotation == []
        assert validator_set.current_proposer_index == 0

    def test_validator_set_default_max_validators(self):
        """Test validator set with default max validators."""
        validator_set = ValidatorSet()
        assert validator_set.max_validators == 100

    def test_add_validator_success(self, validator_set, mock_validator_info):
        """Test adding validator successfully."""
        result = validator_set.add_validator(mock_validator_info)
        assert result is True
        assert "validator_1" in validator_set.validators
        assert "validator_1" in validator_set.active_validators
        assert "validator_1" in validator_set.proposer_rotation

    def test_add_validator_inactive(self, validator_set):
        """Test adding inactive validator."""
        mock_public_key = Mock(spec=PublicKey)
        inactive_validator = ValidatorInfo(
            validator_id="validator_1",
            public_key=mock_public_key,
            status=ValidatorStatus.INACTIVE,
            voting_power=1000,
        )
        result = validator_set.add_validator(inactive_validator)
        assert result is True
        assert "validator_1" in validator_set.validators
        assert "validator_1" not in validator_set.active_validators
        assert "validator_1" not in validator_set.proposer_rotation

    def test_add_validator_max_reached(self, validator_set):
        """Test adding validator when max validators reached."""
        # Fill up the validator set
        for i in range(5):
            mock_public_key = Mock(spec=PublicKey)
            validator_info = ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                status=ValidatorStatus.ACTIVE,
                voting_power=1000,
            )
            validator_set.add_validator(validator_info)

        # Try to add one more
        mock_public_key = Mock(spec=PublicKey)
        extra_validator = ValidatorInfo(
            validator_id="validator_5",
            public_key=mock_public_key,
            status=ValidatorStatus.ACTIVE,
            voting_power=1000,
        )
        result = validator_set.add_validator(extra_validator)
        assert result is False
        assert "validator_5" not in validator_set.validators

    def test_remove_validator_success(self, validator_set, mock_validator_info):
        """Test removing validator successfully."""
        validator_set.add_validator(mock_validator_info)
        result = validator_set.remove_validator("validator_1")
        assert result is True
        assert "validator_1" not in validator_set.validators
        assert "validator_1" not in validator_set.active_validators
        assert "validator_1" not in validator_set.proposer_rotation

    def test_remove_validator_not_found(self, validator_set):
        """Test removing non-existent validator."""
        result = validator_set.remove_validator("nonexistent")
        assert result is False

    def test_update_validator_status_success(self, validator_set, mock_validator_info):
        """Test updating validator status successfully."""
        validator_set.add_validator(mock_validator_info)
        result = validator_set.update_validator_status(
            "validator_1", ValidatorStatus.INACTIVE
        )
        assert result is True
        assert validator_set.validators["validator_1"].status == ValidatorStatus.INACTIVE
        assert "validator_1" not in validator_set.active_validators

    def test_update_validator_status_to_active(self, validator_set):
        """Test updating validator status to active."""
        mock_public_key = Mock(spec=PublicKey)
        inactive_validator = ValidatorInfo(
            validator_id="validator_1",
            public_key=mock_public_key,
            status=ValidatorStatus.INACTIVE,
            voting_power=1000,
        )
        validator_set.add_validator(inactive_validator)
        result = validator_set.update_validator_status(
            "validator_1", ValidatorStatus.ACTIVE
        )
        assert result is True
        assert validator_set.validators["validator_1"].status == ValidatorStatus.ACTIVE
        assert "validator_1" in validator_set.active_validators

    def test_update_validator_status_not_found(self, validator_set):
        """Test updating status of non-existent validator."""
        result = validator_set.update_validator_status(
            "nonexistent", ValidatorStatus.ACTIVE
        )
        assert result is False

    def test_get_next_proposer_single_validator(self, validator_set, mock_validator_info):
        """Test getting next proposer with single validator."""
        validator_set.add_validator(mock_validator_info)
        proposer = validator_set.get_next_proposer()
        assert proposer == "validator_1"

    def test_get_next_proposer_rotation(self, validator_set):
        """Test proposer rotation with multiple validators."""
        # Add multiple validators with different voting powers
        for i, power in enumerate([1000, 2000, 1500]):
            mock_public_key = Mock(spec=PublicKey)
            validator_info = ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                status=ValidatorStatus.ACTIVE,
                voting_power=power,
            )
            validator_set.add_validator(validator_info)

        # Should be sorted by voting power: validator_1 (2000), validator_2 (1500), validator_0 (1000)
        proposer1 = validator_set.get_next_proposer()
        proposer2 = validator_set.get_next_proposer()
        proposer3 = validator_set.get_next_proposer()
        proposer4 = validator_set.get_next_proposer()

        assert proposer1 == "validator_1"  # Highest voting power
        assert proposer2 == "validator_2"  # Second highest
        assert proposer3 == "validator_0"  # Lowest
        assert proposer4 == "validator_1"  # Rotates back to first

    def test_get_next_proposer_no_validators(self, validator_set):
        """Test getting next proposer with no validators."""
        proposer = validator_set.get_next_proposer()
        assert proposer is None

    def test_get_validator_by_power(self, validator_set):
        """Test getting validators sorted by voting power."""
        # Add validators with different voting powers
        for i, power in enumerate([1000, 3000, 2000]):
            mock_public_key = Mock(spec=PublicKey)
            validator_info = ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                status=ValidatorStatus.ACTIVE,
                voting_power=power,
            )
            validator_set.add_validator(validator_info)

        validators = validator_set.get_validator_by_power()
        assert len(validators) == 3
        assert validators[0].validator_id == "validator_1"  # 3000 power
        assert validators[1].validator_id == "validator_2"  # 2000 power
        assert validators[2].validator_id == "validator_0"  # 1000 power

    def test_get_validator_by_power_with_limit(self, validator_set):
        """Test getting validators sorted by voting power with limit."""
        # Add validators with different voting powers
        for i, power in enumerate([1000, 3000, 2000]):
            mock_public_key = Mock(spec=PublicKey)
            validator_info = ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                status=ValidatorStatus.ACTIVE,
                voting_power=power,
            )
            validator_set.add_validator(validator_info)

        validators = validator_set.get_validator_by_power(limit=2)
        assert len(validators) == 2
        assert validators[0].validator_id == "validator_1"  # 3000 power
        assert validators[1].validator_id == "validator_2"  # 2000 power

    def test_get_total_voting_power(self, validator_set):
        """Test getting total voting power."""
        # Add validators with different voting powers
        for i, power in enumerate([1000, 2000, 1500]):
            mock_public_key = Mock(spec=PublicKey)
            validator_info = ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                status=ValidatorStatus.ACTIVE,
                voting_power=power,
            )
            validator_set.add_validator(validator_info)

        total_power = validator_set.get_total_voting_power()
        assert total_power == 4500

    def test_get_total_voting_power_empty(self, validator_set):
        """Test getting total voting power with no validators."""
        total_power = validator_set.get_total_voting_power()
        assert total_power == 0

    def test_to_dict(self, validator_set, mock_validator_info):
        """Test converting to dictionary."""
        validator_set.add_validator(mock_validator_info)
        result = validator_set.to_dict()
        assert result["max_validators"] == 5
        assert "validator_1" in result["validators"]
        assert "validator_1" in result["active_validators"]
        assert "validator_1" in result["proposer_rotation"]
        assert result["current_proposer_index"] == 0

    def test_from_dict(self):
        """Test creating from dictionary."""
        mock_public_key = Mock(spec=PublicKey)
        data = {
            "max_validators": 5,
            "validators": {
                "validator_1": {
                    "validator_id": "validator_1",
                    "public_key": "mock_public_key_hex",
                    "status": "active",
                    "role": "validator",
                    "total_stake": 1000,
                    "self_stake": 500,
                    "delegated_stake": 500,
                    "voting_power": 1000,
                    "commission_rate": 0.1,
                    "created_at": 1234567890.0,
                    "last_active": 1234567890.0,
                    "slashing_count": 0,
                    "total_rewards": 100,
                    "metadata": {},
                }
            },
            "active_validators": ["validator_1"],
            "proposer_rotation": ["validator_1"],
            "current_proposer_index": 0,
        }

        with patch.object(PublicKey, "from_hex", return_value=mock_public_key):
            validator_set = ValidatorSet.from_dict(data)
            assert validator_set.max_validators == 5
            assert "validator_1" in validator_set.validators
            assert "validator_1" in validator_set.active_validators
            assert "validator_1" in validator_set.proposer_rotation
            assert validator_set.current_proposer_index == 0


class TestValidatorManager:
    """Test ValidatorManager class."""

    @pytest.fixture
    def validator_set(self):
        """Fixture for validator set."""
        return ValidatorSet(max_validators=5)

    @pytest.fixture
    def validator_manager(self, validator_set):
        """Fixture for validator manager."""
        return ValidatorManager(validator_set)

    @pytest.fixture
    def mock_validator(self):
        """Fixture for mock validator."""
        mock_private_key = Mock(spec=PrivateKey)
        mock_public_key = Mock(spec=PublicKey)
        mock_private_key.get_public_key.return_value = mock_public_key
        return Validator("validator_1", mock_private_key, 0.1)

    def test_validator_manager_creation(self, validator_manager):
        """Test creating validator manager."""
        assert validator_manager.validator_set is not None
        assert validator_manager.staking_pools == {}
        assert validator_manager.slashing_events == []
        assert validator_manager.reward_pool == 0

    def test_register_validator_success(self, validator_manager, mock_validator):
        """Test registering validator successfully."""
        result = validator_manager.register_validator(mock_validator, 1000)
        assert result is True
        assert mock_validator.is_active is True
        assert "validator_1" in validator_manager.validator_set.validators
        assert "validator_1" in validator_manager.staking_pools

    def test_register_validator_no_initial_stake(self, validator_manager, mock_validator):
        """Test registering validator with no initial stake."""
        result = validator_manager.register_validator(mock_validator)
        assert result is True
        assert mock_validator.is_active is True
        assert "validator_1" in validator_manager.validator_set.validators

    def test_register_validator_max_reached(self, validator_manager, mock_validator):
        """Test registering validator when max validators reached."""
        # Fill up the validator set
        for i in range(5):
            mock_private_key = Mock(spec=PrivateKey)
            mock_public_key = Mock(spec=PublicKey)
            mock_private_key.get_public_key.return_value = mock_public_key
            validator = Validator(f"validator_{i}", mock_private_key)
            validator_manager.register_validator(validator)

        # Try to register one more
        result = validator_manager.register_validator(mock_validator)
        assert result is False
        assert not mock_validator.is_active

    def test_stake_success(self, validator_manager, mock_validator):
        """Test staking tokens successfully."""
        validator_manager.register_validator(mock_validator, 1000)
        result = validator_manager.stake("validator_1", "delegator_1", 500)
        assert result is True
        assert len(validator_manager.staking_pools["validator_1"]) == 1
        staking_info = validator_manager.staking_pools["validator_1"][0]
        assert staking_info.validator_id == "validator_1"
        assert staking_info.delegator_id == "delegator_1"
        assert staking_info.amount == 500

    def test_stake_validator_not_found(self, validator_manager):
        """Test staking to non-existent validator."""
        result = validator_manager.stake("nonexistent", "delegator_1", 500)
        assert result is False

    def test_stake_multiple_delegators(self, validator_manager, mock_validator):
        """Test staking from multiple delegators."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        validator_manager.stake("validator_1", "delegator_2", 300)
        assert len(validator_manager.staking_pools["validator_1"]) == 2

    def test_unstake_success(self, validator_manager, mock_validator):
        """Test unstaking tokens successfully."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        result = validator_manager.unstake("validator_1", "delegator_1", 200)
        assert result is True
        staking_info = validator_manager.staking_pools["validator_1"][0]
        assert staking_info.amount == 300

    def test_unstake_complete(self, validator_manager, mock_validator):
        """Test unstaking all tokens."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        result = validator_manager.unstake("validator_1", "delegator_1", 500)
        assert result is True
        assert len(validator_manager.staking_pools["validator_1"]) == 0

    def test_unstake_validator_not_found(self, validator_manager):
        """Test unstaking from non-existent validator."""
        result = validator_manager.unstake("nonexistent", "delegator_1", 200)
        assert result is False

    def test_unstake_delegator_not_found(self, validator_manager, mock_validator):
        """Test unstaking from non-existent delegator."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        result = validator_manager.unstake("validator_1", "delegator_2", 200)
        assert result is False

    def test_unstake_insufficient_amount(self, validator_manager, mock_validator):
        """Test unstaking more than available."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        result = validator_manager.unstake("validator_1", "delegator_1", 600)
        assert result is False

    @patch("time.time")
    def test_slash_validator_success(self, mock_time, validator_manager, mock_validator):
        """Test slashing validator successfully."""
        mock_time.return_value = 1234567890.0
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        slashed_amount = validator_manager.slash_validator("validator_1", 0.1, "test reason")
        assert slashed_amount == 150  # 10% of 1500 total stake
        assert len(validator_manager.slashing_events) == 1
        slashing_event = validator_manager.slashing_events[0]
        assert slashing_event["validator_id"] == "validator_1"
        assert slashing_event["percentage"] == 0.1
        assert slashing_event["amount"] == 150
        assert slashing_event["reason"] == "test reason"
        assert slashing_event["timestamp"] == 1234567890.0

    def test_slash_validator_not_found(self, validator_manager):
        """Test slashing non-existent validator."""
        slashed_amount = validator_manager.slash_validator("nonexistent", 0.1, "test reason")
        assert slashed_amount == 0

    def test_slash_validator_jail_after_multiple_slashings(self, validator_manager, mock_validator):
        """Test jailing validator after multiple slashings."""
        validator_manager.register_validator(mock_validator, 1000)
        # Slash 3 times
        validator_manager.slash_validator("validator_1", 0.1, "reason 1")
        validator_manager.slash_validator("validator_1", 0.1, "reason 2")
        validator_manager.slash_validator("validator_1", 0.1, "reason 3")
        assert validator_manager.validator_set.validators["validator_1"].status == ValidatorStatus.JAILED

    def test_distribute_rewards_success(self, validator_manager, mock_validator):
        """Test distributing rewards successfully."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        rewards = validator_manager.distribute_rewards(1000)
        assert "validator_1" in rewards
        # Validator has 1500 total stake, should get all rewards
        assert rewards["validator_1"] == 900  # 1000 - 100 (10% commission)

    def test_distribute_rewards_multiple_validators(self, validator_manager):
        """Test distributing rewards to multiple validators."""
        # Register two validators with different stakes
        mock_private_key1 = Mock(spec=PrivateKey)
        mock_public_key1 = Mock(spec=PublicKey)
        mock_private_key1.get_public_key.return_value = mock_public_key1
        validator1 = Validator("validator_1", mock_private_key1, 0.1)
        validator_manager.register_validator(validator1, 1000)

        mock_private_key2 = Mock(spec=PrivateKey)
        mock_public_key2 = Mock(spec=PublicKey)
        mock_private_key2.get_public_key.return_value = mock_public_key2
        validator2 = Validator("validator_2", mock_private_key2, 0.1)
        validator_manager.register_validator(validator2, 2000)

        rewards = validator_manager.distribute_rewards(1000)
        assert "validator_1" in rewards
        assert "validator_2" in rewards
        # validator_1: 1000/3000 * 1000 = 333, commission = 33, delegator = 300
        # validator_2: 2000/3000 * 1000 = 667, commission = 67, delegator = 600
        assert rewards["validator_1"] == 300
        assert rewards["validator_2"] == 600

    def test_distribute_rewards_no_active_validators(self, validator_manager):
        """Test distributing rewards with no active validators."""
        rewards = validator_manager.distribute_rewards(1000)
        assert rewards == {}

    def test_distribute_rewards_zero_total_power(self, validator_manager, mock_validator):
        """Test distributing rewards with zero total voting power."""
        validator_manager.register_validator(mock_validator, 0)  # No stake
        rewards = validator_manager.distribute_rewards(1000)
        assert rewards == {}

    @patch("time.time")
    def test_get_validator_metrics(self, mock_time, validator_manager, mock_validator):
        """Test getting validator metrics."""
        mock_time.return_value = 1234567890.0
        validator_manager.register_validator(mock_validator, 1000)
        metrics = validator_manager.get_validator_metrics()
        assert isinstance(metrics, ConsensusMetrics)
        assert metrics.validator_count == 1
        assert metrics.active_validators == 1
        assert metrics.last_updated == 1234567890.0

    def test_to_dict(self, validator_manager, mock_validator):
        """Test converting to dictionary."""
        validator_manager.register_validator(mock_validator, 1000)
        validator_manager.stake("validator_1", "delegator_1", 500)
        result = validator_manager.to_dict()
        assert "validator_set" in result
        assert "staking_pools" in result
        assert "slashing_events" in result
        assert "reward_pool" in result
        assert "validator_1" in result["staking_pools"]

    def test_from_dict(self):
        """Test creating from dictionary."""
        mock_public_key = Mock(spec=PublicKey)
        data = {
            "validator_set": {
                "max_validators": 5,
                "validators": {
                    "validator_1": {
                        "validator_id": "validator_1",
                        "public_key": "mock_public_key_hex",
                        "status": "active",
                        "role": "validator",
                        "total_stake": 1000,
                        "self_stake": 500,
                        "delegated_stake": 500,
                        "voting_power": 1000,
                        "commission_rate": 0.1,
                        "created_at": 1234567890.0,
                        "last_active": 1234567890.0,
                        "slashing_count": 0,
                        "total_rewards": 100,
                        "metadata": {},
                    }
                },
                "active_validators": ["validator_1"],
                "proposer_rotation": ["validator_1"],
                "current_proposer_index": 0,
            },
            "staking_pools": {
                "validator_1": [
                    {
                        "validator_id": "validator_1",
                        "delegator_id": "delegator_1",
                        "amount": 500,
                    }
                ]
            },
            "slashing_events": [],
            "reward_pool": 0,
        }

        with patch.object(PublicKey, "from_hex", return_value=mock_public_key):
            manager = ValidatorManager.from_dict(data)
            assert "validator_1" in manager.validator_set.validators
            assert "validator_1" in manager.staking_pools
            assert len(manager.staking_pools["validator_1"]) == 1
            assert manager.slashing_events == []
            assert manager.reward_pool == 0
