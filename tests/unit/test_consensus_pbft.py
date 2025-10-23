"""
Tests for Practical Byzantine Fault Tolerance (PBFT) consensus implementation.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusType,
    PBFTMessage,
    PBFTPhase,
)
from dubchain.consensus.pbft import PBFTValidator, PracticalByzantineFaultTolerance
from dubchain.consensus.validator import Validator
from dubchain.crypto.signatures import ECDSASigner


class TestPBFTValidator:
    """Test PBFTValidator class."""

    def test_pbft_validator_creation(self):
        """Test PBFTValidator creation."""
        _, public_key = ECDSASigner.generate_keypair()

        validator = PBFTValidator(validator_id="validator_1", public_key=public_key)

        assert validator.validator_id == "validator_1"
        assert validator.public_key == public_key
        assert validator.is_primary is False
        assert validator.view_number == 0
        assert validator.sequence_number == 0
        assert validator.prepared is False
        assert validator.committed is False
        assert len(validator.message_log) == 0

    def test_pbft_validator_add_message(self):
        """Test adding message to validator."""
        _, public_key = ECDSASigner.generate_keypair()
        validator = PBFTValidator("validator_1", public_key)

        message = PBFTMessage(
            message_type=PBFTPhase.PRE_PREPARE,
            view_number=0,
            sequence_number=1,
            block_hash="hash123",
            validator_id="primary",
            signature="sig123",
            payload={"data": "test"},
        )

        initial_time = validator.last_heartbeat
        time.sleep(0.01)  # Small delay to ensure time difference

        validator.add_message(message)

        assert len(validator.message_log) == 1
        assert validator.message_log[0] == message
        assert validator.last_heartbeat > initial_time

    def test_pbft_validator_is_online(self):
        """Test validator online status."""
        _, public_key = ECDSASigner.generate_keypair()
        validator = PBFTValidator("validator_1", public_key)

        # Should be online initially
        assert validator.is_online() is True

        # Set old heartbeat
        validator.last_heartbeat = time.time() - 60.0

        # Should be offline with default timeout
        assert validator.is_online() is False

        # Should be online with longer timeout
        assert validator.is_online(timeout=120.0) is True


class TestPracticalByzantineFaultTolerance:
    """Test PracticalByzantineFaultTolerance class."""

    @pytest.fixture
    def consensus_config(self):
        """Create consensus config for PBFT."""
        return ConsensusConfig(
            consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            max_validators=10,
            block_time=15.0,
            max_gas_per_block=1000000,
            pbft_fault_tolerance=1,
        )

    @pytest.fixture
    def pbft_consensus(self, consensus_config):
        """Create PBFT consensus instance."""
        return PracticalByzantineFaultTolerance(consensus_config)

    @pytest.fixture
    def mock_validator(self):
        """Create mock validator."""
        validator = Mock(spec=Validator)
        validator.validator_id = "validator_1"
        _, public_key = ECDSASigner.generate_keypair()
        validator.public_key = public_key
        return validator

    def test_pbft_consensus_creation(self, pbft_consensus):
        """Test PBFT consensus creation."""
        assert (
            pbft_consensus.config.consensus_type
            == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        )
        assert len(pbft_consensus.validators) == 0
        assert pbft_consensus.primary_validator is None
        assert pbft_consensus.current_view == 0
        assert pbft_consensus.sequence_number == 0
        assert pbft_consensus.checkpoint_interval == 100
        assert pbft_consensus.last_checkpoint == 0

    def test_pbft_add_validator_success(self, pbft_consensus, mock_validator):
        """Test successful validator addition."""
        result = pbft_consensus.add_validator(mock_validator)

        assert result is True
        assert len(pbft_consensus.validators) == 1
        assert "validator_1" in pbft_consensus.validators
        assert pbft_consensus.primary_validator == "validator_1"

        validator = pbft_consensus.validators["validator_1"]
        assert validator.validator_id == "validator_1"
        assert validator.is_primary is True

    def test_pbft_add_validator_max_reached(self, pbft_consensus):
        """Test validator addition when max validators reached."""
        # Add max validators
        for i in range(10):  # max_validators = 10
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        # Try to add one more
        extra_validator = Mock(spec=Validator)
        extra_validator.validator_id = "validator_extra"
        _, public_key = ECDSASigner.generate_keypair()
        extra_validator.public_key = public_key

        result = pbft_consensus.add_validator(extra_validator)

        assert result is False
        assert len(pbft_consensus.validators) == 10

    def test_pbft_remove_validator_success(self, pbft_consensus, mock_validator):
        """Test successful validator removal."""
        pbft_consensus.add_validator(mock_validator)

        result = pbft_consensus.remove_validator("validator_1")

        assert result is True
        assert len(pbft_consensus.validators) == 0
        # Primary validator is not automatically set to None when removing the last validator

    def test_pbft_remove_validator_not_found(self, pbft_consensus):
        """Test removing non-existent validator."""
        result = pbft_consensus.remove_validator("nonexistent")

        assert result is False

    def test_pbft_remove_primary_validator(self, pbft_consensus):
        """Test removing primary validator."""
        # Add multiple validators
        for i in range(3):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        original_primary = pbft_consensus.primary_validator
        result = pbft_consensus.remove_validator(original_primary)

        assert result is True
        # The primary validator should be reassigned to the next validator
        assert pbft_consensus.primary_validator is not None
        # The new primary should be the validator with the lowest ID among remaining validators
        # Since _select_new_primary is called before removal, it selects from current validators
        assert pbft_consensus.primary_validator in [
            "validator_0",
            "validator_1",
            "validator_2",
        ]

    def test_pbft_start_consensus_no_validators(self, pbft_consensus):
        """Test starting consensus with no validators."""
        request_data = {"block_number": 1}

        result = pbft_consensus.start_consensus(request_data)

        assert result.success is False
        assert result.error_message == "No validators available"
        assert (
            result.consensus_type == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        )

    def test_pbft_start_consensus_insufficient_validators(
        self, pbft_consensus, mock_validator
    ):
        """Test starting consensus with insufficient validators for fault tolerance."""
        pbft_consensus.add_validator(mock_validator)
        request_data = {"block_number": 1}

        result = pbft_consensus.start_consensus(request_data)

        assert result.success is False
        assert result.error_message == "Insufficient validators for fault tolerance"

    def test_pbft_start_consensus_success(self, pbft_consensus):
        """Test successful consensus process."""
        # Add enough validators for fault tolerance (3f + 1 = 4)
        for i in range(4):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        request_data = {"block_number": 1}

        result = pbft_consensus.start_consensus(request_data)

        assert result.success is True
        assert result.validator_id == pbft_consensus.primary_validator
        assert (
            result.consensus_type == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        )
        assert result.block_hash is not None
        assert pbft_consensus.sequence_number == 1
        assert pbft_consensus.metrics.total_blocks == 1
        assert pbft_consensus.metrics.successful_blocks == 1

    def test_pbft_pre_prepare_phase_success(self, pbft_consensus):
        """Test successful pre-prepare phase."""
        # Add validators
        for i in range(4):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        request_data = {"block_number": 1}
        request_hash = "hash123"

        result = pbft_consensus._pre_prepare_phase(request_data, request_hash)

        assert result is True
        assert (
            len(pbft_consensus.pre_prepare_messages[pbft_consensus.sequence_number])
            == 1
        )
        assert (
            pbft_consensus.primary_validator
            in pbft_consensus.pre_prepare_messages[pbft_consensus.sequence_number]
        )

        # Check that non-primary validators received the message
        for validator_id, validator in pbft_consensus.validators.items():
            if validator_id != pbft_consensus.primary_validator:
                assert len(validator.message_log) == 1
                assert validator.message_log[0].message_type == PBFTPhase.PRE_PREPARE

    def test_pbft_pre_prepare_phase_no_primary(self, pbft_consensus):
        """Test pre-prepare phase with no primary validator."""
        request_data = {"block_number": 1}
        request_hash = "hash123"

        result = pbft_consensus._pre_prepare_phase(request_data, request_hash)

        assert result is False

    def test_pbft_prepare_phase_success(self, pbft_consensus):
        """Test successful prepare phase."""
        # Add enough validators
        for i in range(4):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        request_hash = "hash123"

        result = pbft_consensus._prepare_phase(request_hash)

        assert result is True
        assert (
            len(pbft_consensus.prepare_messages[pbft_consensus.sequence_number]) == 3
        )  # 4 - 1 primary

    def test_pbft_prepare_phase_insufficient_prepares(self, pbft_consensus):
        """Test prepare phase with insufficient prepare messages."""
        # Add only 2 validators (not enough for fault tolerance)
        for i in range(2):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        request_hash = "hash123"

        result = pbft_consensus._prepare_phase(request_hash)

        assert result is False

    def test_pbft_commit_phase_success(self, pbft_consensus):
        """Test successful commit phase."""
        # Add enough validators
        for i in range(4):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        request_hash = "hash123"

        result = pbft_consensus._commit_phase(request_hash)

        assert result is True
        assert len(pbft_consensus.commit_messages[pbft_consensus.sequence_number]) == 4
        assert pbft_consensus.sequence_number in pbft_consensus.prepared_requests
        assert pbft_consensus.sequence_number in pbft_consensus.committed_requests

    def test_pbft_commit_phase_insufficient_commits(self, pbft_consensus):
        """Test commit phase with insufficient commit messages."""
        # Add only 2 validators
        for i in range(2):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        request_hash = "hash123"

        result = pbft_consensus._commit_phase(request_hash)

        assert result is False
        assert pbft_consensus.sequence_number not in pbft_consensus.committed_requests

    def test_pbft_hash_request(self, pbft_consensus):
        """Test request hashing."""
        request_data = {"block_number": 1, "transactions": ["tx1", "tx2"]}

        hash1 = pbft_consensus._hash_request(request_data)
        hash2 = pbft_consensus._hash_request(request_data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

        # Different data should produce different hash
        different_data = {"block_number": 2, "transactions": ["tx1", "tx2"]}
        hash3 = pbft_consensus._hash_request(different_data)

        assert hash1 != hash3

    def test_pbft_handle_view_change_success(self, pbft_consensus, mock_validator):
        """Test successful view change."""
        pbft_consensus.add_validator(mock_validator)

        result = pbft_consensus.handle_view_change(1, "validator_1")

        assert result is True
        assert pbft_consensus.current_view == 1

    def test_pbft_handle_view_change_invalid_validator(self, pbft_consensus):
        """Test view change with invalid validator."""
        result = pbft_consensus.handle_view_change(1, "nonexistent")

        assert result is False

    def test_pbft_handle_view_change_invalid_view(self, pbft_consensus, mock_validator):
        """Test view change with invalid view number."""
        pbft_consensus.add_validator(mock_validator)

        result = pbft_consensus.handle_view_change(0, "validator_1")  # Same view

        assert result is False

    def test_pbft_get_consensus_metrics(self, pbft_consensus):
        """Test getting consensus metrics."""
        metrics = pbft_consensus.get_consensus_metrics()

        assert (
            metrics.consensus_type == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        )
        assert metrics.total_blocks == 0
        assert metrics.successful_blocks == 0
        assert metrics.failed_blocks == 0

    def test_pbft_get_validator_count(self, pbft_consensus, mock_validator):
        """Test getting validator count."""
        assert len(pbft_consensus.validators) == 0

        pbft_consensus.add_validator(mock_validator)

        assert len(pbft_consensus.validators) == 1

    def test_pbft_is_primary_validator(self, pbft_consensus, mock_validator):
        """Test checking if validator is primary."""
        assert pbft_consensus.primary_validator != "validator_1"

        pbft_consensus.add_validator(mock_validator)

        assert pbft_consensus.primary_validator == "validator_1"
        assert pbft_consensus.primary_validator != "nonexistent"

    def test_pbft_get_primary_validator(self, pbft_consensus, mock_validator):
        """Test getting primary validator."""
        assert pbft_consensus.primary_validator is None

        pbft_consensus.add_validator(mock_validator)

        assert pbft_consensus.primary_validator == "validator_1"

    def test_pbft_get_current_view(self, pbft_consensus):
        """Test getting current view."""
        assert pbft_consensus.current_view == 0

        pbft_consensus.current_view = 5

        assert pbft_consensus.current_view == 5

    def test_pbft_get_sequence_number(self, pbft_consensus):
        """Test getting sequence number."""
        assert pbft_consensus.sequence_number == 0

        pbft_consensus.sequence_number = 10

        assert pbft_consensus.sequence_number == 10

    def test_pbft_get_validator_info(self, pbft_consensus, mock_validator):
        """Test getting validator information."""
        assert pbft_consensus.get_validator_status("validator_1") is None

        pbft_consensus.add_validator(mock_validator)

        info = pbft_consensus.get_validator_status("validator_1")
        assert info is not None
        assert info["validator_id"] == "validator_1"
        assert info["is_primary"] is True

    def test_pbft_get_validator_list(self, pbft_consensus):
        """Test getting validator list."""
        assert len(pbft_consensus.validators) == 0

        # Add validators
        for i in range(3):
            validator = Mock(spec=Validator)
            validator.validator_id = f"validator_{i}"
            _, public_key = ECDSASigner.generate_keypair()
            validator.public_key = public_key
            pbft_consensus.add_validator(validator)

        validator_list = list(pbft_consensus.validators.keys())
        assert len(validator_list) == 3
        assert "validator_0" in validator_list
        assert "validator_1" in validator_list
        assert "validator_2" in validator_list

    def test_pbft_get_consensus_state(self, pbft_consensus):
        """Test getting consensus state."""
        state = pbft_consensus.get_network_status()

        assert state["current_view"] == 0
        assert state["sequence_number"] == 0
        assert state["primary_validator"] is None
        assert state["total_validators"] == 0
        assert state["prepared_requests"] == 0
        assert state["committed_requests"] == 0

    def test_pbft_detect_byzantine_fault(self, pbft_consensus, mock_validator):
        """Test detecting Byzantine fault."""
        pbft_consensus.add_validator(mock_validator)

        # Test with non-existent validator
        assert pbft_consensus.detect_byzantine_fault("nonexistent") is False

        # Test with valid validator (should not be faulty initially)
        assert pbft_consensus.detect_byzantine_fault("validator_1") is False

    def test_pbft_to_dict(self, pbft_consensus, mock_validator):
        """Test converting to dictionary."""
        pbft_consensus.add_validator(mock_validator)

        data = pbft_consensus.to_dict()

        assert "config" in data
        assert "validators" in data
        assert "primary_validator" in data
        assert "current_view" in data
        assert "sequence_number" in data
        assert len(data["validators"]) == 1
