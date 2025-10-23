"""Tests for sharding consensus module."""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.sharding.shard_consensus import (
    ShardCommittee,
    ShardConsensus,
    ShardProposer,
    ShardValidator,
)
from dubchain.sharding.shard_types import ShardId, ShardMetrics


class TestShardValidator:
    """Test ShardValidator functionality."""

    def test_shard_validator_creation(self):
        """Test creating shard validator."""
        validator = ShardValidator(validator_id="validator_1", shard_id=ShardId.SHARD_1)

        assert validator.validator_id == "validator_1"
        assert validator.shard_id == ShardId.SHARD_1
        assert validator.is_active == True
        assert validator.last_heartbeat > 0

    def test_shard_validator_custom_values(self):
        """Test creating shard validator with custom values."""
        validator = ShardValidator(
            validator_id="validator_2", shard_id=ShardId.SHARD_2, is_active=False
        )

        assert validator.validator_id == "validator_2"
        assert validator.shard_id == ShardId.SHARD_2
        assert validator.is_active == False

    def test_update_heartbeat(self):
        """Test updating validator heartbeat."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)
        initial_heartbeat = validator.last_heartbeat

        time.sleep(0.01)  # Small delay
        validator.update_heartbeat()

        assert validator.last_heartbeat > initial_heartbeat

    def test_is_online_within_timeout(self):
        """Test validator is online within timeout."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)

        assert validator.is_online(30.0) == True
        assert validator.is_online(1.0) == True

    def test_is_online_exceeds_timeout(self):
        """Test validator is offline when exceeding timeout."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)

        # Simulate old heartbeat
        validator.last_heartbeat = time.time() - 60.0

        assert validator.is_online(30.0) == False
        assert validator.is_online(1.0) == False

    def test_is_online_custom_timeout(self):
        """Test validator online check with custom timeout."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)

        # Set heartbeat to 5 seconds ago
        validator.last_heartbeat = time.time() - 5.0

        assert validator.is_online(10.0) == True
        assert validator.is_online(1.0) == False


class TestShardProposer:
    """Test ShardProposer functionality."""

    @pytest.fixture
    def proposer(self):
        """Fixture for shard proposer."""
        return ShardProposer(ShardId.SHARD_1)

    def test_shard_proposer_creation(self, proposer):
        """Test creating shard proposer."""
        assert proposer.shard_id == ShardId.SHARD_1
        assert proposer.current_proposer is None
        assert proposer.proposer_rotation == []
        assert proposer.current_index == 0

    def test_get_next_proposer_empty_rotation(self, proposer):
        """Test getting next proposer with empty rotation."""
        proposer = proposer.get_next_proposer()
        assert proposer is None

    def test_get_next_proposer_single_validator(self, proposer):
        """Test getting next proposer with single validator."""
        proposer.proposer_rotation = ["validator_1"]

        proposer1 = proposer.get_next_proposer()
        proposer2 = proposer.get_next_proposer()

        assert proposer1 == "validator_1"
        assert proposer2 == "validator_1"  # Should cycle back

    def test_get_next_proposer_multiple_validators(self, proposer):
        """Test getting next proposer with multiple validators."""
        proposer.proposer_rotation = ["validator_1", "validator_2", "validator_3"]

        proposer1 = proposer.get_next_proposer()
        proposer2 = proposer.get_next_proposer()
        proposer3 = proposer.get_next_proposer()
        proposer4 = proposer.get_next_proposer()

        assert proposer1 == "validator_1"
        assert proposer2 == "validator_2"
        assert proposer3 == "validator_3"
        assert proposer4 == "validator_1"  # Should cycle back

    def test_update_rotation(self, proposer):
        """Test updating proposer rotation."""
        validators = ["validator_1", "validator_2", "validator_3"]

        proposer.update_rotation(validators)

        assert proposer.proposer_rotation == validators
        assert proposer.current_index == 0

    def test_update_rotation_resets_index(self, proposer):
        """Test that updating rotation resets current index."""
        proposer.proposer_rotation = ["validator_1", "validator_2"]
        proposer.current_index = 1  # Move to second validator

        new_validators = ["validator_3", "validator_4", "validator_5"]
        proposer.update_rotation(new_validators)

        assert proposer.current_index == 0
        assert proposer.get_next_proposer() == "validator_3"


class TestShardCommittee:
    """Test ShardCommittee functionality."""

    @pytest.fixture
    def committee(self):
        """Fixture for shard committee."""
        return ShardCommittee(ShardId.SHARD_1)

    def test_shard_committee_creation(self, committee):
        """Test creating shard committee."""
        assert committee.shard_id == ShardId.SHARD_1
        assert committee.validators == []
        assert isinstance(committee.proposer, ShardProposer)
        assert committee.proposer.shard_id == ShardId.BEACON_CHAIN  # Default

    def test_add_validator(self, committee):
        """Test adding validator to committee."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)

        committee.add_validator(validator)

        assert len(committee.validators) == 1
        assert committee.validators[0] == validator
        assert committee.proposer.proposer_rotation == ["validator_1"]

    def test_add_duplicate_validator(self, committee):
        """Test adding duplicate validator to committee."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)

        committee.add_validator(validator)
        committee.add_validator(validator)  # Add same validator again

        assert len(committee.validators) == 1  # Should not add duplicate

    def test_add_multiple_validators(self, committee):
        """Test adding multiple validators to committee."""
        validator1 = ShardValidator("validator_1", ShardId.SHARD_1)
        validator2 = ShardValidator("validator_2", ShardId.SHARD_1)

        committee.add_validator(validator1)
        committee.add_validator(validator2)

        assert len(committee.validators) == 2
        assert committee.proposer.proposer_rotation == ["validator_1", "validator_2"]

    def test_remove_validator(self, committee):
        """Test removing validator from committee."""
        validator = ShardValidator("validator_1", ShardId.SHARD_1)
        committee.add_validator(validator)

        result = committee.remove_validator("validator_1")

        assert result == True
        assert len(committee.validators) == 0
        assert committee.proposer.proposer_rotation == []

    def test_remove_nonexistent_validator(self, committee):
        """Test removing nonexistent validator from committee."""
        result = committee.remove_validator("nonexistent")

        assert result == False
        assert len(committee.validators) == 0

    def test_get_active_validators(self, committee):
        """Test getting active validators."""
        validator1 = ShardValidator("validator_1", ShardId.SHARD_1, is_active=True)
        validator2 = ShardValidator("validator_2", ShardId.SHARD_1, is_active=False)
        validator3 = ShardValidator("validator_3", ShardId.SHARD_1, is_active=True)

        committee.add_validator(validator1)
        committee.add_validator(validator2)
        committee.add_validator(validator3)

        active_validators = committee.get_active_validators()

        assert len(active_validators) == 2
        assert validator1 in active_validators
        assert validator3 in active_validators
        assert validator2 not in active_validators

    def test_get_active_validators_offline(self, committee):
        """Test getting active validators excludes offline ones."""
        validator1 = ShardValidator("validator_1", ShardId.SHARD_1, is_active=True)
        validator2 = ShardValidator("validator_2", ShardId.SHARD_1, is_active=True)

        # Make validator2 offline
        validator2.last_heartbeat = time.time() - 60.0

        committee.add_validator(validator1)
        committee.add_validator(validator2)

        active_validators = committee.get_active_validators()

        assert len(active_validators) == 1
        assert validator1 in active_validators
        assert validator2 not in active_validators


class TestShardConsensus:
    """Test ShardConsensus functionality."""

    @pytest.fixture
    def consensus(self):
        """Fixture for shard consensus."""
        return ShardConsensus(ShardId.SHARD_1)

    def test_shard_consensus_creation(self, consensus):
        """Test creating shard consensus."""
        assert consensus.shard_id == ShardId.SHARD_1
        assert isinstance(consensus.committee, ShardCommittee)
        assert consensus.committee.shard_id == ShardId.SHARD_1
        assert isinstance(consensus.metrics, ShardMetrics)
        assert consensus.metrics.shard_id == ShardId.SHARD_1
        assert consensus.current_epoch == 0

    def test_add_validator(self, consensus):
        """Test adding validator to shard consensus."""
        consensus.add_validator("validator_1")

        assert len(consensus.committee.validators) == 1
        assert consensus.committee.validators[0].validator_id == "validator_1"
        assert consensus.committee.validators[0].shard_id == ShardId.SHARD_1
        assert consensus.metrics.validator_count == 1
        assert consensus.metrics.active_validators == 1

    def test_add_multiple_validators(self, consensus):
        """Test adding multiple validators to shard consensus."""
        consensus.add_validator("validator_1")
        consensus.add_validator("validator_2")
        consensus.add_validator("validator_3")

        assert len(consensus.committee.validators) == 3
        assert consensus.metrics.validator_count == 3
        assert consensus.metrics.active_validators == 3

    def test_remove_validator(self, consensus):
        """Test removing validator from shard consensus."""
        consensus.add_validator("validator_1")
        consensus.add_validator("validator_2")

        result = consensus.remove_validator("validator_1")

        assert result == True
        assert len(consensus.committee.validators) == 1
        assert consensus.committee.validators[0].validator_id == "validator_2"
        assert consensus.metrics.validator_count == 1
        assert consensus.metrics.active_validators == 1

    def test_remove_nonexistent_validator(self, consensus):
        """Test removing nonexistent validator from shard consensus."""
        result = consensus.remove_validator("nonexistent")

        assert result == False
        assert len(consensus.committee.validators) == 0
        assert consensus.metrics.validator_count == 0

    def test_get_proposer(self, consensus):
        """Test getting proposer from shard consensus."""
        # No validators initially
        proposer = consensus.get_proposer()
        assert proposer is None

        # Add validators
        consensus.add_validator("validator_1")
        consensus.add_validator("validator_2")

        proposer1 = consensus.get_proposer()
        proposer2 = consensus.get_proposer()

        assert proposer1 == "validator_1"
        assert proposer2 == "validator_2"

    def test_update_metrics_successful_block(self, consensus):
        """Test updating metrics for successful block."""
        consensus.update_metrics(success=True, block_time=2.5, gas_used=1000000)

        assert consensus.metrics.total_blocks == 1
        assert consensus.metrics.successful_blocks == 1
        assert consensus.metrics.failed_blocks == 0
        assert consensus.metrics.average_block_time == 2.5
        assert consensus.metrics.average_gas_used == 1000000
        assert consensus.metrics.last_updated > 0

    def test_update_metrics_failed_block(self, consensus):
        """Test updating metrics for failed block."""
        consensus.update_metrics(success=False, block_time=1.0, gas_used=500000)

        assert consensus.metrics.total_blocks == 1
        assert consensus.metrics.successful_blocks == 0
        assert consensus.metrics.failed_blocks == 1
        assert consensus.metrics.average_block_time == 1.0
        assert consensus.metrics.average_gas_used == 500000

    def test_update_metrics_multiple_blocks(self, consensus):
        """Test updating metrics for multiple blocks."""
        # First block
        consensus.update_metrics(success=True, block_time=2.0, gas_used=800000)

        # Second block
        consensus.update_metrics(success=True, block_time=3.0, gas_used=1200000)

        # Third block (failed)
        consensus.update_metrics(success=False, block_time=1.5, gas_used=600000)

        assert consensus.metrics.total_blocks == 3
        assert consensus.metrics.successful_blocks == 2
        assert consensus.metrics.failed_blocks == 1
        assert (
            abs(consensus.metrics.average_block_time - 2.17) < 0.01
        )  # (2.0 + 3.0 + 1.5) / 3
        assert (
            abs(consensus.metrics.average_gas_used - 866667) < 1
        )  # (800000 + 1200000 + 600000) / 3

    def test_get_metrics(self, consensus):
        """Test getting shard metrics."""
        consensus.add_validator("validator_1")
        consensus.update_metrics(success=True, block_time=2.0, gas_used=1000000)

        metrics = consensus.get_metrics()

        assert isinstance(metrics, ShardMetrics)
        assert metrics.shard_id == ShardId.SHARD_1
        assert metrics.total_blocks == 1
        assert metrics.successful_blocks == 1
        assert metrics.validator_count == 1
        assert metrics.active_validators == 1

    def test_validator_heartbeat_updates(self, consensus):
        """Test that validator heartbeats are properly tracked."""
        consensus.add_validator("validator_1")
        validator = consensus.committee.validators[0]

        initial_heartbeat = validator.last_heartbeat
        time.sleep(0.01)

        validator.update_heartbeat()

        assert validator.last_heartbeat > initial_heartbeat
        assert validator.is_online() == True

    def test_consensus_with_inactive_validators(self, consensus):
        """Test consensus behavior with inactive validators."""
        consensus.add_validator("validator_1")
        consensus.add_validator("validator_2")

        # Make one validator inactive
        validator2 = consensus.committee.validators[1]
        validator2.is_active = False

        # Update metrics should reflect only active validators
        consensus.metrics.active_validators = len(
            consensus.committee.get_active_validators()
        )

        assert consensus.metrics.validator_count == 2
        assert consensus.metrics.active_validators == 1

    def test_proposer_rotation_with_validator_changes(self, consensus):
        """Test proposer rotation updates when validators change."""
        consensus.add_validator("validator_1")
        consensus.add_validator("validator_2")

        # Get first proposer
        proposer1 = consensus.get_proposer()
        assert proposer1 == "validator_1"

        # Remove a validator
        consensus.remove_validator("validator_1")

        # Next proposer should be from remaining validators
        proposer2 = consensus.get_proposer()
        assert proposer2 == "validator_2"

        # Add new validator
        consensus.add_validator("validator_3")

        # Next proposer should include new validator
        proposer3 = consensus.get_proposer()
        assert proposer3 == "validator_2"  # Still in rotation
        proposer4 = consensus.get_proposer()
        assert proposer4 == "validator_3"  # New validator in rotation
