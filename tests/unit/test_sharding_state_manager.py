"""Tests for sharding state manager module."""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.sharding.shard_state_manager import (
    ShardStateManager,
    StateSnapshot,
    StateSync,
    StateValidator,
)
from dubchain.sharding.shard_types import ShardId, ShardState, ShardStatus, ShardType


class TestStateSnapshot:
    """Test StateSnapshot functionality."""

    def test_state_snapshot_creation(self):
        """Test creating state snapshot."""
        snapshot = StateSnapshot(
            shard_id=ShardId.SHARD_1, state_root="0x123abc", block_number=100
        )

        assert snapshot.shard_id == ShardId.SHARD_1
        assert snapshot.state_root == "0x123abc"
        assert snapshot.block_number == 100
        assert snapshot.timestamp > 0
        assert snapshot.validator_set == []
        assert snapshot.metadata == {}

    def test_state_snapshot_custom_values(self):
        """Test creating state snapshot with custom values."""
        validator_set = ["validator_1", "validator_2"]
        metadata = {"key": "value"}

        snapshot = StateSnapshot(
            shard_id=ShardId.SHARD_2,
            state_root="0x456def",
            block_number=200,
            validator_set=validator_set,
            metadata=metadata,
        )

        assert snapshot.shard_id == ShardId.SHARD_2
        assert snapshot.state_root == "0x456def"
        assert snapshot.block_number == 200
        assert snapshot.validator_set == validator_set
        assert snapshot.metadata == metadata

    def test_calculate_hash(self):
        """Test calculating snapshot hash."""
        snapshot = StateSnapshot(
            shard_id=ShardId.SHARD_1, state_root="0x123abc", block_number=100
        )

        hash1 = snapshot.calculate_hash()
        hash2 = snapshot.calculate_hash()

        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert isinstance(hash1, str)

    def test_calculate_hash_different_snapshots(self):
        """Test that different snapshots have different hashes."""
        snapshot1 = StateSnapshot(
            shard_id=ShardId.SHARD_1, state_root="0x123abc", block_number=100
        )

        snapshot2 = StateSnapshot(
            shard_id=ShardId.SHARD_2, state_root="0x123abc", block_number=100
        )

        hash1 = snapshot1.calculate_hash()
        hash2 = snapshot2.calculate_hash()

        assert hash1 != hash2

    def test_calculate_hash_includes_all_fields(self):
        """Test that hash calculation includes all relevant fields."""
        snapshot1 = StateSnapshot(
            shard_id=ShardId.SHARD_1, state_root="0x123abc", block_number=100
        )

        # Create identical snapshot
        snapshot2 = StateSnapshot(
            shard_id=ShardId.SHARD_1, state_root="0x123abc", block_number=100
        )

        # Set same timestamp
        snapshot2.timestamp = snapshot1.timestamp

        hash1 = snapshot1.calculate_hash()
        hash2 = snapshot2.calculate_hash()

        assert hash1 == hash2


class TestStateValidator:
    """Test StateValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Fixture for state validator."""
        return StateValidator()

    def test_state_validator_creation(self, validator):
        """Test creating state validator."""
        assert validator.validation_rules == {}

    def test_validate_state_valid(self, validator):
        """Test validating valid shard state."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1", "validator_2"],
        )

        result = validator.validate_state(shard_state)

        assert result == True

    def test_validate_state_no_shard_id(self, validator):
        """Test validating state with no shard ID."""
        shard_state = ShardState(
            shard_id=None,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        result = validator.validate_state(shard_state)

        assert result == False

    def test_validate_state_empty_validator_set(self, validator):
        """Test validating state with empty validator set."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=[],
        )

        result = validator.validate_state(shard_state)

        assert result == False

    def test_validate_state_none_validator_set(self, validator):
        """Test validating state with None validator set."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=None,
        )

        result = validator.validate_state(shard_state)

        assert result == False


class TestStateSync:
    """Test StateSync functionality."""

    @pytest.fixture
    def state_sync(self):
        """Fixture for state sync."""
        return StateSync()

    def test_state_sync_creation(self, state_sync):
        """Test creating state sync."""
        assert state_sync.sync_interval == 32
        assert state_sync.last_sync > 0
        assert state_sync.sync_metrics == {"syncs_performed": 0, "sync_failures": 0}

    def test_should_sync_immediately(self, state_sync):
        """Test that sync is needed immediately after creation."""
        # Set last_sync to past
        state_sync.last_sync = time.time() - 100

        result = state_sync.should_sync()

        assert result == True

    def test_should_sync_not_needed(self, state_sync):
        """Test that sync is not needed immediately after creation."""
        result = state_sync.should_sync()

        assert result == False

    def test_should_sync_custom_interval(self, state_sync):
        """Test should_sync with custom interval."""
        state_sync.sync_interval = 10  # 10 seconds

        # Set last_sync to 5 seconds ago
        state_sync.last_sync = time.time() - 5

        result = state_sync.should_sync()

        assert result == False

        # Set last_sync to 15 seconds ago
        state_sync.last_sync = time.time() - 15

        result = state_sync.should_sync()

        assert result == True

    def test_sync_states_success(self, state_sync):
        """Test successful state synchronization."""
        # Create mock shard states with cross-shard transactions
        shard_state1 = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        shard_state2 = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_2"],
        )

        # Add cross-shard transaction
        transaction = Mock()
        transaction.status = "pending"
        shard_state1.cross_shard_queues = {ShardId.SHARD_2: [transaction]}

        shard_states = {ShardId.SHARD_1: shard_state1, ShardId.SHARD_2: shard_state2}

        result = state_sync.sync_states(shard_states)

        assert result == True
        assert state_sync.sync_metrics["syncs_performed"] == 1
        assert state_sync.sync_metrics["sync_failures"] == 0
        assert transaction.status == "confirmed"
        assert transaction.confirmation_epoch == shard_state1.current_epoch

    def test_sync_states_no_cross_shard_transactions(self, state_sync):
        """Test state synchronization with no cross-shard transactions."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        shard_states = {ShardId.SHARD_1: shard_state}

        result = state_sync.sync_states(shard_states)

        assert result == True
        assert state_sync.sync_metrics["syncs_performed"] == 1
        assert state_sync.sync_metrics["sync_failures"] == 0

    def test_sync_states_failure(self, state_sync):
        """Test state synchronization failure."""
        # Create invalid shard states to cause exception
        shard_states = {ShardId.SHARD_1: None}

        result = state_sync.sync_states(shard_states)

        assert result == False
        assert state_sync.sync_metrics["syncs_performed"] == 0
        assert state_sync.sync_metrics["sync_failures"] == 1


class TestShardStateManager:
    """Test ShardStateManager functionality."""

    @pytest.fixture
    def manager(self):
        """Fixture for shard state manager."""
        return ShardStateManager()

    def test_shard_state_manager_creation(self, manager):
        """Test creating shard state manager."""
        assert manager.shard_states == {}
        assert isinstance(manager.state_validator, StateValidator)
        assert isinstance(manager.state_sync, StateSync)
        assert manager.snapshots == []

    def test_add_shard_state(self, manager):
        """Test adding shard state."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        manager.add_shard_state(shard_state)

        assert ShardId.SHARD_1 in manager.shard_states
        assert manager.shard_states[ShardId.SHARD_1] == shard_state

    def test_add_multiple_shard_states(self, manager):
        """Test adding multiple shard states."""
        shard_state1 = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        shard_state2 = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.CONSENSUS,
            validator_set=["validator_2"],
        )

        manager.add_shard_state(shard_state1)
        manager.add_shard_state(shard_state2)

        assert len(manager.shard_states) == 2
        assert ShardId.SHARD_1 in manager.shard_states
        assert ShardId.SHARD_2 in manager.shard_states

    def test_get_shard_state(self, manager):
        """Test getting shard state."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        manager.add_shard_state(shard_state)

        retrieved_state = manager.get_shard_state(ShardId.SHARD_1)

        assert retrieved_state == shard_state

    def test_get_shard_state_nonexistent(self, manager):
        """Test getting nonexistent shard state."""
        retrieved_state = manager.get_shard_state(ShardId.SHARD_1)

        assert retrieved_state is None

    def test_validate_all_states(self, manager):
        """Test validating all shard states."""
        # Add valid state
        valid_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        # Add invalid state (empty validator set)
        invalid_state = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=[],
        )

        manager.add_shard_state(valid_state)
        manager.add_shard_state(invalid_state)

        results = manager.validate_all_states()

        assert results[ShardId.SHARD_1] == True
        assert results[ShardId.SHARD_2] == False

    def test_validate_all_states_empty(self, manager):
        """Test validating all states when no states exist."""
        results = manager.validate_all_states()

        assert results == {}

    def test_sync_all_states_needed(self, manager):
        """Test synchronizing all states when sync is needed."""
        # Set sync to be needed
        manager.state_sync.last_sync = time.time() - 100

        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        manager.add_shard_state(shard_state)

        result = manager.sync_all_states()

        assert result == True
        assert manager.state_sync.sync_metrics["syncs_performed"] == 1

    def test_sync_all_states_not_needed(self, manager):
        """Test synchronizing all states when sync is not needed."""
        result = manager.sync_all_states()

        assert result == True
        assert manager.state_sync.sync_metrics["syncs_performed"] == 0

    def test_create_snapshot(self, manager):
        """Test creating state snapshot."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1", "validator_2"],
            state_root="0x123abc",
            last_block_number=100,
        )

        manager.add_shard_state(shard_state)

        snapshot = manager.create_snapshot(ShardId.SHARD_1)

        assert snapshot is not None
        assert snapshot.shard_id == ShardId.SHARD_1
        assert snapshot.state_root == "0x123abc"
        assert snapshot.block_number == 100
        assert snapshot.validator_set == ["validator_1", "validator_2"]
        assert len(manager.snapshots) == 1
        assert manager.snapshots[0] == snapshot

    def test_create_snapshot_nonexistent_shard(self, manager):
        """Test creating snapshot for nonexistent shard."""
        snapshot = manager.create_snapshot(ShardId.SHARD_1)

        assert snapshot is None
        assert len(manager.snapshots) == 0

    def test_create_multiple_snapshots(self, manager):
        """Test creating multiple snapshots."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
            state_root="0x123abc",
            last_block_number=100,
        )

        manager.add_shard_state(shard_state)

        snapshot1 = manager.create_snapshot(ShardId.SHARD_1)
        snapshot2 = manager.create_snapshot(ShardId.SHARD_1)

        assert len(manager.snapshots) == 2
        assert snapshot1 != snapshot2
        assert snapshot1.shard_id == snapshot2.shard_id

    def test_get_metrics(self, manager):
        """Test getting state management metrics."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        manager.add_shard_state(shard_state)
        manager.create_snapshot(ShardId.SHARD_1)

        metrics = manager.get_metrics()

        assert metrics["total_shards"] == 1
        assert metrics["total_snapshots"] == 1
        assert "sync_metrics" in metrics
        assert "last_sync" in metrics
        assert metrics["sync_metrics"]["syncs_performed"] == 0
        assert metrics["sync_metrics"]["sync_failures"] == 0

    def test_get_metrics_empty(self, manager):
        """Test getting metrics when no states exist."""
        metrics = manager.get_metrics()

        assert metrics["total_shards"] == 0
        assert metrics["total_snapshots"] == 0
        assert "sync_metrics" in metrics
        assert "last_sync" in metrics

    def test_snapshot_hash_calculation(self, manager):
        """Test that snapshots have proper hash calculation."""
        shard_state = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
            state_root="0x123abc",
            last_block_number=100,
        )

        manager.add_shard_state(shard_state)

        snapshot = manager.create_snapshot(ShardId.SHARD_1)

        hash_value = snapshot.calculate_hash()

        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 hex length
        assert isinstance(hash_value, str)

    def test_state_sync_with_cross_shard_transactions(self, manager):
        """Test state sync with cross-shard transactions."""
        # Create shard states with cross-shard transactions
        shard_state1 = ShardState(
            shard_id=ShardId.SHARD_1,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_1"],
        )

        shard_state2 = ShardState(
            shard_id=ShardId.SHARD_2,
            status=ShardStatus.ACTIVE,
            shard_type=ShardType.EXECUTION,
            validator_set=["validator_2"],
        )

        # Add cross-shard transaction
        transaction = Mock()
        transaction.status = "pending"
        shard_state1.cross_shard_queues = {ShardId.SHARD_2: [transaction]}

        manager.add_shard_state(shard_state1)
        manager.add_shard_state(shard_state2)

        # Force sync
        manager.state_sync.last_sync = time.time() - 100

        result = manager.sync_all_states()

        assert result == True
        assert transaction.status == "confirmed"
        assert transaction.confirmation_epoch == shard_state1.current_epoch
