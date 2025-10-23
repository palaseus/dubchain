"""Tests for sharding manager module."""

import logging

logger = logging.getLogger(__name__)
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.sharding.shard_manager import ShardManager
from dubchain.sharding.shard_types import (
    ShardConfig,
    ShardId,
    ShardMetrics,
    ShardState,
    ShardStatus,
    ShardType,
)


class TestShardManager:
    """Test ShardManager functionality."""

    @pytest.fixture
    def shard_config(self):
        """Fixture for shard configuration."""
        return ShardConfig(
            max_shards=32, min_validators_per_shard=32, max_validators_per_shard=128
        )

    @pytest.fixture
    def shard_manager(self, shard_config):
        """Fixture for shard manager."""
        return ShardManager(shard_config)

    def test_shard_manager_creation(self, shard_config):
        """Test creating shard manager."""
        manager = ShardManager(shard_config)

        assert manager.config == shard_config
        assert isinstance(manager._shards, dict)
        assert isinstance(manager._validators, dict)
        assert manager._running is False

    def test_create_shard(self, shard_manager):
        """Test creating shard."""
        shard = shard_manager.create_shard(
            shard_type=ShardType.EXECUTION, validators=["validator1", "validator2"]
        )

        assert shard is not None
        assert shard.shard_type == ShardType.EXECUTION
        assert len(shard.validator_set) == 2

    def test_get_shard(self, shard_manager):
        """Test getting shard."""
        created_shard = shard_manager.create_shard(shard_type=ShardType.EXECUTION)

        retrieved_shard = shard_manager.get_shard(created_shard.shard_id)

        assert created_shard == retrieved_shard

    def test_get_nonexistent_shard(self, shard_manager):
        """Test getting nonexistent shard."""
        result = shard_manager.get_shard(ShardId.SHARD_9)
        assert result is None

    def test_list_shards(self, shard_manager):
        """Test listing shards."""
        shard1 = shard_manager.create_shard(ShardType.EXECUTION)
        shard2 = shard_manager.create_shard(ShardType.CONSENSUS)

        shards = shard_manager.list_shards()

        assert len(shards) == 2
        assert shard1.shard_id in [s.shard_id for s in shards]
        assert shard2.shard_id in [s.shard_id for s in shards]

    def test_add_validator(self, shard_manager):
        """Test adding validator to shard."""
        shard = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard.shard_id

        shard_manager.add_validator(shard_id, "validator1")

        shard = shard_manager.get_shard(shard_id)
        assert "validator1" in shard.validator_set

    def test_remove_validator(self, shard_manager):
        """Test removing validator from shard."""
        shard = shard_manager.create_shard(
            ShardType.EXECUTION, validators=["validator1", "validator2"]
        )
        shard_id = shard.shard_id

        shard_manager.remove_validator(shard_id, "validator1")

        shard = shard_manager.get_shard(shard_id)
        assert "validator1" not in shard.validator_set
        assert "validator2" in shard.validator_set

    def test_update_shard_status(self, shard_manager):
        """Test updating shard status."""
        shard = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard.shard_id

        shard_manager.update_shard_status(shard_id, ShardStatus.ACTIVE)

        shard = shard_manager.get_shard(shard_id)
        assert shard.status == ShardStatus.ACTIVE

    def test_get_shard_metrics(self, shard_manager):
        """Test getting shard metrics."""
        shard = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard.shard_id

        metrics = shard_manager.get_shard_metrics(shard_id)

        assert metrics is not None
        assert metrics.shard_id == shard_id

    def test_get_active_shards(self, shard_manager):
        """Test getting active shards."""
        shard1 = shard_manager.create_shard(ShardType.EXECUTION)
        shard2 = shard_manager.create_shard(ShardType.CONSENSUS)

        shard_manager.update_shard_status(shard1.shard_id, ShardStatus.ACTIVE)
        shard_manager.update_shard_status(shard2.shard_id, ShardStatus.INACTIVE)

        active_shards = shard_manager.get_active_shards()

        assert len(active_shards) == 1
        assert active_shards[0].shard_id == shard1.shard_id

    def test_validate_shard(self, shard_manager):
        """Test shard validation."""
        shard = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard.shard_id

        # Should not raise exception for valid shard
        shard_manager.validate_shard(shard_id)

    def test_rebalance_shards(self, shard_manager):
        """Test shard rebalancing."""
        # Create multiple shards with different validator counts
        shard1 = shard_manager.create_shard(
            ShardType.EXECUTION, validators=["v1", "v2", "v3", "v4", "v5"]
        )
        shard2 = shard_manager.create_shard(ShardType.EXECUTION, validators=["v6"])

        # Rebalancing should distribute validators more evenly
        shard_manager.rebalance_shards()

        # Check that rebalancing was attempted
        shard1_retrieved = shard_manager.get_shard(shard1.shard_id)
        shard2_retrieved = shard_manager.get_shard(shard2.shard_id)

        assert shard1_retrieved is not None
        assert shard2_retrieved is not None

    def test_start_stop_manager(self, shard_manager):
        """Test starting and stopping shard manager."""
        assert shard_manager._running is False

        shard_manager.start()
        assert shard_manager._running is True

        shard_manager.stop()
        assert shard_manager._running is False
