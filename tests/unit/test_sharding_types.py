"""Tests for sharding types module."""

import logging

logger = logging.getLogger(__name__)
import pytest

from dubchain.sharding.shard_types import (
    CrossShardTransaction,
    ShardConfig,
    ShardId,
    ShardMetrics,
    ShardState,
    ShardStatus,
    ShardType,
)


class TestShardId:
    """Test ShardId functionality."""

    def test_shard_id_creation(self):
        """Test creating shard ID."""
        shard_id = ShardId.SHARD_1
        assert shard_id.value == 1
        assert str(shard_id) == "ShardId.SHARD_1"

    def test_shard_id_comparison(self):
        """Test shard ID comparison."""
        shard1 = ShardId.SHARD_1
        shard2 = ShardId.SHARD_2
        shard3 = ShardId.SHARD_1

        assert shard1 != shard2
        assert shard1 == shard3
        assert shard1 < shard2


class TestShardType:
    """Test ShardType enum."""

    def test_shard_type_values(self):
        """Test shard type values."""
        assert ShardType.BEACON.value == "beacon"
        assert ShardType.EXECUTION.value == "execution"
        assert ShardType.CONSENSUS.value == "consensus"
        assert ShardType.STORAGE.value == "storage"


class TestShardStatus:
    """Test ShardStatus enum."""

    def test_shard_status_values(self):
        """Test shard status values."""
        assert ShardStatus.ACTIVE.value == "active"
        assert ShardStatus.INACTIVE.value == "inactive"
        assert ShardStatus.SYNCING.value == "syncing"
        assert ShardStatus.ERROR.value == "error"


class TestShardState:
    """Test ShardState functionality."""

    def test_shard_state_creation(self):
        """Test creating shard state."""
        shard_id = ShardId.SHARD_1
        state = ShardState(
            shard_id=shard_id,
            shard_type=ShardType.EXECUTION,
            status=ShardStatus.ACTIVE,
            current_epoch=5,
            last_block_number=1000,
        )

        assert state.shard_id == shard_id
        assert state.shard_type == ShardType.EXECUTION
        assert state.status == ShardStatus.ACTIVE
        assert state.current_epoch == 5
        assert state.last_block_number == 1000

    def test_shard_state_defaults(self):
        """Test shard state defaults."""
        shard_id = ShardId.SHARD_1
        state = ShardState(
            shard_id=shard_id,
            status=ShardStatus.INACTIVE,
            shard_type=ShardType.EXECUTION,
        )

        assert state.shard_id == shard_id
        assert state.shard_type == ShardType.EXECUTION
        assert state.status == ShardStatus.INACTIVE
        assert state.current_epoch == 0
        assert state.last_block_number == 0


class TestShardConfig:
    """Test ShardConfig functionality."""

    def test_shard_config_creation(self):
        """Test creating shard config."""
        config = ShardConfig(
            max_shards=32,
            min_validators_per_shard=32,
            max_validators_per_shard=128,
            shard_epoch_length=32,
            cross_shard_delay=2,
        )

        assert config.max_shards == 32
        assert config.min_validators_per_shard == 32
        assert config.max_validators_per_shard == 128
        assert config.shard_epoch_length == 32
        assert config.cross_shard_delay == 2

    def test_shard_config_defaults(self):
        """Test shard config defaults."""
        config = ShardConfig()

        assert config.max_shards == 64
        assert config.min_validators_per_shard == 64
        assert config.max_validators_per_shard == 256
        assert config.shard_epoch_length == 64
        assert config.cross_shard_delay == 4


class TestCrossShardTransaction:
    """Test CrossShardTransaction functionality."""

    def test_cross_shard_transaction_creation(self):
        """Test creating cross shard transaction."""
        source_shard = ShardId.SHARD_1
        target_shard = ShardId.SHARD_2

        tx = CrossShardTransaction(
            transaction_id="tx123",
            source_shard=source_shard,
            target_shard=target_shard,
            sender="sender123",
            receiver="receiver456",
            amount=100,
            gas_limit=21000,
            gas_price=20,
            data=b"test_data",
        )

        assert tx.transaction_id == "tx123"
        assert tx.source_shard == source_shard
        assert tx.target_shard == target_shard
        assert tx.sender == "sender123"
        assert tx.receiver == "receiver456"
        assert tx.amount == 100
        assert tx.gas_limit == 21000
        assert tx.gas_price == 20
        assert tx.data == b"test_data"

    def test_cross_shard_transaction_validation(self):
        """Test cross shard transaction validation."""
        source_shard = ShardId.SHARD_1
        target_shard = ShardId.SHARD_2

        tx = CrossShardTransaction(
            transaction_id="tx123",
            source_shard=source_shard,
            target_shard=target_shard,
            sender="sender123",
            receiver="receiver456",
            amount=100,
            gas_limit=21000,
            gas_price=20,
            data=b"test_data",
        )

        # Test that source and target shards are different
        assert tx.source_shard != tx.target_shard


class TestShardMetrics:
    """Test ShardMetrics functionality."""

    def test_shard_metrics_creation(self):
        """Test creating shard metrics."""
        shard_id = ShardId.SHARD_1
        metrics = ShardMetrics(
            shard_id=shard_id,
            total_blocks=100,
            successful_blocks=95,
            failed_blocks=5,
            validator_count=200,
            average_block_time=12.5,
            average_gas_used=50000,
        )

        assert metrics.shard_id == shard_id
        assert metrics.total_blocks == 100
        assert metrics.successful_blocks == 95
        assert metrics.failed_blocks == 5
        assert metrics.validator_count == 200
        assert metrics.average_block_time == 12.5
        assert metrics.average_gas_used == 50000

    def test_shard_metrics_defaults(self):
        """Test shard metrics defaults."""
        shard_id = ShardId.SHARD_1
        metrics = ShardMetrics(shard_id=shard_id)

        assert metrics.shard_id == shard_id
        assert metrics.total_blocks == 0
        assert metrics.successful_blocks == 0
        assert metrics.failed_blocks == 0
        assert metrics.validator_count == 0
        assert metrics.average_block_time == 0.0
        assert metrics.average_gas_used == 0.0
