"""
Unit tests for shard manager functionality.
"""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.consensus.validator import ValidatorInfo
from dubchain.sharding.shard_manager import ShardAllocator, ShardBalancer, ShardManager
from dubchain.sharding.shard_types import (
    ShardConfig,
    ShardId,
    ShardMetrics,
    ShardState,
    ShardStatus,
    ShardType,
)


class TestShardConfig:
    """Test the ShardConfig class."""

    def test_shard_config_creation(self):
        """Test creating a shard configuration."""
        config = ShardConfig(
            max_shards=32,
            min_validators_per_shard=32,
            max_validators_per_shard=128,
            shard_epoch_length=32,
            cross_shard_delay=2,
            state_sync_interval=16,
            rebalance_threshold=0.05,
            enable_dynamic_sharding=False,
            shard_consensus_type="pbft",
        )

        assert config.max_shards == 32
        assert config.min_validators_per_shard == 32
        assert config.max_validators_per_shard == 128
        assert config.shard_epoch_length == 32
        assert config.cross_shard_delay == 2
        assert config.state_sync_interval == 16
        assert config.rebalance_threshold == 0.05
        assert config.enable_dynamic_sharding is False
        assert config.shard_consensus_type == "pbft"

    def test_shard_config_serialization(self):
        """Test shard configuration serialization."""
        config = ShardConfig(
            max_shards=32,
            min_validators_per_shard=32,
            max_validators_per_shard=128,
            shard_epoch_length=32,
            cross_shard_delay=2,
            state_sync_interval=16,
            rebalance_threshold=0.05,
            enable_dynamic_sharding=False,
            shard_consensus_type="pbft",
        )

        # Test to_dict
        data = config.to_dict()
        assert isinstance(data, dict)
        assert data["max_shards"] == 32
        assert data["min_validators_per_shard"] == 32
        assert data["shard_consensus_type"] == "pbft"

        # Test from_dict
        deserialized = ShardConfig.from_dict(data)
        assert deserialized.max_shards == config.max_shards
        assert deserialized.min_validators_per_shard == config.min_validators_per_shard
        assert deserialized.shard_consensus_type == config.shard_consensus_type


class TestShardMetrics:
    """Test the ShardMetrics class."""

    def test_shard_metrics_creation(self):
        """Test creating shard metrics."""
        metrics = ShardMetrics(
            shard_id=ShardId.SHARD_1,
            total_blocks=1000,
            successful_blocks=950,
            failed_blocks=50,
            average_block_time=2.5,
            average_gas_used=1000000,
            validator_count=64,
            active_validators=60,
        )

        assert metrics.shard_id == ShardId.SHARD_1
        assert metrics.total_blocks == 1000
        assert metrics.successful_blocks == 950
        assert metrics.failed_blocks == 50
        assert metrics.average_block_time == 2.5
        assert metrics.average_gas_used == 1000000
        assert metrics.validator_count == 64
        assert metrics.active_validators == 60

    def test_shard_metrics_calculations(self):
        """Test shard metrics calculations."""
        metrics = ShardMetrics(
            shard_id=ShardId.SHARD_1,
            total_blocks=1000,
            successful_blocks=950,
            failed_blocks=50,
            average_block_time=2.5,
            average_gas_used=1000000,
            validator_count=64,
            active_validators=60,
        )

        # Test success rate calculation
        success_rate = metrics.get_success_rate()
        assert success_rate == 0.95  # 950/1000

        # Test validator utilization
        utilization = metrics.get_validator_utilization()
        assert utilization == 0.9375  # 60/64


class TestShardAllocator:
    """Test the ShardAllocator class."""

    def test_shard_allocator_creation(self):
        """Test creating a shard allocator."""
        allocator = ShardAllocator(
            allocation_strategy="random", rebalance_threshold=0.1
        )

        assert allocator.allocation_strategy == "random"
        assert allocator.rebalance_threshold == 0.1
        assert allocator.last_rebalance > 0

    def test_random_allocation(self):
        """Test random validator allocation."""
        allocator = ShardAllocator(allocation_strategy="random")

        # Create test validators
        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators = [
            ValidatorInfo(
                validator_id="validator_1", public_key=mock_public_key, total_stake=1000
            ),
            ValidatorInfo(
                validator_id="validator_2", public_key=mock_public_key, total_stake=2000
            ),
            ValidatorInfo(
                validator_id="validator_3", public_key=mock_public_key, total_stake=1500
            ),
            ValidatorInfo(
                validator_id="validator_4", public_key=mock_public_key, total_stake=3000
            ),
        ]

        allocation = allocator.allocate_validators(validators, 2)

        assert len(allocation) == 2
        assert ShardId.SHARD_1 in allocation
        assert ShardId.SHARD_2 in allocation

        # Check that all validators are allocated
        all_allocated = []
        for shard_validators in allocation.values():
            all_allocated.extend(shard_validators)

        assert len(all_allocated) == 4
        assert "validator_1" in all_allocated
        assert "validator_2" in all_allocated
        assert "validator_3" in all_allocated
        assert "validator_4" in all_allocated

    def test_round_robin_allocation(self):
        """Test round-robin validator allocation."""
        allocator = ShardAllocator(allocation_strategy="round_robin")

        # Create test validators
        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators = [
            ValidatorInfo(
                validator_id="validator_1", public_key=mock_public_key, total_stake=1000
            ),
            ValidatorInfo(
                validator_id="validator_2", public_key=mock_public_key, total_stake=2000
            ),
            ValidatorInfo(
                validator_id="validator_3", public_key=mock_public_key, total_stake=1500
            ),
            ValidatorInfo(
                validator_id="validator_4", public_key=mock_public_key, total_stake=3000
            ),
        ]

        allocation = allocator.allocate_validators(validators, 2)

        assert len(allocation) == 2
        assert ShardId.SHARD_1 in allocation
        assert ShardId.SHARD_2 in allocation

        # In round-robin, validator_1 and validator_3 should be in SHARD_1
        # validator_2 and validator_4 should be in SHARD_2
        assert "validator_1" in allocation[ShardId.SHARD_1]
        assert "validator_3" in allocation[ShardId.SHARD_1]
        assert "validator_2" in allocation[ShardId.SHARD_2]
        assert "validator_4" in allocation[ShardId.SHARD_2]

    def test_weighted_allocation(self):
        """Test weighted validator allocation."""
        allocator = ShardAllocator(allocation_strategy="weighted")

        # Create test validators with different stakes
        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators = [
            ValidatorInfo(
                validator_id="validator_1", public_key=mock_public_key, total_stake=1000
            ),
            ValidatorInfo(
                validator_id="validator_2", public_key=mock_public_key, total_stake=2000
            ),
            ValidatorInfo(
                validator_id="validator_3", public_key=mock_public_key, total_stake=1500
            ),
            ValidatorInfo(
                validator_id="validator_4", public_key=mock_public_key, total_stake=3000
            ),
        ]

        allocation = allocator.allocate_validators(validators, 2)

        assert len(allocation) == 2
        assert ShardId.SHARD_1 in allocation
        assert ShardId.SHARD_2 in allocation

        # Check that all validators are allocated
        all_allocated = []
        for shard_validators in allocation.values():
            all_allocated.extend(shard_validators)

        assert len(all_allocated) == 4
        assert "validator_1" in all_allocated
        assert "validator_2" in all_allocated
        assert "validator_3" in all_allocated
        assert "validator_4" in all_allocated


class TestShardBalancer:
    """Test the ShardBalancer class."""

    def test_shard_balancer_creation(self):
        """Test creating a shard balancer."""
        balancer = ShardBalancer(balance_threshold=0.1, rebalance_interval=3600.0)

        assert balancer.balance_threshold == 0.1
        assert balancer.rebalance_interval == 3600.0
        assert balancer.last_rebalance > 0

    def test_should_rebalance(self):
        """Test rebalancing decision logic."""
        balancer = ShardBalancer(balance_threshold=0.1)
        # Set last_rebalance to be old enough to allow rebalancing
        balancer.last_rebalance = time.time() - 7200  # 2 hours ago

        # Create balanced shard states
        balanced_states = {
            ShardId.SHARD_1: ShardState(
                shard_id=ShardId.SHARD_1,
                status=ShardStatus.ACTIVE,
                shard_type=ShardType.EXECUTION,
            ),
            ShardId.SHARD_2: ShardState(
                shard_id=ShardId.SHARD_2,
                status=ShardStatus.ACTIVE,
                shard_type=ShardType.EXECUTION,
            ),
        }
        # Set validator sets to simulate balanced state
        balanced_states[ShardId.SHARD_1].validator_set = [
            f"validator_{i}" for i in range(50)
        ]
        balanced_states[ShardId.SHARD_2].validator_set = [
            f"validator_{i+50}" for i in range(50)
        ]

        # Should not rebalance when balanced
        should_rebalance = balancer.should_rebalance(balanced_states)
        assert should_rebalance is False

        # Create imbalanced shard states
        imbalanced_states = {
            ShardId.SHARD_1: ShardState(
                shard_id=ShardId.SHARD_1,
                status=ShardStatus.ACTIVE,
                shard_type=ShardType.EXECUTION,
            ),
            ShardId.SHARD_2: ShardState(
                shard_id=ShardId.SHARD_2,
                status=ShardStatus.ACTIVE,
                shard_type=ShardType.EXECUTION,
            ),
        }
        # Set validator sets to simulate imbalanced state
        imbalanced_states[ShardId.SHARD_1].validator_set = [
            f"validator_{i}" for i in range(30)
        ]
        imbalanced_states[ShardId.SHARD_2].validator_set = [
            f"validator_{i+30}" for i in range(70)
        ]

        # Should rebalance when imbalanced
        # The imbalance is (70-30)/30 = 1.33, which is > 0.1 threshold
        should_rebalance = balancer.should_rebalance(imbalanced_states)
        assert should_rebalance is True


class TestShardManager:
    """Test the ShardManager class."""

    @pytest.fixture
    def shard_config(self):
        """Create a shard configuration."""
        return ShardConfig(
            max_shards=4,
            min_validators_per_shard=16,
            max_validators_per_shard=64,
            shard_epoch_length=32,
            cross_shard_delay=2,
            state_sync_interval=16,
            rebalance_threshold=0.1,
            enable_dynamic_sharding=True,
            shard_consensus_type="proof_of_stake",
        )

    @pytest.fixture
    def shard_manager(self, shard_config):
        """Create a shard manager instance."""
        return ShardManager(shard_config)

    def test_shard_manager_creation(self, shard_manager, shard_config):
        """Test creating a shard manager instance."""
        assert shard_manager is not None
        assert shard_manager.config == shard_config
        assert shard_manager.allocator is not None
        assert shard_manager.balancer is not None
        assert len(shard_manager.shards) == 0
        assert len(shard_manager.shard_states) == 0

    def test_create_shard(self, shard_manager):
        """Test creating a new shard."""
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)

        assert shard_state is not None
        assert shard_state.shard_id in shard_manager.shards
        assert (
            shard_manager.shards[shard_state.shard_id].shard_type == ShardType.EXECUTION
        )
        assert shard_manager.shards[shard_state.shard_id].status == ShardStatus.ACTIVE

    def test_get_shard_info(self, shard_manager):
        """Test getting shard information."""
        # Create a shard first
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id

        shard_info = shard_manager.get_shard_info(shard_id)

        assert shard_info is not None
        assert shard_info.shard_id == shard_id
        assert shard_info.shard_type == ShardType.EXECUTION
        assert shard_info.status == ShardStatus.ACTIVE

    def test_get_shard_info_not_found(self, shard_manager):
        """Test getting information for non-existent shard."""
        shard_info = shard_manager.get_shard_info(ShardId.SHARD_10)
        assert shard_info is None

    def test_assign_validators_to_shard(self, shard_manager):
        """Test assigning validators to a shard."""
        # Create a shard first
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id

        # Create test validators
        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators = [
            ValidatorInfo(
                validator_id="validator_1", public_key=mock_public_key, total_stake=1000
            ),
            ValidatorInfo(
                validator_id="validator_2", public_key=mock_public_key, total_stake=2000
            ),
            ValidatorInfo(
                validator_id="validator_3", public_key=mock_public_key, total_stake=1500
            ),
        ]

        # Assign validators to shard
        result = shard_manager.assign_validators_to_shard(shard_id, validators)

        assert result is True
        assert shard_id in shard_manager.shard_states
        assert len(shard_manager.shard_states[shard_id].validator_set) == 3

    def test_get_shard_validators(self, shard_manager):
        """Test getting validators for a shard."""
        # Create a shard and assign validators
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id

        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators = [
            ValidatorInfo(
                validator_id="validator_1", public_key=mock_public_key, total_stake=1000
            ),
            ValidatorInfo(
                validator_id="validator_2", public_key=mock_public_key, total_stake=2000
            ),
        ]

        shard_manager.assign_validators_to_shard(shard_id, validators)

        # Get shard validators
        shard_validators = shard_manager.get_shard_validators(shard_id)

        assert shard_validators is not None
        assert len(shard_validators) == 2
        assert "validator_1" in shard_validators
        assert "validator_2" in shard_validators

    def test_get_shard_validators_not_found(self, shard_manager):
        """Test getting validators for non-existent shard."""
        shard_validators = shard_manager.get_shard_validators(ShardId.SHARD_10)
        assert shard_validators == []

    def test_rebalance_shards(self, shard_manager):
        """Test rebalancing shards."""
        # Create multiple shards with different validator counts
        shard_state1 = shard_manager.create_shard(ShardType.EXECUTION)
        shard_state2 = shard_manager.create_shard(ShardType.CONSENSUS)
        shard1 = shard_state1.shard_id
        shard2 = shard_state2.shard_id

        # Assign different numbers of validators to create imbalance
        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators1 = [
            ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                total_stake=1000,
            )
            for i in range(10)
        ]
        validators2 = [
            ValidatorInfo(
                validator_id=f"validator_{i+10}",
                public_key=mock_public_key,
                total_stake=1000,
            )
            for i in range(30)
        ]

        shard_manager.assign_validators_to_shard(shard1, validators1)
        shard_manager.assign_validators_to_shard(shard2, validators2)

        # Rebalance shards
        rebalance_result = shard_manager.rebalance_shards()

        assert rebalance_result is not None
        assert "rebalanced_shards" in rebalance_result
        assert "moved_validators" in rebalance_result

    def test_get_shard_metrics(self, shard_manager):
        """Test getting shard metrics."""
        # Create a shard first
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id

        # Get shard metrics
        metrics = shard_manager.get_shard_metrics(shard_id)

        assert metrics is not None
        assert metrics.shard_id == shard_id
        assert metrics.total_blocks >= 0
        assert metrics.validator_count >= 0

    def test_get_shard_metrics_not_found(self, shard_manager):
        """Test getting metrics for non-existent shard."""
        metrics = shard_manager.get_shard_metrics(ShardId.SHARD_10)
        assert metrics is None

    def test_update_shard_status(self, shard_manager):
        """Test updating shard status."""
        # Create a shard first
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id

        # Update status
        result = shard_manager.update_shard_status(shard_id, ShardStatus.MAINTENANCE)

        assert result is True
        assert shard_manager.shards[shard_id].status == ShardStatus.MAINTENANCE

    def test_update_shard_status_not_found(self, shard_manager):
        """Test updating status for non-existent shard."""
        result = shard_manager.update_shard_status(
            ShardId.SHARD_10, ShardStatus.MAINTENANCE
        )
        assert result is False

    def test_get_all_shards(self, shard_manager):
        """Test getting all shards."""
        # Create some shards
        shard_state1 = shard_manager.create_shard(ShardType.EXECUTION)
        shard_state2 = shard_manager.create_shard(ShardType.CONSENSUS)

        # Get all shards
        all_shards = shard_manager.get_all_shards()

        assert all_shards is not None
        assert len(all_shards) == 2
        assert shard_state1 in all_shards
        assert shard_state2 in all_shards

    def test_get_shard_statistics(self, shard_manager):
        """Test getting shard statistics."""
        # Create some shards
        shard_state1 = shard_manager.create_shard(ShardType.EXECUTION)
        shard_state2 = shard_manager.create_shard(ShardType.CONSENSUS)
        shard1 = shard_state1.shard_id
        shard2 = shard_state2.shard_id

        # Assign validators to shards
        from dubchain.crypto.signatures import PublicKey

        mock_public_key = Mock(spec=PublicKey)
        validators1 = [
            ValidatorInfo(
                validator_id=f"validator_{i}",
                public_key=mock_public_key,
                total_stake=1000,
            )
            for i in range(5)
        ]
        validators2 = [
            ValidatorInfo(
                validator_id=f"validator_{i+5}",
                public_key=mock_public_key,
                total_stake=1000,
            )
            for i in range(10)
        ]

        shard_manager.assign_validators_to_shard(shard1, validators1)
        shard_manager.assign_validators_to_shard(shard2, validators2)

        # Get statistics
        stats = shard_manager.get_shard_statistics()

        assert stats is not None
        assert "total_shards" in stats
        assert "active_shards" in stats
        assert "total_validators" in stats
        assert "average_validators_per_shard" in stats
        assert stats["total_shards"] == 2
        assert stats["total_validators"] == 15

    def test_remove_shard(self, shard_manager):
        """Test removing a shard."""
        # Create a shard first
        shard_state = shard_manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id

        # Remove shard
        result = shard_manager.remove_shard(shard_id)

        assert result is True
        assert shard_id not in shard_manager.shards
        assert shard_id not in shard_manager.shard_states

    def test_remove_shard_not_found(self, shard_manager):
        """Test removing non-existent shard."""
        result = shard_manager.remove_shard(ShardId.SHARD_10)
        assert result is False
