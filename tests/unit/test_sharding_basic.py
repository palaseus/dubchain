"""Basic tests for sharding module."""

import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.sharding.shard_manager import ShardAllocator, ShardBalancer
from src.dubchain.sharding.shard_types import ShardId, ShardConfig, ShardStatus, ShardType
from src.dubchain.consensus.validator import ValidatorInfo


class TestShardAllocator:
    """Test ShardAllocator functionality."""

    def test_init(self):
        """Test ShardAllocator initialization."""
        allocator = ShardAllocator()
        assert allocator.allocation_strategy == "random"
        assert allocator.rebalance_threshold == 0.1
        assert allocator.last_rebalance > 0

    def test_allocate_validators_random(self):
        """Test random allocation of validators."""
        allocator = ShardAllocator(allocation_strategy="random")
        
        # Create mock validators
        from unittest.mock import Mock
        from src.dubchain.crypto.signatures import PublicKey
        
        validators = [
            ValidatorInfo(validator_id="val1", public_key=Mock(spec=PublicKey), total_stake=1000),
            ValidatorInfo(validator_id="val2", public_key=Mock(spec=PublicKey), total_stake=2000),
            ValidatorInfo(validator_id="val3", public_key=Mock(spec=PublicKey), total_stake=1500),
        ]
        
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: x  # Don't actually shuffle
            allocation = allocator.allocate_validators(validators, 2)
        
        assert len(allocation) == 2
        assert ShardId(1) in allocation
        assert ShardId(2) in allocation
        
        # Check that all validators are allocated
        all_allocated = []
        for shard_validators in allocation.values():
            all_allocated.extend(shard_validators)
        
        assert len(all_allocated) == 3

    def test_allocate_validators_round_robin(self):
        """Test round-robin allocation of validators."""
        allocator = ShardAllocator(allocation_strategy="round_robin")
        
        # Create mock validators
        from unittest.mock import Mock
        from src.dubchain.crypto.signatures import PublicKey
        
        validators = [
            ValidatorInfo(validator_id="val1", public_key=Mock(spec=PublicKey), total_stake=1000),
            ValidatorInfo(validator_id="val2", public_key=Mock(spec=PublicKey), total_stake=2000),
            ValidatorInfo(validator_id="val3", public_key=Mock(spec=PublicKey), total_stake=1500),
        ]
        
        allocation = allocator.allocate_validators(validators, 2)
        
        assert len(allocation) == 2
        assert ShardId(1) in allocation
        assert ShardId(2) in allocation
        
        # Check round-robin distribution
        assert allocation[ShardId(1)] == ["val1", "val3"]
        assert allocation[ShardId(2)] == ["val2"]

    def test_allocate_validators_empty_list(self):
        """Test allocation with empty validator list."""
        allocator = ShardAllocator()
        
        allocation = allocator.allocate_validators([], 2)
        
        assert len(allocation) == 2
        assert allocation[ShardId(1)] == []
        assert allocation[ShardId(2)] == []


class TestShardBalancer:
    """Test ShardBalancer functionality."""

    def test_init(self):
        """Test ShardBalancer initialization."""
        balancer = ShardBalancer()
        assert balancer.balance_threshold == 0.1
        assert balancer.rebalance_interval == 3600.0
        assert balancer.last_rebalance > 0

    def test_should_rebalance(self):
        """Test checking if rebalancing should occur."""
        balancer = ShardBalancer()
        
        # Set last rebalance to old time to allow rebalancing
        import time
        balancer.last_rebalance = time.time() - 4000  # 4000 seconds ago
        
        # Create mock shard states
        from src.dubchain.sharding.shard_types import ShardState, ShardStatus
        shard_states = {
            ShardId(1): ShardState(
                shard_id=ShardId(1),
                status=ShardStatus.ACTIVE,
                shard_type=ShardType.EXECUTION,
                validator_set=["val1", "val2"]
            ),
            ShardId(2): ShardState(
                shard_id=ShardId(2),
                status=ShardStatus.ACTIVE,
                shard_type=ShardType.EXECUTION,
                validator_set=["val3", "val4", "val5"]
            ),
        }
        
        # Should rebalance due to imbalance
        assert balancer.should_rebalance(shard_states) is True


    def test_should_rebalance_no_states(self):
        """Test checking if rebalancing should occur with no states."""
        balancer = ShardBalancer()
        
        # Should not rebalance with empty states
        assert balancer.should_rebalance({}) is False
        
        # Set last rebalance to old time
        balancer.last_rebalance = time.time() - 400  # 400 seconds ago
        balancer.rebalance_interval = 300  # 300 second interval
        assert balancer.should_rebalance({}) is False  # Still no states


class TestShardTypes:
    """Test shard type definitions."""

    def test_shard_id(self):
        """Test ShardId functionality."""
        shard_id = ShardId(1)
        assert shard_id.value == 1
        
        shard_id2 = ShardId(2)
        assert shard_id != shard_id2
        assert shard_id < shard_id2

    def test_shard_config(self):
        """Test ShardConfig functionality."""
        config = ShardConfig()
        
        assert config.max_shards == 64
        assert config.min_validators_per_shard == 64
        assert config.max_validators_per_shard == 256
        assert config.shard_epoch_length == 64
        assert config.cross_shard_delay == 4
        assert config.state_sync_interval == 32
        assert config.rebalance_threshold == 0.1
        assert config.enable_dynamic_sharding is True

    def test_shard_status(self):
        """Test ShardStatus enum."""
        assert ShardStatus.INACTIVE.value == "inactive"
        assert ShardStatus.ACTIVE.value == "active"
        assert ShardStatus.SYNCING.value == "syncing"
        assert ShardStatus.ERROR.value == "error"

    def test_shard_type(self):
        """Test ShardType enum."""
        assert ShardType.EXECUTION.value == "execution"
        assert ShardType.STORAGE.value == "storage"
        assert ShardType.BEACON.value == "beacon"
        assert ShardType.CONSENSUS.value == "consensus"
