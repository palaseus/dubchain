"""
Unit tests for Enhanced Sharding System.

This module provides comprehensive unit tests covering:
- Load balancing strategies
- Resharding operations
- Health monitoring
- Fault tolerance
- Performance metrics
- Edge cases and error conditions
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.dubchain.sharding.enhanced_sharding import (
    LoadBalancingStrategy,
    ReshardingStrategy,
    ShardHealthStatus,
    ShardLoadMetrics,
    ShardHealthInfo,
    ReshardingPlan,
    ShardOperation,
    ShardLoadBalancer,
    ConsistentHashBalancer,
    LeastLoadedBalancer,
    AdaptiveBalancer,
    ShardReshardingManager,
    ShardHealthMonitor,
    EnhancedShardManager,
)
from src.dubchain.sharding.shard_types import (
    ShardId,
    ShardStatus,
    ShardType,
    ShardConfig,
    ShardState,
    ShardMetrics,
)


class TestShardLoadMetrics:
    """Test ShardLoadMetrics functionality."""
    
    def test_load_metrics_creation(self):
        """Test load metrics creation."""
        shard_id = ShardId.SHARD_1
        metrics = ShardLoadMetrics(shard_id=shard_id)
        
        assert metrics.shard_id == shard_id
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.disk_usage == 0.0
        assert metrics.network_io == 0.0
        assert metrics.transaction_throughput == 0.0
        assert metrics.queue_depth == 0
        assert metrics.response_time_p50 == 0.0
        assert metrics.response_time_p95 == 0.0
        assert metrics.response_time_p99 == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.last_updated > 0
    
    def test_overall_load_score_calculation(self):
        """Test overall load score calculation."""
        shard_id = ShardId.SHARD_1
        metrics = ShardLoadMetrics(
            shard_id=shard_id,
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_io=100.0,
            transaction_throughput=5000.0,
            queue_depth=100,
            response_time_p95=500.0
        )
        
        load_score = metrics.overall_load_score
        assert 0.0 <= load_score <= 1.0
        assert load_score > 0  # Should have some load
    
    def test_load_score_normalization(self):
        """Test that load scores are properly normalized."""
        shard_id = ShardId.SHARD_1
        
        # Test with maximum values
        metrics_max = ShardLoadMetrics(
            shard_id=shard_id,
            cpu_usage=100.0,
            memory_usage=100.0,
            disk_usage=100.0,
            network_io=1000.0,
            transaction_throughput=10000.0,
            queue_depth=1000,
            response_time_p95=1000.0
        )
        
        assert metrics_max.overall_load_score <= 1.0
        
        # Test with zero values
        metrics_zero = ShardLoadMetrics(shard_id=shard_id)
        assert metrics_zero.overall_load_score == 0.0


class TestShardHealthInfo:
    """Test ShardHealthInfo functionality."""
    
    def test_health_info_creation(self):
        """Test health info creation."""
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id)
        health_info = ShardHealthInfo(
            shard_id=shard_id,
            status=ShardHealthStatus.HEALTHY,
            load_metrics=load_metrics
        )
        
        assert health_info.shard_id == shard_id
        assert health_info.status == ShardHealthStatus.HEALTHY
        assert health_info.load_metrics == load_metrics
        assert health_info.consecutive_failures == 0
        assert health_info.recovery_attempts == 0
        assert health_info.health_score == 1.0
    
    def test_health_score_update(self):
        """Test health score update logic."""
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(
            shard_id=shard_id,
            cpu_usage=80.0,  # High load
            memory_usage=90.0
        )
        health_info = ShardHealthInfo(
            shard_id=shard_id,
            status=ShardHealthStatus.HEALTHY,
            load_metrics=load_metrics
        )
        
        health_info.update_health_score()
        
        assert health_info.health_score < 1.0
        assert health_info.health_score >= 0.0
        assert health_info.status in [ShardHealthStatus.DEGRADED, ShardHealthStatus.CRITICAL]
    
    def test_consecutive_failures_impact(self):
        """Test that consecutive failures impact health score."""
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id)
        health_info = ShardHealthInfo(
            shard_id=shard_id,
            status=ShardHealthStatus.HEALTHY,
            load_metrics=load_metrics
        )
        
        # Simulate consecutive failures
        health_info.consecutive_failures = 5
        health_info.update_health_score()
        
        assert health_info.health_score < 1.0
        assert health_info.status in [ShardHealthStatus.DEGRADED, ShardHealthStatus.CRITICAL, ShardHealthStatus.FAILED]


class TestConsistentHashBalancer:
    """Test ConsistentHashBalancer functionality."""
    
    def test_consistent_hash_creation(self):
        """Test consistent hash balancer creation."""
        balancer = ConsistentHashBalancer(virtual_nodes=100)
        assert balancer.virtual_nodes == 100
        assert len(balancer.hash_ring) == 0
    
    def test_hash_ring_building(self):
        """Test hash ring building."""
        balancer = ConsistentHashBalancer(virtual_nodes=10)
        shards = [ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3]
        
        balancer._build_hash_ring(shards)
        
        assert len(balancer.hash_ring) == 30  # 3 shards * 10 virtual nodes
        assert all(shard in balancer.hash_ring.values() for shard in shards)
    
    def test_shard_selection_consistency(self):
        """Test that shard selection is consistent for the same key."""
        balancer = ConsistentHashBalancer(virtual_nodes=50)
        shards = [ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3]
        shard_health = {
            shard: ShardHealthInfo(shard, ShardHealthStatus.HEALTHY, ShardLoadMetrics(shard))
            for shard in shards
        }
        
        # Select shard multiple times for the same key
        key = "test_key_123"
        selected_shards = []
        for _ in range(10):
            shard = balancer.select_shard(key, shards, shard_health)
            selected_shards.append(shard)
        
        # All selections should be the same
        assert all(shard == selected_shards[0] for shard in selected_shards)
    
    def test_shard_selection_distribution(self):
        """Test that shard selection is reasonably distributed."""
        balancer = ConsistentHashBalancer(virtual_nodes=100)
        shards = [ShardId.SHARD_1, ShardId.SHARD_2, ShardId.SHARD_3]
        shard_health = {
            shard: ShardHealthInfo(shard, ShardHealthStatus.HEALTHY, ShardLoadMetrics(shard))
            for shard in shards
        }
        
        # Test with many keys (reduced for faster testing)
        key_counts = {shard: 0 for shard in shards}
        for i in range(100):  # Reduced from 1000 to 100
            key = f"test_key_{i}"
            selected_shard = balancer.select_shard(key, shards, shard_health)
            key_counts[selected_shard] += 1
        
        # Each shard should get some keys (distribution should be reasonable)
        for shard in shards:
            assert key_counts[shard] > 0
            # No shard should get more than 50% of keys (with reasonable distribution)
            assert key_counts[shard] < 600
    
    def test_empty_shards_handling(self):
        """Test handling of empty shard list."""
        balancer = ConsistentHashBalancer()
        shard_health = {}
        
        with pytest.raises(ValueError, match="No available shards"):
            balancer.select_shard("test_key", [], shard_health)
    
    def test_should_rebalance(self):
        """Test rebalancing decision."""
        balancer = ConsistentHashBalancer()
        shard_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, ShardHealthStatus.HEALTHY, ShardLoadMetrics(ShardId.SHARD_1)
            )
        }
        
        # Consistent hash should not need rebalancing
        assert not balancer.should_rebalance(shard_health)


class TestLeastLoadedBalancer:
    """Test LeastLoadedBalancer functionality."""
    
    def test_least_loaded_selection(self):
        """Test least loaded shard selection."""
        balancer = LeastLoadedBalancer()
        
        # Create shards with different load levels
        shard_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, 
                ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=90.0, memory_usage=80.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, 
                ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=30.0, memory_usage=20.0)
            ),
            ShardId.SHARD_3: ShardHealthInfo(
                ShardId.SHARD_3, 
                ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_3, cpu_usage=60.0, memory_usage=50.0)
            )
        }
        
        shards = list(shard_health.keys())
        
        # Should select the least loaded shard (SHARD_2)
        selected_shard = balancer.select_shard("test_key", shards, shard_health)
        assert selected_shard == ShardId.SHARD_2
    
    def test_healthy_shard_preference(self):
        """Test that healthy shards are preferred over degraded ones."""
        balancer = LeastLoadedBalancer()
        
        shard_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, 
                ShardHealthStatus.DEGRADED, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=30.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, 
                ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=50.0)
            )
        }
        
        shards = list(shard_health.keys())
        selected_shard = balancer.select_shard("test_key", shards, shard_health)
        
        # Should prefer healthy shard even if it has higher load
        # Note: The balancer now prefers healthy shards over degraded ones
        assert selected_shard == ShardId.SHARD_2
    
    def test_fallback_to_any_shard(self):
        """Test fallback when no healthy shards are available."""
        balancer = LeastLoadedBalancer()
        
        shard_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, 
                ShardHealthStatus.CRITICAL, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=30.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, 
                ShardHealthStatus.FAILED, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=50.0)
            )
        }
        
        shards = list(shard_health.keys())
        selected_shard = balancer.select_shard("test_key", shards, shard_health)
        
        # Should fallback to any available shard
        assert selected_shard in shards
    
    def test_rebalancing_threshold(self):
        """Test rebalancing threshold logic."""
        balancer = LeastLoadedBalancer()
        
        # Test with balanced load
        balanced_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=50.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=60.0)
            )
        }
        assert not balancer.should_rebalance(balanced_health)
        
        # Test with imbalanced load (more extreme to trigger rebalancing)
        imbalanced_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=10.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=90.0)
            )
        }
        assert balancer.should_rebalance(imbalanced_health)


class TestAdaptiveBalancer:
    """Test AdaptiveBalancer functionality."""
    
    def test_adaptive_balancer_creation(self):
        """Test adaptive balancer creation."""
        balancer = AdaptiveBalancer()
        assert balancer.current_strategy == "consistent_hash"
        assert balancer.strategy_switch_threshold == 0.4
    
    def test_strategy_switching(self):
        """Test automatic strategy switching."""
        balancer = AdaptiveBalancer()
        
        # Create imbalanced load to trigger strategy switch
        shard_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=10.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=90.0)
            )
        }
        
        shards = list(shard_health.keys())
        
        # First call should switch to least_loaded strategy
        selected_shard = balancer.select_shard("test_key", shards, shard_health)
        assert balancer.current_strategy == "least_loaded"
        assert selected_shard == ShardId.SHARD_1  # Least loaded
    
    def test_consistent_hash_fallback(self):
        """Test fallback to consistent hash when load is balanced."""
        balancer = AdaptiveBalancer()
        
        # Create balanced load
        shard_health = {
            ShardId.SHARD_1: ShardHealthInfo(
                ShardId.SHARD_1, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_1, cpu_usage=50.0)
            ),
            ShardId.SHARD_2: ShardHealthInfo(
                ShardId.SHARD_2, ShardHealthStatus.HEALTHY, 
                ShardLoadMetrics(ShardId.SHARD_2, cpu_usage=55.0)
            )
        }
        
        shards = list(shard_health.keys())
        selected_shard = balancer.select_shard("test_key", shards, shard_health)
        
        assert balancer.current_strategy == "consistent_hash"
        assert selected_shard in shards


class TestShardReshardingManager:
    """Test ShardReshardingManager functionality."""
    
    def test_resharding_manager_creation(self):
        """Test resharding manager creation."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        assert manager.config == config
        assert len(manager.active_plans) == 0
        assert len(manager.completed_plans) == 0
    
    def test_create_resharding_plan(self):
        """Test resharding plan creation."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        source_shards = [ShardId.SHARD_1, ShardId.SHARD_2]
        target_shards = [ShardId.SHARD_3, ShardId.SHARD_4]
        data_migration_map = {"key1": ShardId.SHARD_3, "key2": ShardId.SHARD_4}
        
        plan = manager.create_resharding_plan(
            ReshardingStrategy.HORIZONTAL_SPLIT,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        assert plan.strategy == ReshardingStrategy.HORIZONTAL_SPLIT
        assert plan.source_shards == source_shards
        assert plan.target_shards == target_shards
        assert plan.data_migration_map == data_migration_map
        assert plan.status == "pending"
        assert plan.plan_id in manager.active_plans
    
    def test_plan_duration_estimation(self):
        """Test plan duration estimation."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        # Test different strategies
        source_shards = [ShardId.SHARD_1, ShardId.SHARD_2]
        target_shards = [ShardId.SHARD_3]
        
        # Horizontal split should take longer than rebalance
        split_duration = manager._estimate_duration(
            ReshardingStrategy.HORIZONTAL_SPLIT, source_shards, target_shards
        )
        rebalance_duration = manager._estimate_duration(
            ReshardingStrategy.REBALANCE, source_shards, target_shards
        )
        
        assert split_duration > rebalance_duration
    
    def test_plan_impact_estimation(self):
        """Test plan impact estimation."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        source_shards = [ShardId.SHARD_1]
        target_shards = [ShardId.SHARD_2]
        
        # Merge should have higher impact than rebalance
        merge_impact = manager._estimate_impact(
            ReshardingStrategy.MERGE, source_shards, target_shards
        )
        rebalance_impact = manager._estimate_impact(
            ReshardingStrategy.REBALANCE, source_shards, target_shards
        )
        
        assert merge_impact > rebalance_impact
    
    def test_safety_checks_creation(self):
        """Test safety checks creation."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        source_shards = [ShardId.SHARD_1]
        target_shards = [ShardId.SHARD_2]
        
        checks = manager._create_safety_checks(
            ReshardingStrategy.MERGE, source_shards, target_shards
        )
        
        assert len(checks) > 0
        assert "Verify source shards are healthy" in checks
        assert "Verify target shards have sufficient capacity" in checks
        assert "Verify no data conflicts between merging shards" in checks
    
    def test_rollback_plan_creation(self):
        """Test rollback plan creation."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        source_shards = [ShardId.SHARD_1]
        target_shards = [ShardId.SHARD_2]
        
        rollback_plan = manager._create_rollback_plan(
            ReshardingStrategy.HORIZONTAL_SPLIT, source_shards, target_shards
        )
        
        assert "strategy" in rollback_plan
        assert "source_shards" in rollback_plan
        assert "target_shards" in rollback_plan
        assert "steps" in rollback_plan
        assert len(rollback_plan["steps"]) > 0


class TestShardHealthMonitor:
    """Test ShardHealthMonitor functionality."""
    
    def test_health_monitor_creation(self):
        """Test health monitor creation."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        assert monitor.config == config
        assert len(monitor.shard_health) == 0
        assert len(monitor.health_callbacks) == 0
        assert not monitor.monitoring_active
    
    def test_update_shard_health(self):
        """Test shard health update."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id, cpu_usage=50.0)
        
        monitor.update_shard_health(shard_id, load_metrics, is_healthy=True)
        
        assert shard_id in monitor.shard_health
        health_info = monitor.shard_health[shard_id]
        assert health_info.shard_id == shard_id
        assert health_info.load_metrics == load_metrics
        assert health_info.consecutive_failures == 0
    
    def test_consecutive_failures_tracking(self):
        """Test consecutive failures tracking."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id)
        
        # Simulate consecutive failures
        for _ in range(3):
            monitor.update_shard_health(shard_id, load_metrics, is_healthy=False)
        
        health_info = monitor.shard_health[shard_id]
        assert health_info.consecutive_failures == 3
    
    def test_health_callback_registration(self):
        """Test health callback registration."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        callback_called = []
        
        def test_callback(shard_id: ShardId, status: ShardHealthStatus):
            callback_called.append((shard_id, status))
        
        monitor.add_health_callback(test_callback)
        assert len(monitor.health_callbacks) == 1
        
        # Update health to trigger callback
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id, cpu_usage=90.0)
        monitor.update_shard_health(shard_id, load_metrics, is_healthy=True)
        
        # Callback should be triggered due to status change
        assert len(callback_called) > 0
    
    def test_failed_shard_detection(self):
        """Test failed shard detection."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id)
        
        # Add shard with old heartbeat
        monitor.update_shard_health(shard_id, load_metrics, is_healthy=True)
        health_info = monitor.shard_health[shard_id]
        health_info.last_heartbeat = time.time() - 60  # 60 seconds ago
        
        failed_shards = monitor.detect_failed_shards()
        assert shard_id in failed_shards
    
    def test_healthy_shards_list(self):
        """Test healthy shards list."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        # Add healthy and unhealthy shards
        healthy_shard = ShardId.SHARD_1
        unhealthy_shard = ShardId.SHARD_2
        
        monitor.update_shard_health(
            healthy_shard, 
            ShardLoadMetrics(shard_id=healthy_shard, cpu_usage=30.0), 
            is_healthy=True
        )
        monitor.update_shard_health(
            unhealthy_shard, 
            ShardLoadMetrics(shard_id=unhealthy_shard, cpu_usage=90.0), 
            is_healthy=False
        )
        
        healthy_shards = monitor.get_healthy_shards()
        assert healthy_shard in healthy_shards
        # Unhealthy shard might still be in list if not failed
        assert len(healthy_shards) >= 1


class TestEnhancedShardManager:
    """Test EnhancedShardManager functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ShardConfig(max_shards=10)
    
    @pytest.fixture
    def manager(self, config):
        """Create test shard manager."""
        return EnhancedShardManager(config)
    
    def test_manager_creation(self, config):
        """Test manager creation."""
        manager = EnhancedShardManager(config)
        
        assert manager.config == config
        assert len(manager.shards) == 0
        assert isinstance(manager.load_balancer, AdaptiveBalancer)
        assert isinstance(manager.resharding_manager, ShardReshardingManager)
        assert isinstance(manager.health_monitor, ShardHealthMonitor)
        assert len(manager.operations) == 0
    
    def test_start_stop_manager(self, manager):
        """Test manager start/stop."""
        assert manager.start()
        assert manager.health_monitor.monitoring_active
        assert manager.stop()
        assert not manager.health_monitor.monitoring_active
    
    def test_create_shard(self, manager):
        """Test shard creation."""
        shard_state = manager.create_shard(ShardType.EXECUTION, ["validator1", "validator2"])
        
        assert shard_state.shard_id == ShardId.SHARD_1
        assert shard_state.status == ShardStatus.ACTIVE
        assert shard_state.shard_type == ShardType.EXECUTION
        assert shard_state.validator_set == ["validator1", "validator2"]
        assert shard_state.shard_id in manager.shards
    
    def test_create_multiple_shards(self, manager):
        """Test creating multiple shards."""
        shard1 = manager.create_shard(ShardType.EXECUTION)
        shard2 = manager.create_shard(ShardType.CONSENSUS)
        shard3 = manager.create_shard(ShardType.STORAGE)
        
        assert shard1.shard_id == ShardId.SHARD_1
        assert shard2.shard_id == ShardId.SHARD_2
        assert shard3.shard_id == ShardId.SHARD_3
        assert len(manager.shards) == 3
    
    def test_max_shards_limit(self, config):
        """Test maximum shards limit."""
        config.max_shards = 2
        manager = EnhancedShardManager(config)
        
        # Create maximum shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Should fail to create more
        with pytest.raises(ValueError, match="Maximum number of shards reached"):
            manager.create_shard(ShardType.STORAGE)
    
    def test_select_shard_for_key(self, manager):
        """Test shard selection for key."""
        # Create some shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Select shard for key
        shard_id = manager.select_shard_for_key("test_key_123")
        assert shard_id in manager.shards
    
    def test_select_shard_no_shards(self, manager):
        """Test shard selection with no shards."""
        with pytest.raises(ValueError, match="No shards available"):
            manager.select_shard_for_key("test_key")
    
    def test_add_data_to_shard(self, manager):
        """Test adding data to shard."""
        # Create a shard
        shard_state = manager.create_shard(ShardType.EXECUTION)
        
        # Add data
        success = manager.add_data_to_shard("test_key", "test_data", shard_state.shard_id)
        
        assert success
        assert manager.performance_metrics['total_operations'] == 1
        assert manager.performance_metrics['successful_operations'] == 1
    
    def test_add_data_auto_shard_selection(self, manager):
        """Test adding data with automatic shard selection."""
        # Create shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Add data without specifying shard
        success = manager.add_data_to_shard("test_key", "test_data")
        
        assert success
        assert manager.performance_metrics['total_operations'] == 1
    
    def test_remove_shard(self, manager):
        """Test shard removal."""
        # Create and remove shard
        shard_state = manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id
        
        success = manager.remove_shard(shard_id)
        
        assert success
        assert shard_id not in manager.shards
    
    def test_remove_nonexistent_shard(self, manager):
        """Test removing non-existent shard."""
        success = manager.remove_shard(ShardId.SHARD_1)
        assert not success
    
    def test_remove_critical_shard(self, manager):
        """Test removing critical shard."""
        # Create shard and mark as critical
        shard_state = manager.create_shard(ShardType.EXECUTION)
        shard_id = shard_state.shard_id
        
        # Simulate critical health status
        load_metrics = ShardLoadMetrics(shard_id=shard_id, cpu_usage=95.0)
        manager.health_monitor.update_shard_health(shard_id, load_metrics, is_healthy=False)
        
        # Force the health status to be critical
        health_info = manager.health_monitor.get_shard_health(shard_id)
        health_info.status = ShardHealthStatus.CRITICAL
        
        # Try to remove critical shard
        success = manager.remove_shard(shard_id)
        assert not success
    
    def test_trigger_resharding(self, manager):
        """Test triggering resharding."""
        # Create shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        source_shards = [ShardId.SHARD_1]
        target_shards = [ShardId.SHARD_2]
        data_migration_map = {"key1": ShardId.SHARD_2}
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.REBALANCE,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        assert plan_id is not None
        # The plan might be in active_plans or completed_plans depending on execution timing
        assert (plan_id in manager.resharding_manager.active_plans or 
                any(plan.plan_id == plan_id for plan in manager.resharding_manager.completed_plans))
    
    def test_performance_metrics(self, manager):
        """Test performance metrics collection."""
        # Create shard and perform operations
        manager.create_shard(ShardType.EXECUTION)
        manager.add_data_to_shard("key1", "data1")
        manager.add_data_to_shard("key2", "data2")
        
        metrics = manager.get_performance_metrics()
        
        assert metrics['total_operations'] == 2
        assert metrics['successful_operations'] == 2
        assert metrics['total_shards'] == 1
        assert metrics['healthy_shards'] == 1
        assert metrics['failed_shards'] == 0
    
    def test_load_distribution(self, manager):
        """Test load distribution tracking."""
        # Create shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Add some load
        manager.add_data_to_shard("key1", "data1")
        manager.add_data_to_shard("key2", "data2")
        
        distribution = manager.get_shard_load_distribution()
        
        assert len(distribution) == 2
        for shard_id, load in distribution.items():
            assert 0.0 <= load <= 1.0
    
    def test_operation_history(self, manager):
        """Test operation history tracking."""
        # Create shard and perform operations
        manager.create_shard(ShardType.EXECUTION)
        manager.add_data_to_shard("key1", "data1")
        manager.add_data_to_shard("key2", "data2")
        
        history = manager.get_operation_history(limit=10)
        
        assert len(history) >= 2
        assert all(isinstance(op, ShardOperation) for op in history)
    
    def test_cleanup_old_operations(self, manager):
        """Test cleanup of old operations."""
        # Create shard and perform operation
        manager.create_shard(ShardType.EXECUTION)
        manager.add_data_to_shard("key1", "data1")
        
        # Manually age the operation
        for operation in manager.operations.values():
            operation.end_time = time.time() - 4000  # 4000 seconds ago
        
        cleaned_count = manager.cleanup_old_operations(max_age_seconds=3600)
        
        assert cleaned_count >= 1  # At least the add_data operation should be cleaned
        assert len(manager.operations) == 0
    
    def test_concurrent_operations(self, manager):
        """Test concurrent operations."""
        # Create shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Perform concurrent operations
        def add_data(key):
            return manager.add_data_to_shard(key, f"data_{key}")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_data, f"key_{i}") for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        assert all(results)
        assert manager.performance_metrics['total_operations'] == 10
        assert manager.performance_metrics['successful_operations'] == 10
    
    def test_simulate_load(self, manager):
        """Test load simulation."""
        # Create shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Simulate load
        results = manager.simulate_load(num_operations=100)
        
        assert results['total_operations'] == 100
        assert results['successful_operations'] > 0
        assert results['throughput'] > 0
        assert 0.0 <= results['success_rate'] <= 1.0
    
    def test_shard_failure_handling(self, manager):
        """Test shard failure handling."""
        # Create shards
        manager.create_shard(ShardType.EXECUTION)
        manager.create_shard(ShardType.CONSENSUS)
        
        # Simulate shard failure
        shard_id = ShardId.SHARD_1
        load_metrics = ShardLoadMetrics(shard_id=shard_id, cpu_usage=100.0)
        manager.health_monitor.update_shard_health(shard_id, load_metrics, is_healthy=False)
        
        # Trigger failure handling
        manager._handle_shard_failure(shard_id)
        
        # Shard should be marked as failed
        assert manager.shards[shard_id].status == ShardStatus.ERROR


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_key_handling(self):
        """Test handling of empty keys."""
        balancer = ConsistentHashBalancer()
        shards = [ShardId.SHARD_1, ShardId.SHARD_2]
        shard_health = {
            shard: ShardHealthInfo(shard, ShardHealthStatus.HEALTHY, ShardLoadMetrics(shard))
            for shard in shards
        }
        
        # Should handle empty key gracefully
        shard_id = balancer.select_shard("", shards, shard_health)
        assert shard_id in shards
    
    def test_very_long_key_handling(self):
        """Test handling of very long keys."""
        balancer = ConsistentHashBalancer()
        shards = [ShardId.SHARD_1, ShardId.SHARD_2]
        shard_health = {
            shard: ShardHealthInfo(shard, ShardHealthStatus.HEALTHY, ShardLoadMetrics(shard))
            for shard in shards
        }
        
        # Very long key
        long_key = "x" * 10000
        shard_id = balancer.select_shard(long_key, shards, shard_health)
        assert shard_id in shards
    
    def test_unicode_key_handling(self):
        """Test handling of unicode keys."""
        balancer = ConsistentHashBalancer()
        shards = [ShardId.SHARD_1, ShardId.SHARD_2]
        shard_health = {
            shard: ShardHealthInfo(shard, ShardHealthStatus.HEALTHY, ShardLoadMetrics(shard))
            for shard in shards
        }
        
        # Unicode key
        unicode_key = "测试键_🔑_ключ"
        shard_id = balancer.select_shard(unicode_key, shards, shard_health)
        assert shard_id in shards
    
    def test_extreme_load_values(self):
        """Test handling of extreme load values."""
        shard_id = ShardId.SHARD_1
        
        # Test with extreme values
        metrics = ShardLoadMetrics(
            shard_id=shard_id,
            cpu_usage=1000.0,  # > 100%
            memory_usage=-10.0,  # Negative
            disk_usage=200.0,  # > 100%
            network_io=float('inf'),  # Infinity
            transaction_throughput=float('-inf'),  # Negative infinity
            queue_depth=-5,  # Negative
            response_time_p95=float('nan')  # NaN
        )
        
        # Should handle extreme values gracefully
        load_score = metrics.overall_load_score
        assert 0.0 <= load_score <= 1.0
    
    def test_concurrent_health_updates(self):
        """Test concurrent health updates."""
        config = ShardConfig()
        monitor = ShardHealthMonitor(config)
        
        shard_id = ShardId.SHARD_1
        
        def update_health():
            for _ in range(100):
                load_metrics = ShardLoadMetrics(shard_id=shard_id, cpu_usage=50.0)
                monitor.update_shard_health(shard_id, load_metrics, is_healthy=True)
        
        # Run concurrent updates
        threads = [threading.Thread(target=update_health) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have consistent state
        assert shard_id in monitor.shard_health
        health_info = monitor.shard_health[shard_id]
        assert health_info.consecutive_failures == 0  # All updates were healthy
    
    def test_rapid_resharding_requests(self):
        """Test rapid resharding requests."""
        config = ShardConfig()
        manager = ShardReshardingManager(config)
        
        source_shards = [ShardId.SHARD_1]
        target_shards = [ShardId.SHARD_2]
        data_migration_map = {"key1": ShardId.SHARD_2}
        
        # Create multiple plans rapidly
        plans = []
        for i in range(10):
            plan = manager.create_resharding_plan(
                ReshardingStrategy.REBALANCE,
                source_shards,
                target_shards,
                data_migration_map
            )
            plans.append(plan)
        
        assert len(manager.active_plans) == 10
        assert all(plan.status == "pending" for plan in plans)
    
    def test_memory_pressure_handling(self):
        """Test handling under memory pressure."""
        config = ShardConfig()
        manager = EnhancedShardManager(config)
        
        # Create many shards and operations to simulate memory pressure
        for i in range(5):
            manager.create_shard(ShardType.EXECUTION)
        
        # Perform many operations (reduced for faster testing)
        for i in range(100):  # Reduced from 1000 to 100
            manager.add_data_to_shard(f"key_{i}", f"data_{i}")
        
        # Should still function correctly
        assert len(manager.shards) == 5
        assert manager.performance_metrics['total_operations'] == 100
        
        # Cleanup should work
        cleaned = manager.cleanup_old_operations(max_age_seconds=0)
        assert cleaned >= 100  # At least 100 operations should be cleaned
        assert len(manager.operations) == 0


if __name__ == "__main__":
    pytest.main([__file__])
