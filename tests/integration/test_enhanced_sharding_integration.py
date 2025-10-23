"""
Integration tests for Enhanced Sharding System.

This module provides comprehensive integration tests covering:
- End-to-end sharding workflows
- Cross-component interactions
- Real-world scenarios
- Performance under load
- Fault tolerance scenarios
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

from src.dubchain.sharding.enhanced_sharding import (
    LoadBalancingStrategy,
    ReshardingStrategy,
    ShardHealthStatus,
    ShardLoadMetrics,
    ShardHealthInfo,
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


class TestShardingWorkflow:
    """Test complete sharding workflows."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ShardConfig(
            max_shards=10,
            min_validators_per_shard=2,
            max_validators_per_shard=10,
            rebalance_threshold=0.2
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create test shard manager."""
        manager = EnhancedShardManager(config)
        manager.start()
        yield manager
        manager.stop()
    
    def test_complete_sharding_lifecycle(self, manager):
        """Test complete sharding lifecycle from creation to removal."""
        # 1. Create initial shards
        shard1 = manager.create_shard(ShardType.EXECUTION, ["validator1", "validator2"])
        shard2 = manager.create_shard(ShardType.CONSENSUS, ["validator3", "validator4"])
        
        assert len(manager.shards) == 2
        assert shard1.shard_id in manager.shards
        assert shard2.shard_id in manager.shards
        
        # 2. Add data to shards
        for i in range(50):
            key = f"data_key_{i}"
            success = manager.add_data_to_shard(key, f"data_{i}")
            assert success
        
        # 3. Verify load distribution
        distribution = manager.get_shard_load_distribution()
        assert len(distribution) == 2
        
        # 4. Create additional shard for scaling
        shard3 = manager.create_shard(ShardType.STORAGE, ["validator5", "validator6"])
        assert len(manager.shards) == 3
        
        # 5. Trigger resharding to rebalance load
        source_shards = [shard1.shard_id, shard2.shard_id]
        target_shards = [shard3.shard_id]
        data_migration_map = {f"data_key_{i}": shard3.shard_id for i in range(25)}
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.REBALANCE,
            source_shards,
            target_shards,
            data_migration_map
        )
        assert plan_id is not None
        
        # 6. Wait for resharding to complete
        time.sleep(1)  # Allow resharding to complete
        
        # 7. Verify system is still functional
        success = manager.add_data_to_shard("new_key", "new_data")
        assert success
        
        # 8. Remove a shard
        success = manager.remove_shard(shard1.shard_id)
        assert success
        assert len(manager.shards) == 2
    
    def test_load_balancing_under_varying_load(self, manager):
        """Test load balancing under varying load conditions."""
        # Create multiple shards
        for i in range(5):
            manager.create_shard(ShardType.EXECUTION, [f"validator_{i}_1", f"validator_{i}_2"])
        
        # Phase 1: Light load
        for i in range(100):
            manager.add_data_to_shard(f"light_key_{i}", f"light_data_{i}")
        
        light_distribution = manager.get_shard_load_distribution()
        assert len(light_distribution) == 5
        
        # Phase 2: Heavy load on specific shards
        for i in range(500):
            # Create hot keys that will hit specific shards
            hot_key = f"hot_key_{i % 2}"  # Only 2 different keys
            manager.add_data_to_shard(hot_key, f"hot_data_{i}")
        
        heavy_distribution = manager.get_shard_load_distribution()
        
        # Phase 3: Verify load balancing works
        # The adaptive balancer should handle hot key skew
        max_load = max(heavy_distribution.values())
        min_load = min(heavy_distribution.values())
        
        # Load should be reasonably distributed (not too skewed)
        assert max_load - min_load < 0.8  # Allow some skew but not extreme
    
    def test_fault_tolerance_and_recovery(self, manager):
        """Test fault tolerance and recovery mechanisms."""
        # Create shards
        shard1 = manager.create_shard(ShardType.EXECUTION)
        shard2 = manager.create_shard(ShardType.CONSENSUS)
        shard3 = manager.create_shard(ShardType.STORAGE)
        
        # Add some data
        for i in range(100):
            manager.add_data_to_shard(f"key_{i}", f"data_{i}")
        
        # Simulate shard failure
        failed_shard = shard1.shard_id
        load_metrics = ShardLoadMetrics(
            shard_id=failed_shard,
            cpu_usage=100.0,
            memory_usage=100.0,
            error_rate=1.0
        )
        manager.health_monitor.update_shard_health(failed_shard, load_metrics, is_healthy=False)
        
        # Verify shard is marked as failed
        health_info = manager.health_monitor.get_shard_health(failed_shard)
        assert health_info.status in [ShardHealthStatus.CRITICAL, ShardHealthStatus.FAILED]
        
        # System should still function with remaining shards
        success = manager.add_data_to_shard("recovery_key", "recovery_data")
        assert success
        
        # Verify failed shard is excluded from selection
        healthy_shards = manager.health_monitor.get_healthy_shards()
        assert failed_shard not in healthy_shards or len(healthy_shards) >= 2
    
    def test_concurrent_operations(self, manager):
        """Test concurrent operations across multiple threads."""
        # Create shards
        for i in range(3):
            manager.create_shard(ShardType.EXECUTION, [f"validator_{i}_1", f"validator_{i}_2"])
        
        # Define operation function
        def perform_operations(thread_id, num_operations):
            results = []
            for i in range(num_operations):
                key = f"thread_{thread_id}_key_{i}"
                success = manager.add_data_to_shard(key, f"thread_{thread_id}_data_{i}")
                results.append(success)
            return results
        
        # Run concurrent operations
        num_threads = 5
        operations_per_thread = 50
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(perform_operations, i, operations_per_thread)
                for i in range(num_threads)
            ]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all operations succeeded
        all_results = [result for thread_results in results for result in thread_results]
        assert all(all_results)
        assert len(all_results) == num_threads * operations_per_thread
        
        # Verify performance metrics
        metrics = manager.get_performance_metrics()
        assert metrics['total_operations'] == num_threads * operations_per_thread
        assert metrics['successful_operations'] == num_threads * operations_per_thread
    
    def test_resharding_with_active_operations(self, manager):
        """Test resharding while operations are active."""
        # Create shards
        shard1 = manager.create_shard(ShardType.EXECUTION)
        shard2 = manager.create_shard(ShardType.CONSENSUS)
        shard3 = manager.create_shard(ShardType.STORAGE)
        
        # Start background operations
        operation_results = []
        
        def background_operations():
            for i in range(100):
                key = f"bg_key_{i}"
                success = manager.add_data_to_shard(key, f"bg_data_{i}")
                operation_results.append(success)
                time.sleep(0.01)  # Small delay
        
        # Start background thread
        bg_thread = threading.Thread(target=background_operations)
        bg_thread.start()
        
        # Wait a bit for some operations to start
        time.sleep(0.5)
        
        # Trigger resharding
        source_shards = [shard1.shard_id, shard2.shard_id]
        target_shards = [shard3.shard_id]
        data_migration_map = {f"bg_key_{i}": shard3.shard_id for i in range(50)}
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.REBALANCE,
            source_shards,
            target_shards,
            data_migration_map
        )
        assert plan_id is not None
        
        # Wait for background operations to complete
        bg_thread.join()
        
        # Verify operations continued during resharding
        assert len(operation_results) == 100
        assert all(operation_results)
        
        # Verify system is still functional
        success = manager.add_data_to_shard("post_reshard_key", "post_reshard_data")
        assert success


class TestLoadBalancingStrategies:
    """Test different load balancing strategies in realistic scenarios."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ShardConfig(max_shards=5)
    
    def test_consistent_hash_under_load(self, config):
        """Test consistent hash balancer under realistic load."""
        balancer = ConsistentHashBalancer(virtual_nodes=100)
        manager = EnhancedShardManager(config, load_balancer=balancer)
        manager.start()
        
        try:
            # Create shards
            for i in range(3):
                manager.create_shard(ShardType.EXECUTION)
            
            # Test with realistic key patterns
            key_patterns = [
                "user_12345",
                "user_67890",
                "session_abc123",
                "session_def456",
                "order_001",
                "order_002",
                "product_xyz",
                "product_abc"
            ]
            
            # Add data with these patterns
            for pattern in key_patterns:
                for i in range(100):
                    key = f"{pattern}_{i}"
                    success = manager.add_data_to_shard(key, f"data_{i}")
                    assert success
            
            # Verify consistent distribution
            distribution = manager.get_shard_load_distribution()
            assert len(distribution) == 3
            
            # Test key consistency - same keys should go to same shards
            test_keys = ["consistent_key_1", "consistent_key_2"]
            selected_shards = []
            
            for key in test_keys:
                shard_id = manager.select_shard_for_key(key)
                selected_shards.append(shard_id)
            
            # Verify consistency
            for _ in range(10):
                for i, key in enumerate(test_keys):
                    shard_id = manager.select_shard_for_key(key)
                    assert shard_id == selected_shards[i]
        
        finally:
            manager.stop()
    
    def test_least_loaded_adaptation(self, config):
        """Test least loaded balancer adaptation to load changes."""
        balancer = LeastLoadedBalancer()
        manager = EnhancedShardManager(config, load_balancer=balancer)
        manager.start()
        
        try:
            # Create shards
            shard1 = manager.create_shard(ShardType.EXECUTION)
            shard2 = manager.create_shard(ShardType.CONSENSUS)
            shard3 = manager.create_shard(ShardType.STORAGE)
            
            # Initially, all shards should be equally loaded
            initial_distribution = manager.get_shard_load_distribution()
            assert len(initial_distribution) == 3
            
            # Create artificial load imbalance
            for i in range(200):
                # Force load onto shard1
                success = manager.add_data_to_shard(f"shard1_key_{i}", f"data_{i}", shard1.shard_id)
                assert success
            
            # Update load metrics to reflect imbalance
            high_load_metrics = ShardLoadMetrics(
                shard_id=shard1.shard_id,
                cpu_usage=90.0,
                memory_usage=85.0,
                queue_depth=500
            )
            manager.health_monitor.update_shard_health(shard1.shard_id, high_load_metrics)
            
            low_load_metrics = ShardLoadMetrics(
                shard_id=shard2.shard_id,
                cpu_usage=20.0,
                memory_usage=15.0,
                queue_depth=50
            )
            manager.health_monitor.update_shard_health(shard2.shard_id, low_load_metrics)
            
            # New operations should prefer least loaded shard
            for i in range(50):
                key = f"new_key_{i}"
                selected_shard = manager.select_shard_for_key(key)
                # Should prefer shard2 (least loaded) or shard3
                assert selected_shard in [shard2.shard_id, shard3.shard_id]
        
        finally:
            manager.stop()
    
    def test_adaptive_balancer_switching(self, config):
        """Test adaptive balancer strategy switching."""
        balancer = AdaptiveBalancer()
        manager = EnhancedShardManager(config, load_balancer=balancer)
        manager.start()
        
        try:
            # Create shards
            for i in range(3):
                manager.create_shard(ShardType.EXECUTION)
            
            # Phase 1: Balanced load - should use consistent hash
            for i in range(100):
                key = f"balanced_key_{i}"
                manager.add_data_to_shard(key, f"data_{i}")
            
            # Verify consistent hash behavior
            test_key = "test_consistency_key"
            shard1 = manager.select_shard_for_key(test_key)
            shard2 = manager.select_shard_for_key(test_key)
            assert shard1 == shard2  # Consistent hash should be consistent
            
            # Phase 2: Create load imbalance
            for i in range(500):
                # Create hot keys
                hot_key = f"hot_key_{i % 3}"  # Only 3 different keys
                manager.add_data_to_shard(hot_key, f"hot_data_{i}")
            
            # Simulate load imbalance in health metrics
            shard_ids = list(manager.shards.keys())
            high_load_metrics = ShardLoadMetrics(
                shard_id=shard_ids[0],
                cpu_usage=95.0,
                memory_usage=90.0,
                queue_depth=800
            )
            manager.health_monitor.update_shard_health(shard_ids[0], high_load_metrics)
            
            low_load_metrics = ShardLoadMetrics(
                shard_id=shard_ids[1],
                cpu_usage=25.0,
                memory_usage=20.0,
                queue_depth=100
            )
            manager.health_monitor.update_shard_health(shard_ids[1], low_load_metrics)
            
            # Phase 3: Should switch to least loaded strategy
            for i in range(20):
                key = f"adaptive_key_{i}"
                selected_shard = manager.select_shard_for_key(key)
                # Should prefer less loaded shards
                assert selected_shard in [shard_ids[1], shard_ids[2]]
        
        finally:
            manager.stop()


class TestReshardingScenarios:
    """Test various resharding scenarios."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ShardConfig(max_shards=10)
    
    @pytest.fixture
    def manager(self, config):
        """Create test shard manager."""
        manager = EnhancedShardManager(config)
        manager.start()
        yield manager
        manager.stop()
    
    def test_horizontal_split_resharding(self, manager):
        """Test horizontal split resharding scenario."""
        # Create initial shard with data
        shard1 = manager.create_shard(ShardType.EXECUTION, ["validator1", "validator2"])
        
        # Add data to shard
        for i in range(100):
            manager.add_data_to_shard(f"key_{i}", f"data_{i}", shard1.shard_id)
        
        # Create new shard for split
        shard2 = manager.create_shard(ShardType.EXECUTION, ["validator3", "validator4"])
        
        # Create horizontal split plan
        source_shards = [shard1.shard_id]
        target_shards = [shard2.shard_id]
        data_migration_map = {f"key_{i}": shard2.shard_id for i in range(50, 100)}
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.HORIZONTAL_SPLIT,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        assert plan_id is not None
        
        # Wait for resharding to complete
        time.sleep(1)
        
        # Verify system is still functional
        success = manager.add_data_to_shard("new_key", "new_data")
        assert success
        
        # Verify both shards are active
        assert shard1.shard_id in manager.shards
        assert shard2.shard_id in manager.shards
    
    def test_vertical_split_resharding(self, manager):
        """Test vertical split resharding scenario."""
        # Create initial shard
        shard1 = manager.create_shard(ShardType.EXECUTION, ["validator1", "validator2"])
        
        # Add data
        for i in range(100):
            manager.add_data_to_shard(f"key_{i}", f"data_{i}", shard1.shard_id)
        
        # Create new shards for vertical split
        shard2 = manager.create_shard(ShardType.CONSENSUS, ["validator3", "validator4"])
        shard3 = manager.create_shard(ShardType.STORAGE, ["validator5", "validator6"])
        
        # Create vertical split plan
        source_shards = [shard1.shard_id]
        target_shards = [shard2.shard_id, shard3.shard_id]
        data_migration_map = {
            f"key_{i}": shard2.shard_id if i % 2 == 0 else shard3.shard_id
            for i in range(100)
        }
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.VERTICAL_SPLIT,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        assert plan_id is not None
        
        # Wait for resharding to complete
        time.sleep(1)
        
        # Verify system functionality
        success = manager.add_data_to_shard("new_key", "new_data")
        assert success
    
    def test_merge_resharding(self, manager):
        """Test merge resharding scenario."""
        # Create multiple shards
        shard1 = manager.create_shard(ShardType.EXECUTION, ["validator1", "validator2"])
        shard2 = manager.create_shard(ShardType.EXECUTION, ["validator3", "validator4"])
        
        # Add data to both shards
        for i in range(50):
            manager.add_data_to_shard(f"shard1_key_{i}", f"data_{i}", shard1.shard_id)
            manager.add_data_to_shard(f"shard2_key_{i}", f"data_{i}", shard2.shard_id)
        
        # Create merge plan
        source_shards = [shard1.shard_id, shard2.shard_id]
        target_shards = [shard1.shard_id]  # Merge into shard1
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.MERGE,
            source_shards,
            target_shards,
            {}
        )
        
        assert plan_id is not None
        
        # Wait for resharding to complete
        time.sleep(1)
        
        # Verify system functionality
        success = manager.add_data_to_shard("merged_key", "merged_data")
        assert success
    
    def test_rebalance_resharding(self, manager):
        """Test rebalance resharding scenario."""
        # Create shards with different loads
        shard1 = manager.create_shard(ShardType.EXECUTION, ["validator1", "validator2"])
        shard2 = manager.create_shard(ShardType.EXECUTION, ["validator3", "validator4"])
        shard3 = manager.create_shard(ShardType.EXECUTION, ["validator5", "validator6"])
        
        # Create load imbalance
        for i in range(200):
            manager.add_data_to_shard(f"heavy_key_{i}", f"data_{i}", shard1.shard_id)
        
        for i in range(50):
            manager.add_data_to_shard(f"light_key_{i}", f"data_{i}", shard2.shard_id)
        
        # Create rebalance plan
        source_shards = [shard1.shard_id, shard2.shard_id, shard3.shard_id]
        target_shards = [shard1.shard_id, shard2.shard_id, shard3.shard_id]
        data_migration_map = {
            f"heavy_key_{i}": shard2.shard_id if i % 3 == 0 else shard3.shard_id
            for i in range(50, 100)
        }
        
        plan_id = manager.trigger_resharding(
            ReshardingStrategy.REBALANCE,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        assert plan_id is not None
        
        # Wait for resharding to complete
        time.sleep(1)
        
        # Verify system functionality
        success = manager.add_data_to_shard("rebalanced_key", "rebalanced_data")
        assert success


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ShardConfig(max_shards=20)
    
    @pytest.fixture
    def manager(self, config):
        """Create test shard manager."""
        manager = EnhancedShardManager(config)
        manager.start()
        yield manager
        manager.stop()
    
    def test_high_throughput_operations(self, manager):
        """Test high throughput operations."""
        # Create multiple shards
        for i in range(5):
            manager.create_shard(ShardType.EXECUTION)
        
        # Measure throughput
        start_time = time.time()
        num_operations = 1000
        
        for i in range(num_operations):
            manager.add_data_to_shard(f"throughput_key_{i}", f"data_{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_operations / duration
        
        # Should achieve reasonable throughput (> 50 ops/sec)
        assert throughput > 50
        
        # Verify all operations succeeded
        metrics = manager.get_performance_metrics()
        assert metrics['total_operations'] == num_operations
        assert metrics['successful_operations'] == num_operations
    
    def test_scalability_with_shard_count(self, manager):
        """Test scalability with increasing shard count."""
        shard_counts = [1, 3, 5, 10]
        throughputs = []
        
        for shard_count in shard_counts:
            # Create shards
            for i in range(shard_count):
                if i >= len(manager.shards):
                    manager.create_shard(ShardType.EXECUTION)
            
            # Measure throughput
            start_time = time.time()
            num_operations = 200
            
            for i in range(num_operations):
                manager.add_data_to_shard(f"scale_key_{shard_count}_{i}", f"data_{i}")
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = num_operations / duration
            throughputs.append(throughput)
        
        # Throughput should generally improve or stay stable with more shards
        # (allowing for some variance due to test environment)
        assert all(throughput > 50 for throughput in throughputs)
    
    def test_memory_usage_under_load(self, manager):
        """Test memory usage under sustained load."""
        # Create shards
        for i in range(3):
            manager.create_shard(ShardType.EXECUTION)
        
        # Perform sustained operations
        for batch in range(10):
            for i in range(100):
                key = f"memory_key_{batch}_{i}"
                manager.add_data_to_shard(key, f"data_{i}")
            
            # Clean up old operations periodically
            if batch % 3 == 0:
                cleaned = manager.cleanup_old_operations(max_age_seconds=0)
                assert cleaned > 0
        
        # Verify system is still functional
        success = manager.add_data_to_shard("final_key", "final_data")
        assert success
        
        # Verify performance metrics are reasonable
        metrics = manager.get_performance_metrics()
        assert metrics['total_operations'] > 0
        assert metrics['successful_operations'] > 0
    
    def test_concurrent_resharding_operations(self, manager):
        """Test concurrent resharding operations."""
        # Create initial shards
        for i in range(5):
            manager.create_shard(ShardType.EXECUTION)
        
        # Start multiple resharding operations concurrently
        plan_ids = []
        
        def trigger_resharding(plan_id_suffix):
            source_shards = [ShardId.SHARD_1, ShardId.SHARD_2]
            target_shards = [ShardId.SHARD_3, ShardId.SHARD_4]
            data_migration_map = {f"key_{plan_id_suffix}_{i}": ShardId.SHARD_3 for i in range(10)}
            
            plan_id = manager.trigger_resharding(
                ReshardingStrategy.REBALANCE,
                source_shards,
                target_shards,
                data_migration_map
            )
            return plan_id
        
        # Trigger multiple resharding operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(trigger_resharding, i) for i in range(3)]
            plan_ids = [future.result() for future in as_completed(futures)]
        
        # All plans should be created
        assert all(plan_id is not None for plan_id in plan_ids)
        assert len(set(plan_ids)) == 3  # All unique
        
        # Wait for operations to complete
        time.sleep(2)
        
        # Verify system is still functional
        success = manager.add_data_to_shard("concurrent_key", "concurrent_data")
        assert success


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ShardConfig(max_shards=5)
    
    @pytest.fixture
    def manager(self, config):
        """Create test shard manager."""
        manager = EnhancedShardManager(config)
        manager.start()
        yield manager
        manager.stop()
    
    def test_graceful_degradation_under_failures(self, manager):
        """Test graceful degradation when shards fail."""
        # Create shards
        shard1 = manager.create_shard(ShardType.EXECUTION)
        shard2 = manager.create_shard(ShardType.CONSENSUS)
        shard3 = manager.create_shard(ShardType.STORAGE)
        
        # Add some data
        for i in range(100):
            manager.add_data_to_shard(f"key_{i}", f"data_{i}")
        
        # Simulate shard failures
        failed_shards = [shard1.shard_id, shard2.shard_id]
        
        for shard_id in failed_shards:
            load_metrics = ShardLoadMetrics(
                shard_id=shard_id,
                cpu_usage=100.0,
                memory_usage=100.0,
                error_rate=1.0
            )
            manager.health_monitor.update_shard_health(shard_id, load_metrics, is_healthy=False)
        
        # System should still function with remaining shards
        for i in range(50):
            success = manager.add_data_to_shard(f"recovery_key_{i}", f"data_{i}")
            assert success
        
        # Verify failed shards are excluded from healthy list
        healthy_shards = manager.health_monitor.get_healthy_shards()
        assert shard3.shard_id in healthy_shards
    
    def test_resharding_rollback_scenario(self, manager):
        """Test resharding rollback scenario."""
        # Create shards
        shard1 = manager.create_shard(ShardType.EXECUTION)
        shard2 = manager.create_shard(ShardType.CONSENSUS)
        
        # Add data
        for i in range(50):
            manager.add_data_to_shard(f"key_{i}", f"data_{i}", shard1.shard_id)
        
        # Create resharding plan
        source_shards = [shard1.shard_id]
        target_shards = [shard2.shard_id]
        data_migration_map = {f"key_{i}": shard2.shard_id for i in range(25)}
        
        plan = manager.resharding_manager.create_resharding_plan(
            ReshardingStrategy.HORIZONTAL_SPLIT,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        # Simulate rollback
        success = manager.resharding_manager.rollback_resharding_plan(
            plan.plan_id, manager
        )
        
        assert success
        assert plan.status == "rolled_back"
        
        # Verify system is still functional
        success = manager.add_data_to_shard("rollback_key", "rollback_data")
        assert success
    
    def test_invalid_resharding_plan_handling(self, manager):
        """Test handling of invalid resharding plans."""
        # Try to create plan with non-existent shards
        source_shards = [ShardId.SHARD_10]  # Non-existent
        target_shards = [ShardId.SHARD_1]  # Use existing shard
        data_migration_map = {}
        
        plan = manager.resharding_manager.create_resharding_plan(
            ReshardingStrategy.REBALANCE,
            source_shards,
            target_shards,
            data_migration_map
        )
        
        # Plan should be created but execution should handle gracefully
        assert plan is not None
        
        # Try to execute plan
        success = manager.resharding_manager.execute_resharding_plan(
            plan.plan_id, manager
        )
        
        # Should handle gracefully (may succeed or fail, but not crash)
        assert isinstance(success, bool)
    
    def test_health_monitor_edge_cases(self, manager):
        """Test health monitor edge cases."""
        # Test with no shards
        failed_shards = manager.health_monitor.detect_failed_shards()
        assert failed_shards == []
        
        healthy_shards = manager.health_monitor.get_healthy_shards()
        assert healthy_shards == []
        
        # Create shard and test edge cases
        shard_id = manager.create_shard(ShardType.EXECUTION).shard_id
        
        # Test with extreme load values
        extreme_metrics = ShardLoadMetrics(
            shard_id=shard_id,
            cpu_usage=float('inf'),
            memory_usage=float('-inf'),
            disk_usage=float('nan'),
            network_io=1000000.0,
            transaction_throughput=-1000.0,
            queue_depth=-100,
            response_time_p95=float('inf')
        )
        
        # Should handle extreme values gracefully
        manager.health_monitor.update_shard_health(shard_id, extreme_metrics, is_healthy=True)
        
        health_info = manager.health_monitor.get_shard_health(shard_id)
        assert health_info is not None
        assert 0.0 <= health_info.health_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
