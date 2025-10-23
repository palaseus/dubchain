#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Enhanced Sharding System Demo.

This script demonstrates the advanced features of the Enhanced Sharding System:
- Sophisticated load balancing
- Dynamic resharding
- Fault tolerance
- Health monitoring
- Performance metrics
- Real-world scenarios

Usage:
    python enhanced_sharding_demo.py [--scenario SCENARIO] [--operations NUM]
    
Scenarios:
    - basic: Basic sharding operations
    - load_balancing: Load balancing strategies
    - resharding: Dynamic resharding operations
    - fault_tolerance: Fault tolerance and recovery
    - performance: Performance testing
    - stress: Stress testing
    - all: Run all scenarios
"""

import logging
import argparse
import asyncio
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

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
)


class EnhancedShardingDemo:
    """Demo class for Enhanced Sharding System."""
    
    def __init__(self):
        self.config = ShardConfig(
            max_shards=10,
            min_validators_per_shard=2,
            max_validators_per_shard=10,
            rebalance_threshold=0.2
        )
        self.manager = None
        self.results = {}
    
    def setup_manager(self, load_balancer=None):
        """Setup shard manager with optional load balancer."""
        self.manager = EnhancedShardManager(self.config, load_balancer)
        self.manager.start()
        return self.manager
    
    def teardown_manager(self):
        """Teardown shard manager."""
        if self.manager:
            self.manager.stop()
            self.manager = None
    
    def basic_scenario(self, num_operations: int = 1000):
        """Demonstrate basic sharding operations."""
        logger.info("\n=== BASIC SHARDING SCENARIO ===")
        
        manager = self.setup_manager()
        
        try:
            # Create shards
            logger.info("Creating shards...")
            shards = []
            for i in range(3):
                shard = manager.create_shard(ShardType.EXECUTION, [f"validator_{i}_1", f"validator_{i}_2"])
                shards.append(shard)
                logger.info(f"  Created shard {shard.shard_id} with type {shard.shard_type.value}")
            
            # Add data
            logger.info(f"\nAdding {num_operations} data items...")
            start_time = time.time()
            
            for i in range(num_operations):
                key = f"basic_key_{i}"
                success = manager.add_data_to_shard(key, f"data_{i}")
                if not success:
                    logger.info(f"  Failed to add data for key {key}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Show results
            logger.info(f"\nBasic operations completed in {duration:.2f} seconds")
            logger.info(f"Throughput: {num_operations / duration:.2f} ops/sec")
            
            # Show metrics
            metrics = manager.get_performance_metrics()
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Total operations: {metrics['total_operations']}")
            logger.info(f"  Successful operations: {metrics['successful_operations']}")
            logger.info(f"  Failed operations: {metrics['failed_operations']}")
            logger.info(f"  Average operation time: {metrics['average_operation_time']:.4f}s")
            
            # Show load distribution
            distribution = manager.get_shard_load_distribution()
            logger.info(f"\nLoad Distribution:")
            for shard_id, load in distribution.items():
                logger.info(f"  Shard {shard_id}: {load:.3f}")
            
            self.results['basic'] = {
                'duration': duration,
                'throughput': num_operations / duration,
                'metrics': metrics,
                'distribution': distribution
            }
            
        finally:
            self.teardown_manager()
    
    def load_balancing_scenario(self, num_operations: int = 1000):
        """Demonstrate load balancing strategies."""
        logger.info("\n=== LOAD BALANCING SCENARIO ===")
        
        strategies = [
            ("Consistent Hash", ConsistentHashBalancer(virtual_nodes=100)),
            ("Least Loaded", LeastLoadedBalancer()),
            ("Adaptive", AdaptiveBalancer())
        ]
        
        for strategy_name, balancer in strategies:
            logger.info(f"\n--- Testing {strategy_name} Strategy ---")
            
            manager = self.setup_manager(balancer)
            
            try:
                # Create shards
                for i in range(5):
                    manager.create_shard(ShardType.EXECUTION)
                
                # Create load imbalance with hot keys
                logger.info("Creating load imbalance with hot keys...")
                for i in range(num_operations):
                    # Create hot keys (only 10 different keys)
                    hot_key = f"hot_key_{i % 10}"
                    manager.add_data_to_shard(hot_key, f"data_{i}")
                
                # Show load distribution
                distribution = manager.get_shard_load_distribution()
                logger.info(f"Load distribution with {strategy_name}:")
                for shard_id, load in distribution.items():
                    logger.info(f"  Shard {shard_id}: {load:.3f}")
                
                # Calculate load balance metrics
                load_values = list(distribution.values())
                if load_values:
                    max_load = max(load_values)
                    min_load = min(load_values)
                    imbalance = (max_load - min_load) / max_load if max_load > 0 else 0
                    logger.info(f"  Load imbalance: {imbalance:.3f}")
                
                # Test key consistency
                logger.info("Testing key consistency...")
                test_keys = ["consistent_key_1", "consistent_key_2", "consistent_key_3"]
                selected_shards = []
                
                for key in test_keys:
                    shard_id = manager.select_shard_for_key(key)
                    selected_shards.append(shard_id)
                    logger.info(f"  Key '{key}' -> Shard {shard_id}")
                
                # Verify consistency
                for _ in range(5):
                    for i, key in enumerate(test_keys):
                        shard_id = manager.select_shard_for_key(key)
                        assert shard_id == selected_shards[i], f"Key {key} inconsistent"
                
                logger.info("  ✓ Key consistency verified")
                
            finally:
                self.teardown_manager()
    
    def resharding_scenario(self, num_operations: int = 500):
        """Demonstrate dynamic resharding operations."""
        logger.info("\n=== RESHARDING SCENARIO ===")
        
        manager = self.setup_manager()
        
        try:
            # Create initial shards
            logger.info("Creating initial shards...")
            shard1 = manager.create_shard(ShardType.EXECUTION, ["validator_1", "validator_2"])
            shard2 = manager.create_shard(ShardType.CONSENSUS, ["validator_3", "validator_4"])
            
            # Add data
            logger.info(f"Adding {num_operations} data items...")
            for i in range(num_operations):
                key = f"reshard_key_{i}"
                manager.add_data_to_shard(key, f"data_{i}")
            
            # Show initial state
            initial_distribution = manager.get_shard_load_distribution()
            logger.info(f"\nInitial load distribution:")
            for shard_id, load in initial_distribution.items():
                logger.info(f"  Shard {shard_id}: {load:.3f}")
            
            # Test different resharding strategies
            strategies = [
                ("Horizontal Split", ReshardingStrategy.HORIZONTAL_SPLIT),
                ("Vertical Split", ReshardingStrategy.VERTICAL_SPLIT),
                ("Merge", ReshardingStrategy.MERGE),
                ("Rebalance", ReshardingStrategy.REBALANCE)
            ]
            
            for strategy_name, strategy in strategies:
                logger.info(f"\n--- Testing {strategy_name} ---")
                
                # Create new shard for resharding
                new_shard = manager.create_shard(ShardType.STORAGE, ["validator_5", "validator_6"])
                
                # Create resharding plan
                source_shards = [shard1.shard_id, shard2.shard_id]
                target_shards = [new_shard.shard_id]
                data_migration_map = {
                    f"reshard_key_{i}": new_shard.shard_id
                    for i in range(100)  # Migrate 100 keys
                }
                
                logger.info(f"Creating resharding plan...")
                plan = manager.resharding_manager.create_resharding_plan(
                    strategy, source_shards, target_shards, data_migration_map
                )
                
                logger.info(f"  Plan ID: {plan.plan_id}")
                logger.info(f"  Strategy: {plan.strategy.value}")
                logger.info(f"  Estimated duration: {plan.estimated_duration:.2f}s")
                logger.info(f"  Estimated impact: {plan.estimated_impact:.3f}")
                logger.info(f"  Safety checks: {len(plan.safety_checks)}")
                
                # Execute resharding
                logger.info("Executing resharding...")
                start_time = time.time()
                
                plan_id = manager.trigger_resharding(strategy, source_shards, target_shards, data_migration_map)
                
                # Wait for completion
                time.sleep(1)
                
                end_time = time.time()
                duration = end_time - start_time
                
                logger.info(f"  Resharding completed in {duration:.2f}s")
                
                # Verify system is still functional
                success = manager.add_data_to_shard("post_reshard_key", "post_reshard_data")
                logger.info(f"  System functional after resharding: {success}")
                
                # Show new load distribution
                new_distribution = manager.get_shard_load_distribution()
                logger.info(f"  New load distribution:")
                for shard_id, load in new_distribution.items():
                    logger.info(f"    Shard {shard_id}: {load:.3f}")
            
        finally:
            self.teardown_manager()
    
    def fault_tolerance_scenario(self, num_operations: int = 500):
        """Demonstrate fault tolerance and recovery."""
        logger.info("\n=== FAULT TOLERANCE SCENARIO ===")
        
        manager = self.setup_manager()
        
        try:
            # Create shards
            logger.info("Creating shards...")
            shards = []
            for i in range(5):
                shard = manager.create_shard(ShardType.EXECUTION, [f"validator_{i}_1", f"validator_{i}_2"])
                shards.append(shard)
            
            # Add initial data
            logger.info(f"Adding {num_operations} initial data items...")
            for i in range(num_operations):
                key = f"fault_key_{i}"
                manager.add_data_to_shard(key, f"data_{i}")
            
            # Show initial state
            initial_healthy = manager.health_monitor.get_healthy_shards()
            logger.info(f"Initial healthy shards: {len(initial_healthy)}")
            
            # Simulate shard failures
            logger.info("\nSimulating shard failures...")
            failed_shards = shards[:2]  # Fail first 2 shards
            
            for shard in failed_shards:
                logger.info(f"  Failing shard {shard.shard_id}...")
                
                # Mark shard as failed
                load_metrics = ShardLoadMetrics(
                    shard_id=shard.shard_id,
                    cpu_usage=100.0,
                    memory_usage=100.0,
                    error_rate=1.0
                )
                manager.health_monitor.update_shard_health(shard.shard_id, load_metrics, is_healthy=False)
                
                # Update shard status
                manager.shards[shard.shard_id].status = ShardStatus.ERROR
            
            # Check system response
            logger.info("\nChecking system response to failures...")
            healthy_shards = manager.health_monitor.get_healthy_shards()
            failed_shard_list = manager.health_monitor.detect_failed_shards()
            
            logger.info(f"  Healthy shards: {len(healthy_shards)}")
            logger.info(f"  Failed shards: {len(failed_shard_list)}")
            
            # Test system functionality with failures
            logger.info("\nTesting system functionality with failures...")
            recovery_operations = 200
            successful_ops = 0
            failed_ops = 0
            
            start_time = time.time()
            
            for i in range(recovery_operations):
                key = f"recovery_key_{i}"
                try:
                    success = manager.add_data_to_shard(key, f"data_{i}")
                    if success:
                        successful_ops += 1
                    else:
                        failed_ops += 1
                except Exception as e:
                    failed_ops += 1
                    logger.info(f"    Error with key {key}: {e}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"  Recovery operations completed in {duration:.2f}s")
            logger.info(f"  Successful operations: {successful_ops}")
            logger.info(f"  Failed operations: {failed_ops}")
            logger.info(f"  Success rate: {successful_ops / recovery_operations:.2%}")
            
            # Test automatic recovery
            logger.info("\nTesting automatic recovery...")
            for shard in failed_shards:
                # Simulate recovery
                load_metrics = ShardLoadMetrics(
                    shard_id=shard.shard_id,
                    cpu_usage=30.0,
                    memory_usage=25.0,
                    error_rate=0.0
                )
                manager.health_monitor.update_shard_health(shard.shard_id, load_metrics, is_healthy=True)
                manager.shards[shard.shard_id].status = ShardStatus.ACTIVE
            
            # Check recovery
            recovered_healthy = manager.health_monitor.get_healthy_shards()
            logger.info(f"  Shards after recovery: {len(recovered_healthy)}")
            
            # Test system functionality after recovery
            success = manager.add_data_to_shard("post_recovery_key", "post_recovery_data")
            logger.info(f"  System functional after recovery: {success}")
            
        finally:
            self.teardown_manager()
    
    def performance_scenario(self, num_operations: int = 2000):
        """Demonstrate performance characteristics."""
        logger.info("\n=== PERFORMANCE SCENARIO ===")
        
        manager = self.setup_manager()
        
        try:
            # Create shards
            logger.info("Creating shards...")
            for i in range(5):
                manager.create_shard(ShardType.EXECUTION)
            
            # Test throughput
            logger.info(f"\nTesting throughput with {num_operations} operations...")
            start_time = time.time()
            
            for i in range(num_operations):
                key = f"perf_key_{i}"
                manager.add_data_to_shard(key, f"data_{i}")
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = num_operations / duration
            
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"  Throughput: {throughput:.2f} ops/sec")
            
            # Test latency
            logger.info("\nTesting operation latency...")
            latencies = []
            
            for i in range(100):
                key = f"latency_key_{i}"
                start = time.time()
                manager.add_data_to_shard(key, f"data_{i}")
                end = time.time()
                latency = (end - start) * 1000  # Convert to milliseconds
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
            p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
            
            logger.info(f"  Average latency: {avg_latency:.2f}ms")
            logger.info(f"  95th percentile: {p95_latency:.2f}ms")
            logger.info(f"  99th percentile: {p99_latency:.2f}ms")
            
            # Test concurrent operations
            logger.info("\nTesting concurrent operations...")
            num_threads = 10
            operations_per_thread = 100
            
            def perform_operations(thread_id):
                results = []
                for i in range(operations_per_thread):
                    key = f"concurrent_{thread_id}_{i}"
                    success = manager.add_data_to_shard(key, f"data_{i}")
                    results.append(success)
                return results
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(perform_operations, i)
                    for i in range(num_threads)
                ]
                results = [future.result() for future in as_completed(futures)]
            
            end_time = time.time()
            duration = end_time - start_time
            total_operations = num_threads * operations_per_thread
            concurrent_throughput = total_operations / duration
            
            successful_ops = sum(sum(thread_results) for thread_results in results)
            
            logger.info(f"  Threads: {num_threads}")
            logger.info(f"  Operations per thread: {operations_per_thread}")
            logger.info(f"  Total operations: {total_operations}")
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"  Concurrent throughput: {concurrent_throughput:.2f} ops/sec")
            logger.info(f"  Successful operations: {successful_ops}")
            logger.info(f"  Success rate: {successful_ops / total_operations:.2%}")
            
            # Show final metrics
            metrics = manager.get_performance_metrics()
            logger.info(f"\nFinal Performance Metrics:")
            logger.info(f"  Total operations: {metrics['total_operations']}")
            logger.info(f"  Successful operations: {metrics['successful_operations']}")
            logger.info(f"  Failed operations: {metrics['failed_operations']}")
            logger.info(f"  Average operation time: {metrics['average_operation_time']:.4f}s")
            logger.info(f"  Active operations: {metrics['active_operations']}")
            logger.info(f"  Total shards: {metrics['total_shards']}")
            logger.info(f"  Healthy shards: {metrics['healthy_shards']}")
            logger.info(f"  Failed shards: {metrics['failed_shards']}")
            
        finally:
            self.teardown_manager()
    
    def stress_scenario(self, num_operations: int = 5000):
        """Demonstrate stress testing capabilities."""
        logger.info("\n=== STRESS TESTING SCENARIO ===")
        
        manager = self.setup_manager()
        
        try:
            # Create shards
            logger.info("Creating shards...")
            for i in range(8):
                manager.create_shard(ShardType.EXECUTION)
            
            # High load stress test
            logger.info(f"\nHigh load stress test with {num_operations} operations...")
            start_time = time.time()
            
            successful_ops = 0
            failed_ops = 0
            
            for i in range(num_operations):
                key = f"stress_key_{i}"
                try:
                    success = manager.add_data_to_shard(key, f"data_{i}")
                    if success:
                        successful_ops += 1
                    else:
                        failed_ops += 1
                except Exception as e:
                    failed_ops += 1
                    if failed_ops % 100 == 0:
                        logger.info(f"    Error at operation {i}: {e}")
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = num_operations / duration
            success_rate = successful_ops / num_operations
            
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"  Throughput: {throughput:.2f} ops/sec")
            logger.info(f"  Successful operations: {successful_ops}")
            logger.info(f"  Failed operations: {failed_ops}")
            logger.info(f"  Success rate: {success_rate:.2%}")
            
            # Memory pressure test
            logger.info("\nMemory pressure test...")
            memory_start = manager.get_performance_metrics()['memory_usage_mb']
            
            # Perform many operations to create memory pressure
            for i in range(10000):
                key = f"memory_key_{i}"
                manager.add_data_to_shard(key, f"data_{i}")
                
                # Clean up periodically
                if i % 1000 == 0:
                    cleaned = manager.cleanup_old_operations(max_age_seconds=0)
                    if cleaned > 0:
                        logger.info(f"    Cleaned up {cleaned} old operations")
            
            memory_end = manager.get_performance_metrics()['memory_usage_mb']
            memory_delta = memory_end - memory_start
            
            logger.info(f"  Memory usage increase: {memory_delta:.2f} MB")
            logger.info(f"  Memory per operation: {memory_delta / 10000:.4f} MB")
            
            # System stability check
            logger.info("\nSystem stability check...")
            final_metrics = manager.get_performance_metrics()
            
            logger.info(f"  Total operations: {final_metrics['total_operations']}")
            logger.info(f"  System still functional: {final_metrics['total_shards'] > 0}")
            logger.info(f"  Healthy shards: {final_metrics['healthy_shards']}")
            logger.info(f"  Failed shards: {final_metrics['failed_shards']}")
            
            # Test system recovery
            logger.info("\nTesting system recovery...")
            recovery_success = manager.add_data_to_shard("recovery_test_key", "recovery_test_data")
            logger.info(f"  System recovery successful: {recovery_success}")
            
        finally:
            self.teardown_manager()
    
    def run_all_scenarios(self, num_operations: int = 1000):
        """Run all scenarios."""
        logger.info("=== ENHANCED SHARDING SYSTEM DEMO ===")
        logger.info(f"Running all scenarios with {num_operations} operations each...")
        
        scenarios = [
            ("Basic Sharding", self.basic_scenario),
            ("Load Balancing", self.load_balancing_scenario),
            ("Resharding", self.resharding_scenario),
            ("Fault Tolerance", self.fault_tolerance_scenario),
            ("Performance", self.performance_scenario),
            ("Stress Testing", self.stress_scenario),
        ]
        
        for scenario_name, scenario_func in scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {scenario_name} Scenario")
            logger.info(f"{'='*60}")
            
            try:
                scenario_func(num_operations)
                logger.info(f"✓ {scenario_name} completed successfully")
            except Exception as e:
                logger.info(f"✗ {scenario_name} failed: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info("DEMO COMPLETED")
        logger.info(f"{'='*60}")
        
        # Summary
        if self.results:
            logger.info("\nSummary of Results:")
            for scenario, result in self.results.items():
                if 'throughput' in result:
                    logger.info(f"  {scenario}: {result['throughput']:.2f} ops/sec")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Sharding System Demo")
    parser.add_argument(
        "--scenario",
        choices=["basic", "load_balancing", "resharding", "fault_tolerance", "performance", "stress", "all"],
        default="all",
        help="Scenario to run"
    )
    parser.add_argument(
        "--operations",
        type=int,
        default=1000,
        help="Number of operations to perform"
    )
    
    args = parser.parse_args()
    
    demo = EnhancedShardingDemo()
    
    try:
        if args.scenario == "basic":
            demo.basic_scenario(args.operations)
        elif args.scenario == "load_balancing":
            demo.load_balancing_scenario(args.operations)
        elif args.scenario == "resharding":
            demo.resharding_scenario(args.operations)
        elif args.scenario == "fault_tolerance":
            demo.fault_tolerance_scenario(args.operations)
        elif args.scenario == "performance":
            demo.performance_scenario(args.operations)
        elif args.scenario == "stress":
            demo.stress_scenario(args.operations)
        elif args.scenario == "all":
            demo.run_all_scenarios(args.operations)
        
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.info(f"\nDemo failed with error: {e}")
    finally:
        demo.teardown_manager()


if __name__ == "__main__":
    main()
