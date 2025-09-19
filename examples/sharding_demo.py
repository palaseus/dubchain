#!/usr/bin/env python3
"""
Blockchain Sharding Demo for DubChain

This demo showcases the sophisticated sharding system including:
- Shard creation and management
- Validator allocation and rebalancing
- Cross-shard transactions
- Shard coordination and synchronization

Run this demo to see how DubChain scales horizontally through sharding.
"""

import asyncio
import time
import json
from typing import Dict, Any, List

# Import DubChain sharding components
from dubchain.sharding import (
    ShardId,
    ShardStatus,
    ShardType,
    ShardConfig,
    ShardManager,
    ShardState,
    CrossShardTransaction,
    CrossShardMessaging,
    MessageType,
    CrossShardMessage
)
from dubchain.consensus.validator import Validator, ValidatorInfo
from dubchain.crypto.signatures import PrivateKey


class ShardingDemo:
    """Demonstrates blockchain sharding capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.shard_manager = None
        self.cross_shard_messaging = None
        self.validators = []
        self.demo_transactions = []
    
    def create_validators(self, count: int = 20) -> None:
        """Create validators for the demo."""
        print(f"🔧 Creating {count} validators...")
        
        for i in range(count):
            private_key = PrivateKey.generate()
            validator = Validator(
                validator_id=f"validator_{i}",
                private_key=private_key,
                commission_rate=0.1
            )
            
            # Create validator info with varying stakes
            validator_info = ValidatorInfo(
                validator_id=validator.validator_id,
                public_key=validator.public_key,
                total_stake=1000000 + (i * 100000),  # Varying stakes
                self_stake=1000000 + (i * 100000)
            )
            
            self.validators.append(validator_info)
            print(f"  ✅ Created {validator_info.validator_id} with {validator_info.total_stake} stake")
    
    def setup_sharding_system(self) -> None:
        """Setup the sharding system."""
        print("\n🚀 Setting up sharding system...")
        
        # Create shard configuration
        config = ShardConfig(
            max_shards=8,
            min_validators_per_shard=4,
            max_validators_per_shard=8,
            shard_epoch_length=32,
            cross_shard_delay=2,
            state_sync_interval=16,
            enable_dynamic_sharding=True
        )
        
        # Create shard manager
        self.shard_manager = ShardManager(config)
        
        # Create cross-shard messaging system
        self.cross_shard_messaging = CrossShardMessaging()
        
        print("  ✅ Sharding system initialized")
    
    def create_shards(self) -> None:
        """Create shards for the demo."""
        print("\n🏗️  Creating shards...")
        
        # Create beacon chain
        beacon_shard = self.shard_manager.create_shard(ShardId.BEACON_CHAIN, ShardType.BEACON)
        print(f"  ✅ Created beacon chain (Shard {ShardId.BEACON_CHAIN.value})")
        
        # Create execution shards
        for i in range(1, 5):  # Create 4 execution shards
            shard_id = ShardId(i)
            shard = self.shard_manager.create_shard(shard_id, ShardType.EXECUTION)
            print(f"  ✅ Created execution shard {i}")
        
        print(f"  📊 Total shards created: {len(self.shard_manager.shards)}")
    
    def allocate_validators(self) -> None:
        """Allocate validators to shards."""
        print("\n👥 Allocating validators to shards...")
        
        # Allocate validators to shards
        self.shard_manager.allocate_validators_to_shards(self.validators)
        
        # Show allocation
        for shard_id, shard_state in self.shard_manager.shards.items():
            validator_count = len(shard_state.validator_set)
            print(f"  📝 Shard {shard_id.value}: {validator_count} validators")
            if validator_count > 0:
                print(f"    Validators: {', '.join(shard_state.validator_set[:3])}{'...' if validator_count > 3 else ''}")
    
    def demonstrate_shard_operations(self) -> None:
        """Demonstrate shard operations."""
        print("\n⚙️  SHARD OPERATIONS DEMONSTRATION")
        print("=" * 50)
        
        # Show shard information
        print("📊 Shard Information:")
        for shard_id, shard_state in self.shard_manager.shards.items():
            print(f"  Shard {shard_id.value}:")
            print(f"    Status: {shard_state.status.value}")
            print(f"    Type: {shard_state.shard_type.value}")
            print(f"    Validators: {len(shard_state.validator_set)}")
            print(f"    Current Epoch: {shard_state.current_epoch}")
        
        # Demonstrate validator movement
        print("\n🔄 Validator Movement:")
        if len(self.shard_manager.shards) >= 2:
            shard_ids = list(self.shard_manager.shards.keys())
            source_shard = shard_ids[1]  # Use first execution shard
            target_shard = shard_ids[2]  # Use second execution shard
            
            if (len(self.shard_manager.shards[source_shard].validator_set) > 0 and
                len(self.shard_manager.shards[target_shard].validator_set) > 0):
                
                # Move a validator
                validator_to_move = self.shard_manager.shards[source_shard].validator_set[0]
                
                print(f"  Moving {validator_to_move} from Shard {source_shard.value} to Shard {target_shard.value}")
                
                # Remove from source
                self.shard_manager.remove_validator_from_shard(source_shard, validator_to_move)
                print(f"    ✅ Removed from Shard {source_shard.value}")
                
                # Add to target
                self.shard_manager.add_validator_to_shard(target_shard, validator_to_move)
                print(f"    ✅ Added to Shard {target_shard.value}")
                
                print(f"  📊 Shard {source_shard.value} now has {len(self.shard_manager.shards[source_shard].validator_set)} validators")
                print(f"  📊 Shard {target_shard.value} now has {len(self.shard_manager.shards[target_shard].validator_set)} validators")
    
    def demonstrate_cross_shard_transactions(self) -> None:
        """Demonstrate cross-shard transactions."""
        print("\n🌉 CROSS-SHARD TRANSACTIONS DEMONSTRATION")
        print("=" * 50)
        
        # Create cross-shard transactions
        print("📝 Creating cross-shard transactions...")
        
        execution_shards = [shard_id for shard_id in self.shard_manager.shards.keys() 
                          if shard_id != ShardId.BEACON_CHAIN]
        
        if len(execution_shards) >= 2:
            source_shard = execution_shards[0]
            target_shard = execution_shards[1]
            
            # Create sample transactions
            transactions = [
                CrossShardTransaction(
                    transaction_id=f"tx_{i}",
                    source_shard=source_shard,
                    target_shard=target_shard,
                    sender=f"user_{i}",
                    receiver=f"user_{i+10}",
                    amount=1000 + (i * 100),
                    gas_limit=21000,
                    gas_price=20,
                    data=b"cross_shard_transfer"
                )
                for i in range(3)
            ]
            
            print(f"  Created {len(transactions)} transactions from Shard {source_shard.value} to Shard {target_shard.value}")
            
            # Process transactions
            print("\n🔄 Processing cross-shard transactions...")
            for i, transaction in enumerate(transactions):
                print(f"  Processing transaction {i + 1}...")
                
                # Process through shard manager
                success = self.shard_manager.process_cross_shard_transaction(transaction)
                if success:
                    print(f"    ✅ Transaction {transaction.transaction_id} queued")
                else:
                    print(f"    ❌ Transaction {transaction.transaction_id} failed")
                
                # Process through cross-shard messaging
                source_state = self.shard_manager.shards[source_shard]
                target_state = self.shard_manager.shards[target_shard]
                
                messaging_success = self.cross_shard_messaging.process_cross_shard_transaction(
                    transaction, source_state, target_state
                )
                if messaging_success:
                    print(f"    ✅ Message for transaction {transaction.transaction_id} sent")
                else:
                    print(f"    ❌ Message for transaction {transaction.transaction_id} failed")
    
    def demonstrate_shard_coordination(self) -> None:
        """Demonstrate shard coordination."""
        print("\n🎯 SHARD COORDINATION DEMONSTRATION")
        print("=" * 50)
        
        # Show coordination events
        print("📊 Coordination Events:")
        events = self.shard_manager.coordinator.coordination_events
        if events:
            for i, event in enumerate(events[-5:]):  # Show last 5 events
                print(f"  {i + 1}. {event['type']} at {time.ctime(event['timestamp'])}")
        else:
            print("  No coordination events yet")
        
        # Demonstrate state synchronization
        print("\n🔄 State Synchronization:")
        print("  Triggering state sync...")
        self.shard_manager.sync_shard_states()
        print("  ✅ State synchronization completed")
        
        # Show updated coordination events
        print("\n📊 Updated Coordination Events:")
        events = self.shard_manager.coordinator.coordination_events
        if events:
            for i, event in enumerate(events[-3:]):  # Show last 3 events
                print(f"  {i + 1}. {event['type']} at {time.ctime(event['timestamp'])}")
    
    def demonstrate_shard_rebalancing(self) -> None:
        """Demonstrate shard rebalancing."""
        print("\n⚖️  SHARD REBALANCING DEMONSTRATION")
        print("=" * 50)
        
        # Show current validator distribution
        print("📊 Current Validator Distribution:")
        for shard_id, shard_state in self.shard_manager.shards.items():
            validator_count = len(shard_state.validator_set)
            print(f"  Shard {shard_id.value}: {validator_count} validators")
        
        # Check if rebalancing is needed
        print("\n🔍 Checking rebalancing need...")
        should_rebalance = self.shard_manager.balancer.should_rebalance(self.shard_manager.shards)
        print(f"  Rebalancing needed: {should_rebalance}")
        
        if should_rebalance:
            print("  🔄 Performing rebalancing...")
            rebalance_success = self.shard_manager.rebalance_shards()
            if rebalance_success:
                print("  ✅ Rebalancing completed")
                
                # Show new distribution
                print("\n📊 New Validator Distribution:")
                for shard_id, shard_state in self.shard_manager.shards.items():
                    validator_count = len(shard_state.validator_set)
                    print(f"  Shard {shard_id.value}: {validator_count} validators")
            else:
                print("  ❌ Rebalancing failed")
        else:
            print("  ✅ Shards are well balanced")
    
    def show_shard_metrics(self) -> None:
        """Show comprehensive shard metrics."""
        print("\n📊 SHARD METRICS")
        print("=" * 50)
        
        # Global metrics
        global_metrics = self.shard_manager.get_global_metrics()
        print("🌐 Global Metrics:")
        print(f"  Total shards: {global_metrics['total_shards']}")
        print(f"  Active shards: {global_metrics['active_shards']}")
        print(f"  Total validators: {global_metrics['total_validators']}")
        print(f"  Total cross-shard transactions: {global_metrics['total_cross_shard_transactions']}")
        print(f"  Current epoch: {global_metrics['current_epoch']}")
        
        # Individual shard metrics
        print("\n📈 Individual Shard Metrics:")
        for shard_id, shard_state in self.shard_manager.shards.items():
            metrics = shard_state.metrics
            print(f"  Shard {shard_id.value}:")
            print(f"    Total blocks: {metrics.total_blocks}")
            print(f"    Success rate: {metrics.success_rate:.2%}")
            print(f"    Validators: {metrics.validator_count}")
            print(f"    Cross-shard transactions: {metrics.cross_shard_transactions}")
        
        # Cross-shard messaging metrics
        if self.cross_shard_messaging:
            print("\n🌉 Cross-Shard Messaging Metrics:")
            messaging_metrics = self.cross_shard_messaging.get_system_metrics()
            print(f"  Active connections: {messaging_metrics['active_connections']}")
            print(f"  Registered handlers: {messaging_metrics['registered_handlers']}")
            print(f"  Queued messages: {messaging_metrics['relay_metrics']['queued_messages']}")
            print(f"  Processed messages: {messaging_metrics['relay_metrics']['processed_messages']}")
    
    def run_demo(self) -> None:
        """Run the complete sharding demo."""
        print("🚀 DUBCHAIN BLOCKCHAIN SHARDING DEMO")
        print("=" * 60)
        print("This demo showcases horizontal scaling through sharding")
        print("including validator allocation, cross-shard transactions,")
        print("and dynamic shard management.")
        print("=" * 60)
        
        # Setup
        self.create_validators(20)
        self.setup_sharding_system()
        self.create_shards()
        self.allocate_validators()
        
        # Demonstrate features
        self.demonstrate_shard_operations()
        self.demonstrate_cross_shard_transactions()
        self.demonstrate_shard_coordination()
        self.demonstrate_shard_rebalancing()
        self.show_shard_metrics()
        
        print("\n🎉 DEMO COMPLETED!")
        print("=" * 60)
        print("DubChain's sharding system provides:")
        print("✅ Horizontal scaling through sharding")
        print("✅ Dynamic validator allocation")
        print("✅ Cross-shard transaction processing")
        print("✅ Automatic shard rebalancing")
        print("✅ State synchronization")
        print("✅ Enterprise-grade scalability")
        print("=" * 60)


async def main():
    """Main demo function."""
    demo = ShardingDemo()
    demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
