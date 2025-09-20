# Sharding System

This document explains the sharding implementation in DubChain for horizontal scaling.

## Overview

Sharding is a horizontal scaling technique that partitions the blockchain into smaller, manageable pieces called shards. Each shard processes a subset of transactions independently, allowing the system to scale linearly with the number of shards.

## Sharding Architecture

### Components

1. **Shard Manager**: Coordinates shard operations
2. **Cross-Shard Communication**: Inter-shard messaging
3. **Shard Consensus**: Consensus within each shard
4. **State Management**: Distributed state synchronization

### Shard Types

#### Network Sharding
- Partition network into smaller groups
- Each shard has its own validators
- Reduced communication overhead

#### Transaction Sharding
- Partition transactions by address or type
- Each shard processes specific transactions
- Improved throughput

#### State Sharding
- Partition blockchain state
- Each shard maintains subset of state
- Reduced storage requirements

## Implementation

### Shard Manager
```python
class ShardManager:
    def __init__(self, config):
        self.config = config
        self.shards = {}
        self.validators = []
        self.cross_shard_queue = []
    
    def create_shard(self, shard_id, validators):
        """Create new shard."""
        shard = Shard(
            shard_id=shard_id,
            validators=validators,
            state={},
            transactions=[]
        )
        self.shards[shard_id] = shard
        return shard
    
    def assign_transaction(self, transaction):
        """Assign transaction to appropriate shard."""
        shard_id = self.determine_shard(transaction)
        if shard_id in self.shards:
            self.shards[shard_id].add_transaction(transaction)
        return shard_id
    
    def determine_shard(self, transaction):
        """Determine which shard should process transaction."""
        # Simple hash-based sharding
        address_hash = hash(transaction.sender)
        shard_id = address_hash % len(self.shards)
        return f"shard_{shard_id}"
```

### Cross-Shard Communication
```python
class CrossShardMessaging:
    def __init__(self, shard_manager):
        self.shard_manager = shard_manager
        self.message_queue = []
        self.pending_cross_shard_txs = {}
    
    def send_cross_shard_transaction(self, source_shard, target_shard, transaction):
        """Send transaction to another shard."""
        message = CrossShardMessage(
            source_shard=source_shard,
            target_shard=target_shard,
            transaction=transaction,
            timestamp=time.time()
        )
        
        self.message_queue.append(message)
        return message.id
    
    def process_cross_shard_messages(self):
        """Process pending cross-shard messages."""
        for message in self.message_queue:
            target_shard = self.shard_manager.shards[message.target_shard]
            target_shard.add_transaction(message.transaction)
        
        self.message_queue.clear()
```

## Usage Examples

### Basic Sharding
```python
# Create shard manager
shard_config = ShardConfig(
    max_shards=8,
    min_validators_per_shard=64,
    enable_dynamic_sharding=True
)
shard_manager = ShardManager(shard_config)

# Create shards
for i in range(4):
    validators = generate_validators(64)
    shard = shard_manager.create_shard(f"shard_{i}", validators)
    print(f"Created shard {shard.shard_id}")

# Process transactions
transaction = create_transaction("alice", "bob", 100)
shard_id = shard_manager.assign_transaction(transaction)
print(f"Transaction assigned to {shard_id}")
```

## Further Reading

- [Blockchain Fundamentals](../concepts/blockchain.md)
- [Consensus Mechanisms](../concepts/consensus.md)
- [Performance Optimization](../performance/README.md)
