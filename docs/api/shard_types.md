# Shard Types API

**Module:** `dubchain.sharding.shard_types`

## Classes

### CrossShardTransaction

Cross-shard transaction data.

#### Methods

##### `calculate_hash(self)`

Calculate transaction hash.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `CrossShardTransaction`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ShardConfig

Configuration for shard management.

#### Methods

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `ShardConfig`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ShardId

Shard identifier.

**Inherits from:** IntEnum

### ShardMetrics

Metrics for shard performance.

#### Methods

##### `get_success_rate(self)`

Get success rate.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

##### `get_validator_utilization(self)`

Get validator utilization rate.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

#### Properties

##### `failure_rate`

Calculate failure rate.

##### `success_rate`

Calculate success rate.

### ShardState

State of a shard.

#### Methods

##### `add_cross_shard_transaction(self, transaction)`

Add cross-shard transaction to queue.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.sharding.shard_types.CrossShardTransaction'> (required)

**Returns:** `None`

##### `clear_cross_shard_transactions(self, target_shard)`

Clear cross-shard transactions for target shard.

**Parameters:**

- `self`: Any (required)
- `target_shard`: <enum 'ShardId'> (required)

**Returns:** `None`

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `ShardState`

##### `get_cross_shard_transactions(self, target_shard)`

Get cross-shard transactions for target shard.

**Parameters:**

- `self`: Any (required)
- `target_shard`: <enum 'ShardId'> (required)

**Returns:** `typing.List[dubchain.sharding.shard_types.CrossShardTransaction]`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `update_metrics(self, success, block_time, gas_used)`

Update shard metrics.

**Parameters:**

- `self`: Any (required)
- `success`: <class 'bool'> (required)
- `block_time`: <class 'float'> (required)
- `gas_used`: <class 'int'> (required)

**Returns:** `None`

### ShardStatus

Shard status states.

**Inherits from:** Enum

### ShardType

Types of shards.

**Inherits from:** Enum

## Usage Examples

```python
from dubchain.sharding.shard_types import *

# Create instance of CrossShardTransaction
crossshardtransaction = CrossShardTransaction()

# Call method
result = crossshardtransaction.calculate_hash()
```
