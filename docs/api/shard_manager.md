# Shard Manager API

**Module:** `dubchain.sharding.shard_manager`

## Classes

### ShardAllocator

Allocates validators to shards.

#### Methods

##### `allocate_validators(self, validators, shard_count)`

Allocate validators to shards.

**Parameters:**

- `self`: Any (required)
- `validators`: typing.List[dubchain.consensus.validator.ValidatorInfo] (required)
- `shard_count`: <class 'int'> (required)

**Returns:** `typing.Dict[dubchain.sharding.shard_types.ShardId, typing.List[str]]`

### ShardBalancer

Balances validators across shards.

#### Methods

##### `rebalance_shards(self, shard_states, all_validators)`

Rebalance validators across shards.

**Parameters:**

- `self`: Any (required)
- `shard_states`: typing.Dict[dubchain.sharding.shard_types.ShardId, dubchain.sharding.shard_types.ShardState] (required)
- `all_validators`: typing.List[dubchain.consensus.validator.ValidatorInfo] (required)

**Returns:** `typing.Dict[dubchain.sharding.shard_types.ShardId, typing.List[str]]`

##### `should_rebalance(self, shard_states)`

Check if shards need rebalancing.

**Parameters:**

- `self`: Any (required)
- `shard_states`: typing.Dict[dubchain.sharding.shard_types.ShardId, dubchain.sharding.shard_types.ShardState] (required)

**Returns:** `<class 'bool'>`

### ShardCoordinator

Coordinates operations across shards.

#### Methods

##### `coordinate_cross_shard_transaction(self, transaction, source_shard, target_shard)`

Coordinate cross-shard transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.sharding.shard_types.CrossShardTransaction'> (required)
- `source_shard`: <class 'dubchain.sharding.shard_types.ShardState'> (required)
- `target_shard`: <class 'dubchain.sharding.shard_types.ShardState'> (required)

**Returns:** `<class 'bool'>`

##### `should_sync_state(self)`

Check if state sync is needed.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `sync_shard_states(self, shard_states)`

Synchronize state across shards.

**Parameters:**

- `self`: Any (required)
- `shard_states`: typing.Dict[dubchain.sharding.shard_types.ShardId, dubchain.sharding.shard_types.ShardState] (required)

**Returns:** `None`

### ShardManager

Manages all shards in the network.

#### Methods

##### `add_validator(self, shard_id, validator_id)`

Add validator to shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `add_validator_to_shard(self, shard_id, validator_id)`

Add validator to a specific shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `allocate_validators_to_shards(self, validators)`

Allocate validators to shards.

**Parameters:**

- `self`: Any (required)
- `validators`: typing.List[dubchain.consensus.validator.ValidatorInfo] (required)

**Returns:** `None`

##### `assign_validators_to_shard(self, shard_id, validators)`

Assign validators to a shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)
- `validators`: typing.List[dubchain.consensus.validator.ValidatorInfo] (required)

**Returns:** `<class 'bool'>`

##### `create_shard(self, shard_type, validators)`

Create a new shard.

**Parameters:**

- `self`: Any (required)
- `shard_type`: <enum 'ShardType'> = ShardType.EXECUTION
- `validators`: typing.Optional[typing.List[str]] = None

**Returns:** `<class 'dubchain.sharding.shard_types.ShardState'>`

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `ShardManager`

##### `get_active_shards(self)`

Get active shards.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.sharding.shard_types.ShardState]`

##### `get_all_shards(self)`

Get all shards.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.sharding.shard_types.ShardState]`

##### `get_global_metrics(self)`

Get global shard metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_shard(self, shard_id)`

Get shard by ID.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)

**Returns:** `typing.Optional[dubchain.sharding.shard_types.ShardState]`

##### `get_shard_by_validator(self, validator_id)`

Get shard ID for a validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.sharding.shard_types.ShardId]`

##### `get_shard_info(self, shard_id)`

Get shard information.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)

**Returns:** `typing.Optional[dubchain.sharding.shard_types.ShardState]`

##### `get_shard_metrics(self, shard_id)`

Get metrics for a specific shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)

**Returns:** `typing.Optional[dubchain.sharding.shard_types.ShardMetrics]`

##### `get_shard_statistics(self)`

Get shard statistics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_shard_validators(self, shard_id)`

Get validators for a shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)

**Returns:** `typing.List[str]`

##### `list_shards(self)`

List all shards.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.sharding.shard_types.ShardState]`

##### `process_cross_shard_transaction(self, transaction)`

Process cross-shard transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.sharding.shard_types.CrossShardTransaction'> (required)

**Returns:** `<class 'bool'>`

##### `rebalance_shards(self)`

Rebalance validators across shards.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `remove_shard(self, shard_id)`

Remove a shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)

**Returns:** `<class 'bool'>`

##### `remove_validator(self, shard_id, validator_id)`

Remove validator from shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `remove_validator_from_shard(self, shard_id, validator_id)`

Remove validator from a specific shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `start(self)`

Start shard manager.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `stop(self)`

Stop shard manager.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `sync_shard_states(self)`

Synchronize state across all shards.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `update_shard_status(self, shard_id, status)`

Update shard status.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)
- `status`: <enum 'ShardStatus'> (required)

**Returns:** `<class 'bool'>`

##### `validate_shard(self, shard_id)`

Validate shard.

**Parameters:**

- `self`: Any (required)
- `shard_id`: <enum 'ShardId'> (required)

**Returns:** `<class 'bool'>`

#### Properties

##### `shard_states`

Get shard states.

## Usage Examples

```python
from dubchain.sharding.shard_manager import *

# Create instance of ShardAllocator
shardallocator = ShardAllocator()

# Call method
result = shardallocator.allocate_validators(1)
```
