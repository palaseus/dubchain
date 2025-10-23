# Pbft API

**Module:** `dubchain.consensus.pbft`

## Classes

### PracticalByzantineFaultTolerance

PBFT consensus implementation.

#### Methods

##### `add_validator(self, validator)`

Add validator to PBFT network.

**Parameters:**

- `self`: Any (required)
- `validator`: <class 'dubchain.consensus.validator.Validator'> (required)

**Returns:** `<class 'bool'>`

##### `detect_byzantine_fault(self, validator_id)`

Detect Byzantine fault in validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `PracticalByzantineFaultTolerance`

##### `get_consensus_metrics(self)`

Get consensus metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.consensus.consensus_types.ConsensusMetrics'>`

##### `get_network_status(self)`

Get overall network status.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_validator_status(self, validator_id)`

Get status of a validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `typing.Optional[typing.Dict[str, typing.Any]]`

##### `handle_view_change(self, new_view, validator_id)`

Handle view change request.

**Parameters:**

- `self`: Any (required)
- `new_view`: <class 'int'> (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `remove_validator(self, validator_id)`

Remove validator from PBFT network.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `start_consensus(self, request_data)`

Start PBFT consensus process.

**Parameters:**

- `self`: Any (required)
- `request_data`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'dubchain.consensus.consensus_types.ConsensusResult'>`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### PBFTValidator

PBFT-specific validator information.

#### Methods

##### `add_message(self, message)`

Add message to validator's log.

**Parameters:**

- `self`: Any (required)
- `message`: <class 'dubchain.consensus.consensus_types.PBFTMessage'> (required)

**Returns:** `None`

##### `is_online(self, timeout)`

Check if validator is online.

**Parameters:**

- `self`: Any (required)
- `timeout`: <class 'float'> = 30.0

**Returns:** `<class 'bool'>`

### PracticalByzantineFaultTolerance

PBFT consensus implementation.

#### Methods

##### `add_validator(self, validator)`

Add validator to PBFT network.

**Parameters:**

- `self`: Any (required)
- `validator`: <class 'dubchain.consensus.validator.Validator'> (required)

**Returns:** `<class 'bool'>`

##### `detect_byzantine_fault(self, validator_id)`

Detect Byzantine fault in validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `PracticalByzantineFaultTolerance`

##### `get_consensus_metrics(self)`

Get consensus metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.consensus.consensus_types.ConsensusMetrics'>`

##### `get_network_status(self)`

Get overall network status.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_validator_status(self, validator_id)`

Get status of a validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `typing.Optional[typing.Dict[str, typing.Any]]`

##### `handle_view_change(self, new_view, validator_id)`

Handle view change request.

**Parameters:**

- `self`: Any (required)
- `new_view`: <class 'int'> (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `remove_validator(self, validator_id)`

Remove validator from PBFT network.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `start_consensus(self, request_data)`

Start PBFT consensus process.

**Parameters:**

- `self`: Any (required)
- `request_data`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'dubchain.consensus.consensus_types.ConsensusResult'>`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

## Usage Examples

```python
from dubchain.consensus.pbft import *

# Create instance of PracticalByzantineFaultTolerance
practicalbyzantinefaulttolerance = PracticalByzantineFaultTolerance()

# Call method
result = practicalbyzantinefaulttolerance.add_validator()
```
