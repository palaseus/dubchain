# Proof Of Stake API

**Module:** `dubchain.consensus.proof_of_stake`

## Classes

### ProofOfStake

Proof of Stake consensus implementation.

#### Methods

##### `add_validator(self, address, stake_amount)`

Add a validator to the system.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)
- `stake_amount`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `finalize_block(self, block_data, proposer_id)`

Finalize a block through consensus.

**Parameters:**

- `self`: Any (required)
- `block_data`: typing.Dict[str, typing.Any] (required)
- `proposer_id`: <class 'str'> (required)

**Returns:** `<class 'dubchain.consensus.consensus_types.ConsensusResult'>`

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `ProofOfStake`

##### `get_consensus_metrics(self)`

Get consensus metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.consensus.consensus_types.ConsensusMetrics'>`

##### `get_staking_pool_info(self, validator_id)`

Get staking pool information.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.consensus.proof_of_stake.StakingPool]`

##### `get_top_validators(self, limit)`

Get top validators by stake.

**Parameters:**

- `self`: Any (required)
- `limit`: <class 'int'> = 10

**Returns:** `typing.List[dubchain.consensus.validator.ValidatorInfo]`

##### `get_total_stake(self)`

Get total stake in the network.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_validator_info(self, validator_id)`

Get information about a validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.consensus.validator.ValidatorInfo]`

##### `is_validator_active(self, validator_id)`

Check if validator is active.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `register_validator(self, validator, initial_stake)`

Register a new validator.

**Parameters:**

- `self`: Any (required)
- `validator`: <class 'dubchain.consensus.validator.Validator'> (required)
- `initial_stake`: <class 'int'> = 0

**Returns:** `<class 'bool'>`

##### `select_proposer(self, block_number)`

Select proposer for next block based on stake.

**Parameters:**

- `self`: Any (required)
- `block_number`: <class 'int'> (required)

**Returns:** `typing.Optional[str]`

##### `slash_validator(self, validator_id, reason, evidence)`

Slash a validator for misbehavior.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)
- `reason`: <class 'str'> (required)
- `evidence`: typing.Optional[typing.Dict[str, typing.Any]] = None

**Returns:** `<class 'int'>`

##### `stake_to_validator(self, validator_id, delegator_id, amount)`

Stake tokens to a validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)
- `delegator_id`: <class 'str'> (required)
- `amount`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `unstake_from_validator(self, validator_id, delegator_id, amount)`

Unstake tokens from a validator.

**Parameters:**

- `self`: Any (required)
- `validator_id`: <class 'str'> (required)
- `delegator_id`: <class 'str'> (required)
- `amount`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `validate_block_proposal(self, proposer_id, block_data)`

Validate a block proposal.

**Parameters:**

- `self`: Any (required)
- `proposer_id`: <class 'str'> (required)
- `block_data`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'bool'>`

### RewardCalculator

Calculates rewards for validators and delegators.

#### Methods

##### `calculate_block_reward(self, block_number)`

Calculate reward for a block.

**Parameters:**

- `self`: Any (required)
- `block_number`: <class 'int'> (required)

**Returns:** `<class 'int'>`

##### `calculate_validator_reward(self, validator_stake, total_stake, block_reward)`

Calculate reward for a specific validator.

**Parameters:**

- `self`: Any (required)
- `validator_stake`: <class 'int'> (required)
- `total_stake`: <class 'int'> (required)
- `block_reward`: <class 'int'> (required)

**Returns:** `<class 'int'>`

### StakingPool

Manages staking pool for a validator.

#### Methods

##### `add_delegation(self, delegator_id, amount)`

Add delegation to pool.

**Parameters:**

- `self`: Any (required)
- `delegator_id`: <class 'str'> (required)
- `amount`: <class 'int'> (required)

**Returns:** `None`

##### `calculate_delegator_rewards(self, total_rewards, commission_rate)`

Calculate rewards for each delegator.

**Parameters:**

- `self`: Any (required)
- `total_rewards`: <class 'int'> (required)
- `commission_rate`: <class 'float'> (required)

**Returns:** `typing.Dict[str, int]`

##### `remove_delegation(self, delegator_id, amount)`

Remove delegation from pool.

**Parameters:**

- `self`: Any (required)
- `delegator_id`: <class 'str'> (required)
- `amount`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.consensus.proof_of_stake import *

# Create instance of ProofOfStake
proofofstake = ProofOfStake()

# Call method
result = proofofstake.add_validator("stake_amount")
```
