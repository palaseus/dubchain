# Consensus Types API

**Module:** `dubchain.consensus.consensus_types`

## Classes

### ConsensusConfig

Configuration for consensus mechanisms.

#### Methods

##### `from_dict(data)`

Create from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `ConsensusConfig`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ConsensusMetrics

Consensus performance metrics.

#### Properties

##### `failure_rate`

Calculate consensus failure rate.

##### `success_rate`

Calculate consensus success rate.

### ConsensusResult

Result of consensus operation.

### ConsensusState

Current state of consensus mechanism.

#### Methods

##### `should_switch_consensus(self)`

Check if consensus should be switched.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `update_metrics(self, success, block_time, gas_used)`

Update consensus metrics.

**Parameters:**

- `self`: Any (required)
- `success`: <class 'bool'> (required)
- `block_time`: <class 'float'> (required)
- `gas_used`: <class 'int'> (required)

**Returns:** `None`

### ConsensusType

Types of consensus mechanisms.

**Inherits from:** Enum

### DelegateInfo

Delegate information for DPoS.

### HotStuffMessage

HotStuff consensus message.

### HotStuffPhase

HotStuff consensus phases.

**Inherits from:** Enum

### PBFTMessage

PBFT consensus message.

### PBFTPhase

PBFT consensus phases.

**Inherits from:** Enum

### PoAAuthority

Proof-of-Authority authority information.

### PoAStatus

Proof-of-Authority validator status.

**Inherits from:** Enum

### PoHEntry

Proof-of-History entry.

### PoHStatus

Proof-of-History status.

**Inherits from:** Enum

### PoSpaceChallenge

Proof-of-Space challenge.

### PoSpacePlot

Proof-of-Space plot information.

### PoSpaceStatus

Proof-of-Space/Time status.

**Inherits from:** Enum

### StakingInfo

Information about staking operations.

### ValidatorRole

Validator roles in consensus.

**Inherits from:** Enum

### ValidatorStatus

Validator status states.

**Inherits from:** Enum

### VotingPower

Voting power information.

## Usage Examples

```python
from dubchain.consensus.consensus_types import *

# Create instance of ConsensusConfig
consensusconfig = ConsensusConfig()

# Call method
result = consensusconfig.from_dict("data")
```
