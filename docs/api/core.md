# Core API

**Module:** `dubchain.governance.core`

## Classes

### GovernanceConfig

Configuration for the governance system.

#### Methods

##### `validate(self)`

Validate configuration.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

### GovernanceEngine

Main governance engine for DubChain.

#### Methods

##### `cast_vote(self, proposal_id, voter_address, choice, voting_power, signature)`

Cast a vote on a proposal.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)
- `voter_address`: <class 'str'> (required)
- `choice`: <enum 'VoteChoice'> (required)
- `voting_power`: <class 'dubchain.governance.core.VotingPower'> (required)
- `signature`: <class 'str'> (required)

**Returns:** `<class 'dubchain.governance.core.Vote'>`

##### `create_proposal(self, proposer_address, title, description, proposal_type, voting_strategy, quorum_threshold, approval_threshold, execution_delay, execution_data)`

Create a new governance proposal.

**Parameters:**

- `self`: Any (required)
- `proposer_address`: <class 'str'> (required)
- `title`: <class 'str'> (required)
- `description`: <class 'str'> (required)
- `proposal_type`: <enum 'ProposalType'> (required)
- `voting_strategy`: <class 'str'> = token_weighted
- `quorum_threshold`: typing.Optional[int] = None
- `approval_threshold`: typing.Optional[float] = None
- `execution_delay`: typing.Optional[int] = None
- `execution_data`: typing.Optional[typing.Dict[str, typing.Any]] = None

**Returns:** `<class 'dubchain.governance.core.Proposal'>`

##### `emergency_pause(self, reason, block_height)`

Pause governance due to emergency.

**Parameters:**

- `self`: Any (required)
- `reason`: <class 'str'> (required)
- `block_height`: <class 'int'> (required)

**Returns:** `None`

##### `emergency_resume(self)`

Resume governance after emergency.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

##### `get_active_proposals(self)`

Get all active proposals.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.governance.core.Proposal]`

##### `get_proposal(self, proposal_id)`

Get a proposal by ID.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.governance.core.Proposal]`

##### `get_queued_proposals(self)`

Get all queued proposals.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.governance.core.Proposal]`

### GovernanceState

Current state of the governance system.

#### Methods

##### `add_proposal(self, proposal)`

Add a proposal to the state.

**Parameters:**

- `self`: Any (required)
- `proposal`: <class 'dubchain.governance.core.Proposal'> (required)

**Returns:** `None`

##### `get_proposal(self, proposal_id)`

Get a proposal by ID.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.governance.core.Proposal]`

##### `update_proposal_status(self, proposal_id, status)`

Update proposal status.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)
- `status`: <enum 'ProposalStatus'> (required)

**Returns:** `None`

### Proposal

A governance proposal.

#### Methods

##### `add_vote(self, vote)`

Add a vote to this proposal.

**Parameters:**

- `self`: Any (required)
- `vote`: <class 'dubchain.governance.core.Vote'> (required)

**Returns:** `None`

##### `can_execute(self)`

Check if proposal can be executed.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `from_dict(data)`

Create proposal from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `Proposal`

##### `get_vote_summary(self)`

Get summary of votes for this proposal.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `to_dict(self)`

Convert proposal to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ProposalStatus

Status of a governance proposal.

**Inherits from:** Enum

### ProposalType

Type of governance proposal.

**Inherits from:** Enum

### Vote

A governance vote.

#### Methods

##### `from_dict(data)`

Create vote from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `Vote`

##### `to_dict(self)`

Convert vote to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### VoteChoice

Vote choices for governance proposals.

**Inherits from:** Enum

### VotingPower

Represents voting power for a voter.

#### Methods

##### `is_delegated(self)`

Check if this voting power includes delegations.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `total_power(self)`

Get total voting power including delegations.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

## Usage Examples

```python
from dubchain.governance.core import *

# Create instance of GovernanceConfig
governanceconfig = GovernanceConfig()

# Call method
result = governanceconfig.validate()
```
