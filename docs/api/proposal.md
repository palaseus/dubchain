# Proposal API

**Module:** `dubchain.governance.proposal`

## Classes

### GovernanceManager

Manages governance proposals and voting.

#### Methods

##### `add_parameter_to_proposal(self, proposal_id, name, current_value, proposed_value, description)`

Add a parameter to a proposal.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)
- `name`: <class 'str'> (required)
- `current_value`: typing.Any (required)
- `proposed_value`: typing.Any (required)
- `description`: <class 'str'> = 

**Returns:** `<class 'bool'>`

##### `add_validator(self, address)`

Add a validator address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `None`

##### `create_proposal(self, proposer_address, proposal_type, title, description, voting_duration, execution_delay)`

Create a new governance proposal.

**Parameters:**

- `self`: Any (required)
- `proposer_address`: <class 'str'> (required)
- `proposal_type`: <enum 'ProposalType'> (required)
- `title`: <class 'str'> (required)
- `description`: <class 'str'> (required)
- `voting_duration`: <class 'int'> = 604800
- `execution_delay`: <class 'int'> = 86400

**Returns:** `<class 'str'>`

##### `execute_proposal(self, proposal_id)`

Execute a passed proposal.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `get_active_proposals(self)`

Get active proposals.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.governance.proposal.GovernanceProposal]`

##### `get_all_proposals(self)`

Get all proposals.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.governance.proposal.GovernanceProposal]`

##### `get_governance_stats(self)`

Get governance statistics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_passed_proposals(self)`

Get passed proposals.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.governance.proposal.GovernanceProposal]`

##### `get_proposal(self, proposal_id)`

Get a proposal by ID.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.governance.proposal.GovernanceProposal]`

##### `is_validator(self, address)`

Check if address is a validator.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `remove_validator(self, address)`

Remove a validator address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `None`

##### `set_voting_power(self, address, power)`

Set voting power for an address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)
- `power`: <class 'int'> (required)

**Returns:** `None`

##### `vote_on_proposal(self, proposal_id, voter_address, vote_type, private_key)`

Vote on a proposal.

**Parameters:**

- `self`: Any (required)
- `proposal_id`: <class 'str'> (required)
- `voter_address`: <class 'str'> (required)
- `vote_type`: <enum 'VoteType'> (required)
- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)

**Returns:** `<class 'bool'>`

### GovernanceProposal

A governance proposal.

#### Methods

##### `add_parameter(self, parameter)`

Add a parameter to the proposal.

**Parameters:**

- `self`: Any (required)
- `parameter`: <class 'dubchain.governance.proposal.ProposalParameter'> (required)

**Returns:** `None`

##### `add_vote(self, vote)`

Add a vote to the proposal.

**Parameters:**

- `self`: Any (required)
- `vote`: <class 'dubchain.governance.proposal.ProposalVote'> (required)

**Returns:** `<class 'bool'>`

##### `can_be_executed(self)`

Check if proposal can be executed.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `get_vote_summary(self)`

Get summary of votes.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `update_status(self)`

Update proposal status based on votes and timing.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

### ProposalParameter

A parameter in a governance proposal.

#### Methods

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ProposalStatus

Status of a governance proposal.

**Inherits from:** Enum

### ProposalType

Types of governance proposals.

**Inherits from:** Enum

### ProposalVote

A vote on a governance proposal.

#### Methods

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### VoteType

Types of votes.

**Inherits from:** Enum

## Usage Examples

```python
from dubchain.governance.proposal import *

# Create instance of GovernanceManager
governancemanager = GovernanceManager()

# Call method
result = governancemanager.add_parameter_to_proposal("name", "current_value", 1, 1)
```
