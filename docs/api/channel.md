# Channel API

**Module:** `dubchain.state_channels.channel`

## Classes

### ChannelParticipant

A participant in a state channel.

#### Methods

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ChannelState

State of a channel.

**Inherits from:** Enum

### ChannelStatus

Status of a state channel.

**Inherits from:** Enum

### ChannelUpdate

An update to a state channel.

#### Methods

##### `add_signature(self, participant_address, signature)`

Add a signature to the update.

**Parameters:**

- `self`: Any (required)
- `participant_address`: <class 'str'> (required)
- `signature`: <class 'str'> (required)

**Returns:** `None`

##### `is_fully_signed(self, required_participants)`

Check if the update is fully signed.

**Parameters:**

- `self`: Any (required)
- `required_participants`: typing.Set[str] (required)

**Returns:** `<class 'bool'>`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### StateChannel

A state channel between multiple parties.

#### Methods

##### `add_participant(self, participant)`

Add a participant to the channel.

**Parameters:**

- `self`: Any (required)
- `participant`: <class 'dubchain.state_channels.channel.ChannelParticipant'> (required)

**Returns:** `None`

##### `can_close(self)`

Check if the channel can be closed.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `close_channel(self)`

Close the channel.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `create_channel(self, participants, deposits, public_keys)`

Create/initialize the channel with participants and deposits.

**Parameters:**

- `self`: Any (required)
- `participants`: typing.List[dubchain.state_channels.channel.ChannelParticipant] (required)
- `deposits`: typing.List[int] (required)
- `public_keys`: typing.List[dubchain.crypto.signatures.PublicKey] (required)

**Returns:** `<class 'bool'>`

##### `create_update(self, new_balances)`

Create a new channel update.

**Parameters:**

- `self`: Any (required)
- `new_balances`: typing.Dict[str, int] (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel.ChannelUpdate]`

##### `finalize_channel(self)`

Finalize the channel.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `get_latest_state(self)`

Get the latest state update.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel.ChannelUpdate]`

##### `get_participant(self, address)`

Get a participant by address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel.ChannelParticipant]`

##### `is_expired(self)`

Check if the channel is expired.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `sign_update(self, participant_address, private_key)`

Sign the current update.

**Parameters:**

- `self`: Any (required)
- `participant_address`: <class 'str'> (required)
- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)

**Returns:** `<class 'bool'>`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `update_participant_balance(self, address, new_balance)`

Update a participant's balance.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)
- `new_balance`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

### StateChannelManager

Manages state channels.

#### Methods

##### `cleanup_expired_channels(self)`

Clean up expired channels.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `close_channel(self, channel_id)`

Close a channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `create_channel(self, channel_id, participants, dispute_period, challenge_period)`

Create a new state channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)
- `participants`: typing.List[dubchain.state_channels.channel.ChannelParticipant] (required)
- `dispute_period`: <class 'int'> = 604800
- `challenge_period`: <class 'int'> = 86400

**Returns:** `<class 'dubchain.state_channels.channel.StateChannel'>`

##### `finalize_channel(self, channel_id)`

Finalize a channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `get_channel(self, channel_id)`

Get a channel by ID.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel.StateChannel]`

##### `get_channel_stats(self)`

Get channel statistics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_participant_channels(self, address)`

Get all channels for a participant.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `typing.List[dubchain.state_channels.channel.StateChannel]`

##### `update_channel(self, channel_id, new_balances, participant_address, private_key)`

Update a channel with new balances.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)
- `new_balances`: typing.Dict[str, int] (required)
- `participant_address`: <class 'str'> (required)
- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.state_channels.channel import *

# Create instance of ChannelParticipant
channelparticipant = ChannelParticipant()

# Call method
result = channelparticipant.to_dict()
```
