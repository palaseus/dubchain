# Channel Protocol API

**Module:** `dubchain.state_channels.channel_protocol`

## Classes

### ChannelCloseReason

Reasons for channel closure.

**Inherits from:** Enum

### ChannelConfig

Configuration for state channels.

#### Methods

##### `from_dict(data)`

Create configuration from dictionary.

**Parameters:**

- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `ChannelConfig`

##### `to_dict(self)`

Convert configuration to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ChannelError

Channel-specific error.

**Inherits from:** Exception

### ChannelEvent

Channel events.

**Inherits from:** Enum

### ChannelId

Channel ID with generation capability.

#### Methods

##### `generate()`

Generate a new unique channel ID.

**Returns:** `ChannelId`

### ChannelManager

Main channel management system.

#### Methods

##### `cleanup_expired_channels(self)`

Clean up expired channels.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `close_channel(self, channel_id, closer_address)`

Close a state channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)
- `closer_address`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `create_channel(self, channel_id, participants, initial_balance)`

Create a new state channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)
- `participants`: typing.List[dubchain.state_channels.channel_protocol.ChannelParticipant] (required)
- `initial_balance`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `get_channel(self, channel_id)`

Get channel state.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel_protocol.ChannelState]`

##### `initiate_dispute(self, channel_id, initiator_address)`

Initiate a dispute for a channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)
- `initiator_address`: <class 'str'> (required)

**Returns:** `typing.Optional[str]`

##### `list_channels(self, participant_address)`

List channels.

**Parameters:**

- `self`: Any (required)
- `participant_address`: typing.Optional[str] = None

**Returns:** `typing.List[dubchain.state_channels.channel_protocol.ChannelState]`

##### `process_payment(self, channel_id, payment)`

Process a payment in a channel.

**Parameters:**

- `self`: Any (required)
- `channel_id`: <class 'str'> (required)
- `payment`: <class 'dubchain.state_channels.channel_protocol.Payment'> (required)

**Returns:** `<class 'bool'>`

### ChannelParticipant

Participant in a state channel.

### ChannelSecurityError

Channel security error.

**Inherits from:** ChannelError

### ChannelState

State of a state channel.

#### Methods

##### `apply_state_update(self, update)`

Apply a state update.

**Parameters:**

- `self`: Any (required)
- `update`: <class 'dubchain.state_channels.channel_protocol.StateUpdate'> (required)

**Returns:** `<class 'bool'>`

##### `can_update_state(self, update)`

Check if state can be updated with given update.

**Parameters:**

- `self`: Any (required)
- `update`: <class 'dubchain.state_channels.channel_protocol.StateUpdate'> (required)

**Returns:** `<class 'bool'>`

##### `get_total_deposits(self)`

Get total deposits.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `validate_balances(self)`

Validate that balances are consistent.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

#### Properties

##### `participant_names`

Get list of participant names.

### ChannelStatus

Status of a state channel.

**Inherits from:** Enum

### ChannelTimeoutError

Channel timeout error.

**Inherits from:** ChannelError

### InsufficientSignaturesError

Insufficient signatures error.

**Inherits from:** ChannelError

### InvalidStateUpdateError

Invalid state update error.

**Inherits from:** ChannelError

### Payment

Payment within a state channel.

### PaymentProcessor

Processes payments within state channels.

#### Methods

##### `process_payment(self, channel_state, payment)`

Process a payment and return new channel state.

**Parameters:**

- `self`: Any (required)
- `channel_state`: <class 'dubchain.state_channels.channel_protocol.ChannelState'> (required)
- `payment`: <class 'dubchain.state_channels.channel_protocol.Payment'> (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel_protocol.ChannelState]`

### PaymentType

Types of payments.

**Inherits from:** Enum

### StateChannel

Complete state channel.

#### Methods

##### `add_event_handler(self, event_type, handler)`

Add an event handler.

**Parameters:**

- `self`: Any (required)
- `event_type`: <enum 'ChannelEvent'> (required)
- `handler`: typing.Callable (required)

**Returns:** `None`

##### `close_channel(self)`

Close the channel.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `create_channel(self, participants, deposits, public_keys)`

Create/initialize the channel with participants and deposits.

**Parameters:**

- `self`: Any (required)
- `participants`: typing.List[str] (required)
- `deposits`: typing.Dict[str, int] (required)
- `public_keys`: typing.Dict[str, dubchain.crypto.signatures.PublicKey] (required)

**Returns:** `<class 'bool'>`

##### `expire_channel(self)`

Expire the channel.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `freeze_channel(self, reason)`

Freeze the channel.

**Parameters:**

- `self`: Any (required)
- `reason`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `get_channel_info(self)`

Get channel information.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_latest_state(self)`

Get the latest state.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Optional[dubchain.state_channels.channel_protocol.ChannelState]`

##### `is_active(self)`

Check if the channel is active.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `open_channel(self)`

Open the channel for transactions.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `update_state(self, update)`

Update the channel state with a new update.

**Parameters:**

- `self`: Any (required)
- `update`: <class 'dubchain.state_channels.channel_protocol.StateUpdate'> (required)

**Returns:** `<class 'bool'>`

### StateUpdate

State update for a channel.

#### Methods

##### `add_signature(self, participant, signature)`

Add a signature to the update.

**Parameters:**

- `self`: Any (required)
- `participant`: <class 'str'> (required)
- `signature`: <class 'str'> (required)

**Returns:** `None`

##### `get_hash(self)`

Get hash of the state update.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `has_required_signatures(self, config)`

Check if update has required signatures.

**Parameters:**

- `self`: Any (required)
- `config`: ChannelConfig (required)

**Returns:** `<class 'bool'>`

##### `verify_signatures(self, public_keys)`

Verify all signatures on the update.

**Parameters:**

- `self`: Any (required)
- `public_keys`: typing.Dict[str, dubchain.crypto.signatures.PublicKey] (required)

**Returns:** `<class 'bool'>`

### StateUpdateType

Types of state updates.

**Inherits from:** Enum

### StateValidator

Validates channel state transitions.

#### Methods

##### `validate_state_transition(self, old_state, new_state)`

Validate state transition.

**Parameters:**

- `self`: Any (required)
- `old_state`: <class 'dubchain.state_channels.channel_protocol.ChannelState'> (required)
- `new_state`: <class 'dubchain.state_channels.channel_protocol.ChannelState'> (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.state_channels.channel_protocol import *

# Create instance of ChannelCloseReason
channelclosereason = ChannelCloseReason()

```
