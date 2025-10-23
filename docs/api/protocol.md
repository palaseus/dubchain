# Protocol API

**Module:** `dubchain.network.protocol`

## Classes

### MessageRouter

Routes messages to appropriate handlers.

#### Methods

##### `register_route(self, message_type, handler)`

Register a route for a message type.

**Parameters:**

- `self`: Any (required)
- `message_type`: <enum 'MessageType'> (required)
- `handler`: <built-in function callable> (required)

**Returns:** `None`

##### `route_message(self, message)`

Route a message to its handlers.

**Parameters:**

- `self`: Any (required)
- `message`: <class 'dubchain.network.protocol.NetworkMessage'> (required)

**Returns:** `<class 'bool'>`

### MessageType

Types of network messages.

**Inherits from:** Enum

### NetworkManager

Manages network operations and peer connections.

#### Methods

##### `add_peer(self, peer_id, peer_info)`

Add a peer to the network.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `peer_info`: typing.Any (required)

**Returns:** `None`

##### `broadcast_message(self, message)`

Broadcast a message to all peers.

**Parameters:**

- `self`: Any (required)
- `message`: <class 'dubchain.network.protocol.NetworkMessage'> (required)

**Returns:** `<class 'int'>`

##### `get_peer_count(self)`

Get the number of connected peers.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_peer_list(self)`

Get list of peer IDs.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[str]`

##### `process_message(self, message_bytes)`

Process an incoming message.

**Parameters:**

- `self`: Any (required)
- `message_bytes`: <class 'bytes'> (required)

**Returns:** `typing.Optional[dubchain.network.protocol.NetworkMessage]`

##### `remove_peer(self, peer_id)`

Remove a peer from the network.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `None`

##### `send_message_to_peer(self, peer_id, message)`

Send a message to a specific peer.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `message`: <class 'dubchain.network.protocol.NetworkMessage'> (required)

**Returns:** `<class 'bool'>`

### NetworkMessage

A network message.

#### Methods

##### `from_bytes(data)`

Deserialize message from bytes.

**Parameters:**

- `data`: <class 'bytes'> (required)

**Returns:** `NetworkMessage`

##### `sign(self, private_key)`

Sign the message.

**Parameters:**

- `self`: Any (required)
- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)

**Returns:** `NetworkMessage`

##### `to_bytes(self)`

Serialize message to bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `verify(self, public_key)`

Verify the message signature.

**Parameters:**

- `self`: Any (required)
- `public_key`: <class 'dubchain.crypto.signatures.PublicKey'> (required)

**Returns:** `<class 'bool'>`

### NetworkProtocol

Core network protocol implementation.

#### Methods

##### `create_block_message(self, block_data)`

Create a block message.

**Parameters:**

- `self`: Any (required)
- `block_data`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_error_message(self, error_code, error_message)`

Create an error message.

**Parameters:**

- `self`: Any (required)
- `error_code`: <class 'str'> (required)
- `error_message`: <class 'str'> (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_handshake_message(self)`

Create a handshake message.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_message(self, message_type, data)`

Create a new network message.

**Parameters:**

- `self`: Any (required)
- `message_type`: <enum 'MessageType'> (required)
- `data`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_ping_message(self)`

Create a ping message.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_pong_message(self, ping_nonce)`

Create a pong message.

**Parameters:**

- `self`: Any (required)
- `ping_nonce`: <class 'str'> (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_sync_request_message(self, start_height, end_height)`

Create a sync request message.

**Parameters:**

- `self`: Any (required)
- `start_height`: <class 'int'> (required)
- `end_height`: <class 'int'> (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_sync_response_message(self, blocks, request_id)`

Create a sync response message.

**Parameters:**

- `self`: Any (required)
- `blocks`: typing.List[typing.Dict[str, typing.Any]] (required)
- `request_id`: <class 'str'> (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `create_transaction_message(self, transaction_data)`

Create a transaction message.

**Parameters:**

- `self`: Any (required)
- `transaction_data`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'dubchain.network.protocol.NetworkMessage'>`

##### `receive_message(self, message_bytes)`

Receive and process a message.

**Parameters:**

- `self`: Any (required)
- `message_bytes`: <class 'bytes'> (required)

**Returns:** `typing.Optional[dubchain.network.protocol.NetworkMessage]`

##### `register_handler(self, message_type, handler)`

Register a message handler.

**Parameters:**

- `self`: Any (required)
- `message_type`: <enum 'MessageType'> (required)
- `handler`: <built-in function callable> (required)

**Returns:** `None`

##### `send_message(self, peer_id, message)`

Send a message to a peer.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `message`: <class 'dubchain.network.protocol.NetworkMessage'> (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.network.protocol import *

# Create instance of MessageRouter
messagerouter = MessageRouter()

# Call method
result = messagerouter.register_route(1)
```
