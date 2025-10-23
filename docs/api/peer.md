# Peer API

**Module:** `dubchain.network.peer`

## Classes

### ConnectionType

Types of peer connections.

**Inherits from:** Enum

### Peer

Represents a peer in the network.

#### Methods

##### `get_address(self)`

Get peer address.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `get_id(self)`

Get peer ID.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `get_public_key(self)`

Get peer public key.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Optional[dubchain.crypto.signatures.PublicKey]`

##### `is_connected(self)`

Check if peer is connected.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `update_activity(self)`

Update last activity timestamp.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

### PeerAuthenticator

Handles peer authentication and key exchange.

#### Methods

##### `authenticate_peer(self, peer_id, challenge)`

Authenticate a peer using challenge-response.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `challenge`: <class 'bytes'> (required)

**Returns:** `typing.Tuple[bool, typing.Optional[bytes]]`

##### `get_peer_public_key(self, peer_id)`

Get peer's public key.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.crypto.signatures.PublicKey]`

##### `verify_peer_authentication(self, peer_id, response)`

Verify peer authentication response.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `response`: <class 'bytes'> (required)

**Returns:** `<class 'bool'>`

### PeerConfig

Configuration for peer management.

### PeerConnection

Active peer connection.

### PeerConnectionStatus

Peer connection status.

**Inherits from:** Enum

### PeerHealthMonitor

Monitors peer health and connection quality.

#### Methods

##### `check_peer_health(self, peer_id, connection)`

Check if a peer is healthy.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `connection`: <class 'dubchain.network.peer.PeerConnection'> (required)

**Returns:** `<class 'bool'>`

##### `get_peer_health(self, peer_id)`

Get peer health information.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `typing.Optional[typing.Dict[str, typing.Any]]`

##### `get_unhealthy_peers(self)`

Get list of unhealthy peers.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Set[str]`

### PeerInfo

Information about a peer.

#### Methods

##### `add_capability(self, capability)`

Add a capability.

**Parameters:**

- `self`: Any (required)
- `capability`: <class 'str'> (required)

**Returns:** `None`

##### `get_connection_success_rate(self)`

Get the connection success rate.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

##### `has_capability(self, capability)`

Check if peer has a capability.

**Parameters:**

- `self`: Any (required)
- `capability`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `is_healthy(self)`

Check if peer is healthy.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `record_failed_connection(self)`

Record a failed connection.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

##### `record_successful_connection(self)`

Record a successful connection.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

##### `to_dict(self)`

Convert to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `update_last_seen(self)`

Update the last seen timestamp.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

### PeerManager

Main peer management system.

#### Methods

##### `authenticate_peer(self, peer_id)`

Authenticate a peer.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `cleanup_inactive_peers(self)`

Clean up inactive peers.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `connect_to_peer(self, address, port, peer_info)`

Connect to a peer.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)
- `port`: <class 'int'> (required)
- `peer_info`: typing.Optional[dubchain.network.peer.PeerInfo] = None

**Returns:** `typing.Optional[str]`

##### `disconnect_peer(self, peer_id)`

Disconnect from a peer.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `get_connected_peers(self)`

Get list of connected peer IDs.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[str]`

##### `get_peer_count(self)`

Get total number of connected peers.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_peer_info(self, peer_id)`

Get peer information.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.network.peer.PeerInfo]`

##### `get_peer_metrics(self, peer_id)`

Get peer metrics.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.network.peer.PeerMetrics]`

##### `get_peer_reputation(self, peer_id)`

Get peer reputation.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.network.peer.PeerReputation]`

##### `health_check_all_peers(self)`

Perform health check on all connected peers.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, bool]`

##### `receive_message(self, peer_id, message_bytes)`

Receive message from peer.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `message_bytes`: <class 'bytes'> (required)

**Returns:** `typing.Optional[typing.Dict[str, typing.Any]]`

##### `send_message(self, peer_id, message)`

Send message to peer.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `message`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'bool'>`

### PeerMetrics

Peer performance metrics.

### PeerNode

Represents a peer node in the network.

#### Methods

##### `get_address(self)`

Get peer address.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `get_id(self)`

Get peer ID.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `get_public_key(self)`

Get peer public key.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Optional[dubchain.crypto.signatures.PublicKey]`

##### `is_connected(self)`

Check if peer is connected.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `update_activity(self)`

Update last activity timestamp.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

### PeerReputation

Peer reputation and trust score.

### PeerReputationManager

Manages peer reputation and trust scores.

#### Methods

##### `get_reputation(self, peer_id)`

Get peer reputation.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `typing.Optional[dubchain.network.peer.PeerReputation]`

##### `get_top_peers(self, limit)`

Get top peers by reputation score.

**Parameters:**

- `self`: Any (required)
- `limit`: <class 'int'> = 10

**Returns:** `typing.List[typing.Tuple[str, float]]`

##### `is_trusted_peer(self, peer_id)`

Check if peer is trusted.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)

**Returns:** `<class 'bool'>`

##### `update_reputation(self, peer_id, interaction_type, success, severity)`

Update peer reputation based on interaction.

**Parameters:**

- `self`: Any (required)
- `peer_id`: <class 'str'> (required)
- `interaction_type`: <class 'str'> (required)
- `success`: <class 'bool'> (required)
- `severity`: <class 'float'> = 1.0

**Returns:** `None`

### PeerRole

Peer roles in the network.

**Inherits from:** Enum

### PeerStatus

Peer connection status.

**Inherits from:** Enum

## Usage Examples

```python
from dubchain.network.peer import *

# Create instance of ConnectionType
connectiontype = ConnectionType()

```
