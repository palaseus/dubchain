# Signatures API

**Module:** `dubchain.crypto.signatures`

## Classes

### ECDSASigner

ECDSA signature operations with secp256k1 curve.

#### Methods

##### `generate_keypair()`

Generate a new ECDSA key pair.

**Returns:** `typing.Tuple[dubchain.crypto.signatures.PrivateKey, dubchain.crypto.signatures.PublicKey]`

##### `recover_public_key(signature, message)`

Recover the public key from a signature and message.

Note: This is a simplified implementation. In practice, you'd need
to handle the recovery ID and implement proper ECDSA recovery.

This method is intentionally not implemented as ECDSA public key recovery
is complex and requires careful handling of recovery IDs and curve mathematics.
For production use, consider using a cryptographic library like cryptography
or secp256k1 that provides proper ECDSA recovery functionality.

**Parameters:**

- `signature`: <class 'dubchain.crypto.signatures.Signature'> (required)
- `message`: <class 'bytes'> (required)

**Returns:** `typing.Optional[dubchain.crypto.signatures.PublicKey]`

##### `sign_message(private_key, message)`

Sign a message with a private key.

**Parameters:**

- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)
- `message`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `<class 'dubchain.crypto.signatures.Signature'>`

##### `sign_transaction(private_key, transaction_hash)`

Sign a transaction hash.

**Parameters:**

- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)
- `transaction_hash`: <class 'dubchain.crypto.hashing.Hash'> (required)

**Returns:** `<class 'dubchain.crypto.signatures.Signature'>`

##### `verify_signature(public_key, signature, message)`

Verify a signature against a message and public key.

**Parameters:**

- `public_key`: <class 'dubchain.crypto.signatures.PublicKey'> (required)
- `signature`: <class 'dubchain.crypto.signatures.Signature'> (required)
- `message`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `<class 'bool'>`

##### `verify_transaction_signature(public_key, signature, transaction_hash)`

Verify a transaction signature.

**Parameters:**

- `public_key`: <class 'dubchain.crypto.signatures.PublicKey'> (required)
- `signature`: <class 'dubchain.crypto.signatures.Signature'> (required)
- `transaction_hash`: <class 'dubchain.crypto.hashing.Hash'> (required)

**Returns:** `<class 'bool'>`

### PrivateKey

Immutable private key with cryptographic operations.

#### Methods

##### `add_scalar(self, scalar)`

Add a scalar to the private key (mod n).

**Parameters:**

- `self`: Any (required)
- `scalar`: typing.Union[int, bytes] (required)

**Returns:** `PrivateKey`

##### `from_bytes(key_bytes)`

Create a private key from raw bytes.

**Parameters:**

- `key_bytes`: <class 'bytes'> (required)

**Returns:** `PrivateKey`

##### `from_hex(hex_string)`

Create a private key from hexadecimal string.

**Parameters:**

- `hex_string`: <class 'str'> (required)

**Returns:** `PrivateKey`

##### `generate()`

Generate a new random private key.

**Returns:** `PrivateKey`

##### `get_public_key(self)`

Get the corresponding public key.

**Parameters:**

- `self`: Any (required)

**Returns:** `PublicKey`

##### `sign(self, message)`

Sign a message with this private key.

**Parameters:**

- `self`: Any (required)
- `message`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `Signature`

##### `to_bytes(self)`

Convert private key to raw bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `to_hex(self)`

Convert private key to hexadecimal string.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

### PublicKey

Immutable public key with cryptographic operations.

#### Methods

##### `from_bytes(key_bytes)`

Create a public key from raw bytes (compressed or uncompressed).

**Parameters:**

- `key_bytes`: <class 'bytes'> (required)

**Returns:** `PublicKey`

##### `from_hex(hex_string)`

Create a public key from hexadecimal string.

**Parameters:**

- `hex_string`: <class 'str'> (required)

**Returns:** `PublicKey`

##### `to_address(self)`

Convert public key to address (first 20 bytes of hash).

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `to_bytes(self, compressed)`

Convert public key to raw bytes.

**Parameters:**

- `self`: Any (required)
- `compressed`: <class 'bool'> = True

**Returns:** `<class 'bytes'>`

##### `to_hex(self, compressed)`

Convert public key to hexadecimal string.

**Parameters:**

- `self`: Any (required)
- `compressed`: <class 'bool'> = True

**Returns:** `<class 'str'>`

##### `verify(self, signature, message)`

Verify a signature against a message.

**Parameters:**

- `self`: Any (required)
- `signature`: Signature (required)
- `message`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `<class 'bool'>`

### Signature

Immutable digital signature.

#### Methods

##### `from_bytes(signature_bytes, message)`

Create a signature from raw bytes.

**Parameters:**

- `signature_bytes`: <class 'bytes'> (required)
- `message`: <class 'bytes'> (required)

**Returns:** `Signature`

##### `from_hex(hex_string, message)`

Create a signature from hexadecimal string.

**Parameters:**

- `hex_string`: <class 'str'> (required)
- `message`: <class 'bytes'> (required)

**Returns:** `Signature`

##### `to_bytes(self)`

Convert signature to raw bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `to_der(self)`

Convert signature to DER format.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `to_hex(self)`

Convert signature to hexadecimal string.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

## Usage Examples

```python
from dubchain.crypto.signatures import *

# Create instance of ECDSASigner
ecdsasigner = ECDSASigner()

```
