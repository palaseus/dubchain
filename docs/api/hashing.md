# Hashing API

**Module:** `dubchain.crypto.hashing`

## Classes

### Hash

Immutable hash value with comparison and string representation.

#### Methods

##### `from_hex(hex_string)`

Create a Hash from a hexadecimal string.

**Parameters:**

- `hex_string`: <class 'str'> (required)

**Returns:** `Hash`

##### `from_int(value)`

Create a Hash from an integer (big-endian).

**Parameters:**

- `value`: <class 'int'> (required)

**Returns:** `Hash`

##### `max_value()`

Create a maximum hash (all 0xFF bytes).

**Returns:** `Hash`

##### `to_hex(self)`

Convert hash to hexadecimal string.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `to_int(self)`

Convert hash to integer (big-endian).

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `zero()`

Create a zero hash (all zeros).

**Returns:** `Hash`

### SHA256Hasher

SHA-256 hasher with blockchain-specific utilities.

#### Methods

##### `calculate_difficulty_target(difficulty)`

Calculate the target hash for a given difficulty.

Args:
    difficulty: Difficulty in bits (number of leading zeros)

Returns:
    Target hash value

**Parameters:**

- `difficulty`: <class 'int'> (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `double_hash(data)`

Double SHA-256 hash (Bitcoin-style).

Args:
    data: Data to hash (bytes or string)

Returns:
    Hash object containing the double SHA-256 hash

**Parameters:**

- `data`: typing.Union[bytes, str] (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `hash(data)`

Hash data using SHA-256.

Args:
    data: Data to hash (bytes or string)

Returns:
    Hash object containing the SHA-256 hash

**Parameters:**

- `data`: typing.Union[bytes, str] (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `hash_list(items)`

Hash a list of items by concatenating them.

Args:
    items: List of items to hash

Returns:
    Hash of the concatenated items

**Parameters:**

- `items`: typing.List[typing.Union[bytes, str]] (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `hmac_sha256(key, data)`

HMAC-SHA256 for keyed hashing.

Args:
    key: HMAC key
    data: Data to hash

Returns:
    Hash object containing the HMAC-SHA256 hash

**Parameters:**

- `key`: typing.Union[bytes, str] (required)
- `data`: typing.Union[bytes, str] (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `pbkdf2_hmac(password, salt, iterations, key_length)`

PBKDF2-HMAC-SHA256 for key derivation.

Args:
    password: Password to derive key from
    salt: Salt for key derivation
    iterations: Number of iterations
    key_length: Length of derived key

Returns:
    Derived key bytes

**Parameters:**

- `password`: typing.Union[bytes, str] (required)
- `salt`: typing.Union[bytes, str] (required)
- `iterations`: <class 'int'> = 100000
- `key_length`: <class 'int'> = 32

**Returns:** `<class 'bytes'>`

##### `verify_proof_of_work(hash_value, difficulty)`

Verify that a hash meets the proof of work difficulty requirement.

Args:
    hash_value: Hash to verify
    difficulty: Required number of leading zeros (in bits)

Returns:
    True if the hash meets the difficulty requirement

**Parameters:**

- `hash_value`: <class 'dubchain.crypto.hashing.Hash'> (required)
- `difficulty`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.crypto.hashing import *

# Create instance of Hash
hash = Hash()

# Call method
result = hash.from_hex("hex_string")
```
