# Block API

**Module:** `dubchain.core.block`

## Classes

### Block

A blockchain block containing transactions.

#### Methods

##### `create_block(transactions, previous_block, difficulty, gas_limit, extra_data)`

Create a new block.

**Parameters:**

- `transactions`: typing.List[dubchain.core.transaction.Transaction] (required)
- `previous_block`: Block (required)
- `difficulty`: <class 'int'> (required)
- `gas_limit`: <class 'int'> = 10000000
- `extra_data`: typing.Optional[bytes] = None

**Returns:** `Block`

##### `create_genesis_block(coinbase_recipient, coinbase_amount, difficulty)`

Create the genesis block.

**Parameters:**

- `coinbase_recipient`: <class 'str'> (required)
- `coinbase_amount`: <class 'int'> = 1000000
- `difficulty`: <class 'int'> = 1

**Returns:** `Block`

##### `get_coinbase_transaction(self)`

Get the coinbase transaction of this block.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.core.transaction.Transaction'>`

##### `get_hash(self)`

Get the hash of this block.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `get_merkle_tree(self)`

Get the merkle tree of this block's transactions.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.crypto.merkle.MerkleTree'>`

##### `get_regular_transactions(self)`

Get all non-coinbase transactions.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.core.transaction.Transaction]`

##### `get_total_gas_used(self)`

Get the total gas used by all transactions.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_total_transaction_fees(self, utxos)`

Get the total transaction fees in this block.

**Parameters:**

- `self`: Any (required)
- `utxos`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'int'>`

##### `get_transaction_proof(self, transaction)`

Get a merkle proof for a transaction in this block.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)

**Returns:** `typing.Optional[typing.Any]`

##### `is_valid(self, utxos, previous_block)`

Check if this block is valid.

**Parameters:**

- `self`: Any (required)
- `utxos`: typing.Dict[str, typing.Any] (required)
- `previous_block`: typing.Optional[ForwardRef('Block')] = None

**Returns:** `<class 'bool'>`

##### `to_bytes(self)`

Serialize block to bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `verify_transaction_proof(self, transaction, proof)`

Verify a merkle proof for a transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)
- `proof`: typing.Any (required)

**Returns:** `<class 'bool'>`

### BlockHeader

Header of a blockchain block.

#### Methods

##### `get_difficulty_target(self)`

Get the target hash for this difficulty.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `get_hash(self)`

Get the hash of this block header.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `meets_difficulty(self)`

Check if the block header meets the difficulty requirement.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `to_bytes(self)`

Serialize block header to bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `with_merkle_root(self, merkle_root)`

Create a new block header with a different merkle root.

**Parameters:**

- `self`: Any (required)
- `merkle_root`: <class 'dubchain.crypto.hashing.Hash'> (required)

**Returns:** `BlockHeader`

##### `with_nonce(self, nonce)`

Create a new block header with a different nonce.

**Parameters:**

- `self`: Any (required)
- `nonce`: <class 'int'> (required)

**Returns:** `BlockHeader`

## Usage Examples

```python
from dubchain.core.block import *

# Create instance of Block
block = Block()

# Call method
result = block.create_block(1, 1, 1, 1, 1)
```
