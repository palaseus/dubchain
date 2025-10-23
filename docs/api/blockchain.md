# Blockchain API

**Module:** `dubchain.core.blockchain`

## Classes

### Blockchain

Main blockchain implementation.

#### Methods

##### `add_block(self, block)`

Add a block to the blockchain.

Args:
    block: Block to add

Returns:
    True if block was added successfully, False otherwise

**Parameters:**

- `self`: Any (required)
- `block`: <class 'dubchain.core.block.Block'> (required)

**Returns:** `<class 'bool'>`

##### `add_transaction(self, transaction)`

Add a transaction to the pending pool.

Args:
    transaction: Transaction to add

Returns:
    True if transaction was added successfully, False otherwise

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)

**Returns:** `<class 'bool'>`

##### `clear_pending_transactions(self)`

Clear all pending transactions.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

##### `create_genesis_block(self, coinbase_recipient, coinbase_amount)`

Create and add the genesis block.

**Parameters:**

- `self`: Any (required)
- `coinbase_recipient`: <class 'str'> (required)
- `coinbase_amount`: <class 'int'> = 1000000

**Returns:** `<class 'dubchain.core.block.Block'>`

##### `create_transfer_transaction(self, sender_private_key, recipient_address, amount, fee)`

Create a transfer transaction.

Args:
    sender_private_key: Private key of sender
    recipient_address: Address of recipient
    amount: Amount to transfer
    fee: Transaction fee

Returns:
    Created transaction, or None if creation failed

**Parameters:**

- `self`: Any (required)
- `sender_private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)
- `recipient_address`: <class 'str'> (required)
- `amount`: <class 'int'> (required)
- `fee`: <class 'int'> = 1000

**Returns:** `typing.Optional[dubchain.core.transaction.Transaction]`

##### `export_state(self)`

Export the current blockchain state.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_balance(self, address)`

Get the balance of an address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `<class 'int'>`

##### `get_best_chain(self)`

Get the best (longest) chain.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.core.block.Block]`

##### `get_block_by_hash(self, block_hash)`

Get a block by its hash.

**Parameters:**

- `self`: Any (required)
- `block_hash`: <class 'dubchain.crypto.hashing.Hash'> (required)

**Returns:** `typing.Optional[dubchain.core.block.Block]`

##### `get_block_by_height(self, height)`

Get a block by its height.

**Parameters:**

- `self`: Any (required)
- `height`: <class 'int'> (required)

**Returns:** `typing.Optional[dubchain.core.block.Block]`

##### `get_chain_info(self)`

Get information about the blockchain.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_pending_transactions(self)`

Get all pending transactions.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.core.transaction.Transaction]`

##### `get_transaction_by_hash(self, tx_hash)`

Get a transaction by its hash.

**Parameters:**

- `self`: Any (required)
- `tx_hash`: <class 'dubchain.crypto.hashing.Hash'> (required)

**Returns:** `typing.Optional[typing.Tuple[dubchain.core.block.Block, dubchain.core.transaction.Transaction]]`

##### `get_utxos_for_address(self, address)`

Get all UTXOs for an address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `typing.List[dubchain.core.transaction.UTXO]`

##### `initialize_genesis(self, genesis_block)`

Initialize with a genesis block.

**Parameters:**

- `self`: Any (required)
- `genesis_block`: <class 'dubchain.core.block.Block'> (required)

**Returns:** `None`

##### `mine_block(self, miner_address, max_transactions)`

Mine a new block with pending transactions.

Args:
    miner_address: Address to receive block reward
    max_transactions: Maximum number of transactions to include

Returns:
    Mined block, or None if mining failed

**Parameters:**

- `self`: Any (required)
- `miner_address`: <class 'str'> (required)
- `max_transactions`: <class 'int'> = 1000

**Returns:** `typing.Optional[dubchain.core.block.Block]`

##### `process_transaction(self, transaction)`

Process a transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)

**Returns:** `<class 'bool'>`

##### `validate_chain(self)`

Validate the entire blockchain.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `validate_transaction(self, transaction)`

Validate a transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)

**Returns:** `<class 'bool'>`

### BlockchainState

Current state of the blockchain.

#### Methods

##### `add_pending_transaction(self, transaction)`

Add a pending transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)

**Returns:** `None`

##### `add_utxo(self, utxo)`

Add a UTXO to the state.

**Parameters:**

- `self`: Any (required)
- `utxo`: <class 'dubchain.core.transaction.UTXO'> (required)

**Returns:** `None`

##### `get_balance(self, address)`

Get the balance of an address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `<class 'int'>`

##### `get_utxos_for_address(self, address)`

Get all UTXOs for an address.

**Parameters:**

- `self`: Any (required)
- `address`: <class 'str'> (required)

**Returns:** `typing.List[dubchain.core.transaction.UTXO]`

##### `remove_pending_transaction(self, transaction)`

Remove a pending transaction.

**Parameters:**

- `self`: Any (required)
- `transaction`: <class 'dubchain.core.transaction.Transaction'> (required)

**Returns:** `None`

##### `remove_utxo(self, utxo_key)`

Remove a UTXO from the state.

**Parameters:**

- `self`: Any (required)
- `utxo_key`: <class 'str'> (required)

**Returns:** `None`

##### `update_state(self, block)`

Update state with a new block.

**Parameters:**

- `self`: Any (required)
- `block`: <class 'dubchain.core.block.Block'> (required)

**Returns:** `None`

##### `update_utxo_block_height(self, utxo_key, block_height)`

Update the block height of a UTXO.

**Parameters:**

- `self`: Any (required)
- `utxo_key`: <class 'str'> (required)
- `block_height`: <class 'int'> (required)

**Returns:** `None`

## Usage Examples

```python
from dubchain.core.blockchain import *

# Create instance of Blockchain
blockchain = Blockchain()

# Call method
result = blockchain.add_block()
```
