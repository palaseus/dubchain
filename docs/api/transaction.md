# Transaction API

**Module:** `dubchain.core.transaction`

## Classes

### Transaction

A blockchain transaction.

#### Methods

##### `copy_without_signatures(self)`

Create a copy of the transaction without signatures.

**Parameters:**

- `self`: Any (required)

**Returns:** `Transaction`

##### `create_coinbase(recipient_address, amount, block_height)`

Create a coinbase transaction.

**Parameters:**

- `recipient_address`: <class 'str'> (required)
- `amount`: <class 'int'> (required)
- `block_height`: <class 'int'> (required)

**Returns:** `Transaction`

##### `create_transfer(sender_private_key, recipient_address, amount, utxos, fee)`

Create a transfer transaction.

**Parameters:**

- `sender_private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)
- `recipient_address`: <class 'str'> (required)
- `amount`: <class 'int'> (required)
- `utxos`: typing.List[dubchain.core.transaction.UTXO] (required)
- `fee`: <class 'int'> = 0

**Returns:** `Transaction`

##### `get_fee(self, utxos)`

Get the transaction fee.

**Parameters:**

- `self`: Any (required)
- `utxos`: typing.Dict[str, dubchain.core.transaction.UTXO] (required)

**Returns:** `<class 'int'>`

##### `get_hash(self)`

Get the hash of this transaction.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `get_total_input_amount(self, utxos)`

Get the total amount of all inputs.

**Parameters:**

- `self`: Any (required)
- `utxos`: typing.Dict[str, dubchain.core.transaction.UTXO] (required)

**Returns:** `<class 'int'>`

##### `get_total_output_amount(self)`

Get the total amount of all outputs.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_utxos_consumed(self)`

Get the keys of UTXOs that this transaction consumes.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[str]`

##### `get_utxos_created(self)`

Get the UTXOs that this transaction creates.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.core.transaction.UTXO]`

##### `is_valid(self, utxos)`

Check if the transaction is valid.

**Parameters:**

- `self`: Any (required)
- `utxos`: typing.Dict[str, dubchain.core.transaction.UTXO] (required)

**Returns:** `<class 'bool'>`

##### `sign_input(self, input_index, private_key)`

Sign a specific input of the transaction.

**Parameters:**

- `self`: Any (required)
- `input_index`: <class 'int'> (required)
- `private_key`: <class 'dubchain.crypto.signatures.PrivateKey'> (required)

**Returns:** `Transaction`

##### `to_bytes(self)`

Serialize transaction to bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

##### `verify_signature(self, input_index, utxo)`

Verify the signature of a specific input.

**Parameters:**

- `self`: Any (required)
- `input_index`: <class 'int'> (required)
- `utxo`: <class 'dubchain.core.transaction.UTXO'> (required)

**Returns:** `<class 'bool'>`

### TransactionInput

Input to a transaction (UTXO reference).

#### Methods

##### `get_signature_hash(self, transaction, input_index)`

Get the hash that should be signed for this input.

**Parameters:**

- `self`: Any (required)
- `transaction`: Transaction (required)
- `input_index`: <class 'int'> (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `to_bytes(self)`

Serialize transaction input to bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

### TransactionOutput

Output from a transaction (UTXO).

#### Methods

##### `to_bytes(self)`

Serialize transaction output to bytes.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bytes'>`

### TransactionType

Types of transactions.

**Inherits from:** Enum

### UTXO

Unspent Transaction Output.

#### Methods

##### `get_key(self)`

Get unique key for this UTXO.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'str'>`

##### `to_transaction_output(self)`

Convert UTXO to TransactionOutput.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.core.transaction.TransactionOutput'>`

## Usage Examples

```python
from dubchain.core.transaction import *

# Create instance of Transaction
transaction = Transaction()

# Call method
result = transaction.copy_without_signatures()
```
