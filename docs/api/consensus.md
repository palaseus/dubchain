# Consensus API

**Module:** `dubchain.core.consensus`

## Classes

### ConsensusConfig

Configuration for consensus mechanisms.

### ConsensusEngine

Main consensus engine that coordinates different consensus mechanisms.

#### Methods

##### `calculate_next_difficulty(self, blocks)`

Calculate difficulty for the next block.

**Parameters:**

- `self`: Any (required)
- `blocks`: typing.List[dubchain.core.block.Block] (required)

**Returns:** `<class 'int'>`

##### `get_consensus_info(self, blocks)`

Get information about the current consensus state.

**Parameters:**

- `self`: Any (required)
- `blocks`: typing.List[dubchain.core.block.Block] (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `mine_block(self, transactions, previous_block, utxos)`

Mine a new block with the given transactions.

Args:
    transactions: Transactions to include in the block
    previous_block: Previous block in the chain
    utxos: Current UTXO set

Returns:
    Mined block, or None if mining failed

**Parameters:**

- `self`: Any (required)
- `transactions`: typing.List[dubchain.core.transaction.Transaction] (required)
- `previous_block`: <class 'dubchain.core.block.Block'> (required)
- `utxos`: typing.Dict[str, typing.Any] (required)

**Returns:** `typing.Optional[dubchain.core.block.Block]`

##### `validate_block(self, block, previous_blocks, utxos)`

Validate a block according to consensus rules.

Args:
    block: Block to validate
    previous_blocks: List of previous blocks
    utxos: Current UTXO set

Returns:
    True if block is valid, False otherwise

**Parameters:**

- `self`: Any (required)
- `block`: <class 'dubchain.core.block.Block'> (required)
- `previous_blocks`: typing.List[dubchain.core.block.Block] (required)
- `utxos`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'bool'>`

### ProofOfWork

Proof of Work consensus mechanism.

#### Methods

##### `calculate_difficulty(self, blocks, target_block_time)`

Calculate the difficulty for the next block.

Args:
    blocks: List of recent blocks
    target_block_time: Target time between blocks in seconds

Returns:
    Calculated difficulty

**Parameters:**

- `self`: Any (required)
- `blocks`: typing.List[dubchain.core.block.Block] (required)
- `target_block_time`: typing.Optional[int] = None

**Returns:** `<class 'int'>`

##### `estimate_mining_time(self, difficulty, hashrate)`

Estimate the time to mine a block with given difficulty and hashrate.

Args:
    difficulty: Current difficulty
    hashrate: Hash rate in hashes per second

Returns:
    Estimated time in seconds

**Parameters:**

- `self`: Any (required)
- `difficulty`: <class 'int'> (required)
- `hashrate`: <class 'float'> (required)

**Returns:** `<class 'float'>`

##### `get_difficulty_target(self, difficulty)`

Get the target hash for a given difficulty.

**Parameters:**

- `self`: Any (required)
- `difficulty`: <class 'int'> (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `mine_block(self, block_header, max_nonce)`

Mine a block by finding a valid nonce.

Args:
    block_header: Block header to mine
    max_nonce: Maximum nonce to try

Returns:
    Block header with valid nonce, or None if mining failed

**Parameters:**

- `self`: Any (required)
- `block_header`: <class 'dubchain.core.block.BlockHeader'> (required)
- `max_nonce`: <class 'int'> = 4294967295

**Returns:** `typing.Optional[dubchain.core.block.BlockHeader]`

##### `verify_block(self, block)`

Verify that a block meets the proof of work requirements.

**Parameters:**

- `self`: Any (required)
- `block`: <class 'dubchain.core.block.Block'> (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.core.consensus import *

# Create instance of ConsensusConfig
consensusconfig = ConsensusConfig()

```
