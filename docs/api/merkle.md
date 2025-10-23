# Merkle API

**Module:** `dubchain.crypto.merkle`

## Classes

### MerkleProof

Proof of inclusion in a Merkle tree.

#### Methods

##### `verify(self)`

Verify that this proof is valid.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

### MerkleTree

Merkle tree for efficient data verification.

#### Methods

##### `add_leaf(self, item)`

Create a new Merkle tree with an additional leaf.

Args:
    item: Item to add

Returns:
    New MerkleTree with additional leaf

**Parameters:**

- `self`: Any (required)
- `item`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `MerkleTree`

##### `contains(self, item)`

Check if an item is in the tree.

**Parameters:**

- `self`: Any (required)
- `item`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `<class 'bool'>`

##### `get_all_leaves(self)`

Get all leaf hashes.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.crypto.hashing.Hash]`

##### `get_depth(self)`

Get the depth of the tree.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_leaf_count(self)`

Get the number of leaves in the tree.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_proof(self, item)`

Get a Merkle proof for an item.

Args:
    item: Item to get proof for

Returns:
    MerkleProof if item exists, None otherwise

**Parameters:**

- `self`: Any (required)
- `item`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `typing.Optional[dubchain.crypto.merkle.MerkleProof]`

##### `get_root(self)`

Get the root hash of the tree.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `remove_leaf(self, item)`

Create a new Merkle tree with a leaf removed.

Args:
    item: Item to remove

Returns:
    New MerkleTree with leaf removed

**Parameters:**

- `self`: Any (required)
- `item`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `MerkleTree`

##### `update_leaf(self, old_item, new_item)`

Create a new Merkle tree with an updated leaf.

Args:
    old_item: Item to replace
    new_item: New item

Returns:
    New MerkleTree with updated leaf

**Parameters:**

- `self`: Any (required)
- `old_item`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)
- `new_item`: typing.Union[bytes, str, dubchain.crypto.hashing.Hash] (required)

**Returns:** `MerkleTree`

##### `verify_proof(self, proof)`

Verify a Merkle proof.

**Parameters:**

- `self`: Any (required)
- `proof`: <class 'dubchain.crypto.merkle.MerkleProof'> (required)

**Returns:** `<class 'bool'>`

### SparseMerkleTree

Sparse Merkle tree for efficient updates and proofs.

This implementation uses a binary tree where each leaf represents
a possible key in a large key space (e.g., 256-bit keys).

#### Methods

##### `delete(self, key)`

Delete a leaf (set to zero).

**Parameters:**

- `self`: Any (required)
- `key`: <class 'int'> (required)

**Returns:** `None`

##### `get(self, key)`

Get a leaf value.

**Parameters:**

- `self`: Any (required)
- `key`: <class 'int'> (required)

**Returns:** `<class 'dubchain.crypto.hashing.Hash'>`

##### `get_proof(self, key)`

Get a Merkle proof for a key.

**Parameters:**

- `self`: Any (required)
- `key`: <class 'int'> (required)

**Returns:** `typing.List[dubchain.crypto.hashing.Hash]`

##### `update(self, key, value)`

Update a leaf value.

**Parameters:**

- `self`: Any (required)
- `key`: <class 'int'> (required)
- `value`: <class 'dubchain.crypto.hashing.Hash'> (required)

**Returns:** `None`

##### `verify_proof(self, key, value, proof)`

Verify a Merkle proof.

**Parameters:**

- `self`: Any (required)
- `key`: <class 'int'> (required)
- `value`: <class 'dubchain.crypto.hashing.Hash'> (required)
- `proof`: typing.List[dubchain.crypto.hashing.Hash] (required)

**Returns:** `<class 'bool'>`

## Usage Examples

```python
from dubchain.crypto.merkle import *

# Create instance of MerkleProof
merkleproof = MerkleProof()

# Call method
result = merkleproof.verify()
```
