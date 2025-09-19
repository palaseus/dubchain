"""
Merkle tree implementation for efficient data verification.

This module provides Merkle tree functionality for blockchain transaction verification
and other cryptographic proofs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .hashing import Hash, SHA256Hasher


@dataclass(frozen=True)
class MerkleProof:
    """Proof of inclusion in a Merkle tree."""

    leaf_hash: Hash
    path: List[Tuple[Hash, bool]]  # (hash, is_left)
    root_hash: Hash

    def verify(self) -> bool:
        """Verify that this proof is valid."""
        current_hash = self.leaf_hash

        for sibling_hash, is_left in self.path:
            if is_left:
                # Sibling is left, current is right
                combined = sibling_hash.value + current_hash.value
            else:
                # Sibling is right, current is left
                combined = current_hash.value + sibling_hash.value

            current_hash = SHA256Hasher.hash(combined)

        return current_hash == self.root_hash


class MerkleTree:
    """Merkle tree for efficient data verification."""

    def __init__(self, items: List[Union[bytes, str, Hash]]):
        """
        Initialize a Merkle tree with the given items.

        Args:
            items: List of items to include in the tree
        """
        if not items:
            raise ValueError("Merkle tree cannot be empty")

        # Convert all items to hashes
        self.leaves = [self._to_hash(item) for item in items]
        self.tree = self._build_tree()
        self.root = self.tree[-1][0] if self.tree else Hash.zero()

    def _to_hash(self, item: Union[bytes, str, Hash]) -> Hash:
        """Convert an item to a hash."""
        if isinstance(item, Hash):
            return item
        elif isinstance(item, str):
            return SHA256Hasher.hash(item)
        else:
            return SHA256Hasher.hash(item)

    def _build_tree(self) -> List[List[Hash]]:
        """Build the Merkle tree structure."""
        if not self.leaves:
            return []

        # Start with leaves
        current_level = self.leaves.copy()
        tree = [current_level]

        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []

            # Process pairs of nodes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                # Combine and hash
                combined = left.value + right.value
                parent_hash = SHA256Hasher.hash(combined)
                next_level.append(parent_hash)

            tree.append(next_level)
            current_level = next_level

        return tree

    def get_root(self) -> Hash:
        """Get the root hash of the tree."""
        return self.root

    def get_proof(self, item: Union[bytes, str, Hash]) -> Optional[MerkleProof]:
        """
        Get a Merkle proof for an item.

        Args:
            item: Item to get proof for

        Returns:
            MerkleProof if item exists, None otherwise
        """
        item_hash = self._to_hash(item)

        # Find the leaf index
        try:
            leaf_index = self.leaves.index(item_hash)
        except ValueError:
            return None

        # Build the proof path
        path = []
        current_index = leaf_index

        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]

            # Determine if current node is left or right child
            is_left = current_index % 2 == 0

            # Get sibling
            sibling_index = current_index + 1 if is_left else current_index - 1

            if sibling_index < len(current_level):
                sibling_hash = current_level[sibling_index]
                # Store whether the sibling is on the left (not the current node)
                sibling_is_left = not is_left
                path.append((sibling_hash, sibling_is_left))

            # Move to parent level
            current_index //= 2

        return MerkleProof(item_hash, path, self.root)

    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify a Merkle proof."""
        if proof is None:
            return False
        return proof.verify()

    def contains(self, item: Union[bytes, str, Hash]) -> bool:
        """Check if an item is in the tree."""
        item_hash = self._to_hash(item)
        return item_hash in self.leaves

    def get_leaf_count(self) -> int:
        """Get the number of leaves in the tree."""
        return len(self.leaves)

    def get_depth(self) -> int:
        """Get the depth of the tree."""
        return len(self.tree) - 1

    def get_all_leaves(self) -> List[Hash]:
        """Get all leaf hashes."""
        return self.leaves.copy()

    def update_leaf(
        self, old_item: Union[bytes, str, Hash], new_item: Union[bytes, str, Hash]
    ) -> "MerkleTree":
        """
        Create a new Merkle tree with an updated leaf.

        Args:
            old_item: Item to replace
            new_item: New item

        Returns:
            New MerkleTree with updated leaf
        """
        old_hash = self._to_hash(old_item)
        new_hash = self._to_hash(new_item)

        # Find and replace the leaf
        new_leaves = self.leaves.copy()
        try:
            index = new_leaves.index(old_hash)
            new_leaves[index] = new_hash
        except ValueError:
            raise ValueError("Old item not found in tree")

        return MerkleTree(new_leaves)

    def add_leaf(self, item: Union[bytes, str, Hash]) -> "MerkleTree":
        """
        Create a new Merkle tree with an additional leaf.

        Args:
            item: Item to add

        Returns:
            New MerkleTree with additional leaf
        """
        new_leaves = self.leaves + [self._to_hash(item)]
        return MerkleTree(new_leaves)

    def remove_leaf(self, item: Union[bytes, str, Hash]) -> "MerkleTree":
        """
        Create a new Merkle tree with a leaf removed.

        Args:
            item: Item to remove

        Returns:
            New MerkleTree with leaf removed
        """
        item_hash = self._to_hash(item)
        new_leaves = [leaf for leaf in self.leaves if leaf != item_hash]

        if not new_leaves:
            raise ValueError("Cannot remove the last leaf from tree")

        return MerkleTree(new_leaves)

    def __str__(self) -> str:
        return f"MerkleTree(root={self.root}, leaves={len(self.leaves)})"

    def __repr__(self) -> str:
        return f"MerkleTree({len(self.leaves)} leaves)"


class SparseMerkleTree:
    """
    Sparse Merkle tree for efficient updates and proofs.

    This implementation uses a binary tree where each leaf represents
    a possible key in a large key space (e.g., 256-bit keys).
    """

    def __init__(self, depth: int = 256):
        """
        Initialize a sparse Merkle tree.

        Args:
            depth: Depth of the tree (default 256 for 256-bit keys)
        """
        self.depth = depth
        self.zero_hashes = self._compute_zero_hashes()
        self.root = self.zero_hashes[0]
        self.leaves: dict[int, Hash] = {}  # key -> hash mapping

    def _compute_zero_hashes(self) -> List[Hash]:
        """Compute the zero hashes for each level."""
        zero_hashes = [Hash.zero()]

        for i in range(self.depth):
            # Hash two zero hashes together
            combined = zero_hashes[-1].value + zero_hashes[-1].value
            zero_hashes.append(SHA256Hasher.hash(combined))

        return zero_hashes

    def _key_to_path(self, key: int) -> List[bool]:
        """Convert a key to a path (list of left/right decisions)."""
        path = []
        for _ in range(self.depth):
            path.append(key & 1 == 1)
            key >>= 1
        return path

    def _path_to_key(self, path: List[bool]) -> int:
        """Convert a path to a key."""
        key = 0
        for i, is_right in enumerate(path):
            if is_right:
                key |= 1 << i
        return key

    def update(self, key: int, value: Hash) -> None:
        """Update a leaf value."""
        self.leaves[key] = value
        self.root = self._compute_root()

    def delete(self, key: int) -> None:
        """Delete a leaf (set to zero)."""
        if key in self.leaves:
            del self.leaves[key]
        self.root = self._compute_root()

    def get(self, key: int) -> Hash:
        """Get a leaf value."""
        return self.leaves.get(key, Hash.zero())

    def _compute_root(self) -> Hash:
        """Compute the current root hash."""
        if not self.leaves:
            return self.zero_hashes[0]

        # Use a recursive approach to compute the root
        return self._compute_subtree_root(0, self.depth)

    def _compute_subtree_root(self, key: int, depth: int) -> Hash:
        """Compute the root of a subtree."""
        if depth == 0:
            return self.leaves.get(key, Hash.zero())

        left_key = key
        right_key = key | (1 << (depth - 1))

        left_hash = self._compute_subtree_root(left_key, depth - 1)
        right_hash = self._compute_subtree_root(right_key, depth - 1)

        combined = left_hash.value + right_hash.value
        return SHA256Hasher.hash(combined)

    def get_proof(self, key: int) -> List[Hash]:
        """Get a Merkle proof for a key."""
        path = self._key_to_path(key)
        proof = []

        current_key = key
        for level in range(self.depth):
            is_right = path[level]
            sibling_key = current_key ^ (1 << level)

            # Get sibling hash
            if level == 0:
                sibling_hash = self.leaves.get(sibling_key, Hash.zero())
            else:
                sibling_hash = self._compute_subtree_root(sibling_key, level)

            proof.append(sibling_hash)
            current_key >>= 1

        return proof

    def verify_proof(self, key: int, value: Hash, proof: List[Hash]) -> bool:
        """Verify a Merkle proof."""
        if len(proof) != self.depth:
            return False

        path = self._key_to_path(key)
        current_hash = value

        for level in range(self.depth):
            is_right = path[level]
            sibling_hash = proof[level]

            if is_right:
                combined = sibling_hash.value + current_hash.value
            else:
                combined = current_hash.value + sibling_hash.value

            current_hash = SHA256Hasher.hash(combined)

        return current_hash == self.root
