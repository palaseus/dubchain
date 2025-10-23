"""
Unit tests for Merkle tree implementation.
"""

import logging

logger = logging.getLogger(__name__)
import pytest

from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.crypto.merkle import MerkleProof, MerkleTree, SparseMerkleTree


class TestMerkleTree:
    """Test the MerkleTree class."""

    def test_merkle_tree_creation(self):
        """Test creating a Merkle tree."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)
        assert isinstance(tree, MerkleTree)
        assert tree.get_leaf_count() == 4

    def test_merkle_tree_empty(self):
        """Test that empty tree raises ValueError."""
        with pytest.raises(ValueError, match="Merkle tree cannot be empty"):
            MerkleTree([])

    def test_merkle_tree_single_item(self):
        """Test Merkle tree with single item."""
        items = ["single_item"]
        tree = MerkleTree(items)
        assert tree.get_leaf_count() == 1
        assert tree.get_depth() == 0

    def test_merkle_tree_root(self):
        """Test getting the root hash."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)
        root = tree.get_root()
        assert isinstance(root, Hash)
        assert len(root.value) == 32

    def test_merkle_tree_contains(self):
        """Test checking if item is in tree."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)

        assert tree.contains("item1")
        assert tree.contains("item2")
        assert not tree.contains("item5")

    def test_merkle_tree_get_proof(self):
        """Test getting a Merkle proof."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)

        proof = tree.get_proof("item1")
        assert isinstance(proof, MerkleProof)
        assert proof.leaf_hash == SHA256Hasher.hash("item1")
        assert proof.root_hash == tree.get_root()

    def test_merkle_tree_get_proof_nonexistent(self):
        """Test getting proof for non-existent item."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)

        proof = tree.get_proof("nonexistent")
        assert proof is None

    def test_merkle_tree_verify_proof(self):
        """Test verifying a Merkle proof."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)

        proof = tree.get_proof("item1")
        assert tree.verify_proof(proof)

    def test_merkle_tree_verify_invalid_proof(self):
        """Test verifying an invalid Merkle proof."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)

        # Create invalid proof
        invalid_proof = MerkleProof(
            leaf_hash=SHA256Hasher.hash("item1"),
            path=[(SHA256Hasher.hash("wrong"), True)],
            root_hash=tree.get_root(),
        )

        assert not tree.verify_proof(invalid_proof)

    def test_merkle_tree_update_leaf(self):
        """Test updating a leaf in the tree."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)
        original_root = tree.get_root()

        new_tree = tree.update_leaf("item1", "new_item1")
        assert new_tree.get_root() != original_root
        assert new_tree.contains("new_item1")
        assert not new_tree.contains("item1")

    def test_merkle_tree_add_leaf(self):
        """Test adding a leaf to the tree."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)
        original_root = tree.get_root()

        new_tree = tree.add_leaf("item5")
        assert new_tree.get_root() != original_root
        assert new_tree.contains("item5")
        assert new_tree.get_leaf_count() == 5

    def test_merkle_tree_remove_leaf(self):
        """Test removing a leaf from the tree."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)
        original_root = tree.get_root()

        new_tree = tree.remove_leaf("item1")
        assert new_tree.get_root() != original_root
        assert not new_tree.contains("item1")
        assert new_tree.get_leaf_count() == 3

    def test_merkle_tree_remove_last_leaf(self):
        """Test that removing the last leaf raises ValueError."""
        items = ["single_item"]
        tree = MerkleTree(items)

        with pytest.raises(ValueError, match="Cannot remove the last leaf from tree"):
            tree.remove_leaf("single_item")

    def test_merkle_tree_odd_number_of_leaves(self):
        """Test Merkle tree with odd number of leaves."""
        items = ["item1", "item2", "item3"]
        tree = MerkleTree(items)
        assert tree.get_leaf_count() == 3
        assert tree.get_depth() == 2  # Should pad to next power of 2

    def test_merkle_tree_different_data_types(self):
        """Test Merkle tree with different data types."""
        items = ["string", b"bytes", Hash.from_hex("00" * 32)]
        tree = MerkleTree(items)
        assert tree.get_leaf_count() == 3

    def test_merkle_tree_string_representation(self):
        """Test string representation of Merkle tree."""
        items = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(items)

        str_repr = str(tree)
        assert "MerkleTree" in str_repr
        assert "root=" in str_repr
        assert "leaves=4" in str_repr


class TestMerkleProof:
    """Test the MerkleProof class."""

    def test_merkle_proof_creation(self):
        """Test creating a Merkle proof."""
        leaf_hash = SHA256Hasher.hash("item1")
        path = [(SHA256Hasher.hash("sibling"), True)]
        root_hash = SHA256Hasher.hash("root")

        proof = MerkleProof(leaf_hash, path, root_hash)
        assert proof.leaf_hash == leaf_hash
        assert proof.path == path
        assert proof.root_hash == root_hash

    def test_merkle_proof_verify(self):
        """Test verifying a Merkle proof."""
        # Create a simple tree and get a proof
        items = ["item1", "item2"]
        tree = MerkleTree(items)
        proof = tree.get_proof("item1")

        assert proof.verify()

    def test_merkle_proof_verify_invalid(self):
        """Test verifying an invalid Merkle proof."""
        leaf_hash = SHA256Hasher.hash("item1")
        path = [(SHA256Hasher.hash("wrong_sibling"), True)]
        root_hash = SHA256Hasher.hash("wrong_root")

        proof = MerkleProof(leaf_hash, path, root_hash)
        assert not proof.verify()


class TestSparseMerkleTree:
    """Test the SparseMerkleTree class."""

    def test_sparse_merkle_tree_creation(self):
        """Test creating a sparse Merkle tree."""
        tree = SparseMerkleTree(depth=8)
        assert tree.depth == 8
        assert isinstance(tree.root, Hash)

    def test_sparse_merkle_tree_update(self):
        """Test updating a leaf in sparse Merkle tree."""
        tree = SparseMerkleTree(depth=8)
        original_root = tree.root

        key = 42
        value = SHA256Hasher.hash("value")
        tree.update(key, value)

        assert tree.root != original_root
        assert tree.get(key) == value

    def test_sparse_merkle_tree_delete(self):
        """Test deleting a leaf from sparse Merkle tree."""
        tree = SparseMerkleTree(depth=8)
        key = 42
        value = SHA256Hasher.hash("value")

        tree.update(key, value)
        assert tree.get(key) == value

        tree.delete(key)
        assert tree.get(key) == Hash.zero()

    def test_sparse_merkle_tree_get_proof(self):
        """Test getting a proof from sparse Merkle tree."""
        tree = SparseMerkleTree(depth=8)
        key = 42
        value = SHA256Hasher.hash("value")

        tree.update(key, value)
        proof = tree.get_proof(key)

        assert isinstance(proof, list)
        assert len(proof) == tree.depth

    def test_sparse_merkle_tree_verify_proof(self):
        """Test verifying a proof from sparse Merkle tree."""
        tree = SparseMerkleTree(depth=8)
        key = 42
        value = SHA256Hasher.hash("value")

        tree.update(key, value)
        proof = tree.get_proof(key)

        assert tree.verify_proof(key, value, proof)

    def test_sparse_merkle_tree_verify_invalid_proof(self):
        """Test verifying an invalid proof from sparse Merkle tree."""
        tree = SparseMerkleTree(depth=8)
        key = 42
        value = SHA256Hasher.hash("value")

        tree.update(key, value)
        proof = tree.get_proof(key)

        # Verify with wrong value
        wrong_value = SHA256Hasher.hash("wrong")
        assert not tree.verify_proof(key, wrong_value, proof)

    def test_sparse_merkle_tree_key_to_path(self):
        """Test converting key to path."""
        tree = SparseMerkleTree(depth=4)
        key = 5  # Binary: 0101

        path = tree._key_to_path(key)
        expected_path = [True, False, True, False]  # Reversed binary
        assert path == expected_path

    def test_sparse_merkle_tree_path_to_key(self):
        """Test converting path to key."""
        tree = SparseMerkleTree(depth=4)
        path = [True, False, True, False]  # Reversed binary of 5

        key = tree._path_to_key(path)
        assert key == 5


class TestMerkleTreeIntegration:
    """Integration tests for Merkle tree functions."""

    def test_merkle_tree_large_dataset(self):
        """Test Merkle tree with large dataset."""
        items = [f"item_{i}" for i in range(100)]  # Reduced for faster testing
        tree = MerkleTree(items)

        assert tree.get_leaf_count() == 100
        assert tree.get_depth() > 0

        # Test proof for middle item
        proof = tree.get_proof("item_50")  # Adjusted for smaller dataset
        assert proof is not None
        assert tree.verify_proof(proof)

    def test_merkle_tree_consistency(self):
        """Test that same items produce same root."""
        items = ["item1", "item2", "item3", "item4"]
        tree1 = MerkleTree(items)
        tree2 = MerkleTree(items)

        assert tree1.get_root() == tree2.get_root()

    def test_merkle_tree_different_order(self):
        """Test that different order produces different root."""
        items1 = ["item1", "item2", "item3", "item4"]
        items2 = ["item4", "item3", "item2", "item1"]

        tree1 = MerkleTree(items1)
        tree2 = MerkleTree(items2)

        assert tree1.get_root() != tree2.get_root()

    def test_sparse_merkle_tree_multiple_updates(self):
        """Test sparse Merkle tree with multiple updates."""
        tree = SparseMerkleTree(depth=8)

        # Update multiple keys
        for i in range(10):
            key = i * 100
            value = SHA256Hasher.hash(f"value_{i}")
            tree.update(key, value)

        # Verify all values
        for i in range(10):
            key = i * 100
            expected_value = SHA256Hasher.hash(f"value_{i}")
            assert tree.get(key) == expected_value

    def test_merkle_tree_proof_verification_edge_cases(self):
        """Test Merkle proof verification with edge cases."""
        # Test with single item
        items = ["single"]
        tree = MerkleTree(items)
        proof = tree.get_proof("single")
        assert proof is not None
        assert proof.verify()

        # Test with two items
        items = ["item1", "item2"]
        tree = MerkleTree(items)
        proof1 = tree.get_proof("item1")
        proof2 = tree.get_proof("item2")

        assert proof1 is not None
        assert proof2 is not None
        assert proof1.verify()
        assert proof2.verify()
