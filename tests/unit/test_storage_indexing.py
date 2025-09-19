"""Tests for storage indexing module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.storage.indexing import (
    BTreeIndex,
    HashIndex,
    Index,
    IndexConfig,
    IndexError,
    IndexManager,
    IndexNotFoundError,
    IndexType,
)


class TestIndexType:
    """Test IndexType enum."""

    def test_index_type_values(self):
        """Test index type values."""
        assert IndexType.B_TREE.value == "b_tree"
        assert IndexType.HASH.value == "hash"
        assert IndexType.FULL_TEXT.value == "full_text"
        assert IndexType.COMPOSITE.value == "composite"


class TestIndexConfig:
    """Test IndexConfig functionality."""

    def test_index_config_creation(self):
        """Test creating index config."""
        config = IndexConfig(
            name="test_index",
            table="test_table",
            columns=["id", "name"],
            index_type=IndexType.B_TREE,
            unique=True,
        )

        assert config.name == "test_index"
        assert config.table == "test_table"
        assert config.columns == ["id", "name"]
        assert config.index_type == IndexType.B_TREE
        assert config.unique is True

    def test_index_config_defaults(self):
        """Test index config defaults."""
        config = IndexConfig(name="test_index", table="test_table", columns=["id"])

        assert config.name == "test_index"
        assert config.table == "test_table"
        assert config.columns == ["id"]
        assert config.index_type == IndexType.B_TREE
        assert config.unique is False


class TestIndex:
    """Test Index base class."""

    def test_index_creation(self):
        """Test creating index."""
        config = IndexConfig(name="test_index", table="test_table", columns=["id"])

        # Create concrete implementation
        class TestIndex(Index):
            def create(self, backend):
                pass

            def drop(self, backend):
                pass

            def rebuild(self, backend):
                pass

            def analyze(self, backend):
                return self.stats

            def is_used_by_query(self, query):
                return True

            def search(self, value):
                return []

            def insert(self, key, value):
                pass

            def delete(self, key):
                pass

            def update(self, old_key, new_key, value):
                pass

        index = TestIndex(config)
        assert index.config == config
        assert index.name == "test_index"


class TestBTreeIndex:
    """Test BTreeIndex functionality."""

    @pytest.fixture
    def btree_config(self):
        """Fixture for BTree index configuration."""
        return IndexConfig(
            name="btree_index",
            table="test_table",
            columns=["id"],
            index_type=IndexType.B_TREE,
        )

    @pytest.fixture
    def btree_index(self, btree_config):
        """Fixture for BTree index."""
        return BTreeIndex(btree_config)

    def test_btree_index_creation(self, btree_config):
        """Test creating BTree index."""
        index = BTreeIndex(btree_config)

        assert index.config == btree_config
        assert index.name == "btree_index"
        assert index._tree == {}

    def test_btree_index_insert(self, btree_index):
        """Test BTree index insert."""
        btree_index.insert("key1", "value1")
        btree_index.insert("key2", "value2")

        assert "key1" in btree_index._tree
        assert "key2" in btree_index._tree

    def test_btree_index_search(self, btree_index):
        """Test BTree index search."""
        btree_index.insert("key1", "value1")

        result = btree_index.search("key1")
        assert result == ["value1"]

        result = btree_index.search("nonexistent")
        assert result == []

    def test_btree_index_delete(self, btree_index):
        """Test BTree index delete."""
        btree_index.insert("key1", "value1")
        assert "key1" in btree_index._tree

        btree_index.delete("key1")
        assert "key1" not in btree_index._tree

    def test_btree_index_update(self, btree_index):
        """Test BTree index update."""
        btree_index.insert("old_key", "value1")

        btree_index.update("old_key", "new_key", "value1")

        assert "old_key" not in btree_index._tree
        assert "new_key" in btree_index._tree


class TestHashIndex:
    """Test HashIndex functionality."""

    @pytest.fixture
    def hash_config(self):
        """Fixture for Hash index configuration."""
        return IndexConfig(
            name="hash_index",
            table="test_table",
            columns=["id"],
            index_type=IndexType.HASH,
        )

    @pytest.fixture
    def hash_index(self, hash_config):
        """Fixture for Hash index."""
        return HashIndex(hash_config)

    def test_hash_index_creation(self, hash_config):
        """Test creating Hash index."""
        index = HashIndex(hash_config)

        assert index.config == hash_config
        assert index.name == "hash_index"
        assert index._hash_table == {}

    def test_hash_index_insert(self, hash_index):
        """Test Hash index insert."""
        hash_index.insert("key1", "value1")
        hash_index.insert("key2", "value2")

        assert len(hash_index._hash_table) > 0

    def test_hash_index_search(self, hash_index):
        """Test Hash index search."""
        hash_index.insert("key1", "value1")

        result = hash_index.search("key1")
        assert result == ["value1"]

        result = hash_index.search("nonexistent")
        assert result == []

    def test_hash_index_delete(self, hash_index):
        """Test Hash index delete."""
        hash_index.insert("key1", "value1")

        hash_index.delete("key1")

        result = hash_index.search("key1")
        assert result == []

    def test_hash_index_update(self, hash_index):
        """Test Hash index update."""
        hash_index.insert("old_key", "value1")

        hash_index.update("old_key", "new_key", "value1")

        old_result = hash_index.search("old_key")
        new_result = hash_index.search("new_key")

        assert old_result == []
        assert new_result == ["value1"]


class TestIndexManager:
    """Test IndexManager functionality."""

    @pytest.fixture
    def index_manager(self):
        """Fixture for index manager."""
        from unittest.mock import Mock

        mock_backend = Mock()
        return IndexManager(mock_backend)

    def test_index_manager_creation(self):
        """Test creating index manager."""
        from unittest.mock import Mock

        mock_backend = Mock()
        manager = IndexManager(mock_backend)

        assert isinstance(manager._indexes, dict)
        assert manager._lock is not None

    def test_create_index(self, index_manager):
        """Test creating index."""
        config = IndexConfig(
            name="test_index",
            table="test_table",
            columns=["id"],
            index_type=IndexType.B_TREE,
        )

        index = index_manager.create_index(config)

        assert index is not None
        assert index.name == "test_index"
        assert "test_index" in index_manager._indexes

    def test_get_index(self, index_manager):
        """Test getting index."""
        config = IndexConfig(
            name="test_index",
            table="test_table",
            columns=["id"],
            index_type=IndexType.B_TREE,
        )

        created_index = index_manager.create_index(config)
        retrieved_index = index_manager.get_index("test_index")

        assert created_index is retrieved_index

    def test_get_nonexistent_index(self, index_manager):
        """Test getting nonexistent index."""
        with pytest.raises(IndexNotFoundError):
            index_manager.get_index("nonexistent")

    def test_drop_index(self, index_manager):
        """Test dropping index."""
        config = IndexConfig(
            name="test_index",
            table="test_table",
            columns=["id"],
            index_type=IndexType.B_TREE,
        )

        index_manager.create_index(config)
        assert "test_index" in index_manager._indexes

        index_manager.drop_index("test_index")
        assert "test_index" not in index_manager._indexes

    def test_list_indexes(self, index_manager):
        """Test listing indexes."""
        config1 = IndexConfig(
            name="index1", table="table1", columns=["id"], index_type=IndexType.B_TREE
        )

        config2 = IndexConfig(
            name="index2", table="table2", columns=["name"], index_type=IndexType.HASH
        )

        index_manager.create_index(config1)
        index_manager.create_index(config2)

        indexes = index_manager.list_indexes()
        assert len(indexes) == 2
        assert "index1" in indexes
        assert "index2" in indexes

    def test_rebuild_index(self, index_manager):
        """Test rebuilding index."""
        config = IndexConfig(
            name="test_index",
            table="test_table",
            columns=["id"],
            index_type=IndexType.B_TREE,
        )

        index = index_manager.create_index(config)
        index.insert("key1", "value1")

        # Rebuild should clear and recreate
        index_manager.rebuild_index("test_index")

        # Index should still exist but be empty
        rebuilt_index = index_manager.get_index("test_index")
        assert rebuilt_index is not None

    def test_get_table_indexes(self, index_manager):
        """Test getting indexes for a table."""
        config1 = IndexConfig(
            name="index1", table="table1", columns=["id"], index_type=IndexType.B_TREE
        )

        config2 = IndexConfig(
            name="index2", table="table1", columns=["name"], index_type=IndexType.HASH
        )

        config3 = IndexConfig(
            name="index3", table="table2", columns=["id"], index_type=IndexType.B_TREE
        )

        index_manager.create_index(config1)
        index_manager.create_index(config2)
        index_manager.create_index(config3)

        table1_indexes = index_manager.get_table_indexes("table1")
        assert len(table1_indexes) == 2

        table2_indexes = index_manager.get_table_indexes("table2")
        assert len(table2_indexes) == 1
