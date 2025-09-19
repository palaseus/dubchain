"""Indexing system for DubChain database.

This module provides advanced indexing capabilities including B-tree indexes,
hash indexes, full-text search indexes, and composite indexes for optimal
query performance.
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .database import DatabaseBackend, QueryResult


class IndexError(Exception):
    """Base exception for indexing operations."""

    pass


class IndexNotFoundError(IndexError):
    """Index not found error."""

    pass


class IndexCorruptionError(IndexError):
    """Index corruption error."""

    pass


class IndexType(Enum):
    """Types of database indexes."""

    B_TREE = "b_tree"
    HASH = "hash"
    FULL_TEXT = "full_text"
    COMPOSITE = "composite"
    UNIQUE = "unique"
    PARTIAL = "partial"


@dataclass
class IndexConfig:
    """Index configuration."""

    name: str
    table: str
    columns: List[str]
    index_type: IndexType = IndexType.B_TREE
    unique: bool = False
    partial_condition: Optional[str] = None
    fill_factor: int = 100
    include_columns: Optional[List[str]] = None
    where_clause: Optional[str] = None

    # Performance settings
    cache_size: int = 1000
    enable_compression: bool = True
    enable_statistics: bool = True

    # Maintenance settings
    auto_rebuild: bool = True
    rebuild_threshold: float = 0.1  # 10% fragmentation
    maintenance_interval: int = 3600  # seconds


@dataclass
class IndexStats:
    """Index statistics."""

    name: str
    table: str
    size_bytes: int = 0
    row_count: int = 0
    fragmentation_percent: float = 0.0
    last_used: Optional[float] = None
    usage_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    maintenance_count: int = 0
    last_maintenance: Optional[float] = None


@dataclass
class IndexUsage:
    """Index usage information."""

    index_name: str
    query_count: int = 0
    last_used: Optional[float] = None
    average_selectivity: float = 0.0
    is_used: bool = True


class Index(ABC):
    """Abstract index interface."""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.stats = IndexStats(name=config.name, table=config.table)
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Get index name."""
        return self.config.name

    @abstractmethod
    def create(self, backend: DatabaseBackend) -> None:
        """Create the index."""
        pass

    @abstractmethod
    def drop(self, backend: DatabaseBackend) -> None:
        """Drop the index."""
        pass

    @abstractmethod
    def rebuild(self, backend: DatabaseBackend) -> None:
        """Rebuild the index."""
        pass

    @abstractmethod
    def analyze(self, backend: DatabaseBackend) -> IndexStats:
        """Analyze index statistics."""
        pass

    @abstractmethod
    def is_used_by_query(self, query: str) -> bool:
        """Check if index is used by a query."""
        pass

    @abstractmethod
    def insert(self, key: Any, value: Any) -> None:
        """Insert a key-value pair into the index."""
        pass

    @abstractmethod
    def search(self, key: Any) -> List[Any]:
        """Search for values by key in the index."""
        pass

    @abstractmethod
    def delete(self, key: Any) -> bool:
        """Delete a key from the index."""
        pass

    @abstractmethod
    def update(self, old_key: Any, new_key: Any, value: Any) -> None:
        """Update a key in the index."""
        pass


class BTreeIndex(Index):
    """B-tree index implementation."""

    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self._tree = {}  # Simple in-memory tree structure for testing

    def create(self, backend: DatabaseBackend) -> None:
        """Create B-tree index."""
        with self._lock:
            try:
                columns_str = ", ".join(self.config.columns)
                unique_str = "UNIQUE " if self.config.unique else ""
                where_str = (
                    f" WHERE {self.config.where_clause}"
                    if self.config.where_clause
                    else ""
                )

                create_sql = f"""
                CREATE {unique_str}INDEX IF NOT EXISTS {self.config.name}
                ON {self.config.table} ({columns_str}){where_str}
                """

                backend.execute_query(create_sql)
                self._logger.info(f"Created B-tree index: {self.config.name}")

            except Exception as e:
                raise IndexError(
                    f"Failed to create B-tree index {self.config.name}: {e}"
                )

    def drop(self, backend: DatabaseBackend) -> None:
        """Drop B-tree index."""
        with self._lock:
            try:
                drop_sql = f"DROP INDEX IF EXISTS {self.config.name}"
                backend.execute_query(drop_sql)
                self._logger.info(f"Dropped B-tree index: {self.config.name}")

            except Exception as e:
                raise IndexError(f"Failed to drop B-tree index {self.config.name}: {e}")

    def rebuild(self, backend: DatabaseBackend) -> None:
        """Rebuild B-tree index."""
        with self._lock:
            try:
                # Drop and recreate index
                self.drop(backend)
                self.create(backend)

                self.stats.maintenance_count += 1
                self.stats.last_maintenance = time.time()

                self._logger.info(f"Rebuilt B-tree index: {self.config.name}")

            except Exception as e:
                raise IndexError(
                    f"Failed to rebuild B-tree index {self.config.name}: {e}"
                )

    def analyze(self, backend: DatabaseBackend) -> IndexStats:
        """Analyze B-tree index statistics."""
        with self._lock:
            try:
                # Get index information
                info_sql = f"PRAGMA index_info({self.config.name})"
                result = backend.execute_query(info_sql)

                # Get index size and usage
                stats_sql = f"PRAGMA index_list({self.config.table})"
                stats_result = backend.execute_query(stats_sql)

                # Update statistics
                self.stats.row_count = len(result.rows) if result.rows else 0
                self.stats.last_used = time.time()

                return self.stats

            except Exception as e:
                raise IndexError(
                    f"Failed to analyze B-tree index {self.config.name}: {e}"
                )

    def is_used_by_query(self, query: str) -> bool:
        """Check if B-tree index is used by a query."""
        query_upper = query.upper()

        # Check if query references the indexed columns
        for column in self.config.columns:
            if column.upper() in query_upper:
                return True

        return False

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key-value pair into the B-tree index."""
        self._tree[key] = value
        self.stats.usage_count += 1
        self.stats.last_used = time.time()

    def search(self, key: Any) -> List[Any]:
        """Search for values by key in the B-tree index."""
        self.stats.usage_count += 1
        self.stats.last_used = time.time()
        if key in self._tree:
            return [self._tree[key]]
        return []

    def delete(self, key: Any) -> bool:
        """Delete a key from the B-tree index."""
        self.stats.usage_count += 1
        self.stats.last_used = time.time()
        if key in self._tree:
            del self._tree[key]
            return True
        return False

    def update(self, old_key: Any, new_key: Any, value: Any) -> None:
        """Update a key in the B-tree index."""
        if old_key in self._tree:
            del self._tree[old_key]
        self._tree[new_key] = value
        self.stats.usage_count += 1
        self.stats.last_used = time.time()


class HashIndex(Index):
    """Hash index implementation."""

    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self._hash_table = {}  # Simple in-memory hash table for testing

    def create(self, backend: DatabaseBackend) -> None:
        """Create hash index."""
        with self._lock:
            try:
                # SQLite doesn't have native hash indexes, so we create a B-tree
                # with a simple hash-like approach using the columns directly
                columns_str = ", ".join(self.config.columns)

                create_sql = f"""
                CREATE INDEX IF NOT EXISTS {self.config.name}
                ON {self.config.table} ({columns_str})
                """

                backend.execute_query(create_sql)
                self._logger.info(f"Created hash index: {self.config.name}")

            except Exception as e:
                raise IndexError(f"Failed to create hash index {self.config.name}: {e}")

    def drop(self, backend: DatabaseBackend) -> None:
        """Drop hash index."""
        with self._lock:
            try:
                drop_sql = f"DROP INDEX IF EXISTS {self.config.name}"
                backend.execute_query(drop_sql)
                self._logger.info(f"Dropped hash index: {self.config.name}")

            except Exception as e:
                raise IndexError(f"Failed to drop hash index {self.config.name}: {e}")

    def rebuild(self, backend: DatabaseBackend) -> None:
        """Rebuild hash index."""
        with self._lock:
            try:
                self.drop(backend)
                self.create(backend)

                self.stats.maintenance_count += 1
                self.stats.last_maintenance = time.time()

                self._logger.info(f"Rebuilt hash index: {self.config.name}")

            except Exception as e:
                raise IndexError(
                    f"Failed to rebuild hash index {self.config.name}: {e}"
                )

    def analyze(self, backend: DatabaseBackend) -> IndexStats:
        """Analyze hash index statistics."""
        with self._lock:
            try:
                # Similar to B-tree analysis
                info_sql = f"PRAGMA index_info({self.config.name})"
                result = backend.execute_query(info_sql)

                self.stats.row_count = len(result.rows) if result.rows else 0
                self.stats.last_used = time.time()

                return self.stats

            except Exception as e:
                raise IndexError(
                    f"Failed to analyze hash index {self.config.name}: {e}"
                )

    def is_used_by_query(self, query: str) -> bool:
        """Check if hash index is used by a query."""
        query_upper = query.upper()

        for column in self.config.columns:
            if column.upper() in query_upper:
                return True

        return False

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key-value pair into the hash index."""
        self._hash_table[key] = value
        self.stats.usage_count += 1
        self.stats.last_used = time.time()

    def search(self, key: Any) -> List[Any]:
        """Search for values by key in the hash index."""
        self.stats.usage_count += 1
        self.stats.last_used = time.time()
        if key in self._hash_table:
            return [self._hash_table[key]]
        return []

    def delete(self, key: Any) -> bool:
        """Delete a key from the hash index."""
        self.stats.usage_count += 1
        self.stats.last_used = time.time()
        if key in self._hash_table:
            del self._hash_table[key]
            return True
        return False

    def update(self, old_key: Any, new_key: Any, value: Any) -> None:
        """Update a key in the hash index."""
        if old_key in self._hash_table:
            del self._hash_table[old_key]
        self._hash_table[new_key] = value
        self.stats.usage_count += 1
        self.stats.last_used = time.time()


class FullTextIndex(Index):
    """Full-text search index implementation."""

    def create(self, backend: DatabaseBackend) -> None:
        """Create full-text search index."""
        with self._lock:
            try:
                # Create FTS virtual table
                columns_str = ", ".join(self.config.columns)

                create_sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.config.name}_fts
                USING fts5({columns_str})
                """

                backend.execute_query(create_sql)
                self._logger.info(f"Created full-text index: {self.config.name}")

            except Exception as e:
                raise IndexError(
                    f"Failed to create full-text index {self.config.name}: {e}"
                )

    def drop(self, backend: DatabaseBackend) -> None:
        """Drop full-text search index."""
        with self._lock:
            try:
                drop_sql = f"DROP TABLE IF EXISTS {self.config.name}_fts"
                backend.execute_query(drop_sql)
                self._logger.info(f"Dropped full-text index: {self.config.name}")

            except Exception as e:
                raise IndexError(
                    f"Failed to drop full-text index {self.config.name}: {e}"
                )

    def rebuild(self, backend: DatabaseBackend) -> None:
        """Rebuild full-text search index."""
        with self._lock:
            try:
                self.drop(backend)
                self.create(backend)

                # Rebuild FTS index
                rebuild_sql = f"INSERT INTO {self.config.name}_fts({self.config.name}_fts) VALUES('rebuild')"
                backend.execute_query(rebuild_sql)

                self.stats.maintenance_count += 1
                self.stats.last_maintenance = time.time()

                self._logger.info(f"Rebuilt full-text index: {self.config.name}")

            except Exception as e:
                raise IndexError(
                    f"Failed to rebuild full-text index {self.config.name}: {e}"
                )

    def analyze(self, backend: DatabaseBackend) -> IndexStats:
        """Analyze full-text search index statistics."""
        with self._lock:
            try:
                # Get FTS statistics
                stats_sql = f"SELECT COUNT(*) as count FROM {self.config.name}_fts"
                result = backend.execute_query(stats_sql)

                if result.rows:
                    self.stats.row_count = result.rows[0].get("count", 0)

                self.stats.last_used = time.time()

                return self.stats

            except Exception as e:
                raise IndexError(
                    f"Failed to analyze full-text index {self.config.name}: {e}"
                )

    def is_used_by_query(self, query: str) -> bool:
        """Check if full-text index is used by a query."""
        query_upper = query.upper()
        return "MATCH" in query_upper or "FTS" in query_upper

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key-value pair into the full-text index."""
        # Full-text indexes work differently - they index document content
        self.stats.usage_count += 1
        self.stats.last_used = time.time()

    def search(self, key: Any) -> List[Any]:
        """Search for values by key in the full-text index."""
        # Full-text indexes use MATCH queries for search
        self.stats.usage_count += 1
        self.stats.last_used = time.time()
        return []

    def delete(self, key: Any) -> bool:
        """Delete a key from the full-text index."""
        # Full-text indexes update when documents are deleted
        self.stats.usage_count += 1
        self.stats.last_used = time.time()
        return True

    def update(self, old_key: Any, new_key: Any, value: Any) -> None:
        """Update a key in the full-text index."""
        # Full-text indexes update when documents are modified
        self.stats.usage_count += 1
        self.stats.last_used = time.time()


class IndexManager:
    """Manager for database indexes."""

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self._indexes: Dict[str, Index] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        self._maintenance_thread: Optional[threading.Thread] = None
        self._running = False

    def create_index(self, config: IndexConfig) -> Index:
        """Create a new index."""
        with self._lock:
            if config.name in self._indexes:
                raise IndexError(f"Index {config.name} already exists")

            # Create appropriate index type
            if config.index_type == IndexType.B_TREE:
                index = BTreeIndex(config)
            elif config.index_type == IndexType.HASH:
                index = HashIndex(config)
            elif config.index_type == IndexType.FULL_TEXT:
                index = FullTextIndex(config)
            else:
                raise IndexError(f"Unsupported index type: {config.index_type}")

            # Create the index
            index.create(self.backend)

            # Store index
            self._indexes[config.name] = index

            self._logger.info(f"Created index: {config.name}")
            return index

    def drop_index(self, name: str) -> None:
        """Drop an index."""
        with self._lock:
            if name not in self._indexes:
                raise IndexNotFoundError(f"Index {name} not found")

            index = self._indexes[name]
            index.drop(self.backend)

            del self._indexes[name]

            self._logger.info(f"Dropped index: {name}")

    def rebuild_index(self, name: str) -> None:
        """Rebuild an index."""
        with self._lock:
            if name not in self._indexes:
                raise IndexNotFoundError(f"Index {name} not found")

            index = self._indexes[name]
            index.rebuild(self.backend)

            self._logger.info(f"Rebuilt index: {name}")

    def analyze_index(self, name: str) -> IndexStats:
        """Analyze index statistics."""
        with self._lock:
            if name not in self._indexes:
                raise IndexNotFoundError(f"Index {name} not found")

            index = self._indexes[name]
            return index.analyze(self.backend)

    def get_index(self, name: str) -> Index:
        """Get an index by name."""
        with self._lock:
            if name not in self._indexes:
                raise IndexNotFoundError(f"Index {name} not found")
            return self._indexes[name]

    def list_indexes(self) -> List[str]:
        """List all index names."""
        with self._lock:
            return list(self._indexes.keys())

    def get_table_indexes(self, table_name: str) -> List[Index]:
        """Get all indexes for a specific table."""
        with self._lock:
            table_indexes = []
            for index in self._indexes.values():
                if index.config.table == table_name:
                    table_indexes.append(index)
            return table_indexes

    def get_index_stats(self) -> Dict[str, IndexStats]:
        """Get statistics for all indexes."""
        with self._lock:
            stats = {}
            for name, index in self._indexes.items():
                try:
                    stats[name] = index.analyze(self.backend)
                except Exception as e:
                    self._logger.error(f"Failed to get stats for index {name}: {e}")

            return stats

    def optimize_indexes(self) -> None:
        """Optimize all indexes."""
        with self._lock:
            for name, index in self._indexes.items():
                try:
                    stats = index.analyze(self.backend)
                    if stats.fragmentation_percent > 10.0:  # 10% threshold
                        index.rebuild(self.backend)
                        self._logger.info(f"Optimized fragmented index: {name}")
                except Exception as e:
                    self._logger.error(f"Failed to optimize index {name}: {e}")

    def start_maintenance(self) -> None:
        """Start index maintenance thread."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_worker, daemon=True
            )
            self._maintenance_thread.start()

            self._logger.info("Started index maintenance")

    def stop_maintenance(self) -> None:
        """Stop index maintenance thread."""
        with self._lock:
            self._running = False
            if self._maintenance_thread:
                self._maintenance_thread.join(timeout=5.0)

            self._logger.info("Stopped index maintenance")

    def _maintenance_worker(self) -> None:
        """Background maintenance worker."""
        while self._running:
            try:
                time.sleep(3600)  # Run every hour

                if not self._running:
                    break

                self.optimize_indexes()

            except Exception as e:
                self._logger.error(f"Index maintenance error: {e}")

    def create_default_indexes(self) -> None:
        """Create default indexes for DubChain tables."""
        default_indexes = [
            # Block indexes
            IndexConfig(
                name="idx_blocks_height",
                table="blocks",
                columns=["height"],
                index_type=IndexType.B_TREE,
                unique=True,
            ),
            IndexConfig(
                name="idx_blocks_timestamp",
                table="blocks",
                columns=["timestamp"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_blocks_hash",
                table="blocks",
                columns=["hash"],
                index_type=IndexType.HASH,
                unique=True,
            ),
            # Transaction indexes
            IndexConfig(
                name="idx_transactions_block_hash",
                table="transactions",
                columns=["block_hash"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_transactions_type",
                table="transactions",
                columns=["transaction_type"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_transactions_hash",
                table="transactions",
                columns=["hash"],
                index_type=IndexType.HASH,
                unique=True,
            ),
            # UTXO indexes
            IndexConfig(
                name="idx_utxos_recipient",
                table="utxos",
                columns=["recipient_address"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_utxos_spent",
                table="utxos",
                columns=["is_spent"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_utxos_tx_hash",
                table="utxos",
                columns=["tx_hash", "output_index"],
                index_type=IndexType.B_TREE,
                unique=True,
            ),
            # Smart contract indexes
            IndexConfig(
                name="idx_contracts_creator",
                table="smart_contracts",
                columns=["creator"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_contracts_active",
                table="smart_contracts",
                columns=["is_active"],
                index_type=IndexType.B_TREE,
            ),
            IndexConfig(
                name="idx_contracts_address",
                table="smart_contracts",
                columns=["address"],
                index_type=IndexType.HASH,
                unique=True,
            ),
        ]

        for config in default_indexes:
            try:
                self.create_index(config)
            except IndexError as e:
                self._logger.warning(
                    f"Failed to create default index {config.name}: {e}"
                )

    def __enter__(self):
        """Context manager entry."""
        self.start_maintenance()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_maintenance()
