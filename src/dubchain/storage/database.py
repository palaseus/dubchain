"""Database backend implementation for DubChain.

This module provides the core database functionality including connection management,
query execution, transaction handling, and data persistence.
"""

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..crypto.hashing import SHA256Hasher


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class ConnectionError(DatabaseError):
    """Database connection error."""

    pass


class QueryError(DatabaseError):
    """Database query error."""

    pass


class TransactionError(DatabaseError):
    """Database transaction error."""

    pass


class IsolationLevel(Enum):
    """Database transaction isolation levels."""

    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    # Connection settings
    database_path: str = "dubchain.db"
    connection_timeout: float = 30.0
    max_connections: int = 10
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED

    # Performance settings
    cache_size: int = 2000  # SQLite pages
    synchronous: str = "NORMAL"  # OFF, NORMAL, FULL
    journal_mode: str = "WAL"  # DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF
    temp_store: str = "MEMORY"  # DEFAULT, FILE, MEMORY

    # Backup settings
    auto_backup: bool = True
    backup_interval: int = 3600  # seconds
    max_backups: int = 10

    # Monitoring
    enable_profiling: bool = False
    slow_query_threshold: float = 1.0  # seconds
    enable_metrics: bool = True


@dataclass
class QueryResult:
    """Database query result."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time: float = 0.0
    query_plan: Optional[str] = None
    cache_hit: bool = False


@dataclass
class DatabaseStats:
    """Database statistics."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    slow_queries: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    active_connections: int = 0
    database_size: int = 0
    last_backup: Optional[float] = None


class DatabaseBackend(ABC):
    """Abstract database backend interface."""

    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a database query."""
        pass

    @abstractmethod
    def execute_transaction(
        self, queries: List[Tuple[str, Optional[Dict[str, Any]]]]
    ) -> List[QueryResult]:
        """Execute multiple queries in a transaction."""
        pass

    @abstractmethod
    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        pass


class SQLiteBackend(DatabaseBackend):
    """SQLite database backend implementation."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._stats = DatabaseStats()
        self._query_cache: Dict[str, QueryResult] = {}
        self._logger = logging.getLogger(__name__)

        # Ensure database directory exists
        db_path = Path(config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        """Establish SQLite database connection."""
        with self._lock:
            if self._connection is not None:
                return

            try:
                self._connection = sqlite3.connect(
                    self.config.database_path,
                    timeout=self.config.connection_timeout,
                    isolation_level=None,  # We'll handle transactions manually
                )

                # Configure SQLite settings
                self._configure_sqlite()

                # Enable WAL mode for better concurrency
                self._connection.execute("PRAGMA journal_mode = WAL")

                # Create tables if they don't exist
                self._create_tables()

                self._logger.info(
                    f"Connected to SQLite database: {self.config.database_path}"
                )

            except sqlite3.Error as e:
                raise ConnectionError(f"Failed to connect to database: {e}")

    def disconnect(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                    self._connection = None
                    self._logger.info("Disconnected from SQLite database")
                except sqlite3.Error as e:
                    self._logger.error(f"Error closing database connection: {e}")

    def _configure_sqlite(self) -> None:
        """Configure SQLite performance settings."""
        if self._connection is None:
            return

        pragmas = [
            f"PRAGMA cache_size = {self.config.cache_size}",
            f"PRAGMA synchronous = {self.config.synchronous}",
            f"PRAGMA temp_store = {self.config.temp_store}",
            "PRAGMA foreign_keys = ON",
            "PRAGMA optimize",
        ]

        for pragma in pragmas:
            self._connection.execute(pragma)

    def _create_tables(self) -> None:
        """Create database tables."""
        if self._connection is None:
            return

        tables = [
            # Blocks table
            """
            CREATE TABLE IF NOT EXISTS blocks (
                hash TEXT PRIMARY KEY,
                height INTEGER UNIQUE NOT NULL,
                previous_hash TEXT,
                merkle_root TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                difficulty INTEGER NOT NULL,
                nonce INTEGER NOT NULL,
                version INTEGER NOT NULL,
                gas_limit INTEGER NOT NULL,
                gas_used INTEGER NOT NULL,
                extra_data BLOB,
                size INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """,
            # Transactions table
            """
            CREATE TABLE IF NOT EXISTS transactions (
                hash TEXT PRIMARY KEY,
                block_hash TEXT,
                block_height INTEGER,
                transaction_type TEXT NOT NULL,
                inputs TEXT NOT NULL,  -- JSON
                outputs TEXT NOT NULL,  -- JSON
                signature TEXT,
                size INTEGER NOT NULL,
                fee INTEGER NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (block_hash) REFERENCES blocks(hash)
            )
            """,
            # UTXOs table
            """
            CREATE TABLE IF NOT EXISTS utxos (
                tx_hash TEXT NOT NULL,
                output_index INTEGER NOT NULL,
                amount INTEGER NOT NULL,
                recipient_address TEXT NOT NULL,
                is_spent BOOLEAN DEFAULT FALSE,
                spent_by_tx TEXT,
                spent_at REAL,
                created_at REAL NOT NULL,
                PRIMARY KEY (tx_hash, output_index),
                FOREIGN KEY (spent_by_tx) REFERENCES transactions(hash)
            )
            """,
            # Smart contracts table
            """
            CREATE TABLE IF NOT EXISTS smart_contracts (
                address TEXT PRIMARY KEY,
                bytecode TEXT NOT NULL,
                abi TEXT,  -- JSON
                creator TEXT NOT NULL,
                creation_tx TEXT NOT NULL,
                block_height INTEGER NOT NULL,
                gas_limit INTEGER NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at REAL NOT NULL,
                FOREIGN KEY (creation_tx) REFERENCES transactions(hash)
            )
            """,
            # Contract state table
            """
            CREATE TABLE IF NOT EXISTS contract_state (
                contract_address TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (contract_address, key),
                FOREIGN KEY (contract_address) REFERENCES smart_contracts(address)
            )
            """,
            # Indexes for performance
            """
            CREATE INDEX IF NOT EXISTS idx_blocks_height ON blocks(height)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks(timestamp)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_block_hash ON transactions(block_hash)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_utxos_recipient ON utxos(recipient_address)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_utxos_spent ON utxos(is_spent)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_contracts_creator ON smart_contracts(creator)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_contracts_active ON smart_contracts(is_active)
            """,
        ]

        for table_sql in tables:
            self._connection.execute(table_sql)

        self._connection.commit()

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a database query."""
        if self._connection is None:
            raise ConnectionError("Database not connected")

        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"{query}:{hash(str(params)) if params else 'None'}"
            if cache_key in self._query_cache:
                result = self._query_cache[cache_key]
                result.cache_hit = True
                self._stats.cache_hits += 1
                return result

            # Execute query
            cursor = self._connection.cursor()

            if params:
                # Convert dict params to tuple for positional parameters
                if isinstance(params, dict):
                    # For named parameters, use as-is
                    cursor.execute(query, params)
                else:
                    cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get results
            columns = (
                [description[0] for description in cursor.description]
                if cursor.description
                else []
            )
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            execution_time = time.time() - start_time

            # Update statistics
            self._stats.total_queries += 1
            self._stats.total_execution_time += execution_time
            self._stats.average_execution_time = (
                self._stats.total_execution_time / self._stats.total_queries
            )

            if execution_time > self.config.slow_query_threshold:
                self._stats.slow_queries += 1
                self._logger.warning(
                    f"Slow query detected: {execution_time:.3f}s - {query[:100]}..."
                )

            # Cache result for SELECT queries
            if query.strip().upper().startswith("SELECT"):
                result = QueryResult(
                    rows=rows, row_count=len(rows), execution_time=execution_time
                )
                self._query_cache[cache_key] = result
                self._stats.cache_misses += 1
            else:
                # Clear cache for non-SELECT queries
                self._query_cache.clear()
                result = QueryResult(
                    rows=rows, row_count=len(rows), execution_time=execution_time
                )
                self._stats.cache_misses += 1

            return result

        except sqlite3.Error as e:
            raise QueryError(f"Query execution failed: {e}")

    def execute_transaction(
        self, queries: List[Tuple[str, Optional[Dict[str, Any]]]]
    ) -> List[QueryResult]:
        """Execute multiple queries in a transaction."""
        if self._connection is None:
            raise ConnectionError("Database not connected")

        results = []

        try:
            # Begin transaction
            self._connection.execute("BEGIN TRANSACTION")

            for query, params in queries:
                # Execute query directly in transaction context
                cursor = self._connection.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Get results
                columns = (
                    [description[0] for description in cursor.description]
                    if cursor.description
                    else []
                )
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

                result = QueryResult(
                    rows=rows,
                    row_count=len(rows),
                    execution_time=0.0,  # Don't track time in transaction
                )
                results.append(result)

            # Commit transaction
            self._connection.execute("COMMIT")

        except sqlite3.Error as e:
            # Rollback on error
            self._connection.execute("ROLLBACK")
            raise TransactionError(f"Transaction failed: {e}")

        return results

    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        with self._lock:
            stats = DatabaseStats(
                total_queries=self._stats.total_queries,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                slow_queries=self._stats.slow_queries,
                total_execution_time=self._stats.total_execution_time,
                average_execution_time=self._stats.average_execution_time,
                active_connections=1 if self._connection else 0,
                database_size=self._get_database_size(),
            )

            return stats

    def _get_database_size(self) -> int:
        """Get database file size."""
        try:
            db_path = Path(self.config.database_path)
            if db_path.exists():
                return db_path.stat().st_size
        except OSError:
            pass
        return 0

    def clear_cache(self) -> None:
        """Clear query cache."""
        with self._lock:
            self._query_cache.clear()

    def optimize(self) -> None:
        """Optimize database performance."""
        if self._connection is None:
            return

        try:
            self._connection.execute("PRAGMA optimize")
            self._connection.execute("VACUUM")
            self.clear_cache()
            self._logger.info("Database optimization completed")
        except sqlite3.Error as e:
            self._logger.error(f"Database optimization failed: {e}")

    def backup(self, backup_path: str) -> None:
        """Create database backup."""
        if self._connection is None:
            raise ConnectionError("Database not connected")

        try:
            # Ensure backup directory exists
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            # Create backup
            backup_conn = sqlite3.connect(backup_path)
            self._connection.backup(backup_conn)
            backup_conn.close()

            self._logger.info(f"Database backup created: {backup_path}")

        except sqlite3.Error as e:
            raise DatabaseError(f"Backup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class DatabaseManager:
    """Database manager for handling multiple database operations."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.backend = SQLiteBackend(config)
        self._logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize database manager."""
        self.backend.connect()
        self._logger.info("Database manager initialized")

    def shutdown(self) -> None:
        """Shutdown database manager."""
        self.backend.disconnect()
        self._logger.info("Database manager shutdown")

    def get_backend(self) -> DatabaseBackend:
        """Get database backend."""
        return self.backend

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
