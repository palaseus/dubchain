"""Tests for storage database module."""

import pytest

from dubchain.storage.database import (
    ConnectionError,
    DatabaseBackend,
    DatabaseConfig,
    DatabaseError,
    IsolationLevel,
    QueryError,
    SQLiteBackend,
    TransactionError,
)


class TestDatabaseExceptions:
    """Test database exception classes."""

    def test_database_error(self):
        """Test base database error."""
        error = DatabaseError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_connection_error(self):
        """Test connection error."""
        error = ConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, DatabaseError)

    def test_query_error(self):
        """Test query error."""
        error = QueryError("Query failed")
        assert str(error) == "Query failed"
        assert isinstance(error, DatabaseError)

    def test_transaction_error(self):
        """Test transaction error."""
        error = TransactionError("Transaction failed")
        assert str(error) == "Transaction failed"
        assert isinstance(error, DatabaseError)


class TestIsolationLevel:
    """Test IsolationLevel enum."""

    def test_isolation_level_values(self):
        """Test isolation level values."""
        assert IsolationLevel.READ_UNCOMMITTED.value == "read_uncommitted"
        assert IsolationLevel.READ_COMMITTED.value == "read_committed"
        assert IsolationLevel.REPEATABLE_READ.value == "repeatable_read"
        assert IsolationLevel.SERIALIZABLE.value == "serializable"


class TestDatabaseConfig:
    """Test DatabaseConfig functionality."""

    def test_database_config_creation(self):
        """Test creating database config."""
        config = DatabaseConfig(database_path="test.db", max_connections=15)
        assert config.database_path == "test.db"
        assert config.max_connections == 15

    def test_database_config_defaults(self):
        """Test database config defaults."""
        config = DatabaseConfig()
        assert config.database_path == "dubchain.db"
        assert config.max_connections == 10
        assert config.connection_timeout == 30.0
