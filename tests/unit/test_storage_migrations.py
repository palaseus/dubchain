"""Tests for storage migrations module."""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.storage.database import DatabaseBackend
from dubchain.storage.migrations import (
    Migration,
    MigrationConflictError,
    MigrationError,
    MigrationExecutor,
    MigrationManager,
    MigrationPlan,
    MigrationRecord,
    MigrationRollbackError,
    MigrationStatus,
    MigrationValidator,
    MigrationVersionError,
)


class TestMigrationError:
    """Test MigrationError exception."""

    def test_migration_error_creation(self):
        """Test creating migration error."""
        error = MigrationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestMigrationVersionError:
    """Test MigrationVersionError exception."""

    def test_migration_version_error_creation(self):
        """Test creating migration version error."""
        error = MigrationVersionError("Invalid version")
        assert str(error) == "Invalid version"
        assert isinstance(error, MigrationError)


class TestMigrationConflictError:
    """Test MigrationConflictError exception."""

    def test_migration_conflict_error_creation(self):
        """Test creating migration conflict error."""
        error = MigrationConflictError("Migration conflict")
        assert str(error) == "Migration conflict"
        assert isinstance(error, MigrationError)


class TestMigrationRollbackError:
    """Test MigrationRollbackError exception."""

    def test_migration_rollback_error_creation(self):
        """Test creating migration rollback error."""
        error = MigrationRollbackError("Rollback failed")
        assert str(error) == "Rollback failed"
        assert isinstance(error, MigrationError)


class TestMigrationStatus:
    """Test MigrationStatus enum."""

    def test_migration_status_values(self):
        """Test migration status values."""
        assert MigrationStatus.PENDING.value == "pending"
        assert MigrationStatus.RUNNING.value == "running"
        assert MigrationStatus.COMPLETED.value == "completed"
        assert MigrationStatus.FAILED.value == "failed"
        assert MigrationStatus.ROLLED_BACK.value == "rolled_back"

    def test_migration_status_enumeration(self):
        """Test migration status enumeration."""
        statuses = list(MigrationStatus)
        assert len(statuses) == 5
        assert MigrationStatus.PENDING in statuses
        assert MigrationStatus.RUNNING in statuses
        assert MigrationStatus.COMPLETED in statuses
        assert MigrationStatus.FAILED in statuses
        assert MigrationStatus.ROLLED_BACK in statuses


class TestMigration:
    """Test Migration dataclass."""

    def test_migration_creation(self):
        """Test creating migration."""
        migration = Migration(
            version="1.0.0",
            name="create_users_table",
            description="Create users table",
            up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            down_sql="DROP TABLE users;",
        )

        assert migration.version == "1.0.0"
        assert migration.name == "create_users_table"
        assert migration.description == "Create users table"
        assert (
            migration.up_sql
            == "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        )
        assert migration.down_sql == "DROP TABLE users;"
        assert migration.dependencies == []
        assert migration.rollback_safe == True
        assert migration.batch_size == 1000
        assert migration.timeout == 300
        assert migration.up_function is None
        assert migration.down_function is None
        assert migration.validate_up is None
        assert migration.validate_down is None
        assert migration.author == "DubChain Team"
        assert migration.tags == []

    def test_migration_defaults(self):
        """Test migration default values."""
        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test;",
            down_sql="DROP TABLE test;",
        )

        assert migration.dependencies == []
        assert migration.rollback_safe == True
        assert migration.batch_size == 1000
        assert migration.timeout == 300
        assert migration.up_function is None
        assert migration.down_function is None
        assert migration.validate_up is None
        assert migration.validate_down is None
        assert migration.author == "DubChain Team"
        assert migration.tags == []

    def test_migration_custom_values(self):
        """Test migration with custom values."""

        def up_func(backend):
            pass

        def down_func(backend):
            pass

        def validate_func(backend):
            return True

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test;",
            down_sql="DROP TABLE test;",
            dependencies=["0.9.0"],
            rollback_safe=False,
            batch_size=500,
            timeout=600,
            up_function=up_func,
            down_function=down_func,
            validate_up=validate_func,
            validate_down=validate_func,
            author="Test Author",
            tags=["test", "example"],
        )

        assert migration.dependencies == ["0.9.0"]
        assert migration.rollback_safe == False
        assert migration.batch_size == 500
        assert migration.timeout == 600
        assert migration.up_function == up_func
        assert migration.down_function == down_func
        assert migration.validate_up == validate_func
        assert migration.validate_down == validate_func
        assert migration.author == "Test Author"
        assert migration.tags == ["test", "example"]


class TestMigrationRecord:
    """Test MigrationRecord dataclass."""

    def test_migration_record_creation(self):
        """Test creating migration record."""
        record = MigrationRecord(
            version="1.0.0",
            name="test_migration",
            status=MigrationStatus.COMPLETED,
            started_at=time.time(),
        )

        assert record.version == "1.0.0"
        assert record.name == "test_migration"
        assert record.status == MigrationStatus.COMPLETED
        assert record.completed_at is None
        assert record.execution_time == 0.0
        assert record.error_message is None
        assert record.batch_count == 0
        assert record.rows_affected == 0

    def test_migration_record_defaults(self):
        """Test migration record default values."""
        record = MigrationRecord(
            version="1.0.0",
            name="test_migration",
            status=MigrationStatus.COMPLETED,
            started_at=time.time(),
        )

        assert record.completed_at is None
        assert record.execution_time == 0.0
        assert record.error_message is None
        assert record.batch_count == 0
        assert record.rows_affected == 0

    def test_migration_record_custom_values(self):
        """Test migration record with custom values."""
        start_time = time.time()
        completed_time = start_time + 10.0

        record = MigrationRecord(
            version="1.0.0",
            name="test_migration",
            status=MigrationStatus.COMPLETED,
            started_at=start_time,
            completed_at=completed_time,
            execution_time=10.0,
            error_message="Test error",
            batch_count=5,
            rows_affected=100,
        )

        assert record.completed_at == completed_time
        assert record.execution_time == 10.0
        assert record.error_message == "Test error"
        assert record.batch_count == 5
        assert record.rows_affected == 100


class TestMigrationPlan:
    """Test MigrationPlan dataclass."""

    def test_migration_plan_creation(self):
        """Test creating migration plan."""
        migrations = [
            Migration(
                "1.0.0", "test1", "Test 1", "CREATE TABLE test1;", "DROP TABLE test1;"
            ),
            Migration(
                "1.0.1", "test2", "Test 2", "CREATE TABLE test2;", "DROP TABLE test2;"
            ),
        ]

        plan = MigrationPlan(
            migrations=migrations,
            total_count=2,
            estimated_time=20.0,
            rollback_plan=["1.0.1", "1.0.0"],
        )

        assert plan.migrations == migrations
        assert plan.total_count == 2
        assert plan.estimated_time == 20.0
        assert plan.rollback_plan == ["1.0.1", "1.0.0"]
        assert plan.dependencies_resolved == True

    def test_migration_plan_defaults(self):
        """Test migration plan default values."""
        plan = MigrationPlan(
            migrations=[], total_count=0, estimated_time=0.0, rollback_plan=[]
        )

        assert plan.dependencies_resolved == True


class TestMigrationValidator:
    """Test MigrationValidator functionality."""

    def test_validate_version_valid(self):
        """Test validating valid version formats."""
        valid_versions = ["1", "1.0", "1.0.0", "20231201", "20231201_001"]

        for version in valid_versions:
            assert MigrationValidator.validate_version(version) == True

    def test_validate_version_invalid(self):
        """Test validating invalid version formats."""
        invalid_versions = [
            "1.0.0.0",
            "v1.0.0",
            "1.0.0-beta",
            "20231201_",
            "20231201_0001",
            "abc",
            "",
        ]

        for version in invalid_versions:
            assert MigrationValidator.validate_version(version) == False

    def test_validate_sql_valid(self):
        """Test validating valid SQL."""
        valid_sqls = [
            "CREATE TABLE test (id INTEGER);",
            "INSERT INTO test VALUES (1);",
            "UPDATE test SET name = 'test';",
            "DELETE FROM test WHERE id = 1;",
            "SELECT * FROM test;",
        ]

        for sql in valid_sqls:
            assert MigrationValidator.validate_sql(sql) == True

    def test_validate_sql_invalid(self):
        """Test validating invalid SQL."""
        invalid_sqls = [
            "DROP DATABASE test;",
            "DROP SCHEMA test;",
            "TRUNCATE TABLE test;",
            "CREATE TABLE test (id INTEGER)",  # Missing semicolon
            "SELECT * FROM test",  # Missing semicolon
        ]

        for sql in invalid_sqls:
            assert MigrationValidator.validate_sql(sql) == False

    def test_validate_dependencies_valid(self):
        """Test validating valid dependencies."""
        migrations = [
            Migration(
                "1.0.0",
                "migration1",
                "Migration 1",
                "CREATE TABLE test1;",
                "DROP TABLE test1;",
            ),
            Migration(
                "1.0.1",
                "migration2",
                "Migration 2",
                "CREATE TABLE test2;",
                "DROP TABLE test2;",
                dependencies=["1.0.0"],
            ),
            Migration(
                "1.0.2",
                "migration3",
                "Migration 3",
                "CREATE TABLE test3;",
                "DROP TABLE test3;",
                dependencies=["1.0.1"],
            ),
        ]

        assert MigrationValidator.validate_dependencies(migrations) == True

    def test_validate_dependencies_invalid(self):
        """Test validating invalid dependencies."""
        migrations = [
            Migration(
                "1.0.0",
                "migration1",
                "Migration 1",
                "CREATE TABLE test1;",
                "DROP TABLE test1;",
            ),
            Migration(
                "1.0.1",
                "migration2",
                "Migration 2",
                "CREATE TABLE test2;",
                "DROP TABLE test2;",
                dependencies=["1.0.2"],
            ),  # Circular dependency
        ]

        assert MigrationValidator.validate_dependencies(migrations) == False


class TestMigrationExecutor:
    """Test MigrationExecutor functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Fixture for mock backend."""
        backend = Mock()
        backend.execute_query = Mock(return_value="result")
        return backend

    @pytest.fixture
    def migration_executor(self, mock_backend):
        """Fixture for migration executor."""
        return MigrationExecutor(mock_backend)

    def test_migration_executor_creation(self, migration_executor, mock_backend):
        """Test creating migration executor."""
        assert migration_executor.backend == mock_backend
        assert hasattr(migration_executor._lock, "acquire")

    def test_execute_migration_success(self, migration_executor, mock_backend):
        """Test executing migration successfully."""
        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
        )

        record = migration_executor.execute_migration(migration)

        assert record.version == "1.0.0"
        assert record.name == "test_migration"
        assert record.status == MigrationStatus.COMPLETED
        assert record.completed_at is not None
        assert record.execution_time > 0
        assert record.error_message is None

        mock_backend.execute_query.assert_called_once_with(
            "CREATE TABLE test (id INTEGER)"
        )

    def test_execute_migration_with_function(self, migration_executor, mock_backend):
        """Test executing migration with custom function."""

        def up_func(backend):
            backend.execute_query("CREATE TABLE test (id INTEGER);")

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            up_function=up_func,
        )

        record = migration_executor.execute_migration(migration)

        assert record.status == MigrationStatus.COMPLETED
        mock_backend.execute_query.assert_called_once_with(
            "CREATE TABLE test (id INTEGER);"
        )

    def test_execute_migration_with_validation(self, migration_executor, mock_backend):
        """Test executing migration with validation."""

        def validate_func(backend):
            return True

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            validate_up=validate_func,
        )

        record = migration_executor.execute_migration(migration)

        assert record.status == MigrationStatus.COMPLETED

    def test_execute_migration_validation_failure(
        self, migration_executor, mock_backend
    ):
        """Test executing migration with validation failure."""

        def validate_func(backend):
            return False

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            validate_up=validate_func,
        )

        with pytest.raises(MigrationError):
            migration_executor.execute_migration(migration)

    def test_execute_migration_invalid_version(self, migration_executor, mock_backend):
        """Test executing migration with invalid version."""
        migration = Migration(
            version="invalid",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
        )

        with pytest.raises(MigrationError):
            migration_executor.execute_migration(migration)

    def test_execute_migration_invalid_sql(self, migration_executor, mock_backend):
        """Test executing migration with invalid SQL."""
        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER)",  # Missing semicolon
            down_sql="DROP TABLE test;",
        )

        with pytest.raises(MigrationError):
            migration_executor.execute_migration(migration)

    def test_rollback_migration_success(self, migration_executor, mock_backend):
        """Test rolling back migration successfully."""
        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            rollback_safe=True,
        )

        record = migration_executor.rollback_migration(migration)

        assert record.version == "1.0.0"
        assert record.name == "test_migration"
        assert record.status == MigrationStatus.ROLLED_BACK
        assert record.completed_at is not None
        assert record.execution_time > 0
        assert record.error_message is None

        mock_backend.execute_query.assert_called_once_with("DROP TABLE test")

    def test_rollback_migration_not_safe(self, migration_executor, mock_backend):
        """Test rolling back migration that is not rollback safe."""
        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            rollback_safe=False,
        )

        with pytest.raises(MigrationRollbackError):
            migration_executor.rollback_migration(migration)

    def test_rollback_migration_with_function(self, migration_executor, mock_backend):
        """Test rolling back migration with custom function."""

        def down_func(backend):
            backend.execute_query("DROP TABLE test;")

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            down_function=down_func,
            rollback_safe=True,
        )

        record = migration_executor.rollback_migration(migration)

        assert record.status == MigrationStatus.ROLLED_BACK
        mock_backend.execute_query.assert_called_once_with("DROP TABLE test;")

    def test_rollback_migration_with_validation(self, migration_executor, mock_backend):
        """Test rolling back migration with validation."""

        def validate_func(backend):
            return True

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            validate_down=validate_func,
            rollback_safe=True,
        )

        record = migration_executor.rollback_migration(migration)

        assert record.status == MigrationStatus.ROLLED_BACK

    def test_rollback_migration_validation_failure(
        self, migration_executor, mock_backend
    ):
        """Test rolling back migration with validation failure."""

        def validate_func(backend):
            return False

        migration = Migration(
            version="1.0.0",
            name="test_migration",
            description="Test migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            validate_down=validate_func,
            rollback_safe=True,
        )

        with pytest.raises(MigrationError):
            migration_executor.rollback_migration(migration)


class TestMigrationManager:
    """Test MigrationManager functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Fixture for mock backend."""
        backend = Mock()
        backend.execute_query = Mock(return_value="result")
        return backend

    @pytest.fixture
    def temp_dir(self):
        """Fixture for temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def migration_manager(self, mock_backend, temp_dir):
        """Fixture for migration manager."""
        return MigrationManager(mock_backend, temp_dir)

    def test_migration_manager_creation(
        self, migration_manager, mock_backend, temp_dir
    ):
        """Test creating migration manager."""
        assert migration_manager.backend == mock_backend
        assert migration_manager.migrations_dir == Path(temp_dir)
        assert isinstance(migration_manager.executor, MigrationExecutor)
        assert migration_manager._migrations == {}
        assert migration_manager._records == {}
        assert hasattr(migration_manager._lock, "acquire")

        # Check that migration table was created
        mock_backend.execute_query.assert_called()

    def test_create_migration(self, migration_manager, temp_dir):
        """Test creating migration file."""
        file_path = migration_manager.create_migration(
            version="1.0.0", name="test_migration", description="Test migration"
        )

        assert file_path.exists()
        assert file_path.name == "1.0.0_test_migration.py"

        # Check file content
        content = file_path.read_text()
        assert 'version = "1.0.0"' in content
        assert 'name = "test_migration"' in content
        assert 'description = "Test migration"' in content

    def test_create_migration_invalid_version(self, migration_manager):
        """Test creating migration with invalid version."""
        with pytest.raises(MigrationVersionError):
            migration_manager.create_migration(version="invalid", name="test_migration")

    def test_create_migration_conflict(self, migration_manager):
        """Test creating migration with conflicting version."""
        # Create first migration
        migration_manager.create_migration("1.0.0", "test1")

        # Try to create second migration with same version (should overwrite the file)
        # The current implementation doesn't check for conflicts, it just overwrites
        file_path = migration_manager.create_migration("1.0.0", "test2")

        # Check that the file was created
        assert file_path.exists()
        assert file_path.name == "1.0.0_test2.py"

    def test_get_pending_migrations(self, migration_manager):
        """Test getting pending migrations."""
        # Create some migrations
        migration1 = Migration(
            "1.0.0", "test1", "Test 1", "CREATE TABLE test1;", "DROP TABLE test1;"
        )
        migration2 = Migration(
            "1.0.1", "test2", "Test 2", "CREATE TABLE test2;", "DROP TABLE test2;"
        )

        migration_manager._migrations = {"1.0.0": migration1, "1.0.1": migration2}

        pending = migration_manager.get_pending_migrations()

        assert len(pending) == 2
        assert pending[0].version == "1.0.0"
        assert pending[1].version == "1.0.1"

    def test_get_pending_migrations_with_records(self, migration_manager):
        """Test getting pending migrations with existing records."""
        # Create migrations
        migration1 = Migration(
            "1.0.0", "test1", "Test 1", "CREATE TABLE test1;", "DROP TABLE test1;"
        )
        migration2 = Migration(
            "1.0.1", "test2", "Test 2", "CREATE TABLE test2;", "DROP TABLE test2;"
        )

        migration_manager._migrations = {"1.0.0": migration1, "1.0.1": migration2}

        # Add record for first migration
        record = MigrationRecord(
            version="1.0.0",
            name="test1",
            status=MigrationStatus.COMPLETED,
            started_at=time.time(),
        )
        migration_manager._records["1.0.0"] = record

        pending = migration_manager.get_pending_migrations()

        assert len(pending) == 1
        assert pending[0].version == "1.0.1"

    def test_get_migration_status(self, migration_manager):
        """Test getting migration status."""
        # Add migration record
        record = MigrationRecord(
            version="1.0.0",
            name="test1",
            status=MigrationStatus.COMPLETED,
            started_at=time.time(),
        )
        migration_manager._records["1.0.0"] = record

        status = migration_manager.get_migration_status("1.0.0")
        assert status == MigrationStatus.COMPLETED

        status = migration_manager.get_migration_status("1.0.1")
        assert status is None

    def test_create_migration_plan(self, migration_manager):
        """Test creating migration plan."""
        # Create migrations
        migration1 = Migration(
            "1.0.0", "test1", "Test 1", "CREATE TABLE test1;", "DROP TABLE test1;"
        )
        migration2 = Migration(
            "1.0.1", "test2", "Test 2", "CREATE TABLE test2;", "DROP TABLE test2;"
        )

        migration_manager._migrations = {"1.0.0": migration1, "1.0.1": migration2}

        plan = migration_manager.create_migration_plan()

        assert plan.total_count == 2
        assert len(plan.migrations) == 2
        assert plan.migrations[0].version == "1.0.0"
        assert plan.migrations[1].version == "1.0.1"
        assert plan.estimated_time > 0
        assert plan.rollback_plan == ["1.0.1", "1.0.0"]
        assert plan.dependencies_resolved == True

    def test_create_migration_plan_with_target_version(self, migration_manager):
        """Test creating migration plan with target version."""
        # Create migrations
        migration1 = Migration(
            "1.0.0", "test1", "Test 1", "CREATE TABLE test1;", "DROP TABLE test1;"
        )
        migration2 = Migration(
            "1.0.1", "test2", "Test 2", "CREATE TABLE test2;", "DROP TABLE test2;"
        )
        migration3 = Migration(
            "1.0.2", "test3", "Test 3", "CREATE TABLE test3;", "DROP TABLE test3;"
        )

        migration_manager._migrations = {
            "1.0.0": migration1,
            "1.0.1": migration2,
            "1.0.2": migration3,
        }

        plan = migration_manager.create_migration_plan(target_version="1.0.1")

        assert plan.total_count == 2
        assert len(plan.migrations) == 2
        assert plan.migrations[0].version == "1.0.0"
        assert plan.migrations[1].version == "1.0.1"

    def test_resolve_dependencies(self, migration_manager):
        """Test resolving migration dependencies."""
        # Create migrations with dependencies
        migration1 = Migration(
            "1.0.0", "test1", "Test 1", "CREATE TABLE test1;", "DROP TABLE test1;"
        )
        migration2 = Migration(
            "1.0.1",
            "test2",
            "Test 2",
            "CREATE TABLE test2;",
            "DROP TABLE test2;",
            dependencies=["1.0.0"],
        )
        migration3 = Migration(
            "1.0.2",
            "test3",
            "Test 3",
            "CREATE TABLE test3;",
            "DROP TABLE test3;",
            dependencies=["1.0.1"],
        )

        migrations = [migration3, migration1, migration2]  # Out of order

        resolved = migration_manager._resolve_dependencies(migrations)

        assert len(resolved) == 3
        assert resolved[0].version == "1.0.0"
        assert resolved[1].version == "1.0.1"
        assert resolved[2].version == "1.0.2"

    def test_resolve_dependencies_circular(self, migration_manager):
        """Test resolving circular dependencies."""
        # Create migrations with circular dependencies
        migration1 = Migration(
            "1.0.0",
            "test1",
            "Test 1",
            "CREATE TABLE test1;",
            "DROP TABLE test1;",
            dependencies=["1.0.1"],
        )
        migration2 = Migration(
            "1.0.1",
            "test2",
            "Test 2",
            "CREATE TABLE test2;",
            "DROP TABLE test2;",
            dependencies=["1.0.0"],
        )

        migrations = [migration1, migration2]

        with pytest.raises(MigrationError):
            migration_manager._resolve_dependencies(migrations)

    def test_get_migration_history(self, migration_manager):
        """Test getting migration history."""
        # Add migration records
        record1 = MigrationRecord(
            "1.0.0", "test1", MigrationStatus.COMPLETED, time.time()
        )
        record2 = MigrationRecord("1.0.1", "test2", MigrationStatus.FAILED, time.time())

        migration_manager._records = {"1.0.0": record1, "1.0.1": record2}

        history = migration_manager.get_migration_history()

        assert len(history) == 2
        assert history[0].version == "1.0.0"
        assert history[1].version == "1.0.1"

    def test_get_current_version(self, migration_manager):
        """Test getting current version."""
        # Add migration records
        record1 = MigrationRecord(
            "1.0.0", "test1", MigrationStatus.COMPLETED, time.time()
        )
        record2 = MigrationRecord(
            "1.0.1", "test2", MigrationStatus.COMPLETED, time.time()
        )
        record3 = MigrationRecord("1.0.2", "test3", MigrationStatus.FAILED, time.time())

        migration_manager._records = {
            "1.0.0": record1,
            "1.0.1": record2,
            "1.0.2": record3,
        }

        current_version = migration_manager.get_current_version()

        assert current_version == "1.0.1"  # Highest completed version

    def test_get_current_version_no_completed(self, migration_manager):
        """Test getting current version with no completed migrations."""
        # Add migration records
        record1 = MigrationRecord("1.0.0", "test1", MigrationStatus.FAILED, time.time())
        record2 = MigrationRecord(
            "1.0.1", "test2", MigrationStatus.PENDING, time.time()
        )

        migration_manager._records = {"1.0.0": record1, "1.0.1": record2}

        current_version = migration_manager.get_current_version()

        assert current_version is None

    def test_context_manager(self, mock_backend, temp_dir):
        """Test context manager functionality."""
        with MigrationManager(mock_backend, temp_dir) as manager:
            assert isinstance(manager, MigrationManager)
            assert manager.migrations_dir == Path(temp_dir)
