"""Database migration system for DubChain.

This module provides a comprehensive migration system for managing database
schema changes, data transformations, and version control.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .database import DatabaseBackend, DatabaseError


class MigrationError(Exception):
    """Base exception for migration operations."""

    pass


class MigrationVersionError(MigrationError):
    """Migration version error."""

    pass


class MigrationConflictError(MigrationError):
    """Migration conflict error."""

    pass


class MigrationRollbackError(MigrationError):
    """Migration rollback error."""

    pass


class MigrationStatus(Enum):
    """Migration status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Database migration definition."""

    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    rollback_safe: bool = True
    batch_size: int = 1000
    timeout: int = 300  # seconds

    # Custom migration functions
    up_function: Optional[Callable[[DatabaseBackend], None]] = None
    down_function: Optional[Callable[[DatabaseBackend], None]] = None

    # Validation
    validate_up: Optional[Callable[[DatabaseBackend], bool]] = None
    validate_down: Optional[Callable[[DatabaseBackend], bool]] = None

    # Metadata
    author: str = "DubChain Team"
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class MigrationRecord:
    """Migration execution record."""

    version: str
    name: str
    status: MigrationStatus
    started_at: float
    completed_at: Optional[float] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    batch_count: int = 0
    rows_affected: int = 0


@dataclass
class MigrationPlan:
    """Migration execution plan."""

    migrations: List[Migration]
    total_count: int
    estimated_time: float
    rollback_plan: List[str]
    dependencies_resolved: bool = True


class MigrationValidator:
    """Migration validation utilities."""

    @staticmethod
    def validate_version(version: str) -> bool:
        """Validate migration version format."""
        # Support formats: 1, 1.0, 1.0.0, 20231201, 20231201_001
        patterns = [
            r"^\d+$",  # 1
            r"^\d+\.\d+$",  # 1.0
            r"^\d+\.\d+\.\d+$",  # 1.0.0
            r"^\d{8}$",  # 20231201
            r"^\d{8}_\d{3}$",  # 20231201_001
        ]

        return any(re.match(pattern, version) for pattern in patterns)

    @staticmethod
    def validate_sql(sql: str) -> bool:
        """Validate SQL syntax."""
        try:
            # Basic SQL validation
            sql_upper = sql.upper().strip()

            # Check for dangerous operations
            dangerous_ops = ["DROP DATABASE", "DROP SCHEMA", "TRUNCATE"]
            for op in dangerous_ops:
                if op in sql_upper:
                    return False

            # Check for required semicolon
            if not sql_upper.endswith(";"):
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def validate_dependencies(migrations: List[Migration]) -> bool:
        """Validate migration dependencies."""
        versions = {m.version for m in migrations}

        for migration in migrations:
            for dep in migration.dependencies:
                if dep not in versions:
                    return False

        return True


class MigrationExecutor:
    """Migration execution engine."""

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self._logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

    def execute_migration(self, migration: Migration) -> MigrationRecord:
        """Execute a single migration."""
        with self._lock:
            record = MigrationRecord(
                version=migration.version,
                name=migration.name,
                status=MigrationStatus.RUNNING,
                started_at=time.time(),
            )

            try:
                self._logger.info(
                    f"Executing migration {migration.version}: {migration.name}"
                )

                # Validate migration
                if not MigrationValidator.validate_version(migration.version):
                    raise MigrationVersionError(
                        f"Invalid migration version: {migration.version}"
                    )

                if not MigrationValidator.validate_sql(migration.up_sql):
                    raise MigrationError(
                        f"Invalid SQL in migration {migration.version}"
                    )

                # Execute migration
                if migration.up_function:
                    migration.up_function(self.backend)
                else:
                    self._execute_sql(migration.up_sql, migration.batch_size)

                # Validate migration result
                if migration.validate_up:
                    if not migration.validate_up(self.backend):
                        raise MigrationError(
                            f"Migration validation failed: {migration.version}"
                        )

                # Update record
                record.status = MigrationStatus.COMPLETED
                record.completed_at = time.time()
                record.execution_time = record.completed_at - record.started_at

                self._logger.info(
                    f"Migration {migration.version} completed successfully"
                )

            except Exception as e:
                record.status = MigrationStatus.FAILED
                record.error_message = str(e)
                record.completed_at = time.time()
                record.execution_time = record.completed_at - record.started_at

                self._logger.error(f"Migration {migration.version} failed: {e}")
                raise MigrationError(f"Migration {migration.version} failed: {e}")

            return record

    def rollback_migration(self, migration: Migration) -> MigrationRecord:
        """Rollback a single migration."""
        with self._lock:
            record = MigrationRecord(
                version=migration.version,
                name=migration.name,
                status=MigrationStatus.RUNNING,
                started_at=time.time(),
            )

            try:
                self._logger.info(
                    f"Rolling back migration {migration.version}: {migration.name}"
                )

                if not migration.rollback_safe:
                    raise MigrationRollbackError(
                        f"Migration {migration.version} is not rollback safe"
                    )

                # Execute rollback
                if migration.down_function:
                    migration.down_function(self.backend)
                else:
                    self._execute_sql(migration.down_sql, migration.batch_size)

                # Validate rollback result
                if migration.validate_down:
                    if not migration.validate_down(self.backend):
                        raise MigrationError(
                            f"Rollback validation failed: {migration.version}"
                        )

                # Update record
                record.status = MigrationStatus.ROLLED_BACK
                record.completed_at = time.time()
                record.execution_time = record.completed_at - record.started_at

                self._logger.info(
                    f"Migration {migration.version} rolled back successfully"
                )

            except Exception as e:
                record.status = MigrationStatus.FAILED
                record.error_message = str(e)
                record.completed_at = time.time()
                record.execution_time = record.completed_at - record.started_at

                self._logger.error(
                    f"Migration {migration.version} rollback failed: {e}"
                )
                raise MigrationRollbackError(
                    f"Migration {migration.version} rollback failed: {e}"
                )

            return record

    def _execute_sql(self, sql: str, batch_size: int) -> None:
        """Execute SQL with batching support."""
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]

        for statement in statements:
            if statement.upper().startswith("SELECT"):
                # SELECT statements don't need batching
                self.backend.execute_query(statement)
            else:
                # For INSERT/UPDATE/DELETE, we might need batching
                # For now, execute directly
                self.backend.execute_query(statement)


class MigrationManager:
    """Database migration manager."""

    def __init__(self, backend: DatabaseBackend, migrations_dir: str = "migrations"):
        self.backend = backend
        self.migrations_dir = Path(migrations_dir)
        self.executor = MigrationExecutor(backend)
        self._migrations: Dict[str, Migration] = {}
        self._records: Dict[str, MigrationRecord] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize migration tracking table
        self._init_migration_table()

    def _init_migration_table(self) -> None:
        """Initialize migration tracking table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at REAL NOT NULL,
            completed_at REAL,
            execution_time REAL DEFAULT 0,
            error_message TEXT,
            batch_count INTEGER DEFAULT 0,
            rows_affected INTEGER DEFAULT 0,
            created_at REAL DEFAULT (julianday('now'))
        )
        """

        self.backend.execute_query(create_table_sql)

    def load_migrations(self) -> None:
        """Load migrations from directory."""
        with self._lock:
            self._migrations.clear()

            if not self.migrations_dir.exists():
                self._logger.warning(
                    f"Migrations directory not found: {self.migrations_dir}"
                )
                return

            # Load migration files
            for file_path in sorted(self.migrations_dir.glob("*.py")):
                try:
                    migration = self._load_migration_file(file_path)
                    if migration:
                        self._migrations[migration.version] = migration
                        self._logger.debug(f"Loaded migration: {migration.version}")
                except Exception as e:
                    self._logger.error(f"Failed to load migration {file_path}: {e}")

            # Load migration records from database
            self._load_migration_records()

            self._logger.info(f"Loaded {len(self._migrations)} migrations")

    def _load_migration_file(self, file_path: Path) -> Optional[Migration]:
        """Load a single migration file."""
        try:
            # Read file content
            with open(file_path, "r") as f:
                content = f.read()

            # Extract migration metadata (simplified)
            # In a real implementation, you'd use AST parsing or a DSL
            lines = content.split("\n")

            # Find migration definition
            version = None
            name = None
            description = ""
            up_sql = ""
            down_sql = ""

            in_up_sql = False
            in_down_sql = False

            for line in lines:
                line = line.strip()

                if line.startswith("version = "):
                    version = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("name = "):
                    name = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("description = "):
                    description = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith('up_sql = """'):
                    in_up_sql = True
                    up_sql = line[12:]  # Remove 'up_sql = """'
                elif line.startswith('down_sql = """'):
                    in_down_sql = True
                    down_sql = line[14:]  # Remove 'down_sql = """'
                elif in_up_sql and line.endswith('"""'):
                    in_up_sql = False
                elif in_down_sql and line.endswith('"""'):
                    in_down_sql = False
                elif in_up_sql:
                    up_sql += "\n" + line
                elif in_down_sql:
                    down_sql += "\n" + line

            if version and name and up_sql and down_sql:
                return Migration(
                    version=version,
                    name=name,
                    description=description,
                    up_sql=up_sql,
                    down_sql=down_sql,
                )

        except Exception as e:
            self._logger.error(f"Error loading migration file {file_path}: {e}")

        return None

    def _load_migration_records(self) -> None:
        """Load migration records from database."""
        try:
            query = "SELECT * FROM migrations ORDER BY version"
            result = self.backend.execute_query(query)

            for row in result.rows:
                record = MigrationRecord(
                    version=row["version"],
                    name=row["name"],
                    status=MigrationStatus(row["status"]),
                    started_at=row["started_at"],
                    completed_at=row.get("completed_at"),
                    execution_time=row.get("execution_time", 0),
                    error_message=row.get("error_message"),
                    batch_count=row.get("batch_count", 0),
                    rows_affected=row.get("rows_affected", 0),
                )

                self._records[record.version] = record

        except Exception as e:
            self._logger.error(f"Failed to load migration records: {e}")

    def create_migration(self, version: str, name: str, description: str = "") -> Path:
        """Create a new migration file."""
        with self._lock:
            if version in self._migrations:
                raise MigrationConflictError(f"Migration {version} already exists")

            # Validate version
            if not MigrationValidator.validate_version(version):
                raise MigrationVersionError(f"Invalid migration version: {version}")

            # Create migration file
            filename = f"{version}_{name.lower().replace(' ', '_')}.py"
            file_path = self.migrations_dir / filename

            template = f'''"""
Migration: {version} - {name}

{description}
"""

version = "{version}"
name = "{name}"
description = "{description}"

up_sql = """
-- Add your migration SQL here
-- Example:
-- CREATE TABLE example (
--     id INTEGER PRIMARY KEY,
--     name TEXT NOT NULL
-- );
"""

down_sql = """
-- Add your rollback SQL here
-- Example:
-- DROP TABLE IF EXISTS example;
"""
'''

            with open(file_path, "w") as f:
                f.write(template)

            self._logger.info(f"Created migration file: {file_path}")
            return file_path

    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        with self._lock:
            pending = []

            for version, migration in self._migrations.items():
                if version not in self._records:
                    pending.append(migration)
                elif self._records[version].status == MigrationStatus.FAILED:
                    pending.append(migration)

            # Sort by version
            pending.sort(key=lambda m: m.version)

            return pending

    def get_migration_status(self, version: str) -> Optional[MigrationStatus]:
        """Get migration status."""
        with self._lock:
            if version in self._records:
                return self._records[version].status
            return None

    def create_migration_plan(
        self, target_version: Optional[str] = None
    ) -> MigrationPlan:
        """Create migration execution plan."""
        with self._lock:
            pending = self.get_pending_migrations()

            if target_version:
                # Filter to target version
                pending = [m for m in pending if m.version <= target_version]

            # Resolve dependencies
            resolved = self._resolve_dependencies(pending)

            # Create rollback plan
            rollback_plan = [m.version for m in resolved if m.rollback_safe]
            rollback_plan.reverse()  # Rollback in reverse order

            # Estimate execution time
            estimated_time = sum(
                10.0 for _ in resolved
            )  # 10 seconds per migration estimate

            return MigrationPlan(
                migrations=resolved,
                total_count=len(resolved),
                estimated_time=estimated_time,
                rollback_plan=rollback_plan,
                dependencies_resolved=True,
            )

    def _resolve_dependencies(self, migrations: List[Migration]) -> List[Migration]:
        """Resolve migration dependencies."""
        resolved = []
        remaining = migrations.copy()

        while remaining:
            # Find migrations with no unresolved dependencies
            ready = []
            for migration in remaining:
                deps_resolved = all(
                    dep in [m.version for m in resolved]
                    or dep not in [m.version for m in remaining]
                    for dep in migration.dependencies
                )
                if deps_resolved:
                    ready.append(migration)

            if not ready:
                raise MigrationError("Circular dependency detected in migrations")

            # Add ready migrations to resolved list
            ready.sort(key=lambda m: m.version)
            resolved.extend(ready)

            # Remove from remaining
            for migration in ready:
                remaining.remove(migration)

        return resolved

    def execute_migrations(
        self, plan: Optional[MigrationPlan] = None
    ) -> List[MigrationRecord]:
        """Execute migrations according to plan."""
        with self._lock:
            if plan is None:
                plan = self.create_migration_plan()

            executed = []

            for migration in plan.migrations:
                try:
                    # Execute migration
                    record = self.executor.execute_migration(migration)

                    # Save record to database
                    self._save_migration_record(record)
                    self._records[record.version] = record

                    executed.append(record)

                except Exception as e:
                    self._logger.error(f"Migration execution failed: {e}")
                    raise MigrationError(f"Migration execution failed: {e}")

            return executed

    def rollback_migrations(self, target_version: str) -> List[MigrationRecord]:
        """Rollback migrations to target version."""
        with self._lock:
            # Get migrations to rollback
            to_rollback = []
            for version, record in self._records.items():
                if (
                    record.status == MigrationStatus.COMPLETED
                    and version > target_version
                    and version in self._migrations
                ):
                    to_rollback.append(self._migrations[version])

            # Sort in reverse order
            to_rollback.sort(key=lambda m: m.version, reverse=True)

            rolled_back = []

            for migration in to_rollback:
                try:
                    # Rollback migration
                    record = self.executor.rollback_migration(migration)

                    # Update record in database
                    self._update_migration_record(record)
                    self._records[record.version] = record

                    rolled_back.append(record)

                except Exception as e:
                    self._logger.error(f"Migration rollback failed: {e}")
                    raise MigrationRollbackError(f"Migration rollback failed: {e}")

            return rolled_back

    def _save_migration_record(self, record: MigrationRecord) -> None:
        """Save migration record to database."""
        insert_sql = """
        INSERT OR REPLACE INTO migrations 
        (version, name, status, started_at, completed_at, execution_time, 
         error_message, batch_count, rows_affected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            record.version,
            record.name,
            record.status.value,
            record.started_at,
            record.completed_at,
            record.execution_time,
            record.error_message,
            record.batch_count,
            record.rows_affected,
        )

        self.backend.execute_query(insert_sql, params)

    def _update_migration_record(self, record: MigrationRecord) -> None:
        """Update migration record in database."""
        update_sql = """
        UPDATE migrations SET
            status = ?, completed_at = ?, execution_time = ?, error_message = ?
        WHERE version = ?
        """

        params = (
            record.status.value,
            record.completed_at,
            record.execution_time,
            record.error_message,
            record.version,
        )

        self.backend.execute_query(update_sql, params)

    def get_migration_history(self) -> List[MigrationRecord]:
        """Get migration execution history."""
        with self._lock:
            return list(self._records.values())

    def get_current_version(self) -> Optional[str]:
        """Get current database version."""
        with self._lock:
            completed = [
                record
                for record in self._records.values()
                if record.status == MigrationStatus.COMPLETED
            ]

            if not completed:
                return None

            # Return highest version
            return max(completed, key=lambda r: r.version).version

    def __enter__(self):
        """Context manager entry."""
        self.load_migrations()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
