"""Database migration system for DubChain.

This module provides a comprehensive migration system for managing database
schema changes, data transformations, and version control.
"""

import json
import logging

logger = logging.getLogger(__name__)
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

from ..errors import ClientError
from ..logging import get_logger

logger = get_logger(__name__)

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

class MigrationType(Enum):
    """Migration type."""
    SCHEMA = "schema"
    DATA = "data"
    INDEX = "index"
    CONSTRAINT = "constraint"
    FUNCTION = "function"
    TRIGGER = "trigger"

@dataclass
class Migration:
    """Migration definition."""
    version: str
    name: str
    description: str
    migration_type: MigrationType
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    status: MigrationStatus = MigrationStatus.PENDING
    error_message: Optional[str] = None
    rollback_at: Optional[float] = None

@dataclass
class MigrationConfig:
    """Migration configuration."""
    migrations_dir: str = "migrations"
    database_url: str = "sqlite:///dubchain.db"
    backup_dir: str = "backups"
    max_retries: int = 3
    timeout: int = 300  # seconds
    enable_backup: bool = True
    enable_rollback: bool = True

class MigrationExecutor(ABC):
    """Abstract migration executor."""
    
    @abstractmethod
    def execute_up(self, migration: Migration) -> bool:
        """Execute migration up."""
        pass
    
    @abstractmethod
    def execute_down(self, migration: Migration) -> bool:
        """Execute migration down."""
        pass

class SQLiteMigrationExecutor(MigrationExecutor):
    """SQLite migration executor."""
    
    def __init__(self, database_path: str):
        """Initialize SQLite executor."""
        self.database_path = database_path
        self.connection = None
        logger.info(f"Initialized SQLite migration executor for {database_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    def execute_up(self, migration: Migration) -> bool:
        """Execute migration up."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Execute the up migration
            cursor.executescript(migration.up_sql)
            conn.commit()
            
            logger.info(f"Successfully executed migration {migration.version}: {migration.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute migration {migration.version}: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def execute_down(self, migration: Migration) -> bool:
        """Execute migration down."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Execute the down migration
            cursor.executescript(migration.down_sql)
            conn.commit()
            
            logger.info(f"Successfully rolled back migration {migration.version}: {migration.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to roll back migration {migration.version}: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

class MigrationRegistry:
    """Migration registry for tracking migrations."""
    
    def __init__(self, config: MigrationConfig):
        """Initialize migration registry."""
        self.config = config
        self.migrations: Dict[str, Migration] = {}
        self.executor = self._create_executor()
        self._ensure_migrations_table()
        logger.info("Initialized migration registry")
    
    def _create_executor(self) -> MigrationExecutor:
        """Create appropriate migration executor."""
        if self.config.database_url.startswith("sqlite"):
            db_path = self.config.database_url.replace("sqlite:///", "")
            return SQLiteMigrationExecutor(db_path)
        else:
            raise MigrationError(f"Unsupported database: {self.config.database_url}")
    
    def _ensure_migrations_table(self):
        """Ensure migrations table exists."""
        try:
            conn = self.executor._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    migration_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    executed_at REAL,
                    error_message TEXT,
                    rollback_at REAL,
                    created_at REAL NOT NULL
                )
            """)
            
            conn.commit()
            logger.info("Migrations table ensured")
            
        except Exception as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise MigrationError(f"Failed to create migrations table: {e}")
    
    def register_migration(self, migration: Migration) -> None:
        """Register a migration."""
        self.migrations[migration.version] = migration
        logger.info(f"Registered migration {migration.version}: {migration.name}")
    
    def get_migration(self, version: str) -> Optional[Migration]:
        """Get migration by version."""
        return self.migrations.get(version)
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get pending migrations."""
        return [m for m in self.migrations.values() if m.status == MigrationStatus.PENDING]
    
    def get_completed_migrations(self) -> List[Migration]:
        """Get completed migrations."""
        return [m for m in self.migrations.values() if m.status == MigrationStatus.COMPLETED]
    
    def get_failed_migrations(self) -> List[Migration]:
        """Get failed migrations."""
        return [m for m in self.migrations.values() if m.status == MigrationStatus.FAILED]
    
    def load_migrations_from_db(self) -> None:
        """Load migration status from database."""
        try:
            conn = self.executor._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM schema_migrations")
            rows = cursor.fetchall()
            
            for row in rows:
                version = row['version']
                if version in self.migrations:
                    migration = self.migrations[version]
                    migration.status = MigrationStatus(row['status'])
                    migration.executed_at = row['executed_at']
                    migration.error_message = row['error_message']
                    migration.rollback_at = row['rollback_at']
            
            logger.info(f"Loaded {len(rows)} migration statuses from database")
            
        except Exception as e:
            logger.error(f"Failed to load migrations from database: {e}")
            raise MigrationError(f"Failed to load migrations from database: {e}")
    
    def save_migration_status(self, migration: Migration) -> None:
        """Save migration status to database."""
        try:
            conn = self.executor._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO schema_migrations 
                (version, name, description, migration_type, status, executed_at, error_message, rollback_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                migration.version,
                migration.name,
                migration.description,
                migration.migration_type.value,
                migration.status.value,
                migration.executed_at,
                migration.error_message,
                migration.rollback_at,
                migration.created_at
            ))
            
            conn.commit()
            logger.info(f"Saved migration status for {migration.version}")
            
        except Exception as e:
            logger.error(f"Failed to save migration status: {e}")
            raise MigrationError(f"Failed to save migration status: {e}")

class MigrationRunner:
    """Migration runner for executing migrations."""
    
    def __init__(self, registry: MigrationRegistry):
        """Initialize migration runner."""
        self.registry = registry
        self.config = registry.config
        self.backup_manager = BackupManager(self.config) if self.config.enable_backup else None
        logger.info("Initialized migration runner")
    
    def run_migrations(self, target_version: Optional[str] = None) -> bool:
        """Run pending migrations."""
        try:
            logger.info("Starting migration run")
            
            # Load current migration status
            self.registry.load_migrations_from_db()
            
            # Get migrations to run
            migrations_to_run = self._get_migrations_to_run(target_version)
            
            if not migrations_to_run:
                logger.info("No migrations to run")
                return True
            
            # Create backup if enabled
            if self.backup_manager:
                backup_path = self.backup_manager.create_backup()
                logger.info(f"Created backup: {backup_path}")
            
            # Run migrations
            for migration in migrations_to_run:
                if not self._run_single_migration(migration):
                    logger.error(f"Migration {migration.version} failed")
                    return False
            
            logger.info(f"Successfully ran {len(migrations_to_run)} migrations")
            return True
            
        except Exception as e:
            logger.error(f"Migration run failed: {e}")
            return False
    
    def rollback_migrations(self, target_version: str) -> bool:
        """Rollback migrations to target version."""
        try:
            logger.info(f"Starting rollback to version {target_version}")
            
            # Load current migration status
            self.registry.load_migrations_from_db()
            
            # Get migrations to rollback
            migrations_to_rollback = self._get_migrations_to_rollback(target_version)
            
            if not migrations_to_rollback:
                logger.info("No migrations to rollback")
                return True
            
            # Create backup if enabled
            if self.backup_manager:
                backup_path = self.backup_manager.create_backup()
                logger.info(f"Created backup: {backup_path}")
            
            # Rollback migrations
            for migration in reversed(migrations_to_rollback):
                if not self._rollback_single_migration(migration):
                    logger.error(f"Rollback of migration {migration.version} failed")
                    return False
            
            logger.info(f"Successfully rolled back {len(migrations_to_rollback)} migrations")
            return True
            
        except Exception as e:
            logger.error(f"Migration rollback failed: {e}")
            return False
    
    def _get_migrations_to_run(self, target_version: Optional[str] = None) -> List[Migration]:
        """Get migrations to run."""
        pending_migrations = self.registry.get_pending_migrations()
        
        if target_version:
            # Run up to target version
            migrations_to_run = []
            for migration in sorted(pending_migrations, key=lambda m: m.version):
                migrations_to_run.append(migration)
                if migration.version == target_version:
                    break
            return migrations_to_run
        else:
            # Run all pending migrations
            return sorted(pending_migrations, key=lambda m: m.version)
    
    def _get_migrations_to_rollback(self, target_version: str) -> List[Migration]:
        """Get migrations to rollback."""
        completed_migrations = self.registry.get_completed_migrations()
        
        migrations_to_rollback = []
        for migration in sorted(completed_migrations, key=lambda m: m.version, reverse=True):
            migrations_to_rollback.append(migration)
            if migration.version == target_version:
                break
        
        return migrations_to_rollback
    
    def _run_single_migration(self, migration: Migration) -> bool:
        """Run a single migration."""
        try:
            logger.info(f"Running migration {migration.version}: {migration.name}")
            
            # Update status to running
            migration.status = MigrationStatus.RUNNING
            self.registry.save_migration_status(migration)
            
            # Execute migration
            success = self.registry.executor.execute_up(migration)
            
            if success:
                # Update status to completed
                migration.status = MigrationStatus.COMPLETED
                migration.executed_at = time.time()
                migration.error_message = None
                self.registry.save_migration_status(migration)
                
                logger.info(f"Migration {migration.version} completed successfully")
                return True
            else:
                # Update status to failed
                migration.status = MigrationStatus.FAILED
                migration.error_message = "Migration execution failed"
                self.registry.save_migration_status(migration)
                
                logger.error(f"Migration {migration.version} failed")
                return False
                
        except Exception as e:
            # Update status to failed
            migration.status = MigrationStatus.FAILED
            migration.error_message = str(e)
            self.registry.save_migration_status(migration)
            
            logger.error(f"Migration {migration.version} failed with exception: {e}")
            return False
    
    def _rollback_single_migration(self, migration: Migration) -> bool:
        """Rollback a single migration."""
        try:
            logger.info(f"Rolling back migration {migration.version}: {migration.name}")
            
            # Execute rollback
            success = self.registry.executor.execute_down(migration)
            
            if success:
                # Update status to rolled back
                migration.status = MigrationStatus.ROLLED_BACK
                migration.rollback_at = time.time()
                migration.error_message = None
                self.registry.save_migration_status(migration)
                
                logger.info(f"Migration {migration.version} rolled back successfully")
                return True
            else:
                logger.error(f"Rollback of migration {migration.version} failed")
                return False
                
        except Exception as e:
            logger.error(f"Rollback of migration {migration.version} failed with exception: {e}")
            return False

class BackupManager:
    """Backup manager for migrations."""
    
    def __init__(self, config: MigrationConfig):
        """Initialize backup manager."""
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized backup manager")
    
    def create_backup(self) -> str:
        """Create database backup."""
        try:
            timestamp = int(time.time())
            backup_filename = f"backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # Copy database file
            if self.config.database_url.startswith("sqlite"):
                db_path = self.config.database_url.replace("sqlite:///", "")
                
                import shutil
                shutil.copy2(db_path, backup_path)
                
                logger.info(f"Created backup: {backup_path}")
                return str(backup_path)
            else:
                raise MigrationError("Backup only supported for SQLite databases")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise MigrationError(f"Failed to create backup: {e}")
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            if self.config.database_url.startswith("sqlite"):
                db_path = self.config.database_url.replace("sqlite:///", "")
                
                import shutil
                shutil.copy2(backup_path, db_path)
                
                logger.info(f"Restored backup: {backup_path}")
                return True
            else:
                raise MigrationError("Restore only supported for SQLite databases")
                
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

class MigrationLoader:
    """Migration loader for loading migrations from files."""
    
    def __init__(self, migrations_dir: str):
        """Initialize migration loader."""
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized migration loader for {migrations_dir}")
    
    def load_migrations(self) -> List[Migration]:
        """Load migrations from directory."""
        migrations = []
        
        try:
            for file_path in self.migrations_dir.glob("*.py"):
                migration = self._load_migration_from_file(file_path)
                if migration:
                    migrations.append(migration)
            
            logger.info(f"Loaded {len(migrations)} migrations from {self.migrations_dir}")
            return migrations
            
        except Exception as e:
            logger.error(f"Failed to load migrations: {e}")
            raise MigrationError(f"Failed to load migrations: {e}")
    
    def _load_migration_from_file(self, file_path: Path) -> Optional[Migration]:
        """Load migration from Python file."""
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract migration metadata using regex
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
            description_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
            
            if not version_match or not name_match:
                logger.warning(f"Invalid migration file: {file_path}")
                return None
            
            version = version_match.group(1)
            name = name_match.group(1)
            description = description_match.group(1) if description_match else ""
            
            # Extract SQL from functions
            up_sql = self._extract_sql_function(content, 'up')
            down_sql = self._extract_sql_function(content, 'down')
            
            if not up_sql:
                logger.warning(f"No up() function found in {file_path}")
                return None
            
            migration = Migration(
                version=version,
                name=name,
                description=description,
                migration_type=MigrationType.SCHEMA,  # Default type
                up_sql=up_sql,
                down_sql=down_sql or ""
            )
            
            return migration
            
        except Exception as e:
            logger.error(f"Failed to load migration from {file_path}: {e}")
            return None
    
    def _extract_sql_function(self, content: str, function_name: str) -> Optional[str]:
        """Extract SQL from function."""
        try:
            # Find function definition
            pattern = rf'def\s+{function_name}\s*\([^)]*\):\s*\n(.*?)(?=\ndef\s+|\Z)'
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                return None
            
            function_body = match.group(1)
            
            # Extract SQL from triple quotes
            sql_pattern = r'"""([^"]*)"""'
            sql_match = re.search(sql_pattern, function_body, re.DOTALL)
            
            if sql_match:
                return sql_match.group(1).strip()
            
            # Extract SQL from single quotes
            sql_pattern = r"'''([^']*)'''"
            sql_match = re.search(sql_pattern, function_body, re.DOTALL)
            
            if sql_match:
                return sql_match.group(1).strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract SQL from {function_name} function: {e}")
            return None

class MigrationManager:
    """Main migration manager."""
    
    def __init__(self, config: MigrationConfig):
        """Initialize migration manager."""
        self.config = config
        self.registry = MigrationRegistry(config)
        self.runner = MigrationRunner(self.registry)
        self.loader = MigrationLoader(config.migrations_dir)
        logger.info("Initialized migration manager")
    
    def initialize(self) -> bool:
        """Initialize migration system."""
        try:
            # Load migrations from files
            migrations = self.loader.load_migrations()
            
            # Register migrations
            for migration in migrations:
                self.registry.register_migration(migration)
            
            logger.info("Migration system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize migration system: {e}")
            return False
    
    def migrate(self, target_version: Optional[str] = None) -> bool:
        """Run migrations."""
        return self.runner.run_migrations(target_version)
    
    def rollback(self, target_version: str) -> bool:
        """Rollback migrations."""
        return self.runner.rollback_migrations(target_version)
    
    def status(self) -> Dict[str, Any]:
        """Get migration status."""
        return {
            "pending": len(self.registry.get_pending_migrations()),
            "completed": len(self.registry.get_completed_migrations()),
            "failed": len(self.registry.get_failed_migrations()),
            "total": len(self.registry.migrations)
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.registry.executor, 'close'):
            self.registry.executor.close()

__all__ = [
    "MigrationManager",
    "MigrationRegistry",
    "MigrationRunner",
    "MigrationLoader",
    "BackupManager",
    "MigrationExecutor",
    "SQLiteMigrationExecutor",
    "Migration",
    "MigrationConfig",
    "MigrationStatus",
    "MigrationType",
    "MigrationError",
    "MigrationVersionError",
    "MigrationConflictError",
    "MigrationRollbackError",
]