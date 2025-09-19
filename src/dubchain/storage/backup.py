"""Backup and recovery system for DubChain database.

This module provides comprehensive backup and recovery capabilities including
incremental backups, compression, encryption, and point-in-time recovery.
"""

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from cryptography.fernet import Fernet

from .database import DatabaseBackend, DatabaseError


class BackupError(Exception):
    """Base exception for backup operations."""

    pass


class BackupCorruptionError(BackupError):
    """Backup corruption error."""

    pass


class BackupNotFoundError(BackupError):
    """Backup not found error."""

    pass


class RecoveryError(BackupError):
    """Recovery error."""

    pass


class BackupType(Enum):
    """Types of backups."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class BackupConfig:
    """Backup configuration."""

    # Backup settings
    backup_directory: str = "backups"
    backup_type: BackupType = BackupType.FULL
    compression: bool = True
    encryption: bool = False
    encryption_key: Optional[str] = None

    # Retention settings
    max_backups: int = 10
    retention_days: int = 30
    auto_cleanup: bool = True

    # Scheduling
    auto_backup: bool = True
    backup_interval: int = 3600  # seconds
    backup_time: str = "02:00"  # HH:MM format

    # Performance
    parallel_backup: bool = True
    max_workers: int = 4
    chunk_size: int = 1024 * 1024  # 1MB

    # Validation
    verify_backup: bool = True
    checksum_algorithm: str = "sha256"

    # Notifications
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_callback: Optional[Callable[[str, bool], None]] = None


@dataclass
class BackupInfo:
    """Backup information."""

    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: float
    size_bytes: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    checksum: Optional[str] = None
    database_version: Optional[str] = None
    backup_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Recovery information
    recovery_time: float = 0.0
    is_recoverable: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class RecoveryPlan:
    """Recovery plan."""

    target_backup: BackupInfo
    recovery_type: str
    estimated_time: float
    steps: List[str]
    rollback_plan: List[str]
    data_loss_risk: str = "low"


class BackupValidator:
    """Backup validation utilities."""

    @staticmethod
    def validate_backup_file(backup_path: str, checksum: Optional[str] = None) -> bool:
        """Validate backup file integrity."""
        try:
            if not os.path.exists(backup_path):
                return False

            # Check file size
            if os.path.getsize(backup_path) == 0:
                return False

            # Verify checksum if provided
            if checksum:
                calculated_checksum = BackupValidator.calculate_checksum(backup_path)
                if calculated_checksum != checksum:
                    return False

            return True

        except Exception:
            return False

    @staticmethod
    def calculate_checksum(file_path: str, algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    @staticmethod
    def validate_database_backup(backup_path: str) -> bool:
        """Validate database backup integrity."""
        try:
            # Try to open as SQLite database
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()

            # Check if it's a valid SQLite database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            conn.close()

            # Should have at least the migrations table
            return len(tables) > 0

        except Exception:
            return False


class BackupCompressor:
    """Backup compression utilities."""

    @staticmethod
    def compress_file(input_path: str, output_path: str) -> float:
        """Compress a file using gzip."""
        try:
            with open(input_path, "rb") as f_in:
                with gzip.open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Calculate compression ratio
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)

            return compressed_size / original_size if original_size > 0 else 0.0

        except Exception as e:
            raise BackupError(f"Compression failed: {e}")

    @staticmethod
    def decompress_file(input_path: str, output_path: str) -> None:
        """Decompress a gzip file."""
        try:
            with gzip.open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        except Exception as e:
            raise BackupError(f"Decompression failed: {e}")


class BackupEncryptor:
    """Backup encryption utilities."""

    def __init__(self, key: Optional[str] = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """Encrypt a file."""
        try:
            with open(input_path, "rb") as f_in:
                data = f_in.read()

            encrypted_data = self.cipher.encrypt(data)

            with open(output_path, "wb") as f_out:
                f_out.write(encrypted_data)

        except Exception as e:
            raise BackupError(f"Encryption failed: {e}")

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """Decrypt a file."""
        try:
            with open(input_path, "rb") as f_in:
                encrypted_data = f_in.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)

            with open(output_path, "wb") as f_out:
                f_out.write(decrypted_data)

        except Exception as e:
            raise BackupError(f"Decryption failed: {e}")


class BackupManager:
    """Database backup manager."""

    def __init__(self, backend: DatabaseBackend, config: BackupConfig):
        self.backend = backend
        self.config = config
        self.backup_dir = Path(config.backup_directory)
        self._backups: Dict[str, BackupInfo] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        self._backup_thread: Optional[threading.Thread] = None
        self._running = False

        # Initialize components
        self.validator = BackupValidator()
        self.compressor = BackupCompressor()
        self.encryptor = (
            BackupEncryptor(config.encryption_key) if config.encryption else None
        )

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Load existing backups
        self._load_backup_index()

    def _initialize(self) -> None:
        """Initialize backup manager components."""
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)

        # Load existing backups
        self._load_backup_index()

    def _load_backup_index(self) -> None:
        """Load backup index from disk."""
        index_file = self.backup_dir / "backup_index.json"

        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)

                for backup_data in data.get("backups", []):
                    backup_info = BackupInfo(**backup_data)
                    self._backups[backup_info.backup_id] = backup_info

                self._logger.info(f"Loaded {len(self._backups)} backup records")

            except Exception as e:
                self._logger.error(f"Failed to load backup index: {e}")

    def _save_backup_index(self) -> None:
        """Save backup index to disk."""
        index_file = self.backup_dir / "backup_index.json"

        try:
            data = {
                "backups": [
                    {
                        "backup_id": backup.backup_id,
                        "backup_type": backup.backup_type.value,
                        "status": backup.status.value,
                        "created_at": backup.created_at,
                        "size_bytes": backup.size_bytes,
                        "compressed_size": backup.compressed_size,
                        "compression_ratio": backup.compression_ratio,
                        "checksum": backup.checksum,
                        "database_version": backup.database_version,
                        "backup_path": backup.backup_path,
                        "metadata": backup.metadata,
                        "recovery_time": backup.recovery_time,
                        "is_recoverable": backup.is_recoverable,
                        "dependencies": backup.dependencies,
                    }
                    for backup in self._backups.values()
                ]
            }

            with open(index_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self._logger.error(f"Failed to save backup index: {e}")

    def create_backup(
        self,
        backup_type: Optional[BackupType] = None,
        database_path: Optional[str] = None,
    ) -> BackupInfo:
        """Create a new backup."""
        with self._lock:
            backup_type = backup_type or self.config.backup_type
            backup_id = self._generate_backup_id(backup_type)

            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                status=BackupStatus.RUNNING,
                created_at=time.time(),
                backup_path="",
            )

            try:
                self._logger.info(f"Creating {backup_type.value} backup: {backup_id}")

                # Create backup file
                backup_path = self._create_backup_file(backup_info, database_path)
                backup_info.backup_path = str(backup_path)

                # Calculate file size
                backup_info.size_bytes = os.path.getsize(backup_path)

                # Compress if enabled
                if self.config.compression:
                    compressed_path = backup_path.with_suffix(
                        backup_path.suffix + ".gz"
                    )
                    compression_ratio = self.compressor.compress_file(
                        backup_path, compressed_path
                    )
                    backup_info.compression_ratio = compression_ratio
                    backup_info.compressed_size = os.path.getsize(compressed_path)

                    # Remove original file
                    os.remove(backup_path)
                    backup_info.backup_path = str(compressed_path)
                    backup_path = compressed_path

                # Encrypt if enabled
                if self.config.encryption and self.encryptor:
                    encrypted_path = backup_path.with_suffix(
                        backup_path.suffix + ".enc"
                    )
                    self.encryptor.encrypt_file(backup_path, encrypted_path)
                    backup_info.compressed_size = os.path.getsize(encrypted_path)

                    # Remove unencrypted file
                    os.remove(backup_path)
                    backup_info.backup_path = str(encrypted_path)
                    backup_path = encrypted_path

                # Calculate checksum
                if self.config.verify_backup:
                    backup_info.checksum = self.validator.calculate_checksum(
                        backup_path, self.config.checksum_algorithm
                    )

                # Validate backup
                if self.config.verify_backup:
                    if not self.validator.validate_backup_file(
                        backup_path, backup_info.checksum
                    ):
                        raise BackupCorruptionError("Backup validation failed")

                # Update status
                backup_info.status = BackupStatus.COMPLETED
                backup_info.is_recoverable = True

                # Store backup info
                self._backups[backup_id] = backup_info
                self._save_backup_index()

                self._logger.info(f"Backup {backup_id} created successfully")

                # Send notification
                if self.config.notify_on_success and self.config.notification_callback:
                    self.config.notification_callback(
                        f"Backup {backup_id} completed successfully", True
                    )

                return backup_info

            except Exception as e:
                backup_info.status = BackupStatus.FAILED
                backup_info.is_recoverable = False

                self._logger.error(f"Backup {backup_id} failed: {e}")

                # Send notification
                if self.config.notify_on_failure and self.config.notification_callback:
                    self.config.notification_callback(
                        f"Backup {backup_id} failed: {e}", False
                    )

                raise BackupError(f"Backup creation failed: {e}")

    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID."""
        timestamp = int(time.time())
        return f"backup_{timestamp}_{backup_type.value}"

    def _create_backup_file(
        self, backup_info: BackupInfo, database_path: Optional[str] = None
    ) -> Path:
        """Create backup file."""
        filename = f"{backup_info.backup_id}.db"
        backup_path = self.backup_dir / filename

        # Use provided database path or backend's path
        if database_path:
            if os.path.exists(database_path):
                shutil.copy2(database_path, backup_path)
            else:
                raise BackupError(f"Database file not found: {database_path}")
        elif hasattr(self.backend, "_connection") and self.backend._connection:
            # Direct SQLite backup
            backup_conn = sqlite3.connect(backup_path)
            self.backend._connection.backup(backup_conn)
            backup_conn.close()
        else:
            # Fallback: copy database file
            db_path = Path(self.backend.config.database_path)
            if db_path.exists():
                shutil.copy2(db_path, backup_path)
            else:
                raise BackupError("Database file not found")

        return backup_path

    def restore_backup(self, backup_id: str, target_path: Optional[str] = None) -> None:
        """Restore from backup."""
        with self._lock:
            if backup_id not in self._backups:
                raise BackupNotFoundError(f"Backup {backup_id} not found")

            backup_info = self._backups[backup_id]

            if not backup_info.is_recoverable:
                raise RecoveryError(f"Backup {backup_id} is not recoverable")

            try:
                self._logger.info(f"Restoring backup: {backup_id}")

                # Determine target path
                if target_path is None:
                    target_path = self.backend.config.database_path

                # Create temporary file for restoration
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = temp_file.name

                try:
                    # Copy backup to temporary file
                    shutil.copy2(backup_info.backup_path, temp_path)

                    # Decrypt if needed
                    if self.config.encryption and self.encryptor:
                        decrypted_path = temp_path + ".dec"
                        self.encryptor.decrypt_file(temp_path, decrypted_path)
                        os.remove(temp_path)
                        temp_path = decrypted_path

                    # Decompress if needed
                    if self.config.compression:
                        decompressed_path = temp_path + ".db"
                        self.compressor.decompress_file(temp_path, decompressed_path)
                        os.remove(temp_path)
                        temp_path = decompressed_path

                    # Validate restored database
                    if not self.validator.validate_database_backup(temp_path):
                        raise BackupCorruptionError(
                            "Restored database validation failed"
                        )

                    # Replace current database
                    shutil.move(temp_path, target_path)

                    self._logger.info(f"Backup {backup_id} restored successfully")

                finally:
                    # Clean up temporary files
                    for path in [temp_path, temp_path + ".dec", temp_path + ".db"]:
                        if os.path.exists(path):
                            os.remove(path)

            except Exception as e:
                self._logger.error(f"Backup restoration failed: {e}")
                raise RecoveryError(f"Backup restoration failed: {e}")

    def list_backups(self) -> List[BackupInfo]:
        """List all backups."""
        with self._lock:
            return list(self._backups.values())

    def get_backup(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup information."""
        with self._lock:
            return self._backups.get(backup_id)

    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup info by ID."""
        with self._lock:
            return self._backups.get(backup_id)

    def delete_backup(self, backup_id: str) -> None:
        """Delete a backup."""
        with self._lock:
            if backup_id not in self._backups:
                raise BackupNotFoundError(f"Backup {backup_id} not found")

            backup_info = self._backups[backup_id]

            try:
                # Remove backup file
                if os.path.exists(backup_info.backup_path):
                    os.remove(backup_info.backup_path)

                # Remove from index
                del self._backups[backup_id]
                self._save_backup_index()

                self._logger.info(f"Backup {backup_id} deleted")

            except Exception as e:
                self._logger.error(f"Failed to delete backup {backup_id}: {e}")
                raise BackupError(f"Failed to delete backup: {e}")

    def cleanup_old_backups(self) -> int:
        """Clean up old backups based on retention policy."""
        with self._lock:
            if not self.config.auto_cleanup:
                return 0

            current_time = time.time()
            cutoff_time = current_time - (self.config.retention_days * 24 * 3600)

            # Get backups to delete
            backups_to_delete = []

            # Sort by creation time (oldest first)
            sorted_backups = sorted(self._backups.values(), key=lambda b: b.created_at)

            # Keep only the most recent backups
            if len(sorted_backups) > self.config.max_backups:
                excess_count = len(sorted_backups) - self.config.max_backups
                backups_to_delete.extend(sorted_backups[:excess_count])

            # Add old backups
            for backup in sorted_backups:
                if backup.created_at < cutoff_time:
                    backups_to_delete.append(backup)

            # Delete backups
            deleted_count = 0
            for backup in backups_to_delete:
                try:
                    self.delete_backup(backup.backup_id)
                    deleted_count += 1
                except Exception as e:
                    self._logger.error(
                        f"Failed to delete old backup {backup.backup_id}: {e}"
                    )

            if deleted_count > 0:
                self._logger.info(f"Cleaned up {deleted_count} old backups")

            return deleted_count

    def start_auto_backup(self) -> None:
        """Start automatic backup scheduling."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._backup_thread = threading.Thread(
                target=self._backup_worker, daemon=True
            )
            self._backup_thread.start()

            self._logger.info("Started automatic backup scheduling")

    def stop_auto_backup(self) -> None:
        """Stop automatic backup scheduling."""
        with self._lock:
            self._running = False
            if self._backup_thread:
                self._backup_thread.join(timeout=5.0)

            self._logger.info("Stopped automatic backup scheduling")

    def _backup_worker(self) -> None:
        """Background backup worker."""
        while self._running:
            try:
                # Wait for next backup time
                time.sleep(self.config.backup_interval)

                if not self._running:
                    break

                # Create backup
                self.create_backup()

                # Cleanup old backups
                self.cleanup_old_backups()

            except Exception as e:
                self._logger.error(f"Automatic backup error: {e}")

    def create_recovery_plan(self, backup_id: str) -> RecoveryPlan:
        """Create a recovery plan for a backup."""
        with self._lock:
            if backup_id not in self._backups:
                raise BackupNotFoundError(f"Backup {backup_id} not found")

            backup_info = self._backups[backup_id]

            steps = [
                f"1. Stop database connections",
                f"2. Create current database backup",
                f"3. Restore backup {backup_id}",
                f"4. Validate restored database",
                f"5. Restart database connections",
            ]

            rollback_plan = [
                f"1. Stop database connections",
                f"2. Restore current database backup",
                f"3. Restart database connections",
            ]

            return RecoveryPlan(
                target_backup=backup_info,
                recovery_type="full_restore",
                estimated_time=30.0,  # 30 seconds estimate
                steps=steps,
                rollback_plan=rollback_plan,
                data_loss_risk="low" if backup_info.is_recoverable else "high",
            )

    def __enter__(self):
        """Context manager entry."""
        if self.config.auto_backup:
            self.start_auto_backup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_auto_backup()
