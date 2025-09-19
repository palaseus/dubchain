"""Log rotation and retention for DubChain.

This module provides log rotation, retention policies, and compression
capabilities for the DubChain logging system.
"""

import gzip
import logging
import os
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .core import LogEntry, LogHandler


class RotationTrigger(Enum):
    """Rotation trigger types."""

    SIZE = "size"
    TIME = "time"
    MANUAL = "manual"


class RetentionPolicy(Enum):
    """Retention policy types."""

    COUNT = "count"
    AGE = "age"
    SIZE = "size"


@dataclass
class RotationPolicy:
    """Log rotation policy configuration."""

    trigger: RotationTrigger
    max_size: int = 10 * 1024 * 1024  # 10MB
    max_age: float = 86400.0  # 1 day
    max_files: int = 5
    backup_count: int = 5
    compress: bool = True
    compress_level: int = 6
    suffix_format: str = "%Y-%m-%d_%H-%M-%S"

    def should_rotate(self, file_path: str) -> bool:
        """Check if rotation is needed."""
        if not os.path.exists(file_path):
            return False

        if self.trigger == RotationTrigger.SIZE:
            return os.path.getsize(file_path) >= self.max_size
        elif self.trigger == RotationTrigger.TIME:
            file_age = time.time() - os.path.getmtime(file_path)
            return file_age >= self.max_age
        else:
            return False


@dataclass
class RetentionPolicyConfig:
    """Log retention policy configuration."""

    policy_type: RetentionPolicy
    max_count: int = 10
    max_age: float = 7 * 86400.0  # 7 days
    max_size: int = 100 * 1024 * 1024  # 100MB
    compress_after: int = 1  # Compress after 1 day

    def should_retain(self, file_path: str) -> bool:
        """Check if file should be retained."""
        if not os.path.exists(file_path):
            return False

        if self.policy_type == RetentionPolicy.COUNT:
            # This would need to be checked against total file count
            return True
        elif self.policy_type == RetentionPolicy.AGE:
            file_age = time.time() - os.path.getmtime(file_path)
            return file_age <= self.max_age
        elif self.policy_type == RetentionPolicy.SIZE:
            # This would need to be checked against total size
            return True
        else:
            return True


class LogRotator:
    """Log rotator for managing log file rotation."""

    def __init__(
        self,
        file_path: str,
        rotation_policy: RotationPolicy,
        retention_policy: RetentionPolicyConfig,
    ):
        self.file_path = file_path
        self.rotation_policy = rotation_policy
        self.retention_policy = retention_policy
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Background thread for rotation
        self._rotation_thread = None
        self._running = False

        self._start_rotation_thread()

    def _start_rotation_thread(self) -> None:
        """Start background rotation thread."""
        self._running = True
        self._rotation_thread = threading.Thread(
            target=self._rotation_worker, daemon=True
        )
        self._rotation_thread.start()

    def _rotation_worker(self) -> None:
        """Background worker for rotation."""
        while self._running:
            try:
                time.sleep(60)  # Check every minute

                if not self._running:
                    break

                if self.rotation_policy.should_rotate(self.file_path):
                    self.rotate()

                self.cleanup_old_files()

            except Exception as e:
                self._logger.error(f"Error in rotation worker: {e}")

    def rotate(self) -> None:
        """Perform log rotation."""
        with self._lock:
            if not os.path.exists(self.file_path):
                return

            try:
                # Generate backup filename
                timestamp = time.strftime(
                    self.rotation_policy.suffix_format, time.gmtime()
                )
                backup_path = f"{self.file_path}.{timestamp}"

                # Move current file to backup
                shutil.move(self.file_path, backup_path)

                # Compress if enabled
                if self.rotation_policy.compress:
                    compressed_path = f"{backup_path}.gz"
                    with open(backup_path, "rb") as f_in:
                        with gzip.open(
                            compressed_path,
                            "wb",
                            compresslevel=self.rotation_policy.compress_level,
                        ) as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(backup_path)
                    backup_path = compressed_path

                self._logger.info(f"Log rotated: {self.file_path} -> {backup_path}")

            except Exception as e:
                self._logger.error(f"Error rotating log file: {e}")

    def cleanup_old_files(self) -> None:
        """Clean up old log files based on retention policy."""
        with self._lock:
            try:
                # Get all backup files
                backup_files = []
                base_name = os.path.basename(self.file_path)
                dir_name = os.path.dirname(self.file_path)

                for filename in os.listdir(dir_name):
                    if filename.startswith(base_name + "."):
                        file_path = os.path.join(dir_name, filename)
                        if os.path.isfile(file_path):
                            backup_files.append(file_path)

                # Sort by modification time
                backup_files.sort(key=os.path.getmtime, reverse=True)

                # Apply retention policy
                if self.retention_policy.policy_type == RetentionPolicy.COUNT:
                    # Keep only max_count files
                    files_to_remove = backup_files[self.retention_policy.max_count :]
                elif self.retention_policy.policy_type == RetentionPolicy.AGE:
                    # Remove files older than max_age
                    cutoff_time = time.time() - self.retention_policy.max_age
                    files_to_remove = [
                        f for f in backup_files if os.path.getmtime(f) < cutoff_time
                    ]
                else:
                    files_to_remove = []

                # Remove old files
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        self._logger.info(f"Removed old log file: {file_path}")
                    except Exception as e:
                        self._logger.error(
                            f"Error removing old log file {file_path}: {e}"
                        )

            except Exception as e:
                self._logger.error(f"Error cleaning up old files: {e}")

    def force_rotate(self) -> None:
        """Force log rotation."""
        self.rotate()

    def get_rotation_status(self) -> Dict[str, Any]:
        """Get rotation status."""
        with self._lock:
            status = {
                "file_path": self.file_path,
                "file_exists": os.path.exists(self.file_path),
                "file_size": os.path.getsize(self.file_path)
                if os.path.exists(self.file_path)
                else 0,
                "file_age": time.time() - os.path.getmtime(self.file_path)
                if os.path.exists(self.file_path)
                else 0,
                "rotation_policy": {
                    "trigger": self.rotation_policy.trigger.value,
                    "max_size": self.rotation_policy.max_size,
                    "max_age": self.rotation_policy.max_age,
                    "max_files": self.rotation_policy.max_files,
                    "compress": self.rotation_policy.compress,
                },
                "retention_policy": {
                    "type": self.retention_policy.policy_type.value,
                    "max_count": self.retention_policy.max_count,
                    "max_age": self.retention_policy.max_age,
                    "max_size": self.retention_policy.max_size,
                },
            }

            # Count backup files
            backup_count = 0
            if os.path.exists(os.path.dirname(self.file_path)):
                base_name = os.path.basename(self.file_path)
                dir_name = os.path.dirname(self.file_path)
                for filename in os.listdir(dir_name):
                    if filename.startswith(base_name + "."):
                        backup_count += 1

            status["backup_files_count"] = backup_count

            return status

    def shutdown(self) -> None:
        """Shutdown rotator."""
        with self._lock:
            self._running = False

            # Wait for rotation thread
            if self._rotation_thread and self._rotation_thread.is_alive():
                self._rotation_thread.join(timeout=5.0)


class CompressionHandler(LogHandler):
    """Compression handler for compressing log files."""

    def __init__(
        self,
        target_handler: LogHandler,
        compression_level: int = 6,
        compress_after: int = 1000,
    ):
        super().__init__()
        self.target_handler = target_handler
        self.compression_level = compression_level
        self.compress_after = compress_after
        self.buffer = []
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry with compression."""
        with self._lock:
            if self.formatter:
                formatted = self.formatter.format(entry)
                self.buffer.append(formatted)

                # Compress and forward if buffer is full
                if len(self.buffer) >= self.compress_after:
                    self._compress_and_forward()

    def _compress_and_forward(self) -> None:
        """Compress buffer and forward to target handler."""
        if not self.buffer:
            return

        try:
            # Compress buffer
            data = "\n".join(self.buffer).encode("utf-8")
            compressed = gzip.compress(data, compresslevel=self.compression_level)

            # Create a compressed log entry
            from .core import LogEntry, LogLevel

            compressed_entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                message=f"Compressed {len(self.buffer)} log entries",
                logger_name="compression_handler",
                context=None,
            )
            compressed_entry.extra = {
                "compressed_size": len(compressed),
                "original_size": len(data),
                "compression_ratio": len(compressed) / len(data),
                "entry_count": len(self.buffer),
            }

            # Forward to target handler
            self.target_handler.emit(compressed_entry)

            # Clear buffer
            self.buffer.clear()

        except Exception as e:
            self._logger.error(f"Error compressing logs: {e}")
            # Clear buffer on error
            self.buffer.clear()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            # Flush remaining buffer
            if self.buffer:
                self._compress_and_forward()

            # Close target handler
            if self.target_handler:
                self.target_handler.close()


class SizeBasedRotator(LogRotator):
    """Size-based log rotator."""

    def __init__(
        self,
        file_path: str,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        compress: bool = True,
    ):
        rotation_policy = RotationPolicy(
            trigger=RotationTrigger.SIZE,
            max_size=max_size,
            backup_count=backup_count,
            compress=compress,
        )

        retention_policy = RetentionPolicyConfig(
            policy_type=RetentionPolicy.COUNT, max_count=backup_count
        )

        super().__init__(file_path, rotation_policy, retention_policy)


class TimeBasedRotator(LogRotator):
    """Time-based log rotator."""

    def __init__(
        self,
        file_path: str,
        max_age: float = 86400.0,  # 1 day
        backup_count: int = 5,
        compress: bool = True,
    ):
        rotation_policy = RotationPolicy(
            trigger=RotationTrigger.TIME,
            max_age=max_age,
            backup_count=backup_count,
            compress=compress,
        )

        retention_policy = RetentionPolicyConfig(
            policy_type=RetentionPolicy.COUNT, max_count=backup_count
        )

        super().__init__(file_path, rotation_policy, retention_policy)


class AgeBasedRetention:
    """Age-based retention policy."""

    def __init__(self, max_age: float = 7 * 86400.0):  # 7 days
        self.max_age = max_age

    def should_retain(self, file_path: str) -> bool:
        """Check if file should be retained based on age."""
        if not os.path.exists(file_path):
            return False

        file_age = time.time() - os.path.getmtime(file_path)
        return file_age <= self.max_age

    def cleanup(self, file_paths: List[str]) -> List[str]:
        """Clean up files based on age."""
        files_to_remove = []

        for file_path in file_paths:
            if not self.should_retain(file_path):
                files_to_remove.append(file_path)

        return files_to_remove


class CountBasedRetention:
    """Count-based retention policy."""

    def __init__(self, max_count: int = 10):
        self.max_count = max_count

    def should_retain(self, file_paths: List[str], file_path: str) -> bool:
        """Check if file should be retained based on count."""
        # Sort files by modification time
        sorted_files = sorted(file_paths, key=os.path.getmtime, reverse=True)

        # Keep only the most recent max_count files
        return file_path in sorted_files[: self.max_count]

    def cleanup(self, file_paths: List[str]) -> List[str]:
        """Clean up files based on count."""
        if len(file_paths) <= self.max_count:
            return []

        # Sort files by modification time
        sorted_files = sorted(file_paths, key=os.path.getmtime, reverse=True)

        # Remove oldest files
        return sorted_files[self.max_count :]


class SizeBasedRetention:
    """Size-based retention policy."""

    def __init__(self, max_size: int = 100 * 1024 * 1024):  # 100MB
        self.max_size = max_size

    def should_retain(self, file_paths: List[str], file_path: str) -> bool:
        """Check if file should be retained based on total size."""
        total_size = sum(os.path.getsize(f) for f in file_paths if os.path.exists(f))
        return total_size <= self.max_size

    def cleanup(self, file_paths: List[str]) -> List[str]:
        """Clean up files based on total size."""
        if not file_paths:
            return []

        # Sort files by modification time (oldest first)
        sorted_files = sorted(file_paths, key=os.path.getmtime)

        total_size = sum(os.path.getsize(f) for f in file_paths if os.path.exists(f))
        files_to_remove = []

        for file_path in sorted_files:
            if total_size <= self.max_size:
                break

            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                files_to_remove.append(file_path)
                total_size -= file_size

        return files_to_remove
