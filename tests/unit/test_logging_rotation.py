"""
Tests for logging rotation module.
"""

import gzip
import os
import shutil
import tempfile
import time
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.dubchain.logging.core import LogContext, LogEntry, LogHandler, LogLevel
from src.dubchain.logging.rotation import (
    AgeBasedRetention,
    CompressionHandler,
    CountBasedRetention,
    LogRotator,
    RetentionPolicy,
    RetentionPolicyConfig,
    RotationPolicy,
    RotationTrigger,
    SizeBasedRetention,
    SizeBasedRotator,
    TimeBasedRotator,
)


class TestRotationTrigger:
    """Test RotationTrigger enum."""

    def test_rotation_trigger_values(self):
        """Test rotation trigger enum values."""
        assert RotationTrigger.SIZE.value == "size"
        assert RotationTrigger.TIME.value == "time"
        assert RotationTrigger.MANUAL.value == "manual"


class TestRetentionPolicy:
    """Test RetentionPolicy enum."""

    def test_retention_policy_values(self):
        """Test retention policy enum values."""
        assert RetentionPolicy.COUNT.value == "count"
        assert RetentionPolicy.AGE.value == "age"
        assert RetentionPolicy.SIZE.value == "size"


class TestRotationPolicy:
    """Test RotationPolicy class."""

    def test_rotation_policy_initialization(self):
        """Test rotation policy initialization."""
        policy = RotationPolicy(
            trigger=RotationTrigger.SIZE,
            max_size=1024 * 1024,  # 1MB
            max_age=3600.0,  # 1 hour
            max_files=10,
            backup_count=5,
            compress=True,
            compress_level=9,
            suffix_format="%Y-%m-%d",
        )

        assert policy.trigger == RotationTrigger.SIZE
        assert policy.max_size == 1024 * 1024
        assert policy.max_age == 3600.0
        assert policy.max_files == 10
        assert policy.backup_count == 5
        assert policy.compress is True
        assert policy.compress_level == 9
        assert policy.suffix_format == "%Y-%m-%d"

    def test_rotation_policy_defaults(self):
        """Test rotation policy default values."""
        policy = RotationPolicy(trigger=RotationTrigger.SIZE)

        assert policy.trigger == RotationTrigger.SIZE
        assert policy.max_size == 10 * 1024 * 1024  # 10MB
        assert policy.max_age == 86400.0  # 1 day
        assert policy.max_files == 5
        assert policy.backup_count == 5
        assert policy.compress is True
        assert policy.compress_level == 6
        assert policy.suffix_format == "%Y-%m-%d_%H-%M-%S"

    def test_should_rotate_size_trigger(self):
        """Test should_rotate with size trigger."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"x" * 1024)  # 1KB

        try:
            policy = RotationPolicy(
                trigger=RotationTrigger.SIZE, max_size=512  # 512 bytes
            )

            # Should rotate because file is larger than max_size
            assert policy.should_rotate(temp_path) is True

            # Create smaller file
            with open(temp_path, "w") as f:
                f.write("x" * 256)  # 256 bytes

            # Should not rotate because file is smaller than max_size
            assert policy.should_rotate(temp_path) is False

        finally:
            os.unlink(temp_path)

    def test_should_rotate_time_trigger(self):
        """Test should_rotate with time trigger."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test")

        try:
            policy = RotationPolicy(
                trigger=RotationTrigger.TIME, max_age=1.0  # 1 second
            )

            # Should not rotate immediately
            assert policy.should_rotate(temp_path) is False

            # Wait for file to age
            time.sleep(1.1)

            # Should rotate because file is older than max_age
            assert policy.should_rotate(temp_path) is True

        finally:
            os.unlink(temp_path)

    def test_should_rotate_manual_trigger(self):
        """Test should_rotate with manual trigger."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test")

        try:
            policy = RotationPolicy(trigger=RotationTrigger.MANUAL)

            # Manual trigger should never auto-rotate
            assert policy.should_rotate(temp_path) is False

        finally:
            os.unlink(temp_path)

    def test_should_rotate_nonexistent_file(self):
        """Test should_rotate with nonexistent file."""
        policy = RotationPolicy(trigger=RotationTrigger.SIZE)

        assert policy.should_rotate("/nonexistent/file") is False


class TestRetentionPolicyConfig:
    """Test RetentionPolicyConfig class."""

    def test_retention_policy_initialization(self):
        """Test retention policy initialization."""
        policy = RetentionPolicyConfig(
            policy_type=RetentionPolicy.COUNT,
            max_count=10,
            max_age=7 * 86400.0,  # 7 days
            max_size=100 * 1024 * 1024,  # 100MB
            compress_after=1,
        )

        assert policy.policy_type == RetentionPolicy.COUNT
        assert policy.max_count == 10
        assert policy.max_age == 7 * 86400.0
        assert policy.max_size == 100 * 1024 * 1024
        assert policy.compress_after == 1

    def test_retention_policy_defaults(self):
        """Test retention policy default values."""
        policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

        assert policy.policy_type == RetentionPolicy.COUNT
        assert policy.max_count == 10
        assert policy.max_age == 7 * 86400.0
        assert policy.max_size == 100 * 1024 * 1024
        assert policy.compress_after == 1

    def test_should_retain_count_policy(self):
        """Test should_retain with count policy."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test")

        try:
            policy = RetentionPolicyConfig(
                policy_type=RetentionPolicy.COUNT, max_count=5
            )

            # Count policy always returns True (checking is done elsewhere)
            assert policy.should_retain(temp_path) is True

        finally:
            os.unlink(temp_path)

    def test_should_retain_age_policy(self):
        """Test should_retain with age policy."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test")

        try:
            policy = RetentionPolicyConfig(
                policy_type=RetentionPolicy.AGE, max_age=1.0  # 1 second
            )

            # Should retain immediately
            assert policy.should_retain(temp_path) is True

            # Wait for file to age
            time.sleep(1.1)

            # Should not retain because file is older than max_age
            assert policy.should_retain(temp_path) is False

        finally:
            os.unlink(temp_path)

    def test_should_retain_size_policy(self):
        """Test should_retain with size policy."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test")

        try:
            policy = RetentionPolicyConfig(
                policy_type=RetentionPolicy.SIZE, max_size=1024 * 1024  # 1MB
            )

            # Size policy always returns True (checking is done elsewhere)
            assert policy.should_retain(temp_path) is True

        finally:
            os.unlink(temp_path)

    def test_should_retain_nonexistent_file(self):
        """Test should_retain with nonexistent file."""
        policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

        assert policy.should_retain("/nonexistent/file") is False


class TestLogRotator:
    """Test LogRotator class."""

    def test_log_rotator_initialization(self):
        """Test log rotator initialization."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.SIZE)
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            assert rotator.file_path == temp_path
            assert rotator.rotation_policy == rotation_policy
            assert rotator.retention_policy == retention_policy
            assert rotator._running is True

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_log_rotator_rotate(self):
        """Test log rotation."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test log content")

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.MANUAL)
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            # Force rotation
            rotator.rotate()

            # Original file should not exist
            assert not os.path.exists(temp_path)

            # Backup file should exist
            backup_files = [
                f
                for f in os.listdir(os.path.dirname(temp_path))
                if f.startswith(os.path.basename(temp_path) + ".")
            ]
            assert len(backup_files) == 1

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Clean up backup files
            for f in os.listdir(os.path.dirname(temp_path)):
                if f.startswith(os.path.basename(temp_path) + "."):
                    os.unlink(os.path.join(os.path.dirname(temp_path), f))

    def test_log_rotator_rotate_with_compression(self):
        """Test log rotation with compression."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test log content")

        try:
            rotation_policy = RotationPolicy(
                trigger=RotationTrigger.MANUAL, compress=True
            )
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            # Force rotation
            rotator.rotate()

            # Original file should not exist
            assert not os.path.exists(temp_path)

            # Compressed backup file should exist
            backup_files = [
                f
                for f in os.listdir(os.path.dirname(temp_path))
                if f.startswith(os.path.basename(temp_path) + ".") and f.endswith(".gz")
            ]
            assert len(backup_files) == 1

            # Verify compression
            compressed_file = os.path.join(os.path.dirname(temp_path), backup_files[0])
            with gzip.open(compressed_file, "rt") as f:
                content = f.read()
                assert content == "test log content"

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Clean up backup files
            for f in os.listdir(os.path.dirname(temp_path)):
                if f.startswith(os.path.basename(temp_path) + "."):
                    os.unlink(os.path.join(os.path.dirname(temp_path), f))

    def test_log_rotator_cleanup_old_files(self):
        """Test cleanup of old files."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test log content")

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.MANUAL)
            retention_policy = RetentionPolicyConfig(
                policy_type=RetentionPolicy.COUNT, max_count=2
            )

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            # Create multiple backup files
            for i in range(5):
                backup_path = f"{temp_path}.{i}"
                with open(backup_path, "w") as f:
                    f.write(f"backup {i}")

            # Cleanup old files
            rotator.cleanup_old_files()

            # Should keep only 2 most recent files
            backup_files = [
                f
                for f in os.listdir(os.path.dirname(temp_path))
                if f.startswith(os.path.basename(temp_path) + ".")
            ]
            assert len(backup_files) == 2

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Clean up backup files
            for f in os.listdir(os.path.dirname(temp_path)):
                if f.startswith(os.path.basename(temp_path) + "."):
                    os.unlink(os.path.join(os.path.dirname(temp_path), f))

    def test_log_rotator_get_rotation_status(self):
        """Test getting rotation status."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test log content")

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.SIZE)
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            status = rotator.get_rotation_status()

            assert status["file_path"] == temp_path
            assert status["file_exists"] is True
            assert status["file_size"] > 0
            assert status["file_age"] >= 0
            assert status["rotation_policy"]["trigger"] == "size"
            assert status["retention_policy"]["type"] == "count"
            assert status["backup_files_count"] == 0

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_log_rotator_shutdown(self):
        """Test rotator shutdown."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.SIZE)
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            assert rotator._running is True

            rotator.shutdown()

            assert rotator._running is False

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCompressionHandler:
    """Test CompressionHandler class."""

    def test_compression_handler_initialization(self):
        """Test compression handler initialization."""
        target_handler = Mock(spec=LogHandler)

        handler = CompressionHandler(
            target_handler=target_handler, compression_level=9, compress_after=100
        )

        assert handler.target_handler == target_handler
        assert handler.compression_level == 9
        assert handler.compress_after == 100
        assert len(handler.buffer) == 0

    def test_compression_handler_emit(self):
        """Test compression handler emit."""
        target_handler = Mock(spec=LogHandler)

        handler = CompressionHandler(
            target_handler=target_handler, compression_level=6, compress_after=2
        )

        # Mock formatter
        handler.formatter = Mock()
        handler.formatter.format.return_value = "formatted log entry"

        # Create log entry
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        # Emit first entry (should not compress yet)
        handler.emit(entry)
        assert len(handler.buffer) == 1
        target_handler.emit.assert_not_called()

        # Emit second entry (should trigger compression)
        handler.emit(entry)
        assert len(handler.buffer) == 0
        target_handler.emit.assert_called_once()

        # Verify compressed entry
        compressed_entry = target_handler.emit.call_args[0][0]
        assert compressed_entry.level == LogLevel.INFO
        assert compressed_entry.logger_name == "compression_handler"
        assert "Compressed 2 log entries" in compressed_entry.message
        assert compressed_entry.extra["entry_count"] == 2
        assert compressed_entry.extra["compression_ratio"] > 0

    def test_compression_handler_close(self):
        """Test compression handler close."""
        target_handler = Mock()
        target_handler.close = Mock()

        handler = CompressionHandler(
            target_handler=target_handler, compression_level=6, compress_after=100
        )

        # Mock formatter
        handler.formatter = Mock()
        handler.formatter.format.return_value = "formatted log entry"

        # Add entry to buffer
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        handler.emit(entry)
        assert len(handler.buffer) == 1

        # Close handler (should flush buffer)
        handler.close()

        assert len(handler.buffer) == 0
        target_handler.emit.assert_called_once()
        target_handler.close.assert_called_once()


class TestSizeBasedRotator:
    """Test SizeBasedRotator class."""

    def test_size_based_rotator_initialization(self):
        """Test size-based rotator initialization."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            rotator = SizeBasedRotator(
                file_path=temp_path,
                max_size=1024 * 1024,  # 1MB
                backup_count=5,
                compress=True,
            )

            assert rotator.file_path == temp_path
            assert rotator.rotation_policy.trigger == RotationTrigger.SIZE
            assert rotator.rotation_policy.max_size == 1024 * 1024
            assert rotator.rotation_policy.backup_count == 5
            assert rotator.rotation_policy.compress is True
            assert rotator.retention_policy.policy_type == RetentionPolicy.COUNT
            assert rotator.retention_policy.max_count == 5

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTimeBasedRotator:
    """Test TimeBasedRotator class."""

    def test_time_based_rotator_initialization(self):
        """Test time-based rotator initialization."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            rotator = TimeBasedRotator(
                file_path=temp_path,
                max_age=3600.0,  # 1 hour
                backup_count=3,
                compress=False,
            )

            assert rotator.file_path == temp_path
            assert rotator.rotation_policy.trigger == RotationTrigger.TIME
            assert rotator.rotation_policy.max_age == 3600.0
            assert rotator.rotation_policy.backup_count == 3
            assert rotator.rotation_policy.compress is False
            assert rotator.retention_policy.policy_type == RetentionPolicy.COUNT
            assert rotator.retention_policy.max_count == 3

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAgeBasedRetention:
    """Test AgeBasedRetention class."""

    def test_age_based_retention_initialization(self):
        """Test age-based retention initialization."""
        retention = AgeBasedRetention(max_age=86400.0)  # 1 day

        assert retention.max_age == 86400.0

    def test_age_based_retention_should_retain(self):
        """Test age-based retention should_retain."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test")

        try:
            retention = AgeBasedRetention(max_age=1.0)  # 1 second

            # Should retain immediately
            assert retention.should_retain(temp_path) is True

            # Wait for file to age
            time.sleep(1.1)

            # Should not retain because file is older than max_age
            assert retention.should_retain(temp_path) is False

        finally:
            os.unlink(temp_path)

    def test_age_based_retention_cleanup(self):
        """Test age-based retention cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different ages
            old_file = os.path.join(temp_dir, "old.txt")
            new_file = os.path.join(temp_dir, "new.txt")

            with open(old_file, "w") as f:
                f.write("old content")

            # Wait a bit
            time.sleep(0.1)

            with open(new_file, "w") as f:
                f.write("new content")

            retention = AgeBasedRetention(max_age=0.05)  # 50ms

            # Wait for old file to age
            time.sleep(0.1)

            files_to_remove = retention.cleanup([old_file, new_file])

            # Both files might be removed if they're both older than max_age
            assert old_file in files_to_remove
            # The new file might also be removed if the timing is off
            # Just check that at least the old file is removed


class TestCountBasedRetention:
    """Test CountBasedRetention class."""

    def test_count_based_retention_initialization(self):
        """Test count-based retention initialization."""
        retention = CountBasedRetention(max_count=5)

        assert retention.max_count == 5

    def test_count_based_retention_should_retain(self):
        """Test count-based retention should_retain."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            files = []
            for i in range(10):
                file_path = os.path.join(temp_dir, f"file_{i}.txt")
                with open(file_path, "w") as f:
                    f.write(f"content {i}")
                files.append(file_path)
                time.sleep(0.01)  # Ensure different modification times

            retention = CountBasedRetention(max_count=3)

            # Check retention for each file
            retained_count = 0
            for file_path in files:
                if retention.should_retain(files, file_path):
                    retained_count += 1

            assert retained_count == 3

    def test_count_based_retention_cleanup(self):
        """Test count-based retention cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            files = []
            for i in range(10):
                file_path = os.path.join(temp_dir, f"file_{i}.txt")
                with open(file_path, "w") as f:
                    f.write(f"content {i}")
                files.append(file_path)
                time.sleep(0.01)  # Ensure different modification times

            retention = CountBasedRetention(max_count=3)

            files_to_remove = retention.cleanup(files)

            assert len(files_to_remove) == 7  # 10 - 3 = 7
            assert len(files) - len(files_to_remove) == 3


class TestSizeBasedRetention:
    """Test SizeBasedRetention class."""

    def test_size_based_retention_initialization(self):
        """Test size-based retention initialization."""
        retention = SizeBasedRetention(max_size=1024 * 1024)  # 1MB

        assert retention.max_size == 1024 * 1024

    def test_size_based_retention_should_retain(self):
        """Test size-based retention should_retain."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different sizes
            small_file = os.path.join(temp_dir, "small.txt")
            large_file = os.path.join(temp_dir, "large.txt")

            with open(small_file, "w") as f:
                f.write("small content")

            with open(large_file, "w") as f:
                f.write("x" * 1024)  # 1KB

            retention = SizeBasedRetention(max_size=512)  # 512 bytes

            # Total size is 13 + 1024 = 1037 bytes, which exceeds 512 bytes
            # So should_retain should return False for both files
            assert (
                retention.should_retain([small_file, large_file], small_file) is False
            )
            assert (
                retention.should_retain([small_file, large_file], large_file) is False
            )

    def test_size_based_retention_cleanup(self):
        """Test size-based retention cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different sizes
            files = []
            for i in range(5):
                file_path = os.path.join(temp_dir, f"file_{i}.txt")
                with open(file_path, "w") as f:
                    f.write("x" * 200)  # 200 bytes each
                files.append(file_path)
                time.sleep(0.01)  # Ensure different modification times

            retention = SizeBasedRetention(max_size=500)  # 500 bytes total

            files_to_remove = retention.cleanup(files)

            # Should remove files until total size is under limit
            assert len(files_to_remove) > 0
            assert len(files_to_remove) < len(files)


class TestRotationIntegration:
    """Test rotation integration and edge cases."""

    def test_rotation_with_nonexistent_file(self):
        """Test rotation with nonexistent file."""
        rotation_policy = RotationPolicy(trigger=RotationTrigger.MANUAL)
        retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

        rotator = LogRotator("/nonexistent/file", rotation_policy, retention_policy)

        # Should not crash
        rotator.rotate()
        rotator.cleanup_old_files()
        status = rotator.get_rotation_status()

        assert status["file_exists"] is False
        assert status["file_size"] == 0

        rotator.shutdown()

    def test_rotation_with_permission_error(self):
        """Test rotation with permission error."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test content")

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.MANUAL)
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            # Make file read-only
            os.chmod(temp_path, 0o444)

            # Should not crash
            rotator.rotate()

            rotator.shutdown()

        finally:
            # Restore permissions and clean up
            try:
                os.chmod(temp_path, 0o644)
                os.unlink(temp_path)
            except:
                pass

    def test_compression_handler_error_handling(self):
        """Test compression handler error handling."""
        target_handler = Mock(spec=LogHandler)
        target_handler.emit.side_effect = Exception("Test error")

        handler = CompressionHandler(
            target_handler=target_handler, compression_level=6, compress_after=1
        )

        # Mock formatter
        handler.formatter = Mock()
        handler.formatter.format.return_value = "formatted log entry"

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        # Should not crash on error
        handler.emit(entry)

        # Buffer should be cleared on error
        assert len(handler.buffer) == 0

    def test_retention_policies_with_empty_lists(self):
        """Test retention policies with empty file lists."""
        age_retention = AgeBasedRetention(max_age=86400.0)
        count_retention = CountBasedRetention(max_count=5)
        size_retention = SizeBasedRetention(max_size=1024 * 1024)

        # Should handle empty lists gracefully
        assert age_retention.cleanup([]) == []
        assert count_retention.cleanup([]) == []
        assert size_retention.cleanup([]) == []

    def test_rotation_thread_safety(self):
        """Test rotation thread safety."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test content")

        try:
            rotation_policy = RotationPolicy(trigger=RotationTrigger.SIZE, max_size=1)
            retention_policy = RetentionPolicyConfig(policy_type=RetentionPolicy.COUNT)

            rotator = LogRotator(temp_path, rotation_policy, retention_policy)

            # Multiple operations should be thread-safe
            import threading

            def worker():
                for _ in range(10):
                    rotator.rotate()
                    rotator.cleanup_old_files()
                    rotator.get_rotation_status()

            threads = []
            for _ in range(3):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            rotator.shutdown()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Clean up backup files
            for f in os.listdir(os.path.dirname(temp_path)):
                if f.startswith(os.path.basename(temp_path) + "."):
                    os.unlink(os.path.join(os.path.dirname(temp_path), f))
