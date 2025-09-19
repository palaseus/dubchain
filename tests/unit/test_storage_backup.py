"""Tests for storage backup module."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.storage.backup import (
    BackupCompressor,
    BackupConfig,
    BackupCorruptionError,
    BackupEncryptor,
    BackupError,
    BackupInfo,
    BackupManager,
    BackupNotFoundError,
    BackupStatus,
    BackupType,
    BackupValidator,
    RecoveryError,
    RecoveryPlan,
)


class TestBackupExceptions:
    """Test backup exception classes."""

    def test_backup_error(self):
        """Test base backup error."""
        error = BackupError("Test backup error")
        assert str(error) == "Test backup error"
        assert isinstance(error, Exception)

    def test_backup_corruption_error(self):
        """Test backup corruption error."""
        error = BackupCorruptionError("Backup corrupted")
        assert str(error) == "Backup corrupted"
        assert isinstance(error, BackupError)

    def test_backup_not_found_error(self):
        """Test backup not found error."""
        error = BackupNotFoundError("Backup not found")
        assert str(error) == "Backup not found"
        assert isinstance(error, BackupError)

    def test_recovery_error(self):
        """Test recovery error."""
        error = RecoveryError("Recovery failed")
        assert str(error) == "Recovery failed"
        assert isinstance(error, BackupError)


class TestBackupType:
    """Test BackupType enum."""

    def test_backup_type_values(self):
        """Test backup type values."""
        assert BackupType.FULL.value == "full"
        assert BackupType.INCREMENTAL.value == "incremental"
        assert BackupType.DIFFERENTIAL.value == "differential"
        assert BackupType.SNAPSHOT.value == "snapshot"


class TestBackupConfig:
    """Test BackupConfig functionality."""

    def test_backup_config_creation(self):
        """Test creating backup config."""
        config = BackupConfig(
            backup_directory="/tmp/backups",
            max_backups=15,
            compression=True,
            encryption=True,
            backup_interval=1800,
        )

        assert config.backup_directory == "/tmp/backups"
        assert config.max_backups == 15
        assert config.compression is True
        assert config.encryption is True
        assert config.backup_interval == 1800

    def test_backup_config_defaults(self):
        """Test backup config defaults."""
        config = BackupConfig()

        assert config.backup_directory == "backups"
        assert config.max_backups == 10
        assert config.compression is True
        assert config.encryption is False
        assert config.backup_interval == 3600


class TestBackupManager:
    """Test BackupManager functionality."""

    @pytest.fixture
    def backup_config(self):
        """Fixture for backup configuration."""
        return BackupConfig(
            backup_directory="/tmp/test_backups",
            max_backups=3,
            compression=False,
            encryption=False,
            verify_backup=False,
        )

    @pytest.fixture
    def backup_manager(self, backup_config):
        """Fixture for backup manager."""
        mock_backend = Mock()
        return BackupManager(mock_backend, backup_config)

    def test_backup_manager_creation(self, backup_config):
        """Test creating backup manager."""
        mock_backend = Mock()
        manager = BackupManager(mock_backend, backup_config)

        assert manager.config == backup_config
        assert manager.backend == mock_backend
        assert isinstance(manager._backups, dict)
        assert manager._running is False

    @patch("os.makedirs")
    def test_backup_manager_initialization(self, mock_makedirs, backup_manager):
        """Test backup manager initialization."""
        backup_manager._initialize()

        mock_makedirs.assert_called_once()

    @patch("os.path.getsize")
    @patch("shutil.copy2")
    @patch("os.path.exists")
    def test_create_full_backup(
        self, mock_exists, mock_copy, mock_getsize, backup_manager
    ):
        """Test creating full backup."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # Mock file size

        result = backup_manager.create_backup(BackupType.FULL, "test.db")

        assert result is not None
        assert mock_copy.called

    def test_list_backups(self, backup_manager):
        """Test listing backups."""
        backups = backup_manager.list_backups()

        assert isinstance(backups, list)

    def test_get_backup_info(self, backup_manager):
        """Test getting backup info."""
        # Should handle case when backup doesn't exist
        info = backup_manager.get_backup_info("nonexistent")
        assert info is None


class TestBackupStatus:
    """Test BackupStatus enum."""

    def test_backup_status_values(self):
        """Test backup status values."""
        assert BackupStatus.PENDING.value == "pending"
        assert BackupStatus.RUNNING.value == "running"
        assert BackupStatus.COMPLETED.value == "completed"
        assert BackupStatus.FAILED.value == "failed"
        assert BackupStatus.CORRUPTED.value == "corrupted"


class TestBackupInfo:
    """Test BackupInfo dataclass."""

    def test_backup_info_creation(self):
        """Test creating backup info."""
        backup_info = BackupInfo(
            backup_id="test_backup_1",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=1234567890.0,
            size_bytes=1024,
            compressed_size=512,
            compression_ratio=0.5,
            checksum="abc123",
            database_version="1.0.0",
            backup_path="/tmp/backup.db",
            metadata={"key": "value"},
            recovery_time=30.0,
            is_recoverable=True,
            dependencies=["backup_0"],
        )

        assert backup_info.backup_id == "test_backup_1"
        assert backup_info.backup_type == BackupType.FULL
        assert backup_info.status == BackupStatus.COMPLETED
        assert backup_info.created_at == 1234567890.0
        assert backup_info.size_bytes == 1024
        assert backup_info.compressed_size == 512
        assert backup_info.compression_ratio == 0.5
        assert backup_info.checksum == "abc123"
        assert backup_info.database_version == "1.0.0"
        assert backup_info.backup_path == "/tmp/backup.db"
        assert backup_info.metadata == {"key": "value"}
        assert backup_info.recovery_time == 30.0
        assert backup_info.is_recoverable is True
        assert backup_info.dependencies == ["backup_0"]

    def test_backup_info_defaults(self):
        """Test backup info with default values."""
        backup_info = BackupInfo(
            backup_id="test_backup_1",
            backup_type=BackupType.FULL,
            status=BackupStatus.PENDING,
            created_at=1234567890.0,
        )

        assert backup_info.size_bytes == 0
        assert backup_info.compressed_size == 0
        assert backup_info.compression_ratio == 0.0
        assert backup_info.checksum is None
        assert backup_info.database_version is None
        assert backup_info.backup_path == ""
        assert backup_info.metadata == {}
        assert backup_info.recovery_time == 0.0
        assert backup_info.is_recoverable is True
        assert backup_info.dependencies == []


class TestRecoveryPlan:
    """Test RecoveryPlan dataclass."""

    def test_recovery_plan_creation(self):
        """Test creating recovery plan."""
        backup_info = BackupInfo(
            backup_id="test_backup_1",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=1234567890.0,
        )

        recovery_plan = RecoveryPlan(
            target_backup=backup_info,
            recovery_type="full_restore",
            estimated_time=30.0,
            steps=["step1", "step2"],
            rollback_plan=["rollback1", "rollback2"],
            data_loss_risk="low",
        )

        assert recovery_plan.target_backup == backup_info
        assert recovery_plan.recovery_type == "full_restore"
        assert recovery_plan.estimated_time == 30.0
        assert recovery_plan.steps == ["step1", "step2"]
        assert recovery_plan.rollback_plan == ["rollback1", "rollback2"]
        assert recovery_plan.data_loss_risk == "low"

    def test_recovery_plan_defaults(self):
        """Test recovery plan with default values."""
        backup_info = BackupInfo(
            backup_id="test_backup_1",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=1234567890.0,
        )

        recovery_plan = RecoveryPlan(
            target_backup=backup_info,
            recovery_type="full_restore",
            estimated_time=30.0,
            steps=["step1"],
            rollback_plan=["rollback1"],
        )

        assert recovery_plan.data_loss_risk == "low"


class TestBackupValidator:
    """Test BackupValidator class."""

    def test_validate_backup_file_exists(self):
        """Test validating existing backup file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_path = temp_file.name

        try:
            result = BackupValidator.validate_backup_file(temp_path)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_validate_backup_file_not_exists(self):
        """Test validating non-existent backup file."""
        result = BackupValidator.validate_backup_file("/nonexistent/path")
        assert result is False

    def test_validate_backup_file_empty(self):
        """Test validating empty backup file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result = BackupValidator.validate_backup_file(temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)

    def test_validate_backup_file_with_checksum(self):
        """Test validating backup file with checksum."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_path = temp_file.name

        try:
            checksum = BackupValidator.calculate_checksum(temp_path)
            result = BackupValidator.validate_backup_file(temp_path, checksum)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_validate_backup_file_wrong_checksum(self):
        """Test validating backup file with wrong checksum."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_path = temp_file.name

        try:
            result = BackupValidator.validate_backup_file(temp_path, "wrong_checksum")
            assert result is False
        finally:
            os.unlink(temp_path)

    def test_calculate_checksum_sha256(self):
        """Test calculating SHA256 checksum."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_path = temp_file.name

        try:
            checksum = BackupValidator.calculate_checksum(temp_path, "sha256")
            assert len(checksum) == 64  # SHA256 hex length
            assert isinstance(checksum, str)
        finally:
            os.unlink(temp_path)

    def test_calculate_checksum_md5(self):
        """Test calculating MD5 checksum."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test data")
            temp_path = temp_file.name

        try:
            checksum = BackupValidator.calculate_checksum(temp_path, "md5")
            assert len(checksum) == 32  # MD5 hex length
            assert isinstance(checksum, str)
        finally:
            os.unlink(temp_path)

    def test_validate_database_backup_valid(self):
        """Test validating valid database backup."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Create a minimal SQLite database
            import sqlite3
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            result = BackupValidator.validate_database_backup(temp_path)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_validate_database_backup_invalid(self):
        """Test validating invalid database backup."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"not a database")
            temp_path = temp_file.name

        try:
            result = BackupValidator.validate_database_backup(temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)

    def test_validate_database_backup_nonexistent(self):
        """Test validating non-existent database backup."""
        result = BackupValidator.validate_database_backup("/nonexistent/path")
        assert result is False


class TestBackupCompressor:
    """Test BackupCompressor class."""

    def test_compress_file(self):
        """Test compressing a file."""
        with tempfile.NamedTemporaryFile(delete=False) as input_file:
            # Write more data to ensure compression works
            input_file.write(b"test data for compression" * 100)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = output_file.name

        try:
            compression_ratio = BackupCompressor.compress_file(input_path, output_path)
            assert compression_ratio < 1.0  # Should be compressed
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_compress_file_empty(self):
        """Test compressing an empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as input_file:
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = output_file.name

        try:
            compression_ratio = BackupCompressor.compress_file(input_path, output_path)
            assert compression_ratio == 0.0  # Empty file
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_compress_file_error(self):
        """Test compressing file with error."""
        with pytest.raises(BackupError, match="Compression failed"):
            BackupCompressor.compress_file("/nonexistent/input", "/nonexistent/output")

    def test_decompress_file(self):
        """Test decompressing a file."""
        with tempfile.NamedTemporaryFile(delete=False) as input_file:
            input_file.write(b"test data for compression")
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(delete=False) as compressed_file:
            compressed_path = compressed_file.name

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = output_file.name

        try:
            # First compress
            BackupCompressor.compress_file(input_path, compressed_path)
            
            # Then decompress
            BackupCompressor.decompress_file(compressed_path, output_path)
            
            # Verify content matches
            with open(input_path, "rb") as f:
                original_data = f.read()
            with open(output_path, "rb") as f:
                decompressed_data = f.read()
            
            assert original_data == decompressed_data
        finally:
            for path in [input_path, compressed_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_decompress_file_error(self):
        """Test decompressing file with error."""
        with pytest.raises(BackupError, match="Decompression failed"):
            BackupCompressor.decompress_file("/nonexistent/input", "/nonexistent/output")


class TestBackupEncryptor:
    """Test BackupEncryptor class."""

    def test_encryptor_creation_default_key(self):
        """Test creating encryptor with default key."""
        encryptor = BackupEncryptor()
        assert encryptor.key is not None
        assert encryptor.cipher is not None

    def test_encryptor_creation_custom_key(self):
        """Test creating encryptor with custom key."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        encryptor = BackupEncryptor(key.decode())
        assert encryptor.key == key.decode()  # Key is stored as string

    def test_encrypt_file(self):
        """Test encrypting a file."""
        encryptor = BackupEncryptor()
        
        with tempfile.NamedTemporaryFile(delete=False) as input_file:
            input_file.write(b"test data for encryption")
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = output_file.name

        try:
            encryptor.encrypt_file(input_path, output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Verify content is different (encrypted)
            with open(input_path, "rb") as f:
                original_data = f.read()
            with open(output_path, "rb") as f:
                encrypted_data = f.read()
            
            assert original_data != encrypted_data
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_encrypt_file_error(self):
        """Test encrypting file with error."""
        encryptor = BackupEncryptor()
        with pytest.raises(BackupError, match="Encryption failed"):
            encryptor.encrypt_file("/nonexistent/input", "/nonexistent/output")

    def test_decrypt_file(self):
        """Test decrypting a file."""
        encryptor = BackupEncryptor()
        
        with tempfile.NamedTemporaryFile(delete=False) as input_file:
            input_file.write(b"test data for encryption")
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(delete=False) as encrypted_file:
            encrypted_path = encrypted_file.name

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = output_file.name

        try:
            # First encrypt
            encryptor.encrypt_file(input_path, encrypted_path)
            
            # Then decrypt
            encryptor.decrypt_file(encrypted_path, output_path)
            
            # Verify content matches
            with open(input_path, "rb") as f:
                original_data = f.read()
            with open(output_path, "rb") as f:
                decrypted_data = f.read()
            
            assert original_data == decrypted_data
        finally:
            for path in [input_path, encrypted_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_decrypt_file_error(self):
        """Test decrypting file with error."""
        encryptor = BackupEncryptor()
        with pytest.raises(BackupError, match="Decryption failed"):
            encryptor.decrypt_file("/nonexistent/input", "/nonexistent/output")


class TestBackupManagerAdvanced:
    """Test advanced BackupManager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Fixture for temporary backup directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_backend(self):
        """Fixture for mock database backend."""
        mock_backend = Mock()
        mock_backend.config.database_path = "/tmp/test.db"
        return mock_backend

    @pytest.fixture
    def backup_config(self, temp_backup_dir):
        """Fixture for backup configuration."""
        return BackupConfig(
            backup_directory=temp_backup_dir,
            max_backups=3,
            compression=False,
            encryption=False,
            verify_backup=False,
            auto_cleanup=True,
            retention_days=1,
        )

    @pytest.fixture
    def backup_manager(self, mock_backend, backup_config):
        """Fixture for backup manager."""
        return BackupManager(mock_backend, backup_config)

    def test_generate_backup_id(self, backup_manager):
        """Test generating backup ID."""
        backup_id = backup_manager._generate_backup_id(BackupType.FULL)
        assert backup_id.startswith("backup_")
        assert backup_id.endswith("_full")
        assert "_" in backup_id

    def test_create_backup_file_with_path(self, backup_manager):
        """Test creating backup file with provided path."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test database content")
            temp_path = temp_file.name

        try:
            backup_info = BackupInfo(
                backup_id="test_backup",
                backup_type=BackupType.FULL,
                status=BackupStatus.RUNNING,
                created_at=time.time(),
            )
            
            result_path = backup_manager._create_backup_file(backup_info, temp_path)
            assert os.path.exists(result_path)
            
            with open(result_path, "rb") as f:
                content = f.read()
            assert content == b"test database content"
        finally:
            for path in [temp_path, result_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_create_backup_file_database_not_found(self, backup_manager):
        """Test creating backup file when database not found."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.RUNNING,
            created_at=time.time(),
        )
        
        with pytest.raises(BackupError, match="Database file not found"):
            backup_manager._create_backup_file(backup_info, "/nonexistent/db")

    def test_restore_backup_not_found(self, backup_manager):
        """Test restoring non-existent backup."""
        with pytest.raises(BackupNotFoundError, match="Backup nonexistent not found"):
            backup_manager.restore_backup("nonexistent")

    def test_restore_backup_not_recoverable(self, backup_manager):
        """Test restoring non-recoverable backup."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.FAILED,
            created_at=time.time(),
            is_recoverable=False,
        )
        backup_manager._backups["test_backup"] = backup_info
        
        with pytest.raises(RecoveryError, match="Backup test_backup is not recoverable"):
            backup_manager.restore_backup("test_backup")

    def test_delete_backup_not_found(self, backup_manager):
        """Test deleting non-existent backup."""
        with pytest.raises(BackupNotFoundError, match="Backup nonexistent not found"):
            backup_manager.delete_backup("nonexistent")

    def test_delete_backup_success(self, backup_manager):
        """Test deleting backup successfully."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            backup_info = BackupInfo(
                backup_id="test_backup",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                created_at=time.time(),
                backup_path=temp_path,
            )
            backup_manager._backups["test_backup"] = backup_info
            
            backup_manager.delete_backup("test_backup")
            assert "test_backup" not in backup_manager._backups
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_cleanup_old_backups_disabled(self, backup_manager):
        """Test cleanup when auto cleanup is disabled."""
        backup_manager.config.auto_cleanup = False
        result = backup_manager.cleanup_old_backups()
        assert result == 0

    def test_cleanup_old_backups_by_count(self, backup_manager):
        """Test cleanup by backup count."""
        # Add more backups than max_backups
        for i in range(5):
            backup_info = BackupInfo(
                backup_id=f"backup_{i}",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                created_at=time.time() - i * 3600,  # Different timestamps
            )
            backup_manager._backups[f"backup_{i}"] = backup_info

        result = backup_manager.cleanup_old_backups()
        assert result == 2  # Should delete 2 backups (5 - 3 max_backups)

    def test_cleanup_old_backups_by_age(self, backup_manager):
        """Test cleanup by backup age."""
        # Add old backup
        old_backup = BackupInfo(
            backup_id="old_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=time.time() - 2 * 24 * 3600,  # 2 days ago
        )
        backup_manager._backups["old_backup"] = old_backup

        result = backup_manager.cleanup_old_backups()
        assert result == 1  # Should delete 1 old backup

    def test_start_auto_backup(self, backup_manager):
        """Test starting auto backup."""
        backup_manager.start_auto_backup()
        assert backup_manager._running is True
        assert backup_manager._backup_thread is not None
        backup_manager.stop_auto_backup()

    def test_start_auto_backup_already_running(self, backup_manager):
        """Test starting auto backup when already running."""
        backup_manager._running = True
        backup_manager.start_auto_backup()  # Should not start again
        assert backup_manager._running is True

    def test_stop_auto_backup(self, backup_manager):
        """Test stopping auto backup."""
        backup_manager._running = True
        backup_manager._backup_thread = Mock()
        backup_manager.stop_auto_backup()
        assert backup_manager._running is False

    def test_create_recovery_plan_not_found(self, backup_manager):
        """Test creating recovery plan for non-existent backup."""
        with pytest.raises(BackupNotFoundError, match="Backup nonexistent not found"):
            backup_manager.create_recovery_plan("nonexistent")

    def test_create_recovery_plan_success(self, backup_manager):
        """Test creating recovery plan successfully."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=time.time(),
            is_recoverable=True,
        )
        backup_manager._backups["test_backup"] = backup_info
        
        recovery_plan = backup_manager.create_recovery_plan("test_backup")
        assert isinstance(recovery_plan, RecoveryPlan)
        assert recovery_plan.target_backup == backup_info
        assert recovery_plan.recovery_type == "full_restore"
        assert recovery_plan.estimated_time == 30.0
        assert len(recovery_plan.steps) == 5
        assert len(recovery_plan.rollback_plan) == 3
        assert recovery_plan.data_loss_risk == "low"

    def test_create_recovery_plan_high_risk(self, backup_manager):
        """Test creating recovery plan for non-recoverable backup."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.FAILED,
            created_at=time.time(),
            is_recoverable=False,
        )
        backup_manager._backups["test_backup"] = backup_info
        
        recovery_plan = backup_manager.create_recovery_plan("test_backup")
        assert recovery_plan.data_loss_risk == "high"

    def test_context_manager_auto_backup_enabled(self, mock_backend, backup_config):
        """Test context manager with auto backup enabled."""
        backup_config.auto_backup = True
        
        with BackupManager(mock_backend, backup_config) as manager:
            assert manager._running is True

    def test_context_manager_auto_backup_disabled(self, mock_backend, backup_config):
        """Test context manager with auto backup disabled."""
        backup_config.auto_backup = False
        
        with BackupManager(mock_backend, backup_config) as manager:
            assert manager._running is False

    def test_load_backup_index_nonexistent(self, backup_manager):
        """Test loading backup index when file doesn't exist."""
        # Should not raise an error
        backup_manager._load_backup_index()
        assert len(backup_manager._backups) == 0

    def test_load_backup_index_invalid_json(self, backup_manager):
        """Test loading backup index with invalid JSON."""
        index_file = backup_manager.backup_dir / "backup_index.json"
        with open(index_file, "w") as f:
            f.write("invalid json")
        
        # Should not raise an error
        backup_manager._load_backup_index()
        assert len(backup_manager._backups) == 0

    def test_save_backup_index(self, backup_manager):
        """Test saving backup index."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=time.time(),
        )
        backup_manager._backups["test_backup"] = backup_info
        
        backup_manager._save_backup_index()
        
        index_file = backup_manager.backup_dir / "backup_index.json"
        assert index_file.exists()
        
        with open(index_file, "r") as f:
            data = json.load(f)
        
        assert "backups" in data
        assert len(data["backups"]) == 1
        assert data["backups"][0]["backup_id"] == "test_backup"

    def test_get_backup(self, backup_manager):
        """Test getting backup by ID."""
        backup_info = BackupInfo(
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            created_at=time.time(),
        )
        backup_manager._backups["test_backup"] = backup_info
        
        result = backup_manager.get_backup("test_backup")
        assert result == backup_info
        
        result = backup_manager.get_backup("nonexistent")
        assert result is None
