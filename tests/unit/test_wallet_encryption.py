"""
Unit tests for wallet encryption functionality.
"""

import logging

logger = logging.getLogger(__name__)
import json
import os
import tempfile
from typing import List

import pytest

from dubchain.wallet import (
    EncryptedData,
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionError,
    KeyDerivationFunction,
    PasswordManager,
    SecureStorage,
    WalletEncryption,
)


class TestEncryptionConfig:
    """Test EncryptionConfig functionality."""

    def test_encryption_config_creation(self):
        """Test encryption config creation."""
        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.PBKDF2,
            iterations=100000,
            salt_length=32,
            key_length=32,
            iv_length=12,
        )

        assert config.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert config.kdf == KeyDerivationFunction.PBKDF2
        assert config.iterations == 100000
        assert config.salt_length == 32
        assert config.key_length == 32
        assert config.iv_length == 12
        assert config.tag_length == 16

    def test_encryption_config_validation(self):
        """Test encryption config validation."""
        # Test zero iterations
        with pytest.raises(ValueError, match="Iterations must be positive"):
            EncryptionConfig(iterations=0)

        # Test zero salt length
        with pytest.raises(ValueError, match="Salt length must be positive"):
            EncryptionConfig(salt_length=0)

        # Test zero key length
        with pytest.raises(ValueError, match="Key length must be positive"):
            EncryptionConfig(key_length=0)

        # Test zero IV length
        with pytest.raises(ValueError, match="IV length must be positive"):
            EncryptionConfig(iv_length=0)

        # Test GCM with wrong IV length
        with pytest.raises(ValueError, match="GCM requires 12-byte IV"):
            EncryptionConfig(algorithm=EncryptionAlgorithm.AES_256_GCM, iv_length=16)

        # Test CBC with wrong IV length
        with pytest.raises(ValueError, match="CBC requires 16-byte IV"):
            EncryptionConfig(algorithm=EncryptionAlgorithm.AES_256_CBC, iv_length=12)

    def test_encryption_config_serialization(self):
        """Test encryption config serialization."""
        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.SCRYPT,
            iterations=200000,
            salt_length=64,
            key_length=32,
            iv_length=12,
            memory_cost=2097152,
            parallelism=2,
            metadata={"security_level": "high"},
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["algorithm"] == "aes_256_gcm"
        assert config_dict["kdf"] == "scrypt"
        assert config_dict["iterations"] == 200000
        assert config_dict["salt_length"] == 64
        assert config_dict["key_length"] == 32
        assert config_dict["iv_length"] == 12
        assert config_dict["memory_cost"] == 2097152
        assert config_dict["parallelism"] == 2
        assert config_dict["metadata"]["security_level"] == "high"

        # Test from_dict
        restored_config = EncryptionConfig.from_dict(config_dict)
        assert restored_config.algorithm == config.algorithm
        assert restored_config.kdf == config.kdf
        assert restored_config.iterations == config.iterations
        assert restored_config.salt_length == config.salt_length
        assert restored_config.key_length == config.key_length
        assert restored_config.iv_length == config.iv_length
        assert restored_config.memory_cost == config.memory_cost
        assert restored_config.parallelism == config.parallelism
        assert restored_config.metadata == config.metadata


class TestEncryptedData:
    """Test EncryptedData functionality."""

    def test_encrypted_data_creation(self):
        """Test encrypted data creation."""
        encrypted_data = EncryptedData(
            ciphertext=b"encrypted_data",
            salt=b"salt_data",
            iv=b"iv_data",
            tag=b"tag_data",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.PBKDF2,
            iterations=100000,
        )

        assert encrypted_data.ciphertext == b"encrypted_data"
        assert encrypted_data.salt == b"salt_data"
        assert encrypted_data.iv == b"iv_data"
        assert encrypted_data.tag == b"tag_data"
        assert encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encrypted_data.kdf == KeyDerivationFunction.PBKDF2
        assert encrypted_data.iterations == 100000
        assert encrypted_data.created_at > 0

    def test_encrypted_data_serialization(self):
        """Test encrypted data serialization."""
        encrypted_data = EncryptedData(
            ciphertext=b"encrypted_data",
            salt=b"salt_data",
            iv=b"iv_data",
            tag=b"tag_data",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivationFunction.PBKDF2,
            iterations=100000,
            metadata={"test": "value"},
        )

        # Test to_dict
        data_dict = encrypted_data.to_dict()
        assert (
            data_dict["ciphertext"] == "656e637279707465645f64617461"
        )  # hex of "encrypted_data"
        assert data_dict["salt"] == "73616c745f64617461"  # hex of "salt_data"
        assert data_dict["iv"] == "69765f64617461"  # hex of "iv_data"
        assert data_dict["tag"] == "7461675f64617461"  # hex of "tag_data"
        assert data_dict["algorithm"] == "aes_256_gcm"
        assert data_dict["kdf"] == "pbkdf2"
        assert data_dict["iterations"] == 100000
        assert data_dict["metadata"]["test"] == "value"

        # Test from_dict
        restored_data = EncryptedData.from_dict(data_dict)
        assert restored_data.ciphertext == encrypted_data.ciphertext
        assert restored_data.salt == encrypted_data.salt
        assert restored_data.iv == encrypted_data.iv
        assert restored_data.tag == encrypted_data.tag
        assert restored_data.algorithm == encrypted_data.algorithm
        assert restored_data.kdf == encrypted_data.kdf
        assert restored_data.iterations == encrypted_data.iterations
        assert restored_data.metadata == encrypted_data.metadata


class TestWalletEncryption:
    """Test WalletEncryption functionality."""

    def test_wallet_encryption_creation(self):
        """Test wallet encryption creation."""
        config = EncryptionConfig()
        encryption = WalletEncryption(config)

        assert encryption.config == config

    def test_wallet_encryption_default_config(self):
        """Test wallet encryption with default config."""
        encryption = WalletEncryption()

        assert encryption.config.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert encryption.config.kdf == KeyDerivationFunction.PBKDF2
        assert encryption.config.iterations == 100000

    def test_wallet_encryption_encrypt_decrypt_bytes(self):
        """Test wallet encryption encrypt/decrypt bytes."""
        encryption = WalletEncryption()
        data = b"test data to encrypt"
        password = "test_password_123"

        # Encrypt data
        encrypted_data = encryption.encrypt(data, password)

        assert isinstance(encrypted_data, EncryptedData)
        assert encrypted_data.ciphertext != data
        assert len(encrypted_data.salt) == 32
        assert len(encrypted_data.iv) == 12
        assert encrypted_data.tag is not None

        # Decrypt data
        decrypted_data = encryption.decrypt(encrypted_data, password)

        assert decrypted_data == data

    def test_wallet_encryption_encrypt_decrypt_string(self):
        """Test wallet encryption encrypt/decrypt string."""
        encryption = WalletEncryption()
        text = "test string to encrypt"
        password = "test_password_123"

        # Encrypt string
        encrypted_data = encryption.encrypt_string(text, password)

        assert isinstance(encrypted_data, EncryptedData)

        # Decrypt string
        decrypted_text = encryption.decrypt_string(encrypted_data, password)

        assert decrypted_text == text

    def test_wallet_encryption_encrypt_decrypt_dict(self):
        """Test wallet encryption encrypt/decrypt dictionary."""
        encryption = WalletEncryption()
        data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        password = "test_password_123"

        # Encrypt dictionary
        encrypted_data = encryption.encrypt_dict(data, password)

        assert isinstance(encrypted_data, EncryptedData)

        # Decrypt dictionary
        decrypted_data = encryption.decrypt_dict(encrypted_data, password)

        assert decrypted_data == data

    def test_wallet_encryption_wrong_password(self):
        """Test wallet encryption with wrong password."""
        encryption = WalletEncryption()
        data = b"test data"
        password = "correct_password"
        wrong_password = "wrong_password"

        # Encrypt with correct password
        encrypted_data = encryption.encrypt(data, password)

        # Try to decrypt with wrong password
        with pytest.raises(EncryptionError, match="Invalid password or corrupted data"):
            encryption.decrypt(encrypted_data, wrong_password)

    def test_wallet_encryption_empty_data(self):
        """Test wallet encryption with empty data."""
        encryption = WalletEncryption()
        password = "test_password"

        # Test empty bytes
        with pytest.raises(EncryptionError, match="Cannot encrypt empty data"):
            encryption.encrypt(b"", password)

        # Test empty password
        with pytest.raises(EncryptionError, match="Password cannot be empty"):
            encryption.encrypt(b"test data", "")

    def test_wallet_encryption_password_verification(self):
        """Test wallet encryption password verification."""
        encryption = WalletEncryption()
        data = b"test data"
        password = "test_password"
        wrong_password = "wrong_password"

        # Encrypt data
        encrypted_data = encryption.encrypt(data, password)

        # Verify correct password
        assert encryption.verify_password(encrypted_data, password) is True

        # Verify wrong password
        assert encryption.verify_password(encrypted_data, wrong_password) is False

    def test_wallet_encryption_password_change(self):
        """Test wallet encryption password change."""
        encryption = WalletEncryption()
        data = b"test data"
        old_password = "old_password"
        new_password = "new_password"

        # Encrypt with old password
        encrypted_data = encryption.encrypt(data, old_password)

        # Change password
        new_encrypted_data = encryption.change_password(
            encrypted_data, old_password, new_password
        )

        # Verify old password doesn't work
        with pytest.raises(EncryptionError):
            encryption.decrypt(new_encrypted_data, old_password)

        # Verify new password works
        decrypted_data = encryption.decrypt(new_encrypted_data, new_password)
        assert decrypted_data == data

    def test_wallet_encryption_info(self):
        """Test wallet encryption info retrieval."""
        encryption = WalletEncryption()
        data = b"test data"
        password = "test_password"

        # Encrypt data
        encrypted_data = encryption.encrypt(data, password)

        # Get encryption info
        info = encryption.get_encryption_info(encrypted_data)

        assert info["algorithm"] == "aes_256_gcm"
        assert info["kdf"] == "pbkdf2"
        assert info["iterations"] == 100000
        assert info["salt_length"] == 32
        assert info["iv_length"] == 12
        assert info["ciphertext_length"] == len(encrypted_data.ciphertext)
        assert info["has_tag"] is True
        assert info["created_at"] > 0


class TestSecureStorage:
    """Test SecureStorage functionality."""

    def test_secure_storage_creation(self):
        """Test secure storage creation."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        assert storage.encryption == encryption
        assert storage.storage_path is None

    def test_secure_storage_path_setting(self):
        """Test secure storage path setting."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.set_storage_path(temp_dir)
            assert storage.storage_path == temp_dir
            assert os.path.exists(temp_dir)

    def test_secure_storage_save_load_wallet(self):
        """Test secure storage save/load wallet."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.set_storage_path(temp_dir)

            # Test data
            wallet_data = {"wallet_id": "test_wallet", "balance": 1000}
            password = "test_password"
            filename = "test_wallet"

            # Save wallet
            file_path = storage.save_wallet(wallet_data, password, filename)

            assert os.path.exists(file_path)
            assert file_path.endswith(".meta")

            # Check metadata file exists
            metadata_path = os.path.join(temp_dir, f"{filename}.meta")
            assert os.path.exists(metadata_path)

            # Load wallet
            loaded_data = storage.load_wallet(filename, password)

            assert loaded_data == wallet_data

    def test_secure_storage_wrong_password(self):
        """Test secure storage with wrong password."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.set_storage_path(temp_dir)

            # Save wallet
            wallet_data = {"wallet_id": "test_wallet"}
            storage.save_wallet(wallet_data, "correct_password", "test_wallet")

            # Try to load with wrong password
            with pytest.raises(EncryptionError):
                storage.load_wallet("test_wallet", "wrong_password")

    def test_secure_storage_list_wallets(self):
        """Test secure storage wallet listing."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.set_storage_path(temp_dir)

            # Save multiple wallets
            storage.save_wallet({"wallet1": "data1"}, "password1", "wallet1")
            storage.save_wallet({"wallet2": "data2"}, "password2", "wallet2")
            storage.save_wallet({"wallet3": "data3"}, "password3", "wallet3")

            # List wallets
            wallets = storage.list_wallets()

            assert len(wallets) == 3
            assert "wallet1" in wallets
            assert "wallet2" in wallets
            assert "wallet3" in wallets

    def test_secure_storage_delete_wallet(self):
        """Test secure storage wallet deletion."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.set_storage_path(temp_dir)

            # Save wallet
            storage.save_wallet({"wallet_id": "test_wallet"}, "password", "test_wallet")

            # Verify wallet exists
            assert storage.wallet_exists("test_wallet") is True

            # Delete wallet
            success = storage.delete_wallet("test_wallet")

            assert success is True
            assert storage.wallet_exists("test_wallet") is False

    def test_secure_storage_wallet_exists(self):
        """Test secure storage wallet existence check."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        with tempfile.TemporaryDirectory() as temp_dir:
            storage.set_storage_path(temp_dir)

            # Wallet doesn't exist
            assert storage.wallet_exists("nonexistent_wallet") is False

            # Save wallet
            storage.save_wallet({"wallet_id": "test_wallet"}, "password", "test_wallet")

            # Wallet exists
            assert storage.wallet_exists("test_wallet") is True

    def test_secure_storage_no_storage_path(self):
        """Test secure storage without storage path."""
        encryption = WalletEncryption()
        storage = SecureStorage(encryption)

        # Test operations without storage path
        with pytest.raises(EncryptionError, match="Storage path not set"):
            storage.save_wallet({"data": "test"}, "password", "wallet")

        with pytest.raises(EncryptionError, match="Storage path not set"):
            storage.load_wallet("wallet", "password")

        assert storage.list_wallets() == []
        assert storage.delete_wallet("wallet") is False
        assert storage.wallet_exists("wallet") is False


class TestPasswordManager:
    """Test PasswordManager functionality."""

    def test_password_manager_creation(self):
        """Test password manager creation."""
        manager = PasswordManager()

        assert manager.min_length == 8
        assert manager.require_special is True

    def test_password_manager_custom_config(self):
        """Test password manager with custom config."""
        manager = PasswordManager(min_length=12, require_special=False)

        assert manager.min_length == 12
        assert manager.require_special is False

    def test_password_validation(self):
        """Test password validation."""
        manager = PasswordManager()

        # Valid password
        is_valid, errors = manager.validate_password("StrongPass123!")
        assert is_valid is True
        assert len(errors) == 0

        # Too short
        is_valid, errors = manager.validate_password("Short1!")
        assert is_valid is False
        assert any("at least 8 characters" in error for error in errors)

        # No uppercase
        is_valid, errors = manager.validate_password("lowercase123!")
        assert is_valid is False
        assert any("uppercase letter" in error for error in errors)

        # No lowercase
        is_valid, errors = manager.validate_password("UPPERCASE123!")
        assert is_valid is False
        assert any("lowercase letter" in error for error in errors)

        # No digit
        is_valid, errors = manager.validate_password("NoDigits!")
        assert is_valid is False
        assert any("digit" in error for error in errors)

        # No special character
        is_valid, errors = manager.validate_password("NoSpecial123")
        assert is_valid is False
        assert any("special character" in error for error in errors)

    def test_password_validation_no_special_required(self):
        """Test password validation without special character requirement."""
        manager = PasswordManager(require_special=False)

        # Valid password without special character
        is_valid, errors = manager.validate_password("StrongPass123")
        assert is_valid is True
        assert len(errors) == 0

    def test_strong_password_generation(self):
        """Test strong password generation."""
        manager = PasswordManager()

        # Generate password
        password = manager.generate_strong_password(16)

        assert len(password) == 16
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    def test_strong_password_generation_minimum_length(self):
        """Test strong password generation with minimum length."""
        manager = PasswordManager()

        # Generate password with length less than minimum
        password = manager.generate_strong_password(4)

        assert len(password) == 8  # Should be at least 8
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    def test_password_entropy_calculation(self):
        """Test password entropy calculation."""
        manager = PasswordManager()

        # Test different password types
        assert manager.calculate_entropy("password") > 0
        assert manager.calculate_entropy("Password123!") > manager.calculate_entropy(
            "password"
        )
        assert manager.calculate_entropy("") == 0

    def test_password_strength_rating(self):
        """Test password strength rating."""
        manager = PasswordManager()

        # Test different strength levels
        assert manager.get_password_strength("password") in ["Very Weak", "Weak"]
        assert manager.get_password_strength("Password123!") in [
            "Good",
            "Strong",
            "Very Strong",
        ]
        assert manager.get_password_strength("VeryStrongPassword123!@#") in [
            "Strong",
            "Very Strong",
        ]
