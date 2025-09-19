"""
Unit tests for wallet manager functionality.
"""

import json
import os
import tempfile
from typing import List

import pytest

from dubchain.wallet import (
    EncryptionConfig,
    HDWallet,
    MultisigConfig,
    MultisigType,
    MultisigWallet,
    WalletInfo,
    WalletManager,
    WalletManagerConfig,
    WalletManagerError,
    WalletType,
)


class TestWalletInfo:
    """Test WalletInfo functionality."""

    def test_wallet_info_creation(self):
        """Test wallet info creation."""
        info = WalletInfo(
            wallet_id="wallet_123",
            wallet_type=WalletType.HD_WALLET,
            name="Test Wallet",
            created_at=1234567890,
            last_accessed=1234567890,
            is_encrypted=True,
            network="mainnet",
        )

        assert info.wallet_id == "wallet_123"
        assert info.wallet_type == WalletType.HD_WALLET
        assert info.name == "Test Wallet"
        assert info.created_at == 1234567890
        assert info.last_accessed == 1234567890
        assert info.is_encrypted is True
        assert info.network == "mainnet"

    def test_wallet_info_serialization(self):
        """Test wallet info serialization."""
        info = WalletInfo(
            wallet_id="wallet_123",
            wallet_type=WalletType.MULTISIG_WALLET,
            name="Test Wallet",
            created_at=1234567890,
            last_accessed=1234567890,
            is_encrypted=False,
            network="testnet",
            metadata={"custom": "value"},
        )

        # Test to_dict
        info_dict = info.to_dict()
        assert info_dict["wallet_id"] == "wallet_123"
        assert info_dict["wallet_type"] == "multisig_wallet"
        assert info_dict["name"] == "Test Wallet"
        assert info_dict["created_at"] == 1234567890
        assert info_dict["last_accessed"] == 1234567890
        assert info_dict["is_encrypted"] is False
        assert info_dict["network"] == "testnet"
        assert info_dict["metadata"]["custom"] == "value"

        # Test from_dict
        restored_info = WalletInfo.from_dict(info_dict)
        assert restored_info.wallet_id == info.wallet_id
        assert restored_info.wallet_type == info.wallet_type
        assert restored_info.name == info.name
        assert restored_info.created_at == info.created_at
        assert restored_info.last_accessed == info.last_accessed
        assert restored_info.is_encrypted == info.is_encrypted
        assert restored_info.network == info.network
        assert restored_info.metadata == info.metadata


class TestWalletManagerConfig:
    """Test WalletManagerConfig functionality."""

    def test_wallet_manager_config_creation(self):
        """Test wallet manager config creation."""
        config = WalletManagerConfig(
            storage_path="/tmp/wallets",
            default_network="mainnet",
            encryption_enabled=True,
            mnemonic_language="english",
            auto_backup=True,
            backup_interval=86400,
            max_wallets=100,
        )

        assert config.storage_path == "/tmp/wallets"
        assert config.default_network == "mainnet"
        assert config.encryption_enabled is True
        assert config.mnemonic_language == "english"
        assert config.auto_backup is True
        assert config.backup_interval == 86400
        assert config.max_wallets == 100

    def test_wallet_manager_config_validation(self):
        """Test wallet manager config validation."""
        # Test empty storage path
        with pytest.raises(ValueError, match="Storage path cannot be empty"):
            WalletManagerConfig(storage_path="")

        # Test negative backup interval
        with pytest.raises(ValueError, match="Backup interval must be positive"):
            WalletManagerConfig(storage_path="/tmp/wallets", backup_interval=0)

        # Test negative max wallets
        with pytest.raises(ValueError, match="Max wallets must be positive"):
            WalletManagerConfig(storage_path="/tmp/wallets", max_wallets=0)

    def test_wallet_manager_config_serialization(self):
        """Test wallet manager config serialization."""
        encryption_config = EncryptionConfig()
        config = WalletManagerConfig(
            storage_path="/tmp/wallets",
            default_network="testnet",
            encryption_enabled=True,
            encryption_config=encryption_config,
            mnemonic_language="english",
            auto_backup=False,
            backup_interval=43200,
            max_wallets=50,
            metadata={"custom": "value"},
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["storage_path"] == "/tmp/wallets"
        assert config_dict["default_network"] == "testnet"
        assert config_dict["encryption_enabled"] is True
        assert config_dict["encryption_config"] is not None
        assert config_dict["mnemonic_language"] == "english"
        assert config_dict["auto_backup"] is False
        assert config_dict["backup_interval"] == 43200
        assert config_dict["max_wallets"] == 50
        assert config_dict["metadata"]["custom"] == "value"

        # Test from_dict
        restored_config = WalletManagerConfig.from_dict(config_dict)
        assert restored_config.storage_path == config.storage_path
        assert restored_config.default_network == config.default_network
        assert restored_config.encryption_enabled == config.encryption_enabled
        assert restored_config.mnemonic_language == config.mnemonic_language
        assert restored_config.auto_backup == config.auto_backup
        assert restored_config.backup_interval == config.backup_interval
        assert restored_config.max_wallets == config.max_wallets
        assert restored_config.metadata == config.metadata


class TestWalletManager:
    """Test WalletManager functionality."""

    def test_wallet_manager_creation(self):
        """Test wallet manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            assert manager.config == config
            assert len(manager.wallets) == 0
            assert len(manager.wallet_info) == 0

    def test_wallet_manager_hd_wallet_creation(self):
        """Test HD wallet creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create HD wallet
            wallet_id = manager.create_hd_wallet(
                name="Test HD Wallet", password="test_password", network="mainnet"
            )

            assert wallet_id in manager.wallet_info
            assert manager.wallet_info[wallet_id].wallet_type == WalletType.HD_WALLET
            assert manager.wallet_info[wallet_id].name == "Test HD Wallet"
            assert manager.wallet_info[wallet_id].is_encrypted is True
            assert manager.wallet_info[wallet_id].network == "mainnet"

    def test_wallet_manager_hd_wallet_creation_with_mnemonic(self):
        """Test HD wallet creation with custom mnemonic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Generate mnemonic
            mnemonic = manager.mnemonic_generator.generate()

            # Create HD wallet with mnemonic
            wallet_id = manager.create_hd_wallet(
                name="Test HD Wallet", mnemonic=mnemonic, network="testnet"
            )

            assert wallet_id in manager.wallet_info
            wallet = manager.load_wallet(wallet_id)
            assert wallet.mnemonic == mnemonic

    def test_wallet_manager_multisig_wallet_creation(self):
        """Test multisig wallet creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create multisig config
            multisig_config = MultisigConfig(
                multisig_type=MultisigType.M_OF_N,
                required_signatures=2,
                total_participants=3,
            )

            # Create multisig wallet
            wallet_id = manager.create_multisig_wallet(
                name="Test Multisig Wallet",
                config=multisig_config,
                password="test_password",
            )

            assert wallet_id in manager.wallet_info
            assert (
                manager.wallet_info[wallet_id].wallet_type == WalletType.MULTISIG_WALLET
            )
            assert manager.wallet_info[wallet_id].name == "Test Multisig Wallet"
            assert manager.wallet_info[wallet_id].is_encrypted is True

    def test_wallet_manager_max_wallets_limit(self):
        """Test wallet manager max wallets limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir, max_wallets=2)
            manager = WalletManager(config)

            # Create first wallet
            wallet_id1 = manager.create_hd_wallet("Wallet 1")

            # Create second wallet
            wallet_id2 = manager.create_hd_wallet("Wallet 2")

            # Try to create third wallet
            with pytest.raises(
                WalletManagerError, match="Maximum number of wallets reached"
            ):
                manager.create_hd_wallet("Wallet 3")

    def test_wallet_manager_wallet_loading(self):
        """Test wallet loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet
            wallet_id = manager.create_hd_wallet(
                name="Test Wallet", password="test_password"
            )

            # Load wallet
            wallet = manager.load_wallet(wallet_id, "test_password")

            assert isinstance(wallet, HDWallet)
            assert wallet.metadata.name == "Test Wallet"

            # Unload wallet first
            manager.unload_wallet(wallet_id)

            # Test loading without password for encrypted wallet
            with pytest.raises(
                WalletManagerError, match="Password required for encrypted wallet"
            ):
                manager.load_wallet(wallet_id)

    def test_wallet_manager_wallet_unloading(self):
        """Test wallet unloading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create and load wallet
            wallet_id = manager.create_hd_wallet("Test Wallet")
            wallet = manager.load_wallet(wallet_id)

            assert wallet_id in manager.wallets

            # Unload wallet
            manager.unload_wallet(wallet_id)

            assert wallet_id not in manager.wallets

    def test_wallet_manager_wallet_deletion(self):
        """Test wallet deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet
            wallet_id = manager.create_hd_wallet(
                name="Test Wallet", password="test_password"
            )

            assert wallet_id in manager.wallet_info

            # Delete wallet
            success = manager.delete_wallet(wallet_id, "test_password")

            assert success is True
            assert wallet_id not in manager.wallet_info
            assert wallet_id not in manager.wallets

    def test_wallet_manager_wallet_deletion_wrong_password(self):
        """Test wallet deletion with wrong password."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create encrypted wallet
            wallet_id = manager.create_hd_wallet(
                name="Test Wallet", password="correct_password"
            )

            # Try to delete with wrong password
            with pytest.raises(WalletManagerError, match="Invalid password"):
                manager.delete_wallet(wallet_id, "wrong_password")

    def test_wallet_manager_wallet_list(self):
        """Test wallet list retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create multiple wallets
            wallet_id1 = manager.create_hd_wallet("Wallet 1")
            wallet_id2 = manager.create_hd_wallet("Wallet 2")

            # Get wallet list
            wallet_list = manager.get_wallet_list()

            assert len(wallet_list) == 2
            wallet_ids = [info.wallet_id for info in wallet_list]
            assert wallet_id1 in wallet_ids
            assert wallet_id2 in wallet_ids

    def test_wallet_manager_wallet_info(self):
        """Test wallet info retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet
            wallet_id = manager.create_hd_wallet("Test Wallet")

            # Get wallet info
            info = manager.get_wallet_info(wallet_id)

            assert info.wallet_id == wallet_id
            assert info.name == "Test Wallet"
            assert info.wallet_type == WalletType.HD_WALLET

    def test_wallet_manager_wallet_info_not_found(self):
        """Test wallet info retrieval for non-existent wallet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            with pytest.raises(
                WalletManagerError, match="Wallet nonexistent_wallet not found"
            ):
                manager.get_wallet_info("nonexistent_wallet")

    def test_wallet_manager_wallet_renaming(self):
        """Test wallet renaming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet
            wallet_id = manager.create_hd_wallet("Original Name")

            # Rename wallet
            manager.rename_wallet(wallet_id, "New Name")

            # Check name changed
            info = manager.get_wallet_info(wallet_id)
            assert info.name == "New Name"

    def test_wallet_manager_wallet_export(self):
        """Test wallet export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet
            wallet_id = manager.create_hd_wallet(
                name="Test Wallet", password="test_password"
            )

            # Export wallet
            export_data = manager.export_wallet(
                wallet_id, password="test_password", include_private_keys=True
            )

            assert "mnemonic" in export_data
            assert "metadata" in export_data
            assert "accounts" in export_data

    def test_wallet_manager_wallet_import(self):
        """Test wallet import."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet and export it
            original_wallet_id = manager.create_hd_wallet("Original Wallet")
            original_wallet = manager.load_wallet(original_wallet_id)
            export_data = original_wallet.to_dict(include_private_keys=True)

            # Import wallet
            imported_wallet_id = manager.import_wallet(
                export_data, name="Imported Wallet", password="test_password"
            )

            assert imported_wallet_id in manager.wallet_info
            assert manager.wallet_info[imported_wallet_id].name == "Imported Wallet"

            # Load imported wallet
            imported_wallet = manager.load_wallet(imported_wallet_id, "test_password")
            assert isinstance(imported_wallet, HDWallet)

    def test_wallet_manager_wallet_backup_restore(self):
        """Test wallet backup and restore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create wallet
            wallet_id = manager.create_hd_wallet(
                name="Test Wallet", password="test_password"
            )

            # Create backup
            backup_file = manager.backup_wallet(
                wallet_id, backup_path=temp_dir, password="test_password"
            )

            assert os.path.exists(backup_file)

            # Delete original wallet
            manager.delete_wallet(wallet_id, "test_password")

            # Restore wallet
            restored_wallet_id = manager.restore_wallet(
                backup_file, name="Restored Wallet", password="test_password"
            )

            assert restored_wallet_id in manager.wallet_info
            assert manager.wallet_info[restored_wallet_id].name == "Restored Wallet"

    def test_wallet_manager_password_validation(self):
        """Test password validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Test valid password
            is_valid, errors = manager.validate_password("StrongPass123!")
            assert is_valid is True
            assert len(errors) == 0

            # Test invalid password
            is_valid, errors = manager.validate_password("weak")
            assert is_valid is False
            assert len(errors) > 0

    def test_wallet_manager_password_generation(self):
        """Test password generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Generate password
            password = manager.generate_strong_password(16)

            assert len(password) == 16
            assert any(c.isupper() for c in password)
            assert any(c.islower() for c in password)
            assert any(c.isdigit() for c in password)
            assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    def test_wallet_manager_password_strength(self):
        """Test password strength rating."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Test different strength levels
            assert manager.get_password_strength("password") in ["Very Weak", "Weak"]
            assert manager.get_password_strength("Password123!") in [
                "Good",
                "Strong",
                "Very Strong",
            ]

    def test_wallet_manager_info(self):
        """Test wallet manager info retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create some wallets
            manager.create_hd_wallet("Wallet 1")
            manager.create_hd_wallet("Wallet 2")

            # Get manager info
            info = manager.get_manager_info()

            assert info["wallet_count"] == 2
            assert info["loaded_wallets"] == 2  # Wallets are loaded after creation
            assert info["storage_path"] == temp_dir
            assert info["encryption_enabled"] is True
            assert info["auto_backup"] is True
            assert info["max_wallets"] == 100

    def test_wallet_manager_cleanup_backups(self):
        """Test wallet manager backup cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            # Create some backup files
            old_backup = os.path.join(
                temp_dir, "wallet_1234567890_backup_1234567890.json"
            )
            with open(old_backup, "w") as f:
                json.dump({"backup": "data"}, f)

            # Cleanup old backups
            cleaned_count = manager.cleanup_old_backups(max_age_days=0)  # Clean all

            assert cleaned_count == 1
            assert not os.path.exists(old_backup)

    def test_wallet_manager_string_representation(self):
        """Test wallet manager string representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WalletManagerConfig(storage_path=temp_dir)
            manager = WalletManager(config)

            str_repr = str(manager)
            assert "WalletManager" in str_repr
            assert "wallets=0" in str_repr
            assert "loaded=0" in str_repr

            repr_str = repr(manager)
            assert "WalletManager" in repr_str
            assert temp_dir in repr_str
