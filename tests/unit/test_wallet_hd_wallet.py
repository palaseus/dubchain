"""
Unit tests for HD wallet functionality.
"""

import time
from typing import List

import pytest

from dubchain.wallet import (
    AccountType,
    HDWallet,
    Language,
    MnemonicConfig,
    MnemonicGenerator,
    WalletAccount,
    WalletError,
    WalletMetadata,
)


class TestWalletAccount:
    """Test WalletAccount functionality."""

    def test_wallet_account_creation(self):
        """Test wallet account creation."""
        from dubchain.crypto.signatures import PrivateKey, PublicKey
        from dubchain.wallet import DerivationPath

        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        derivation_path = DerivationPath.bip44(0, 0, 0, 0)

        account = WalletAccount(
            account_index=0,
            account_type=AccountType.STANDARD,
            derivation_path=derivation_path,
            public_key=public_key,
            private_key=private_key,
            label="Test Account",
        )

        assert account.account_index == 0
        assert account.account_type == AccountType.STANDARD
        assert account.derivation_path == derivation_path
        assert account.public_key == public_key
        assert account.private_key == private_key
        assert account.balance == 0
        assert account.transaction_count == 0
        assert account.label == "Test Account"

    def test_wallet_account_validation(self):
        """Test wallet account validation."""
        from dubchain.crypto.signatures import PrivateKey, PublicKey
        from dubchain.wallet import DerivationPath

        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        derivation_path = DerivationPath.bip44(0, 0, 0, 0)

        # Test negative account index
        with pytest.raises(ValueError, match="Account index must be non-negative"):
            WalletAccount(
                account_index=-1,
                account_type=AccountType.STANDARD,
                derivation_path=derivation_path,
                public_key=public_key,
            )

        # Test negative balance
        account = WalletAccount(
            account_index=0,
            account_type=AccountType.STANDARD,
            derivation_path=derivation_path,
            public_key=public_key,
        )

        # The balance is set via update_balance method, not direct assignment
        with pytest.raises(ValueError, match="Balance cannot be negative"):
            account.update_balance(-1)

        # Test negative transaction count - this is validated in __post_init__, not during assignment
        # So we need to create an account with negative transaction count
        with pytest.raises(ValueError, match="Transaction count must be non-negative"):
            WalletAccount(
                account_index=0,
                account_type=AccountType.STANDARD,
                derivation_path=derivation_path,
                public_key=public_key,
                transaction_count=-1,
            )

    def test_wallet_account_address_generation(self):
        """Test wallet account address generation."""
        from dubchain.crypto.signatures import PrivateKey, PublicKey
        from dubchain.wallet import DerivationPath

        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        derivation_path = DerivationPath.bip44(0, 0, 0, 0)

        account = WalletAccount(
            account_index=0,
            account_type=AccountType.STANDARD,
            derivation_path=derivation_path,
            public_key=public_key,
        )

        address = account.get_address()
        assert address.startswith("GC")
        assert len(address) == 42  # GC + 40 hex chars

    def test_wallet_account_balance_update(self):
        """Test wallet account balance update."""
        from dubchain.crypto.signatures import PrivateKey, PublicKey
        from dubchain.wallet import DerivationPath

        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        derivation_path = DerivationPath.bip44(0, 0, 0, 0)

        account = WalletAccount(
            account_index=0,
            account_type=AccountType.STANDARD,
            derivation_path=derivation_path,
            public_key=public_key,
        )

        # Update balance
        account.update_balance(1000)
        assert account.balance == 1000
        assert account.last_used is not None

        # Test negative balance
        with pytest.raises(ValueError, match="Balance cannot be negative"):
            account.update_balance(-100)

    def test_wallet_account_transaction_count(self):
        """Test wallet account transaction count increment."""
        from dubchain.crypto.signatures import PrivateKey, PublicKey
        from dubchain.wallet import DerivationPath

        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        derivation_path = DerivationPath.bip44(0, 0, 0, 0)

        account = WalletAccount(
            account_index=0,
            account_type=AccountType.STANDARD,
            derivation_path=derivation_path,
            public_key=public_key,
        )

        # Increment transaction count
        account.increment_transaction_count()
        assert account.transaction_count == 1
        assert account.last_used is not None

        account.increment_transaction_count()
        assert account.transaction_count == 2

    def test_wallet_account_serialization(self):
        """Test wallet account serialization."""
        from dubchain.crypto.signatures import PrivateKey, PublicKey
        from dubchain.wallet import DerivationPath

        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        derivation_path = DerivationPath.bip44(0, 0, 0, 0)

        account = WalletAccount(
            account_index=0,
            account_type=AccountType.STANDARD,
            derivation_path=derivation_path,
            public_key=public_key,
            private_key=private_key,
            balance=1000,
            transaction_count=5,
            label="Test Account",
            metadata={"test": "value"},
        )

        # Test to_dict
        account_dict = account.to_dict()
        assert account_dict["account_index"] == 0
        assert account_dict["account_type"] == "standard"
        assert account_dict["balance"] == 1000
        assert account_dict["transaction_count"] == 5
        assert account_dict["label"] == "Test Account"
        assert account_dict["metadata"]["test"] == "value"

        # Test from_dict
        restored_account = WalletAccount.from_dict(account_dict)
        assert restored_account.account_index == account.account_index
        assert restored_account.account_type == account.account_type
        assert restored_account.balance == account.balance
        assert restored_account.transaction_count == account.transaction_count
        assert restored_account.label == account.label
        assert restored_account.metadata == account.metadata


class TestWalletMetadata:
    """Test WalletMetadata functionality."""

    def test_wallet_metadata_creation(self):
        """Test wallet metadata creation."""
        metadata = WalletMetadata(
            name="Test Wallet",
            version="1.0.0",
            network="mainnet",
            coin_type=0,
            language=Language.ENGLISH,
        )

        assert metadata.name == "Test Wallet"
        assert metadata.version == "1.0.0"
        assert metadata.network == "mainnet"
        assert metadata.coin_type == 0
        assert metadata.language == Language.ENGLISH
        assert metadata.encryption_enabled is False
        assert metadata.backup_count == 0
        assert metadata.created_at > 0
        assert metadata.last_accessed > 0

    def test_wallet_metadata_access_time_update(self):
        """Test wallet metadata access time update."""
        metadata = WalletMetadata(name="Test Wallet")
        original_time = metadata.last_accessed

        time.sleep(0.01)  # Small delay
        metadata.update_access_time()

        # The time might be the same if the system clock resolution is low
        assert metadata.last_accessed >= original_time

    def test_wallet_metadata_serialization(self):
        """Test wallet metadata serialization."""
        metadata = WalletMetadata(
            name="Test Wallet",
            version="2.0.0",
            network="testnet",
            coin_type=1,
            language=Language.ENGLISH,
            encryption_enabled=True,
            backup_count=3,
            metadata={"custom": "value"},
        )

        # Test to_dict
        metadata_dict = metadata.to_dict()
        assert metadata_dict["name"] == "Test Wallet"
        assert metadata_dict["version"] == "2.0.0"
        assert metadata_dict["network"] == "testnet"
        assert metadata_dict["coin_type"] == 1
        assert metadata_dict["language"] == "english"
        assert metadata_dict["encryption_enabled"] is True
        assert metadata_dict["backup_count"] == 3
        assert metadata_dict["metadata"]["custom"] == "value"

        # Test from_dict
        restored_metadata = WalletMetadata.from_dict(metadata_dict)
        assert restored_metadata.name == metadata.name
        assert restored_metadata.version == metadata.version
        assert restored_metadata.network == metadata.network
        assert restored_metadata.coin_type == metadata.coin_type
        assert restored_metadata.language == metadata.language
        assert restored_metadata.encryption_enabled == metadata.encryption_enabled
        assert restored_metadata.backup_count == metadata.backup_count
        assert restored_metadata.metadata == metadata.metadata


class TestHDWallet:
    """Test HDWallet functionality."""

    def test_hd_wallet_creation_with_mnemonic(self):
        """Test HD wallet creation with mnemonic."""
        # Generate a valid mnemonic
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet", network="mainnet")

        assert wallet.mnemonic == mnemonic
        assert wallet.metadata.name == "Test Wallet"
        assert wallet.metadata.network == "mainnet"
        assert len(wallet.accounts) == 1
        assert wallet.current_account_index == 0

    def test_hd_wallet_creation_with_invalid_mnemonic(self):
        """Test HD wallet creation with invalid mnemonic."""
        with pytest.raises(WalletError, match="Invalid mnemonic phrase"):
            HDWallet(mnemonic="invalid mnemonic phrase", name="Test Wallet")

    def test_hd_wallet_account_creation(self):
        """Test HD wallet account creation."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Create additional accounts
        account1 = wallet.create_account(AccountType.STANDARD, "Account 1")
        account2 = wallet.create_account(AccountType.MULTISIG, "Account 2")

        assert len(wallet.accounts) == 3  # Default + 2 new
        assert account1.account_index == 1
        assert account2.account_index == 2
        assert account1.account_type == AccountType.STANDARD
        assert account2.account_type == AccountType.MULTISIG
        assert account1.label == "Account 1"
        assert account2.label == "Account 2"

    def test_hd_wallet_account_management(self):
        """Test HD wallet account management."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Create additional accounts
        wallet.create_account()
        wallet.create_account()

        # Test get account
        account = wallet.get_account(1)
        assert account.account_index == 1

        # Test get current account
        current = wallet.get_current_account()
        assert current.account_index == wallet.current_account_index

        # Test set current account
        wallet.set_current_account(2)
        assert wallet.current_account_index == 2

        # Test get all accounts
        all_accounts = wallet.get_all_accounts()
        assert len(all_accounts) == 3

    def test_hd_wallet_account_not_found(self):
        """Test HD wallet account not found error."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        with pytest.raises(WalletError, match="Account 5 not found"):
            wallet.get_account(5)

    def test_hd_wallet_address_generation(self):
        """Test HD wallet address generation."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Generate addresses for account 0
        addresses = wallet.get_account_addresses(0, 5)
        assert len(addresses) == 5

        for address in addresses:
            assert address.startswith("GC")
            assert len(address) == 42

    def test_hd_wallet_balance_management(self):
        """Test HD wallet balance management."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Update balance
        wallet.update_balance(0, 1000)
        assert wallet.get_balance(0) == 1000

        # Test total balance
        wallet.create_account()
        wallet.update_balance(1, 500)
        assert wallet.get_total_balance() == 1500

    def test_hd_wallet_transaction_signing(self):
        """Test HD wallet transaction signing."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Sign transaction
        transaction_data = b"test transaction data"
        signature = wallet.sign_transaction(0, transaction_data)

        assert isinstance(signature, bytes)
        assert len(signature) > 0

    def test_hd_wallet_private_key_export(self):
        """Test HD wallet private key export."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Export private key
        private_key_hex = wallet.export_private_key(0)
        assert isinstance(private_key_hex, str)
        assert len(private_key_hex) > 0

    def test_hd_wallet_public_key_export(self):
        """Test HD wallet public key export."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Export public key
        public_key_hex = wallet.export_public_key(0)
        assert isinstance(public_key_hex, str)
        assert len(public_key_hex) > 0

    def test_hd_wallet_account_import(self):
        """Test HD wallet account import."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Import account
        from dubchain.crypto.signatures import PrivateKey

        private_key = PrivateKey.generate()
        public_key_hex = private_key.get_public_key().to_hex()

        imported_account = wallet.import_account(
            public_key_hex, AccountType.WATCH_ONLY, "Imported Account"
        )

        assert imported_account.account_type == AccountType.WATCH_ONLY
        assert imported_account.label == "Imported Account"
        assert imported_account.private_key is None  # Watch-only

    def test_hd_wallet_account_removal(self):
        """Test HD wallet account removal."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Create additional accounts
        wallet.create_account()
        wallet.create_account()

        # Remove account
        wallet.remove_account(1)
        assert len(wallet.accounts) == 2
        assert 1 not in wallet.accounts

        # Test removing last account
        with pytest.raises(WalletError, match="Cannot remove the last account"):
            wallet.remove_account(0)
            wallet.remove_account(2)

    def test_hd_wallet_wallet_info(self):
        """Test HD wallet info retrieval."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        info = wallet.get_wallet_info()
        assert info["account_count"] == 1
        assert info["current_account"] == 0
        assert info["total_balance"] == 0
        assert info["network"] == "mainnet"
        assert info["coin_type"] == 0

    def test_hd_wallet_backup(self):
        """Test HD wallet backup."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Create backup
        backup = wallet.backup_wallet(include_private_keys=True)
        assert "version" in backup
        assert "timestamp" in backup
        assert "metadata" in backup
        assert "accounts" in backup
        assert len(backup["accounts"]) == 1

    def test_hd_wallet_restore(self):
        """Test HD wallet restore."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Create backup
        backup = wallet.backup_wallet(include_private_keys=True)

        # Create new wallet and restore
        new_wallet = HDWallet(mnemonic=mnemonic, name="Restored Wallet")
        new_wallet.restore_wallet(backup)

        assert len(new_wallet.accounts) == len(wallet.accounts)
        assert new_wallet.current_account_index == wallet.current_account_index

    def test_hd_wallet_serialization(self):
        """Test HD wallet serialization."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        # Test to_dict
        wallet_dict = wallet.to_dict(include_private_keys=True)
        assert "mnemonic" in wallet_dict
        assert "metadata" in wallet_dict
        assert "accounts" in wallet_dict
        assert "current_account_index" in wallet_dict

        # Test from_dict
        restored_wallet = HDWallet.from_dict(wallet_dict)
        assert restored_wallet.mnemonic == wallet.mnemonic
        assert restored_wallet.metadata.name == wallet.metadata.name
        assert len(restored_wallet.accounts) == len(wallet.accounts)

    def test_hd_wallet_create_new(self):
        """Test HD wallet creation with generated mnemonic."""
        wallet = HDWallet.create_new(name="New Wallet", network="testnet")

        assert wallet.metadata.name == "New Wallet"
        assert wallet.metadata.network == "testnet"
        assert wallet.mnemonic is not None
        assert len(wallet.mnemonic.split()) == 24  # 24-word mnemonic

    def test_hd_wallet_from_mnemonic(self):
        """Test HD wallet creation from existing mnemonic."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet.from_mnemonic(mnemonic=mnemonic, name="From Mnemonic Wallet")

        assert wallet.mnemonic == mnemonic
        assert wallet.metadata.name == "From Mnemonic Wallet"

    def test_hd_wallet_string_representation(self):
        """Test HD wallet string representation."""
        config = MnemonicConfig(language=Language.ENGLISH)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        wallet = HDWallet(mnemonic=mnemonic, name="Test Wallet")

        str_repr = str(wallet)
        assert "HDWallet" in str_repr
        assert "Test Wallet" in str_repr

        repr_str = repr(wallet)
        assert "HDWallet" in repr_str
        assert "Test Wallet" in repr_str
