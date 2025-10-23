"""
Advanced HD (Hierarchical Deterministic) wallet implementation for GodChain.

This module provides a sophisticated HD wallet system with multi-account support,
advanced key derivation, and comprehensive wallet management features.
"""

import logging

logger = logging.getLogger(__name__)
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey
from .key_derivation import DerivationPath, DerivationType, HDKeyDerivation
from .mnemonic import Language, MnemonicConfig, MnemonicGenerator, MnemonicValidator


class WalletError(Exception):
    """Exception raised for wallet-related errors."""

    pass


class AccountType(Enum):
    """Types of wallet accounts."""

    STANDARD = "standard"
    MULTISIG = "multisig"
    HARDWARE = "hardware"
    WATCH_ONLY = "watch_only"


@dataclass
class WalletAccount:
    """Represents a wallet account with key derivation."""

    account_index: int
    account_type: AccountType
    derivation_path: DerivationPath
    public_key: PublicKey
    private_key: Optional[PrivateKey] = None
    balance: int = 0
    transaction_count: int = 0
    last_used: Optional[int] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate account data."""
        if self.account_index < 0:
            raise ValueError("Account index must be non-negative")

        if hasattr(self, "balance") and self.balance < 0:
            raise ValueError("Balance must be non-negative")

        if hasattr(self, "transaction_count") and self.transaction_count < 0:
            raise ValueError("Transaction count must be non-negative")

    def get_address(self, address_type: str = "standard") -> str:
        """Get address for this account."""
        # This would integrate with address generation
        # For now, return a simple hash-based address
        address_hash = SHA256Hasher.hash(self.public_key.to_bytes())
        return f"GC{address_hash.to_hex()[:40]}"

    def update_balance(self, new_balance: int) -> None:
        """Update account balance."""
        if new_balance < 0:
            raise ValueError("Balance cannot be negative")
        self.balance = new_balance
        self.last_used = int(time.time())

    def increment_transaction_count(self) -> None:
        """Increment transaction count."""
        self.transaction_count += 1
        self.last_used = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert account to dictionary."""
        return {
            "account_index": self.account_index,
            "account_type": self.account_type.value,
            "derivation_path": self.derivation_path.to_string(),
            "public_key": self.public_key.to_hex(),
            "private_key": self.private_key.to_hex() if self.private_key else None,
            "balance": self.balance,
            "transaction_count": self.transaction_count,
            "last_used": self.last_used,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalletAccount":
        """Create account from dictionary."""
        derivation_path = DerivationPath.from_string(data["derivation_path"])
        public_key = PublicKey.from_hex(data["public_key"])
        private_key = (
            PrivateKey.from_hex(data["private_key"])
            if data.get("private_key")
            else None
        )

        return cls(
            account_index=data["account_index"],
            account_type=AccountType(data["account_type"]),
            derivation_path=derivation_path,
            public_key=public_key,
            private_key=private_key,
            balance=data.get("balance", 0),
            transaction_count=data.get("transaction_count", 0),
            last_used=data.get("last_used"),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WalletMetadata:
    """Wallet metadata and configuration."""

    name: str
    version: str = "1.0.0"
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_accessed: int = field(default_factory=lambda: int(time.time()))
    network: str = "mainnet"
    coin_type: int = 0  # GodChain coin type
    language: Language = Language.ENGLISH
    encryption_enabled: bool = False
    backup_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access_time(self) -> None:
        """Update last accessed time."""
        self.last_accessed = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "network": self.network,
            "coin_type": self.coin_type,
            "language": self.language.value,
            "encryption_enabled": self.encryption_enabled,
            "backup_count": self.backup_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalletMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", int(time.time())),
            last_accessed=data.get("last_accessed", int(time.time())),
            network=data.get("network", "mainnet"),
            coin_type=data.get("coin_type", 0),
            language=Language(data.get("language", "english")),
            encryption_enabled=data.get("encryption_enabled", False),
            backup_count=data.get("backup_count", 0),
            metadata=data.get("metadata", {}),
        )


class HDWallet:
    """Advanced HD wallet implementation."""

    def __init__(
        self,
        mnemonic: str,
        passphrase: str = "",
        name: str = "GodChain Wallet",
        network: str = "mainnet",
    ):
        """Initialize HD wallet."""
        self.mnemonic = mnemonic
        self.passphrase = passphrase
        self.metadata = WalletMetadata(name=name, network=network)

        # Validate mnemonic
        from .mnemonic import MnemonicConfig

        config = MnemonicConfig(validate_checksum=False)
        validator = MnemonicValidator(config)
        if not validator.validate(mnemonic):
            raise WalletError("Invalid mnemonic phrase")

        # Generate seed from mnemonic
        self.seed = self._mnemonic_to_seed(mnemonic, passphrase)

        # Initialize key derivation
        self.key_derivation = HDKeyDerivation(self.seed, network)

        # Initialize accounts
        self.accounts: Dict[int, WalletAccount] = {}
        self.current_account_index = 0

        # Create default account
        self.create_account()

    def _mnemonic_to_seed(self, mnemonic: str, passphrase: str) -> bytes:
        """Convert mnemonic to seed."""
        import hashlib
        import unicodedata

        # Normalize mnemonic and passphrase
        mnemonic_normalized = unicodedata.normalize("NFKD", mnemonic)
        passphrase_normalized = unicodedata.normalize("NFKD", passphrase)

        # Use PBKDF2 with HMAC-SHA512
        salt = ("mnemonic" + passphrase_normalized).encode("utf-8")
        return hashlib.pbkdf2_hmac(
            "sha512", mnemonic_normalized.encode("utf-8"), salt, 2048
        )

    def create_account(
        self,
        account_type: AccountType = AccountType.STANDARD,
        label: Optional[str] = None,
    ) -> WalletAccount:
        """Create a new account."""
        account_index = len(self.accounts)

        # Create derivation path
        derivation_path = DerivationPath.bip44(
            coin_type=self.metadata.coin_type,
            account=account_index,
            change=0,
            address_index=0,
        )

        # Derive keys
        private_key = self.key_derivation.derive_key(derivation_path)
        public_key = private_key.get_public_key()

        # Create account
        account = WalletAccount(
            account_index=account_index,
            account_type=account_type,
            derivation_path=derivation_path,
            public_key=public_key,
            private_key=private_key,
            label=label or f"Account {account_index + 1}",
        )

        self.accounts[account_index] = account
        return account

    def get_account(self, account_index: int) -> WalletAccount:
        """Get account by index."""
        if account_index not in self.accounts:
            raise WalletError(f"Account {account_index} not found")
        return self.accounts[account_index]

    def get_current_account(self) -> WalletAccount:
        """Get current active account."""
        return self.get_account(self.current_account_index)

    def set_current_account(self, account_index: int) -> None:
        """Set current active account."""
        if account_index not in self.accounts:
            raise WalletError(f"Account {account_index} not found")
        self.current_account_index = account_index
        self.metadata.update_access_time()

    def get_all_accounts(self) -> List[WalletAccount]:
        """Get all accounts."""
        return list(self.accounts.values())

    def get_account_addresses(self, account_index: int, count: int = 10) -> List[str]:
        """Get addresses for an account."""
        account = self.get_account(account_index)
        addresses = []

        for i in range(count):
            # Create derivation path for this address
            address_path = DerivationPath(
                purpose=account.derivation_path.purpose,
                coin_type=account.derivation_path.coin_type,
                account=account.derivation_path.account,
                change=account.derivation_path.change,
                address_index=i,
            )

            # Derive public key
            public_key = self.key_derivation.derive_public_key(address_path)

            # Generate address (simplified)
            address_hash = SHA256Hasher.hash(public_key.to_bytes())
            address = f"GC{address_hash.to_hex()[:40]}"
            addresses.append(address)

        return addresses

    def get_balance(self, account_index: Optional[int] = None) -> int:
        """Get balance for account."""
        if account_index is None:
            account_index = self.current_account_index

        account = self.get_account(account_index)
        return account.balance

    def get_total_balance(self) -> int:
        """Get total balance across all accounts."""
        return sum(account.balance for account in self.accounts.values())

    def update_balance(self, account_index: int, new_balance: int) -> None:
        """Update account balance."""
        account = self.get_account(account_index)
        account.update_balance(new_balance)
        self.metadata.update_access_time()

    def sign_transaction(self, account_index: int, transaction_data: bytes) -> bytes:
        """Sign transaction with account private key."""
        account = self.get_account(account_index)
        if not account.private_key:
            raise WalletError("Account does not have private key")

        # Sign transaction
        signature = account.private_key.sign(transaction_data)
        return signature.to_bytes()

    def verify_signature(
        self, account_index: int, data: bytes, signature: bytes
    ) -> bool:
        """Verify signature with account public key."""
        account = self.get_account(account_index)

        # Implement actual signature verification
        try:
            from ..crypto.signatures import Signature
            
            # Parse signature if it's a hex string
            if isinstance(signature, str):
                signature_bytes = bytes.fromhex(signature)
            else:
                signature_bytes = signature
            
            # Create signature object
            sig = Signature.from_bytes(signature_bytes)
            
            # Get account's public key
            public_key = account.public_key
            
            # Verify signature
            return public_key.verify_signature(data, sig)
            
        except Exception as e:
            # Log error and return False
            logger.info(f"Signature verification failed: {e}")
            return False

    def export_private_key(self, account_index: int) -> str:
        """Export private key for account."""
        account = self.get_account(account_index)
        if not account.private_key:
            raise WalletError("Account does not have private key")

        return account.private_key.to_hex()

    def export_public_key(self, account_index: int) -> str:
        """Export public key for account."""
        account = self.get_account(account_index)
        return account.public_key.to_hex()

    def import_account(
        self,
        public_key: str,
        account_type: AccountType = AccountType.WATCH_ONLY,
        label: Optional[str] = None,
    ) -> WalletAccount:
        """Import account from public key."""
        try:
            imported_public_key = PublicKey.from_hex(public_key)
        except Exception:
            raise WalletError("Invalid public key format")

        account_index = len(self.accounts)

        # Create account with imported public key
        account = WalletAccount(
            account_index=account_index,
            account_type=account_type,
            derivation_path=DerivationPath.bip44(
                self.metadata.coin_type, account_index, 0, 0
            ),
            public_key=imported_public_key,
            private_key=None,  # Watch-only account
            label=label or f"Imported Account {account_index + 1}",
        )

        self.accounts[account_index] = account
        return account

    def remove_account(self, account_index: int) -> None:
        """Remove account from wallet."""
        if account_index not in self.accounts:
            raise WalletError(f"Account {account_index} not found")

        if len(self.accounts) <= 1:
            raise WalletError("Cannot remove the last account")

        del self.accounts[account_index]

        # Adjust current account index if necessary
        if self.current_account_index >= account_index:
            self.current_account_index = max(0, self.current_account_index - 1)

    def get_wallet_info(self) -> Dict[str, Any]:
        """Get comprehensive wallet information."""
        return {
            "metadata": self.metadata.to_dict(),
            "account_count": len(self.accounts),
            "current_account": self.current_account_index,
            "total_balance": self.get_total_balance(),
            "accounts": [account.to_dict() for account in self.accounts.values()],
            "network": self.metadata.network,
            "coin_type": self.metadata.coin_type,
        }

    def backup_wallet(self, include_private_keys: bool = False) -> Dict[str, Any]:
        """Create wallet backup."""
        backup_data = {
            "version": "1.0.0",
            "timestamp": int(time.time()),
            "mnemonic": self.mnemonic,
            "passphrase": self.passphrase,
            "metadata": self.metadata.to_dict(),
            "accounts": [],
            "current_account_index": self.current_account_index,
        }

        for account in self.accounts.values():
            account_data = account.to_dict()
            if not include_private_keys:
                account_data["private_key"] = None
            backup_data["accounts"].append(account_data)

        self.metadata.backup_count += 1
        return backup_data

    def restore_wallet(self, backup_data: Dict[str, Any]) -> None:
        """Restore wallet from backup."""
        # Validate backup format
        if "version" not in backup_data or "metadata" not in backup_data:
            raise WalletError("Invalid backup format")

        # Restore metadata
        self.metadata = WalletMetadata.from_dict(backup_data["metadata"])

        # Clear existing accounts
        self.accounts.clear()

        # Restore accounts
        for account_data in backup_data.get("accounts", []):
            account = WalletAccount.from_dict(account_data)
            self.accounts[account.account_index] = account

        # Set current account
        if self.accounts:
            self.current_account_index = min(self.accounts.keys())

    def to_dict(self, include_private_keys: bool = False) -> Dict[str, Any]:
        """Convert wallet to dictionary."""
        return {
            "mnemonic": self.mnemonic if include_private_keys else None,
            "passphrase": self.passphrase if include_private_keys else None,
            "metadata": self.metadata.to_dict(),
            "accounts": [
                account.to_dict()
                if include_private_keys
                else {**account.to_dict(), "private_key": None}
                for account in self.accounts.values()
            ],
            "current_account_index": self.current_account_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HDWallet":
        """Create wallet from dictionary."""
        if not data.get("mnemonic"):
            raise WalletError("Cannot restore wallet without mnemonic")

        wallet = cls(
            mnemonic=data["mnemonic"],
            passphrase=data.get("passphrase", ""),
            name=data["metadata"]["name"],
            network=data["metadata"]["network"],
        )

        # Restore accounts
        for account_data in data.get("accounts", []):
            account = WalletAccount.from_dict(account_data)
            wallet.accounts[account.account_index] = account

        wallet.current_account_index = data.get("current_account_index", 0)
        return wallet

    @classmethod
    def create_new(
        cls,
        name: str = "GodChain Wallet",
        network: str = "mainnet",
        language: Language = Language.ENGLISH,
    ) -> "HDWallet":
        """Create a new wallet with generated mnemonic."""
        # Generate mnemonic
        config = MnemonicConfig(language=language)
        generator = MnemonicGenerator(config)
        mnemonic = generator.generate()

        return cls(mnemonic=mnemonic, name=name, network=network)

    @classmethod
    def from_mnemonic(
        cls,
        mnemonic: str,
        passphrase: str = "",
        name: str = "GodChain Wallet",
        network: str = "mainnet",
    ) -> "HDWallet":
        """Create wallet from existing mnemonic."""
        return cls(mnemonic=mnemonic, passphrase=passphrase, name=name, network=network)

    def __str__(self) -> str:
        """String representation."""
        return f"HDWallet(name={self.metadata.name}, accounts={len(self.accounts)}, balance={self.get_total_balance()})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"HDWallet(name={self.metadata.name}, network={self.metadata.network}, "
            f"accounts={len(self.accounts)}, total_balance={self.get_total_balance()})"
        )
