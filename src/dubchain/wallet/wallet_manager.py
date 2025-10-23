"""
Advanced wallet manager for GodChain.

This module provides a comprehensive wallet management system that integrates
HD wallets, multi-signature wallets, encryption, and secure storage.
"""

import logging

logger = logging.getLogger(__name__)
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey
from .encryption import (
    EncryptionConfig,
    PasswordManager,
    SecureStorage,
    WalletEncryption,
)
from .hd_wallet import AccountType, HDWallet, WalletAccount, WalletError
from .mnemonic import Language, MnemonicConfig, MnemonicGenerator, MnemonicValidator
from .multisig_wallet import MultisigConfig, MultisigType, MultisigWallet


class WalletManagerError(Exception):
    """Exception raised for wallet manager errors."""

    pass


class WalletType(Enum):
    """Types of wallets managed by the wallet manager."""

    HD_WALLET = "hd_wallet"
    MULTISIG_WALLET = "multisig_wallet"
    HARDWARE_WALLET = "hardware_wallet"
    WATCH_ONLY_WALLET = "watch_only_wallet"


@dataclass
class WalletInfo:
    """Information about a managed wallet."""

    wallet_id: str
    wallet_type: WalletType
    name: str
    created_at: int
    last_accessed: int
    is_encrypted: bool
    network: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wallet_id": self.wallet_id,
            "wallet_type": self.wallet_type.value,
            "name": self.name,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "is_encrypted": self.is_encrypted,
            "network": self.network,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalletInfo":
        """Create from dictionary."""
        return cls(
            wallet_id=data["wallet_id"],
            wallet_type=WalletType(data["wallet_type"]),
            name=data["name"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            is_encrypted=data["is_encrypted"],
            network=data["network"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class WalletManagerConfig:
    """Configuration for wallet manager."""

    storage_path: str
    default_network: str = "mainnet"
    encryption_enabled: bool = True
    encryption_config: Optional[EncryptionConfig] = None
    mnemonic_language: Language = Language.ENGLISH
    auto_backup: bool = True
    backup_interval: int = 86400  # 24 hours
    max_wallets: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not self.storage_path:
            raise ValueError("Storage path cannot be empty")

        if self.backup_interval <= 0:
            raise ValueError("Backup interval must be positive")

        if self.max_wallets <= 0:
            raise ValueError("Max wallets must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_path": self.storage_path,
            "default_network": self.default_network,
            "encryption_enabled": self.encryption_enabled,
            "encryption_config": self.encryption_config.to_dict()
            if self.encryption_config
            else None,
            "mnemonic_language": self.mnemonic_language.value
            if hasattr(self.mnemonic_language, "value")
            else self.mnemonic_language,
            "auto_backup": self.auto_backup,
            "backup_interval": self.backup_interval,
            "max_wallets": self.max_wallets,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalletManagerConfig":
        """Create from dictionary."""
        encryption_config = None
        if data.get("encryption_config"):
            encryption_config = EncryptionConfig.from_dict(data["encryption_config"])

        return cls(
            storage_path=data["storage_path"],
            default_network=data.get("default_network", "mainnet"),
            encryption_enabled=data.get("encryption_enabled", True),
            encryption_config=encryption_config,
            mnemonic_language=data.get("mnemonic_language", "english"),
            auto_backup=data.get("auto_backup", True),
            backup_interval=data.get("backup_interval", 86400),
            max_wallets=data.get("max_wallets", 100),
            metadata=data.get("metadata", {}),
        )


class WalletManager:
    """Advanced wallet management system."""

    def __init__(self, config: WalletManagerConfig):
        """Initialize wallet manager."""
        self.config = config
        self.wallets: Dict[str, Union[HDWallet, MultisigWallet]] = {}
        self.wallet_info: Dict[str, WalletInfo] = {}

        # Initialize encryption
        self.encryption = WalletEncryption(config.encryption_config)
        self.secure_storage = SecureStorage(self.encryption)
        self.password_manager = PasswordManager()

        # Initialize mnemonic components
        mnemonic_config = MnemonicConfig(
            language=config.mnemonic_language, validate_checksum=False
        )
        self.mnemonic_generator = MnemonicGenerator(mnemonic_config)
        self.mnemonic_validator = MnemonicValidator(mnemonic_config)

        # Set up storage
        self.secure_storage.set_storage_path(config.storage_path)

        # Load existing wallets
        self._load_wallets()

    def _generate_wallet_id(self) -> str:
        """Generate unique wallet ID."""
        timestamp = int(time.time())
        random_data = os.urandom(16)
        wallet_hash = SHA256Hasher.hash(f"{timestamp}{random_data.hex()}")
        return f"wallet_{wallet_hash.to_hex()[:16]}"

    def _load_wallets(self) -> None:
        """Load existing wallets from storage."""
        try:
            # Load wallet registry
            registry_path = os.path.join(
                self.config.storage_path, "wallet_registry.json"
            )
            if os.path.exists(registry_path):
                with open(registry_path, "r") as f:
                    registry_data = json.load(f)

                for wallet_id, wallet_info_data in registry_data.get(
                    "wallets", {}
                ).items():
                    self.wallet_info[wallet_id] = WalletInfo.from_dict(wallet_info_data)
        except Exception as e:
            # If loading fails, start with empty registry
            pass

    def _save_wallet_registry(self) -> None:
        """Save wallet registry to storage."""
        registry_data = {
            "version": "1.0.0",
            "created_at": int(time.time()),
            "wallets": {
                wallet_id: info.to_dict()
                for wallet_id, info in self.wallet_info.items()
            },
        }

        registry_path = os.path.join(self.config.storage_path, "wallet_registry.json")
        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

    def create_hd_wallet(
        self,
        name: str,
        password: Optional[str] = None,
        network: Optional[str] = None,
        mnemonic: Optional[str] = None,
    ) -> str:
        """Create a new HD wallet."""
        if len(self.wallets) >= self.config.max_wallets:
            raise WalletManagerError("Maximum number of wallets reached")

        network = network or self.config.default_network
        wallet_id = self._generate_wallet_id()

        # Generate or validate mnemonic
        if mnemonic:
            if not self.mnemonic_validator.validate(mnemonic):
                raise WalletManagerError("Invalid mnemonic phrase")
        else:
            mnemonic = self.mnemonic_generator.generate()

        # Create HD wallet
        hd_wallet = HDWallet(mnemonic=mnemonic, name=name, network=network)

        # Store wallet
        self.wallets[wallet_id] = hd_wallet

        # Create wallet info
        wallet_info = WalletInfo(
            wallet_id=wallet_id,
            wallet_type=WalletType.HD_WALLET,
            name=name,
            created_at=int(time.time()),
            last_accessed=int(time.time()),
            is_encrypted=bool(password),
            network=network,
        )
        self.wallet_info[wallet_id] = wallet_info

        # Save wallet if encrypted
        if password and self.config.encryption_enabled:
            self._save_wallet(wallet_id, password)

        # Save registry
        self._save_wallet_registry()

        return wallet_id

    def create_multisig_wallet(
        self, name: str, config: MultisigConfig, password: Optional[str] = None
    ) -> str:
        """Create a new multi-signature wallet."""
        if len(self.wallets) >= self.config.max_wallets:
            raise WalletManagerError("Maximum number of wallets reached")

        wallet_id = self._generate_wallet_id()

        # Create multisig wallet
        multisig_wallet = MultisigWallet(wallet_id=wallet_id, config=config, name=name)

        # Store wallet
        self.wallets[wallet_id] = multisig_wallet

        # Create wallet info
        wallet_info = WalletInfo(
            wallet_id=wallet_id,
            wallet_type=WalletType.MULTISIG_WALLET,
            name=name,
            created_at=int(time.time()),
            last_accessed=int(time.time()),
            is_encrypted=bool(password),
            network=self.config.default_network,
        )
        self.wallet_info[wallet_id] = wallet_info

        # Save wallet if encrypted
        if password and self.config.encryption_enabled:
            self._save_wallet(wallet_id, password)

        # Save registry
        self._save_wallet_registry()

        return wallet_id

    def load_wallet(
        self, wallet_id: str, password: Optional[str] = None
    ) -> Union[HDWallet, MultisigWallet]:
        """Load wallet into memory."""
        if wallet_id not in self.wallet_info:
            raise WalletManagerError(f"Wallet {wallet_id} not found")

        # Check if already loaded
        if wallet_id in self.wallets:
            self.wallet_info[wallet_id].last_accessed = int(time.time())
            return self.wallets[wallet_id]

        # Load from storage
        wallet_info = self.wallet_info[wallet_id]

        if wallet_info.is_encrypted and not password:
            raise WalletManagerError("Password required for encrypted wallet")

        try:
            if wallet_info.wallet_type == WalletType.HD_WALLET:
                wallet_data = self._load_wallet_data(wallet_id, password)
                wallet = HDWallet.from_dict(wallet_data)
            elif wallet_info.wallet_type == WalletType.MULTISIG_WALLET:
                wallet_data = self._load_wallet_data(wallet_id, password)
                wallet = MultisigWallet.from_dict(wallet_data)
            else:
                raise WalletManagerError(
                    f"Unsupported wallet type: {wallet_info.wallet_type}"
                )

            # Store in memory
            self.wallets[wallet_id] = wallet
            wallet_info.last_accessed = int(time.time())

            return wallet

        except Exception as e:
            raise WalletManagerError(f"Failed to load wallet: {str(e)}")

    def unload_wallet(self, wallet_id: str) -> None:
        """Unload wallet from memory."""
        if wallet_id in self.wallets:
            del self.wallets[wallet_id]

    def delete_wallet(self, wallet_id: str, password: Optional[str] = None) -> bool:
        """Delete wallet permanently."""
        if wallet_id not in self.wallet_info:
            raise WalletManagerError(f"Wallet {wallet_id} not found")

        wallet_info = self.wallet_info[wallet_id]

        # Verify password for encrypted wallets
        if wallet_info.is_encrypted and password:
            try:
                self._load_wallet_data(wallet_id, password)
            except Exception:
                raise WalletManagerError("Invalid password")

        # Remove from memory
        if wallet_id in self.wallets:
            del self.wallets[wallet_id]

        # Remove from registry
        del self.wallet_info[wallet_id]

        # Remove from storage
        success = self.secure_storage.delete_wallet(wallet_id)

        # Save registry
        self._save_wallet_registry()

        return success

    def _save_wallet(self, wallet_id: str, password: str) -> None:
        """Save wallet to storage."""
        if wallet_id not in self.wallets:
            raise WalletManagerError(f"Wallet {wallet_id} not found")

        wallet = self.wallets[wallet_id]
        wallet_data = wallet.to_dict(include_private_keys=True)

        self.secure_storage.save_wallet(wallet_data, password, wallet_id)

    def _load_wallet_data(
        self, wallet_id: str, password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load wallet data from storage."""
        if password:
            return self.secure_storage.load_wallet(wallet_id, password)
        else:
            # Load unencrypted wallet
            wallet_path = os.path.join(self.config.storage_path, f"{wallet_id}.json")
            with open(wallet_path, "r") as f:
                return json.load(f)

    def get_wallet_list(self) -> List[WalletInfo]:
        """Get list of all wallets."""
        return list(self.wallet_info.values())

    def get_wallet_info(self, wallet_id: str) -> WalletInfo:
        """Get wallet information."""
        if wallet_id not in self.wallet_info:
            raise WalletManagerError(f"Wallet {wallet_id} not found")
        return self.wallet_info[wallet_id]

    def rename_wallet(self, wallet_id: str, new_name: str) -> None:
        """Rename wallet."""
        if wallet_id not in self.wallet_info:
            raise WalletManagerError(f"Wallet {wallet_id} not found")

        self.wallet_info[wallet_id].name = new_name
        self._save_wallet_registry()

    def export_wallet(
        self,
        wallet_id: str,
        password: Optional[str] = None,
        include_private_keys: bool = False,
    ) -> Dict[str, Any]:
        """Export wallet data."""
        wallet = self.load_wallet(wallet_id, password)
        return wallet.to_dict(include_private_keys=include_private_keys)

    def import_wallet(
        self, wallet_data: Dict[str, Any], name: str, password: Optional[str] = None
    ) -> str:
        """Import wallet from data."""
        if len(self.wallets) >= self.config.max_wallets:
            raise WalletManagerError("Maximum number of wallets reached")

        wallet_id = self._generate_wallet_id()

        # Determine wallet type
        if "mnemonic" in wallet_data:
            wallet_type = WalletType.HD_WALLET
            wallet = HDWallet.from_dict(wallet_data)
        elif "config" in wallet_data and "multisig_type" in wallet_data["config"]:
            wallet_type = WalletType.MULTISIG_WALLET
            wallet = MultisigWallet.from_dict(wallet_data)
        else:
            raise WalletManagerError("Unknown wallet format")

        # Store wallet
        self.wallets[wallet_id] = wallet

        # Create wallet info
        wallet_info = WalletInfo(
            wallet_id=wallet_id,
            wallet_type=wallet_type,
            name=name,
            created_at=int(time.time()),
            last_accessed=int(time.time()),
            is_encrypted=bool(password),
            network=wallet_data.get("metadata", {}).get(
                "network", self.config.default_network
            ),
        )
        self.wallet_info[wallet_id] = wallet_info

        # Save wallet if encrypted
        if password and self.config.encryption_enabled:
            self._save_wallet(wallet_id, password)

        # Save registry
        self._save_wallet_registry()

        return wallet_id

    def backup_wallet(
        self, wallet_id: str, backup_path: str, password: Optional[str] = None
    ) -> str:
        """Create wallet backup."""
        wallet = self.load_wallet(wallet_id, password)
        backup_data = wallet.backup_wallet(include_private_keys=bool(password))

        # Add metadata
        backup_data["backup_info"] = {
            "wallet_id": wallet_id,
            "backup_created": int(time.time()),
            "wallet_type": self.wallet_info[wallet_id].wallet_type.value,
            "network": self.wallet_info[wallet_id].network,
        }

        # Save backup
        backup_file = os.path.join(
            backup_path, f"{wallet_id}_backup_{int(time.time())}.json"
        )
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)

        return backup_file

    def restore_wallet(
        self, backup_file: str, name: str, password: Optional[str] = None
    ) -> str:
        """Restore wallet from backup."""
        with open(backup_file, "r") as f:
            backup_data = json.load(f)

        # Extract wallet data
        wallet_data = {
            k: v
            for k, v in backup_data.items()
            if k not in ["backup_info", "version", "timestamp"]
        }

        return self.import_wallet(wallet_data, name, password)

    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        return self.password_manager.validate_password(password)

    def generate_strong_password(self, length: int = 16) -> str:
        """Generate strong password."""
        return self.password_manager.generate_strong_password(length)

    def get_password_strength(self, password: str) -> str:
        """Get password strength rating."""
        return self.password_manager.get_password_strength(password)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get wallet manager information."""
        return {
            "config": self.config.to_dict(),
            "wallet_count": len(self.wallet_info),
            "loaded_wallets": len(self.wallets),
            "storage_path": self.config.storage_path,
            "encryption_enabled": self.config.encryption_enabled,
            "auto_backup": self.config.auto_backup,
            "max_wallets": self.config.max_wallets,
        }

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """Clean up old backup files."""
        cleaned_count = 0
        current_time = int(time.time())
        max_age_seconds = max_age_days * 86400

        for filename in os.listdir(self.config.storage_path):
            if "_backup_" in filename and filename.endswith(".json"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = filename.split("_backup_")[1].split(".")[0]
                    backup_time = int(timestamp_str)

                    if current_time - backup_time > max_age_seconds:
                        file_path = os.path.join(self.config.storage_path, filename)
                        os.remove(file_path)
                        cleaned_count += 1
                except (ValueError, IndexError):
                    # Skip files with invalid naming
                    continue

        return cleaned_count

    def __str__(self) -> str:
        """String representation."""
        return f"WalletManager(wallets={len(self.wallet_info)}, loaded={len(self.wallets)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"WalletManager(storage_path={self.config.storage_path}, "
            f"wallets={len(self.wallet_info)}, loaded={len(self.wallets)})"
        )
