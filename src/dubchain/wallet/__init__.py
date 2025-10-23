"""
GodChain Advanced Wallet System.

This package provides a sophisticated wallet system with HD key derivation,
multi-signature support, encryption, and advanced security features.
"""

import logging

logger = logging.getLogger(__name__)
from .encryption import (
    EncryptedData,
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionError,
    KeyDerivationFunction,
    PasswordManager,
    SecureStorage,
    WalletEncryption,
)
from .hd_wallet import AccountType, HDWallet, WalletAccount, WalletError, WalletMetadata
from .key_derivation import DerivationPath, DerivationType, HDKeyDerivation
from .mnemonic import Language, MnemonicConfig, MnemonicGenerator, MnemonicValidator
from .multisig_wallet import (
    MultisigConfig,
    MultisigParticipant,
    MultisigSignature,
    MultisigTransaction,
    MultisigType,
    MultisigWallet,
    SignatureStatus,
)
from .wallet_manager import (
    WalletInfo,
    WalletManager,
    WalletManagerConfig,
    WalletManagerError,
    WalletType,
)

__all__ = [
    # Key derivation
    "HDKeyDerivation",
    "DerivationPath",
    "DerivationType",
    # Mnemonic
    "MnemonicGenerator",
    "MnemonicValidator",
    "MnemonicConfig",
    "Language",
    # HD Wallet
    "HDWallet",
    "WalletAccount",
    "AccountType",
    "WalletError",
    "WalletMetadata",
    # Multi-signature
    "MultisigWallet",
    "MultisigConfig",
    "MultisigType",
    "MultisigParticipant",
    "MultisigTransaction",
    "MultisigSignature",
    "SignatureStatus",
    # Encryption
    "WalletEncryption",
    "EncryptionConfig",
    "EncryptedData",
    "SecureStorage",
    "PasswordManager",
    "EncryptionError",
    "KeyDerivationFunction",
    "EncryptionAlgorithm",
    # Wallet Manager
    "WalletManager",
    "WalletManagerConfig",
    "WalletInfo",
    "WalletType",
    "WalletManagerError",
]
