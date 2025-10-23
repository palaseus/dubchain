"""
Advanced key derivation for GodChain wallets.

This module implements BIP32/44/49/84 compliant hierarchical deterministic
key derivation with advanced security features and custom derivation paths.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey


class KeyDerivationError(Exception):
    """Exception raised for key derivation errors."""

    pass


class DerivationType(Enum):
    """Types of key derivation."""

    BIP32 = "bip32"
    BIP44 = "bip44"
    BIP49 = "bip49"
    BIP84 = "bip84"
    CUSTOM = "custom"


@dataclass
class DerivationPath:
    """Hierarchical derivation path for HD wallets."""

    purpose: int
    coin_type: int
    account: int
    change: int
    address_index: int

    def __post_init__(self):
        """Validate derivation path components."""
        if self.purpose < 0 or self.purpose > 0x7FFFFFFF:
            raise ValueError("Purpose must be between 0 and 0x7FFFFFFF")
        if self.coin_type < 0 or self.coin_type > 0x7FFFFFFF:
            raise ValueError("Coin type must be between 0 and 0x7FFFFFFF")
        if self.account < 0 or self.account > 0x7FFFFFFF:
            raise ValueError("Account must be between 0 and 0x7FFFFFFF")
        if self.change not in [0, 1]:
            raise ValueError("Change must be 0 (external) or 1 (internal)")
        if self.address_index < 0 or self.address_index > 0x7FFFFFFF:
            raise ValueError("Address index must be between 0 and 0x7FFFFFFF")

    def to_string(self, hardened: bool = True) -> str:
        """Convert derivation path to string format."""
        components = [
            f"{self.purpose}'" if hardened else str(self.purpose),
            f"{self.coin_type}'" if hardened else str(self.coin_type),
            f"{self.account}'" if hardened else str(self.account),
            str(self.change),
            str(self.address_index),
        ]
        return "m/" + "/".join(components)

    def to_bytes(self) -> bytes:
        """Convert derivation path to bytes."""
        return b"/".join(
            [
                f"{self.purpose}'".encode(),
                f"{self.coin_type}'".encode(),
                f"{self.account}'".encode(),
                str(self.change).encode(),
                str(self.address_index).encode(),
            ]
        )

    @classmethod
    def from_string(cls, path_string: str) -> "DerivationPath":
        """Create derivation path from string."""
        if not path_string.startswith("m/"):
            raise ValueError("Path must start with 'm/'")

        components = path_string[2:].split("/")
        if len(components) != 5:
            raise ValueError("Path must have exactly 5 components")

        purpose = int(components[0].rstrip("'"))
        coin_type = int(components[1].rstrip("'"))
        account = int(components[2].rstrip("'"))
        change = int(components[3])
        address_index = int(components[4])

        return cls(purpose, coin_type, account, change, address_index)

    @classmethod
    def bip44(
        cls, coin_type: int, account: int = 0, change: int = 0, address_index: int = 0
    ) -> "DerivationPath":
        """Create BIP44 derivation path."""
        return cls(44, coin_type, account, change, address_index)

    @classmethod
    def bip49(
        cls, coin_type: int, account: int = 0, change: int = 0, address_index: int = 0
    ) -> "DerivationPath":
        """Create BIP49 derivation path (P2WPKH-P2SH)."""
        return cls(49, coin_type, account, change, address_index)

    @classmethod
    def bip84(
        cls, coin_type: int, account: int = 0, change: int = 0, address_index: int = 0
    ) -> "DerivationPath":
        """Create BIP84 derivation path (P2WPKH)."""
        return cls(84, coin_type, account, change, address_index)

    def __str__(self) -> str:
        """String representation."""
        return self.to_string()

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DerivationPath(purpose={self.purpose}, coin_type={self.coin_type}, account={self.account}, change={self.change}, address_index={self.address_index})"


@dataclass
class ExtendedKey:
    """Extended key for HD wallet derivation."""

    key: bytes
    chain_code: bytes
    depth: int
    parent_fingerprint: bytes
    child_number: int

    def __post_init__(self):
        """Validate extended key."""
        if len(self.key) != 32:
            raise ValueError("Key must be 32 bytes")
        if len(self.chain_code) != 32:
            raise ValueError("Chain code must be 32 bytes")
        if len(self.parent_fingerprint) != 4:
            raise ValueError("Parent fingerprint must be 4 bytes")
        if self.depth < 0 or self.depth > 255:
            raise ValueError("Depth must be between 0 and 255")
        if self.child_number < 0 or self.child_number > 0x7FFFFFFF:
            raise ValueError("Child number must be between 0 and 0x7FFFFFFF")

    def fingerprint(self) -> bytes:
        """Get key fingerprint."""
        public_key = PrivateKey.from_bytes(self.key).get_public_key()
        public_key_bytes = public_key.to_bytes()
        return SHA256Hasher.hash(public_key_bytes).value[:4]

    def to_private_key(self) -> PrivateKey:
        """Convert to private key."""
        return PrivateKey.from_bytes(self.key)

    def to_public_key(self) -> PublicKey:
        """Convert to public key."""
        return self.to_private_key().get_public_key()


class KeyDerivation:
    """Base class for key derivation algorithms."""

    def __init__(self, seed: bytes):
        """Initialize with seed."""
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")
        self.seed = seed

    def derive_key(self, path: DerivationPath) -> PrivateKey:
        """
        Derive private key from path.
        
        This is an abstract base class method that must be implemented by subclasses.
        The NotImplementedError is intentional - concrete implementations should
        inherit from this class and provide their own derive_key implementation.
        """
        raise NotImplementedError("derive_key must be implemented by subclasses")

    def derive_public_key(self, path: DerivationPath) -> PublicKey:
        """Derive public key from path."""
        return self.derive_key(path).get_public_key()


class HDKeyDerivation(KeyDerivation):
    """BIP32 compliant HD key derivation."""

    def __init__(self, seed: bytes, network: str = "mainnet"):
        """Initialize HD key derivation."""
        super().__init__(seed)
        self.network = network
        self.master_key, self.master_chain_code = self._derive_master_key()

    def _derive_master_key(self) -> Tuple[bytes, bytes]:
        """Derive master key and chain code from seed."""
        # Use HMAC-SHA512 as specified in BIP32
        hmac_result = hmac.new(b"Bitcoin seed", self.seed, hashlib.sha512).digest()

        master_key = hmac_result[:32]
        master_chain_code = hmac_result[32:]

        return master_key, master_chain_code

    def _derive_child_key(
        self, parent_key: bytes, parent_chain_code: bytes, child_number: int
    ) -> Tuple[bytes, bytes]:
        """Derive child key from parent."""
        if child_number >= 0x80000000:
            # Hardened derivation
            key_data = b"\x00" + parent_key + child_number.to_bytes(4, "big")
        else:
            # Non-hardened derivation
            parent_public_key = PrivateKey.from_bytes(parent_key).get_public_key()
            key_data = parent_public_key.to_bytes() + child_number.to_bytes(4, "big")

        hmac_result = hmac.new(parent_chain_code, key_data, hashlib.sha512).digest()

        child_key = hmac_result[:32]
        child_chain_code = hmac_result[32:]

        # Add child key to parent key (mod n)
        parent_private_key = PrivateKey.from_bytes(parent_key)
        child_private_key = parent_private_key.add_scalar(child_key)

        return child_private_key.to_bytes(), child_chain_code

    def derive_key(self, path: DerivationPath) -> PrivateKey:
        """Derive private key from derivation path."""
        current_key = self.master_key
        current_chain_code = self.master_chain_code

        # Derive through each level of the path
        components = [
            path.purpose | 0x80000000,  # Hardened
            path.coin_type | 0x80000000,  # Hardened
            path.account | 0x80000000,  # Hardened
            path.change,  # Non-hardened
            path.address_index,  # Non-hardened
        ]

        for component in components:
            current_key, current_chain_code = self._derive_child_key(
                current_key, current_chain_code, component
            )

        return PrivateKey.from_bytes(current_key)

    def derive_extended_key(self, path: DerivationPath) -> ExtendedKey:
        """Derive extended key from path."""
        current_key = self.master_key
        current_chain_code = self.master_chain_code
        depth = 0
        parent_fingerprint = b"\x00\x00\x00\x00"
        child_number = 0

        components = [
            (path.purpose | 0x80000000, True),
            (path.coin_type | 0x80000000, True),
            (path.account | 0x80000000, True),
            (path.change, False),
            (path.address_index, False),
        ]

        for component, is_hardened in components:
            parent_fingerprint = SHA256Hasher.hash(
                PrivateKey.from_bytes(current_key).get_public_key().to_bytes()
            ).value[:4]

            current_key, current_chain_code = self._derive_child_key(
                current_key, current_chain_code, component
            )

            depth += 1
            child_number = component

        return ExtendedKey(
            key=current_key,
            chain_code=current_chain_code,
            depth=depth,
            parent_fingerprint=parent_fingerprint,
            child_number=child_number,
        )

    def derive_key_range(
        self, base_path: DerivationPath, start_index: int, count: int
    ) -> List[PrivateKey]:
        """Derive a range of keys from base path."""
        keys = []
        for i in range(count):
            path = DerivationPath(
                purpose=base_path.purpose,
                coin_type=base_path.coin_type,
                account=base_path.account,
                change=base_path.change,
                address_index=start_index + i,
            )
            keys.append(self.derive_key(path))
        return keys

    def derive_account_keys(
        self, coin_type: int, account: int, change: int, count: int
    ) -> List[PrivateKey]:
        """Derive keys for a specific account."""
        base_path = DerivationPath.bip44(coin_type, account, change, 0)
        return self.derive_key_range(base_path, 0, count)

    def get_public_key_derivation(self, path: DerivationPath) -> "PublicKeyDerivation":
        """Get public key derivation for this path."""
        return PublicKeyDerivation(self, path)


class PublicKeyDerivation:
    """Public key derivation from HD wallet."""

    def __init__(self, hd_derivation: HDKeyDerivation, base_path: DerivationPath):
        """Initialize public key derivation."""
        self.hd_derivation = hd_derivation
        self.base_path = base_path
        self.base_extended_key = hd_derivation.derive_extended_key(base_path)

    def derive_public_key(self, address_index: int) -> PublicKey:
        """Derive public key for address index."""
        path = DerivationPath(
            purpose=self.base_path.purpose,
            coin_type=self.base_path.coin_type,
            account=self.base_path.account,
            change=self.base_path.change,
            address_index=address_index,
        )
        return self.hd_derivation.derive_public_key(path)

    def derive_public_key_range(self, start_index: int, count: int) -> List[PublicKey]:
        """Derive range of public keys."""
        public_keys = []
        for i in range(count):
            public_keys.append(self.derive_public_key(start_index + i))
        return public_keys


class AdvancedKeyDerivation(HDKeyDerivation):
    """Advanced key derivation with additional security features."""

    def __init__(
        self,
        seed: bytes,
        network: str = "mainnet",
        additional_entropy: Optional[bytes] = None,
    ):
        """Initialize advanced key derivation."""
        if additional_entropy:
            # Mix additional entropy into seed
            seed = self._mix_entropy(seed, additional_entropy)

        super().__init__(seed, network)
        self.additional_entropy = additional_entropy

    def _mix_entropy(self, seed: bytes, additional_entropy: bytes) -> bytes:
        """Mix additional entropy into seed."""
        # Use HKDF to mix entropy
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=len(seed),
            salt=b"dubchain_entropy_mix",
            info=b"additional_entropy",
            backend=default_backend(),
        )

        mixed_entropy = hkdf.derive(seed + additional_entropy)
        return mixed_entropy

    def derive_key_with_salt(self, path: DerivationPath, salt: bytes) -> PrivateKey:
        """Derive key with additional salt."""
        # Create salted path
        salted_path = self._create_salted_path(path, salt)
        return self.derive_key(salted_path)

    def _create_salted_path(self, path: DerivationPath, salt: bytes) -> DerivationPath:
        """Create salted derivation path."""
        # Use salt to modify address index
        salt_hash = SHA256Hasher.hash(salt).value
        salt_offset = int.from_bytes(salt_hash[:4], "big") % 1000000

        return DerivationPath(
            purpose=path.purpose,
            coin_type=path.coin_type,
            account=path.account,
            change=path.change,
            address_index=(path.address_index + salt_offset) % 0x7FFFFFFF,
        )

    def derive_multi_purpose_key(
        self, path: DerivationPath, purpose: str
    ) -> PrivateKey:
        """Derive key for specific purpose."""
        purpose_hash = SHA256Hasher.hash(purpose.encode()).value
        purpose_offset = int.from_bytes(purpose_hash[:4], "big") % 1000

        purpose_path = DerivationPath(
            purpose=path.purpose,
            coin_type=path.coin_type,
            account=path.account,
            change=path.change,
            address_index=(path.address_index + purpose_offset) % 0x7FFFFFFF,
        )

        return self.derive_key(purpose_path)

    def create_key_derivation_chain(
        self, base_path: DerivationPath, chain_length: int
    ) -> List[PrivateKey]:
        """Create a chain of derived keys."""
        keys = []
        for i in range(chain_length):
            chain_path = DerivationPath(
                purpose=base_path.purpose,
                coin_type=base_path.coin_type,
                account=base_path.account,
                change=base_path.change,
                address_index=base_path.address_index + i,
            )
            keys.append(self.derive_key(chain_path))
        return keys

    def derive_encryption_key(
        self, path: DerivationPath, encryption_purpose: str
    ) -> bytes:
        """Derive encryption key for specific purpose."""
        base_key = self.derive_key(path)
        purpose_hash = SHA256Hasher.hash(encryption_purpose.encode()).value

        # Use HKDF to derive encryption key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=purpose_hash,
            info=b"encryption_key",
            backend=default_backend(),
        )

        return hkdf.derive(base_key.to_bytes())


class KeyDerivationFactory:
    """Factory for creating key derivation instances."""

    @staticmethod
    def create_derivation(
        derivation_type: DerivationType, seed: bytes, **kwargs
    ) -> KeyDerivation:
        """Create key derivation instance."""
        if derivation_type == DerivationType.BIP32:
            return HDKeyDerivation(seed, **kwargs)
        elif derivation_type == DerivationType.BIP44:
            return HDKeyDerivation(seed, **kwargs)
        elif derivation_type == DerivationType.BIP49:
            return HDKeyDerivation(seed, **kwargs)
        elif derivation_type == DerivationType.BIP84:
            return HDKeyDerivation(seed, **kwargs)
        elif derivation_type == DerivationType.CUSTOM:
            return AdvancedKeyDerivation(seed, **kwargs)
        else:
            raise ValueError(f"Unsupported derivation type: {derivation_type}")

    @staticmethod
    def create_from_mnemonic(
        mnemonic: str,
        passphrase: str = "",
        derivation_type: DerivationType = DerivationType.BIP44,
    ) -> KeyDerivation:
        """Create key derivation from mnemonic."""
        # This would integrate with the mnemonic module
        # For now, we'll create a simple seed
        seed = SHA256Hasher.hash((mnemonic + passphrase).encode()).value
        return KeyDerivationFactory.create_derivation(derivation_type, seed)

    @staticmethod
    def create_random(
        derivation_type: DerivationType = DerivationType.BIP44,
    ) -> KeyDerivation:
        """Create key derivation with random seed."""
        seed = secrets.token_bytes(64)
        return KeyDerivationFactory.create_derivation(derivation_type, seed)
