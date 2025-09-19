"""
Advanced wallet encryption implementation for GodChain.

This module provides sophisticated encryption and security features for wallet
data protection, including AES encryption, key derivation, and secure storage.
"""

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from ..crypto.hashing import Hash, SHA256Hasher


class EncryptionError(Exception):
    """Exception raised for encryption-related errors."""

    pass


class KeyDerivationFunction(Enum):
    """Key derivation functions for encryption."""

    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"  # Would require argon2-cffi


class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""

    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"


@dataclass
class EncryptionConfig:
    """Configuration for wallet encryption."""

    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2
    iterations: int = 100000
    salt_length: int = 32
    key_length: int = 32
    iv_length: int = 12  # For GCM
    tag_length: int = 16  # For GCM
    memory_cost: int = 1048576  # For Scrypt
    parallelism: int = 1  # For Scrypt
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.iterations <= 0:
            raise ValueError("Iterations must be positive")

        if self.salt_length <= 0:
            raise ValueError("Salt length must be positive")

        if self.key_length <= 0:
            raise ValueError("Key length must be positive")

        if self.iv_length <= 0:
            raise ValueError("IV length must be positive")

        if self.algorithm == EncryptionAlgorithm.AES_256_GCM and self.iv_length != 12:
            raise ValueError("GCM requires 12-byte IV")

        if self.algorithm == EncryptionAlgorithm.AES_256_CBC and self.iv_length != 16:
            raise ValueError("CBC requires 16-byte IV")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "kdf": self.kdf.value,
            "iterations": self.iterations,
            "salt_length": self.salt_length,
            "key_length": self.key_length,
            "iv_length": self.iv_length,
            "tag_length": self.tag_length,
            "memory_cost": self.memory_cost,
            "parallelism": self.parallelism,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptionConfig":
        """Create config from dictionary."""
        return cls(
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            kdf=KeyDerivationFunction(data["kdf"]),
            iterations=data.get("iterations", 100000),
            salt_length=data.get("salt_length", 32),
            key_length=data.get("key_length", 32),
            iv_length=data.get("iv_length", 12),
            tag_length=data.get("tag_length", 16),
            memory_cost=data.get("memory_cost", 1048576),
            parallelism=data.get("parallelism", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EncryptedData:
    """Represents encrypted data with metadata."""

    ciphertext: bytes
    salt: bytes
    iv: bytes
    tag: Optional[bytes] = None
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2
    iterations: int = 100000
    created_at: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ciphertext": self.ciphertext.hex(),
            "salt": self.salt.hex(),
            "iv": self.iv.hex(),
            "tag": self.tag.hex() if self.tag else None,
            "algorithm": self.algorithm.value,
            "kdf": self.kdf.value,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedData":
        """Create from dictionary."""
        return cls(
            ciphertext=bytes.fromhex(data["ciphertext"]),
            salt=bytes.fromhex(data["salt"]),
            iv=bytes.fromhex(data["iv"]),
            tag=bytes.fromhex(data["tag"]) if data.get("tag") else None,
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            kdf=KeyDerivationFunction(data["kdf"]),
            iterations=data.get("iterations", 100000),
            created_at=data.get("created_at", int(time.time())),
            metadata=data.get("metadata", {}),
        )


class WalletEncryption:
    """Advanced wallet encryption implementation."""

    def __init__(self, config: Optional[EncryptionConfig] = None):
        """Initialize encryption with configuration."""
        self.config = config or EncryptionConfig()

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password and salt."""
        password_bytes = password.encode("utf-8")

        if self.config.kdf == KeyDerivationFunction.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                iterations=self.config.iterations,
                backend=default_backend(),
            )
            return kdf.derive(password_bytes)

        elif self.config.kdf == KeyDerivationFunction.SCRYPT:
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                n=self.config.iterations,
                r=8,
                p=self.config.parallelism,
                backend=default_backend(),
            )
            return kdf.derive(password_bytes)

        else:
            raise EncryptionError(f"Unsupported KDF: {self.config.kdf}")

    def _generate_salt(self) -> bytes:
        """Generate random salt."""
        return secrets.token_bytes(self.config.salt_length)

    def _generate_iv(self) -> bytes:
        """Generate random IV."""
        return secrets.token_bytes(self.config.iv_length)

    def encrypt(self, data: bytes, password: str) -> EncryptedData:
        """Encrypt data with password."""
        if not data:
            raise EncryptionError("Cannot encrypt empty data")

        if not password:
            raise EncryptionError("Password cannot be empty")

        # Generate salt and IV
        salt = self._generate_salt()
        iv = self._generate_iv()

        # Derive key
        key = self._derive_key(password, salt)

        # Encrypt data
        if self.config.algorithm == EncryptionAlgorithm.AES_256_GCM:
            cipher = Cipher(
                algorithms.AES(key), modes.GCM(iv), backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            tag = encryptor.tag

            return EncryptedData(
                ciphertext=ciphertext,
                salt=salt,
                iv=iv,
                tag=tag,
                algorithm=self.config.algorithm,
                kdf=self.config.kdf,
                iterations=self.config.iterations,
            )

        elif self.config.algorithm == EncryptionAlgorithm.AES_256_CBC:
            cipher = Cipher(
                algorithms.AES(key), modes.CBC(iv), backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()

            return EncryptedData(
                ciphertext=ciphertext,
                salt=salt,
                iv=iv,
                algorithm=self.config.algorithm,
                kdf=self.config.kdf,
                iterations=self.config.iterations,
            )

        else:
            raise EncryptionError(f"Unsupported algorithm: {self.config.algorithm}")

    def decrypt(self, encrypted_data: EncryptedData, password: str) -> bytes:
        """Decrypt data with password."""
        if not password:
            raise EncryptionError("Password cannot be empty")

        # Derive key
        key = self._derive_key(password, encrypted_data.salt)

        # Decrypt data
        try:
            if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
                if not encrypted_data.tag:
                    raise EncryptionError("GCM requires authentication tag")

                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(encrypted_data.iv, encrypted_data.tag),
                    backend=default_backend(),
                )
                decryptor = cipher.decryptor()
                plaintext = (
                    decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
                )

                return plaintext

            elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_CBC:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CBC(encrypted_data.iv),
                    backend=default_backend(),
                )
                decryptor = cipher.decryptor()
                plaintext = (
                    decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
                )

                return plaintext

            else:
                raise EncryptionError(
                    f"Unsupported algorithm: {encrypted_data.algorithm}"
                )

        except InvalidTag:
            raise EncryptionError("Invalid password or corrupted data")
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {str(e)}")

    def encrypt_string(self, text: str, password: str) -> EncryptedData:
        """Encrypt string data."""
        return self.encrypt(text.encode("utf-8"), password)

    def decrypt_string(self, encrypted_data: EncryptedData, password: str) -> str:
        """Decrypt string data."""
        return self.decrypt(encrypted_data, password).decode("utf-8")

    def encrypt_dict(self, data: Dict[str, Any], password: str) -> EncryptedData:
        """Encrypt dictionary data."""
        import json

        json_data = json.dumps(data, sort_keys=True)
        return self.encrypt(json_data.encode("utf-8"), password)

    def decrypt_dict(
        self, encrypted_data: EncryptedData, password: str
    ) -> Dict[str, Any]:
        """Decrypt dictionary data."""
        import json

        json_data = self.decrypt(encrypted_data, password).decode("utf-8")
        return json.loads(json_data)

    def verify_password(self, encrypted_data: EncryptedData, password: str) -> bool:
        """Verify password without decrypting data."""
        try:
            self.decrypt(encrypted_data, password)
            return True
        except EncryptionError:
            return False

    def change_password(
        self, encrypted_data: EncryptedData, old_password: str, new_password: str
    ) -> EncryptedData:
        """Change password for encrypted data."""
        # Decrypt with old password
        plaintext = self.decrypt(encrypted_data, old_password)

        # Encrypt with new password
        return self.encrypt(plaintext, new_password)

    def get_encryption_info(self, encrypted_data: EncryptedData) -> Dict[str, Any]:
        """Get information about encrypted data."""
        return {
            "algorithm": encrypted_data.algorithm.value,
            "kdf": encrypted_data.kdf.value,
            "iterations": encrypted_data.iterations,
            "salt_length": len(encrypted_data.salt),
            "iv_length": len(encrypted_data.iv),
            "ciphertext_length": len(encrypted_data.ciphertext),
            "has_tag": encrypted_data.tag is not None,
            "created_at": encrypted_data.created_at,
            "metadata": encrypted_data.metadata,
        }


class SecureStorage:
    """Secure storage implementation for wallet data."""

    def __init__(self, encryption: WalletEncryption):
        """Initialize secure storage."""
        self.encryption = encryption
        self.storage_path: Optional[str] = None

    def set_storage_path(self, path: str) -> None:
        """Set storage path for wallet files."""
        self.storage_path = path
        os.makedirs(path, exist_ok=True)

    def save_wallet(
        self, wallet_data: Dict[str, Any], password: str, filename: str
    ) -> str:
        """Save encrypted wallet to file."""
        if not self.storage_path:
            raise EncryptionError("Storage path not set")

        # Encrypt wallet data
        encrypted_data = self.encryption.encrypt_dict(wallet_data, password)

        # Save metadata (includes ciphertext)
        metadata_path = os.path.join(self.storage_path, f"{filename}.meta")
        with open(metadata_path, "w") as f:
            import json

            json.dump(encrypted_data.to_dict(), f, indent=2)

        return metadata_path

    def load_wallet(self, filename: str, password: str) -> Dict[str, Any]:
        """Load encrypted wallet from file."""
        if not self.storage_path:
            raise EncryptionError("Storage path not set")

        # Load metadata
        metadata_path = os.path.join(self.storage_path, f"{filename}.meta")
        with open(metadata_path, "r") as f:
            import json

            metadata = json.load(f)

        # Reconstruct encrypted data
        encrypted_data = EncryptedData.from_dict(metadata)

        # Decrypt and return
        return self.encryption.decrypt_dict(encrypted_data, password)

    def list_wallets(self) -> List[str]:
        """List available wallet files."""
        if not self.storage_path:
            return []

        wallets = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".meta"):
                wallet_name = filename[:-5]  # Remove .meta suffix
                wallets.append(wallet_name)

        return wallets

    def delete_wallet(self, filename: str) -> bool:
        """Delete wallet files."""
        if not self.storage_path:
            return False

        try:
            # Delete metadata file
            metadata_path = os.path.join(self.storage_path, f"{filename}.meta")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            return True
        except Exception:
            return False

    def wallet_exists(self, filename: str) -> bool:
        """Check if wallet exists."""
        if not self.storage_path:
            return False

        metadata_path = os.path.join(self.storage_path, f"{filename}.meta")

        return os.path.exists(metadata_path)


class PasswordManager:
    """Password management and validation."""

    def __init__(self, min_length: int = 8, require_special: bool = True):
        """Initialize password manager."""
        self.min_length = min_length
        self.require_special = require_special

    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        errors = []

        if len(password) < self.min_length:
            errors.append(
                f"Password must be at least {self.min_length} characters long"
            )

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")

        if self.require_special and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

    def generate_strong_password(self, length: int = 16) -> str:
        """Generate a strong password."""
        import random
        import string

        if length < 8:
            length = 8

        # Character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"

        # Ensure at least one character from each set
        password = [
            random.choice(lowercase),
            random.choice(uppercase),
            random.choice(digits),
            random.choice(special),
        ]

        # Fill remaining length
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(random.choice(all_chars))

        # Shuffle password
        random.shuffle(password)

        return "".join(password)

    def calculate_entropy(self, password: str) -> float:
        """Calculate password entropy."""
        import math

        # Character set size estimation
        charset_size = 0
        if any(c.islower() for c in password):
            charset_size += 26
        if any(c.isupper() for c in password):
            charset_size += 26
        if any(c.isdigit() for c in password):
            charset_size += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            charset_size += 32

        if charset_size == 0:
            return 0

        return len(password) * math.log2(charset_size)

    def get_password_strength(self, password: str) -> str:
        """Get password strength rating."""
        entropy = self.calculate_entropy(password)

        if entropy < 30:
            return "Very Weak"
        elif entropy < 40:
            return "Weak"
        elif entropy < 50:
            return "Fair"
        elif entropy < 60:
            return "Good"
        elif entropy < 70:
            return "Strong"
        else:
            return "Very Strong"
