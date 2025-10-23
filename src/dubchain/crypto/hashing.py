"""
Hash functions and utilities for GodChain.

Implements SHA-256 hashing with additional utilities for blockchain operations.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes


@dataclass(frozen=True)
class Hash:
    """Immutable hash value with comparison and string representation."""

    value: bytes

    def __post_init__(self) -> None:
        if len(self.value) != 32:
            raise ValueError("Hash must be exactly 32 bytes")

    def __str__(self) -> str:
        return self.value.hex()

    def __repr__(self) -> str:
        return f"Hash('{self.value.hex()}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hash):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __lt__(self, other: "Hash") -> bool:
        return self.value < other.value

    def __le__(self, other: "Hash") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "Hash") -> bool:
        return self.value > other.value

    def __ge__(self, other: "Hash") -> bool:
        return self.value >= other.value

    @classmethod
    def from_hex(cls, hex_string: str) -> "Hash":
        """Create a Hash from a hexadecimal string."""
        return cls(bytes.fromhex(hex_string))

    @classmethod
    def from_int(cls, value: int) -> "Hash":
        """Create a Hash from an integer (big-endian)."""
        return cls(value.to_bytes(32, byteorder="big"))

    @classmethod
    def zero(cls) -> "Hash":
        """Create a zero hash (all zeros)."""
        return cls(b"\x00" * 32)

    @classmethod
    def max_value(cls) -> "Hash":
        """Create a maximum hash (all 0xFF bytes)."""
        return cls(b"\xff" * 32)

    def to_hex(self) -> str:
        """Convert hash to hexadecimal string."""
        return self.value.hex()

    def to_int(self) -> int:
        """Convert hash to integer (big-endian)."""
        return int.from_bytes(self.value, byteorder="big")


class SHA256Hasher:
    """SHA-256 hasher with blockchain-specific utilities."""

    @staticmethod
    def hash(data: Union[bytes, str]) -> Hash:
        """
        Hash data using SHA-256.

        Args:
            data: Data to hash (bytes or string)

        Returns:
            Hash object containing the SHA-256 hash
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        digest = hashlib.sha256(data).digest()
        return Hash(digest)

    @staticmethod
    def double_hash(data: Union[bytes, str]) -> Hash:
        """
        Double SHA-256 hash (Bitcoin-style).

        Args:
            data: Data to hash (bytes or string)

        Returns:
            Hash object containing the double SHA-256 hash
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        first_hash = hashlib.sha256(data).digest()
        second_hash = hashlib.sha256(first_hash).digest()
        return Hash(second_hash)

    @staticmethod
    def hash_list(items: List[Union[bytes, str]]) -> Hash:
        """
        Hash a list of items by concatenating them.

        Args:
            items: List of items to hash

        Returns:
            Hash of the concatenated items
        """
        combined = b""
        for item in items:
            if isinstance(item, str):
                combined += item.encode("utf-8")
            else:
                combined += item

        return SHA256Hasher.hash(combined)

    @staticmethod
    def hmac_sha256(key: Union[bytes, str], data: Union[bytes, str]) -> Hash:
        """
        HMAC-SHA256 for keyed hashing.

        Args:
            key: HMAC key
            data: Data to hash

        Returns:
            Hash object containing the HMAC-SHA256 hash
        """
        if isinstance(key, str):
            key = key.encode("utf-8")
        if isinstance(data, str):
            data = data.encode("utf-8")

        digest = hashlib.sha256(key + data).digest()
        return Hash(digest)

    @staticmethod
    def pbkdf2_hmac(
        password: Union[bytes, str],
        salt: Union[bytes, str],
        iterations: int = 100000,
        key_length: int = 32,
    ) -> bytes:
        """
        PBKDF2-HMAC-SHA256 for key derivation.

        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            iterations: Number of iterations
            key_length: Length of derived key

        Returns:
            Derived key bytes
        """
        if isinstance(password, str):
            password = password.encode("utf-8")
        if isinstance(salt, str):
            salt = salt.encode("utf-8")

        return hashlib.pbkdf2_hmac("sha256", password, salt, iterations, key_length)

    @staticmethod
    def verify_proof_of_work(hash_value: Hash, difficulty: int) -> bool:
        """
        Verify that a hash meets the proof of work difficulty requirement.

        Args:
            hash_value: Hash to verify
            difficulty: Required number of leading zeros (in bits)

        Returns:
            True if the hash meets the difficulty requirement
        """
        # Convert difficulty to number of leading zero bytes
        zero_bytes = difficulty // 8
        zero_bits = difficulty % 8

        # Check leading zero bytes
        for i in range(zero_bytes):
            if hash_value.value[i] != 0:
                return False

        # Check partial byte if needed
        if zero_bits > 0 and zero_bytes < len(hash_value.value):
            # For partial byte, we need to check that the high bits are zero
            # If we need 28 bits, that's 3 full bytes + 4 bits in the 4th byte
            # The 4th byte should be <= 0x0F (binary: 00001111)
            max_value = (1 << (8 - zero_bits)) - 1
            if hash_value.value[zero_bytes] > max_value:
                return False

        return True

    @staticmethod
    def calculate_difficulty_target(difficulty: int) -> Hash:
        """
        Calculate the target hash for a given difficulty.

        Args:
            difficulty: Difficulty in bits (number of leading zeros)

        Returns:
            Target hash value
        """
        if difficulty <= 0:
            return Hash.max_value()
        if difficulty >= 256:
            return Hash.zero()

        # Create target with leading zeros
        target_bytes = bytearray(32)

        zero_bytes = difficulty // 8
        zero_bits = difficulty % 8

        # Set leading zero bytes
        for i in range(zero_bytes):
            target_bytes[i] = 0

        # Set partial byte - this should be the maximum value that still meets the difficulty
        if zero_bits > 0 and zero_bytes < 32:
            max_value = (1 << (8 - zero_bits)) - 1
            target_bytes[zero_bytes] = max_value

        # Fill remaining bytes with max value
        for i in range(zero_bytes + 1, 32):
            target_bytes[i] = 0xFF

        return Hash(bytes(target_bytes))
