"""
Digital signature implementation using ECDSA with secp256k1 curve.

This module provides cryptographic signatures compatible with Bitcoin and Ethereum.
"""

import logging

logger = logging.getLogger(__name__)
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from .hashing import Hash, SHA256Hasher


@dataclass(frozen=True)
class PrivateKey:
    """Immutable private key with cryptographic operations."""

    _key: ec.EllipticCurvePrivateKey

    def __post_init__(self) -> None:
        # Ensure we're using secp256k1
        if not isinstance(self._key.curve, ec.SECP256K1):
            raise ValueError("Private key must use secp256k1 curve")

    @classmethod
    def generate(cls) -> "PrivateKey":
        """Generate a new random private key."""
        key = ec.generate_private_key(ec.SECP256K1())
        return cls(key)

    @classmethod
    def from_bytes(cls, key_bytes: bytes) -> "PrivateKey":
        """Create a private key from raw bytes."""
        if len(key_bytes) != 32:
            raise ValueError("Private key must be exactly 32 bytes")

        # Create private key from raw bytes
        key = ec.derive_private_key(
            int.from_bytes(key_bytes, byteorder="big"), ec.SECP256K1()
        )
        return cls(key)

    @classmethod
    def from_hex(cls, hex_string: str) -> "PrivateKey":
        """Create a private key from hexadecimal string."""
        return cls.from_bytes(bytes.fromhex(hex_string))

    def to_bytes(self) -> bytes:
        """Convert private key to raw bytes."""
        private_numbers = self._key.private_numbers()
        return private_numbers.private_value.to_bytes(32, byteorder="big")

    def to_hex(self) -> str:
        """Convert private key to hexadecimal string."""
        return self.to_bytes().hex()

    def add_scalar(self, scalar: Union[int, bytes]) -> "PrivateKey":
        """Add a scalar to the private key (mod n)."""
        # Get the current private key value
        current_value = int.from_bytes(self.to_bytes(), byteorder="big")

        # Convert scalar to int if it's bytes
        if isinstance(scalar, bytes):
            scalar = int.from_bytes(scalar, byteorder="big")

        # Add the scalar (mod n where n is the curve order)
        # For secp256k1, n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        new_value = (current_value + scalar) % n

        # Create new private key
        return self.from_bytes(new_value.to_bytes(32, byteorder="big"))

    def get_public_key(self) -> "PublicKey":
        """Get the corresponding public key."""
        public_key = self._key.public_key()
        return PublicKey(public_key)

    def sign(self, message: Union[bytes, str, Hash]) -> "Signature":
        """Sign a message with this private key."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        elif isinstance(message, Hash):
            message = message.value

        # Get DER-encoded signature
        der_signature = self._key.sign(message, ec.ECDSA(hashes.SHA256()))

        # Convert DER to raw signature (r, s)
        r, s = self._der_to_raw_signature(der_signature)

        return Signature(r, s, message)

    def _der_to_raw_signature(self, der_signature: bytes) -> Tuple[int, int]:
        """Convert DER-encoded signature to raw (r, s) values."""
        # Simple DER parsing for ECDSA signatures
        # Format: 0x30 [length] 0x02 [r_length] [r] 0x02 [s_length] [s]

        if len(der_signature) < 8:
            raise ValueError("Invalid DER signature")

        # Skip sequence header (0x30)
        if der_signature[0] != 0x30:
            raise ValueError("Invalid DER signature format")

        # Get total length
        total_length = der_signature[1]
        if total_length & 0x80:  # Long form
            length_bytes = total_length & 0x7F
            total_length = int.from_bytes(der_signature[2 : 2 + length_bytes], "big")
            offset = 2 + length_bytes
        else:
            offset = 2

        # Parse r component
        if der_signature[offset] != 0x02:
            raise ValueError("Invalid DER signature format")

        r_length = der_signature[offset + 1]
        if r_length & 0x80:  # Long form
            r_length_bytes = r_length & 0x7F
            r_length = int.from_bytes(
                der_signature[offset + 2 : offset + 2 + r_length_bytes], "big"
            )
            r_offset = offset + 2 + r_length_bytes
        else:
            r_offset = offset + 2

        r_bytes = der_signature[r_offset : r_offset + r_length]
        r = int.from_bytes(r_bytes, "big")

        # Parse s component
        s_offset = r_offset + r_length
        if der_signature[s_offset] != 0x02:
            raise ValueError("Invalid DER signature format")

        s_length = der_signature[s_offset + 1]
        if s_length & 0x80:  # Long form
            s_length_bytes = s_length & 0x7F
            s_length = int.from_bytes(
                der_signature[s_offset + 2 : s_offset + 2 + s_length_bytes], "big"
            )
            s_data_offset = s_offset + 2 + s_length_bytes
        else:
            s_data_offset = s_offset + 2

        s_bytes = der_signature[s_data_offset : s_data_offset + s_length]
        s = int.from_bytes(s_bytes, "big")

        return r, s

    def __str__(self) -> str:
        return f"PrivateKey('{self.to_hex()[:8]}...')"

    def __repr__(self) -> str:
        return f"PrivateKey.from_hex('{self.to_hex()}')"


@dataclass(frozen=True)
class PublicKey:
    """Immutable public key with cryptographic operations."""

    _key: ec.EllipticCurvePublicKey

    def __post_init__(self) -> None:
        # Ensure we're using secp256k1
        if not isinstance(self._key.curve, ec.SECP256K1):
            raise ValueError("Public key must use secp256k1 curve")

    @classmethod
    def from_bytes(cls, key_bytes: bytes) -> "PublicKey":
        """Create a public key from raw bytes (compressed or uncompressed)."""
        if len(key_bytes) == 33:
            # Compressed public key
            if key_bytes[0] not in (0x02, 0x03):
                raise ValueError("Invalid compressed public key format")
        elif len(key_bytes) == 65:
            # Uncompressed public key
            if key_bytes[0] != 0x04:
                raise ValueError("Invalid uncompressed public key format")
        else:
            raise ValueError(
                "Public key must be 33 (compressed) or 65 (uncompressed) bytes"
            )

        try:
            key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), key_bytes
            )
            return cls(key)
        except Exception as e:
            raise ValueError(f"Invalid public key: {e}")

    @classmethod
    def from_hex(cls, hex_string: str) -> "PublicKey":
        """Create a public key from hexadecimal string."""
        return cls.from_bytes(bytes.fromhex(hex_string))

    def to_bytes(self, compressed: bool = True) -> bytes:
        """Convert public key to raw bytes."""
        encoding = (
            PublicFormat.CompressedPoint
            if compressed
            else PublicFormat.UncompressedPoint
        )
        return self._key.public_bytes(Encoding.X962, encoding)

    def to_hex(self, compressed: bool = True) -> str:
        """Convert public key to hexadecimal string."""
        return self.to_bytes(compressed).hex()

    def to_address(self) -> str:
        """Convert public key to address (first 20 bytes of hash)."""
        # Hash the public key and take first 20 bytes
        pub_key_hash = SHA256Hasher.hash(self.to_bytes(compressed=True))
        address_hash = SHA256Hasher.hash(pub_key_hash.value)
        return address_hash.value[:20].hex()

    def verify(self, signature: "Signature", message: Union[bytes, str, Hash]) -> bool:
        """Verify a signature against a message."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        elif isinstance(message, Hash):
            message = message.value

        try:
            # Convert raw signature back to DER format for verification
            der_signature = signature.to_der()
            self._key.verify(der_signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False

    def __str__(self) -> str:
        return f"PublicKey('{self.to_hex()[:8]}...')"

    def __repr__(self) -> str:
        return f"PublicKey.from_hex('{self.to_hex()}')"


@dataclass(frozen=True)
class Signature:
    """Immutable digital signature."""

    r: int
    s: int
    message: bytes

    def __post_init__(self) -> None:
        # Validate signature components
        if self.r <= 0 or self.s <= 0:
            raise ValueError("Signature components must be positive")

        # secp256k1 order
        order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        if self.r >= order or self.s >= order:
            raise ValueError("Signature components must be less than curve order")

    @classmethod
    def from_bytes(cls, signature_bytes: bytes, message: bytes) -> "Signature":
        """Create a signature from raw bytes."""
        if len(signature_bytes) != 64:
            raise ValueError("Signature must be exactly 64 bytes")

        r = int.from_bytes(signature_bytes[:32], byteorder="big")
        s = int.from_bytes(signature_bytes[32:], byteorder="big")

        return cls(r, s, message)

    @classmethod
    def from_hex(cls, hex_string: str, message: bytes) -> "Signature":
        """Create a signature from hexadecimal string."""
        return cls.from_bytes(bytes.fromhex(hex_string), message)

    def to_bytes(self) -> bytes:
        """Convert signature to raw bytes."""
        return self.r.to_bytes(32, byteorder="big") + self.s.to_bytes(
            32, byteorder="big"
        )

    def to_hex(self) -> str:
        """Convert signature to hexadecimal string."""
        return self.to_bytes().hex()

    def to_der(self) -> bytes:
        """Convert signature to DER format."""
        # This is a simplified DER encoding
        # In practice, you'd use the cryptography library's DER encoding
        r_bytes = self.r.to_bytes(32, byteorder="big")
        s_bytes = self.s.to_bytes(32, byteorder="big")

        # Remove leading zeros
        r_bytes = r_bytes.lstrip(b"\x00")
        s_bytes = s_bytes.lstrip(b"\x00")

        # Add leading zero if high bit is set
        if r_bytes[0] & 0x80:
            r_bytes = b"\x00" + r_bytes
        if s_bytes[0] & 0x80:
            s_bytes = b"\x00" + s_bytes

        # DER encoding
        der = bytearray()
        der.append(0x30)  # SEQUENCE
        der.append(4 + len(r_bytes) + len(s_bytes))  # Length

        # r component
        der.append(0x02)  # INTEGER
        der.append(len(r_bytes))
        der.extend(r_bytes)

        # s component
        der.append(0x02)  # INTEGER
        der.append(len(s_bytes))
        der.extend(s_bytes)

        return bytes(der)

    def __str__(self) -> str:
        return f"Signature('{self.to_hex()[:16]}...')"

    def __repr__(self) -> str:
        return f"Signature.from_hex('{self.to_hex()}', {self.message!r})"


class ECDSASigner:
    """ECDSA signature operations with secp256k1 curve."""

    @staticmethod
    def generate_keypair() -> Tuple[PrivateKey, PublicKey]:
        """Generate a new ECDSA key pair."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        return private_key, public_key

    @staticmethod
    def sign_message(
        private_key: PrivateKey, message: Union[bytes, str, Hash]
    ) -> Signature:
        """Sign a message with a private key."""
        return private_key.sign(message)

    @staticmethod
    def verify_signature(
        public_key: PublicKey, signature: Signature, message: Union[bytes, str, Hash]
    ) -> bool:
        """Verify a signature against a message and public key."""
        return public_key.verify(signature, message)

    @staticmethod
    def recover_public_key(signature: Signature, message: bytes) -> Optional[PublicKey]:
        """
        Recover the public key from a signature and message.

        Note: This is a simplified implementation. In practice, you'd need
        to handle the recovery ID and implement proper ECDSA recovery.

        This method is intentionally not implemented as ECDSA public key recovery
        is complex and requires careful handling of recovery IDs and curve mathematics.
        For production use, consider using a cryptographic library like cryptography
        or secp256k1 that provides proper ECDSA recovery functionality.
        """
        # ECDSA public key recovery is not implemented in this educational/research
        # blockchain implementation. This is intentional to keep the codebase
        # focused on core blockchain concepts rather than complex cryptographic
        # implementations.
        return None

    @staticmethod
    def sign_transaction(private_key: PrivateKey, transaction_hash: Hash) -> Signature:
        """Sign a transaction hash."""
        return private_key.sign(transaction_hash)

    @staticmethod
    def verify_transaction_signature(
        public_key: PublicKey, signature: Signature, transaction_hash: Hash
    ) -> bool:
        """Verify a transaction signature."""
        return public_key.verify(signature, transaction_hash)
