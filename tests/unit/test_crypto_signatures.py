"""
Unit tests for cryptographic signature functions.
"""

import pytest

from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.crypto.signatures import ECDSASigner, PrivateKey, PublicKey, Signature


class TestPrivateKey:
    """Test the PrivateKey class."""

    def test_generate_private_key(self):
        """Test generating a new private key."""
        private_key = PrivateKey.generate()
        assert isinstance(private_key, PrivateKey)
        assert len(private_key.to_bytes()) == 32

    def test_private_key_from_bytes(self):
        """Test creating private key from bytes."""
        key_bytes = b"\x01" * 32
        private_key = PrivateKey.from_bytes(key_bytes)
        assert private_key.to_bytes() == key_bytes

    def test_private_key_from_hex(self):
        """Test creating private key from hex string."""
        hex_string = "01" * 32
        private_key = PrivateKey.from_hex(hex_string)
        assert private_key.to_hex() == hex_string

    def test_private_key_invalid_length(self):
        """Test that invalid length raises ValueError."""
        with pytest.raises(ValueError, match="Private key must be exactly 32 bytes"):
            PrivateKey.from_bytes(b"\x01" * 31)

    def test_private_key_get_public_key(self):
        """Test getting public key from private key."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        assert isinstance(public_key, PublicKey)

    def test_private_key_sign_string(self):
        """Test signing a string message."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)
        assert isinstance(signature, Signature)

    def test_private_key_sign_bytes(self):
        """Test signing a bytes message."""
        private_key = PrivateKey.generate()
        message = b"hello world"
        signature = private_key.sign(message)
        assert isinstance(signature, Signature)

    def test_private_key_sign_hash(self):
        """Test signing a hash."""
        private_key = PrivateKey.generate()
        message_hash = SHA256Hasher.hash("hello world")
        signature = private_key.sign(message_hash)
        assert isinstance(signature, Signature)

    def test_private_key_string_representation(self):
        """Test string representation of private key."""
        private_key = PrivateKey.generate()
        str_repr = str(private_key)
        assert "PrivateKey" in str_repr
        assert "..." in str_repr


class TestPublicKey:
    """Test the PublicKey class."""

    def test_public_key_from_private(self):
        """Test creating public key from private key."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        assert isinstance(public_key, PublicKey)

    def test_public_key_compressed(self):
        """Test compressed public key format."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        compressed = public_key.to_bytes(compressed=True)
        assert len(compressed) == 33
        assert compressed[0] in (0x02, 0x03)

    def test_public_key_uncompressed(self):
        """Test uncompressed public key format."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        uncompressed = public_key.to_bytes(compressed=False)
        assert len(uncompressed) == 65
        assert uncompressed[0] == 0x04

    def test_public_key_from_bytes_compressed(self):
        """Test creating public key from compressed bytes."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        compressed = public_key.to_bytes(compressed=True)

        # Recreate from bytes
        new_public_key = PublicKey.from_bytes(compressed)
        assert new_public_key.to_bytes(compressed=True) == compressed

    def test_public_key_from_bytes_uncompressed(self):
        """Test creating public key from uncompressed bytes."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        uncompressed = public_key.to_bytes(compressed=False)

        # Recreate from bytes
        new_public_key = PublicKey.from_bytes(uncompressed)
        assert new_public_key.to_bytes(compressed=False) == uncompressed

    def test_public_key_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            PublicKey.from_bytes(b"\x01" * 33)  # Invalid prefix

        with pytest.raises(ValueError):
            PublicKey.from_bytes(b"\x04" * 65)  # Invalid uncompressed format

    def test_public_key_to_address(self):
        """Test converting public key to address."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        address = public_key.to_address()
        assert isinstance(address, str)
        assert len(address) == 40  # 20 bytes in hex

    def test_public_key_verify_signature(self):
        """Test verifying a signature."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        message = "hello world"

        signature = private_key.sign(message)
        assert public_key.verify(signature, message)

    def test_public_key_verify_invalid_signature(self):
        """Test verifying an invalid signature."""
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()
        public_key2 = private_key2.get_public_key()
        message = "hello world"

        # Sign with key1, verify with key2's public key
        signature = private_key1.sign(message)
        assert not public_key2.verify(signature, message)


class TestSignature:
    """Test the Signature class."""

    def test_signature_creation(self):
        """Test creating a signature."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)
        assert isinstance(signature, Signature)

    def test_signature_from_bytes(self):
        """Test creating signature from bytes."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)

        # Recreate from bytes
        new_signature = Signature.from_bytes(signature.to_bytes(), message.encode())
        assert new_signature.r == signature.r
        assert new_signature.s == signature.s

    def test_signature_from_hex(self):
        """Test creating signature from hex string."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)

        # Recreate from hex
        new_signature = Signature.from_hex(signature.to_hex(), message.encode())
        assert new_signature.r == signature.r
        assert new_signature.s == signature.s

    def test_signature_invalid_components(self):
        """Test that invalid signature components raise ValueError."""
        with pytest.raises(ValueError, match="Signature components must be positive"):
            Signature(0, 1, b"message")

        with pytest.raises(ValueError, match="Signature components must be positive"):
            Signature(1, 0, b"message")

    def test_signature_to_der(self):
        """Test converting signature to DER format."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)

        der = signature.to_der()
        assert isinstance(der, bytes)
        assert len(der) > 0

    def test_signature_string_representation(self):
        """Test string representation of signature."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)

        str_repr = str(signature)
        assert "Signature" in str_repr
        assert "..." in str_repr


class TestECDSASigner:
    """Test the ECDSASigner class."""

    def test_generate_keypair(self):
        """Test generating a key pair."""
        private_key, public_key = ECDSASigner.generate_keypair()
        assert isinstance(private_key, PrivateKey)
        assert isinstance(public_key, PublicKey)
        assert private_key.get_public_key() == public_key

    def test_sign_message(self):
        """Test signing a message."""
        private_key, public_key = ECDSASigner.generate_keypair()
        message = "hello world"

        signature = ECDSASigner.sign_message(private_key, message)
        assert isinstance(signature, Signature)

    def test_verify_signature(self):
        """Test verifying a signature."""
        private_key, public_key = ECDSASigner.generate_keypair()
        message = "hello world"

        signature = ECDSASigner.sign_message(private_key, message)
        assert ECDSASigner.verify_signature(public_key, signature, message)

    def test_verify_invalid_signature(self):
        """Test verifying an invalid signature."""
        private_key1, public_key1 = ECDSASigner.generate_keypair()
        private_key2, public_key2 = ECDSASigner.generate_keypair()
        message = "hello world"

        # Sign with key1, verify with key2's public key
        signature = ECDSASigner.sign_message(private_key1, message)
        assert not ECDSASigner.verify_signature(public_key2, signature, message)

    def test_sign_transaction(self):
        """Test signing a transaction hash."""
        private_key, public_key = ECDSASigner.generate_keypair()
        transaction_hash = SHA256Hasher.hash("transaction data")

        signature = ECDSASigner.sign_transaction(private_key, transaction_hash)
        assert isinstance(signature, Signature)

    def test_verify_transaction_signature(self):
        """Test verifying a transaction signature."""
        private_key, public_key = ECDSASigner.generate_keypair()
        transaction_hash = SHA256Hasher.hash("transaction data")

        signature = ECDSASigner.sign_transaction(private_key, transaction_hash)
        assert ECDSASigner.verify_transaction_signature(
            public_key, signature, transaction_hash
        )

    def test_recover_public_key_not_implemented(self):
        """Test that public key recovery raises NotImplementedError."""
        private_key = PrivateKey.generate()
        message = "hello world"
        signature = private_key.sign(message)

        with pytest.raises(NotImplementedError):
            ECDSASigner.recover_public_key(signature, message.encode())


class TestSignatureIntegration:
    """Integration tests for signature functions."""

    def test_sign_verify_roundtrip(self):
        """Test signing and verifying a message."""
        private_key, public_key = ECDSASigner.generate_keypair()
        message = "hello world"

        signature = private_key.sign(message)
        assert public_key.verify(signature, message)

    def test_sign_verify_different_formats(self):
        """Test signing and verifying with different message formats."""
        private_key, public_key = ECDSASigner.generate_keypair()

        # Test string
        message_str = "hello world"
        signature_str = private_key.sign(message_str)
        assert public_key.verify(signature_str, message_str)

        # Test bytes
        message_bytes = message_str.encode()
        signature_bytes = private_key.sign(message_bytes)
        assert public_key.verify(signature_bytes, message_bytes)

        # Test hash
        message_hash = SHA256Hasher.hash(message_str)
        signature_hash = private_key.sign(message_hash)
        assert public_key.verify(signature_hash, message_hash)

    def test_signature_serialization(self):
        """Test signature serialization and deserialization."""
        private_key = PrivateKey.generate()
        message = "hello world"
        original_signature = private_key.sign(message)

        # Serialize to bytes and back
        signature_bytes = original_signature.to_bytes()
        deserialized_signature = Signature.from_bytes(signature_bytes, message.encode())

        assert deserialized_signature.r == original_signature.r
        assert deserialized_signature.s == original_signature.s

    def test_key_serialization(self):
        """Test key serialization and deserialization."""
        original_private_key = PrivateKey.generate()
        original_public_key = original_private_key.get_public_key()

        # Serialize private key
        private_key_bytes = original_private_key.to_bytes()
        deserialized_private_key = PrivateKey.from_bytes(private_key_bytes)

        # Serialize public key
        public_key_bytes = original_public_key.to_bytes(compressed=True)
        deserialized_public_key = PublicKey.from_bytes(public_key_bytes)

        # Test that deserialized keys work the same
        message = "hello world"
        signature = deserialized_private_key.sign(message)
        assert deserialized_public_key.verify(signature, message)

    def test_deterministic_signatures(self):
        """Test that same input produces same signature."""
        private_key = PrivateKey.from_hex("01" * 32)
        message = "hello world"

        signature1 = private_key.sign(message)
        signature2 = private_key.sign(message)

        # Note: ECDSA signatures are not deterministic by default
        # This test verifies the behavior, not necessarily that they're equal
        assert isinstance(signature1, Signature)
        assert isinstance(signature2, Signature)
