"""
Unit tests for cryptographic hashing functions.
"""

import pytest

from dubchain.crypto.hashing import Hash, SHA256Hasher


class TestHash:
    """Test the Hash class."""

    def test_hash_creation(self):
        """Test creating a hash from bytes."""
        data = b"\x00" * 32
        hash_obj = Hash(data)
        assert hash_obj.value == data

    def test_hash_invalid_length(self):
        """Test that invalid length raises ValueError."""
        with pytest.raises(ValueError, match="Hash must be exactly 32 bytes"):
            Hash(b"\x00" * 31)

        with pytest.raises(ValueError, match="Hash must be exactly 32 bytes"):
            Hash(b"\x00" * 33)

    def test_hash_from_hex(self):
        """Test creating a hash from hex string."""
        hex_string = "00" * 32
        hash_obj = Hash.from_hex(hex_string)
        assert hash_obj.value == b"\x00" * 32

    def test_hash_zero(self):
        """Test creating a zero hash."""
        zero_hash = Hash.zero()
        assert zero_hash.value == b"\x00" * 32

    def test_hash_max_value(self):
        """Test creating a max value hash."""
        max_hash = Hash.max_value()
        assert max_hash.value == b"\xff" * 32

    def test_hash_string_representation(self):
        """Test string representation of hash."""
        hash_obj = Hash.from_hex("00" * 32)
        assert str(hash_obj) == "00" * 32
        assert "Hash(" in repr(hash_obj)
        assert "000000" in repr(hash_obj)

    def test_hash_equality(self):
        """Test hash equality."""
        hash1 = Hash.from_hex("00" * 32)
        hash2 = Hash.from_hex("00" * 32)
        hash3 = Hash.from_hex("01" + "00" * 31)

        assert hash1 == hash2
        assert hash1 != hash3
        assert hash1 != "not a hash"

    def test_hash_ordering(self):
        """Test hash ordering."""
        hash1 = Hash.from_hex("00" * 32)
        hash2 = Hash.from_hex("01" + "00" * 31)

        assert hash1 < hash2
        assert hash2 > hash1
        assert hash1 <= hash2
        assert hash2 >= hash1

    def test_hash_to_int(self):
        """Test converting hash to integer."""
        hash_obj = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001"
        )
        assert hash_obj.to_int() == 1

    def test_hash_hashable(self):
        """Test that hash is hashable."""
        hash_obj = Hash.from_hex("00" * 32)
        hash_set = {hash_obj}
        assert hash_obj in hash_set


class TestSHA256Hasher:
    """Test the SHA256Hasher class."""

    def test_hash_bytes(self):
        """Test hashing bytes."""
        data = b"hello world"
        hash_obj = SHA256Hasher.hash(data)
        assert isinstance(hash_obj, Hash)
        assert len(hash_obj.value) == 32

    def test_hash_string(self):
        """Test hashing string."""
        data = "hello world"
        hash_obj = SHA256Hasher.hash(data)
        assert isinstance(hash_obj, Hash)
        assert len(hash_obj.value) == 32

    def test_hash_consistency(self):
        """Test that same input produces same hash."""
        data = "hello world"
        hash1 = SHA256Hasher.hash(data)
        hash2 = SHA256Hasher.hash(data)
        assert hash1 == hash2

    def test_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = SHA256Hasher.hash("hello")
        hash2 = SHA256Hasher.hash("world")
        assert hash1 != hash2

    def test_double_hash(self):
        """Test double hashing."""
        data = "hello world"
        single_hash = SHA256Hasher.hash(data)
        double_hash = SHA256Hasher.double_hash(data)
        assert single_hash != double_hash

    def test_hash_list(self):
        """Test hashing a list of items."""
        items = ["hello", "world", b"bytes"]
        hash_obj = SHA256Hasher.hash_list(items)
        assert isinstance(hash_obj, Hash)

    def test_hmac_sha256(self):
        """Test HMAC-SHA256."""
        key = "secret"
        data = "message"
        hmac_hash = SHA256Hasher.hmac_sha256(key, data)
        assert isinstance(hmac_hash, Hash)

    def test_pbkdf2_hmac(self):
        """Test PBKDF2-HMAC-SHA256."""
        password = "password"
        salt = "salt"
        key = SHA256Hasher.pbkdf2_hmac(password, salt, 1000, 32)
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_verify_proof_of_work_zero_difficulty(self):
        """Test proof of work verification with zero difficulty."""
        hash_obj = Hash.from_hex("ff" * 32)
        assert SHA256Hasher.verify_proof_of_work(hash_obj, 0)

    def test_verify_proof_of_work_high_difficulty(self):
        """Test proof of work verification with high difficulty."""
        hash_obj = Hash.from_hex("ff" * 32)
        assert not SHA256Hasher.verify_proof_of_work(hash_obj, 1)

    def test_verify_proof_of_work_valid(self):
        """Test proof of work verification with valid hash."""
        hash_obj = Hash.from_hex("00" * 4 + "ff" * 28)
        assert SHA256Hasher.verify_proof_of_work(hash_obj, 32)

    def test_verify_proof_of_work_partial_byte(self):
        """Test proof of work verification with partial byte."""
        hash_obj = Hash.from_hex("00" * 3 + "0f" + "ff" * 28)
        assert SHA256Hasher.verify_proof_of_work(hash_obj, 28)
        assert not SHA256Hasher.verify_proof_of_work(hash_obj, 29)

    def test_calculate_difficulty_target_zero(self):
        """Test calculating difficulty target for zero difficulty."""
        target = SHA256Hasher.calculate_difficulty_target(0)
        assert target == Hash.max_value()

    def test_calculate_difficulty_target_max(self):
        """Test calculating difficulty target for max difficulty."""
        target = SHA256Hasher.calculate_difficulty_target(256)
        assert target == Hash.zero()

    def test_calculate_difficulty_target_partial_byte(self):
        """Test calculating difficulty target with partial byte."""
        target = SHA256Hasher.calculate_difficulty_target(28)
        assert target.value[:3] == b"\x00" * 3
        assert target.value[3] == 0x0F
        assert target.value[4:] == b"\xff" * 28


class TestHashIntegration:
    """Integration tests for hashing functions."""

    def test_hash_chain(self):
        """Test chaining multiple hash operations."""
        data = "initial data"
        hash1 = SHA256Hasher.hash(data)
        hash2 = SHA256Hasher.hash(hash1.value)
        hash3 = SHA256Hasher.double_hash(hash2.value)

        assert isinstance(hash1, Hash)
        assert isinstance(hash2, Hash)
        assert isinstance(hash3, Hash)
        assert hash1 != hash2 != hash3

    def test_hash_large_data(self):
        """Test hashing large data."""
        large_data = b"x" * 1000000  # 1MB
        hash_obj = SHA256Hasher.hash(large_data)
        assert isinstance(hash_obj, Hash)
        assert len(hash_obj.value) == 32

    def test_hash_empty_data(self):
        """Test hashing empty data."""
        hash_obj = SHA256Hasher.hash(b"")
        assert isinstance(hash_obj, Hash)
        assert len(hash_obj.value) == 32

    def test_hash_unicode_data(self):
        """Test hashing unicode data."""
        unicode_data = "Hello 世界 🌍"
        hash_obj = SHA256Hasher.hash(unicode_data)
        assert isinstance(hash_obj, Hash)
        assert len(hash_obj.value) == 32
