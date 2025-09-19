"""
Comprehensive tests for key derivation module.

This module tests the advanced key derivation system including:
- BIP32/44/49/84 compliant HD key derivation
- Extended key management
- Advanced security features
- Custom derivation paths
"""

import secrets
from unittest.mock import MagicMock, Mock, patch

import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from dubchain.crypto.hashing import SHA256Hasher
from dubchain.crypto.signatures import PrivateKey, PublicKey
from dubchain.wallet.key_derivation import (
    AdvancedKeyDerivation,
    DerivationPath,
    DerivationType,
    ExtendedKey,
    HDKeyDerivation,
    KeyDerivation,
    KeyDerivationError,
    KeyDerivationFactory,
    PublicKeyDerivation,
)


class TestKeyDerivationError:
    """Test KeyDerivationError exception."""

    def test_key_derivation_error_creation(self):
        """Test creating KeyDerivationError."""
        error = KeyDerivationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestDerivationType:
    """Test DerivationType enum."""

    def test_derivation_type_values(self):
        """Test derivation type values."""
        assert DerivationType.BIP32.value == "bip32"
        assert DerivationType.BIP44.value == "bip44"
        assert DerivationType.BIP49.value == "bip49"
        assert DerivationType.BIP84.value == "bip84"
        assert DerivationType.CUSTOM.value == "custom"


class TestDerivationPath:
    """Test DerivationPath functionality."""

    def test_derivation_path_creation(self):
        """Test creating derivation path."""
        path = DerivationPath(
            purpose=44, coin_type=0, account=0, change=0, address_index=0
        )

        assert path.purpose == 44
        assert path.coin_type == 0
        assert path.account == 0
        assert path.change == 0
        assert path.address_index == 0

    def test_derivation_path_validation_purpose(self):
        """Test derivation path purpose validation."""
        # Valid purpose
        path = DerivationPath(44, 0, 0, 0, 0)
        assert path.purpose == 44

        # Invalid purpose (negative)
        with pytest.raises(
            ValueError, match="Purpose must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(-1, 0, 0, 0, 0)

        # Invalid purpose (too large)
        with pytest.raises(
            ValueError, match="Purpose must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(0x80000000, 0, 0, 0, 0)

    def test_derivation_path_validation_coin_type(self):
        """Test derivation path coin type validation."""
        # Valid coin type
        path = DerivationPath(44, 0, 0, 0, 0)
        assert path.coin_type == 0

        # Invalid coin type (negative)
        with pytest.raises(
            ValueError, match="Coin type must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(44, -1, 0, 0, 0)

        # Invalid coin type (too large)
        with pytest.raises(
            ValueError, match="Coin type must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(44, 0x80000000, 0, 0, 0)

    def test_derivation_path_validation_account(self):
        """Test derivation path account validation."""
        # Valid account
        path = DerivationPath(44, 0, 0, 0, 0)
        assert path.account == 0

        # Invalid account (negative)
        with pytest.raises(
            ValueError, match="Account must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(44, 0, -1, 0, 0)

        # Invalid account (too large)
        with pytest.raises(
            ValueError, match="Account must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(44, 0, 0x80000000, 0, 0)

    def test_derivation_path_validation_change(self):
        """Test derivation path change validation."""
        # Valid change (external)
        path = DerivationPath(44, 0, 0, 0, 0)
        assert path.change == 0

        # Valid change (internal)
        path = DerivationPath(44, 0, 0, 1, 0)
        assert path.change == 1

        # Invalid change
        with pytest.raises(
            ValueError, match="Change must be 0 \\(external\\) or 1 \\(internal\\)"
        ):
            DerivationPath(44, 0, 0, 2, 0)

    def test_derivation_path_validation_address_index(self):
        """Test derivation path address index validation."""
        # Valid address index
        path = DerivationPath(44, 0, 0, 0, 0)
        assert path.address_index == 0

        # Invalid address index (negative)
        with pytest.raises(
            ValueError, match="Address index must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(44, 0, 0, 0, -1)

        # Invalid address index (too large)
        with pytest.raises(
            ValueError, match="Address index must be between 0 and 0x7FFFFFFF"
        ):
            DerivationPath(44, 0, 0, 0, 0x80000000)

    def test_to_string(self):
        """Test converting derivation path to string."""
        path = DerivationPath(44, 0, 0, 0, 0)

        # Hardened format
        hardened_string = path.to_string(hardened=True)
        assert hardened_string == "m/44'/0'/0'/0/0"

        # Non-hardened format
        non_hardened_string = path.to_string(hardened=False)
        assert non_hardened_string == "m/44/0/0/0/0"

    def test_to_bytes(self):
        """Test converting derivation path to bytes."""
        path = DerivationPath(44, 0, 0, 0, 0)
        path_bytes = path.to_bytes()

        assert isinstance(path_bytes, bytes)
        assert b"44'" in path_bytes
        assert b"0'" in path_bytes
        assert b"0" in path_bytes

    def test_from_string(self):
        """Test creating derivation path from string."""
        # Valid path
        path = DerivationPath.from_string("m/44'/0'/0'/0/0")

        assert path.purpose == 44
        assert path.coin_type == 0
        assert path.account == 0
        assert path.change == 0
        assert path.address_index == 0

        # Invalid path (no m/ prefix)
        with pytest.raises(ValueError, match="Path must start with 'm/'"):
            DerivationPath.from_string("44'/0'/0'/0/0")

        # Invalid path (wrong number of components)
        with pytest.raises(ValueError, match="Path must have exactly 5 components"):
            DerivationPath.from_string("m/44'/0'/0'/0")

    def test_bip44(self):
        """Test BIP44 derivation path creation."""
        path = DerivationPath.bip44(0, 0, 0, 0)

        assert path.purpose == 44
        assert path.coin_type == 0
        assert path.account == 0
        assert path.change == 0
        assert path.address_index == 0

        # With custom values
        path = DerivationPath.bip44(1, 2, 1, 5)

        assert path.purpose == 44
        assert path.coin_type == 1
        assert path.account == 2
        assert path.change == 1
        assert path.address_index == 5

    def test_bip49(self):
        """Test BIP49 derivation path creation."""
        path = DerivationPath.bip49(0, 0, 0, 0)

        assert path.purpose == 49
        assert path.coin_type == 0
        assert path.account == 0
        assert path.change == 0
        assert path.address_index == 0

    def test_bip84(self):
        """Test BIP84 derivation path creation."""
        path = DerivationPath.bip84(0, 0, 0, 0)

        assert path.purpose == 84
        assert path.coin_type == 0
        assert path.account == 0
        assert path.change == 0
        assert path.address_index == 0

    def test_str_repr(self):
        """Test string and representation methods."""
        path = DerivationPath(44, 0, 0, 0, 0)

        # String representation
        assert str(path) == "m/44'/0'/0'/0/0"

        # Detailed representation
        repr_str = repr(path)
        assert "DerivationPath" in repr_str
        assert "purpose=44" in repr_str
        assert "coin_type=0" in repr_str


class TestExtendedKey:
    """Test ExtendedKey functionality."""

    def test_extended_key_creation(self):
        """Test creating extended key."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        extended_key = ExtendedKey(
            key=key,
            chain_code=chain_code,
            depth=0,
            parent_fingerprint=parent_fingerprint,
            child_number=0,
        )

        assert extended_key.key == key
        assert extended_key.chain_code == chain_code
        assert extended_key.depth == 0
        assert extended_key.parent_fingerprint == parent_fingerprint
        assert extended_key.child_number == 0

    def test_extended_key_validation_key(self):
        """Test extended key validation for key."""
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        # Valid key
        key = secrets.token_bytes(32)
        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        assert extended_key.key == key

        # Invalid key (wrong length)
        with pytest.raises(ValueError, match="Key must be 32 bytes"):
            ExtendedKey(secrets.token_bytes(16), chain_code, 0, parent_fingerprint, 0)

    def test_extended_key_validation_chain_code(self):
        """Test extended key validation for chain code."""
        key = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        # Valid chain code
        chain_code = secrets.token_bytes(32)
        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        assert extended_key.chain_code == chain_code

        # Invalid chain code (wrong length)
        with pytest.raises(ValueError, match="Chain code must be 32 bytes"):
            ExtendedKey(key, secrets.token_bytes(16), 0, parent_fingerprint, 0)

    def test_extended_key_validation_parent_fingerprint(self):
        """Test extended key validation for parent fingerprint."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)

        # Valid parent fingerprint
        parent_fingerprint = secrets.token_bytes(4)
        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        assert extended_key.parent_fingerprint == parent_fingerprint

        # Invalid parent fingerprint (wrong length)
        with pytest.raises(ValueError, match="Parent fingerprint must be 4 bytes"):
            ExtendedKey(key, chain_code, 0, secrets.token_bytes(8), 0)

    def test_extended_key_validation_depth(self):
        """Test extended key validation for depth."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        # Valid depth
        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        assert extended_key.depth == 0

        # Invalid depth (negative)
        with pytest.raises(ValueError, match="Depth must be between 0 and 255"):
            ExtendedKey(key, chain_code, -1, parent_fingerprint, 0)

        # Invalid depth (too large)
        with pytest.raises(ValueError, match="Depth must be between 0 and 255"):
            ExtendedKey(key, chain_code, 256, parent_fingerprint, 0)

    def test_extended_key_validation_child_number(self):
        """Test extended key validation for child number."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        # Valid child number
        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        assert extended_key.child_number == 0

        # Invalid child number (negative)
        with pytest.raises(
            ValueError, match="Child number must be between 0 and 0x7FFFFFFF"
        ):
            ExtendedKey(key, chain_code, 0, parent_fingerprint, -1)

        # Invalid child number (too large)
        with pytest.raises(
            ValueError, match="Child number must be between 0 and 0x7FFFFFFF"
        ):
            ExtendedKey(key, chain_code, 0, parent_fingerprint, 0x80000000)

    def test_fingerprint(self):
        """Test getting key fingerprint."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        fingerprint = extended_key.fingerprint()

        assert isinstance(fingerprint, bytes)
        assert len(fingerprint) == 4

    def test_to_private_key(self):
        """Test converting to private key."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        private_key = extended_key.to_private_key()

        assert isinstance(private_key, PrivateKey)

    def test_to_public_key(self):
        """Test converting to public key."""
        key = secrets.token_bytes(32)
        chain_code = secrets.token_bytes(32)
        parent_fingerprint = secrets.token_bytes(4)

        extended_key = ExtendedKey(key, chain_code, 0, parent_fingerprint, 0)
        public_key = extended_key.to_public_key()

        assert isinstance(public_key, PublicKey)


class TestKeyDerivation:
    """Test base KeyDerivation class."""

    def test_key_derivation_creation(self):
        """Test creating key derivation."""
        seed = secrets.token_bytes(32)
        derivation = KeyDerivation(seed)

        assert derivation.seed == seed

    def test_key_derivation_short_seed(self):
        """Test creating key derivation with short seed."""
        seed = secrets.token_bytes(8)  # Too short

        with pytest.raises(ValueError, match="Seed must be at least 16 bytes"):
            KeyDerivation(seed)

    def test_derive_key_not_implemented(self):
        """Test that derive_key raises NotImplementedError."""
        seed = secrets.token_bytes(32)
        derivation = KeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)

        with pytest.raises(NotImplementedError):
            derivation.derive_key(path)

    def test_derive_public_key(self):
        """Test deriving public key."""
        seed = secrets.token_bytes(32)
        derivation = KeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)

        # Mock derive_key to return a private key
        mock_private_key = Mock(spec=PrivateKey)
        mock_public_key = Mock(spec=PublicKey)
        mock_private_key.get_public_key.return_value = mock_public_key

        with patch.object(derivation, "derive_key", return_value=mock_private_key):
            public_key = derivation.derive_public_key(path)
            assert public_key == mock_public_key
            mock_private_key.get_public_key.assert_called_once()


class TestHDKeyDerivation:
    """Test HDKeyDerivation functionality."""

    def test_hd_key_derivation_creation(self):
        """Test creating HD key derivation."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)

        assert derivation.seed == seed
        assert derivation.network == "mainnet"
        assert len(derivation.master_key) == 32
        assert len(derivation.master_chain_code) == 32

    def test_hd_key_derivation_creation_with_network(self):
        """Test creating HD key derivation with custom network."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed, network="testnet")

        assert derivation.seed == seed
        assert derivation.network == "testnet"

    def test_derive_master_key(self):
        """Test deriving master key."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)

        master_key, master_chain_code = derivation._derive_master_key()

        assert len(master_key) == 32
        assert len(master_chain_code) == 32
        assert master_key == derivation.master_key
        assert master_chain_code == derivation.master_chain_code

    def test_derive_child_key_hardened(self):
        """Test deriving hardened child key."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)

        parent_key = secrets.token_bytes(32)
        parent_chain_code = secrets.token_bytes(32)
        child_number = 0x80000000  # Hardened

        child_key, child_chain_code = derivation._derive_child_key(
            parent_key, parent_chain_code, child_number
        )

        assert len(child_key) == 32
        assert len(child_chain_code) == 32
        assert child_key != parent_key
        assert child_chain_code != parent_chain_code

    def test_derive_child_key_non_hardened(self):
        """Test deriving non-hardened child key."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)

        parent_key = secrets.token_bytes(32)
        parent_chain_code = secrets.token_bytes(32)
        child_number = 0  # Non-hardened

        child_key, child_chain_code = derivation._derive_child_key(
            parent_key, parent_chain_code, child_number
        )

        assert len(child_key) == 32
        assert len(child_chain_code) == 32
        assert child_key != parent_key
        assert child_chain_code != parent_chain_code

    def test_derive_key(self):
        """Test deriving key from path."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)

        private_key = derivation.derive_key(path)

        assert isinstance(private_key, PrivateKey)

    def test_derive_extended_key(self):
        """Test deriving extended key from path."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)

        extended_key = derivation.derive_extended_key(path)

        assert isinstance(extended_key, ExtendedKey)
        assert extended_key.depth == 5  # 5 levels in path
        assert len(extended_key.key) == 32
        assert len(extended_key.chain_code) == 32

    def test_derive_key_range(self):
        """Test deriving range of keys."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)
        base_path = DerivationPath(44, 0, 0, 0, 0)

        keys = derivation.derive_key_range(base_path, 0, 5)

        assert len(keys) == 5
        for key in keys:
            assert isinstance(key, PrivateKey)

        # Keys should be different
        key_bytes = [key.to_bytes() for key in keys]
        assert len(set(key_bytes)) == 5  # All unique

    def test_derive_account_keys(self):
        """Test deriving keys for specific account."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)

        keys = derivation.derive_account_keys(0, 0, 0, 3)

        assert len(keys) == 3
        for key in keys:
            assert isinstance(key, PrivateKey)

    def test_get_public_key_derivation(self):
        """Test getting public key derivation."""
        seed = secrets.token_bytes(64)
        derivation = HDKeyDerivation(seed)
        base_path = DerivationPath(44, 0, 0, 0, 0)

        public_derivation = derivation.get_public_key_derivation(base_path)

        assert isinstance(public_derivation, PublicKeyDerivation)
        assert public_derivation.hd_derivation == derivation
        assert public_derivation.base_path == base_path


class TestPublicKeyDerivation:
    """Test PublicKeyDerivation functionality."""

    def test_public_key_derivation_creation(self):
        """Test creating public key derivation."""
        seed = secrets.token_bytes(64)
        hd_derivation = HDKeyDerivation(seed)
        base_path = DerivationPath(44, 0, 0, 0, 0)

        public_derivation = PublicKeyDerivation(hd_derivation, base_path)

        assert public_derivation.hd_derivation == hd_derivation
        assert public_derivation.base_path == base_path
        assert isinstance(public_derivation.base_extended_key, ExtendedKey)

    def test_derive_public_key(self):
        """Test deriving public key for address index."""
        seed = secrets.token_bytes(64)
        hd_derivation = HDKeyDerivation(seed)
        base_path = DerivationPath(44, 0, 0, 0, 0)

        public_derivation = PublicKeyDerivation(hd_derivation, base_path)
        public_key = public_derivation.derive_public_key(0)

        assert isinstance(public_key, PublicKey)

    def test_derive_public_key_range(self):
        """Test deriving range of public keys."""
        seed = secrets.token_bytes(64)
        hd_derivation = HDKeyDerivation(seed)
        base_path = DerivationPath(44, 0, 0, 0, 0)

        public_derivation = PublicKeyDerivation(hd_derivation, base_path)
        public_keys = public_derivation.derive_public_key_range(0, 3)

        assert len(public_keys) == 3
        for public_key in public_keys:
            assert isinstance(public_key, PublicKey)


class TestAdvancedKeyDerivation:
    """Test AdvancedKeyDerivation functionality."""

    def test_advanced_key_derivation_creation(self):
        """Test creating advanced key derivation."""
        seed = secrets.token_bytes(64)
        derivation = AdvancedKeyDerivation(seed)

        assert derivation.seed == seed
        assert derivation.network == "mainnet"
        assert derivation.additional_entropy is None

    def test_advanced_key_derivation_with_entropy(self):
        """Test creating advanced key derivation with additional entropy."""
        seed = secrets.token_bytes(64)
        additional_entropy = secrets.token_bytes(32)

        derivation = AdvancedKeyDerivation(seed, additional_entropy=additional_entropy)

        assert derivation.seed != seed  # Should be mixed
        assert derivation.additional_entropy == additional_entropy

    def test_mix_entropy(self):
        """Test mixing additional entropy."""
        seed = secrets.token_bytes(64)
        additional_entropy = secrets.token_bytes(32)

        derivation = AdvancedKeyDerivation(seed)
        mixed_entropy = derivation._mix_entropy(seed, additional_entropy)

        assert len(mixed_entropy) == len(seed)
        assert mixed_entropy != seed
        assert mixed_entropy != additional_entropy

    def test_derive_key_with_salt(self):
        """Test deriving key with additional salt."""
        seed = secrets.token_bytes(64)
        derivation = AdvancedKeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)
        salt = secrets.token_bytes(16)

        private_key = derivation.derive_key_with_salt(path, salt)

        assert isinstance(private_key, PrivateKey)

        # Should be different from normal derivation
        normal_key = derivation.derive_key(path)
        assert private_key.to_bytes() != normal_key.to_bytes()

    def test_create_salted_path(self):
        """Test creating salted derivation path."""
        seed = secrets.token_bytes(64)
        derivation = AdvancedKeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)
        salt = secrets.token_bytes(16)

        salted_path = derivation._create_salted_path(path, salt)

        assert salted_path.purpose == path.purpose
        assert salted_path.coin_type == path.coin_type
        assert salted_path.account == path.account
        assert salted_path.change == path.change
        assert salted_path.address_index != path.address_index  # Should be modified

    def test_derive_multi_purpose_key(self):
        """Test deriving key for specific purpose."""
        seed = secrets.token_bytes(64)
        derivation = AdvancedKeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)

        private_key = derivation.derive_multi_purpose_key(path, "encryption")

        assert isinstance(private_key, PrivateKey)

        # Should be different from normal derivation
        normal_key = derivation.derive_key(path)
        assert private_key.to_bytes() != normal_key.to_bytes()

    def test_create_key_derivation_chain(self):
        """Test creating chain of derived keys."""
        seed = secrets.token_bytes(64)
        derivation = AdvancedKeyDerivation(seed)
        base_path = DerivationPath(44, 0, 0, 0, 0)

        keys = derivation.create_key_derivation_chain(base_path, 5)

        assert len(keys) == 5
        for key in keys:
            assert isinstance(key, PrivateKey)

        # Keys should be different
        key_bytes = [key.to_bytes() for key in keys]
        assert len(set(key_bytes)) == 5  # All unique

    def test_derive_encryption_key(self):
        """Test deriving encryption key for specific purpose."""
        seed = secrets.token_bytes(64)
        derivation = AdvancedKeyDerivation(seed)
        path = DerivationPath(44, 0, 0, 0, 0)

        encryption_key = derivation.derive_encryption_key(path, "wallet_encryption")

        assert isinstance(encryption_key, bytes)
        assert len(encryption_key) == 32


class TestKeyDerivationFactory:
    """Test KeyDerivationFactory functionality."""

    def test_create_derivation_bip32(self):
        """Test creating BIP32 derivation."""
        seed = secrets.token_bytes(64)

        derivation = KeyDerivationFactory.create_derivation(DerivationType.BIP32, seed)

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_derivation_bip44(self):
        """Test creating BIP44 derivation."""
        seed = secrets.token_bytes(64)

        derivation = KeyDerivationFactory.create_derivation(DerivationType.BIP44, seed)

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_derivation_bip49(self):
        """Test creating BIP49 derivation."""
        seed = secrets.token_bytes(64)

        derivation = KeyDerivationFactory.create_derivation(DerivationType.BIP49, seed)

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_derivation_bip84(self):
        """Test creating BIP84 derivation."""
        seed = secrets.token_bytes(64)

        derivation = KeyDerivationFactory.create_derivation(DerivationType.BIP84, seed)

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_derivation_custom(self):
        """Test creating custom derivation."""
        seed = secrets.token_bytes(64)

        derivation = KeyDerivationFactory.create_derivation(DerivationType.CUSTOM, seed)

        assert isinstance(derivation, AdvancedKeyDerivation)

    def test_create_derivation_unsupported_type(self):
        """Test creating derivation with unsupported type."""
        seed = secrets.token_bytes(64)

        with pytest.raises(ValueError, match="Unsupported derivation type"):
            KeyDerivationFactory.create_derivation("unsupported", seed)

    def test_create_derivation_with_kwargs(self):
        """Test creating derivation with additional kwargs."""
        seed = secrets.token_bytes(64)

        derivation = KeyDerivationFactory.create_derivation(
            DerivationType.BIP44, seed, network="testnet"
        )

        assert isinstance(derivation, HDKeyDerivation)
        assert derivation.network == "testnet"

    def test_create_from_mnemonic(self):
        """Test creating derivation from mnemonic."""
        mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"

        derivation = KeyDerivationFactory.create_from_mnemonic(mnemonic)

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_from_mnemonic_with_passphrase(self):
        """Test creating derivation from mnemonic with passphrase."""
        mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        passphrase = "test_passphrase"

        derivation = KeyDerivationFactory.create_from_mnemonic(mnemonic, passphrase)

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_from_mnemonic_with_derivation_type(self):
        """Test creating derivation from mnemonic with specific type."""
        mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"

        derivation = KeyDerivationFactory.create_from_mnemonic(
            mnemonic, derivation_type=DerivationType.BIP84
        )

        assert isinstance(derivation, HDKeyDerivation)

    def test_create_random(self):
        """Test creating derivation with random seed."""
        derivation = KeyDerivationFactory.create_random()

        assert isinstance(derivation, HDKeyDerivation)
        assert len(derivation.seed) == 64

    def test_create_random_with_type(self):
        """Test creating random derivation with specific type."""
        derivation = KeyDerivationFactory.create_random(DerivationType.CUSTOM)

        assert isinstance(derivation, AdvancedKeyDerivation)
        assert len(derivation.seed) == 64
