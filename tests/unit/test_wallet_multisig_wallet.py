"""
Unit tests for multi-signature wallet functionality.
"""

import time
from typing import List

import pytest

from dubchain.crypto.signatures import PrivateKey, PublicKey, Signature
from dubchain.wallet import (
    MultisigConfig,
    MultisigParticipant,
    MultisigSignature,
    MultisigTransaction,
    MultisigType,
    MultisigWallet,
    SignatureStatus,
    WalletError,
)


class TestMultisigParticipant:
    """Test MultisigParticipant functionality."""

    def test_multisig_participant_creation(self):
        """Test multisig participant creation."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        participant = MultisigParticipant(
            participant_id="participant_1", public_key=public_key, weight=2
        )

        assert participant.participant_id == "participant_1"
        assert participant.public_key == public_key
        assert participant.weight == 2
        assert participant.is_active is True
        assert participant.added_at > 0
        assert participant.last_used is None

    def test_multisig_participant_validation(self):
        """Test multisig participant validation."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Test empty participant ID
        with pytest.raises(ValueError, match="Participant ID cannot be empty"):
            MultisigParticipant(participant_id="", public_key=public_key)

        # Test negative weight
        with pytest.raises(ValueError, match="Participant weight must be positive"):
            MultisigParticipant(
                participant_id="participant_1", public_key=public_key, weight=0
            )

    def test_multisig_participant_usage_update(self):
        """Test multisig participant usage update."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        participant = MultisigParticipant(
            participant_id="participant_1", public_key=public_key
        )

        assert participant.last_used is None

        participant.update_usage()
        assert participant.last_used is not None
        assert participant.last_used > 0

    def test_multisig_participant_serialization(self):
        """Test multisig participant serialization."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        participant = MultisigParticipant(
            participant_id="participant_1",
            public_key=public_key,
            weight=3,
            is_active=False,
            metadata={"role": "admin"},
        )

        # Test to_dict
        participant_dict = participant.to_dict()
        assert participant_dict["participant_id"] == "participant_1"
        assert participant_dict["weight"] == 3
        assert participant_dict["is_active"] is False
        assert participant_dict["metadata"]["role"] == "admin"

        # Test from_dict
        restored_participant = MultisigParticipant.from_dict(participant_dict)
        assert restored_participant.participant_id == participant.participant_id
        assert restored_participant.weight == participant.weight
        assert restored_participant.is_active == participant.is_active
        assert restored_participant.metadata == participant.metadata


class TestMultisigSignature:
    """Test MultisigSignature functionality."""

    def test_multisig_signature_creation(self):
        """Test multisig signature creation."""
        private_key = PrivateKey.generate()
        signature = private_key.sign(b"test data")

        multisig_sig = MultisigSignature(
            participant_id="participant_1", signature=signature
        )

        assert multisig_sig.participant_id == "participant_1"
        assert multisig_sig.signature == signature
        assert multisig_sig.status == SignatureStatus.PENDING
        assert multisig_sig.timestamp > 0

    def test_multisig_signature_serialization(self):
        """Test multisig signature serialization."""
        private_key = PrivateKey.generate()
        signature = private_key.sign(b"test data")

        multisig_sig = MultisigSignature(
            participant_id="participant_1",
            signature=signature,
            status=SignatureStatus.SIGNED,
            metadata={"verified": True},
        )

        # Test to_dict
        sig_dict = multisig_sig.to_dict()
        assert sig_dict["participant_id"] == "participant_1"
        assert sig_dict["status"] == "signed"
        assert sig_dict["metadata"]["verified"] is True

        # Test from_dict
        restored_sig = MultisigSignature.from_dict(sig_dict)
        assert restored_sig.participant_id == multisig_sig.participant_id
        assert restored_sig.status == multisig_sig.status
        assert restored_sig.metadata == multisig_sig.metadata


class TestMultisigTransaction:
    """Test MultisigTransaction functionality."""

    def test_multisig_transaction_creation(self):
        """Test multisig transaction creation."""
        transaction_data = b"test transaction data"

        transaction = MultisigTransaction(
            transaction_id="tx_123",
            transaction_data=transaction_data,
            required_signatures=2,
            total_participants=3,
        )

        assert transaction.transaction_id == "tx_123"
        assert transaction.transaction_data == transaction_data
        assert transaction.required_signatures == 2
        assert transaction.total_participants == 3
        assert len(transaction.signatures) == 0
        assert transaction.status == "pending"
        assert transaction.created_at > 0

    def test_multisig_transaction_validation(self):
        """Test multisig transaction validation."""
        transaction_data = b"test transaction data"

        # Test zero required signatures
        with pytest.raises(ValueError, match="Required signatures must be positive"):
            MultisigTransaction(
                transaction_id="tx_123",
                transaction_data=transaction_data,
                required_signatures=0,
                total_participants=3,
            )

        # Test zero total participants
        with pytest.raises(ValueError, match="Total participants must be positive"):
            MultisigTransaction(
                transaction_id="tx_123",
                transaction_data=transaction_data,
                required_signatures=2,
                total_participants=0,
            )

        # Test required signatures > total participants
        with pytest.raises(
            ValueError, match="Required signatures cannot exceed total participants"
        ):
            MultisigTransaction(
                transaction_id="tx_123",
                transaction_data=transaction_data,
                required_signatures=5,
                total_participants=3,
            )

    def test_multisig_transaction_signature_management(self):
        """Test multisig transaction signature management."""
        transaction_data = b"test transaction data"

        transaction = MultisigTransaction(
            transaction_id="tx_123",
            transaction_data=transaction_data,
            required_signatures=2,
            total_participants=3,
        )

        # Create signatures
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()

        sig1 = MultisigSignature(
            participant_id="participant_1",
            signature=private_key1.sign(transaction_data),
            status=SignatureStatus.SIGNED,
        )

        sig2 = MultisigSignature(
            participant_id="participant_2",
            signature=private_key2.sign(transaction_data),
            status=SignatureStatus.SIGNED,
        )

        # Add signatures
        assert transaction.add_signature(sig1) is True
        assert transaction.add_signature(sig2) is True

        # Test duplicate signature
        assert transaction.add_signature(sig1) is False

        # Test completion
        assert transaction.is_complete() is True
        assert transaction.get_signature_count() == 2

        # Test participant signatures
        participant_sigs = transaction.get_participant_signatures()
        assert "participant_1" in participant_sigs
        assert "participant_2" in participant_sigs

    def test_multisig_transaction_expiration(self):
        """Test multisig transaction expiration."""
        transaction_data = b"test transaction data"

        transaction = MultisigTransaction(
            transaction_id="tx_123",
            transaction_data=transaction_data,
            required_signatures=1,
            total_participants=2,
            expires_at=int(time.time()) - 1,  # Expired
        )

        private_key = PrivateKey.generate()
        sig = MultisigSignature(
            participant_id="participant_1", signature=private_key.sign(transaction_data)
        )

        # Should fail due to expiration
        assert transaction.add_signature(sig) is False
        assert sig.status == SignatureStatus.EXPIRED

    def test_multisig_transaction_serialization(self):
        """Test multisig transaction serialization."""
        transaction_data = b"test transaction data"

        transaction = MultisigTransaction(
            transaction_id="tx_123",
            transaction_data=transaction_data,
            required_signatures=2,
            total_participants=3,
            expires_at=int(time.time()) + 3600,
            metadata={"priority": "high"},
        )

        # Add a signature
        private_key = PrivateKey.generate()
        sig = MultisigSignature(
            participant_id="participant_1",
            signature=private_key.sign(transaction_data),
            status=SignatureStatus.SIGNED,
        )
        transaction.add_signature(sig)

        # Test to_dict
        tx_dict = transaction.to_dict()
        assert tx_dict["transaction_id"] == "tx_123"
        assert tx_dict["required_signatures"] == 2
        assert tx_dict["total_participants"] == 3
        assert len(tx_dict["signatures"]) == 1
        assert tx_dict["metadata"]["priority"] == "high"

        # Test from_dict
        restored_tx = MultisigTransaction.from_dict(tx_dict)
        assert restored_tx.transaction_id == transaction.transaction_id
        assert restored_tx.required_signatures == transaction.required_signatures
        assert restored_tx.total_participants == transaction.total_participants
        assert len(restored_tx.signatures) == 1


class TestMultisigConfig:
    """Test MultisigConfig functionality."""

    def test_multisig_config_creation(self):
        """Test multisig config creation."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
            timeout_seconds=3600,
        )

        assert config.multisig_type == MultisigType.M_OF_N
        assert config.required_signatures == 2
        assert config.total_participants == 3
        assert config.timeout_seconds == 3600
        assert config.allow_duplicate_signatures is False
        assert config.require_all_participants is False

    def test_multisig_config_validation(self):
        """Test multisig config validation."""
        # Test zero required signatures
        with pytest.raises(ValueError, match="Required signatures must be positive"):
            MultisigConfig(
                multisig_type=MultisigType.M_OF_N,
                required_signatures=0,
                total_participants=3,
            )

        # Test zero total participants
        with pytest.raises(ValueError, match="Total participants must be positive"):
            MultisigConfig(
                multisig_type=MultisigType.M_OF_N,
                required_signatures=2,
                total_participants=0,
            )

        # Test required signatures > total participants
        with pytest.raises(
            ValueError, match="Required signatures cannot exceed total participants"
        ):
            MultisigConfig(
                multisig_type=MultisigType.M_OF_N,
                required_signatures=5,
                total_participants=3,
            )

        # Test negative timeout
        with pytest.raises(ValueError, match="Timeout must be positive"):
            MultisigConfig(
                multisig_type=MultisigType.M_OF_N,
                required_signatures=2,
                total_participants=3,
                timeout_seconds=-1,
            )

    def test_multisig_config_serialization(self):
        """Test multisig config serialization."""
        config = MultisigConfig(
            multisig_type=MultisigType.THRESHOLD,
            required_signatures=3,
            total_participants=5,
            timeout_seconds=7200,
            allow_duplicate_signatures=True,
            require_all_participants=True,
            metadata={"security_level": "high"},
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["multisig_type"] == "threshold"
        assert config_dict["required_signatures"] == 3
        assert config_dict["total_participants"] == 5
        assert config_dict["timeout_seconds"] == 7200
        assert config_dict["allow_duplicate_signatures"] is True
        assert config_dict["require_all_participants"] is True
        assert config_dict["metadata"]["security_level"] == "high"

        # Test from_dict
        restored_config = MultisigConfig.from_dict(config_dict)
        assert restored_config.multisig_type == config.multisig_type
        assert restored_config.required_signatures == config.required_signatures
        assert restored_config.total_participants == config.total_participants
        assert restored_config.timeout_seconds == config.timeout_seconds
        assert (
            restored_config.allow_duplicate_signatures
            == config.allow_duplicate_signatures
        )
        assert (
            restored_config.require_all_participants == config.require_all_participants
        )
        assert restored_config.metadata == config.metadata


class TestMultisigWallet:
    """Test MultisigWallet functionality."""

    def test_multisig_wallet_creation(self):
        """Test multisig wallet creation."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        assert wallet.wallet_id == "multisig_1"
        assert wallet.name == "Test Multisig Wallet"
        assert wallet.config == config
        assert len(wallet.participants) == 0
        assert len(wallet.transactions) == 0
        assert wallet.created_at > 0
        assert wallet.last_accessed > 0

    def test_multisig_wallet_participant_management(self):
        """Test multisig wallet participant management."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Add participants
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()
        private_key3 = PrivateKey.generate()

        participant1 = wallet.add_participant(
            "participant_1", private_key1.get_public_key(), weight=1
        )

        participant2 = wallet.add_participant(
            "participant_2", private_key2.get_public_key(), weight=2
        )

        participant3 = wallet.add_participant(
            "participant_3", private_key3.get_public_key(), weight=1
        )

        assert len(wallet.participants) == 3
        assert participant1.participant_id == "participant_1"
        assert participant2.weight == 2

        # Test get participant
        retrieved = wallet.get_participant("participant_1")
        assert retrieved == participant1

        # Test get active participants
        active = wallet.get_active_participants()
        assert len(active) == 3

        # Test update weight
        wallet.update_participant_weight("participant_1", 3)
        assert wallet.get_participant("participant_1").weight == 3

        # Test remove participant
        wallet.remove_participant("participant_3")
        assert len(wallet.participants) == 2
        assert "participant_3" not in wallet.participants

    def test_multisig_wallet_participant_errors(self):
        """Test multisig wallet participant error cases."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        private_key = PrivateKey.generate()

        # Test duplicate participant
        wallet.add_participant("participant_1", private_key.get_public_key())
        with pytest.raises(
            WalletError, match="Participant participant_1 already exists"
        ):
            wallet.add_participant("participant_1", private_key.get_public_key())

        # Test max participants
        wallet.add_participant("participant_2", private_key.get_public_key())
        wallet.add_participant("participant_3", private_key.get_public_key())

        with pytest.raises(WalletError, match="Maximum number of participants reached"):
            wallet.add_participant("participant_4", private_key.get_public_key())

        # Test remove participant not found
        with pytest.raises(WalletError, match="Participant participant_4 not found"):
            wallet.remove_participant("participant_4")

        # Test remove participant would violate requirements
        # First remove one participant, then try to remove another
        wallet.remove_participant(
            "participant_3"
        )  # This should work (2 participants left)

        # Now try to remove another participant - this should fail
        with pytest.raises(
            WalletError,
            match="Cannot remove participant: would violate signature requirements",
        ):
            wallet.remove_participant("participant_1")

    def test_multisig_wallet_transaction_management(self):
        """Test multisig wallet transaction management."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Add participants
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()
        private_key3 = PrivateKey.generate()

        wallet.add_participant("participant_1", private_key1.get_public_key())
        wallet.add_participant("participant_2", private_key2.get_public_key())
        wallet.add_participant("participant_3", private_key3.get_public_key())

        # Create transaction
        transaction_data = b"test transaction data"
        transaction = wallet.create_transaction(
            transaction_data, expires_in_seconds=3600
        )

        assert transaction.transaction_id in wallet.transactions
        assert transaction.required_signatures == 2
        assert transaction.total_participants == 3
        assert transaction.expires_at is not None

        # Sign transaction
        assert (
            wallet.sign_transaction(
                transaction.transaction_id, "participant_1", private_key1
            )
            is True
        )
        assert (
            wallet.sign_transaction(
                transaction.transaction_id, "participant_2", private_key2
            )
            is True
        )

        # Test duplicate signature
        assert (
            wallet.sign_transaction(
                transaction.transaction_id, "participant_1", private_key1
            )
            is False
        )

        # Verify transaction
        assert wallet.verify_transaction(transaction.transaction_id) is True

        # Test get transaction
        retrieved_tx = wallet.get_transaction(transaction.transaction_id)
        assert retrieved_tx == transaction

        # Test get pending transactions
        pending = wallet.get_pending_transactions()
        assert len(pending) == 0  # Transaction is complete

        # Test get completed transactions
        completed = wallet.get_completed_transactions()
        assert len(completed) == 1

    def test_multisig_wallet_transaction_errors(self):
        """Test multisig wallet transaction error cases."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Test transaction not found
        with pytest.raises(WalletError, match="Transaction not found"):
            wallet.sign_transaction(
                "nonexistent_tx", "participant_1", PrivateKey.generate()
            )

        with pytest.raises(WalletError, match="Transaction not found"):
            wallet.verify_transaction("nonexistent_tx")

        with pytest.raises(WalletError, match="Transaction not found"):
            wallet.get_transaction("nonexistent_tx")

        # Test participant not found
        transaction_data = b"test transaction data"
        transaction = wallet.create_transaction(transaction_data)

        with pytest.raises(WalletError, match="Participant not found"):
            wallet.sign_transaction(
                transaction.transaction_id,
                "nonexistent_participant",
                PrivateKey.generate(),
            )

    def test_multisig_wallet_transaction_cancellation(self):
        """Test multisig wallet transaction cancellation."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Create transaction
        transaction_data = b"test transaction data"
        transaction = wallet.create_transaction(transaction_data)

        # Cancel transaction
        assert wallet.cancel_transaction(transaction.transaction_id) is True
        assert transaction.status == "cancelled"

        # Test cancel completed transaction
        # First complete the transaction
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()

        wallet.add_participant("participant_1", private_key1.get_public_key())
        wallet.add_participant("participant_2", private_key2.get_public_key())

        wallet.sign_transaction(
            transaction.transaction_id, "participant_1", private_key1
        )
        wallet.sign_transaction(
            transaction.transaction_id, "participant_2", private_key2
        )

        # Try to cancel completed transaction
        assert wallet.cancel_transaction(transaction.transaction_id) is False

    def test_multisig_wallet_wallet_info(self):
        """Test multisig wallet info retrieval."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Add participants
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()

        wallet.add_participant("participant_1", private_key1.get_public_key())
        wallet.add_participant("participant_2", private_key2.get_public_key())

        # Create transaction
        transaction_data = b"test transaction data"
        wallet.create_transaction(transaction_data)

        info = wallet.get_wallet_info()
        assert info["wallet_id"] == "multisig_1"
        assert info["name"] == "Test Multisig Wallet"
        assert info["participant_count"] == 2
        assert info["active_participants"] == 2
        assert info["transaction_count"] == 1
        assert info["pending_transactions"] == 1
        assert info["completed_transactions"] == 0

    def test_multisig_wallet_export_import(self):
        """Test multisig wallet export and import."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Add participants
        private_key1 = PrivateKey.generate()
        private_key2 = PrivateKey.generate()

        wallet.add_participant("participant_1", private_key1.get_public_key())
        wallet.add_participant("participant_2", private_key2.get_public_key())

        # Export wallet
        export_data = wallet.export_wallet()
        assert "wallet_id" in export_data
        assert "name" in export_data
        assert "config" in export_data
        assert "participants" in export_data
        assert len(export_data["participants"]) == 2

        # Create new wallet and import
        new_wallet = MultisigWallet(
            wallet_id="multisig_2", config=config, name="Imported Wallet"
        )

        new_wallet.import_wallet(export_data)
        assert len(new_wallet.participants) == 2
        assert "participant_1" in new_wallet.participants
        assert "participant_2" in new_wallet.participants

    def test_multisig_wallet_serialization(self):
        """Test multisig wallet serialization."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        # Test to_dict
        wallet_dict = wallet.to_dict()
        assert wallet_dict["wallet_id"] == "multisig_1"
        assert wallet_dict["name"] == "Test Multisig Wallet"
        assert "config" in wallet_dict
        assert "participants" in wallet_dict
        assert "transactions" in wallet_dict

        # Test from_dict
        restored_wallet = MultisigWallet.from_dict(wallet_dict)
        assert restored_wallet.wallet_id == wallet.wallet_id
        assert restored_wallet.name == wallet.name
        assert restored_wallet.config.multisig_type == wallet.config.multisig_type

    def test_multisig_wallet_string_representation(self):
        """Test multisig wallet string representation."""
        config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
        )

        wallet = MultisigWallet(
            wallet_id="multisig_1", config=config, name="Test Multisig Wallet"
        )

        str_repr = str(wallet)
        assert "MultisigWallet" in str_repr
        assert "multisig_1" in str_repr

        repr_str = repr(wallet)
        assert "MultisigWallet" in repr_str
        assert "multisig_1" in repr_str
        assert "Test Multisig Wallet" in repr_str
