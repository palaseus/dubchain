"""
Multi-signature wallet implementation for GodChain.

This module provides sophisticated multi-signature wallet functionality with
support for various signature schemes and threshold configurations.
"""

import logging

logger = logging.getLogger(__name__)
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature
from .hd_wallet import AccountType, HDWallet, WalletAccount, WalletError
from .key_derivation import DerivationPath


class MultisigType(Enum):
    """Types of multi-signature schemes."""

    M_OF_N = "m_of_n"  # M signatures required out of N total
    THRESHOLD = "threshold"  # Threshold-based signing
    WEIGHTED = "weighted"  # Weighted signature scheme
    TIMELOCK = "timelock"  # Time-locked multi-signature


class SignatureStatus(Enum):
    """Status of signature in multi-signature process."""

    PENDING = "pending"
    SIGNED = "signed"
    INVALID = "invalid"
    EXPIRED = "expired"


@dataclass
class MultisigParticipant:
    """Represents a participant in a multi-signature wallet."""

    participant_id: str
    public_key: PublicKey
    weight: int = 1
    is_active: bool = True
    added_at: int = field(default_factory=lambda: int(time.time()))
    last_used: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate participant data."""
        if self.weight <= 0:
            raise ValueError("Participant weight must be positive")

        if not self.participant_id:
            raise ValueError("Participant ID cannot be empty")

    def update_usage(self) -> None:
        """Update last used timestamp."""
        self.last_used = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert participant to dictionary."""
        return {
            "participant_id": self.participant_id,
            "public_key": self.public_key.to_hex(),
            "weight": self.weight,
            "is_active": self.is_active,
            "added_at": self.added_at,
            "last_used": self.last_used,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultisigParticipant":
        """Create participant from dictionary."""
        return cls(
            participant_id=data["participant_id"],
            public_key=PublicKey.from_hex(data["public_key"]),
            weight=data.get("weight", 1),
            is_active=data.get("is_active", True),
            added_at=data.get("added_at", int(time.time())),
            last_used=data.get("last_used"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MultisigSignature:
    """Represents a signature in a multi-signature transaction."""

    participant_id: str
    signature: Signature
    timestamp: int = field(default_factory=lambda: int(time.time()))
    status: SignatureStatus = SignatureStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary."""
        return {
            "participant_id": self.participant_id,
            "signature": self.signature.to_hex(),
            "timestamp": self.timestamp,
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultisigSignature":
        """Create signature from dictionary."""
        # Parse signature from hex - assume it's raw r,s values
        signature_bytes = bytes.fromhex(data["signature"])
        if len(signature_bytes) != 64:
            raise ValueError("Signature must be 64 bytes (32 bytes r + 32 bytes s)")

        r = int.from_bytes(signature_bytes[:32], "big")
        s = int.from_bytes(signature_bytes[32:], "big")
        
        # Create signature with actual message hash
        message_hash = SHA256Hasher.hash(message).value
        signature = Signature(r, s, message_hash)

        return cls(
            participant_id=data["participant_id"],
            signature=signature,
            timestamp=data.get("timestamp", int(time.time())),
            status=SignatureStatus(data.get("status", "pending")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MultisigTransaction:
    """Represents a multi-signature transaction."""

    transaction_id: str
    transaction_data: bytes
    required_signatures: int
    total_participants: int
    signatures: List[MultisigSignature] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: Optional[int] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate transaction data."""
        if self.required_signatures <= 0:
            raise ValueError("Required signatures must be positive")

        if self.total_participants <= 0:
            raise ValueError("Total participants must be positive")

        if self.required_signatures > self.total_participants:
            raise ValueError("Required signatures cannot exceed total participants")

    def add_signature(self, signature: MultisigSignature) -> bool:
        """Add signature to transaction."""
        # Check if participant already signed
        for existing_sig in self.signatures:
            if existing_sig.participant_id == signature.participant_id:
                return False

        # Check expiration
        if self.expires_at and int(time.time()) > self.expires_at:
            signature.status = SignatureStatus.EXPIRED
            return False

        self.signatures.append(signature)
        return True

    def is_complete(self) -> bool:
        """Check if transaction has enough signatures."""
        valid_signatures = [
            sig for sig in self.signatures if sig.status == SignatureStatus.SIGNED
        ]
        return len(valid_signatures) >= self.required_signatures

    def get_signature_count(self) -> int:
        """Get count of valid signatures."""
        return len(
            [sig for sig in self.signatures if sig.status == SignatureStatus.SIGNED]
        )

    def get_participant_signatures(self) -> Dict[str, MultisigSignature]:
        """Get signatures by participant ID."""
        return {sig.participant_id: sig for sig in self.signatures}

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_data": self.transaction_data.hex(),
            "required_signatures": self.required_signatures,
            "total_participants": self.total_participants,
            "signatures": [sig.to_dict() for sig in self.signatures],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultisigTransaction":
        """Create transaction from dictionary."""
        transaction = cls(
            transaction_id=data["transaction_id"],
            transaction_data=bytes.fromhex(data["transaction_data"]),
            required_signatures=data["required_signatures"],
            total_participants=data["total_participants"],
            created_at=data.get("created_at", int(time.time())),
            expires_at=data.get("expires_at"),
            status=data.get("status", "pending"),
            metadata=data.get("metadata", {}),
        )

        # Add signatures
        for sig_data in data.get("signatures", []):
            signature = MultisigSignature.from_dict(sig_data)
            transaction.signatures.append(signature)

        return transaction


@dataclass
class MultisigConfig:
    """Configuration for multi-signature wallet."""

    multisig_type: MultisigType
    required_signatures: int
    total_participants: int
    timeout_seconds: Optional[int] = None
    allow_duplicate_signatures: bool = False
    require_all_participants: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.required_signatures <= 0:
            raise ValueError("Required signatures must be positive")

        if self.total_participants <= 0:
            raise ValueError("Total participants must be positive")

        if self.required_signatures > self.total_participants:
            raise ValueError("Required signatures cannot exceed total participants")

        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "multisig_type": self.multisig_type.value,
            "required_signatures": self.required_signatures,
            "total_participants": self.total_participants,
            "timeout_seconds": self.timeout_seconds,
            "allow_duplicate_signatures": self.allow_duplicate_signatures,
            "require_all_participants": self.require_all_participants,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultisigConfig":
        """Create config from dictionary."""
        return cls(
            multisig_type=MultisigType(data["multisig_type"]),
            required_signatures=data["required_signatures"],
            total_participants=data["total_participants"],
            timeout_seconds=data.get("timeout_seconds"),
            allow_duplicate_signatures=data.get("allow_duplicate_signatures", False),
            require_all_participants=data.get("require_all_participants", False),
            metadata=data.get("metadata", {}),
        )


class MultisigWallet:
    """Advanced multi-signature wallet implementation."""

    def __init__(
        self, wallet_id: str, config: MultisigConfig, name: str = "Multisig Wallet"
    ):
        """Initialize multi-signature wallet."""
        self.wallet_id = wallet_id
        self.name = name
        self.config = config
        self.participants: Dict[str, MultisigParticipant] = {}
        self.transactions: Dict[str, MultisigTransaction] = {}
        self.created_at = int(time.time())
        self.last_accessed = int(time.time())
        self.metadata: Dict[str, Any] = {}

    def add_participant(
        self,
        participant_id: str,
        public_key: PublicKey,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultisigParticipant:
        """Add participant to multi-signature wallet."""
        if participant_id in self.participants:
            raise WalletError(f"Participant {participant_id} already exists")

        if len(self.participants) >= self.config.total_participants:
            raise WalletError("Maximum number of participants reached")

        participant = MultisigParticipant(
            participant_id=participant_id,
            public_key=public_key,
            weight=weight,
            metadata=metadata or {},
        )

        self.participants[participant_id] = participant
        self.last_accessed = int(time.time())
        return participant

    def remove_participant(self, participant_id: str) -> None:
        """Remove participant from wallet."""
        if participant_id not in self.participants:
            raise WalletError(f"Participant {participant_id} not found")

        if len(self.participants) <= self.config.required_signatures:
            raise WalletError(
                "Cannot remove participant: would violate signature requirements"
            )

        del self.participants[participant_id]
        self.last_accessed = int(time.time())

    def update_participant_weight(self, participant_id: str, new_weight: int) -> None:
        """Update participant weight."""
        if participant_id not in self.participants:
            raise WalletError(f"Participant {participant_id} not found")

        if new_weight <= 0:
            raise ValueError("Weight must be positive")

        self.participants[participant_id].weight = new_weight
        self.last_accessed = int(time.time())

    def get_participant(self, participant_id: str) -> MultisigParticipant:
        """Get participant by ID."""
        if participant_id not in self.participants:
            raise WalletError(f"Participant {participant_id} not found")
        return self.participants[participant_id]

    def get_active_participants(self) -> List[MultisigParticipant]:
        """Get all active participants."""
        return [p for p in self.participants.values() if p.is_active]

    def create_transaction(
        self, transaction_data: bytes, expires_in_seconds: Optional[int] = None
    ) -> MultisigTransaction:
        """Create a new multi-signature transaction."""
        transaction_id = SHA256Hasher.hash(transaction_data).to_hex()

        # Check if transaction already exists
        if transaction_id in self.transactions:
            raise WalletError("Transaction already exists")

        # Calculate expiration time
        expires_at = None
        if expires_in_seconds:
            expires_at = int(time.time()) + expires_in_seconds
        elif self.config.timeout_seconds:
            expires_at = int(time.time()) + self.config.timeout_seconds

        transaction = MultisigTransaction(
            transaction_id=transaction_id,
            transaction_data=transaction_data,
            required_signatures=self.config.required_signatures,
            total_participants=self.config.total_participants,
            expires_at=expires_at,
        )

        self.transactions[transaction_id] = transaction
        self.last_accessed = int(time.time())
        return transaction

    def sign_transaction(
        self, transaction_id: str, participant_id: str, private_key: PrivateKey
    ) -> bool:
        """Sign a transaction."""
        if transaction_id not in self.transactions:
            raise WalletError("Transaction not found")

        if participant_id not in self.participants:
            raise WalletError("Participant not found")

        transaction = self.transactions[transaction_id]
        participant = self.participants[participant_id]

        # Check if participant already signed
        if not self.config.allow_duplicate_signatures:
            for existing_sig in transaction.signatures:
                if existing_sig.participant_id == participant_id:
                    return False

        # Check expiration
        if transaction.expires_at and int(time.time()) > transaction.expires_at:
            return False

        # Verify private key matches participant's public key
        if private_key.get_public_key() != participant.public_key:
            raise WalletError("Private key does not match participant's public key")

        # Sign transaction
        signature = private_key.sign(transaction.transaction_data)

        # Create multisig signature
        multisig_sig = MultisigSignature(
            participant_id=participant_id,
            signature=signature,
            status=SignatureStatus.SIGNED,
        )

        # Add signature
        success = transaction.add_signature(multisig_sig)
        if success:
            participant.update_usage()
            self.last_accessed = int(time.time())

        return success

    def verify_transaction(self, transaction_id: str) -> bool:
        """Verify if transaction has enough valid signatures."""
        if transaction_id not in self.transactions:
            raise WalletError("Transaction not found")

        transaction = self.transactions[transaction_id]

        # Check expiration
        if transaction.expires_at and int(time.time()) > transaction.expires_at:
            return False

        # Verify signatures
        valid_signatures = 0
        for sig in transaction.signatures:
            if sig.status == SignatureStatus.SIGNED:
                participant = self.participants.get(sig.participant_id)
                if participant and participant.is_active:
                    # Verify signature
                    try:
                        if participant.public_key.verify(
                            sig.signature, transaction.transaction_data
                        ):
                            valid_signatures += participant.weight
                        else:
                            sig.status = SignatureStatus.INVALID
                    except Exception:
                        sig.status = SignatureStatus.INVALID

        return valid_signatures >= self.config.required_signatures

    def get_transaction(self, transaction_id: str) -> MultisigTransaction:
        """Get transaction by ID."""
        if transaction_id not in self.transactions:
            raise WalletError("Transaction not found")
        return self.transactions[transaction_id]

    def get_pending_transactions(self) -> List[MultisigTransaction]:
        """Get all pending transactions."""
        return [
            tx
            for tx in self.transactions.values()
            if not tx.is_complete() and tx.status == "pending"
        ]

    def get_completed_transactions(self) -> List[MultisigTransaction]:
        """Get all completed transactions."""
        return [tx for tx in self.transactions.values() if tx.is_complete()]

    def cancel_transaction(self, transaction_id: str) -> bool:
        """Cancel a transaction."""
        if transaction_id not in self.transactions:
            raise WalletError("Transaction not found")

        transaction = self.transactions[transaction_id]
        if transaction.is_complete():
            return False  # Cannot cancel completed transaction

        transaction.status = "cancelled"
        self.last_accessed = int(time.time())
        return True

    def get_wallet_info(self) -> Dict[str, Any]:
        """Get comprehensive wallet information."""
        return {
            "wallet_id": self.wallet_id,
            "name": self.name,
            "config": self.config.to_dict(),
            "participant_count": len(self.participants),
            "active_participants": len(self.get_active_participants()),
            "transaction_count": len(self.transactions),
            "pending_transactions": len(self.get_pending_transactions()),
            "completed_transactions": len(self.get_completed_transactions()),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    def export_wallet(self, include_private_keys: bool = False) -> Dict[str, Any]:
        """Export wallet data."""
        return {
            "wallet_id": self.wallet_id,
            "name": self.name,
            "config": self.config.to_dict(),
            "participants": [p.to_dict() for p in self.participants.values()],
            "transactions": [tx.to_dict() for tx in self.transactions.values()],
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    def import_wallet(self, data: Dict[str, Any]) -> None:
        """Import wallet data."""
        # Clear existing data
        self.participants.clear()
        self.transactions.clear()

        # Import participants
        for participant_data in data.get("participants", []):
            participant = MultisigParticipant.from_dict(participant_data)
            self.participants[participant.participant_id] = participant

        # Import transactions
        for transaction_data in data.get("transactions", []):
            transaction = MultisigTransaction.from_dict(transaction_data)
            self.transactions[transaction.transaction_id] = transaction

        # Update metadata
        self.metadata = data.get("metadata", {})
        self.last_accessed = int(time.time())

    def to_dict(self, include_private_keys: bool = False) -> Dict[str, Any]:
        """Convert wallet to dictionary."""
        return self.export_wallet(include_private_keys=include_private_keys)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultisigWallet":
        """Create wallet from dictionary."""
        config = MultisigConfig.from_dict(data["config"])
        wallet = cls(
            wallet_id=data["wallet_id"],
            config=config,
            name=data.get("name", "Multisig Wallet"),
        )
        wallet.import_wallet(data)
        return wallet

    def __str__(self) -> str:
        """String representation."""
        return f"MultisigWallet(id={self.wallet_id}, participants={len(self.participants)}, transactions={len(self.transactions)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"MultisigWallet(id={self.wallet_id}, name={self.name}, "
            f"participants={len(self.participants)}, transactions={len(self.transactions)})"
        )
