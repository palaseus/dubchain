"""
ZKP proof generation components.

This module provides proof generation capabilities, trusted setup,
and key management for zero-knowledge proofs.
"""

import hashlib
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .circuits import PrivateInputs, PublicInputs, Witness, ZKCircuit
from .core import ZKPError, ZKPStatus, ZKPType


class SetupType(Enum):
    """Types of trusted setup."""

    UNIVERSAL = "universal"  # Universal setup (e.g., for zk-SNARKs)
    CIRCUIT_SPECIFIC = "circuit_specific"  # Circuit-specific setup
    NO_SETUP = "no_setup"  # No setup required (e.g., for zk-STARKs)


@dataclass
class TrustedSetup:
    """Represents a trusted setup for proof generation."""

    setup_id: str
    setup_type: SetupType
    circuit_id: str
    proving_key: bytes
    verification_key: bytes
    setup_parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if setup is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def validate(self) -> bool:
        """Validate setup data."""
        if not self.setup_id or not self.circuit_id:
            return False
        if not self.proving_key or not self.verification_key:
            return False
        if self.expires_at and self.expires_at <= self.created_at:
            return False
        return True


@dataclass
class ProvingKey:
    """Proving key for generating proofs."""

    key_data: bytes
    key_type: ZKPType
    circuit_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_hash(self) -> str:
        """Get hash of the proving key."""
        return hashlib.sha256(self.key_data).hexdigest()

    def validate(self) -> bool:
        """Validate proving key."""
        return bool(self.key_data and self.circuit_id)


@dataclass
class VerificationKey:
    """Verification key for verifying proofs."""

    key_data: bytes
    key_type: ZKPType
    circuit_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_hash(self) -> str:
        """Get hash of the verification key."""
        return hashlib.sha256(self.key_data).hexdigest()

    def validate(self) -> bool:
        """Validate verification key."""
        return bool(self.key_data and self.circuit_id)


class ProofGenerator(ABC):
    """Abstract base class for proof generators."""

    def __init__(self, circuit: ZKCircuit, setup: Optional[TrustedSetup] = None):
        self.circuit = circuit
        self.setup = setup
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the proof generator."""
        pass

    @abstractmethod
    def generate_proof(
        self, witness: Witness, public_inputs: PublicInputs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Generate a proof for the given witness and public inputs."""
        pass

    @abstractmethod
    def get_proving_key(self) -> ProvingKey:
        """Get the proving key."""
        pass

    @abstractmethod
    def get_verification_key(self) -> VerificationKey:
        """Get the verification key."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if generator is initialized."""
        return self._initialized

    def validate_witness(self, witness: Witness) -> bool:
        """Validate witness against circuit."""
        return self.circuit.verify_witness(witness)


class MockProofGenerator(ProofGenerator):
    """Mock proof generator for testing."""

    def __init__(self, circuit: ZKCircuit, setup: Optional[TrustedSetup] = None):
        super().__init__(circuit, setup)
        self._proving_key: Optional[ProvingKey] = None
        self._verification_key: Optional[VerificationKey] = None

    def initialize(self) -> None:
        """Initialize the mock proof generator."""
        if self._initialized:
            return

        # Generate mock keys
        self._proving_key = ProvingKey(
            key_data=secrets.token_bytes(256),
            key_type=ZKPType.MOCK,
            circuit_id=self.circuit.circuit_id,
        )

        self._verification_key = VerificationKey(
            key_data=secrets.token_bytes(128),
            key_type=ZKPType.MOCK,
            circuit_id=self.circuit.circuit_id,
        )

        self._initialized = True

    def generate_proof(
        self, witness: Witness, public_inputs: PublicInputs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Generate a mock proof."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")

        if not self.validate_witness(witness):
            raise ZKPError("Invalid witness")

        # Generate mock proof data
        proof_data = secrets.token_bytes(128)

        # Add some deterministic elements based on witness
        witness_hash = hashlib.sha256(witness.to_bytes()).digest()
        proof_data = witness_hash + proof_data[32:]

        metadata = {
            "generator": "mock",
            "circuit_id": self.circuit.circuit_id,
            "witness_hash": witness_hash.hex(),
            "public_inputs_count": len(public_inputs.inputs),
        }

        return proof_data, metadata

    def get_proving_key(self) -> ProvingKey:
        """Get the proving key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._proving_key

    def get_verification_key(self) -> VerificationKey:
        """Get the verification key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._verification_key


class ZKSNARKProofGenerator(ProofGenerator):
    """Proof generator for zk-SNARKs."""

    def __init__(self, circuit: ZKCircuit, setup: TrustedSetup):
        super().__init__(circuit, setup)
        self._proving_key: Optional[ProvingKey] = None
        self._verification_key: Optional[VerificationKey] = None

    def initialize(self) -> None:
        """Initialize the zk-SNARK proof generator."""
        if self._initialized:
            return

        if not self.setup:
            raise ZKPError("Trusted setup required for zk-SNARKs")

        if not self.setup.validate():
            raise ZKPError("Invalid trusted setup")

        if self.setup.is_expired():
            raise ZKPError("Trusted setup has expired")

        # Extract keys from setup
        self._proving_key = ProvingKey(
            key_data=self.setup.proving_key,
            key_type=ZKPType.ZK_SNARK,
            circuit_id=self.circuit.circuit_id,
            parameters=self.setup.setup_parameters,
        )

        self._verification_key = VerificationKey(
            key_data=self.setup.verification_key,
            key_type=ZKPType.ZK_SNARK,
            circuit_id=self.circuit.circuit_id,
            parameters=self.setup.setup_parameters,
        )

        self._initialized = True

    def generate_proof(
        self, witness: Witness, public_inputs: PublicInputs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Generate a zk-SNARK proof."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")

        if not self.validate_witness(witness):
            raise ZKPError("Invalid witness")

        # In a real implementation, this would use a zk-SNARK library
        # For now, we'll simulate the proof generation

        # Simulate proof generation time
        time.sleep(0.1)

        # Generate proof data (in reality, this would be the actual SNARK proof)
        proof_data = secrets.token_bytes(256)
        # Replace any null bytes with non-null bytes
        proof_data = proof_data.replace(b"\x00", b"\x01")

        # Add deterministic elements
        witness_hash = hashlib.sha256(witness.to_bytes()).digest()
        public_hash = hashlib.sha256(public_inputs.to_bytes()).digest()

        # Combine hashes for deterministic proof
        combined_hash = hashlib.sha256(witness_hash + public_hash).digest()
        # Replace any null bytes in hash with non-null bytes
        combined_hash = combined_hash.replace(b"\x00", b"\x01")
        proof_data = combined_hash + proof_data[32:]

        metadata = {
            "generator": "zk_snark",
            "circuit_id": self.circuit.circuit_id,
            "witness_hash": witness_hash.hex(),
            "public_hash": public_hash.hex(),
            "proof_size": len(proof_data),
            "setup_id": self.setup.setup_id,
        }

        return proof_data, metadata

    def get_proving_key(self) -> ProvingKey:
        """Get the proving key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._proving_key

    def get_verification_key(self) -> VerificationKey:
        """Get the verification key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._verification_key


class ZKSTARKProofGenerator(ProofGenerator):
    """Proof generator for zk-STARKs."""

    def __init__(self, circuit: ZKCircuit, setup: Optional[TrustedSetup] = None):
        super().__init__(circuit, setup)
        self._proving_key: Optional[ProvingKey] = None
        self._verification_key: Optional[VerificationKey] = None

    def initialize(self) -> None:
        """Initialize the zk-STARK proof generator."""
        if self._initialized:
            return

        # zk-STARKs don't require trusted setup
        # Generate keys based on circuit

        circuit_hash = hashlib.sha256(self.circuit.circuit_id.encode()).digest()

        self._proving_key = ProvingKey(
            key_data=circuit_hash + secrets.token_bytes(128),
            key_type=ZKPType.ZK_STARK,
            circuit_id=self.circuit.circuit_id,
        )

        self._verification_key = VerificationKey(
            key_data=circuit_hash + secrets.token_bytes(64),
            key_type=ZKPType.ZK_STARK,
            circuit_id=self.circuit.circuit_id,
        )

        self._initialized = True

    def generate_proof(
        self, witness: Witness, public_inputs: PublicInputs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Generate a zk-STARK proof."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")

        if not self.validate_witness(witness):
            raise ZKPError("Invalid witness")

        # In a real implementation, this would use a zk-STARK library
        # For now, we'll simulate the proof generation

        # Simulate proof generation time (STARKs are typically slower)
        time.sleep(0.2)

        # Generate proof data (in reality, this would be the actual STARK proof)
        proof_data = secrets.token_bytes(512)  # STARK proofs are typically larger
        # Replace any null bytes with non-null bytes
        proof_data = proof_data.replace(b"\x00", b"\x01")

        # Add deterministic elements
        witness_hash = hashlib.sha256(witness.to_bytes()).digest()
        public_hash = hashlib.sha256(public_inputs.to_bytes()).digest()

        # Combine hashes for deterministic proof
        combined_hash = hashlib.sha256(witness_hash + public_hash).digest()
        # Replace any null bytes in hash with non-null bytes
        combined_hash = combined_hash.replace(b"\x00", b"\x01")
        proof_data = combined_hash + proof_data[32:]

        metadata = {
            "generator": "zk_stark",
            "circuit_id": self.circuit.circuit_id,
            "witness_hash": witness_hash.hex(),
            "public_hash": public_hash.hex(),
            "proof_size": len(proof_data),
        }

        return proof_data, metadata

    def get_proving_key(self) -> ProvingKey:
        """Get the proving key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._proving_key

    def get_verification_key(self) -> VerificationKey:
        """Get the verification key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._verification_key


class BulletproofGenerator(ProofGenerator):
    """Proof generator for Bulletproofs."""

    def __init__(self, circuit: ZKCircuit, setup: Optional[TrustedSetup] = None):
        super().__init__(circuit, setup)
        self._proving_key: Optional[ProvingKey] = None
        self._verification_key: Optional[VerificationKey] = None

    def initialize(self) -> None:
        """Initialize the Bulletproof generator."""
        if self._initialized:
            return

        # Bulletproofs don't require trusted setup
        # Generate keys based on circuit

        circuit_hash = hashlib.sha256(self.circuit.circuit_id.encode()).digest()

        self._proving_key = ProvingKey(
            key_data=circuit_hash + secrets.token_bytes(96),
            key_type=ZKPType.BULLETPROOF,
            circuit_id=self.circuit.circuit_id,
        )

        self._verification_key = VerificationKey(
            key_data=circuit_hash + secrets.token_bytes(32),
            key_type=ZKPType.BULLETPROOF,
            circuit_id=self.circuit.circuit_id,
        )

        self._initialized = True

    def generate_proof(
        self, witness: Witness, public_inputs: PublicInputs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Generate a Bulletproof."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")

        if not self.validate_witness(witness):
            raise ZKPError("Invalid witness")

        # In a real implementation, this would use a Bulletproof library
        # For now, we'll simulate the proof generation

        # Simulate proof generation time
        time.sleep(0.05)

        # Generate proof data (in reality, this would be the actual Bulletproof)
        proof_data = secrets.token_bytes(64)  # Bulletproofs are compact
        # Replace any null bytes with non-null bytes
        proof_data = proof_data.replace(b"\x00", b"\x01")

        # Add deterministic elements
        witness_hash = hashlib.sha256(witness.to_bytes()).digest()
        public_hash = hashlib.sha256(public_inputs.to_bytes()).digest()

        # Combine hashes for deterministic proof
        combined_hash = hashlib.sha256(witness_hash + public_hash).digest()
        # Replace any null bytes in hash with non-null bytes
        combined_hash = combined_hash.replace(b"\x00", b"\x01")
        proof_data = combined_hash + proof_data[32:]

        metadata = {
            "generator": "bulletproof",
            "circuit_id": self.circuit.circuit_id,
            "witness_hash": witness_hash.hex(),
            "public_hash": public_hash.hex(),
            "proof_size": len(proof_data),
        }

        return proof_data, metadata

    def get_proving_key(self) -> ProvingKey:
        """Get the proving key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._proving_key

    def get_verification_key(self) -> VerificationKey:
        """Get the verification key."""
        if not self._initialized:
            raise ZKPError("Proof generator not initialized")
        return self._verification_key


def create_proof_generator(
    circuit: ZKCircuit, zkp_type: ZKPType, setup: Optional[TrustedSetup] = None
) -> ProofGenerator:
    """Create a proof generator for the specified ZKP type."""
    if zkp_type == ZKPType.MOCK:
        return MockProofGenerator(circuit, setup)
    elif zkp_type == ZKPType.ZK_SNARK:
        if not setup:
            raise ZKPError("Trusted setup required for zk-SNARKs")
        return ZKSNARKProofGenerator(circuit, setup)
    elif zkp_type == ZKPType.ZK_STARK:
        return ZKSTARKProofGenerator(circuit, setup)
    elif zkp_type == ZKPType.BULLETPROOF:
        return BulletproofGenerator(circuit, setup)
    else:
        raise ValueError(f"Unsupported ZKP type: {zkp_type}")
