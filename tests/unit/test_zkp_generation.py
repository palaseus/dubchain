"""
Unit tests for ZKP generation components.

This module tests proof generation, trusted setup, and key management.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
import secrets
from unittest.mock import Mock, patch

from src.dubchain.crypto.zkp.generation import (
    SetupType,
    TrustedSetup,
    ProvingKey,
    VerificationKey,
    ProofGenerator,
    MockProofGenerator,
    ZKSNARKProofGenerator,
    ZKSTARKProofGenerator,
    BulletproofGenerator,
    create_proof_generator,
)
from src.dubchain.crypto.zkp.core import ZKPType, ZKPError
from src.dubchain.crypto.zkp.circuits import (
    ZKCircuit,
    Witness,
    PublicInputs,
    PrivateInputs,
)


class MockZKCircuit(ZKCircuit):
    """Mock ZK circuit for testing."""
    
    def __init__(self, circuit_id: str):
        super().__init__(circuit_id)
        self._built = True
    
    def build(self) -> None:
        pass
    
    def generate_witness(self, public_inputs: PublicInputs, private_inputs: PrivateInputs) -> Witness:
        witness = Witness()
        for name, value in zip(public_inputs.input_names, public_inputs.inputs):
            witness.set_value(name, value, is_public=True)
        for name, value in zip(private_inputs.input_names, private_inputs.inputs):
            witness.set_value(name, value, is_public=False)
        return witness
    
    def verify_witness(self, witness: Witness) -> bool:
        return True


class TestTrustedSetup:
    """Test TrustedSetup data structure."""
    
    def test_trusted_setup_creation(self):
        """Test trusted setup creation."""
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.UNIVERSAL,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        assert setup.setup_id == "test_setup"
        assert setup.setup_type == SetupType.UNIVERSAL
        assert setup.circuit_id == "test_circuit"
        assert setup.proving_key == b"proving_key_data"
        assert setup.verification_key == b"verification_key_data"
        assert setup.created_at > 0
        assert setup.expires_at is None
        assert setup.setup_parameters == {}
    
    def test_trusted_setup_with_expiration(self):
        """Test trusted setup with expiration."""
        expires_at = time.time() + 3600  # 1 hour from now
        
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data",
            expires_at=expires_at
        )
        
        assert setup.expires_at == expires_at
        assert not setup.is_expired()
    
    def test_trusted_setup_expired(self):
        """Test expired trusted setup."""
        expires_at = time.time() - 3600  # 1 hour ago
        
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data",
            expires_at=expires_at
        )
        
        assert setup.is_expired()
    
    def test_trusted_setup_validation(self):
        """Test trusted setup validation."""
        # Valid setup
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.UNIVERSAL,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        assert setup.validate() is True
        
        # Invalid setup - empty setup ID
        setup.setup_id = ""
        assert setup.validate() is False
        
        # Invalid setup - empty circuit ID
        setup.setup_id = "test_setup"
        setup.circuit_id = ""
        assert setup.validate() is False
        
        # Invalid setup - empty proving key
        setup.circuit_id = "test_circuit"
        setup.proving_key = b""
        assert setup.validate() is False
        
        # Invalid setup - empty verification key
        setup.proving_key = b"proving_key_data"
        setup.verification_key = b""
        assert setup.validate() is False
        
        # Invalid setup - expires before created
        setup.verification_key = b"verification_key_data"
        setup.expires_at = setup.created_at - 1
        assert setup.validate() is False


class TestProvingKey:
    """Test ProvingKey data structure."""
    
    def test_proving_key_creation(self):
        """Test proving key creation."""
        key = ProvingKey(
            key_data=b"proving_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        assert key.key_data == b"proving_key_data"
        assert key.key_type == ZKPType.ZK_SNARK
        assert key.circuit_id == "test_circuit"
        assert key.parameters == {}
    
    def test_proving_key_hash(self):
        """Test proving key hashing."""
        key1 = ProvingKey(
            key_data=b"proving_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        key2 = ProvingKey(
            key_data=b"proving_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        # Same key should have same hash
        assert key1.get_hash() == key2.get_hash()
        
        # Different key should have different hash
        key3 = ProvingKey(
            key_data=b"different_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        assert key1.get_hash() != key3.get_hash()
    
    def test_proving_key_validation(self):
        """Test proving key validation."""
        # Valid key
        key = ProvingKey(
            key_data=b"proving_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        assert key.validate() is True
        
        # Invalid key - empty data
        key.key_data = b""
        assert key.validate() is False
        
        # Invalid key - empty circuit ID
        key.key_data = b"proving_key_data"
        key.circuit_id = ""
        assert key.validate() is False


class TestVerificationKey:
    """Test VerificationKey data structure."""
    
    def test_verification_key_creation(self):
        """Test verification key creation."""
        key = VerificationKey(
            key_data=b"verification_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        assert key.key_data == b"verification_key_data"
        assert key.key_type == ZKPType.ZK_SNARK
        assert key.circuit_id == "test_circuit"
        assert key.parameters == {}
    
    def test_verification_key_hash(self):
        """Test verification key hashing."""
        key1 = VerificationKey(
            key_data=b"verification_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        key2 = VerificationKey(
            key_data=b"verification_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        # Same key should have same hash
        assert key1.get_hash() == key2.get_hash()
        
        # Different key should have different hash
        key3 = VerificationKey(
            key_data=b"different_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        assert key1.get_hash() != key3.get_hash()
    
    def test_verification_key_validation(self):
        """Test verification key validation."""
        # Valid key
        key = VerificationKey(
            key_data=b"verification_key_data",
            key_type=ZKPType.ZK_SNARK,
            circuit_id="test_circuit"
        )
        
        assert key.validate() is True
        
        # Invalid key - empty data
        key.key_data = b""
        assert key.validate() is False
        
        # Invalid key - empty circuit ID
        key.key_data = b"verification_key_data"
        key.circuit_id = ""
        assert key.validate() is False


class TestMockProofGenerator:
    """Test MockProofGenerator."""
    
    def test_mock_generator_creation(self):
        """Test mock generator creation."""
        circuit = MockZKCircuit("test_circuit")
        generator = MockProofGenerator(circuit)
        
        assert generator.circuit == circuit
        assert generator.setup is None
        assert not generator.is_initialized
    
    def test_mock_generator_initialization(self):
        """Test mock generator initialization."""
        circuit = MockZKCircuit("test_circuit")
        generator = MockProofGenerator(circuit)
        
        generator.initialize()
        
        assert generator.is_initialized
        assert generator._proving_key is not None
        assert generator._verification_key is not None
        assert generator._proving_key.key_type == ZKPType.MOCK
        assert generator._verification_key.key_type == ZKPType.MOCK
    
    def test_mock_generator_proof_generation(self):
        """Test mock proof generation."""
        circuit = MockZKCircuit("test_circuit")
        generator = MockProofGenerator(circuit)
        generator.initialize()
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        proof_data, metadata = generator.generate_proof(public_inputs, private_inputs)
        
        assert isinstance(proof_data, bytes)
        assert len(proof_data) == 128
        assert metadata["generator"] == "mock"
        assert metadata["circuit_id"] == "test_circuit"
        assert "witness_hash" in metadata
        assert metadata["public_inputs_count"] == 1
    
    def test_mock_generator_invalid_witness(self):
        """Test mock generator with invalid witness."""
        circuit = MockZKCircuit("test_circuit")
        generator = MockProofGenerator(circuit)
        generator.initialize()
        
        # Mock circuit to return invalid witness
        circuit.verify_witness = Mock(return_value=False)
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        with pytest.raises(ZKPError, match="Invalid witness"):
            generator.generate_proof(public_inputs, private_inputs)
    
    def test_mock_generator_not_initialized(self):
        """Test mock generator operations when not initialized."""
        circuit = MockZKCircuit("test_circuit")
        generator = MockProofGenerator(circuit)
        
        with pytest.raises(ZKPError, match="Proof generator not initialized"):
            generator.get_proving_key()
        
        with pytest.raises(ZKPError, match="Proof generator not initialized"):
            generator.get_verification_key()
    
    def test_mock_generator_get_keys(self):
        """Test getting proving and verification keys."""
        circuit = MockZKCircuit("test_circuit")
        generator = MockProofGenerator(circuit)
        generator.initialize()
        
        proving_key = generator.get_proving_key()
        verification_key = generator.get_verification_key()
        
        assert isinstance(proving_key, ProvingKey)
        assert isinstance(verification_key, VerificationKey)
        assert proving_key.circuit_id == "test_circuit"
        assert verification_key.circuit_id == "test_circuit"


class TestZKSNARKProofGenerator:
    """Test ZKSNARKProofGenerator."""
    
    def test_snark_generator_creation(self):
        """Test zk-SNARK generator creation."""
        circuit = MockZKCircuit("test_circuit")
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        generator = ZKSNARKProofGenerator(circuit, setup)
        
        assert generator.circuit == circuit
        assert generator.setup == setup
        assert not generator.is_initialized
    
    def test_snark_generator_initialization(self):
        """Test zk-SNARK generator initialization."""
        circuit = MockZKCircuit("test_circuit")
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        generator = ZKSNARKProofGenerator(circuit, setup)
        generator.initialize()
        
        assert generator.is_initialized
        assert generator._proving_key is not None
        assert generator._verification_key is not None
        assert generator._proving_key.key_type == ZKPType.ZK_SNARK
        assert generator._verification_key.key_type == ZKPType.ZK_SNARK
    
    def test_snark_generator_no_setup(self):
        """Test zk-SNARK generator without setup."""
        circuit = MockZKCircuit("test_circuit")
        generator = ZKSNARKProofGenerator(circuit, None)
        
        with pytest.raises(ZKPError, match="Trusted setup required for zk-SNARKs"):
            generator.initialize()
    
    def test_snark_generator_invalid_setup(self):
        """Test zk-SNARK generator with invalid setup."""
        circuit = MockZKCircuit("test_circuit")
        setup = TrustedSetup(
            setup_id="",  # Invalid setup
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        generator = ZKSNARKProofGenerator(circuit, setup)
        
        with pytest.raises(ZKPError, match="Invalid trusted setup"):
            generator.initialize()
    
    def test_snark_generator_expired_setup(self):
        """Test zk-SNARK generator with expired setup."""
        circuit = MockZKCircuit("test_circuit")
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data",
            expires_at=time.time() - 3600  # Expired
        )
        
        generator = ZKSNARKProofGenerator(circuit, setup)
        
        with pytest.raises(ZKPError, match="Invalid trusted setup"):
            generator.initialize()
    
    def test_snark_generator_proof_generation(self):
        """Test zk-SNARK proof generation."""
        circuit = MockZKCircuit("test_circuit")
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        generator = ZKSNARKProofGenerator(circuit, setup)
        generator.initialize()
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        proof_data, metadata = generator.generate_proof(public_inputs, private_inputs)
        
        assert isinstance(proof_data, bytes)
        assert len(proof_data) == 256
        assert metadata["generator"] == "zk_snark"
        assert metadata["circuit_id"] == "test_circuit"
        assert "witness_hash" in metadata
        assert "public_hash" in metadata
        assert metadata["setup_id"] == "test_setup"


class TestZKSTARKProofGenerator:
    """Test ZKSTARKProofGenerator."""
    
    def test_stark_generator_creation(self):
        """Test zk-STARK generator creation."""
        circuit = MockZKCircuit("test_circuit")
        generator = ZKSTARKProofGenerator(circuit)
        
        assert generator.circuit == circuit
        assert generator.setup is None
        assert not generator.is_initialized
    
    def test_stark_generator_initialization(self):
        """Test zk-STARK generator initialization."""
        circuit = MockZKCircuit("test_circuit")
        generator = ZKSTARKProofGenerator(circuit)
        
        generator.initialize()
        
        assert generator.is_initialized
        assert generator._proving_key is not None
        assert generator._verification_key is not None
        assert generator._proving_key.key_type == ZKPType.ZK_STARK
        assert generator._verification_key.key_type == ZKPType.ZK_STARK
    
    def test_stark_generator_proof_generation(self):
        """Test zk-STARK proof generation."""
        circuit = MockZKCircuit("test_circuit")
        generator = ZKSTARKProofGenerator(circuit)
        generator.initialize()
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        proof_data, metadata = generator.generate_proof(public_inputs, private_inputs)
        
        assert isinstance(proof_data, bytes)
        assert len(proof_data) == 512  # STARK proofs are larger
        assert metadata["generator"] == "zk_stark"
        assert metadata["circuit_id"] == "test_circuit"
        assert "witness_hash" in metadata
        assert "public_hash" in metadata


class TestBulletproofGenerator:
    """Test BulletproofGenerator."""
    
    def test_bulletproof_generator_creation(self):
        """Test Bulletproof generator creation."""
        circuit = MockZKCircuit("test_circuit")
        generator = BulletproofGenerator(circuit)
        
        assert generator.circuit == circuit
        assert generator.setup is None
        assert not generator.is_initialized
    
    def test_bulletproof_generator_initialization(self):
        """Test Bulletproof generator initialization."""
        circuit = MockZKCircuit("test_circuit")
        generator = BulletproofGenerator(circuit)
        
        generator.initialize()
        
        assert generator.is_initialized
        assert generator._proving_key is not None
        assert generator._verification_key is not None
        assert generator._proving_key.key_type == ZKPType.BULLETPROOF
        assert generator._verification_key.key_type == ZKPType.BULLETPROOF
    
    def test_bulletproof_generator_proof_generation(self):
        """Test Bulletproof proof generation."""
        circuit = MockZKCircuit("test_circuit")
        generator = BulletproofGenerator(circuit)
        generator.initialize()
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        proof_data, metadata = generator.generate_proof(public_inputs, private_inputs)
        
        assert isinstance(proof_data, bytes)
        assert len(proof_data) == 64  # Bulletproofs are compact
        assert metadata["generator"] == "bulletproof"
        assert metadata["circuit_id"] == "test_circuit"
        assert "witness_hash" in metadata
        assert "public_hash" in metadata


class TestCreateProofGenerator:
    """Test create_proof_generator function."""
    
    def test_create_mock_generator(self):
        """Test creating mock proof generator."""
        circuit = MockZKCircuit("test_circuit")
        generator = create_proof_generator(circuit, ZKPType.MOCK)
        
        assert isinstance(generator, MockProofGenerator)
        assert generator.circuit == circuit
    
    def test_create_snark_generator(self):
        """Test creating zk-SNARK proof generator."""
        circuit = MockZKCircuit("test_circuit")
        setup = TrustedSetup(
            setup_id="test_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"proving_key_data",
            verification_key=b"verification_key_data"
        )
        
        generator = create_proof_generator(circuit, ZKPType.ZK_SNARK, setup)
        
        assert isinstance(generator, ZKSNARKProofGenerator)
        assert generator.circuit == circuit
        assert generator.setup == setup
    
    def test_create_snark_generator_no_setup(self):
        """Test creating zk-SNARK generator without setup."""
        circuit = MockZKCircuit("test_circuit")
        
        with pytest.raises(ZKPError, match="Trusted setup required for zk-SNARKs"):
            create_proof_generator(circuit, ZKPType.ZK_SNARK)
    
    def test_create_stark_generator(self):
        """Test creating zk-STARK proof generator."""
        circuit = MockZKCircuit("test_circuit")
        generator = create_proof_generator(circuit, ZKPType.ZK_STARK)
        
        assert isinstance(generator, ZKSTARKProofGenerator)
        assert generator.circuit == circuit
    
    def test_create_bulletproof_generator(self):
        """Test creating Bulletproof generator."""
        circuit = MockZKCircuit("test_circuit")
        generator = create_proof_generator(circuit, ZKPType.BULLETPROOF)
        
        assert isinstance(generator, BulletproofGenerator)
        assert generator.circuit == circuit
    
    def test_create_unsupported_generator(self):
        """Test creating unsupported generator type."""
        circuit = MockZKCircuit("test_circuit")
        
        with pytest.raises(ValueError, match="Unsupported ZKP type"):
            create_proof_generator(circuit, "unsupported")  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__])
