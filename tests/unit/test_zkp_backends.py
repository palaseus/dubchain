"""
Unit tests for ZKP backend implementations.

This module tests the concrete backend implementations for different
proof systems (zk-SNARKs, zk-STARKs, Bulletproofs, and mock).
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.crypto.zkp.backends import (
    MockZKPBackend,
    ZKSNARKBackend,
    ZKSTARKBackend,
    BulletproofBackend,
)
from src.dubchain.crypto.zkp.core import (
    ZKPConfig,
    ZKPType,
    ZKPStatus,
    ProofRequest,
    Proof,
    VerificationResult,
)


class TestMockZKPBackend:
    """Test MockZKPBackend."""
    
    def test_mock_backend_creation(self):
        """Test mock backend creation."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        
        assert backend.config == config
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._generators) == 0
    
    def test_mock_backend_initialization(self):
        """Test mock backend initialization."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        
        backend.initialize()
        
        assert backend.is_initialized
    
    def test_mock_backend_proof_generation(self):
        """Test mock backend proof generation."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1", b"public2"],
            private_inputs=[b"private1", b"private2"],
            proof_type=ZKPType.MOCK
        )
        
        result = backend.generate_proof(request)
        
        assert result.is_success
        assert result.proof is not None
        assert result.proof.circuit_id == "test_circuit"
        assert result.proof.proof_type == ZKPType.MOCK
        assert result.proof.public_inputs == [b"public1", b"public2"]
        assert result.generation_time > 0
    
    def test_mock_backend_proof_generation_invalid_inputs(self):
        """Test mock backend proof generation with invalid inputs."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        # Empty public inputs
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[],  # Invalid
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        
        result = backend.generate_proof(request)
        
        assert not result.is_success
        assert result.status == ZKPStatus.INVALID_INPUT
        assert "Invalid public inputs" in result.error_message
    
    def test_mock_backend_proof_verification(self):
        """Test mock backend proof verification."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        proof = Proof(
            proof_data=b"test_proof_data",
            public_inputs=[b"public1", b"public2"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        result = backend.verify_proof(proof, [b"public1", b"public2"])
        
        assert result.is_success
        assert result.is_valid
        assert result.verification_time > 0
    
    def test_mock_backend_proof_verification_invalid_proof(self):
        """Test mock backend proof verification with invalid proof."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        # Proof with too short data
        proof = Proof(
            proof_data=b"short",  # Too short
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        result = backend.verify_proof(proof, [b"public1"])
        
        assert result.status == ZKPStatus.INVALID_PROOF
        assert "Proof data too short" in result.error_message
    
    def test_mock_backend_proof_verification_mismatched_inputs(self):
        """Test mock backend proof verification with mismatched inputs."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        proof = Proof(
            proof_data=b"test_proof_data",
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        # Different public inputs
        result = backend.verify_proof(proof, [b"different_input"])
        
        assert result.status == ZKPStatus.INVALID_PROOF
        assert "Public inputs don't match" in result.error_message
    
    def test_mock_backend_circuit_info(self):
        """Test mock backend circuit info."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        # Generate a proof to create circuit
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        backend.generate_proof(request)
        
        # Get circuit info
        info = backend.get_circuit_info("test_circuit")
        
        assert info["circuit_id"] == "test_circuit"
        assert info["exists"] is True
        assert info["constraint_count"] >= 0  # May be 0 for simple circuits
        assert info["variable_count"] > 0
    
    def test_mock_backend_circuit_info_nonexistent(self):
        """Test mock backend circuit info for nonexistent circuit."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        info = backend.get_circuit_info("nonexistent_circuit")
        
        assert info["circuit_id"] == "nonexistent_circuit"
        assert info["exists"] is False
        assert info["constraint_count"] == 0
        assert info["variable_count"] == 0
    
    def test_mock_backend_cleanup(self):
        """Test mock backend cleanup."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        backend.initialize()
        
        # Add some circuits
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        backend.generate_proof(request)
        
        assert len(backend._circuits) > 0
        
        backend.cleanup()
        
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._generators) == 0
    
    def test_mock_backend_not_initialized(self):
        """Test mock backend operations when not initialized."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        backend = MockZKPBackend(config)
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        
        result = backend.generate_proof(request)
        
        assert not result.is_success
        assert result.status == ZKPStatus.BACKEND_ERROR
        assert "Backend not initialized" in result.error_message
        
        proof = Proof(
            proof_data=b"test_proof",
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        result = backend.verify_proof(proof, [b"public1"])
        
        assert result.status == ZKPStatus.BACKEND_ERROR
        assert "Backend not initialized" in result.error_message


class TestZKSNARKBackend:
    """Test ZKSNARKBackend."""
    
    def test_snark_backend_creation(self):
        """Test zk-SNARK backend creation."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        
        assert backend.config == config
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._setups) == 0
        assert len(backend._generators) == 0
    
    def test_snark_backend_initialization(self):
        """Test zk-SNARK backend initialization."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        
        backend.initialize()
        
        assert backend.is_initialized
    
    def test_snark_backend_proof_generation(self):
        """Test zk-SNARK backend proof generation."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        backend.initialize()
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1", b"public2"],
            private_inputs=[b"private1", b"private2"],
            proof_type=ZKPType.ZK_SNARK
        )
        
        result = backend.generate_proof(request)
        
        assert result.is_success
        assert result.proof is not None
        assert result.proof.circuit_id == "test_circuit"
        assert result.proof.proof_type == ZKPType.ZK_SNARK
        assert result.proof.public_inputs == [b"public1", b"public2"]
        assert result.generation_time > 0
    
    def test_snark_backend_proof_verification(self):
        """Test zk-SNARK backend proof verification."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        backend.initialize()
        
        proof = Proof(
            proof_data=b"test_snark_proof_data" * 10,  # Larger proof
            public_inputs=[b"public1", b"public2"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK
        )
        
        result = backend.verify_proof(proof, [b"public1", b"public2"])
        
        assert result.is_success
        assert result.is_valid
        assert result.verification_time > 0
    
    def test_snark_backend_proof_verification_invalid_size(self):
        """Test zk-SNARK backend proof verification with invalid size."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        backend.initialize()
        
        # Proof with too small data for SNARK
        proof = Proof(
            proof_data=b"short",  # Too short for SNARK
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK
        )
        
        result = backend.verify_proof(proof, [b"public1"])
        
        assert result.status == ZKPStatus.INVALID_PROOF
        assert "Invalid SNARK proof size" in result.error_message
    
    def test_snark_backend_circuit_info(self):
        """Test zk-SNARK backend circuit info."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        backend.initialize()
        
        # Generate a proof to create circuit
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.ZK_SNARK
        )
        backend.generate_proof(request)
        
        # Get circuit info
        info = backend.get_circuit_info("test_circuit")
        
        assert info["circuit_id"] == "test_circuit"
        assert info["exists"] is True
        assert info["constraint_count"] > 0
        assert info["variable_count"] > 0
    
    def test_snark_backend_cleanup(self):
        """Test zk-SNARK backend cleanup."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        backend = ZKSNARKBackend(config)
        backend.initialize()
        
        # Add some circuits and setups
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.ZK_SNARK
        )
        backend.generate_proof(request)
        
        assert len(backend._circuits) > 0
        assert len(backend._setups) > 0
        assert len(backend._generators) > 0
        
        backend.cleanup()
        
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._setups) == 0
        assert len(backend._generators) == 0


class TestZKSTARKBackend:
    """Test ZKSTARKBackend."""
    
    def test_stark_backend_creation(self):
        """Test zk-STARK backend creation."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        
        assert backend.config == config
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._generators) == 0
    
    def test_stark_backend_initialization(self):
        """Test zk-STARK backend initialization."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        
        backend.initialize()
        
        assert backend.is_initialized
    
    def test_stark_backend_proof_generation(self):
        """Test zk-STARK backend proof generation."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        backend.initialize()
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1", b"public2"],
            private_inputs=[b"private1", b"private2"],
            proof_type=ZKPType.ZK_STARK
        )
        
        result = backend.generate_proof(request)
        
        assert result.is_success
        assert result.proof is not None
        assert result.proof.circuit_id == "test_circuit"
        assert result.proof.proof_type == ZKPType.ZK_STARK
        assert result.proof.public_inputs == [b"public1", b"public2"]
        assert result.generation_time > 0
    
    def test_stark_backend_proof_verification(self):
        """Test zk-STARK backend proof verification."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        backend.initialize()
        
        proof = Proof(
            proof_data=b"test_stark_proof_data" * 20,  # Larger proof
            public_inputs=[b"public1", b"public2"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_STARK
        )
        
        result = backend.verify_proof(proof, [b"public1", b"public2"])
        
        assert result.is_success
        assert result.is_valid
        assert result.verification_time > 0
    
    def test_stark_backend_proof_verification_invalid_size(self):
        """Test zk-STARK backend proof verification with invalid size."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        backend.initialize()
        
        # Proof with too small data for STARK
        proof = Proof(
            proof_data=b"short",  # Too short for STARK
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_STARK
        )
        
        result = backend.verify_proof(proof, [b"public1"])
        
        assert result.status == ZKPStatus.INVALID_PROOF
        assert "Invalid STARK proof size" in result.error_message
    
    def test_stark_backend_circuit_info(self):
        """Test zk-STARK backend circuit info."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        backend.initialize()
        
        # Generate a proof to create circuit
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.ZK_STARK
        )
        backend.generate_proof(request)
        
        # Get circuit info
        info = backend.get_circuit_info("test_circuit")
        
        assert info["circuit_id"] == "test_circuit"
        assert info["exists"] is True
        assert info["constraint_count"] > 0
        assert info["variable_count"] > 0
    
    def test_stark_backend_cleanup(self):
        """Test zk-STARK backend cleanup."""
        config = ZKPConfig(backend_type=ZKPType.ZK_STARK)
        backend = ZKSTARKBackend(config)
        backend.initialize()
        
        # Add some circuits
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.ZK_STARK
        )
        backend.generate_proof(request)
        
        assert len(backend._circuits) > 0
        assert len(backend._generators) > 0
        
        backend.cleanup()
        
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._generators) == 0


class TestBulletproofBackend:
    """Test BulletproofBackend."""
    
    def test_bulletproof_backend_creation(self):
        """Test Bulletproof backend creation."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        
        assert backend.config == config
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._generators) == 0
    
    def test_bulletproof_backend_initialization(self):
        """Test Bulletproof backend initialization."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        
        backend.initialize()
        
        assert backend.is_initialized
    
    def test_bulletproof_backend_proof_generation(self):
        """Test Bulletproof backend proof generation."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        backend.initialize()
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1", b"public2"],
            private_inputs=[b"private1", b"private2"],
            proof_type=ZKPType.BULLETPROOF
        )
        
        result = backend.generate_proof(request)
        
        assert result.is_success
        assert result.proof is not None
        assert result.proof.circuit_id == "test_circuit"
        assert result.proof.proof_type == ZKPType.BULLETPROOF
        assert result.proof.public_inputs == [b"public1", b"public2"]
        assert result.generation_time > 0
    
    def test_bulletproof_backend_proof_verification(self):
        """Test Bulletproof backend proof verification."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        backend.initialize()
        
        proof = Proof(
            proof_data=b"test_bulletproof_data" * 5,  # Make it larger
            public_inputs=[b"public1", b"public2"],
            circuit_id="test_circuit",
            proof_type=ZKPType.BULLETPROOF
        )
        
        result = backend.verify_proof(proof, [b"public1", b"public2"])
        
        assert result.is_success
        assert result.is_valid
        assert result.verification_time > 0
    
    def test_bulletproof_backend_proof_verification_invalid_size(self):
        """Test Bulletproof backend proof verification with invalid size."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        backend.initialize()
        
        # Proof with too small data for Bulletproof
        proof = Proof(
            proof_data=b"short",  # Too short for Bulletproof
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.BULLETPROOF
        )
        
        result = backend.verify_proof(proof, [b"public1"])
        
        assert result.status == ZKPStatus.INVALID_PROOF
        assert "Invalid Bulletproof size" in result.error_message
    
    def test_bulletproof_backend_circuit_info(self):
        """Test Bulletproof backend circuit info."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        backend.initialize()
        
        # Generate a proof to create circuit
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.BULLETPROOF
        )
        backend.generate_proof(request)
        
        # Get circuit info
        info = backend.get_circuit_info("test_circuit")
        
        assert info["circuit_id"] == "test_circuit"
        assert info["exists"] is True
        assert info["constraint_count"] > 0
        assert info["variable_count"] > 0
    
    def test_bulletproof_backend_cleanup(self):
        """Test Bulletproof backend cleanup."""
        config = ZKPConfig(backend_type=ZKPType.BULLETPROOF)
        backend = BulletproofBackend(config)
        backend.initialize()
        
        # Add some circuits
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.BULLETPROOF
        )
        backend.generate_proof(request)
        
        assert len(backend._circuits) > 0
        assert len(backend._generators) > 0
        
        backend.cleanup()
        
        assert not backend.is_initialized
        assert len(backend._circuits) == 0
        assert len(backend._generators) == 0


class TestBackendCommon:
    """Test common backend functionality."""
    
    def test_backend_proof_size_validation(self):
        """Test backend proof size validation."""
        config = ZKPConfig(max_proof_size=100)
        backend = MockZKPBackend(config)
        
        assert backend.validate_proof_size(b"small_proof")
        assert not backend.validate_proof_size(b"x" * 101)
    
    def test_backend_input_validation(self):
        """Test backend input validation."""
        config = ZKPConfig()
        backend = MockZKPBackend(config)
        
        # Valid inputs
        assert backend.validate_inputs([b"input1", b"input2"])
        
        # Empty inputs
        assert not backend.validate_inputs([])
        
        # Empty input
        assert not backend.validate_inputs([b"", b"input2"])
        
        # Too large input
        large_input = b"x" * 1025
        assert not backend.validate_inputs([b"input1", large_input])
    
    def test_backend_config_validation(self):
        """Test backend configuration validation."""
        config = ZKPConfig()
        config.validate()  # Should not raise
        
        # Invalid config
        config.max_proof_size = 0
        with pytest.raises(ValueError):
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__])
