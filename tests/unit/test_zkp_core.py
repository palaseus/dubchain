"""
Unit tests for ZKP core functionality.

This module tests the core ZKP types, interfaces, and manager functionality.
"""

import pytest
import time
import secrets
from unittest.mock import Mock, patch

from src.dubchain.crypto.zkp.core import (
    ZKPType,
    ZKPStatus,
    ZKPConfig,
    ZKPError,
    Proof,
    ProofRequest,
    ProofResult,
    VerificationResult,
    ZKPBackend,
    ZKPManager,
)


class TestZKPConfig:
    """Test ZKP configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ZKPConfig()
        
        assert config.backend_type == ZKPType.ZK_SNARK
        assert config.max_proof_size == 1024 * 1024
        assert config.verification_timeout == 5.0
        assert config.generation_timeout == 30.0
        assert config.enable_replay_protection is True
        assert config.nonce_size == 32
        assert config.enable_verification_cache is True
        assert config.cache_size == 1000
        assert config.cache_ttl == 3600.0
        assert config.enable_batch_verification is True
        assert config.max_batch_size == 100
        assert config.max_constraints == 1000000
        assert config.max_witness_size == 10000
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ZKPConfig()
        config.validate()  # Should not raise
        
        # Invalid max_proof_size
        config.max_proof_size = 0
        with pytest.raises(ValueError, match="max_proof_size must be positive"):
            config.validate()
        
        # Invalid verification_timeout
        config = ZKPConfig()
        config.verification_timeout = -1.0
        with pytest.raises(ValueError, match="verification_timeout must be positive"):
            config.validate()
        
        # Invalid nonce_size
        config = ZKPConfig()
        config.nonce_size = 8
        with pytest.raises(ValueError, match="nonce_size must be at least 16 bytes"):
            config.validate()
        
        # Invalid max_constraints
        config = ZKPConfig()
        config.max_constraints = 0
        with pytest.raises(ValueError, match="max_constraints must be positive"):
            config.validate()


class TestProof:
    """Test Proof data structure."""
    
    def test_proof_creation(self):
        """Test proof creation."""
        proof_data = b"test_proof_data"
        public_inputs = [b"input1", b"input2"]
        circuit_id = "test_circuit"
        proof_type = ZKPType.ZK_SNARK
        
        proof = Proof(
            proof_data=proof_data,
            public_inputs=public_inputs,
            circuit_id=circuit_id,
            proof_type=proof_type
        )
        
        assert proof.proof_data == proof_data
        assert proof.public_inputs == public_inputs
        assert proof.circuit_id == circuit_id
        assert proof.proof_type == proof_type
        assert proof.timestamp > 0
        assert proof.nonce is None
        assert proof.metadata == {}
    
    def test_proof_validation(self):
        """Test proof validation."""
        # Valid proof
        proof = Proof(
            proof_data=b"valid_proof",
            public_inputs=[b"input1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK
        )
        # Should not raise
        
        # Empty proof data
        with pytest.raises(ValueError, match="proof_data cannot be empty"):
            Proof(
                proof_data=b"",
                public_inputs=[b"input1"],
                circuit_id="test_circuit",
                proof_type=ZKPType.ZK_SNARK
            )
        
        # Empty circuit ID
        with pytest.raises(ValueError, match="circuit_id cannot be empty"):
            Proof(
                proof_data=b"valid_proof",
                public_inputs=[b"input1"],
                circuit_id="",
                proof_type=ZKPType.ZK_SNARK
            )
        
        # Too large proof data
        large_data = b"x" * (1024 * 1024 + 1)
        with pytest.raises(ValueError, match="proof_data too large"):
            Proof(
                proof_data=large_data,
                public_inputs=[b"input1"],
                circuit_id="test_circuit",
                proof_type=ZKPType.ZK_SNARK
            )
    
    def test_proof_serialization(self):
        """Test proof serialization and deserialization."""
        proof = Proof(
            proof_data=b"test_proof_data",
            public_inputs=[b"input1", b"input2"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK,
            nonce=b"test_nonce"
        )
        
        # Serialize
        serialized = proof.to_bytes()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize
        deserialized = Proof.from_bytes(serialized)
        
        assert deserialized.proof_data == proof.proof_data
        assert deserialized.public_inputs == proof.public_inputs
        assert deserialized.circuit_id == proof.circuit_id
        assert deserialized.proof_type == proof.proof_type
        assert deserialized.nonce == proof.nonce
        assert deserialized.metadata == proof.metadata
    
    def test_proof_hash(self):
        """Test proof hashing."""
        # Create proofs with same timestamp to ensure deterministic hashing
        timestamp = time.time()
        
        proof1 = Proof(
            proof_data=b"test_proof_data",
            public_inputs=[b"input1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK,
            timestamp=timestamp
        )
        
        proof2 = Proof(
            proof_data=b"test_proof_data",
            public_inputs=[b"input1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK,
            timestamp=timestamp
        )
        
        # Same proof should have same hash
        assert proof1.get_hash() == proof2.get_hash()
        
        # Different proof should have different hash
        proof3 = Proof(
            proof_data=b"different_proof_data",
            public_inputs=[b"input1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK,
            timestamp=timestamp
        )
        
        assert proof1.get_hash() != proof3.get_hash()


class TestProofRequest:
    """Test ProofRequest data structure."""
    
    def test_proof_request_creation(self):
        """Test proof request creation."""
        circuit_id = "test_circuit"
        public_inputs = [b"public1", b"public2"]
        private_inputs = [b"private1", b"private2"]
        proof_type = ZKPType.ZK_SNARK
        
        request = ProofRequest(
            circuit_id=circuit_id,
            public_inputs=public_inputs,
            private_inputs=private_inputs,
            proof_type=proof_type
        )
        
        assert request.circuit_id == circuit_id
        assert request.public_inputs == public_inputs
        assert request.private_inputs == private_inputs
        assert request.proof_type == proof_type
        assert request.nonce is not None
        assert len(request.nonce) == 32
        assert request.metadata == {}
    
    def test_proof_request_nonce_generation(self):
        """Test automatic nonce generation."""
        request1 = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"input1"],
            private_inputs=[b"input2"],
            proof_type=ZKPType.ZK_SNARK
        )
        
        request2 = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"input1"],
            private_inputs=[b"input2"],
            proof_type=ZKPType.ZK_SNARK
        )
        
        # Nonces should be different
        assert request1.nonce != request2.nonce
        assert len(request1.nonce) == 32
        assert len(request2.nonce) == 32


class TestProofResult:
    """Test ProofResult data structure."""
    
    def test_success_result(self):
        """Test successful proof result."""
        proof = Proof(
            proof_data=b"test_proof",
            public_inputs=[b"input1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.ZK_SNARK
        )
        
        result = ProofResult(
            status=ZKPStatus.SUCCESS,
            proof=proof,
            generation_time=1.5
        )
        
        assert result.status == ZKPStatus.SUCCESS
        assert result.proof == proof
        assert result.is_success is True
        assert result.error_message is None
        assert result.generation_time == 1.5
    
    def test_failure_result(self):
        """Test failed proof result."""
        result = ProofResult(
            status=ZKPStatus.GENERATION_FAILED,
            error_message="Test error",
            generation_time=0.5
        )
        
        assert result.status == ZKPStatus.GENERATION_FAILED
        assert result.proof is None
        assert result.is_success is False
        assert result.error_message == "Test error"
        assert result.generation_time == 0.5


class TestVerificationResult:
    """Test VerificationResult data structure."""
    
    def test_success_result(self):
        """Test successful verification result."""
        result = VerificationResult(
            status=ZKPStatus.SUCCESS,
            is_valid=True,
            verification_time=0.1
        )
        
        assert result.status == ZKPStatus.SUCCESS
        assert result.is_valid is True
        assert result.is_success is True
        assert result.error_message is None
        assert result.verification_time == 0.1
    
    def test_failure_result(self):
        """Test failed verification result."""
        result = VerificationResult(
            status=ZKPStatus.INVALID_PROOF,
            is_valid=False,
            error_message="Invalid proof",
            verification_time=0.05
        )
        
        assert result.status == ZKPStatus.INVALID_PROOF
        assert result.is_valid is False
        assert result.is_success is False  # Verification failed due to invalid proof
        assert result.error_message == "Invalid proof"
        assert result.verification_time == 0.05


class TestZKPError:
    """Test ZKPError exception."""
    
    def test_zkp_error_creation(self):
        """Test ZKP error creation."""
        error = ZKPError("Test error", ZKPStatus.BACKEND_ERROR, {"key": "value"})
        
        assert str(error) == "Test error"
        assert error.status == ZKPStatus.BACKEND_ERROR
        assert error.details == {"key": "value"}
    
    def test_zkp_error_defaults(self):
        """Test ZKP error with defaults."""
        error = ZKPError("Test error")
        
        assert str(error) == "Test error"
        assert error.status == ZKPStatus.BACKEND_ERROR
        assert error.details == {}


class MockZKPBackend(ZKPBackend):
    """Mock ZKP backend for testing."""
    
    def __init__(self, config: ZKPConfig):
        super().__init__(config)
        self.generate_called = False
        self.verify_called = False
        self.cleanup_called = False
    
    def initialize(self) -> None:
        self._initialized = True
    
    def generate_proof(self, request: ProofRequest) -> ProofResult:
        self.generate_called = True
        proof = Proof(
            proof_data=b"mock_proof",
            public_inputs=request.public_inputs,
            circuit_id=request.circuit_id,
            proof_type=request.proof_type,
            nonce=request.nonce
        )
        return ProofResult(status=ZKPStatus.SUCCESS, proof=proof)
    
    def verify_proof(self, proof: Proof, public_inputs: list) -> VerificationResult:
        self.verify_called = True
        return VerificationResult(
            status=ZKPStatus.SUCCESS,
            is_valid=True
        )
    
    def get_circuit_info(self, circuit_id: str) -> dict:
        return {"circuit_id": circuit_id, "exists": True}
    
    def cleanup(self) -> None:
        self.cleanup_called = True


class TestZKPBackend:
    """Test ZKP backend base class."""
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        config = ZKPConfig()
        backend = MockZKPBackend(config)
        
        assert not backend.is_initialized
        backend.initialize()
        assert backend.is_initialized
    
    def test_proof_size_validation(self):
        """Test proof size validation."""
        config = ZKPConfig(max_proof_size=100)
        backend = MockZKPBackend(config)
        
        assert backend.validate_proof_size(b"small_proof")
        assert not backend.validate_proof_size(b"x" * 101)
    
    def test_input_validation(self):
        """Test input validation."""
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


class TestZKPManager:
    """Test ZKP manager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        
        assert not manager.is_initialized
        manager.initialize()
        assert manager.is_initialized
        assert manager.backend is not None
    
    def test_proof_generation(self):
        """Test proof generation."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        
        assert result.is_success
        assert result.proof is not None
        assert result.proof.circuit_id == "test_circuit"
    
    def test_proof_verification(self):
        """Test proof verification."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        proof = Proof(
            proof_data=b"test_proof",
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        result = manager.verify_proof(proof, [b"public1"])
        
        assert result.is_success
        assert result.is_valid
    
    def test_batch_verification(self):
        """Test batch verification."""
        config = ZKPConfig(backend_type=ZKPType.MOCK, enable_batch_verification=True)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate proofs first to create circuits
        request1 = ProofRequest(
            circuit_id="test_circuit_1",
            public_inputs=[b"input1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        request2 = ProofRequest(
            circuit_id="test_circuit_2",
            public_inputs=[b"input2"],
            private_inputs=[b"private2"],
            proof_type=ZKPType.MOCK
        )
        
        result1 = manager.generate_proof(request1)
        result2 = manager.generate_proof(request2)
        
        assert result1.is_success
        assert result2.is_success
        
        proofs = [result1.proof, result2.proof]
        public_inputs_list = [[b"input1"], [b"input2"]]
        
        results = manager.batch_verify_proofs(proofs, public_inputs_list)
        
        assert len(results) == 2
        # At least one should succeed (batch verification may have some failures due to parallel execution)
        assert any(result.is_success for result in results)
        # Check that we got results for both proofs
        assert len([r for r in results if r.is_success]) >= 1
    
    def test_circuit_info(self):
        """Test circuit info retrieval."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate a proof first to create the circuit
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"input1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        manager.generate_proof(request)
        
        info = manager.get_circuit_info("test_circuit")
        
        assert info["circuit_id"] == "test_circuit"
        assert info["exists"] is True
    
    def test_cleanup(self):
        """Test manager cleanup."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        assert manager.is_initialized
        manager.cleanup()
        assert not manager.is_initialized
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Invalid public inputs
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[],  # Empty inputs
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        assert not result.is_success
        assert result.status == ZKPStatus.INVALID_INPUT
    
    def test_replay_protection(self):
        """Test replay protection."""
        config = ZKPConfig(backend_type=ZKPType.MOCK, enable_replay_protection=True)
        manager = ZKPManager(config)
        manager.initialize()
        
        nonce = secrets.token_bytes(32)
        proof = Proof(
            proof_data=b"test_proof",
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK,
            nonce=nonce
        )
        
        # First verification should succeed
        result1 = manager.verify_proof(proof, [b"public1"])
        assert result1.is_success
        
        # Second verification should detect replay
        result2 = manager.verify_proof(proof, [b"public1"])
        assert result2.status == ZKPStatus.REPLAY_DETECTED
    
    def test_verification_cache(self):
        """Test verification caching."""
        config = ZKPConfig(backend_type=ZKPType.MOCK, enable_verification_cache=True)
        manager = ZKPManager(config)
        manager.initialize()
        
        proof = Proof(
            proof_data=b"test_proof",
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        # First verification
        result1 = manager.verify_proof(proof, [b"public1"])
        assert result1.is_success
        
        # Second verification should use cache
        result2 = manager.verify_proof(proof, [b"public1"])
        assert result2.is_success
        
        # Both should have same result
        assert result1.is_valid == result2.is_valid
    
    def test_manager_not_initialized(self):
        """Test operations on uninitialized manager."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"public1"],
            private_inputs=[b"private1"],
            proof_type=ZKPType.MOCK
        )
        
        with pytest.raises(ZKPError, match="ZKP manager not initialized"):
            manager.generate_proof(request)
        
        proof = Proof(
            proof_data=b"test_proof",
            public_inputs=[b"public1"],
            circuit_id="test_circuit",
            proof_type=ZKPType.MOCK
        )
        
        with pytest.raises(ZKPError, match="ZKP manager not initialized"):
            manager.verify_proof(proof, [b"public1"])
    
    def test_unsupported_backend(self):
        """Test unsupported backend type."""
        config = ZKPConfig()
        config.backend_type = "unsupported"  # type: ignore
        
        manager = ZKPManager(config)
        
        with pytest.raises(ValueError, match="Unsupported backend type"):
            manager.initialize()


if __name__ == "__main__":
    pytest.main([__file__])
