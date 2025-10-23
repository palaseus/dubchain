"""
Integration tests for ZKP system.

This module tests the integration between different ZKP components,
end-to-end workflows, and real-world usage scenarios.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from src.dubchain.crypto.zkp.core import (
    ZKPConfig,
    ZKPType,
    ZKPStatus,
    ZKPManager,
    ProofRequest,
    Proof,
)
from src.dubchain.crypto.zkp.circuits import (
    CircuitBuilder,
    ConstraintType,
    PublicInputs,
    PrivateInputs,
)
from src.dubchain.crypto.zkp.generation import (
    TrustedSetup,
    SetupType,
)


class TestZKPIntegration:
    """Test ZKP system integration."""
    
    def test_end_to_end_proof_workflow(self):
        """Test complete proof generation and verification workflow."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Create a proof request
        request = ProofRequest(
            circuit_id="integration_test_circuit",
            public_inputs=[b"public_data_1", b"public_data_2"],
            private_inputs=[b"private_data_1", b"private_data_2"],
            proof_type=ZKPType.MOCK
        )
        
        # Generate proof
        generation_result = manager.generate_proof(request)
        
        assert generation_result.is_success
        assert generation_result.proof is not None
        assert generation_result.generation_time > 0
        
        proof = generation_result.proof
        
        # Verify proof
        verification_result = manager.verify_proof(proof, request.public_inputs)
        
        assert verification_result.is_success
        assert verification_result.is_valid
        assert verification_result.verification_time > 0
        
        # Verify with different public inputs should fail
        wrong_verification = manager.verify_proof(proof, [b"wrong_input"])
        assert not wrong_verification.is_valid
    
    def test_multiple_backend_types(self):
        """Test integration with different backend types."""
        backends = [ZKPType.MOCK, ZKPType.ZK_SNARK, ZKPType.ZK_STARK, ZKPType.BULLETPROOF]
        
        for backend_type in backends:
            config = ZKPConfig(backend_type=backend_type)
            manager = ZKPManager(config)
            manager.initialize()
            
            request = ProofRequest(
                circuit_id=f"test_circuit_{backend_type.value}",
                public_inputs=[b"test_input"],
                private_inputs=[b"test_private"],
                proof_type=backend_type
            )
            
            # Generate proof
            result = manager.generate_proof(request)
            assert result.is_success
            
            # Verify proof
            verify_result = manager.verify_proof(result.proof, request.public_inputs)
            assert verify_result.is_success
            assert verify_result.is_valid
            
            manager.cleanup()
    
    def test_circuit_builder_integration(self):
        """Test integration with circuit builder."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Build a custom circuit
        builder = CircuitBuilder("custom_circuit")
        builder.add_variable("public_var", "bytes", is_public=True)
        builder.add_variable("private_var", "bytes", is_public=False)
        builder.add_equality_constraint("public_var", "private_var", "Test equality")
        
        circuit = builder.build()
        
        # Create inputs that satisfy the constraint
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"same_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"same_data")
        
        # Generate witness
        witness = circuit.generate_witness(public_inputs, private_inputs)
        
        # Verify witness
        assert circuit.verify_witness(witness)
        
        # Test with inputs that don't satisfy constraint
        bad_public_inputs = PublicInputs()
        bad_public_inputs.add_input("public_var", b"different_data")
        
        bad_witness = circuit.generate_witness(bad_public_inputs, private_inputs)
        assert not circuit.verify_witness(bad_witness)
    
    def test_trusted_setup_integration(self):
        """Test integration with trusted setup for zk-SNARKs."""
        config = ZKPConfig(backend_type=ZKPType.ZK_SNARK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Create a trusted setup
        setup = TrustedSetup(
            setup_id="integration_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"mock_proving_key_data",
            verification_key=b"mock_verification_key_data"
        )
        
        assert setup.validate()
        assert not setup.is_expired()
        
        # Test with expired setup
        expired_setup = TrustedSetup(
            setup_id="expired_setup",
            setup_type=SetupType.CIRCUIT_SPECIFIC,
            circuit_id="test_circuit",
            proving_key=b"mock_proving_key_data",
            verification_key=b"mock_verification_key_data",
            expires_at=time.time() - 3600  # Expired
        )
        
        assert expired_setup.is_expired()
    
    def test_batch_operations_integration(self):
        """Test batch proof generation and verification."""
        config = ZKPConfig(
            backend_type=ZKPType.MOCK,
            enable_batch_verification=True,
            max_batch_size=5
        )
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate multiple proofs
        proofs = []
        public_inputs_list = []
        
        for i in range(5):
            request = ProofRequest(
                circuit_id=f"batch_circuit_{i}",
                public_inputs=[f"input_{i}".encode()],
                private_inputs=[f"private_{i}".encode()],
                proof_type=ZKPType.MOCK
            )
            
            result = manager.generate_proof(request)
            assert result.is_success
            
            proofs.append(result.proof)
            public_inputs_list.append(request.public_inputs)
        
        # Batch verify all proofs
        batch_results = manager.batch_verify_proofs(proofs, public_inputs_list)
        
        assert len(batch_results) == 5
        assert all(result.is_success for result in batch_results)
        assert all(result.is_valid for result in batch_results)
    
    def test_caching_integration(self):
        """Test verification caching integration."""
        config = ZKPConfig(
            backend_type=ZKPType.MOCK,
            enable_verification_cache=True,
            enable_replay_protection=False,  # Disable replay protection for caching test
            cache_size=10,
            cache_ttl=1.0  # 1 second TTL
        )
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate a proof
        request = ProofRequest(
            circuit_id="cache_test_circuit",
            public_inputs=[b"cache_input"],
            private_inputs=[b"cache_private"],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        assert result.is_success
        
        proof = result.proof
        
        # First verification (should not be cached)
        start_time = time.time()
        verify1 = manager.verify_proof(proof, request.public_inputs)
        first_time = time.time() - start_time
        
        assert verify1.is_success
        assert verify1.is_valid
        
        # Second verification (should be cached)
        start_time = time.time()
        verify2 = manager.verify_proof(proof, request.public_inputs)
        second_time = time.time() - start_time
        
        assert verify2.is_success
        assert verify2.is_valid
        
        # Cached verification should be faster
        assert second_time < first_time
        
        # Wait for cache expiration
        time.sleep(1.1)
        
        # Third verification (cache should be expired)
        start_time = time.time()
        verify3 = manager.verify_proof(proof, request.public_inputs)
        third_time = time.time() - start_time
        
        assert verify3.is_success
        assert verify3.is_valid
    
    def test_replay_protection_integration(self):
        """Test replay protection integration."""
        config = ZKPConfig(
            backend_type=ZKPType.MOCK,
            enable_replay_protection=True,
            max_replay_window=100
        )
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate a proof with nonce
        request = ProofRequest(
            circuit_id="replay_test_circuit",
            public_inputs=[b"replay_input"],
            private_inputs=[b"replay_private"],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        assert result.is_success
        
        proof = result.proof
        assert proof.nonce is not None
        
        # First verification should succeed
        verify1 = manager.verify_proof(proof, request.public_inputs)
        assert verify1.is_success
        assert verify1.is_valid
        
        # Second verification with same nonce should fail (replay)
        verify2 = manager.verify_proof(proof, request.public_inputs)
        assert verify2.status == ZKPStatus.REPLAY_DETECTED
    
    def test_concurrent_operations(self):
        """Test concurrent proof generation and verification."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        def generate_and_verify(circuit_id: str, input_data: bytes) -> bool:
            """Generate and verify a proof."""
            request = ProofRequest(
                circuit_id=circuit_id,
                public_inputs=[input_data],
                private_inputs=[input_data + b"_private"],
                proof_type=ZKPType.MOCK
            )
            
            # Generate proof
            result = manager.generate_proof(request)
            if not result.is_success:
                return False
            
            # Verify proof
            verify_result = manager.verify_proof(result.proof, request.public_inputs)
            return verify_result.is_success and verify_result.is_valid
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(
                    generate_and_verify,
                    f"concurrent_circuit_{i}",
                    f"input_{i}".encode()
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # All operations should succeed
        assert len(results) == 10
        assert all(results)
    
    def test_error_handling_integration(self):
        """Test error handling across the system."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Test invalid proof request - empty circuit ID should fail during proof generation
        invalid_request = ProofRequest(
            circuit_id="",  # Empty circuit ID
            public_inputs=[b"input"],
            private_inputs=[b"private"],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(invalid_request)
        assert not result.is_success
        assert result.status == ZKPStatus.GENERATION_FAILED
        
        # Test with empty inputs
        empty_request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[],  # Empty inputs
            private_inputs=[b"private"],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(empty_request)
        assert not result.is_success
        assert result.status == ZKPStatus.INVALID_INPUT
        
        # Test with malformed proof
        valid_request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"input"],
            private_inputs=[b"private"],
            proof_type=ZKPType.MOCK
        )
        
        valid_result = manager.generate_proof(valid_request)
        assert valid_result.is_success
        
        # Test malformed proof creation - this should fail during creation
        with pytest.raises(ValueError, match="proof_data cannot be empty"):
            malformed_proof = Proof(
                proof_data=b"",  # Empty proof data
                public_inputs=[b"input"],
                circuit_id="test_circuit",
                proof_type=ZKPType.MOCK
            )
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the system."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Test proof generation performance
        generation_times = []
        for i in range(10):
            request = ProofRequest(
                circuit_id=f"perf_circuit_{i}",
                public_inputs=[f"input_{i}".encode()],
                private_inputs=[f"private_{i}".encode()],
                proof_type=ZKPType.MOCK
            )
            
            start_time = time.time()
            result = manager.generate_proof(request)
            generation_time = time.time() - start_time
            
            assert result.is_success
            generation_times.append(generation_time)
        
        # Generation should be reasonably fast
        avg_generation_time = sum(generation_times) / len(generation_times)
        assert avg_generation_time < 1.0  # Less than 1 second on average
        
        # Test verification performance
        verification_times = []
        for i in range(10):
            request = ProofRequest(
                circuit_id=f"verify_circuit_{i}",
                public_inputs=[f"input_{i}".encode()],
                private_inputs=[f"private_{i}".encode()],
                proof_type=ZKPType.MOCK
            )
            
            result = manager.generate_proof(request)
            assert result.is_success
            
            start_time = time.time()
            verify_result = manager.verify_proof(result.proof, request.public_inputs)
            verification_time = time.time() - start_time
            
            assert verify_result.is_success
            verification_times.append(verification_time)
        
        # Verification should be very fast
        avg_verification_time = sum(verification_times) / len(verification_times)
        assert avg_verification_time < 0.1  # Less than 100ms on average
    
    def test_system_cleanup(self):
        """Test proper system cleanup."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Generate some proofs to create state
        for i in range(5):
            request = ProofRequest(
                circuit_id=f"cleanup_circuit_{i}",
                public_inputs=[f"input_{i}".encode()],
                private_inputs=[f"private_{i}".encode()],
                proof_type=ZKPType.MOCK
            )
            
            result = manager.generate_proof(request)
            assert result.is_success
        
        # Verify system is working
        assert manager.is_initialized
        
        # Cleanup
        manager.cleanup()
        
        # Verify system is cleaned up
        assert not manager.is_initialized
        
        # Operations should fail after cleanup
        request = ProofRequest(
            circuit_id="test_circuit",
            public_inputs=[b"input"],
            private_inputs=[b"private"],
            proof_type=ZKPType.MOCK
        )
        
        with pytest.raises(Exception):  # Should raise ZKPError
            manager.generate_proof(request)


class TestZKPRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_authentication_workflow(self):
        """Test ZKP-based authentication workflow."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Simulate user authentication
        user_id = b"user_123"
        secret = b"secret_password"
        
        # Build authentication circuit
        builder = CircuitBuilder("auth_circuit")
        builder.add_variable("user_id", "bytes", is_public=True)
        builder.add_variable("secret", "bytes", is_public=False)
        builder.add_variable("secret_hash", "bytes", is_public=False)
        builder.add_hash_constraint("secret", "secret_hash", "sha256", "Hash secret")
        
        circuit = builder.build()
        
        # Generate authentication proof
        import hashlib
        secret_hash = hashlib.sha256(secret).digest()
        
        public_inputs = PublicInputs()
        public_inputs.add_input("user_id", user_id)
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("secret", secret)
        private_inputs.add_input("secret_hash", secret_hash)
        
        witness = circuit.generate_witness(public_inputs, private_inputs)
        assert circuit.verify_witness(witness)
        
        # In a real system, this witness would be used to generate a ZKP
        # For this test, we'll simulate the proof generation
        request = ProofRequest(
            circuit_id="auth_circuit",
            public_inputs=[user_id],
            private_inputs=[secret, secret_hash],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        assert result.is_success
        
        # Verify authentication proof
        verify_result = manager.verify_proof(result.proof, [user_id])
        assert verify_result.is_success
        assert verify_result.is_valid
    
    def test_privacy_preserving_transaction(self):
        """Test privacy-preserving transaction scenario."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Simulate a transaction where we prove we have enough balance
        # without revealing the actual balance
        sender = b"sender_address"
        amount = 100
        balance = 500
        
        # Build transaction circuit
        builder = CircuitBuilder("transaction_circuit")
        builder.add_variable("sender", "bytes", is_public=True)
        builder.add_variable("amount", "int", is_public=True)
        builder.add_variable("balance", "int", is_public=False)
        builder.add_range_constraint("amount", 0, 10000, "Valid amount")
        builder.add_range_constraint("balance", amount, 1000000, "Sufficient balance")
        
        circuit = builder.build()
        
        # Generate transaction proof
        public_inputs = PublicInputs()
        public_inputs.add_input("sender", sender)
        public_inputs.add_input("amount", amount.to_bytes(8, 'little'))
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("balance", balance.to_bytes(8, 'little'))
        
        witness = circuit.generate_witness(public_inputs, private_inputs)
        assert circuit.verify_witness(witness)
        
        # Generate proof
        request = ProofRequest(
            circuit_id="transaction_circuit",
            public_inputs=[sender, amount.to_bytes(8, 'little')],
            private_inputs=[balance.to_bytes(8, 'little')],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        assert result.is_success
        
        # Verify transaction proof
        verify_result = manager.verify_proof(result.proof, [sender, amount.to_bytes(8, 'little')])
        assert verify_result.is_success
        assert verify_result.is_valid
    
    def test_membership_proof(self):
        """Test membership proof scenario."""
        config = ZKPConfig(backend_type=ZKPType.MOCK)
        manager = ZKPManager(config)
        manager.initialize()
        
        # Simulate proving membership in a set without revealing which element
        merkle_root = b"merkle_root_hash"
        element = b"secret_element"
        merkle_proof = [b"proof_hash_1", b"proof_hash_2", b"proof_hash_3"]
        
        # Build membership circuit
        builder = CircuitBuilder("membership_circuit")
        builder.add_variable("merkle_root", "bytes", is_public=True)
        builder.add_variable("element", "bytes", is_public=False)
        builder.add_variable("merkle_proof", "bytes", is_public=False)
        
        circuit = builder.build()
        
        # Generate membership proof
        public_inputs = PublicInputs()
        public_inputs.add_input("merkle_root", merkle_root)
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("element", element)
        private_inputs.add_input("merkle_proof", b"".join(merkle_proof))
        
        witness = circuit.generate_witness(public_inputs, private_inputs)
        assert circuit.verify_witness(witness)
        
        # Generate proof
        request = ProofRequest(
            circuit_id="membership_circuit",
            public_inputs=[merkle_root],
            private_inputs=[element, b"".join(merkle_proof)],
            proof_type=ZKPType.MOCK
        )
        
        result = manager.generate_proof(request)
        assert result.is_success
        
        # Verify membership proof
        verify_result = manager.verify_proof(result.proof, [merkle_root])
        assert verify_result.is_success
        assert verify_result.is_valid


if __name__ == "__main__":
    pytest.main([__file__])
