"""
ZKP backend implementations.

This module provides concrete implementations of ZKP backends for different
proof systems (zk-SNARKs, zk-STARKs, Bulletproofs, and mock for testing).
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from .circuits import PrivateInputs, PublicInputs, Witness, ZKCircuit
from .core import (
    Proof,
    ProofRequest,
    ProofResult,
    VerificationResult,
    ZKPBackend,
    ZKPConfig,
    ZKPStatus,
    ZKPType,
)
from .generation import (
    ProvingKey,
    SetupType,
    TrustedSetup,
    VerificationKey,
    create_proof_generator,
)
from .verification import ProofVerifier


class MockZKPBackend(ZKPBackend):
    """Mock ZKP backend for testing and development."""

    def __init__(self, config: ZKPConfig):
        super().__init__(config)
        self._circuits: Dict[str, ZKCircuit] = {}
        self._generators: Dict[str, Any] = {}
        self._verifier = ProofVerifier(
            {
                "max_proof_size": config.max_proof_size,
                "verification_timeout": config.verification_timeout,
                "max_input_size": 1024,
                "max_input_count": 100,
            }
        )

    def initialize(self) -> None:
        """Initialize the mock backend."""
        self._initialized = True

    def generate_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a mock proof."""
        if not self._initialized:
            return ProofResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate inputs
            if not self.validate_inputs(request.public_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid public inputs",
                )

            if not self.validate_inputs(request.private_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid private inputs",
                )

            # Create mock circuit if not exists
            if request.circuit_id not in self._circuits:
                from .circuits import BuiltCircuit, CircuitBuilder

                builder = CircuitBuilder(request.circuit_id)
                # Add basic constraints
                for i in range(len(request.public_inputs)):
                    builder.add_variable(f"public_{i}", "bytes", is_public=True)
                for i in range(len(request.private_inputs)):
                    builder.add_variable(f"private_{i}", "bytes", is_public=False)
                self._circuits[request.circuit_id] = builder.build()

            circuit = self._circuits[request.circuit_id]

            # Create witness
            public_inputs = PublicInputs()
            for i, inp in enumerate(request.public_inputs):
                public_inputs.add_input(f"public_{i}", inp)

            private_inputs = PrivateInputs()
            for i, inp in enumerate(request.private_inputs):
                private_inputs.add_input(f"private_{i}", inp)

            witness = circuit.generate_witness(public_inputs, private_inputs)

            # Generate mock proof (avoid null bytes)
            proof_data = secrets.token_bytes(128)
            # Ensure no null bytes in proof data
            proof_data = proof_data.replace(b"\x00", b"\x01")

            # Add deterministic elements
            witness_hash = hashlib.sha256(witness.to_bytes()).digest()
            # Ensure witness hash doesn't contain null bytes
            witness_hash = witness_hash.replace(b"\x00", b"\x01")
            proof_data = witness_hash + proof_data[32:]

            proof = Proof(
                proof_data=proof_data,
                public_inputs=request.public_inputs,
                circuit_id=request.circuit_id,
                proof_type=request.proof_type,
                nonce=request.nonce,
                metadata={"generator": "mock"},
            )

            return ProofResult(
                status=ZKPStatus.SUCCESS,
                proof=proof,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            return ProofResult(
                status=ZKPStatus.GENERATION_FAILED,
                error_message=f"Proof generation failed: {e}",
                generation_time=time.time() - start_time,
            )

    def verify_proof(
        self, proof: Proof, public_inputs: List[bytes]
    ) -> VerificationResult:
        """Verify a mock proof."""
        if not self._initialized:
            return VerificationResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate proof format
            is_valid, error_msg = self._verifier.validate_proof_format(proof)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Validate public inputs
            is_valid, error_msg = self._verifier.validate_public_inputs(public_inputs)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.INVALID_INPUT, error_message=error_msg
                )

            # Check for malformed data
            is_valid, error_msg = self._verifier.detect_malformed_data(
                proof, public_inputs
            )
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Mock verification - check if proof data looks valid
            if len(proof.proof_data) < 8:  # Reduced minimum size for testing
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Proof data too short",
                    verification_time=time.time() - start_time,
                )

            # Check if public inputs match
            if proof.public_inputs != public_inputs:
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Public inputs don't match",
                    verification_time=time.time() - start_time,
                )

            # Mock verification success
            return VerificationResult(
                status=ZKPStatus.SUCCESS,
                is_valid=True,
                verification_time=time.time() - start_time,
            )

        except Exception as e:
            return VerificationResult(
                status=ZKPStatus.VERIFICATION_FAILED,
                error_message=f"Proof verification failed: {e}",
                verification_time=time.time() - start_time,
            )

    def get_circuit_info(self, circuit_id: str) -> Dict[str, Any]:
        """Get circuit information."""
        if circuit_id in self._circuits:
            info = self._circuits[circuit_id].get_circuit_info()
            info["exists"] = True
            return info
        else:
            return {
                "circuit_id": circuit_id,
                "exists": False,
                "constraint_count": 0,
                "variable_count": 0,
            }

    def cleanup(self) -> None:
        """Cleanup backend resources."""
        self._circuits.clear()
        self._generators.clear()
        self._initialized = False


class ZKSNARKBackend(ZKPBackend):
    """Backend for zk-SNARK proofs."""

    def __init__(self, config: ZKPConfig):
        super().__init__(config)
        self._circuits: Dict[str, ZKCircuit] = {}
        self._setups: Dict[str, TrustedSetup] = {}
        self._generators: Dict[str, Any] = {}
        self._verifier = ProofVerifier(
            {
                "max_proof_size": config.max_proof_size,
                "verification_timeout": config.verification_timeout,
                "max_input_size": 1024,
                "max_input_count": 100,
            }
        )

    def initialize(self) -> None:
        """Initialize the zk-SNARK backend."""
        self._initialized = True

    def generate_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a zk-SNARK proof."""
        if not self._initialized:
            return ProofResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate inputs
            if not self.validate_inputs(request.public_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid public inputs",
                )

            if not self.validate_inputs(request.private_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid private inputs",
                )

            # Get or create circuit
            circuit = self._get_or_create_circuit(
                request.circuit_id, request.public_inputs, request.private_inputs
            )

            # Get or create trusted setup
            setup = self._get_or_create_setup(request.circuit_id, circuit)

            # Get or create generator
            generator = self._get_or_create_generator(
                request.circuit_id, circuit, setup
            )

            # Create witness
            public_inputs = PublicInputs()
            for i, inp in enumerate(request.public_inputs):
                public_inputs.add_input(f"public_{i}", inp)

            private_inputs = PrivateInputs()
            for i, inp in enumerate(request.private_inputs):
                private_inputs.add_input(f"private_{i}", inp)

            witness = circuit.generate_witness(public_inputs, private_inputs)

            # Generate proof
            proof_data, metadata = generator.generate_proof(witness, public_inputs)

            proof = Proof(
                proof_data=proof_data,
                public_inputs=request.public_inputs,
                circuit_id=request.circuit_id,
                proof_type=request.proof_type,
                nonce=request.nonce,
                metadata=metadata,
            )

            return ProofResult(
                status=ZKPStatus.SUCCESS,
                proof=proof,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            return ProofResult(
                status=ZKPStatus.GENERATION_FAILED,
                error_message=f"Proof generation failed: {e}",
                generation_time=time.time() - start_time,
            )

    def verify_proof(
        self, proof: Proof, public_inputs: List[bytes]
    ) -> VerificationResult:
        """Verify a zk-SNARK proof."""
        if not self._initialized:
            return VerificationResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate proof format
            is_valid, error_msg = self._verifier.validate_proof_format(proof)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Validate public inputs
            is_valid, error_msg = self._verifier.validate_public_inputs(public_inputs)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.INVALID_INPUT, error_message=error_msg
                )

            # Check for malformed data
            is_valid, error_msg = self._verifier.detect_malformed_data(
                proof, public_inputs
            )
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Mock zk-SNARK verification
            # In a real implementation, this would use the verification key
            # and perform actual SNARK verification

            if len(proof.proof_data) < 64:  # SNARK proofs are typically larger
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Invalid SNARK proof size",
                    verification_time=time.time() - start_time,
                )

            # Check if public inputs match
            if proof.public_inputs != public_inputs:
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Public inputs don't match",
                    verification_time=time.time() - start_time,
                )

            # Mock verification success
            return VerificationResult(
                status=ZKPStatus.SUCCESS,
                is_valid=True,
                verification_time=time.time() - start_time,
            )

        except Exception as e:
            return VerificationResult(
                status=ZKPStatus.VERIFICATION_FAILED,
                error_message=f"Proof verification failed: {e}",
                verification_time=time.time() - start_time,
            )

    def get_circuit_info(self, circuit_id: str) -> Dict[str, Any]:
        """Get circuit information."""
        if circuit_id in self._circuits:
            info = self._circuits[circuit_id].get_circuit_info()
            info["exists"] = True
            return info
        else:
            return {
                "circuit_id": circuit_id,
                "exists": False,
                "constraint_count": 0,
                "variable_count": 0,
            }

    def cleanup(self) -> None:
        """Cleanup backend resources."""
        self._circuits.clear()
        self._setups.clear()
        self._generators.clear()
        self._initialized = False

    def _get_or_create_circuit(
        self, circuit_id: str, public_inputs: List[bytes], private_inputs: List[bytes]
    ) -> ZKCircuit:
        """Get or create a circuit."""
        if circuit_id not in self._circuits:
            from .circuits import BuiltCircuit, CircuitBuilder

            builder = CircuitBuilder(circuit_id)

            # Add variables
            for i in range(len(public_inputs)):
                builder.add_variable(f"public_{i}", "bytes", is_public=True)
            for i in range(len(private_inputs)):
                builder.add_variable(f"private_{i}", "bytes", is_public=False)

            # Add basic constraints
            for i in range(len(public_inputs)):
                builder.add_range_constraint(f"public_{i}", 0, 2**256 - 1)
            for i in range(len(private_inputs)):
                builder.add_range_constraint(f"private_{i}", 0, 2**256 - 1)

            self._circuits[circuit_id] = builder.build()

        return self._circuits[circuit_id]

    def _get_or_create_setup(self, circuit_id: str, circuit: ZKCircuit) -> TrustedSetup:
        """Get or create a trusted setup."""
        if circuit_id not in self._setups:
            # Create mock trusted setup
            setup_id = f"setup_{circuit_id}_{int(time.time())}"

            # Generate mock keys
            proving_key = secrets.token_bytes(512)
            verification_key = secrets.token_bytes(256)

            setup = TrustedSetup(
                setup_id=setup_id,
                setup_type=SetupType.CIRCUIT_SPECIFIC,
                circuit_id=circuit_id,
                proving_key=proving_key,
                verification_key=verification_key,
                setup_parameters={"mock": True},
            )

            self._setups[circuit_id] = setup

        return self._setups[circuit_id]

    def _get_or_create_generator(
        self, circuit_id: str, circuit: ZKCircuit, setup: TrustedSetup
    ) -> Any:
        """Get or create a proof generator."""
        if circuit_id not in self._generators:
            generator = create_proof_generator(circuit, ZKPType.ZK_SNARK, setup)
            generator.initialize()
            self._generators[circuit_id] = generator

        return self._generators[circuit_id]


class ZKSTARKBackend(ZKPBackend):
    """Backend for zk-STARK proofs."""

    def __init__(self, config: ZKPConfig):
        super().__init__(config)
        self._circuits: Dict[str, ZKCircuit] = {}
        self._generators: Dict[str, Any] = {}
        self._verifier = ProofVerifier(
            {
                "max_proof_size": config.max_proof_size,
                "verification_timeout": config.verification_timeout,
                "max_input_size": 1024,
                "max_input_count": 100,
            }
        )

    def initialize(self) -> None:
        """Initialize the zk-STARK backend."""
        self._initialized = True

    def generate_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a zk-STARK proof."""
        if not self._initialized:
            return ProofResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate inputs
            if not self.validate_inputs(request.public_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid public inputs",
                )

            if not self.validate_inputs(request.private_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid private inputs",
                )

            # Get or create circuit
            circuit = self._get_or_create_circuit(
                request.circuit_id, request.public_inputs, request.private_inputs
            )

            # Get or create generator (STARKs don't need trusted setup)
            generator = self._get_or_create_generator(request.circuit_id, circuit)

            # Create witness
            public_inputs = PublicInputs()
            for i, inp in enumerate(request.public_inputs):
                public_inputs.add_input(f"public_{i}", inp)

            private_inputs = PrivateInputs()
            for i, inp in enumerate(request.private_inputs):
                private_inputs.add_input(f"private_{i}", inp)

            witness = circuit.generate_witness(public_inputs, private_inputs)

            # Generate proof
            proof_data, metadata = generator.generate_proof(witness, public_inputs)

            proof = Proof(
                proof_data=proof_data,
                public_inputs=request.public_inputs,
                circuit_id=request.circuit_id,
                proof_type=request.proof_type,
                nonce=request.nonce,
                metadata=metadata,
            )

            return ProofResult(
                status=ZKPStatus.SUCCESS,
                proof=proof,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            return ProofResult(
                status=ZKPStatus.GENERATION_FAILED,
                error_message=f"Proof generation failed: {e}",
                generation_time=time.time() - start_time,
            )

    def verify_proof(
        self, proof: Proof, public_inputs: List[bytes]
    ) -> VerificationResult:
        """Verify a zk-STARK proof."""
        if not self._initialized:
            return VerificationResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate proof format
            is_valid, error_msg = self._verifier.validate_proof_format(proof)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Validate public inputs
            is_valid, error_msg = self._verifier.validate_public_inputs(public_inputs)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.INVALID_INPUT, error_message=error_msg
                )

            # Check for malformed data
            is_valid, error_msg = self._verifier.detect_malformed_data(
                proof, public_inputs
            )
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Mock zk-STARK verification
            # In a real implementation, this would perform actual STARK verification

            if len(proof.proof_data) < 128:  # STARK proofs are typically larger
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Invalid STARK proof size",
                    verification_time=time.time() - start_time,
                )

            # Check if public inputs match
            if proof.public_inputs != public_inputs:
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Public inputs don't match",
                    verification_time=time.time() - start_time,
                )

            # Mock verification success
            return VerificationResult(
                status=ZKPStatus.SUCCESS,
                is_valid=True,
                verification_time=time.time() - start_time,
            )

        except Exception as e:
            return VerificationResult(
                status=ZKPStatus.VERIFICATION_FAILED,
                error_message=f"Proof verification failed: {e}",
                verification_time=time.time() - start_time,
            )

    def get_circuit_info(self, circuit_id: str) -> Dict[str, Any]:
        """Get circuit information."""
        if circuit_id in self._circuits:
            info = self._circuits[circuit_id].get_circuit_info()
            info["exists"] = True
            return info
        else:
            return {
                "circuit_id": circuit_id,
                "exists": False,
                "constraint_count": 0,
                "variable_count": 0,
            }

    def cleanup(self) -> None:
        """Cleanup backend resources."""
        self._circuits.clear()
        self._generators.clear()
        self._initialized = False

    def _get_or_create_circuit(
        self, circuit_id: str, public_inputs: List[bytes], private_inputs: List[bytes]
    ) -> ZKCircuit:
        """Get or create a circuit."""
        if circuit_id not in self._circuits:
            from .circuits import BuiltCircuit, CircuitBuilder

            builder = CircuitBuilder(circuit_id)

            # Add variables
            for i in range(len(public_inputs)):
                builder.add_variable(f"public_{i}", "bytes", is_public=True)
            for i in range(len(private_inputs)):
                builder.add_variable(f"private_{i}", "bytes", is_public=False)

            # Add basic constraints
            for i in range(len(public_inputs)):
                builder.add_range_constraint(f"public_{i}", 0, 2**256 - 1)
            for i in range(len(private_inputs)):
                builder.add_range_constraint(f"private_{i}", 0, 2**256 - 1)

            self._circuits[circuit_id] = builder.build()

        return self._circuits[circuit_id]

    def _get_or_create_generator(self, circuit_id: str, circuit: ZKCircuit) -> Any:
        """Get or create a proof generator."""
        if circuit_id not in self._generators:
            generator = create_proof_generator(circuit, ZKPType.ZK_STARK)
            generator.initialize()
            self._generators[circuit_id] = generator

        return self._generators[circuit_id]


class BulletproofBackend(ZKPBackend):
    """Backend for Bulletproofs."""

    def __init__(self, config: ZKPConfig):
        super().__init__(config)
        self._circuits: Dict[str, ZKCircuit] = {}
        self._generators: Dict[str, Any] = {}
        self._verifier = ProofVerifier(
            {
                "max_proof_size": config.max_proof_size,
                "verification_timeout": config.verification_timeout,
                "max_input_size": 1024,
                "max_input_count": 100,
            }
        )

    def initialize(self) -> None:
        """Initialize the Bulletproof backend."""
        self._initialized = True

    def generate_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a Bulletproof."""
        if not self._initialized:
            return ProofResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate inputs
            if not self.validate_inputs(request.public_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid public inputs",
                )

            if not self.validate_inputs(request.private_inputs):
                return ProofResult(
                    status=ZKPStatus.INVALID_INPUT,
                    error_message="Invalid private inputs",
                )

            # Get or create circuit
            circuit = self._get_or_create_circuit(
                request.circuit_id, request.public_inputs, request.private_inputs
            )

            # Get or create generator (Bulletproofs don't need trusted setup)
            generator = self._get_or_create_generator(request.circuit_id, circuit)

            # Create witness
            public_inputs = PublicInputs()
            for i, inp in enumerate(request.public_inputs):
                public_inputs.add_input(f"public_{i}", inp)

            private_inputs = PrivateInputs()
            for i, inp in enumerate(request.private_inputs):
                private_inputs.add_input(f"private_{i}", inp)

            witness = circuit.generate_witness(public_inputs, private_inputs)

            # Generate proof
            proof_data, metadata = generator.generate_proof(witness, public_inputs)

            proof = Proof(
                proof_data=proof_data,
                public_inputs=request.public_inputs,
                circuit_id=request.circuit_id,
                proof_type=request.proof_type,
                nonce=request.nonce,
                metadata=metadata,
            )

            return ProofResult(
                status=ZKPStatus.SUCCESS,
                proof=proof,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            return ProofResult(
                status=ZKPStatus.GENERATION_FAILED,
                error_message=f"Proof generation failed: {e}",
                generation_time=time.time() - start_time,
            )

    def verify_proof(
        self, proof: Proof, public_inputs: List[bytes]
    ) -> VerificationResult:
        """Verify a Bulletproof."""
        if not self._initialized:
            return VerificationResult(
                status=ZKPStatus.BACKEND_ERROR, error_message="Backend not initialized"
            )

        start_time = time.time()

        try:
            # Validate proof format
            is_valid, error_msg = self._verifier.validate_proof_format(proof)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Validate public inputs
            is_valid, error_msg = self._verifier.validate_public_inputs(public_inputs)
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.INVALID_INPUT, error_message=error_msg
                )

            # Check for malformed data
            is_valid, error_msg = self._verifier.detect_malformed_data(
                proof, public_inputs
            )
            if not is_valid:
                return VerificationResult(
                    status=ZKPStatus.MALFORMED_DATA, error_message=error_msg
                )

            # Mock Bulletproof verification
            # In a real implementation, this would perform actual Bulletproof verification

            if len(proof.proof_data) < 32:  # Bulletproofs are compact
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Invalid Bulletproof size",
                    verification_time=time.time() - start_time,
                )

            # Check if public inputs match
            if proof.public_inputs != public_inputs:
                return VerificationResult(
                    status=ZKPStatus.INVALID_PROOF,
                    error_message="Public inputs don't match",
                    verification_time=time.time() - start_time,
                )

            # Mock verification success
            return VerificationResult(
                status=ZKPStatus.SUCCESS,
                is_valid=True,
                verification_time=time.time() - start_time,
            )

        except Exception as e:
            return VerificationResult(
                status=ZKPStatus.VERIFICATION_FAILED,
                error_message=f"Proof verification failed: {e}",
                verification_time=time.time() - start_time,
            )

    def get_circuit_info(self, circuit_id: str) -> Dict[str, Any]:
        """Get circuit information."""
        if circuit_id in self._circuits:
            info = self._circuits[circuit_id].get_circuit_info()
            info["exists"] = True
            return info
        else:
            return {
                "circuit_id": circuit_id,
                "exists": False,
                "constraint_count": 0,
                "variable_count": 0,
            }

    def cleanup(self) -> None:
        """Cleanup backend resources."""
        self._circuits.clear()
        self._generators.clear()
        self._initialized = False

    def _get_or_create_circuit(
        self, circuit_id: str, public_inputs: List[bytes], private_inputs: List[bytes]
    ) -> ZKCircuit:
        """Get or create a circuit."""
        if circuit_id not in self._circuits:
            from .circuits import BuiltCircuit, CircuitBuilder

            builder = CircuitBuilder(circuit_id)

            # Add variables
            for i in range(len(public_inputs)):
                builder.add_variable(f"public_{i}", "bytes", is_public=True)
            for i in range(len(private_inputs)):
                builder.add_variable(f"private_{i}", "bytes", is_public=False)

            # Add basic constraints
            for i in range(len(public_inputs)):
                builder.add_range_constraint(f"public_{i}", 0, 2**256 - 1)
            for i in range(len(private_inputs)):
                builder.add_range_constraint(f"private_{i}", 0, 2**256 - 1)

            self._circuits[circuit_id] = builder.build()

        return self._circuits[circuit_id]

    def _get_or_create_generator(self, circuit_id: str, circuit: ZKCircuit) -> Any:
        """Get or create a proof generator."""
        if circuit_id not in self._generators:
            generator = create_proof_generator(circuit, ZKPType.BULLETPROOF)
            generator.initialize()
            self._generators[circuit_id] = generator

        return self._generators[circuit_id]
