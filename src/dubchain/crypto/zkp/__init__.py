"""
Zero-Knowledge Proof (ZKP) integration for DubChain.

This module provides a modular ZK-proof verification layer that can be plugged
into existing transaction or authentication flows. It supports both proof
generation and verification with a configurable backend abstraction.

Key Features:
- Modular backend abstraction for different ZK systems (zk-SNARKs, zk-STARKs, Bulletproofs)
- High-performance proof verification optimized for throughput
- Security-first design with adversarial input handling
- Support for proof generation and verification
- Comprehensive testing and benchmarking

Security Considerations:
- All inputs are validated and sanitized
- Malformed proofs are rejected gracefully
- Replay attacks are prevented through nonce mechanisms
- Cryptographic parameters are validated
- Adversarial inputs are handled safely
"""

from .backends import BulletproofBackend, MockZKPBackend, ZKSNARKBackend, ZKSTARKBackend
from .circuits import (
    CircuitBuilder,
    ConstraintSystem,
    ConstraintType,
    PrivateInputs,
    PublicInputs,
    Witness,
    ZKCircuit,
)
from .core import (
    Proof,
    ProofRequest,
    ProofResult,
    VerificationResult,
    ZKPBackend,
    ZKPConfig,
    ZKPError,
    ZKPManager,
    ZKPStatus,
    ZKPType,
)
from .generation import ProofGenerator, ProvingKey, TrustedSetup, VerificationKey
from .verification import (
    BatchVerifier,
    ProofVerifier,
    ReplayProtection,
    VerificationCache,
)

__all__ = [
    # Core types
    "ZKPBackend",
    "ZKPConfig",
    "ZKPError",
    "ZKPManager",
    "Proof",
    "ProofRequest",
    "ProofResult",
    "VerificationResult",
    "ZKPType",
    "ZKPStatus",
    # Backends
    "ZKSNARKBackend",
    "ZKSTARKBackend",
    "BulletproofBackend",
    "MockZKPBackend",
    # Circuits
    "ZKCircuit",
    "CircuitBuilder",
    "ConstraintSystem",
    "Witness",
    "PublicInputs",
    "PrivateInputs",
    "ConstraintType",
    # Verification
    "ProofVerifier",
    "BatchVerifier",
    "VerificationCache",
    "ReplayProtection",
    # Generation
    "ProofGenerator",
    "TrustedSetup",
    "ProvingKey",
    "VerificationKey",
]
