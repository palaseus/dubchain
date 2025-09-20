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

from .core import (
    ZKPBackend,
    ZKPConfig,
    ZKPError,
    ZKPManager,
    Proof,
    ProofRequest,
    ProofResult,
    VerificationResult,
    ZKPType,
    ZKPStatus,
)
from .backends import (
    ZKSNARKBackend,
    ZKSTARKBackend,
    BulletproofBackend,
    MockZKPBackend,
)
from .circuits import (
    ZKCircuit,
    CircuitBuilder,
    ConstraintSystem,
    Witness,
    PublicInputs,
    PrivateInputs,
    ConstraintType,
)
from .verification import (
    ProofVerifier,
    BatchVerifier,
    VerificationCache,
    ReplayProtection,
)
from .generation import (
    ProofGenerator,
    TrustedSetup,
    ProvingKey,
    VerificationKey,
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
