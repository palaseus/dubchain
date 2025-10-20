"""
Cryptographic primitives for DubChain.

This module provides all the cryptographic functions needed for:
- Digital signatures (ECDSA)
- Hash functions (SHA-256)
- Merkle trees
- Key derivation
- Random number generation
- Zero-Knowledge Proofs (ZKP)
"""

# GPU-accelerated crypto imports
from .gpu_crypto import GPUConfig, GPUCrypto
from .hashing import Hash, SHA256Hasher
from .merkle import MerkleTree
from .optimized_gpu_crypto import OptimizedGPUConfig, OptimizedGPUCrypto
from .signatures import ECDSASigner, PrivateKey, PublicKey, Signature

# ZKP imports
from .zkp import (  # Core types; Backends; Circuits; Verification; Generation
    BatchVerifier,
    BulletproofBackend,
    CircuitBuilder,
    ConstraintSystem,
    MockZKPBackend,
    PrivateInputs,
    Proof,
    ProofGenerator,
    ProofRequest,
    ProofResult,
    ProofVerifier,
    ProvingKey,
    PublicInputs,
    ReplayProtection,
    TrustedSetup,
    VerificationCache,
    VerificationKey,
    VerificationResult,
    Witness,
    ZKCircuit,
    ZKPBackend,
    ZKPConfig,
    ZKPError,
    ZKPManager,
    ZKPStatus,
    ZKPType,
    ZKSNARKBackend,
    ZKSTARKBackend,
)

__all__ = [
    # Traditional crypto
    "ECDSASigner",
    "Signature",
    "PublicKey",
    "PrivateKey",
    "SHA256Hasher",
    "Hash",
    "MerkleTree",
    # GPU-accelerated crypto
    "GPUCrypto",
    "GPUConfig",
    "OptimizedGPUCrypto",
    "OptimizedGPUConfig",
    # ZKP Core types
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
    # ZKP Backends
    "ZKSNARKBackend",
    "ZKSTARKBackend",
    "BulletproofBackend",
    "MockZKPBackend",
    # ZKP Circuits
    "ZKCircuit",
    "CircuitBuilder",
    "ConstraintSystem",
    "Witness",
    "PublicInputs",
    "PrivateInputs",
    # ZKP Verification
    "ProofVerifier",
    "BatchVerifier",
    "VerificationCache",
    "ReplayProtection",
    # ZKP Generation
    "ProofGenerator",
    "TrustedSetup",
    "ProvingKey",
    "VerificationKey",
]
