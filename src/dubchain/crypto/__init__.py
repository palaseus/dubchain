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

from .hashing import Hash, SHA256Hasher
from .merkle import MerkleTree
from .signatures import ECDSASigner, PrivateKey, PublicKey, Signature

# GPU-accelerated crypto imports
from .gpu_crypto import GPUCrypto, GPUConfig
from .optimized_gpu_crypto import OptimizedGPUCrypto, OptimizedGPUConfig

# ZKP imports
from .zkp import (
    # Core types
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
    # Backends
    ZKSNARKBackend,
    ZKSTARKBackend, 
    BulletproofBackend,
    MockZKPBackend,
    # Circuits
    ZKCircuit,
    CircuitBuilder,
    ConstraintSystem,
    Witness,
    PublicInputs,
    PrivateInputs,
    # Verification
    ProofVerifier,
    BatchVerifier,
    VerificationCache,
    ReplayProtection,
    # Generation
    ProofGenerator,
    TrustedSetup,
    ProvingKey,
    VerificationKey,
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
