"""
Core ZKP types and interfaces.

This module defines the fundamental types and interfaces for the ZKP system,
including the backend abstraction, configuration, and core data structures.
"""

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
import secrets
import json


class ZKPType(Enum):
    """Types of zero-knowledge proof systems supported."""
    ZK_SNARK = "zk_snark"
    ZK_STARK = "zk_stark" 
    BULLETPROOF = "bulletproof"
    MOCK = "mock"  # For testing


class ZKPStatus(IntEnum):
    """Status codes for ZKP operations."""
    SUCCESS = 0
    INVALID_PROOF = 1
    INVALID_INPUT = 2
    VERIFICATION_FAILED = 3
    GENERATION_FAILED = 4
    BACKEND_ERROR = 5
    TIMEOUT = 6
    REPLAY_DETECTED = 7
    MALFORMED_DATA = 8
    CRYPTOGRAPHIC_ERROR = 9


@dataclass
class ZKPConfig:
    """Configuration for ZKP operations."""
    # Backend configuration
    backend_type: ZKPType = ZKPType.ZK_SNARK
    backend_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    max_proof_size: int = 1024 * 1024  # 1MB max proof size
    verification_timeout: float = 5.0  # 5 seconds max verification time
    generation_timeout: float = 30.0   # 30 seconds max generation time
    
    # Security settings
    enable_replay_protection: bool = True
    max_replay_window: int = 10000  # Max operations to track for replay protection
    nonce_size: int = 32  # Size of nonces for replay protection
    
    # Caching settings
    enable_verification_cache: bool = True
    cache_size: int = 1000
    cache_ttl: float = 3600.0  # 1 hour cache TTL
    
    # Batch processing
    enable_batch_verification: bool = True
    max_batch_size: int = 100
    
    # Circuit settings
    max_constraints: int = 1000000
    max_witness_size: int = 10000
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_proof_size <= 0:
            raise ValueError("max_proof_size must be positive")
        if self.verification_timeout <= 0:
            raise ValueError("verification_timeout must be positive")
        if self.generation_timeout <= 0:
            raise ValueError("generation_timeout must be positive")
        if self.nonce_size < 16:
            raise ValueError("nonce_size must be at least 16 bytes")
        if self.max_constraints <= 0:
            raise ValueError("max_constraints must be positive")
        if self.max_witness_size <= 0:
            raise ValueError("max_witness_size must be positive")


@dataclass
class Proof:
    """Represents a zero-knowledge proof."""
    proof_data: bytes
    public_inputs: List[bytes]
    circuit_id: str
    proof_type: ZKPType
    timestamp: float = field(default_factory=time.time)
    nonce: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate proof data after initialization."""
        if not self.proof_data:
            raise ValueError("proof_data cannot be empty")
        if not self.circuit_id:
            raise ValueError("circuit_id cannot be empty")
        if len(self.proof_data) > 1024 * 1024:  # 1MB limit
            raise ValueError("proof_data too large")
    
    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        data = {
            'proof_data': self.proof_data.hex(),
            'public_inputs': [inp.hex() for inp in self.public_inputs],
            'circuit_id': self.circuit_id,
            'proof_type': self.proof_type.value,
            'timestamp': self.timestamp,
            'nonce': self.nonce.hex() if self.nonce else None,
            'metadata': self.metadata
        }
        return json.dumps(data, sort_keys=True).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Proof':
        """Deserialize proof from bytes."""
        try:
            parsed = json.loads(data.decode('utf-8'))
            return cls(
                proof_data=bytes.fromhex(parsed['proof_data']),
                public_inputs=[bytes.fromhex(inp) for inp in parsed['public_inputs']],
                circuit_id=parsed['circuit_id'],
                proof_type=ZKPType(parsed['proof_type']),
                timestamp=parsed['timestamp'],
                nonce=bytes.fromhex(parsed['nonce']) if parsed['nonce'] else None,
                metadata=parsed['metadata']
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid proof data: {e}")
    
    def get_hash(self) -> str:
        """Get a unique hash for this proof."""
        return hashlib.sha256(self.to_bytes()).hexdigest()


@dataclass
class ProofRequest:
    """Request for proof generation."""
    circuit_id: str
    public_inputs: List[bytes]
    private_inputs: List[bytes]
    proof_type: ZKPType
    nonce: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate nonce if not provided."""
        if self.nonce is None:
            self.nonce = secrets.token_bytes(32)


@dataclass
class ProofResult:
    """Result of proof generation."""
    status: ZKPStatus
    proof: Optional[Proof] = None
    error_message: Optional[str] = None
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if proof generation was successful."""
        return self.status == ZKPStatus.SUCCESS and self.proof is not None


@dataclass
class VerificationResult:
    """Result of proof verification."""
    status: ZKPStatus
    is_valid: bool = False
    error_message: Optional[str] = None
    verification_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if verification was successful."""
        return self.status == ZKPStatus.SUCCESS


class ZKPError(Exception):
    """Base exception for ZKP operations."""
    
    def __init__(self, message: str, status: ZKPStatus = ZKPStatus.BACKEND_ERROR, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


class ZKPBackend(ABC):
    """Abstract base class for ZKP backends."""
    
    def __init__(self, config: ZKPConfig):
        self.config = config
        self.config.validate()
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    def generate_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a zero-knowledge proof."""
        pass
    
    @abstractmethod
    def verify_proof(self, proof: Proof, public_inputs: List[bytes]) -> VerificationResult:
        """Verify a zero-knowledge proof."""
        pass
    
    @abstractmethod
    def get_circuit_info(self, circuit_id: str) -> Dict[str, Any]:
        """Get information about a circuit."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup backend resources."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized
    
    def validate_proof_size(self, proof_data: bytes) -> bool:
        """Validate proof size is within limits."""
        return len(proof_data) <= self.config.max_proof_size
    
    def validate_inputs(self, inputs: List[bytes]) -> bool:
        """Validate input data."""
        if not inputs:
            return False
        for inp in inputs:
            if not inp or len(inp) > 1024:  # 1KB max per input
                return False
        return True


class ZKPManager:
    """Main manager for ZKP operations."""
    
    def __init__(self, config: ZKPConfig):
        self.config = config
        self.backend: Optional[ZKPBackend] = None
        self._replay_protection: Optional['ReplayProtection'] = None
        self._verification_cache: Optional['VerificationCache'] = None
        self._batch_verifier: Optional['BatchVerifier'] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the ZKP manager."""
        if self._initialized:
            return
        
        # Initialize backend
        self.backend = self._create_backend()
        self.backend.initialize()
        
        # Initialize security components
        if self.config.enable_replay_protection:
            from .verification import ReplayProtection
            self._replay_protection = ReplayProtection(self.config.max_replay_window)
        
        if self.config.enable_verification_cache:
            from .verification import VerificationCache
            self._verification_cache = VerificationCache(
                self.config.cache_size, 
                self.config.cache_ttl
            )
        
        if self.config.enable_batch_verification:
            from .verification import BatchVerifier
            self._batch_verifier = BatchVerifier(self.config.max_batch_size)
        
        self._initialized = True
    
    def generate_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a zero-knowledge proof."""
        if not self._initialized:
            raise ZKPError("ZKP manager not initialized")
        
        # Validate request
        if not self.backend.validate_inputs(request.public_inputs):
            return ProofResult(
                status=ZKPStatus.INVALID_INPUT,
                error_message="Invalid public inputs"
            )
        
        if not self.backend.validate_inputs(request.private_inputs):
            return ProofResult(
                status=ZKPStatus.INVALID_INPUT,
                error_message="Invalid private inputs"
            )
        
        # Generate proof
        start_time = time.time()
        try:
            result = self.backend.generate_proof(request)
            result.generation_time = time.time() - start_time
            return result
        except Exception as e:
            return ProofResult(
                status=ZKPStatus.GENERATION_FAILED,
                error_message=f"Proof generation failed: {e}",
                generation_time=time.time() - start_time
            )
    
    def verify_proof(self, proof: Proof, public_inputs: List[bytes]) -> VerificationResult:
        """Verify a zero-knowledge proof."""
        if not self._initialized:
            raise ZKPError("ZKP manager not initialized")
        
        # Check replay protection
        if self._replay_protection and proof.nonce:
            if self._replay_protection.is_replay(proof.nonce):
                return VerificationResult(
                    status=ZKPStatus.REPLAY_DETECTED,
                    error_message="Replay attack detected"
                )
        
        # Check cache
        if self._verification_cache:
            cache_key = self._get_cache_key(proof, public_inputs)
            cached_result = self._verification_cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Verify proof
        start_time = time.time()
        try:
            result = self.backend.verify_proof(proof, public_inputs)
            result.verification_time = time.time() - start_time
            
            # Cache successful verifications
            if result.is_success and self._verification_cache:
                self._verification_cache.set(cache_key, result)
            
            # Record nonce for replay protection
            if result.is_success and self._replay_protection and proof.nonce:
                self._replay_protection.record_nonce(proof.nonce)
            
            return result
        except Exception as e:
            return VerificationResult(
                status=ZKPStatus.VERIFICATION_FAILED,
                error_message=f"Proof verification failed: {e}",
                verification_time=time.time() - start_time
            )
    
    def batch_verify_proofs(self, proofs: List[Proof], 
                           public_inputs_list: List[List[bytes]]) -> List[VerificationResult]:
        """Verify multiple proofs in batch."""
        if not self._initialized:
            raise ZKPError("ZKP manager not initialized")
        
        if len(proofs) != len(public_inputs_list):
            raise ValueError("Number of proofs must match number of public input lists")
        
        if self._batch_verifier:
            return self._batch_verifier.verify_batch(
                self.verify_proof, proofs, public_inputs_list
            )
        else:
            # Fallback to individual verification
            results = []
            for proof, public_inputs in zip(proofs, public_inputs_list):
                results.append(self.verify_proof(proof, public_inputs))
            return results
    
    def get_circuit_info(self, circuit_id: str) -> Dict[str, Any]:
        """Get information about a circuit."""
        if not self._initialized:
            raise ZKPError("ZKP manager not initialized")
        
        return self.backend.get_circuit_info(circuit_id)
    
    def cleanup(self) -> None:
        """Cleanup ZKP manager resources."""
        if self.backend:
            self.backend.cleanup()
        
        if self._verification_cache:
            self._verification_cache.clear()
        
        self._initialized = False
    
    def _create_backend(self) -> ZKPBackend:
        """Create backend based on configuration."""
        if self.config.backend_type == ZKPType.ZK_SNARK:
            from .backends import ZKSNARKBackend
            return ZKSNARKBackend(self.config)
        elif self.config.backend_type == ZKPType.ZK_STARK:
            from .backends import ZKSTARKBackend
            return ZKSTARKBackend(self.config)
        elif self.config.backend_type == ZKPType.BULLETPROOF:
            from .backends import BulletproofBackend
            return BulletproofBackend(self.config)
        elif self.config.backend_type == ZKPType.MOCK:
            from .backends import MockZKPBackend
            return MockZKPBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend type: {self.config.backend_type}")
    
    def _get_cache_key(self, proof: Proof, public_inputs: List[bytes]) -> str:
        """Generate cache key for proof verification."""
        data = proof.get_hash() + "|" + "|".join(inp.hex() for inp in public_inputs)
        return hashlib.sha256(data.encode()).hexdigest()
    
    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized
