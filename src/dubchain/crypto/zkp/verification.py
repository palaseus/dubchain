"""
ZKP verification components.

This module provides proof verification, caching, replay protection,
and batch verification capabilities.
"""

import hashlib
import time
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import Proof, VerificationResult, ZKPStatus


@dataclass
class CacheEntry:
    """Entry in the verification cache."""
    result: VerificationResult
    timestamp: float
    access_count: int = 0
    
    def is_expired(self, ttl: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > ttl


class VerificationCache:
    """Cache for proof verification results."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[VerificationResult]:
        """Get a cached verification result."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired(self.ttl):
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    entry.access_count += 1
                    self._hits += 1
                    return entry.result
                else:
                    # Remove expired entry
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, result: VerificationResult) -> None:
        """Cache a verification result."""
        with self._lock:
            # Remove oldest entries if cache is full
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            entry = CacheEntry(result, time.time())
            self._cache[key] = entry
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class ReplayProtection:
    """Protection against replay attacks using nonces."""
    
    def __init__(self, max_window_size: int = 10000):
        self.max_window_size = max_window_size
        self._nonces: Set[bytes] = set()
        self._nonce_queue: deque = deque()
        self._lock = threading.RLock()
    
    def is_replay(self, nonce: bytes) -> bool:
        """Check if a nonce is a replay."""
        with self._lock:
            return nonce in self._nonces
    
    def record_nonce(self, nonce: bytes) -> None:
        """Record a nonce to prevent replay."""
        with self._lock:
            # Add to set and queue
            self._nonces.add(nonce)
            self._nonce_queue.append(nonce)
            
            # Remove oldest nonces if window is full
            while len(self._nonce_queue) > self.max_window_size:
                old_nonce = self._nonce_queue.popleft()
                self._nonces.discard(old_nonce)
    
    def clear(self) -> None:
        """Clear all recorded nonces."""
        with self._lock:
            self._nonces.clear()
            self._nonce_queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replay protection statistics."""
        with self._lock:
            return {
                'recorded_nonces': len(self._nonces),
                'max_window_size': self.max_window_size
            }


class BatchVerifier:
    """Batch verifier for multiple proofs."""
    
    def __init__(self, max_batch_size: int = 100, max_workers: int = 4):
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
    
    def verify_batch(self, verify_func: Callable[[Proof, List[bytes]], VerificationResult],
                    proofs: List[Proof], 
                    public_inputs_list: List[List[bytes]]) -> List[VerificationResult]:
        """Verify multiple proofs in batch."""
        if len(proofs) != len(public_inputs_list):
            raise ValueError("Number of proofs must match number of public input lists")
        
        if len(proofs) == 0:
            return []
        
        # Split into batches if necessary
        batches = []
        for i in range(0, len(proofs), self.max_batch_size):
            batch_proofs = proofs[i:i + self.max_batch_size]
            batch_inputs = public_inputs_list[i:i + self.max_batch_size]
            batches.append((batch_proofs, batch_inputs))
        
        results = []
        for batch_proofs, batch_inputs in batches:
            batch_results = self._verify_batch_parallel(verify_func, batch_proofs, batch_inputs)
            results.extend(batch_results)
        
        return results
    
    def _verify_batch_parallel(self, verify_func: Callable[[Proof, List[bytes]], VerificationResult],
                              proofs: List[Proof], 
                              public_inputs_list: List[List[bytes]]) -> List[VerificationResult]:
        """Verify a batch of proofs in parallel."""
        results = [None] * len(proofs)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all verification tasks
            future_to_index = {}
            for i, (proof, public_inputs) in enumerate(zip(proofs, public_inputs_list)):
                future = executor.submit(verify_func, proof, public_inputs)
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    results[index] = VerificationResult(
                        status=ZKPStatus.VERIFICATION_FAILED,
                        error_message=f"Batch verification failed: {e}"
                    )
        
        return results


class ProofVerifier:
    """Main proof verifier with security checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_proof_size = config.get('max_proof_size', 1024 * 1024)
        self.verification_timeout = config.get('verification_timeout', 5.0)
        self.max_input_size = config.get('max_input_size', 1024)
        self.max_input_count = config.get('max_input_count', 100)
    
    def validate_proof_format(self, proof: Proof) -> Tuple[bool, Optional[str]]:
        """Validate proof format and structure."""
        # Check proof data size
        if len(proof.proof_data) > self.max_proof_size:
            return False, f"Proof data too large: {len(proof.proof_data)} bytes"
        
        if len(proof.proof_data) == 0:
            return False, "Proof data is empty"
        
        # Check public inputs
        if len(proof.public_inputs) > self.max_input_count:
            return False, f"Too many public inputs: {len(proof.public_inputs)}"
        
        for i, inp in enumerate(proof.public_inputs):
            if len(inp) > self.max_input_size:
                return False, f"Public input {i} too large: {len(inp)} bytes"
        
        # Check circuit ID
        if not proof.circuit_id or len(proof.circuit_id) > 256:
            return False, "Invalid circuit ID"
        
        # Check timestamp (not too old, not in future)
        current_time = time.time()
        if proof.timestamp > current_time + 300:  # 5 minutes in future
            return False, "Proof timestamp is in the future"
        
        if proof.timestamp < current_time - 86400:  # 24 hours old
            return False, "Proof timestamp is too old"
        
        return True, None
    
    def validate_public_inputs(self, public_inputs: List[bytes]) -> Tuple[bool, Optional[str]]:
        """Validate public inputs."""
        if len(public_inputs) > self.max_input_count:
            return False, f"Too many public inputs: {len(public_inputs)}"
        
        for i, inp in enumerate(public_inputs):
            if len(inp) > self.max_input_size:
                return False, f"Public input {i} too large: {len(inp)} bytes"
            
            if len(inp) == 0:
                return False, f"Public input {i} is empty"
        
        return True, None
    
    def detect_malformed_data(self, proof: Proof, public_inputs: List[bytes]) -> Tuple[bool, Optional[str]]:
        """Detect malformed or adversarial data."""
        # Check for null bytes in proof data (potential injection)
        if b'\x00' in proof.proof_data:
            return False, "Proof data contains null bytes"
        
        # Check for extremely large numbers in public inputs
        for i, inp in enumerate(public_inputs):
            if len(inp) > self.max_input_size:  # Use configured limit
                return False, f"Public input {i} contains extremely large value"
            
            # Check for all zeros (potential padding attack)
            if inp == b'\x00' * len(inp):
                return False, f"Public input {i} is all zeros"
        
        # Check for duplicate public inputs
        if len(set(public_inputs)) != len(public_inputs):
            return False, "Duplicate public inputs detected"
        
        # Check proof data for suspicious patterns
        if self._has_suspicious_patterns(proof.proof_data):
            return False, "Proof data contains suspicious patterns"
        
        return True, None
    
    def _has_suspicious_patterns(self, data: bytes) -> bool:
        """Check for suspicious patterns in proof data."""
        # Check for repeated patterns (potential padding)
        if len(data) > 100:
            # Look for repeated 16-byte patterns
            for i in range(0, len(data) - 32, 16):
                pattern = data[i:i+16]
                if data[i+16:i+32] == pattern:
                    return True
        
        # Check for all same bytes
        if len(set(data)) == 1:
            return True
        
        return False
    
    def verify_with_timeout(self, verify_func: Callable[[], VerificationResult]) -> VerificationResult:
        """Verify with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Verification timeout")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.verification_timeout))
        
        try:
            result = verify_func()
            return result
        except TimeoutError:
            return VerificationResult(
                status=ZKPStatus.TIMEOUT,
                error_message=f"Verification timed out after {self.verification_timeout} seconds"
            )
        except Exception as e:
            return VerificationResult(
                status=ZKPStatus.VERIFICATION_FAILED,
                error_message=f"Verification failed: {e}"
            )
        finally:
            # Restore old signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            'max_proof_size': self.max_proof_size,
            'verification_timeout': self.verification_timeout,
            'max_input_size': self.max_input_size,
            'max_input_count': self.max_input_count
        }
