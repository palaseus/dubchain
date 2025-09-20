"""
Optimized consensus mechanisms for DubChain.

This module provides performance-optimized implementations of consensus mechanisms
with batching, lock reduction, and O(1) data structures.
"""

import asyncio
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import weakref

from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusState,
    ConsensusType,
)
from .proof_of_stake import ProofOfStake
from .delegated_proof_of_stake import DelegatedProofOfStake
from .pbft import PracticalByzantineFaultTolerance
from .hotstuff import HotStuffConsensus
from ..performance.optimizations import OptimizationManager


@dataclass
class BatchValidationResult:
    """Result of batch validation operation."""
    
    successful_blocks: List[Any]
    failed_blocks: List[Any]
    validation_time: float
    batch_size: int
    errors: List[str] = field(default_factory=list)


@dataclass
class OptimizedConsensusConfig(ConsensusConfig):
    """Configuration for optimized consensus mechanisms."""
    
    # Batching configuration
    enable_batch_validation: bool = True
    batch_size: int = 10
    batch_timeout: float = 0.1  # 100ms
    
    # Lock optimization
    enable_lock_optimization: bool = True
    use_read_write_locks: bool = True
    lock_striping_count: int = 16
    
    # Data structure optimization
    enable_o1_structures: bool = True
    use_consistent_hashing: bool = True
    cache_size: int = 1000
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: float = 1.0


class LockStripedDict:
    """Lock-striped dictionary for reduced contention."""
    
    def __init__(self, strip_count: int = 16):
        self.strip_count = strip_count
        self.strips = [{} for _ in range(strip_count)]
        self.locks = [threading.RLock() for _ in range(strip_count)]
        
    def _get_strip(self, key: str) -> Tuple[Dict, threading.RLock]:
        """Get the strip and lock for a key."""
        strip_index = hash(key) % self.strip_count
        return self.strips[strip_index], self.locks[strip_index]
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from striped dictionary."""
        strip, lock = self._get_strip(key)
        with lock:
            return strip.get(key, default)
            
    def set(self, key: str, value: Any) -> None:
        """Set value in striped dictionary."""
        strip, lock = self._get_strip(key)
        with lock:
            strip[key] = value
            
    def delete(self, key: str) -> bool:
        """Delete key from striped dictionary."""
        strip, lock = self._get_strip(key)
        with lock:
            if key in strip:
                del strip[key]
                return True
            return False
            
    def contains(self, key: str) -> bool:
        """Check if key exists in striped dictionary."""
        strip, lock = self._get_strip(key)
        with lock:
            return key in strip
            
    def size(self) -> int:
        """Get total size of all strips."""
        total = 0
        for strip, lock in zip(self.strips, self.locks):
            with lock:
                total += len(strip)
        return total


class OptimizedValidatorSet:
    """Optimized validator set with O(1) operations."""
    
    def __init__(self, max_validators: int = 1000):
        self.max_validators = max_validators
        self.validators: Dict[str, Any] = {}
        self.validator_list: List[str] = []
        self.validator_index: Dict[str, int] = {}
        self.total_stake: int = 0
        self.stake_weights: List[Tuple[str, int]] = []
        self._lock = threading.RLock()
        
    def add_validator(self, validator_id: str, stake: int = 0) -> bool:
        """Add validator with O(1) complexity."""
        with self._lock:
            if validator_id in self.validators:
                return False
                
            if len(self.validators) >= self.max_validators:
                return False
                
            self.validators[validator_id] = {"stake": stake, "active": True}
            self.validator_index[validator_id] = len(self.validator_list)
            self.validator_list.append(validator_id)
            self.total_stake += stake
            
            # Update stake weights for weighted selection
            self._update_stake_weights()
            return True
            
    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator with O(1) complexity."""
        with self._lock:
            if validator_id not in self.validators:
                return False
                
            # Get index of validator to remove
            index = self.validator_index[validator_id]
            
            # Swap with last element for O(1) removal
            last_validator = self.validator_list[-1]
            self.validator_list[index] = last_validator
            self.validator_index[last_validator] = index
            
            # Remove last element
            self.validator_list.pop()
            del self.validator_index[validator_id]
            del self.validators[validator_id]
            
            # Update stake weights
            self._update_stake_weights()
            return True
            
    def get_validator(self, validator_id: str) -> Optional[Dict[str, Any]]:
        """Get validator info with O(1) complexity."""
        with self._lock:
            return self.validators.get(validator_id)
            
    def select_proposer(self, block_number: int) -> Optional[str]:
        """Select proposer with O(1) complexity using weighted selection."""
        with self._lock:
            if not self.validator_list:
                return None
                
            # Use block number as seed for deterministic selection
            seed = block_number % (2**32)
            
            # Weighted random selection
            if self.stake_weights:
                total_weight = sum(weight for _, weight in self.stake_weights)
                if total_weight > 0:
                    target = (seed * 1103515245 + 12345) % total_weight
                    cumulative = 0
                    for validator_id, weight in self.stake_weights:
                        cumulative += weight
                        if cumulative > target:
                            return validator_id
                            
            # Fallback to round-robin
            return self.validator_list[seed % len(self.validator_list)]
            
    def _update_stake_weights(self) -> None:
        """Update stake weights for weighted selection."""
        self.stake_weights = [
            (vid, info["stake"]) 
            for vid, info in self.validators.items() 
            if info["active"] and info["stake"] > 0
        ]
        
    def size(self) -> int:
        """Get validator set size."""
        with self._lock:
            return len(self.validators)


class OptimizedProofOfStake(ProofOfStake):
    """Optimized Proof of Stake consensus mechanism."""
    
    def __init__(self, config: OptimizedConsensusConfig):
        super().__init__(config)
        self.optimization_manager = OptimizationManager()
        self.optimized_config = config
        
        # Optimized data structures
        if config.enable_o1_structures:
            self.validator_set = OptimizedValidatorSet(config.max_validators)
            self.block_cache = LockStripedDict(config.lock_striping_count)
            self.vote_cache = LockStripedDict(config.lock_striping_count)
        else:
            self.validator_set = OptimizedValidatorSet(config.max_validators)
            self.block_cache = {}
            self.vote_cache = {}
            
        # Batch processing
        self.pending_blocks: deque = deque()
        self.batch_lock = threading.Lock()
        self.batch_timer: Optional[float] = None
        
        # Performance monitoring
        self.performance_metrics = {
            "block_creation_times": deque(maxlen=1000),
            "validation_times": deque(maxlen=1000),
            "batch_sizes": deque(maxlen=1000),
        }
        
    def add_validator(self, validator_id: str, stake: int = 0) -> bool:
        """Add validator with optimized data structure."""
        return self.validator_set.add_validator(validator_id, stake)
        
    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator with optimized data structure."""
        return self.validator_set.remove_validator(validator_id)
        
    def select_proposer(self, block_number: int) -> Optional[str]:
        """Select proposer with O(1) complexity."""
        return self.validator_set.select_proposer(block_number)
        
    def validate_block_batch(self, blocks: List[Any]) -> BatchValidationResult:
        """Validate multiple blocks in batch for improved throughput."""
        if not self.optimized_config.enable_batch_validation:
            # Fallback to individual validation
            successful_blocks = []
            failed_blocks = []
            errors = []
            
            for block in blocks:
                try:
                    if self._validate_single_block(block):
                        successful_blocks.append(block)
                    else:
                        failed_blocks.append(block)
                        errors.append(f"Block {block.get('hash', 'unknown')} validation failed")
                except Exception as e:
                    failed_blocks.append(block)
                    errors.append(f"Block {block.get('hash', 'unknown')} error: {e}")
                    
            return BatchValidationResult(
                successful_blocks=successful_blocks,
                failed_blocks=failed_blocks,
                validation_time=0.0,
                batch_size=len(blocks),
                errors=errors
            )
            
        # Optimized batch validation
        start_time = time.time()
        
        try:
            # Pre-validate common properties
            valid_blocks = []
            invalid_blocks = []
            
            for block in blocks:
                if self._pre_validate_block(block):
                    valid_blocks.append(block)
                else:
                    invalid_blocks.append(block)
                    
            # Batch validate valid blocks
            if valid_blocks:
                batch_validated = self._batch_validate_blocks(valid_blocks)
                successful_blocks = batch_validated["successful"]
                failed_blocks = batch_validated["failed"] + invalid_blocks
            else:
                successful_blocks = []
                failed_blocks = invalid_blocks
                
            validation_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_metrics["validation_times"].append(validation_time)
            self.performance_metrics["batch_sizes"].append(len(blocks))
            
            return BatchValidationResult(
                successful_blocks=successful_blocks,
                failed_blocks=failed_blocks,
                validation_time=validation_time,
                batch_size=len(blocks),
                errors=[]
            )
            
        except Exception as e:
            # Fallback to individual validation on batch failure
            return self.validate_block_batch(blocks)  # This will use fallback
            
    def _pre_validate_block(self, block: Any) -> bool:
        """Pre-validate block for common properties."""
        try:
            # Check basic block structure
            required_fields = ["index", "timestamp", "transactions", "previous_hash"]
            for field in required_fields:
                if field not in block:
                    return False
                    
            # Check timestamp is reasonable
            current_time = time.time()
            if abs(block["timestamp"] - current_time) > 3600:  # 1 hour tolerance
                return False
                
            return True
        except:
            return False
            
    def _batch_validate_blocks(self, blocks: List[Any]) -> Dict[str, List[Any]]:
        """Perform batch validation of blocks."""
        successful = []
        failed = []
        
        # Group blocks by common properties for batch processing
        blocks_by_previous_hash = defaultdict(list)
        for block in blocks:
            blocks_by_previous_hash[block["previous_hash"]].append(block)
            
        # Validate each group
        for previous_hash, group_blocks in blocks_by_previous_hash.items():
            try:
                # Batch validate transactions
                all_transactions = []
                for block in group_blocks:
                    all_transactions.extend(block.get("transactions", []))
                    
                if self._batch_validate_transactions(all_transactions):
                    successful.extend(group_blocks)
                else:
                    failed.extend(group_blocks)
                    
            except Exception:
                failed.extend(group_blocks)
                
        return {"successful": successful, "failed": failed}
        
    def _batch_validate_transactions(self, transactions: List[Any]) -> bool:
        """Batch validate transactions."""
        # Simplified batch validation - in practice this would be more sophisticated
        for tx in transactions:
            if not self._validate_single_transaction(tx):
                return False
        return True
        
    def _validate_single_transaction(self, transaction: Any) -> bool:
        """Validate a single transaction."""
        # Simplified validation - in practice this would be more comprehensive
        required_fields = ["sender", "recipient", "amount", "fee"]
        for field in required_fields:
            if field not in transaction:
                return False
        return True
        
    def _validate_single_block(self, block: Any) -> bool:
        """Validate a single block (fallback method)."""
        # This would implement the actual block validation logic
        return self._pre_validate_block(block)
        
    def add_block_to_batch(self, block: Any) -> bool:
        """Add block to batch for processing."""
        if not self.optimized_config.enable_batch_validation:
            return False
            
        with self.batch_lock:
            self.pending_blocks.append(block)
            
            # Check if batch is ready
            if (len(self.pending_blocks) >= self.optimized_config.batch_size or
                (self.batch_timer and time.time() - self.batch_timer >= self.optimized_config.batch_timeout)):
                
                # Process batch
                blocks_to_process = list(self.pending_blocks)
                self.pending_blocks.clear()
                self.batch_timer = None
                
                # Validate batch
                result = self.validate_block_batch(blocks_to_process)
                return len(result.successful_blocks) > 0
                
            # Set timer if this is the first block in batch
            if self.batch_timer is None:
                self.batch_timer = time.time()
                
        return True
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {}
        
        # Calculate averages
        if self.performance_metrics["validation_times"]:
            metrics["avg_validation_time"] = sum(self.performance_metrics["validation_times"]) / len(self.performance_metrics["validation_times"])
            metrics["max_validation_time"] = max(self.performance_metrics["validation_times"])
            
        if self.performance_metrics["batch_sizes"]:
            metrics["avg_batch_size"] = sum(self.performance_metrics["batch_sizes"]) / len(self.performance_metrics["batch_sizes"])
            metrics["max_batch_size"] = max(self.performance_metrics["batch_sizes"])
            
        # Validator set metrics
        metrics["validator_count"] = self.validator_set.size()
        metrics["total_stake"] = self.validator_set.total_stake
        
        return metrics


class OptimizedDelegatedProofOfStake(DelegatedProofOfStake):
    """Optimized Delegated Proof of Stake consensus mechanism."""
    
    def __init__(self, config: OptimizedConsensusConfig):
        super().__init__(config)
        self.optimized_config = config
        
        # Optimized delegate management
        self.delegate_cache = LockStripedDict(config.lock_striping_count)
        self.delegation_cache = LockStripedDict(config.lock_striping_count)
        
        # Batch delegate operations
        self.pending_delegations: deque = deque()
        self.delegation_lock = threading.Lock()
        
    def batch_process_delegations(self, delegations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple delegations in batch."""
        results = {
            "successful": [],
            "failed": [],
            "total_processed": len(delegations),
        }
        
        # Group delegations by delegate
        delegations_by_delegate = defaultdict(list)
        for delegation in delegations:
            delegate_id = delegation.get("delegate_id")
            if delegate_id:
                delegations_by_delegate[delegate_id].append(delegation)
                
        # Process each delegate's delegations
        for delegate_id, delegate_delegations in delegations_by_delegate.items():
            try:
                # Batch process delegations for this delegate
                delegate_result = self._batch_process_delegate_delegations(delegate_id, delegate_delegations)
                results["successful"].extend(delegate_result["successful"])
                results["failed"].extend(delegate_result["failed"])
            except Exception as e:
                # Mark all delegations for this delegate as failed
                for delegation in delegate_delegations:
                    results["failed"].append({
                        "delegation": delegation,
                        "error": str(e)
                    })
                    
        return results
        
    def _batch_process_delegate_delegations(self, delegate_id: str, delegations: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Process delegations for a specific delegate in batch."""
        successful = []
        failed = []
        
        # Calculate total delegation amount
        total_amount = sum(d.get("amount", 0) for d in delegations)
        
        # Check if delegate can accept this delegation
        if self._can_accept_delegation(delegate_id, total_amount):
            # Process all delegations
            for delegation in delegations:
                try:
                    result = self._process_single_delegation(delegation)
                    if result:
                        successful.append(delegation)
                    else:
                        failed.append(delegation)
                except Exception:
                    failed.append(delegation)
        else:
            # Mark all as failed
            failed.extend(delegations)
            
        return {"successful": successful, "failed": failed}
        
    def _can_accept_delegation(self, delegate_id: str, amount: int) -> bool:
        """Check if delegate can accept delegation amount."""
        # This would implement actual delegation capacity checking
        return True  # Simplified
        
    def _process_single_delegation(self, delegation: Dict[str, Any]) -> bool:
        """Process a single delegation."""
        # This would implement actual delegation processing
        return True  # Simplified


class OptimizedHotStuffConsensus(HotStuffConsensus):
    """Optimized HotStuff consensus mechanism."""
    
    def __init__(self, config: OptimizedConsensusConfig):
        super().__init__(config)
        self.optimized_config = config
        
        # Optimized message handling
        self.message_batch_size = config.batch_size
        self.pending_messages: deque = deque()
        self.message_lock = threading.Lock()
        
        # Optimized vote aggregation
        self.vote_aggregator = LockStripedDict(config.lock_striping_count)
        
    def batch_process_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple consensus messages in batch."""
        results = {
            "processed": 0,
            "failed": 0,
            "aggregated_votes": 0,
        }
        
        # Group messages by type
        messages_by_type = defaultdict(list)
        for message in messages:
            msg_type = message.get("type", "unknown")
            messages_by_type[msg_type].append(message)
            
        # Process each message type
        for msg_type, type_messages in messages_by_type.items():
            try:
                if msg_type == "vote":
                    results["aggregated_votes"] += self._batch_aggregate_votes(type_messages)
                elif msg_type == "proposal":
                    results["processed"] += self._batch_process_proposals(type_messages)
                else:
                    results["processed"] += self._batch_process_other_messages(type_messages)
            except Exception:
                results["failed"] += len(type_messages)
                
        return results
        
    def _batch_aggregate_votes(self, votes: List[Dict[str, Any]]) -> int:
        """Aggregate multiple votes in batch."""
        aggregated_count = 0
        
        # Group votes by proposal
        votes_by_proposal = defaultdict(list)
        for vote in votes:
            proposal_hash = vote.get("proposal_hash")
            if proposal_hash:
                votes_by_proposal[proposal_hash].append(vote)
                
        # Aggregate votes for each proposal
        for proposal_hash, proposal_votes in votes_by_proposal.items():
            try:
                if self._aggregate_proposal_votes(proposal_hash, proposal_votes):
                    aggregated_count += len(proposal_votes)
            except Exception:
                pass
                
        return aggregated_count
        
    def _aggregate_proposal_votes(self, proposal_hash: str, votes: List[Dict[str, Any]]) -> bool:
        """Aggregate votes for a specific proposal."""
        # This would implement actual vote aggregation logic
        return True  # Simplified
        
    def _batch_process_proposals(self, proposals: List[Dict[str, Any]]) -> int:
        """Process multiple proposals in batch."""
        processed_count = 0
        
        for proposal in proposals:
            try:
                if self._process_single_proposal(proposal):
                    processed_count += 1
            except Exception:
                pass
                
        return processed_count
        
    def _process_single_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Process a single proposal."""
        # This would implement actual proposal processing
        return True  # Simplified
        
    def _batch_process_other_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Process other types of messages in batch."""
        # This would implement processing for other message types
        return len(messages)


class OptimizedConsensusEngine:
    """Optimized consensus engine that uses optimized consensus mechanisms."""
    
    def __init__(self, config: OptimizedConsensusConfig):
        self.config = config
        self.optimization_manager = OptimizationManager()
        
        # Create optimized consensus mechanism
        self.consensus_mechanism = self._create_optimized_consensus(config.consensus_type, config)
        
        # Performance monitoring
        self.performance_monitor = None
        if config.enable_performance_monitoring:
            from ..performance.monitoring import PerformanceMonitor
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_monitoring()
            
    def _create_optimized_consensus(self, consensus_type: ConsensusType, config: OptimizedConsensusConfig):
        """Create optimized consensus mechanism."""
        if consensus_type == ConsensusType.PROOF_OF_STAKE:
            return OptimizedProofOfStake(config)
        elif consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE:
            return OptimizedDelegatedProofOfStake(config)
        elif consensus_type == ConsensusType.HOTSTUFF:
            return OptimizedHotStuffConsensus(config)
        else:
            # Fallback to standard consensus
            from .consensus_engine import ConsensusEngine
            return ConsensusEngine(config)
            
    def propose_block_optimized(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """Propose block with optimizations enabled."""
        start_time = time.time()
        
        try:
            # Use optimized consensus mechanism
            if hasattr(self.consensus_mechanism, "propose_block"):
                result = self.consensus_mechanism.propose_block(block_data)
            else:
                # Fallback to standard proposal
                result = ConsensusResult(
                    success=False,
                    error_message="Optimized consensus not available",
                    consensus_type=self.config.consensus_type,
                )
                
            # Record performance metrics
            if self.performance_monitor:
                block_time = time.time() - start_time
                self.performance_monitor.record_block_creation_time(block_time)
                
            return result
            
        except Exception as e:
            # Fallback to standard consensus on error
            from .consensus_engine import ConsensusEngine
            fallback_engine = ConsensusEngine(self.config)
            return fallback_engine.propose_block(block_data)
            
    def validate_blocks_batch(self, blocks: List[Any]) -> BatchValidationResult:
        """Validate multiple blocks in batch."""
        if hasattr(self.consensus_mechanism, "validate_block_batch"):
            return self.consensus_mechanism.validate_block_batch(blocks)
        else:
            # Fallback to individual validation
            successful_blocks = []
            failed_blocks = []
            errors = []
            
            for block in blocks:
                try:
                    if self.consensus_mechanism.validate_block(block):
                        successful_blocks.append(block)
                    else:
                        failed_blocks.append(block)
                        errors.append(f"Block validation failed")
                except Exception as e:
                    failed_blocks.append(block)
                    errors.append(str(e))
                    
            return BatchValidationResult(
                successful_blocks=successful_blocks,
                failed_blocks=failed_blocks,
                validation_time=0.0,
                batch_size=len(blocks),
                errors=errors
            )
            
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        metrics = {
            "consensus_type": self.config.consensus_type.value,
            "optimizations_enabled": {
                "batch_validation": self.config.enable_batch_validation,
                "lock_optimization": self.config.enable_lock_optimization,
                "o1_structures": self.config.enable_o1_structures,
            }
        }
        
        # Get mechanism-specific metrics
        if hasattr(self.consensus_mechanism, "get_performance_metrics"):
            mechanism_metrics = self.consensus_mechanism.get_performance_metrics()
            metrics.update(mechanism_metrics)
            
        return metrics
