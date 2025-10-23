"""
Optimized Batching and Aggregation implementation for DubChain.

This module provides performance optimizations for batching and aggregation including:
- Transaction batching and aggregation
- State write batching
- Signature aggregation
- Message batching
- Shard-aware batching
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import hashlib
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import weakref

from ..performance.optimizations import OptimizationManager, OptimizationFallback


@dataclass
class BatchEntry:
    """Entry in a batch operation."""
    data: Any
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    shard_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Batch operation configuration."""
    max_batch_size: int = 1000
    max_batch_time: float = 1.0  # seconds
    enable_priority_batching: bool = True
    enable_shard_aware_batching: bool = True
    enable_signature_aggregation: bool = True
    enable_state_batching: bool = True
    batch_timeout: float = 0.5  # seconds


@dataclass
class AggregatedSignature:
    """Aggregated signature data."""
    signatures: List[bytes]
    public_keys: List[bytes]
    message_hash: bytes
    aggregated_signature: Optional[bytes] = None
    verification_result: Optional[bool] = None


class OptimizedBatching:
    """
    Optimized Batching and Aggregation with performance enhancements.
    
    Features:
    - Transaction batching and aggregation
    - State write batching
    - Signature aggregation
    - Message batching
    - Shard-aware batching
    """
    
    def __init__(self, optimization_manager: OptimizationManager, config: Optional[BatchConfig] = None):
        """Initialize optimized batching manager."""
        self.optimization_manager = optimization_manager
        self.config = config or BatchConfig()
        
        # Batch queues by type
        self.batch_queues: Dict[str, deque] = defaultdict(deque)
        self.batch_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Shard-aware batching
        self.shard_batches: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.shard_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Signature aggregation
        self.signature_aggregator = SignatureAggregator()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Performance metrics
        self.metrics = {
            "total_batches": 0,
            "total_aggregations": 0,
            "batch_processing_time": 0.0,
            "aggregation_processing_time": 0.0,
            "signature_aggregations": 0,
            "state_batches": 0,
            "transaction_batches": 0,
            "message_batches": 0,
        }
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        
        # Start background processing
        self._start_background_processing()
    
    def _start_background_processing(self):
        """Start background batch processing tasks."""
        # Background tasks will be started when needed
        # This avoids async issues during initialization
        pass
    
    async def _create_background_tasks(self):
        """Create background processing tasks."""
        tasks = [
            asyncio.create_task(self._batch_processor_task("transactions")),
            asyncio.create_task(self._batch_processor_task("state_writes")),
            asyncio.create_task(self._batch_processor_task("messages")),
            asyncio.create_task(self._signature_aggregation_task()),
        ]
        
        self.background_tasks.update(tasks)
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    @OptimizationFallback
    def batch_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch transactions for efficient processing.
        
        Args:
            transactions: List of transactions to batch
            
        Returns:
            Batch processing result
        """
        if not self.optimization_manager.is_optimization_enabled("batching_transaction_aggregation"):
            return self._process_transactions_sequential(transactions)
        
        start_time = time.time()
        
        # Group transactions by shard if shard-aware batching is enabled
        if self.config.enable_shard_aware_batching:
            shard_groups = self._group_by_shard(transactions)
            results = {}
            
            for shard_id, shard_transactions in shard_groups.items():
                shard_result = self._process_transaction_batch(shard_transactions)
                results[shard_id] = shard_result
        else:
            results = self._process_transaction_batch(transactions)
        
        processing_time = time.time() - start_time
        self._update_batch_metrics("transaction_batches", processing_time)
        
        return {
            "success": True,
            "batch_size": len(transactions),
            "processing_time": processing_time,
            "results": results,
        }
    
    def _process_transactions_sequential(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process transactions sequentially without batching."""
        start_time = time.time()
        
        results = []
        for transaction in transactions:
            # Simple transaction processing
            result = {
                "tx_hash": hashlib.sha256(str(transaction).encode()).hexdigest(),
                "processed": True,
            }
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Update metrics for sequential processing
        self.metrics["total_batches"] += 1
        
        return {
            "success": True,
            "batch_size": len(transactions),
            "processing_time": processing_time,
            "results": results,
        }
    
    def _process_transaction_batch(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of transactions."""
        # Optimized batch processing
        batch_hash = hashlib.sha256(str(transactions).encode()).hexdigest()
        
        # Validate all transactions in batch
        valid_transactions = []
        for transaction in transactions:
            if self._validate_transaction(transaction):
                valid_transactions.append(transaction)
        
        # Process valid transactions
        results = []
        for transaction in valid_transactions:
            result = {
                "tx_hash": hashlib.sha256(str(transaction).encode()).hexdigest(),
                "processed": True,
                "batch_hash": batch_hash,
            }
            results.append(result)
        
        return {
            "batch_hash": batch_hash,
            "total_transactions": len(transactions),
            "valid_transactions": len(valid_transactions),
            "results": results,
        }
    
    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate a single transaction."""
        # Simple validation
        required_fields = ["from", "to", "value", "nonce"]
        return all(field in transaction for field in required_fields)
    
    def _group_by_shard(self, transactions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group transactions by shard."""
        shard_groups = defaultdict(list)
        
        for transaction in transactions:
            # Simple shard assignment based on 'to' address
            shard_id = self._get_shard_id(transaction.get("to", ""))
            shard_groups[shard_id].append(transaction)
        
        return dict(shard_groups)
    
    def _get_shard_id(self, address: str) -> str:
        """Get shard ID for an address."""
        if not address:
            return "default"
        
        # Simple shard assignment based on address hash
        address_hash = hashlib.sha256(address.encode()).hexdigest()
        shard_id = int(address_hash[:8], 16) % 4  # 4 shards
        return f"shard_{shard_id}"
    
    @OptimizationFallback
    def batch_state_writes(self, state_writes: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """
        Batch state writes for efficient processing.
        
        Args:
            state_writes: List of (key, value) tuples
            
        Returns:
            Batch processing result
        """
        if not self.optimization_manager.is_optimization_enabled("batching_state_aggregation"):
            return self._process_state_writes_sequential(state_writes)
        
        start_time = time.time()
        
        # Group state writes by key patterns for optimization
        grouped_writes = self._group_state_writes(state_writes)
        
        # Process grouped writes
        results = {}
        for group_key, writes in grouped_writes.items():
            group_result = self._process_state_write_group(writes)
            results[group_key] = group_result
        
        processing_time = time.time() - start_time
        self._update_batch_metrics("state_batches", processing_time)
        
        return {
            "success": True,
            "total_writes": len(state_writes),
            "processing_time": processing_time,
            "results": results,
        }
    
    def _process_state_writes_sequential(self, state_writes: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Process state writes sequentially without batching."""
        start_time = time.time()
        
        results = []
        for key, value in state_writes:
            result = {
                "key": key,
                "value": value,
                "processed": True,
            }
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "total_writes": len(state_writes),
            "processing_time": processing_time,
            "results": results,
        }
    
    def _group_state_writes(self, state_writes: List[Tuple[str, Any]]) -> Dict[str, List[Tuple[str, Any]]]:
        """Group state writes by key patterns."""
        groups = defaultdict(list)
        
        for key, value in state_writes:
            # Group by key prefix
            group_key = key.split('.')[0] if '.' in key else "default"
            groups[group_key].append((key, value))
        
        return dict(groups)
    
    def _process_state_write_group(self, writes: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Process a group of state writes."""
        # Optimized group processing
        group_hash = hashlib.sha256(str(writes).encode()).hexdigest()
        
        results = []
        for key, value in writes:
            result = {
                "key": key,
                "value": value,
                "group_hash": group_hash,
                "processed": True,
            }
            results.append(result)
        
        return {
            "group_hash": group_hash,
            "write_count": len(writes),
            "results": results,
        }
    
    @OptimizationFallback
    def aggregate_signatures(self, 
                           signatures: List[bytes], 
                           public_keys: List[bytes], 
                           message_hash: bytes) -> AggregatedSignature:
        """
        Aggregate multiple signatures for efficient verification.
        
        Args:
            signatures: List of signatures
            public_keys: List of corresponding public keys
            message_hash: Hash of the message
            
        Returns:
            Aggregated signature result
        """
        if not self.optimization_manager.is_optimization_enabled("batching_signature_aggregation"):
            return self._verify_signatures_sequential(signatures, public_keys, message_hash)
        
        start_time = time.time()
        
        # Create aggregated signature
        aggregated = AggregatedSignature(
            signatures=signatures,
            public_keys=public_keys,
            message_hash=message_hash
        )
        
        # Perform signature aggregation
        aggregated.aggregated_signature = self._combine_signatures(signatures)
        aggregated.verification_result = self._verify_aggregated_signature(aggregated)
        
        processing_time = time.time() - start_time
        self._update_batch_metrics("signature_aggregations", processing_time)
        
        return aggregated
    
    def _verify_signatures_sequential(self, 
                                    signatures: List[bytes], 
                                    public_keys: List[bytes], 
                                    message_hash: bytes) -> AggregatedSignature:
        """Verify signatures sequentially without aggregation."""
        aggregated = AggregatedSignature(
            signatures=signatures,
            public_keys=public_keys,
            message_hash=message_hash
        )
        
        # Simple sequential verification
        all_valid = True
        for signature, public_key in zip(signatures, public_keys):
            if not self._verify_single_signature(signature, public_key, message_hash):
                all_valid = False
                break
        
        aggregated.verification_result = all_valid
        return aggregated
    
    def _combine_signatures(self, signatures: List[bytes]) -> bytes:
        """Combine multiple signatures into one."""
        # Simple signature combination (XOR for demonstration)
        if not signatures:
            return b''
        
        combined = signatures[0]
        for signature in signatures[1:]:
            combined = bytes(a ^ b for a, b in zip(combined, signature))
        
        return combined
    
    def _verify_aggregated_signature(self, aggregated: AggregatedSignature) -> bool:
        """Verify an aggregated signature."""
        # Simple aggregated verification
        if not aggregated.aggregated_signature:
            return False
        
        # Check that all individual signatures are valid
        for signature, public_key in zip(aggregated.signatures, aggregated.public_keys):
            if not self._verify_single_signature(signature, public_key, aggregated.message_hash):
                return False
        
        return True
    
    def _verify_single_signature(self, signature: bytes, public_key: bytes, message_hash: bytes) -> bool:
        """Verify a single signature."""
        # Simple signature verification (placeholder)
        return len(signature) == 64 and len(public_key) == 33 and len(message_hash) == 32
    
    @OptimizationFallback
    def batch_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch messages for efficient network transmission.
        
        Args:
            messages: List of messages to batch
            
        Returns:
            Batch processing result
        """
        if not self.optimization_manager.is_optimization_enabled("batching_message_aggregation"):
            return self._process_messages_sequential(messages)
        
        start_time = time.time()
        
        # Group messages by type and priority
        grouped_messages = self._group_messages(messages)
        
        # Process grouped messages
        results = {}
        for group_key, group_messages in grouped_messages.items():
            group_result = self._process_message_group(group_messages)
            results[group_key] = group_result
        
        processing_time = time.time() - start_time
        self._update_batch_metrics("message_batches", processing_time)
        
        return {
            "success": True,
            "total_messages": len(messages),
            "processing_time": processing_time,
            "results": results,
        }
    
    def _process_messages_sequential(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process messages sequentially without batching."""
        start_time = time.time()
        
        results = []
        for message in messages:
            result = {
                "message_id": hashlib.sha256(str(message).encode()).hexdigest(),
                "processed": True,
            }
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "total_messages": len(messages),
            "processing_time": processing_time,
            "results": results,
        }
    
    def _group_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group messages by type and priority."""
        groups = defaultdict(list)
        
        for message in messages:
            # Group by message type
            message_type = message.get("type", "default")
            groups[message_type].append(message)
        
        return dict(groups)
    
    def _process_message_group(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a group of messages."""
        group_hash = hashlib.sha256(str(messages).encode()).hexdigest()
        
        results = []
        for message in messages:
            result = {
                "message_id": hashlib.sha256(str(message).encode()).hexdigest(),
                "group_hash": group_hash,
                "processed": True,
            }
            results.append(result)
        
        return {
            "group_hash": group_hash,
            "message_count": len(messages),
            "results": results,
        }
    
    async def _batch_processor_task(self, batch_type: str):
        """Background task for processing batches."""
        while True:
            try:
                await asyncio.sleep(self.config.batch_timeout)
                
                # Process pending batches
                if batch_type in self.batch_queues:
                    with self.batch_locks[batch_type]:
                        if self.batch_queues[batch_type]:
                            batch = list(self.batch_queues[batch_type])
                            self.batch_queues[batch_type].clear()
                            
                            # Process batch
                            if batch_type == "transactions":
                                await self._process_batch_async(batch, self.batch_transactions)
                            elif batch_type == "state_writes":
                                await self._process_batch_async(batch, self.batch_state_writes)
                            elif batch_type == "messages":
                                await self._process_batch_async(batch, self.batch_messages)
                
            except Exception as e:
                logger.info(f"Batch processor error for {batch_type}: {e}")
    
    async def _signature_aggregation_task(self):
        """Background task for signature aggregation."""
        while True:
            try:
                await asyncio.sleep(self.config.batch_timeout)
                
                # Process pending signature aggregations
                # This would be implemented based on specific requirements
                
            except Exception as e:
                logger.info(f"Signature aggregation error: {e}")
    
    async def _process_batch_async(self, batch: List[Any], processor_func: Callable):
        """Process a batch asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, processor_func, batch)
    
    def _update_batch_metrics(self, metric_name: str, processing_time: float):
        """Update batch processing metrics."""
        with self._metrics_lock:
            self.metrics["total_batches"] += 1
            self.metrics[metric_name] += 1
            self.metrics["batch_processing_time"] += processing_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **self.metrics,
            "avg_batch_processing_time": (
                self.metrics["batch_processing_time"] / max(self.metrics["total_batches"], 1)
            ),
            "optimization_enabled": {
                "transaction_aggregation": self.optimization_manager.is_optimization_enabled("batching_transaction_aggregation"),
                "state_aggregation": self.optimization_manager.is_optimization_enabled("batching_state_aggregation"),
                "signature_aggregation": self.optimization_manager.is_optimization_enabled("batching_signature_aggregation"),
                "message_aggregation": self.optimization_manager.is_optimization_enabled("batching_message_aggregation"),
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Shutdown executor
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            # Ignore shutdown errors during teardown
            pass
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


class SignatureAggregator:
    """Specialized signature aggregation utility."""
    
    def __init__(self):
        """Initialize signature aggregator."""
        self.aggregation_cache = {}
        self.cache_lock = threading.RLock()
    
    def aggregate_batch(self, signature_batch: List[Tuple[bytes, bytes, bytes]]) -> bytes:
        """
        Aggregate a batch of signatures.
        
        Args:
            signature_batch: List of (signature, public_key, message_hash) tuples
            
        Returns:
            Aggregated signature
        """
        if not signature_batch:
            return b''
        
        # Simple aggregation (XOR for demonstration)
        aggregated = signature_batch[0][0]  # First signature
        for signature, _, _ in signature_batch[1:]:
            aggregated = bytes(a ^ b for a, b in zip(aggregated, signature))
        
        return aggregated
    
    def verify_aggregated(self, aggregated_signature: bytes, 
                         public_keys: List[bytes], 
                         message_hash: bytes) -> bool:
        """
        Verify an aggregated signature.
        
        Args:
            aggregated_signature: Aggregated signature to verify
            public_keys: List of public keys
            message_hash: Message hash
            
        Returns:
            True if verification succeeds
        """
        # Simple verification (placeholder)
        return (len(aggregated_signature) == 64 and 
                len(public_keys) > 0 and 
                len(message_hash) == 32)
