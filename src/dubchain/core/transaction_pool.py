"""
Advanced Transaction Pool Management System for DubChain.

This module implements a sophisticated transaction pool with:
- Transaction prioritization based on fee rate and age
- Replace-by-Fee (RBF) mechanisms
- Transaction expiration and cleanup
- Validation caching
- Fee estimation algorithms
- Memory management and performance optimization
"""

import logging

logger = logging.getLogger(__name__)
import heapq
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash
from .blockchain import BlockchainState
from .transaction import UTXO, Transaction, TransactionType

class TransactionPriority(Enum):
    """Transaction priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TransactionEntry:
    """Entry in the transaction pool with metadata."""

    transaction: Transaction
    priority: TransactionPriority = TransactionPriority.NORMAL
    fee_rate: float = 0.0  # satoshis per byte
    age: float = 0.0  # seconds since first seen
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    validation_cache: Optional[bool] = None
    validation_timestamp: float = 0.0
    rbf_enabled: bool = False
    rbf_sequence: int = 0
    mempool_ancestors: Set[str] = field(default_factory=set)
    mempool_descendants: Set[str] = field(default_factory=set)
    conflicts_with: Set[str] = field(default_factory=set)
    replacement_tx: Optional[str] = None  # hash of transaction this replaces

    def __post_init__(self) -> None:
        """Calculate fee rate after initialization."""
        if self.fee_rate == 0.0:
            self.fee_rate = self._calculate_fee_rate()

    def _calculate_fee_rate(self) -> float:
        """Calculate fee rate in satoshis per byte."""
        if self.transaction.transaction_type == TransactionType.COINBASE:
            return 0.0

        tx_size = len(self.transaction.to_bytes())
        if tx_size == 0:
            return 0.0

        # For now, use a simple fee calculation based on transaction size
        # In a real implementation, this would use actual UTXO data
        base_fee = 1000  # Base fee in satoshis
        size_fee = tx_size * 10  # 10 satoshis per byte
        estimated_fee = base_fee + size_fee
        return estimated_fee / tx_size

    def update_age(self) -> None:
        """Update the age of the transaction."""
        self.age = time.time() - self.first_seen
        self.last_seen = time.time()

    def get_priority_score(self) -> float:
        """Calculate priority score for transaction ordering."""
        # Higher fee rate = higher priority
        fee_score = self.fee_rate * 1000

        # Age bonus for older transactions (prevents starvation)
        age_bonus = min(self.age / 3600, 1.0) * 100  # Max 1 hour bonus

        # Priority multiplier
        priority_multiplier = {
            TransactionPriority.LOW: 0.5,
            TransactionPriority.NORMAL: 1.0,
            TransactionPriority.HIGH: 1.5,
            TransactionPriority.URGENT: 2.0,
        }.get(self.priority, 1.0)

        # Base score for transactions (including coinbase)
        base_score = (
            100.0
            if self.transaction.transaction_type == TransactionType.COINBASE
            else 0.0
        )

        return (base_score + fee_score + age_bonus) * priority_multiplier

    def is_expired(self, max_age: float = 3600) -> bool:
        """Check if transaction has expired."""
        return self.age > max_age

    def can_replace(self, other: "TransactionEntry") -> bool:
        """Check if this transaction can replace another."""
        if not self.rbf_enabled or not other.rbf_enabled:
            return False

        # Must have higher fee rate
        if self.fee_rate <= other.fee_rate:
            return False

        # Must have higher sequence number
        if self.rbf_sequence <= other.rbf_sequence:
            return False

        return True

@dataclass
class FeeEstimate:
    """Fee estimate for different confirmation targets."""

    target_blocks: int
    fee_rate: float  # satoshis per byte
    confidence: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)

class TransactionPool:
    """Advanced transaction pool with sophisticated management."""

    def __init__(
        self,
        max_size: int = 100000,
        max_memory_mb: int = 512,
        validation_cache_ttl: float = 300.0,
        cleanup_interval: float = 60.0,
        fee_estimate_blocks: List[int] = None,
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.validation_cache_ttl = validation_cache_ttl
        self.cleanup_interval = cleanup_interval
        self.fee_estimate_blocks = fee_estimate_blocks or [1, 3, 6, 12, 24]

        # Core data structures
        self._transactions: Dict[str, TransactionEntry] = {}
        self._priority_queue: List[
            Tuple[float, str]
        ] = []  # (negative_priority, tx_hash)
        self._by_fee_rate: List[Tuple[float, str]] = []  # (negative_fee_rate, tx_hash)
        self._by_address: Dict[str, Set[str]] = defaultdict(set)
        self._conflicts: Dict[str, Set[str]] = defaultdict(set)

        # Validation cache
        self._validation_cache: Dict[str, Tuple[bool, float]] = {}

        # Fee estimation
        self._fee_estimates: Dict[int, FeeEstimate] = {}
        self._fee_history: deque = deque(maxlen=1000)

        # Memory management
        self._memory_usage = 0
        self._memory_pool = weakref.WeakValueDictionary()

        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks
        self._validation_callback: Optional[
            Callable[[Transaction, Dict[str, UTXO]], bool]
        ] = None
        self._conflict_callback: Optional[
            Callable[[Transaction, Transaction], bool]
        ] = None

    def start(self) -> None:
        """Start the transaction pool background processes."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker, daemon=True
            )
            self._cleanup_thread.start()

    def stop(self) -> None:
        """Stop the transaction pool background processes."""
        with self._lock:
            self._running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5.0)

    def set_validation_callback(
        self, callback: Callable[[Transaction, Dict[str, UTXO]], bool]
    ) -> None:
        """Set the transaction validation callback."""
        self._validation_callback = callback

    def set_conflict_callback(
        self, callback: Callable[[Transaction, Transaction], bool]
    ) -> None:
        """Set the transaction conflict resolution callback."""
        self._conflict_callback = callback

    def add_transaction(
        self,
        transaction: Transaction,
        priority: TransactionPriority = TransactionPriority.NORMAL,
        rbf_enabled: bool = False,
        rbf_sequence: int = 0,
        utxos: Optional[Dict[str, UTXO]] = None,
    ) -> Tuple[bool, str]:
        """
        Add a transaction to the pool.

        Returns:
            (success, message)
        """
        with self._lock:
            tx_hash = transaction.get_hash().to_hex()

            # Check if transaction already exists
            if tx_hash in self._transactions:
                return False, "Transaction already in pool"

            # Check pool size limits
            if len(self._transactions) >= self.max_size:
                if not self._evict_low_priority():
                    return False, "Pool is full and cannot evict transactions"

            # Check memory limits
            if self._memory_usage > self.max_memory_mb * 1024 * 1024:
                if not self._evict_memory():
                    return False, "Pool memory limit exceeded"

            # Validate transaction
            if not self._validate_transaction(transaction, utxos):
                return False, "Transaction validation failed"

            # Check for conflicts
            conflicts = self._find_conflicts(transaction)
            if conflicts:
                if not self._handle_conflicts(transaction, conflicts):
                    return False, "Transaction conflicts with existing transactions"

            # Create transaction entry
            entry = TransactionEntry(
                transaction=transaction,
                priority=priority,
                rbf_enabled=rbf_enabled,
                rbf_sequence=rbf_sequence,
            )

            # Add to data structures
            self._transactions[tx_hash] = entry
            self._add_to_priority_queue(entry)
            self._add_to_fee_queue(entry)
            self._update_address_index(entry)
            self._update_conflicts(entry)

            # Update memory usage
            self._memory_usage += len(transaction.to_bytes())

            # Update fee estimates
            self._update_fee_estimates(entry)

            return True, "Transaction added successfully"

    def remove_transaction(self, tx_hash: str) -> bool:
        """Remove a transaction from the pool."""
        with self._lock:
            if tx_hash not in self._transactions:
                return False

            entry = self._transactions[tx_hash]

            # Remove from all data structures
            del self._transactions[tx_hash]
            self._remove_from_priority_queue(tx_hash)
            self._remove_from_fee_queue(tx_hash)
            self._remove_from_address_index(entry)
            self._remove_from_conflicts(entry)

            # Update memory usage
            self._memory_usage -= len(entry.transaction.to_bytes())

            # Remove from validation cache
            self._validation_cache.pop(tx_hash, None)

            return True

    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get a transaction by hash."""
        with self._lock:
            entry = self._transactions.get(tx_hash)
            if entry:
                entry.update_age()
                return entry.transaction
            return None

    def get_priority_transactions(self, limit: int = 1000) -> List[Transaction]:
        """Get transactions ordered by priority."""
        with self._lock:
            transactions = []
            for _, tx_hash in heapq.nsmallest(limit, self._priority_queue):
                if tx_hash in self._transactions:
                    entry = self._transactions[tx_hash]
                    entry.update_age()
                    transactions.append(entry.transaction)
            return transactions

    def get_high_fee_transactions(self, limit: int = 1000) -> List[Transaction]:
        """Get transactions ordered by fee rate."""
        with self._lock:
            transactions = []
            for _, tx_hash in heapq.nsmallest(limit, self._by_fee_rate):
                if tx_hash in self._transactions:
                    entry = self._transactions[tx_hash]
                    entry.update_age()
                    transactions.append(entry.transaction)
            return transactions

    def get_transactions_by_address(self, address: str) -> List[Transaction]:
        """Get all transactions for a specific address."""
        with self._lock:
            transactions = []
            for tx_hash in self._by_address.get(address, set()):
                if tx_hash in self._transactions:
                    entry = self._transactions[tx_hash]
                    entry.update_age()
                    transactions.append(entry.transaction)
            return transactions

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the transaction pool."""
        with self._lock:
            total_fee_rate = sum(
                entry.fee_rate for entry in self._transactions.values()
            )
            avg_fee_rate = (
                total_fee_rate / len(self._transactions) if self._transactions else 0
            )

            return {
                "size": len(self._transactions),
                "max_size": self.max_size,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_mb,
                "average_fee_rate": avg_fee_rate,
                "validation_cache_size": len(self._validation_cache),
                "conflicts": len(self._conflicts),
                "fee_estimates": {
                    str(blocks): estimate.fee_rate
                    for blocks, estimate in self._fee_estimates.items()
                },
            }

    def get_fee_estimate(self, target_blocks: int) -> Optional[FeeEstimate]:
        """Get fee estimate for a specific confirmation target."""
        with self._lock:
            return self._fee_estimates.get(target_blocks)

    def clear_expired_transactions(self) -> int:
        """Remove expired transactions from the pool."""
        with self._lock:
            expired = []
            for tx_hash, entry in self._transactions.items():
                if entry.is_expired():
                    expired.append(tx_hash)

            for tx_hash in expired:
                self.remove_transaction(tx_hash)

            return len(expired)

    def _validate_transaction(
        self, transaction: Transaction, utxos: Optional[Dict[str, UTXO]]
    ) -> bool:
        """Validate a transaction."""
        tx_hash = transaction.get_hash().to_hex()

        # Check validation cache
        if tx_hash in self._validation_cache:
            is_valid, timestamp = self._validation_cache[tx_hash]
            if time.time() - timestamp < self.validation_cache_ttl:
                return is_valid

        # Perform validation
        is_valid = False
        if self._validation_callback and utxos:
            is_valid = self._validation_callback(transaction, utxos)
        else:
            # Basic validation - for coinbase transactions, always valid
            if transaction.transaction_type == TransactionType.COINBASE:
                is_valid = True
            else:
                # For regular transactions, check basic structure
                is_valid = (
                    len(transaction.inputs) > 0
                    and len(transaction.outputs) > 0
                    and transaction.get_total_output_amount() > 0
                )

        # Cache result
        self._validation_cache[tx_hash] = (is_valid, time.time())

        return is_valid

    def _find_conflicts(self, transaction: Transaction) -> List[str]:
        """Find transactions that conflict with the given transaction."""
        conflicts = []
        tx_inputs = {
            f"{inp.previous_tx_hash.to_hex()}:{inp.output_index}"
            for inp in transaction.inputs
        }

        for tx_hash, entry in self._transactions.items():
            existing_inputs = {
                f"{inp.previous_tx_hash.to_hex()}:{inp.output_index}"
                for inp in entry.transaction.inputs
            }
            if tx_inputs & existing_inputs:  # If any inputs overlap
                conflicts.append(tx_hash)

        return conflicts

    def _handle_conflicts(self, transaction: Transaction, conflicts: List[str]) -> bool:
        """Handle transaction conflicts using RBF or other mechanisms."""
        if not self._conflict_callback:
            return False

        for conflict_hash in conflicts:
            conflict_tx = self._transactions[conflict_hash].transaction
            if not self._conflict_callback(transaction, conflict_tx):
                return False

        # Remove conflicting transactions
        for conflict_hash in conflicts:
            self.remove_transaction(conflict_hash)

        return True

    def _add_to_priority_queue(self, entry: TransactionEntry) -> None:
        """Add transaction to priority queue."""
        tx_hash = entry.transaction.get_hash().to_hex()
        priority_score = entry.get_priority_score()
        heapq.heappush(self._priority_queue, (-priority_score, tx_hash))

    def _remove_from_priority_queue(self, tx_hash: str) -> None:
        """Remove transaction from priority queue."""
        # Note: This is inefficient for large queues
        # In production, consider using a more sophisticated data structure
        self._priority_queue = [
            (score, h) for score, h in self._priority_queue if h != tx_hash
        ]
        heapq.heapify(self._priority_queue)

    def _add_to_fee_queue(self, entry: TransactionEntry) -> None:
        """Add transaction to fee rate queue."""
        tx_hash = entry.transaction.get_hash().to_hex()
        heapq.heappush(self._by_fee_rate, (-entry.fee_rate, tx_hash))

    def _remove_from_fee_queue(self, tx_hash: str) -> None:
        """Remove transaction from fee rate queue."""
        self._by_fee_rate = [
            (score, h) for score, h in self._by_fee_rate if h != tx_hash
        ]
        heapq.heapify(self._by_fee_rate)

    def _update_address_index(self, entry: TransactionEntry) -> None:
        """Update address index for transaction."""
        tx_hash = entry.transaction.get_hash().to_hex()

        # Add addresses from inputs and outputs
        for inp in entry.transaction.inputs:
            if inp.public_key:
                address = inp.public_key.to_address()
                self._by_address[address].add(tx_hash)

        for output in entry.transaction.outputs:
            self._by_address[output.recipient_address].add(tx_hash)

    def _remove_from_address_index(self, entry: TransactionEntry) -> None:
        """Remove transaction from address index."""
        tx_hash = entry.transaction.get_hash().to_hex()

        for address_set in self._by_address.values():
            address_set.discard(tx_hash)

    def _update_conflicts(self, entry: TransactionEntry) -> None:
        """Update conflict tracking for transaction."""
        tx_hash = entry.transaction.get_hash().to_hex()
        tx_inputs = {
            f"{inp.previous_tx_hash.to_hex()}:{inp.output_index}"
            for inp in entry.transaction.inputs
        }

        for other_hash, other_entry in self._transactions.items():
            if other_hash == tx_hash:
                continue

            other_inputs = {
                f"{inp.previous_tx_hash.to_hex()}:{inp.output_index}"
                for inp in other_entry.transaction.inputs
            }
            if tx_inputs & other_inputs:
                self._conflicts[tx_hash].add(other_hash)
                self._conflicts[other_hash].add(tx_hash)

    def _remove_from_conflicts(self, entry: TransactionEntry) -> None:
        """Remove transaction from conflict tracking."""
        tx_hash = entry.transaction.get_hash().to_hex()

        # Remove from other transactions' conflict sets
        for other_hash in self._conflicts.get(tx_hash, set()):
            self._conflicts[other_hash].discard(tx_hash)

        # Remove this transaction's conflict set
        self._conflicts.pop(tx_hash, None)

    def _update_fee_estimates(self, entry: TransactionEntry) -> None:
        """Update fee estimates based on new transaction."""
        self._fee_history.append((entry.fee_rate, time.time()))

        # Calculate estimates for different confirmation targets
        for target_blocks in self.fee_estimate_blocks:
            # Simple implementation - in production, use more sophisticated algorithms
            recent_fees = [
                fee
                for fee, timestamp in self._fee_history
                if time.time() - timestamp < 3600
            ]
            if recent_fees:
                # Use percentile-based estimation
                recent_fees.sort()
                percentile = min(
                    90, 100 - (target_blocks * 2)
                )  # Higher blocks = lower percentile
                index = int(len(recent_fees) * percentile / 100)
                estimated_fee = (
                    recent_fees[index] if index < len(recent_fees) else recent_fees[-1]
                )

                self._fee_estimates[target_blocks] = FeeEstimate(
                    target_blocks=target_blocks,
                    fee_rate=estimated_fee,
                    confidence=self._calculate_fee_confidence(
                        estimated_fee, target_blocks
                    ),  
                )

    def _evict_low_priority(self) -> bool:
        """Evict low priority transactions to make room."""
        # Remove transactions with lowest priority scores
        evicted = 0
        target_evictions = max(1, len(self._transactions) // 10)  # Evict 10%

        # Sort by priority (lowest first)
        sorted_txs = sorted(
            self._transactions.items(), key=lambda x: x[1].get_priority_score()
        )

        for tx_hash, _ in sorted_txs[:target_evictions]:
            if self.remove_transaction(tx_hash):
                evicted += 1

        return evicted > 0

    def _evict_memory(self) -> bool:
        """Evict transactions to free memory."""
        # Remove oldest transactions first
        evicted = 0
        target_memory = self.max_memory_mb * 1024 * 1024 * 0.8  # Target 80% of max

        # Sort by age (oldest first)
        sorted_txs = sorted(self._transactions.items(), key=lambda x: x[1].first_seen)

        for tx_hash, _ in sorted_txs:
            if self._memory_usage <= target_memory:
                break

            if self.remove_transaction(tx_hash):
                evicted += 1

        return evicted > 0

    def _cleanup_worker(self) -> None:
        """Background worker for cleanup tasks."""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)

                if not self._running:
                    break

                # Clear expired transactions
                expired_count = self.clear_expired_transactions()

                # Clean validation cache
                current_time = time.time()
                expired_cache = [
                    tx_hash
                    for tx_hash, (_, timestamp) in self._validation_cache.items()
                    if current_time - timestamp > self.validation_cache_ttl
                ]
                for tx_hash in expired_cache:
                    self._validation_cache.pop(tx_hash, None)

                # Update fee estimates
                self._update_fee_estimates_from_history()

            except Exception as e:
                # Log error in production
                logger.info(f"Transaction pool cleanup error: {e}")

    def force_cleanup(self) -> None:
        """Force cleanup of expired transactions (for testing)."""
        with self._lock:
            # Clear expired transactions
            self.clear_expired_transactions()

            # Clean validation cache
            current_time = time.time()
            expired_cache = [
                tx_hash
                for tx_hash, (_, timestamp) in self._validation_cache.items()
                if current_time - timestamp > self.validation_cache_ttl
            ]
            for tx_hash in expired_cache:
                self._validation_cache.pop(tx_hash, None)

    def _update_fee_estimates_from_history(self) -> None:
        """Update fee estimates from historical data."""
        if not self._fee_history:
            return

        current_time = time.time()
        recent_fees = [
            fee
            for fee, timestamp in self._fee_history
            if current_time - timestamp < 3600
        ]

        if recent_fees:
            recent_fees.sort()
            for target_blocks in self.fee_estimate_blocks:
                percentile = min(90, 100 - (target_blocks * 2))
                index = int(len(recent_fees) * percentile / 100)
                estimated_fee = (
                    recent_fees[index] if index < len(recent_fees) else recent_fees[-1]
                )

                self._fee_estimates[target_blocks] = FeeEstimate(
                    target_blocks=target_blocks, fee_rate=estimated_fee, confidence=0.8
                )

    def __len__(self) -> int:
        """Get the number of transactions in the pool."""
        return len(self._transactions)

    def _calculate_fee_confidence(self, fee_rate: float, target_blocks: int) -> float:
        """
        Calculate confidence level for fee estimate.

        TODO: Implement actual confidence calculation
        This would involve:
        1. Analyzing historical fee data
        2. Considering network congestion
        3. Factoring in target block time
        4. Using statistical models for prediction
        """
        # For now, return a basic confidence based on target blocks
        if target_blocks <= 1:
            return 0.9
        elif target_blocks <= 6:
            return 0.8
        else:
            return 0.7

    def __contains__(self, tx_hash: str) -> bool:
        """Check if transaction is in the pool."""
        return tx_hash in self._transactions

    def __iter__(self):
        """Iterate over transactions in the pool."""
        return iter(self._transactions.values())
