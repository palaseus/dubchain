"""Transaction isolation and concurrency control for DubChain database.

This module provides advanced transaction isolation mechanisms including
MVCC (Multi-Version Concurrency Control), deadlock detection, and
transaction conflict resolution.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .database import DatabaseBackend, IsolationLevel


class IsolationError(Exception):
    """Base exception for isolation operations."""

    pass


class DeadlockError(IsolationError):
    """Deadlock detection error."""

    pass


class TransactionConflictError(IsolationError):
    """Transaction conflict error."""

    pass


class TransactionTimeoutError(IsolationError):
    """Transaction timeout error."""

    pass


class TransactionState(Enum):
    """Transaction state."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    WAITING = "waiting"


class LockType(Enum):
    """Lock types."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"
    INTENT_SHARED = "intent_shared"
    INTENT_EXCLUSIVE = "intent_exclusive"
    SHARED_INTENT_EXCLUSIVE = "shared_intent_exclusive"


class LockMode(Enum):
    """Lock modes."""

    READ = "read"
    WRITE = "write"
    UPDATE = "update"


@dataclass
class Lock:
    """Database lock."""

    lock_id: str
    transaction_id: str
    resource: str
    lock_type: LockType
    mode: LockMode
    acquired_at: float
    timeout: float = 30.0
    is_granted: bool = False
    waiting_transactions: Set[str] = field(default_factory=set)


@dataclass
class Transaction:
    """Database transaction."""

    transaction_id: str
    state: TransactionState
    isolation_level: IsolationLevel
    started_at: float
    timeout: float = 30.0
    read_set: Set[str] = field(default_factory=set)
    write_set: Set[str] = field(default_factory=set)
    locks: Dict[str, Lock] = field(default_factory=dict)
    waiting_for: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class DeadlockInfo:
    """Deadlock information."""

    cycle: List[str]
    transactions: List[Transaction]
    detected_at: float
    resolution_strategy: str = "abort_youngest"


class LockManager:
    """Database lock manager."""

    def __init__(self):
        self._locks: Dict[str, List[Lock]] = {}  # resource -> locks
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    def acquire_lock(
        self,
        transaction_id: str,
        resource: str,
        lock_type: LockType,
        mode: LockMode,
        timeout: float = 30.0,
    ) -> bool:
        """Acquire a lock on a resource."""
        with self._lock:
            lock_id = str(uuid.uuid4())
            lock = Lock(
                lock_id=lock_id,
                transaction_id=transaction_id,
                resource=resource,
                lock_type=lock_type,
                mode=mode,
                acquired_at=time.time(),
                timeout=timeout,
            )

            # Check if lock can be granted
            if self._can_grant_lock(lock):
                lock.is_granted = True
                self._add_lock(lock)
                self._logger.debug(f"Lock granted: {lock_id} for {resource}")
                return True
            else:
                # Add to waiting queue
                self._add_waiting_lock(lock)
                self._logger.debug(f"Lock queued: {lock_id} for {resource}")
                return False

    def release_lock(self, transaction_id: str, resource: str) -> None:
        """Release a lock on a resource."""
        with self._lock:
            if resource in self._locks:
                # Remove locks for this transaction
                locks_to_remove = []
                for lock in self._locks[resource]:
                    if lock.transaction_id == transaction_id:
                        locks_to_remove.append(lock)

                for lock in locks_to_remove:
                    self._locks[resource].remove(lock)

                # Grant waiting locks
                self._grant_waiting_locks(resource)

                self._logger.debug(f"Lock released: {transaction_id} for {resource}")

    def release_all_locks(self, transaction_id: str) -> None:
        """Release all locks for a transaction."""
        with self._lock:
            resources_to_update = []

            for resource, locks in self._locks.items():
                locks_to_remove = []
                for lock in locks:
                    if lock.transaction_id == transaction_id:
                        locks_to_remove.append(lock)

                if locks_to_remove:
                    for lock in locks_to_remove:
                        locks.remove(lock)
                    resources_to_update.append(resource)

            # Grant waiting locks for affected resources
            for resource in resources_to_update:
                self._grant_waiting_locks(resource)

            self._logger.debug(f"All locks released for transaction: {transaction_id}")

    def _can_grant_lock(self, new_lock: Lock) -> bool:
        """Check if a lock can be granted."""
        resource = new_lock.resource

        if resource not in self._locks:
            return True

        existing_locks = self._locks[resource]

        for existing_lock in existing_locks:
            if existing_lock.transaction_id == new_lock.transaction_id:
                # Same transaction - check for lock upgrade
                if self._can_upgrade_lock(existing_lock, new_lock):
                    return True
                else:
                    return False

            # Check for lock compatibility
            if not self._are_locks_compatible(existing_lock, new_lock):
                return False

        return True

    def _can_upgrade_lock(self, existing_lock: Lock, new_lock: Lock) -> bool:
        """Check if a lock can be upgraded."""
        # Can upgrade from shared to exclusive
        if (
            existing_lock.lock_type == LockType.SHARED
            and new_lock.lock_type == LockType.EXCLUSIVE
        ):
            return True

        # Can upgrade from intent shared to intent exclusive
        if (
            existing_lock.lock_type == LockType.INTENT_SHARED
            and new_lock.lock_type == LockType.INTENT_EXCLUSIVE
        ):
            return True

        return False

    def _are_locks_compatible(self, lock1: Lock, lock2: Lock) -> bool:
        """Check if two locks are compatible."""
        # Same transaction
        if lock1.transaction_id == lock2.transaction_id:
            return True

        # Lock compatibility matrix
        compatibility = {
            (LockType.SHARED, LockType.SHARED): True,
            (LockType.SHARED, LockType.EXCLUSIVE): False,
            (LockType.SHARED, LockType.INTENT_SHARED): True,
            (LockType.SHARED, LockType.INTENT_EXCLUSIVE): False,
            (LockType.EXCLUSIVE, LockType.SHARED): False,
            (LockType.EXCLUSIVE, LockType.EXCLUSIVE): False,
            (LockType.EXCLUSIVE, LockType.INTENT_SHARED): False,
            (LockType.EXCLUSIVE, LockType.INTENT_EXCLUSIVE): False,
            (LockType.INTENT_SHARED, LockType.SHARED): True,
            (LockType.INTENT_SHARED, LockType.EXCLUSIVE): False,
            (LockType.INTENT_SHARED, LockType.INTENT_SHARED): True,
            (LockType.INTENT_SHARED, LockType.INTENT_EXCLUSIVE): True,
            (LockType.INTENT_EXCLUSIVE, LockType.SHARED): False,
            (LockType.INTENT_EXCLUSIVE, LockType.EXCLUSIVE): False,
            (LockType.INTENT_EXCLUSIVE, LockType.INTENT_SHARED): True,
            (LockType.INTENT_EXCLUSIVE, LockType.INTENT_EXCLUSIVE): True,
        }

        return compatibility.get((lock1.lock_type, lock2.lock_type), False)

    def _add_lock(self, lock: Lock) -> None:
        """Add a granted lock."""
        resource = lock.resource
        if resource not in self._locks:
            self._locks[resource] = []

        self._locks[resource].append(lock)

    def _add_waiting_lock(self, lock: Lock) -> None:
        """Add a waiting lock."""
        resource = lock.resource
        if resource not in self._locks:
            self._locks[resource] = []

        # Add to waiting queue
        self._locks[resource].append(lock)

        # Add to waiting transactions of existing locks
        for existing_lock in self._locks[resource]:
            if existing_lock.is_granted:
                existing_lock.waiting_transactions.add(lock.transaction_id)

    def _grant_waiting_locks(self, resource: str) -> None:
        """Grant waiting locks for a resource."""
        if resource not in self._locks:
            return

        waiting_locks = [lock for lock in self._locks[resource] if not lock.is_granted]

        for lock in waiting_locks:
            if self._can_grant_lock(lock):
                lock.is_granted = True
                self._logger.debug(f"Waiting lock granted: {lock.lock_id}")

    def get_locks_for_transaction(self, transaction_id: str) -> List[Lock]:
        """Get all locks for a transaction."""
        with self._lock:
            locks = []
            for resource_locks in self._locks.values():
                for lock in resource_locks:
                    if lock.transaction_id == transaction_id:
                        locks.append(lock)
            return locks

    def detect_deadlock(
        self, transactions: Dict[str, Transaction]
    ) -> Optional[DeadlockInfo]:
        """Detect deadlock in the system."""
        with self._lock:
            # Build wait-for graph
            wait_for_graph = {}

            for transaction_id, transaction in transactions.items():
                if transaction.waiting_for:
                    wait_for_graph[transaction_id] = transaction.waiting_for

            # Detect cycles using DFS
            visited = set()
            rec_stack = set()

            def has_cycle(node):
                visited.add(node)
                rec_stack.add(node)

                if node in wait_for_graph:
                    neighbor = wait_for_graph[node]
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            # Check for cycles
            for transaction_id in transactions:
                if transaction_id not in visited:
                    if has_cycle(transaction_id):
                        # Found a cycle - build deadlock info
                        cycle = self._extract_cycle(wait_for_graph, transaction_id)
                        cycle_transactions = [
                            transactions[tid] for tid in cycle if tid in transactions
                        ]

                        return DeadlockInfo(
                            cycle=cycle,
                            transactions=cycle_transactions,
                            detected_at=time.time(),
                        )

            return None

    def _extract_cycle(
        self, wait_for_graph: Dict[str, str], start_node: str
    ) -> List[str]:
        """Extract cycle from wait-for graph."""
        cycle = []
        current = start_node
        visited = set()

        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = wait_for_graph.get(current)

            if current is None:
                break

        return cycle


class TransactionIsolation:
    """Transaction isolation manager."""

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self.lock_manager = LockManager()
        self._transactions: Dict[str, Transaction] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        self._deadlock_detection_interval = 5.0  # seconds
        self._deadlock_detection_thread: Optional[threading.Thread] = None
        self._running = False

    def begin_transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout: float = 30.0,
    ) -> str:
        """Begin a new transaction."""
        with self._lock:
            transaction_id = str(uuid.uuid4())

            transaction = Transaction(
                transaction_id=transaction_id,
                state=TransactionState.ACTIVE,
                isolation_level=isolation_level,
                started_at=time.time(),
                timeout=timeout,
            )

            self._transactions[transaction_id] = transaction

            self._logger.debug(f"Started transaction: {transaction_id}")

            return transaction_id

    def commit_transaction(self, transaction_id: str) -> None:
        """Commit a transaction."""
        with self._lock:
            if transaction_id not in self._transactions:
                raise IsolationError(f"Transaction {transaction_id} not found")

            transaction = self._transactions[transaction_id]

            if transaction.state != TransactionState.ACTIVE:
                raise IsolationError(f"Transaction {transaction_id} is not active")

            try:
                # Release all locks
                self.lock_manager.release_all_locks(transaction_id)

                # Update transaction state
                transaction.state = TransactionState.COMMITTED

                # Remove from active transactions
                del self._transactions[transaction_id]

                self._logger.debug(f"Committed transaction: {transaction_id}")

            except Exception as e:
                transaction.state = TransactionState.ABORTED
                raise IsolationError(f"Transaction commit failed: {e}")

    def abort_transaction(self, transaction_id: str) -> None:
        """Abort a transaction."""
        with self._lock:
            if transaction_id not in self._transactions:
                raise IsolationError(f"Transaction {transaction_id} not found")

            transaction = self._transactions[transaction_id]

            try:
                # Release all locks
                self.lock_manager.release_all_locks(transaction_id)

                # Update transaction state
                transaction.state = TransactionState.ABORTED

                # Remove from active transactions
                del self._transactions[transaction_id]

                self._logger.debug(f"Aborted transaction: {transaction_id}")

            except Exception as e:
                self._logger.error(f"Transaction abort failed: {e}")

    def read_with_isolation(
        self,
        transaction_id: str,
        resource: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Read data with proper isolation."""
        with self._lock:
            if transaction_id not in self._transactions:
                raise IsolationError(f"Transaction {transaction_id} not found")

            transaction = self._transactions[transaction_id]

            if transaction.state != TransactionState.ACTIVE:
                raise IsolationError(f"Transaction {transaction_id} is not active")

            # Check timeout
            if time.time() - transaction.started_at > transaction.timeout:
                self.abort_transaction(transaction_id)
                raise TransactionTimeoutError(f"Transaction {transaction_id} timed out")

            # Acquire appropriate lock based on isolation level
            if transaction.isolation_level in [
                IsolationLevel.READ_COMMITTED,
                IsolationLevel.REPEATABLE_READ,
            ]:
                lock_type = LockType.SHARED
            else:
                lock_type = LockType.SHARED

            # Try to acquire lock
            if not self.lock_manager.acquire_lock(
                transaction_id, resource, lock_type, LockMode.READ, transaction.timeout
            ):
                # Handle lock timeout
                transaction.state = TransactionState.BLOCKED
                raise TransactionTimeoutError(f"Lock timeout for resource {resource}")

            # Add to read set
            transaction.read_set.add(resource)

            # Execute query
            try:
                result = self.backend.execute_query(query, params)
                return result
            except Exception as e:
                self.abort_transaction(transaction_id)
                raise IsolationError(f"Read operation failed: {e}")

    def write_with_isolation(
        self,
        transaction_id: str,
        resource: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Write data with proper isolation."""
        with self._lock:
            if transaction_id not in self._transactions:
                raise IsolationError(f"Transaction {transaction_id} not found")

            transaction = self._transactions[transaction_id]

            if transaction.state != TransactionState.ACTIVE:
                raise IsolationError(f"Transaction {transaction_id} is not active")

            # Check timeout
            if time.time() - transaction.started_at > transaction.timeout:
                self.abort_transaction(transaction_id)
                raise TransactionTimeoutError(f"Transaction {transaction_id} timed out")

            # Acquire exclusive lock
            if not self.lock_manager.acquire_lock(
                transaction_id,
                resource,
                LockType.EXCLUSIVE,
                LockMode.WRITE,
                transaction.timeout,
            ):
                # Handle lock timeout
                transaction.state = TransactionState.BLOCKED
                raise TransactionTimeoutError(f"Lock timeout for resource {resource}")

            # Add to write set
            transaction.write_set.add(resource)

            # Execute query
            try:
                result = self.backend.execute_query(query, params)
                return result
            except Exception as e:
                self.abort_transaction(transaction_id)
                raise IsolationError(f"Write operation failed: {e}")

    def start_deadlock_detection(self) -> None:
        """Start deadlock detection thread."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._deadlock_detection_thread = threading.Thread(
                target=self._deadlock_detection_worker, daemon=True
            )
            self._deadlock_detection_thread.start()

            self._logger.info("Started deadlock detection")

    def stop_deadlock_detection(self) -> None:
        """Stop deadlock detection thread."""
        with self._lock:
            self._running = False
            if self._deadlock_detection_thread:
                self._deadlock_detection_thread.join(timeout=5.0)

            self._logger.info("Stopped deadlock detection")

    def _deadlock_detection_worker(self) -> None:
        """Background deadlock detection worker."""
        while self._running:
            try:
                time.sleep(self._deadlock_detection_interval)

                if not self._running:
                    break

                # Detect deadlocks
                deadlock_info = self.lock_manager.detect_deadlock(self._transactions)

                if deadlock_info:
                    self._resolve_deadlock(deadlock_info)

            except Exception as e:
                self._logger.error(f"Deadlock detection error: {e}")

    def _resolve_deadlock(self, deadlock_info: DeadlockInfo) -> None:
        """Resolve a detected deadlock."""
        with self._lock:
            self._logger.warning(f"Deadlock detected: {deadlock_info.cycle}")

            # Abort the youngest transaction in the cycle
            youngest_transaction = None
            youngest_time = 0

            for transaction in deadlock_info.transactions:
                if transaction.started_at > youngest_time:
                    youngest_time = transaction.started_at
                    youngest_transaction = transaction

            if youngest_transaction:
                self.abort_transaction(youngest_transaction.transaction_id)
                self._logger.info(
                    f"Aborted transaction {youngest_transaction.transaction_id} to resolve deadlock"
                )

    def get_transaction_info(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction information."""
        with self._lock:
            return self._transactions.get(transaction_id)

    def get_active_transactions(self) -> List[Transaction]:
        """Get all active transactions."""
        with self._lock:
            return [
                t
                for t in self._transactions.values()
                if t.state == TransactionState.ACTIVE
            ]

    def get_lock_info(self) -> Dict[str, List[Lock]]:
        """Get current lock information."""
        return self.lock_manager._locks.copy()

    def __enter__(self):
        """Context manager entry."""
        self.start_deadlock_detection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_deadlock_detection()

        # Abort any remaining transactions
        for transaction_id in list(self._transactions.keys()):
            self.abort_transaction(transaction_id)
