"""
Database Isolation and Transaction Management for DubChain

This module provides comprehensive isolation capabilities including:
- Transaction isolation levels
- Lock management
- Deadlock detection and resolution
- MVCC (Multi-Version Concurrency Control)
- Snapshot isolation
- Read/write isolation
"""

import logging

logger = logging.getLogger(__name__)
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque

from ..errors import ClientError
from ..logging import get_logger

logger = get_logger(__name__)

class IsolationLevel(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"
    SNAPSHOT = "snapshot"

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

class TransactionStatus(Enum):
    """Transaction status."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"
    BLOCKED = "blocked"

@dataclass
class Lock:
    """Lock information."""
    lock_id: str
    resource_id: str
    transaction_id: str
    lock_type: LockType
    granted_at: float
    timeout: Optional[float] = None

@dataclass
class Transaction:
    """Transaction information."""
    transaction_id: str
    isolation_level: IsolationLevel
    status: TransactionStatus
    started_at: float
    committed_at: Optional[float] = None
    aborted_at: Optional[float] = None
    locks: Set[str] = field(default_factory=set)
    snapshot_version: Optional[int] = None
    read_set: Set[str] = field(default_factory=set)
    write_set: Set[str] = field(default_factory=set)

@dataclass
class IsolationConfig:
    """Isolation configuration."""
    default_isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    lock_timeout: float = 30.0  # seconds
    deadlock_detection_interval: float = 1.0  # seconds
    max_transaction_age: float = 300.0  # seconds
    enable_mvcc: bool = True
    enable_snapshot_isolation: bool = True
    max_concurrent_transactions: int = 1000

class LockManager:
    """Lock manager for resource locking."""
    
    def __init__(self, config: IsolationConfig):
        """Initialize lock manager."""
        self.config = config
        self.locks: Dict[str, Lock] = {}
        self.resource_locks: Dict[str, List[str]] = defaultdict(list)
        self.transaction_locks: Dict[str, Set[str]] = defaultdict(set)
        self.lock_queue: Dict[str, deque] = defaultdict(deque)
        self.lock_mutex = threading.RLock()
        
        logger.info("Initialized lock manager")
    
    def acquire_lock(self, transaction_id: str, resource_id: str, 
                    lock_type: LockType, timeout: Optional[float] = None) -> bool:
        """Acquire a lock on a resource."""
        try:
            with self.lock_mutex:
                timeout = timeout or self.config.lock_timeout
                lock_id = str(uuid.uuid4())
                
                # Check if lock can be granted immediately
                if self._can_grant_lock(resource_id, lock_type):
                    lock = Lock(
                        lock_id=lock_id,
                        resource_id=resource_id,
                        transaction_id=transaction_id,
                        lock_type=lock_type,
                        granted_at=time.time()
                    )
                    
                    self.locks[lock_id] = lock
                    self.resource_locks[resource_id].append(lock_id)
                    self.transaction_locks[transaction_id].add(lock_id)
                    
                    logger.debug(f"Granted lock {lock_id} to transaction {transaction_id}")
                    return True
                
                # Add to lock queue
                self.lock_queue[resource_id].append({
                    'transaction_id': transaction_id,
                    'lock_type': lock_type,
                    'timeout': time.time() + timeout,
                    'lock_id': lock_id
                })
                
                logger.debug(f"Queued lock request for transaction {transaction_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            return False
    
    def release_lock(self, transaction_id: str, resource_id: str) -> bool:
        """Release a lock on a resource."""
        try:
            with self.lock_mutex:
                # Find locks to release
                locks_to_release = []
                for lock_id in self.transaction_locks[transaction_id]:
                    lock = self.locks.get(lock_id)
                    if lock and lock.resource_id == resource_id:
                        locks_to_release.append(lock_id)
                
                # Release locks
                for lock_id in locks_to_release:
                    self._release_single_lock(lock_id)
                
                # Process lock queue
                self._process_lock_queue(resource_id)
                
                logger.debug(f"Released locks for transaction {transaction_id} on resource {resource_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            return False
    
    def release_all_locks(self, transaction_id: str) -> bool:
        """Release all locks for a transaction."""
        try:
            with self.lock_mutex:
                locks_to_release = list(self.transaction_locks[transaction_id])
                
                for lock_id in locks_to_release:
                    self._release_single_lock(lock_id)
                
                logger.debug(f"Released all locks for transaction {transaction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error releasing all locks: {e}")
            return False
    
    def _can_grant_lock(self, resource_id: str, lock_type: LockType) -> bool:
        """Check if a lock can be granted."""
        existing_locks = self.resource_locks.get(resource_id, [])
        
        if not existing_locks:
            return True
        
        # Check compatibility
        for lock_id in existing_locks:
            existing_lock = self.locks.get(lock_id)
            if existing_lock and not self._are_locks_compatible(existing_lock.lock_type, lock_type):
                return False
        
        return True
    
    def _are_locks_compatible(self, existing_type: LockType, requested_type: LockType) -> bool:
        """Check if two lock types are compatible."""
        compatibility_matrix = {
            LockType.SHARED: {LockType.SHARED, LockType.INTENT_SHARED},
            LockType.EXCLUSIVE: set(),
            LockType.INTENT_SHARED: {LockType.SHARED, LockType.INTENT_SHARED, LockType.INTENT_EXCLUSIVE},
            LockType.INTENT_EXCLUSIVE: {LockType.INTENT_SHARED},
            LockType.SHARED_INTENT_EXCLUSIVE: {LockType.INTENT_SHARED}
        }
        
        return requested_type in compatibility_matrix.get(existing_type, set())
    
    def _release_single_lock(self, lock_id: str) -> None:
        """Release a single lock."""
        if lock_id not in self.locks:
            return
        
        lock = self.locks[lock_id]
        
        # Remove from resource locks
        if lock.resource_id in self.resource_locks:
            try:
                self.resource_locks[lock.resource_id].remove(lock_id)
            except ValueError:
                pass
        
        # Remove from transaction locks
        if lock.transaction_id in self.transaction_locks:
            self.transaction_locks[lock.transaction_id].discard(lock_id)
        
        # Remove lock
        del self.locks[lock_id]
    
    def _process_lock_queue(self, resource_id: str) -> None:
        """Process lock queue for a resource."""
        queue = self.lock_queue.get(resource_id, deque())
        
        while queue:
            request = queue[0]
            
            # Check timeout
            if time.time() > request['timeout']:
                queue.popleft()
                continue
            
            # Check if lock can be granted
            if self._can_grant_lock(resource_id, request['lock_type']):
                lock = Lock(
                    lock_id=request['lock_id'],
                    resource_id=resource_id,
                    transaction_id=request['transaction_id'],
                    lock_type=request['lock_type'],
                    granted_at=time.time()
                )
                
                self.locks[request['lock_id']] = lock
                self.resource_locks[resource_id].append(request['lock_id'])
                self.transaction_locks[request['transaction_id']].add(request['lock_id'])
                
                queue.popleft()
                
                logger.debug(f"Granted queued lock {request['lock_id']}")
            else:
                break

class DeadlockDetector:
    """Deadlock detection and resolution."""
    
    def __init__(self, lock_manager: LockManager):
        """Initialize deadlock detector."""
        self.lock_manager = lock_manager
        self.running = False
        self.detection_thread = None
        
        logger.info("Initialized deadlock detector")
    
    def start(self) -> None:
        """Start deadlock detection."""
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("Deadlock detector started")
    
    def stop(self) -> None:
        """Stop deadlock detection."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        logger.info("Deadlock detector stopped")
    
    def _detection_loop(self) -> None:
        """Deadlock detection loop."""
        while self.running:
            try:
                time.sleep(1.0)  # Check every second
                
                if self.running:
                    deadlocks = self._detect_deadlocks()
                    
                    for deadlock in deadlocks:
                        self._resolve_deadlock(deadlock)
                
            except Exception as e:
                logger.error(f"Error in deadlock detection loop: {e}")
    
    def _detect_deadlocks(self) -> List[List[str]]:
        """Detect deadlocks using wait-for graph."""
        try:
            # Build wait-for graph
            wait_for_graph = self._build_wait_for_graph()
            
            # Find cycles
            deadlocks = []
            visited = set()
            
            for transaction_id in wait_for_graph:
                if transaction_id not in visited:
                    cycle = self._find_cycle(wait_for_graph, transaction_id, visited)
                    if cycle:
                        deadlocks.append(cycle)
            
            return deadlocks
            
        except Exception as e:
            logger.error(f"Error detecting deadlocks: {e}")
            return []
    
    def _build_wait_for_graph(self) -> Dict[str, Set[str]]:
        """Build wait-for graph."""
        wait_for_graph = defaultdict(set)
        
        with self.lock_manager.lock_mutex:
            # For each resource, find transactions waiting for locks held by other transactions
            for resource_id, lock_ids in self.lock_manager.resource_locks.items():
                if len(lock_ids) > 1:
                    # Find transactions holding locks
                    holding_transactions = set()
                    for lock_id in lock_ids:
                        lock = self.lock_manager.locks.get(lock_id)
                        if lock:
                            holding_transactions.add(lock.transaction_id)
                    
                    # Find transactions waiting for locks
                    queue = self.lock_manager.lock_queue.get(resource_id, deque())
                    for request in queue:
                        waiting_transaction = request['transaction_id']
                        for holding_transaction in holding_transactions:
                            if waiting_transaction != holding_transaction:
                                wait_for_graph[waiting_transaction].add(holding_transaction)
        
        return dict(wait_for_graph)
    
    def _find_cycle(self, graph: Dict[str, Set[str]], start: str, visited: Set[str]) -> Optional[List[str]]:
        """Find cycle in graph starting from start node."""
        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return None
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, set()):
                cycle = dfs(neighbor, path)
                if cycle:
                    return cycle
            
            path.pop()
            return None
        
        return dfs(start, [])
    
    def _resolve_deadlock(self, deadlock: List[str]) -> None:
        """Resolve deadlock by aborting a transaction."""
        try:
            # Choose transaction to abort (simplified: abort the first one)
            transaction_to_abort = deadlock[0]
            
            logger.warning(f"Deadlock detected, aborting transaction {transaction_to_abort}")
            
            # Abort transaction (in real implementation, would notify transaction manager)
            # For now, just release all locks
            self.lock_manager.release_all_locks(transaction_to_abort)
            
        except Exception as e:
            logger.error(f"Error resolving deadlock: {e}")

class MVCCManager:
    """Multi-Version Concurrency Control manager."""
    
    def __init__(self, config: IsolationConfig):
        """Initialize MVCC manager."""
        self.config = config
        self.version_counter = 0
        self.data_versions: Dict[str, Dict[int, Any]] = defaultdict(dict)
        self.transaction_snapshots: Dict[str, int] = {}
        self.mvcc_mutex = threading.RLock()
        
        logger.info("Initialized MVCC manager")
    
    def create_snapshot(self, transaction_id: str) -> int:
        """Create snapshot for transaction."""
        with self.mvcc_mutex:
            self.version_counter += 1
            snapshot_version = self.version_counter
            self.transaction_snapshots[transaction_id] = snapshot_version
            
            logger.debug(f"Created snapshot {snapshot_version} for transaction {transaction_id}")
            return snapshot_version
    
    def read_data(self, transaction_id: str, resource_id: str) -> Optional[Any]:
        """Read data using MVCC."""
        with self.mvcc_mutex:
            snapshot_version = self.transaction_snapshots.get(transaction_id)
            if not snapshot_version:
                return None
            
            # Find the latest version visible to this transaction
            versions = self.data_versions.get(resource_id, {})
            
            for version in sorted(versions.keys(), reverse=True):
                if version <= snapshot_version:
                    return versions[version]
            
            return None
    
    def write_data(self, transaction_id: str, resource_id: str, data: Any) -> None:
        """Write data using MVCC."""
        with self.mvcc_mutex:
            self.version_counter += 1
            new_version = self.version_counter
            
            self.data_versions[resource_id][new_version] = data
            
            logger.debug(f"Wrote data version {new_version} for resource {resource_id}")
    
    def commit_transaction(self, transaction_id: str) -> None:
        """Commit transaction and clean up old versions."""
        with self.mvcc_mutex:
            if transaction_id in self.transaction_snapshots:
                del self.transaction_snapshots[transaction_id]
            
            # Clean up old versions (simplified)
            self._cleanup_old_versions()
            
            logger.debug(f"Committed transaction {transaction_id}")
    
    def abort_transaction(self, transaction_id: str) -> None:
        """Abort transaction."""
        with self.mvcc_mutex:
            if transaction_id in self.transaction_snapshots:
                del self.transaction_snapshots[transaction_id]
            
            logger.debug(f"Aborted transaction {transaction_id}")
    
    def _cleanup_old_versions(self) -> None:
        """Clean up old data versions."""
        try:
            # Find the oldest active snapshot
            oldest_snapshot = min(self.transaction_snapshots.values()) if self.transaction_snapshots else self.version_counter
            
            # Remove versions older than the oldest active snapshot
            for resource_id in list(self.data_versions.keys()):
                versions = self.data_versions[resource_id]
                versions_to_remove = [v for v in versions.keys() if v < oldest_snapshot]
                
                for version in versions_to_remove:
                    del versions[version]
                
                # Remove empty resource entries
                if not versions:
                    del self.data_versions[resource_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")

class TransactionManager:
    """Transaction manager for isolation control."""
    
    def __init__(self, config: IsolationConfig):
        """Initialize transaction manager."""
        self.config = config
        self.transactions: Dict[str, Transaction] = {}
        self.lock_manager = LockManager(config)
        self.mvcc_manager = MVCCManager(config) if config.enable_mvcc else None
        self.deadlock_detector = DeadlockDetector(self.lock_manager)
        self.transaction_mutex = threading.RLock()
        
        # Start deadlock detection
        self.deadlock_detector.start()
        
        logger.info("Initialized transaction manager")
    
    def start_transaction(self, isolation_level: Optional[IsolationLevel] = None) -> str:
        """Start a new transaction."""
        try:
            with self.transaction_mutex:
                transaction_id = str(uuid.uuid4())
                isolation_level = isolation_level or self.config.default_isolation_level
                
                transaction = Transaction(
                    transaction_id=transaction_id,
                    isolation_level=isolation_level,
                    status=TransactionStatus.ACTIVE,
                    started_at=time.time()
                )
                
                # Create snapshot for MVCC
                if self.mvcc_manager and isolation_level in [IsolationLevel.SNAPSHOT, IsolationLevel.SERIALIZABLE]:
                    transaction.snapshot_version = self.mvcc_manager.create_snapshot(transaction_id)
                
                self.transactions[transaction_id] = transaction
                
                logger.debug(f"Started transaction {transaction_id} with isolation level {isolation_level}")
                return transaction_id
                
        except Exception as e:
            logger.error(f"Error starting transaction: {e}")
            raise ClientError(f"Failed to start transaction: {e}")
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction."""
        try:
            with self.transaction_mutex:
                if transaction_id not in self.transactions:
                    logger.error(f"Transaction {transaction_id} not found")
                    return False
                
                transaction = self.transactions[transaction_id]
                
                if transaction.status != TransactionStatus.ACTIVE:
                    logger.error(f"Transaction {transaction_id} is not active")
                    return False
                
                # Commit MVCC transaction
                if self.mvcc_manager:
                    self.mvcc_manager.commit_transaction(transaction_id)
                
                # Release all locks
                self.lock_manager.release_all_locks(transaction_id)
                
                # Update transaction status
                transaction.status = TransactionStatus.COMMITTED
                transaction.committed_at = time.time()
                
                logger.debug(f"Committed transaction {transaction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error committing transaction: {e}")
            return False
    
    def abort_transaction(self, transaction_id: str) -> bool:
        """Abort a transaction."""
        try:
            with self.transaction_mutex:
                if transaction_id not in self.transactions:
                    logger.error(f"Transaction {transaction_id} not found")
                    return False
                
                transaction = self.transactions[transaction_id]
                
                # Abort MVCC transaction
                if self.mvcc_manager:
                    self.mvcc_manager.abort_transaction(transaction_id)
                
                # Release all locks
                self.lock_manager.release_all_locks(transaction_id)
                
                # Update transaction status
                transaction.status = TransactionStatus.ABORTED
                transaction.aborted_at = time.time()
                
                logger.debug(f"Aborted transaction {transaction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error aborting transaction: {e}")
            return False
    
    def read_data(self, transaction_id: str, resource_id: str) -> Optional[Any]:
        """Read data within a transaction."""
        try:
            with self.transaction_mutex:
                if transaction_id not in self.transactions:
                    logger.error(f"Transaction {transaction_id} not found")
                    return None
                
                transaction = self.transactions[transaction_id]
                
                if transaction.status != TransactionStatus.ACTIVE:
                    logger.error(f"Transaction {transaction_id} is not active")
                    return None
                
                # Add to read set
                transaction.read_set.add(resource_id)
                
                # Use MVCC if available
                if self.mvcc_manager and transaction.isolation_level in [IsolationLevel.SNAPSHOT, IsolationLevel.SERIALIZABLE]:
                    return self.mvcc_manager.read_data(transaction_id, resource_id)
                
                # Otherwise, use locking
                if transaction.isolation_level in [IsolationLevel.READ_COMMITTED, IsolationLevel.REPEATABLE_READ]:
                    # Acquire shared lock
                    self.lock_manager.acquire_lock(transaction_id, resource_id, LockType.SHARED)
                
                # Simulate data read
                return {"data": f"data_for_{resource_id}", "version": transaction.snapshot_version}
                
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return None
    
    def write_data(self, transaction_id: str, resource_id: str, data: Any) -> bool:
        """Write data within a transaction."""
        try:
            with self.transaction_mutex:
                if transaction_id not in self.transactions:
                    logger.error(f"Transaction {transaction_id} not found")
                    return False
                
                transaction = self.transactions[transaction_id]
                
                if transaction.status != TransactionStatus.ACTIVE:
                    logger.error(f"Transaction {transaction_id} is not active")
                    return False
                
                # Add to write set
                transaction.write_set.add(resource_id)
                
                # Use MVCC if available
                if self.mvcc_manager:
                    self.mvcc_manager.write_data(transaction_id, resource_id, data)
                else:
                    # Acquire exclusive lock
                    self.lock_manager.acquire_lock(transaction_id, resource_id, LockType.EXCLUSIVE)
                
                logger.debug(f"Wrote data for resource {resource_id} in transaction {transaction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error writing data: {e}")
            return False
    
    def get_transaction_info(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction information."""
        return self.transactions.get(transaction_id)
    
    def cleanup_old_transactions(self) -> None:
        """Clean up old transactions."""
        try:
            with self.transaction_mutex:
                current_time = time.time()
                cutoff_time = current_time - self.config.max_transaction_age
                
                transactions_to_remove = []
                for transaction_id, transaction in self.transactions.items():
                    if transaction.started_at < cutoff_time:
                        transactions_to_remove.append(transaction_id)
                
                for transaction_id in transactions_to_remove:
                    # Abort old transaction
                    self.abort_transaction(transaction_id)
                    del self.transactions[transaction_id]
                
                if transactions_to_remove:
                    logger.info(f"Cleaned up {len(transactions_to_remove)} old transactions")
                
        except Exception as e:
            logger.error(f"Error cleaning up old transactions: {e}")
    
    def shutdown(self) -> None:
        """Shutdown transaction manager."""
        self.deadlock_detector.stop()
        logger.info("Transaction manager shutdown")

class TransactionIsolation:
    """Transaction isolation manager."""
    
    def __init__(self, level: IsolationLevel = IsolationLevel.READ_COMMITTED):
        """Initialize transaction isolation."""
        self.level = level
        self.lock_manager = LockManager()
        self.mvcc_manager = MVCCManager()
        logger.info(f"Initialized transaction isolation with level: {level}")
    
    def set_isolation_level(self, level: IsolationLevel) -> None:
        """Set the isolation level."""
        self.level = level
        logger.info(f"Changed isolation level to: {level}")
    
    def get_isolation_level(self) -> IsolationLevel:
        """Get the current isolation level."""
        return self.level

__all__ = [
    "TransactionManager",
    "LockManager",
    "MVCCManager",
    "DeadlockDetector",
    "Transaction",
    "Lock",
    "IsolationConfig",
    "IsolationLevel",
    "LockType",
    "LockMode",
    "TransactionStatus",
    "TransactionIsolation",
]