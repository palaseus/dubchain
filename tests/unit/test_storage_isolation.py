"""Tests for storage isolation module."""

import logging

logger = logging.getLogger(__name__)
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.storage.database import IsolationLevel
from dubchain.storage.isolation import (
    DeadlockError,
    DeadlockInfo,
    IsolationError,
    Lock,
    LockManager,
    LockMode,
    LockType,
    Transaction,
    TransactionConflictError,
    TransactionIsolation,
    TransactionState,
    TransactionTimeoutError,
)


class TestIsolationError:
    """Test IsolationError exception."""

    def test_isolation_error_creation(self):
        """Test creating isolation error."""
        error = IsolationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestDeadlockError:
    """Test DeadlockError exception."""

    def test_deadlock_error_creation(self):
        """Test creating deadlock error."""
        error = DeadlockError("Deadlock detected")
        assert str(error) == "Deadlock detected"
        assert isinstance(error, IsolationError)


class TestTransactionConflictError:
    """Test TransactionConflictError exception."""

    def test_transaction_conflict_error_creation(self):
        """Test creating transaction conflict error."""
        error = TransactionConflictError("Transaction conflict")
        assert str(error) == "Transaction conflict"
        assert isinstance(error, IsolationError)


class TestTransactionTimeoutError:
    """Test TransactionTimeoutError exception."""

    def test_transaction_timeout_error_creation(self):
        """Test creating transaction timeout error."""
        error = TransactionTimeoutError("Transaction timeout")
        assert str(error) == "Transaction timeout"
        assert isinstance(error, IsolationError)


class TestTransactionState:
    """Test TransactionState enum."""

    def test_transaction_state_values(self):
        """Test transaction state values."""
        assert TransactionState.ACTIVE.value == "active"
        assert TransactionState.COMMITTED.value == "committed"
        assert TransactionState.ABORTED.value == "aborted"
        assert TransactionState.BLOCKED.value == "blocked"
        assert TransactionState.WAITING.value == "waiting"

    def test_transaction_state_enumeration(self):
        """Test transaction state enumeration."""
        states = list(TransactionState)
        assert len(states) == 5
        assert TransactionState.ACTIVE in states
        assert TransactionState.COMMITTED in states
        assert TransactionState.ABORTED in states
        assert TransactionState.BLOCKED in states
        assert TransactionState.WAITING in states


class TestLockType:
    """Test LockType enum."""

    def test_lock_type_values(self):
        """Test lock type values."""
        assert LockType.SHARED.value == "shared"
        assert LockType.EXCLUSIVE.value == "exclusive"
        assert LockType.INTENT_SHARED.value == "intent_shared"
        assert LockType.INTENT_EXCLUSIVE.value == "intent_exclusive"
        assert LockType.SHARED_INTENT_EXCLUSIVE.value == "shared_intent_exclusive"

    def test_lock_type_enumeration(self):
        """Test lock type enumeration."""
        types = list(LockType)
        assert len(types) == 5
        assert LockType.SHARED in types
        assert LockType.EXCLUSIVE in types
        assert LockType.INTENT_SHARED in types
        assert LockType.INTENT_EXCLUSIVE in types
        assert LockType.SHARED_INTENT_EXCLUSIVE in types


class TestLockMode:
    """Test LockMode enum."""

    def test_lock_mode_values(self):
        """Test lock mode values."""
        assert LockMode.READ.value == "read"
        assert LockMode.WRITE.value == "write"
        assert LockMode.UPDATE.value == "update"

    def test_lock_mode_enumeration(self):
        """Test lock mode enumeration."""
        modes = list(LockMode)
        assert len(modes) == 3
        assert LockMode.READ in modes
        assert LockMode.WRITE in modes
        assert LockMode.UPDATE in modes


class TestLock:
    """Test Lock dataclass."""

    def test_lock_creation(self):
        """Test creating lock."""
        lock = Lock(
            lock_id="lock1",
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
            acquired_at=time.time(),
        )

        assert lock.lock_id == "lock1"
        assert lock.transaction_id == "tx1"
        assert lock.resource == "table1"
        assert lock.lock_type == LockType.SHARED
        assert lock.mode == LockMode.READ
        assert lock.timeout == 30.0
        assert lock.is_granted == False
        assert lock.waiting_transactions == set()

    def test_lock_defaults(self):
        """Test lock default values."""
        lock = Lock(
            lock_id="lock1",
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
            acquired_at=time.time(),
        )

        assert lock.timeout == 30.0
        assert lock.is_granted == False
        assert lock.waiting_transactions == set()

    def test_lock_custom_timeout(self):
        """Test lock with custom timeout."""
        lock = Lock(
            lock_id="lock1",
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
            acquired_at=time.time(),
            timeout=60.0,
        )

        assert lock.timeout == 60.0


class TestTransaction:
    """Test Transaction dataclass."""

    def test_transaction_creation(self):
        """Test creating transaction."""
        transaction = Transaction(
            transaction_id="tx1",
            state=TransactionState.ACTIVE,
            isolation_level=IsolationLevel.READ_COMMITTED,
            started_at=time.time(),
        )

        assert transaction.transaction_id == "tx1"
        assert transaction.state == TransactionState.ACTIVE
        assert transaction.isolation_level == IsolationLevel.READ_COMMITTED
        assert transaction.timeout == 30.0
        assert transaction.read_set == set()
        assert transaction.write_set == set()
        assert transaction.locks == {}
        assert transaction.waiting_for is None
        assert transaction.retry_count == 0
        assert transaction.max_retries == 3

    def test_transaction_defaults(self):
        """Test transaction default values."""
        transaction = Transaction(
            transaction_id="tx1",
            state=TransactionState.ACTIVE,
            isolation_level=IsolationLevel.READ_COMMITTED,
            started_at=time.time(),
        )

        assert transaction.timeout == 30.0
        assert transaction.read_set == set()
        assert transaction.write_set == set()
        assert transaction.locks == {}
        assert transaction.waiting_for is None
        assert transaction.retry_count == 0
        assert transaction.max_retries == 3

    def test_transaction_custom_values(self):
        """Test transaction with custom values."""
        transaction = Transaction(
            transaction_id="tx1",
            state=TransactionState.ACTIVE,
            isolation_level=IsolationLevel.READ_COMMITTED,
            started_at=time.time(),
            timeout=60.0,
            max_retries=5,
        )

        assert transaction.timeout == 60.0
        assert transaction.max_retries == 5


class TestDeadlockInfo:
    """Test DeadlockInfo dataclass."""

    def test_deadlock_info_creation(self):
        """Test creating deadlock info."""
        cycle = ["tx1", "tx2", "tx3"]
        transactions = [
            Transaction(
                "tx1",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
            Transaction(
                "tx2",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
            Transaction(
                "tx3",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
        ]

        deadlock_info = DeadlockInfo(
            cycle=cycle, transactions=transactions, detected_at=time.time()
        )

        assert deadlock_info.cycle == cycle
        assert deadlock_info.transactions == transactions
        assert deadlock_info.resolution_strategy == "abort_youngest"

    def test_deadlock_info_defaults(self):
        """Test deadlock info default values."""
        deadlock_info = DeadlockInfo(
            cycle=["tx1"], transactions=[], detected_at=time.time()
        )

        assert deadlock_info.resolution_strategy == "abort_youngest"


class TestLockManager:
    """Test LockManager functionality."""

    @pytest.fixture
    def lock_manager(self):
        """Fixture for lock manager."""
        return LockManager()

    def test_lock_manager_creation(self, lock_manager):
        """Test creating lock manager."""
        assert lock_manager._locks == {}
        assert hasattr(lock_manager._lock, "acquire")

    def test_acquire_lock_success(self, lock_manager):
        """Test acquiring lock successfully."""
        result = lock_manager.acquire_lock(
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
        )

        assert result == True
        assert "table1" in lock_manager._locks
        assert len(lock_manager._locks["table1"]) == 1

        lock = lock_manager._locks["table1"][0]
        assert lock.transaction_id == "tx1"
        assert lock.resource == "table1"
        assert lock.lock_type == LockType.SHARED
        assert lock.mode == LockMode.READ
        assert lock.is_granted == True

    def test_acquire_lock_conflict(self, lock_manager):
        """Test acquiring conflicting lock."""
        # Acquire first lock
        lock_manager.acquire_lock(
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.EXCLUSIVE,
            mode=LockMode.WRITE,
        )

        # Try to acquire conflicting lock
        result = lock_manager.acquire_lock(
            transaction_id="tx2",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
        )

        assert result == False
        assert len(lock_manager._locks["table1"]) == 2

        # Check that second lock is not granted
        second_lock = lock_manager._locks["table1"][1]
        assert second_lock.is_granted == False

    def test_acquire_compatible_locks(self, lock_manager):
        """Test acquiring compatible locks."""
        # Acquire first shared lock
        result1 = lock_manager.acquire_lock(
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
        )

        # Acquire second shared lock
        result2 = lock_manager.acquire_lock(
            transaction_id="tx2",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
        )

        assert result1 == True
        assert result2 == True
        assert len(lock_manager._locks["table1"]) == 2

        # Both locks should be granted
        for lock in lock_manager._locks["table1"]:
            assert lock.is_granted == True

    def test_release_lock(self, lock_manager):
        """Test releasing lock."""
        # Acquire lock
        lock_manager.acquire_lock(
            transaction_id="tx1",
            resource="table1",
            lock_type=LockType.SHARED,
            mode=LockMode.READ,
        )

        # Release lock
        lock_manager.release_lock("tx1", "table1")

        # The resource should still be in _locks but with an empty list
        assert "table1" in lock_manager._locks
        assert len(lock_manager._locks["table1"]) == 0

    def test_release_all_locks(self, lock_manager):
        """Test releasing all locks for transaction."""
        # Acquire multiple locks
        lock_manager.acquire_lock("tx1", "table1", LockType.SHARED, LockMode.READ)
        lock_manager.acquire_lock("tx1", "table2", LockType.SHARED, LockMode.READ)
        lock_manager.acquire_lock("tx2", "table1", LockType.SHARED, LockMode.READ)

        # Release all locks for tx1
        lock_manager.release_all_locks("tx1")

        # Check that tx1 locks are released but tx2 lock remains
        assert "table1" in lock_manager._locks
        assert "table2" in lock_manager._locks
        assert len(lock_manager._locks["table1"]) == 1
        assert lock_manager._locks["table1"][0].transaction_id == "tx2"
        assert len(lock_manager._locks["table2"]) == 0

    def test_get_locks_for_transaction(self, lock_manager):
        """Test getting locks for transaction."""
        # Acquire locks
        lock_manager.acquire_lock("tx1", "table1", LockType.SHARED, LockMode.READ)
        lock_manager.acquire_lock("tx1", "table2", LockType.SHARED, LockMode.READ)
        lock_manager.acquire_lock("tx2", "table1", LockType.SHARED, LockMode.READ)

        # Get locks for tx1
        locks = lock_manager.get_locks_for_transaction("tx1")

        assert len(locks) == 2
        for lock in locks:
            assert lock.transaction_id == "tx1"

    def test_detect_deadlock_no_cycle(self, lock_manager):
        """Test deadlock detection with no cycle."""
        transactions = {
            "tx1": Transaction(
                "tx1",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
            "tx2": Transaction(
                "tx2",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
        }

        deadlock_info = lock_manager.detect_deadlock(transactions)

        assert deadlock_info is None

    def test_detect_deadlock_with_cycle(self, lock_manager):
        """Test deadlock detection with cycle."""
        transactions = {
            "tx1": Transaction(
                "tx1",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
            "tx2": Transaction(
                "tx2",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
            "tx3": Transaction(
                "tx3",
                TransactionState.ACTIVE,
                IsolationLevel.READ_COMMITTED,
                time.time(),
            ),
        }

        # Create cycle: tx1 -> tx2 -> tx3 -> tx1
        transactions["tx1"].waiting_for = "tx2"
        transactions["tx2"].waiting_for = "tx3"
        transactions["tx3"].waiting_for = "tx1"

        deadlock_info = lock_manager.detect_deadlock(transactions)

        assert deadlock_info is not None
        assert "tx1" in deadlock_info.cycle
        assert "tx2" in deadlock_info.cycle
        assert "tx3" in deadlock_info.cycle
        assert len(deadlock_info.transactions) == 3


class TestTransactionIsolation:
    """Test TransactionIsolation functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Fixture for mock backend."""
        backend = Mock()
        backend.execute_query = Mock(return_value="result")
        return backend

    @pytest.fixture
    def transaction_isolation(self, mock_backend):
        """Fixture for transaction isolation."""
        return TransactionIsolation(mock_backend)

    def test_transaction_isolation_creation(self, transaction_isolation, mock_backend):
        """Test creating transaction isolation."""
        assert transaction_isolation.backend == mock_backend
        assert isinstance(transaction_isolation.lock_manager, LockManager)
        assert transaction_isolation._transactions == {}
        assert hasattr(transaction_isolation._lock, "acquire")
        assert transaction_isolation._running == False

    def test_begin_transaction(self, transaction_isolation):
        """Test beginning transaction."""
        transaction_id = transaction_isolation.begin_transaction()

        assert transaction_id is not None
        assert transaction_id in transaction_isolation._transactions

        transaction = transaction_isolation._transactions[transaction_id]
        assert transaction.state == TransactionState.ACTIVE
        assert transaction.isolation_level == IsolationLevel.READ_COMMITTED
        assert transaction.timeout == 30.0

    def test_begin_transaction_custom_isolation(self, transaction_isolation):
        """Test beginning transaction with custom isolation level."""
        transaction_id = transaction_isolation.begin_transaction(
            isolation_level=IsolationLevel.SERIALIZABLE, timeout=60.0
        )

        transaction = transaction_isolation._transactions[transaction_id]
        assert transaction.isolation_level == IsolationLevel.SERIALIZABLE
        assert transaction.timeout == 60.0

    def test_commit_transaction(self, transaction_isolation):
        """Test committing transaction."""
        transaction_id = transaction_isolation.begin_transaction()

        transaction_isolation.commit_transaction(transaction_id)

        assert transaction_id not in transaction_isolation._transactions

    def test_commit_nonexistent_transaction(self, transaction_isolation):
        """Test committing nonexistent transaction."""
        with pytest.raises(IsolationError):
            transaction_isolation.commit_transaction("nonexistent")

    def test_abort_transaction(self, transaction_isolation):
        """Test aborting transaction."""
        transaction_id = transaction_isolation.begin_transaction()

        transaction_isolation.abort_transaction(transaction_id)

        assert transaction_id not in transaction_isolation._transactions

    def test_abort_nonexistent_transaction(self, transaction_isolation):
        """Test aborting nonexistent transaction."""
        with pytest.raises(IsolationError):
            transaction_isolation.abort_transaction("nonexistent")

    def test_read_with_isolation(self, transaction_isolation, mock_backend):
        """Test reading with isolation."""
        transaction_id = transaction_isolation.begin_transaction()

        result = transaction_isolation.read_with_isolation(
            transaction_id, "table1", "SELECT * FROM table1"
        )

        assert result == "result"
        mock_backend.execute_query.assert_called_once_with("SELECT * FROM table1", None)

        # Check that resource was added to read set
        transaction = transaction_isolation._transactions[transaction_id]
        assert "table1" in transaction.read_set

    def test_read_with_isolation_nonexistent_transaction(self, transaction_isolation):
        """Test reading with nonexistent transaction."""
        with pytest.raises(IsolationError):
            transaction_isolation.read_with_isolation(
                "nonexistent", "table1", "SELECT * FROM table1"
            )

    def test_write_with_isolation(self, transaction_isolation, mock_backend):
        """Test writing with isolation."""
        transaction_id = transaction_isolation.begin_transaction()

        result = transaction_isolation.write_with_isolation(
            transaction_id, "table1", "INSERT INTO table1 VALUES (?)", {"value": "test"}
        )

        assert result == "result"
        mock_backend.execute_query.assert_called_once_with(
            "INSERT INTO table1 VALUES (?)", {"value": "test"}
        )

        # Check that resource was added to write set
        transaction = transaction_isolation._transactions[transaction_id]
        assert "table1" in transaction.write_set

    def test_write_with_isolation_nonexistent_transaction(self, transaction_isolation):
        """Test writing with nonexistent transaction."""
        with pytest.raises(IsolationError):
            transaction_isolation.write_with_isolation(
                "nonexistent", "table1", "INSERT INTO table1 VALUES (?)"
            )

    def test_get_transaction_info(self, transaction_isolation):
        """Test getting transaction info."""
        transaction_id = transaction_isolation.begin_transaction()

        info = transaction_isolation.get_transaction_info(transaction_id)

        assert info is not None
        assert info.transaction_id == transaction_id
        assert info.state == TransactionState.ACTIVE

    def test_get_transaction_info_nonexistent(self, transaction_isolation):
        """Test getting info for nonexistent transaction."""
        info = transaction_isolation.get_transaction_info("nonexistent")

        assert info is None

    def test_get_active_transactions(self, transaction_isolation):
        """Test getting active transactions."""
        # Begin multiple transactions
        tx1 = transaction_isolation.begin_transaction()
        tx2 = transaction_isolation.begin_transaction()

        active_transactions = transaction_isolation.get_active_transactions()

        assert len(active_transactions) == 2
        transaction_ids = [t.transaction_id for t in active_transactions]
        assert tx1 in transaction_ids
        assert tx2 in transaction_ids

    def test_get_lock_info(self, transaction_isolation):
        """Test getting lock information."""
        transaction_id = transaction_isolation.begin_transaction()

        # Perform a read operation to acquire a lock
        transaction_isolation.read_with_isolation(
            transaction_id, "table1", "SELECT * FROM table1"
        )

        lock_info = transaction_isolation.get_lock_info()

        assert "table1" in lock_info
        assert len(lock_info["table1"]) == 1

    def test_context_manager(self, mock_backend):
        """Test context manager functionality."""
        with TransactionIsolation(mock_backend) as isolation:
            assert isolation._running == True

        # After context exit, deadlock detection should be stopped
        assert isolation._running == False

    def test_start_stop_deadlock_detection(self, transaction_isolation):
        """Test starting and stopping deadlock detection."""
        # Start deadlock detection
        transaction_isolation.start_deadlock_detection()
        assert transaction_isolation._running == True
        assert transaction_isolation._deadlock_detection_thread is not None

        # Stop deadlock detection
        transaction_isolation.stop_deadlock_detection()
        assert transaction_isolation._running == False

    def test_resolve_deadlock(self, transaction_isolation):
        """Test resolving deadlock."""
        # Create transactions with different start times
        tx1 = transaction_isolation.begin_transaction()
        time.sleep(0.01)  # Ensure different start times
        tx2 = transaction_isolation.begin_transaction()

        # Create deadlock info
        transactions = [
            transaction_isolation._transactions[tx1],
            transaction_isolation._transactions[tx2],
        ]

        deadlock_info = DeadlockInfo(
            cycle=[tx1, tx2], transactions=transactions, detected_at=time.time()
        )

        # Resolve deadlock
        transaction_isolation._resolve_deadlock(deadlock_info)

        # The younger transaction should be aborted
        assert tx2 not in transaction_isolation._transactions
        assert tx1 in transaction_isolation._transactions
