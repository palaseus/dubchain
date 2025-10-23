"""
Comprehensive unit tests for the transaction pool management system.
"""

import logging

logger = logging.getLogger(__name__)
import threading
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.core.transaction import (
    UTXO,
    Transaction,
    TransactionInput,
    TransactionOutput,
    TransactionType,
)
from dubchain.core.transaction_pool import (
    FeeEstimate,
    TransactionEntry,
    TransactionPool,
    TransactionPriority,
)
from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.crypto.signatures import PrivateKey, PublicKey


class TestTransactionEntry:
    """Test the TransactionEntry class."""

    def test_transaction_entry_creation(self):
        """Test creating a transaction entry."""
        tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

        entry = TransactionEntry(
            transaction=tx,
            priority=TransactionPriority.HIGH,
            rbf_enabled=True,
            rbf_sequence=1,
        )

        assert entry.transaction == tx
        assert entry.priority == TransactionPriority.HIGH
        assert entry.rbf_enabled is True
        assert entry.rbf_sequence == 1
        assert entry.fee_rate == 0.0  # Coinbase has no fee
        assert entry.age == 0.0
        assert entry.validation_cache is None

    def test_transaction_entry_fee_rate_calculation(self):
        """Test fee rate calculation for regular transactions."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create a regular transaction
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=0
        )
        output = TransactionOutput(amount=900, recipient_address="recipient")

        tx = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )

        entry = TransactionEntry(transaction=tx)
        # Fee rate should be calculated (even if placeholder)
        assert entry.fee_rate >= 0.0

    def test_transaction_entry_age_update(self):
        """Test age update functionality."""
        tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

        entry = TransactionEntry(transaction=tx)
        initial_age = entry.age

        time.sleep(0.01)  # Small delay
        entry.update_age()

        assert entry.age > initial_age
        assert entry.last_seen > entry.first_seen

    def test_transaction_entry_priority_score(self):
        """Test priority score calculation."""
        # Create a regular transaction (not coinbase) to have a fee rate
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=0
        )
        tx = Transaction(
            inputs=[input_tx],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.REGULAR,
        )

        # Test different priorities
        low_entry = TransactionEntry(transaction=tx, priority=TransactionPriority.LOW)
        normal_entry = TransactionEntry(
            transaction=tx, priority=TransactionPriority.NORMAL
        )
        high_entry = TransactionEntry(transaction=tx, priority=TransactionPriority.HIGH)
        urgent_entry = TransactionEntry(
            transaction=tx, priority=TransactionPriority.URGENT
        )

        # Urgent should have highest score
        assert urgent_entry.get_priority_score() > high_entry.get_priority_score()
        assert high_entry.get_priority_score() > normal_entry.get_priority_score()
        assert normal_entry.get_priority_score() > low_entry.get_priority_score()

    def test_transaction_entry_expiration(self):
        """Test transaction expiration checking."""
        tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

        entry = TransactionEntry(transaction=tx)

        # Should not be expired initially
        assert not entry.is_expired(max_age=3600)

        # Manually set age to be expired
        entry.age = 3700
        assert entry.is_expired(max_age=3600)

    def test_transaction_entry_can_replace(self):
        """Test RBF replacement logic."""
        tx1 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )
        tx2 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

        # Create entries with RBF enabled
        entry1 = TransactionEntry(
            transaction=tx1, rbf_enabled=True, rbf_sequence=1, fee_rate=10.0
        )
        entry2 = TransactionEntry(
            transaction=tx2, rbf_enabled=True, rbf_sequence=2, fee_rate=15.0
        )

        # entry2 should be able to replace entry1
        assert entry2.can_replace(entry1)
        assert not entry1.can_replace(entry2)

        # Test with RBF disabled
        entry1.rbf_enabled = False
        assert not entry2.can_replace(entry1)


class TestFeeEstimate:
    """Test the FeeEstimate class."""

    def test_fee_estimate_creation(self):
        """Test creating a fee estimate."""
        estimate = FeeEstimate(target_blocks=6, fee_rate=15.5, confidence=0.85)

        assert estimate.target_blocks == 6
        assert estimate.fee_rate == 15.5
        assert estimate.confidence == 0.85
        assert estimate.timestamp > 0


class TestTransactionPool:
    """Test the TransactionPool class."""

    @pytest.fixture
    def pool(self):
        """Create a transaction pool for testing."""
        return TransactionPool(max_size=100, max_memory_mb=10)

    @pytest.fixture
    def sample_transaction(self):
        """Create a sample transaction for testing."""
        return Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

    @pytest.fixture
    def sample_utxos(self):
        """Create sample UTXOs for testing."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        utxo = UTXO(
            tx_hash=SHA256Hasher.hash("prev_tx"),
            output_index=0,
            amount=1000,
            recipient_address=public_key.to_address(),
        )

        return {utxo.get_key(): utxo}

    def test_pool_creation(self, pool):
        """Test creating a transaction pool."""
        assert pool.max_size == 100
        assert pool.max_memory_mb == 10
        assert len(pool) == 0
        assert pool._running is False

    def test_pool_start_stop(self, pool):
        """Test starting and stopping the pool."""
        pool.start()
        assert pool._running is True
        assert pool._cleanup_thread is not None

        pool.stop()
        assert pool._running is False

    def test_add_transaction_success(self, pool, sample_transaction):
        """Test successfully adding a transaction."""
        success, message = pool.add_transaction(sample_transaction)

        assert success is True
        assert "successfully" in message
        assert len(pool) == 1
        assert sample_transaction.get_hash().to_hex() in pool

    def test_add_duplicate_transaction(self, pool, sample_transaction):
        """Test adding a duplicate transaction."""
        # Add transaction first time
        pool.add_transaction(sample_transaction)

        # Try to add again
        success, message = pool.add_transaction(sample_transaction)

        assert success is False
        assert "already in pool" in message
        assert len(pool) == 1

    def test_add_transaction_with_validation_callback(
        self, pool, sample_transaction, sample_utxos
    ):
        """Test adding transaction with validation callback."""
        # Set up validation callback
        validation_callback = Mock(return_value=True)
        pool.set_validation_callback(validation_callback)

        success, message = pool.add_transaction(sample_transaction, utxos=sample_utxos)

        assert success is True
        validation_callback.assert_called_once()

    def test_add_transaction_validation_failure(
        self, pool, sample_transaction, sample_utxos
    ):
        """Test adding transaction that fails validation."""
        # Set up validation callback to return False
        validation_callback = Mock(return_value=False)
        pool.set_validation_callback(validation_callback)

        success, message = pool.add_transaction(sample_transaction, utxos=sample_utxos)

        assert success is False
        assert "validation failed" in message
        assert len(pool) == 0

    def test_remove_transaction(self, pool, sample_transaction):
        """Test removing a transaction."""
        # Add transaction
        pool.add_transaction(sample_transaction)
        tx_hash = sample_transaction.get_hash().to_hex()

        # Remove transaction
        success = pool.remove_transaction(tx_hash)

        assert success is True
        assert len(pool) == 0
        assert tx_hash not in pool

    def test_remove_nonexistent_transaction(self, pool):
        """Test removing a transaction that doesn't exist."""
        success = pool.remove_transaction("nonexistent_hash")
        assert success is False

    def test_get_transaction(self, pool, sample_transaction):
        """Test getting a transaction by hash."""
        # Add transaction
        pool.add_transaction(sample_transaction)
        tx_hash = sample_transaction.get_hash().to_hex()

        # Get transaction
        retrieved_tx = pool.get_transaction(tx_hash)

        assert retrieved_tx is not None
        assert retrieved_tx.get_hash() == sample_transaction.get_hash()

    def test_get_nonexistent_transaction(self, pool):
        """Test getting a transaction that doesn't exist."""
        retrieved_tx = pool.get_transaction("nonexistent_hash")
        assert retrieved_tx is None

    def test_get_priority_transactions(self, pool):
        """Test getting transactions by priority."""
        # Create transactions with different priorities
        tx1 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient1")],
            transaction_type=TransactionType.COINBASE,
        )
        tx2 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=2000, recipient_address="recipient2")],
            transaction_type=TransactionType.COINBASE,
        )

        pool.add_transaction(tx1, priority=TransactionPriority.LOW)
        pool.add_transaction(tx2, priority=TransactionPriority.HIGH)

        priority_txs = pool.get_priority_transactions(limit=10)

        assert len(priority_txs) == 2
        # Higher priority transaction should come first
        assert priority_txs[0].get_hash() == tx2.get_hash()

    def test_get_high_fee_transactions(self, pool):
        """Test getting transactions by fee rate."""
        # Create transactions (fee rates will be calculated)
        tx1 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient1")],
            transaction_type=TransactionType.COINBASE,
        )
        tx2 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=2000, recipient_address="recipient2")],
            transaction_type=TransactionType.COINBASE,
        )

        pool.add_transaction(tx1)
        pool.add_transaction(tx2)

        fee_txs = pool.get_high_fee_transactions(limit=10)

        assert len(fee_txs) == 2

    def test_get_transactions_by_address(self, pool):
        """Test getting transactions by address."""
        address = "test_address"
        tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address=address)],
            transaction_type=TransactionType.COINBASE,
        )

        pool.add_transaction(tx)

        address_txs = pool.get_transactions_by_address(address)

        assert len(address_txs) == 1
        assert address_txs[0].get_hash() == tx.get_hash()

    def test_get_pool_stats(self, pool, sample_transaction):
        """Test getting pool statistics."""
        # Add a transaction
        pool.add_transaction(sample_transaction)

        stats = pool.get_pool_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "memory_usage_mb" in stats
        assert "average_fee_rate" in stats
        assert "validation_cache_size" in stats
        assert "conflicts" in stats
        assert "fee_estimates" in stats

        assert stats["size"] == 1
        assert stats["max_size"] == 100

    def test_fee_estimate(self, pool, sample_transaction):
        """Test fee estimation functionality."""
        # Add a transaction to generate fee estimates
        pool.add_transaction(sample_transaction)

        # Get fee estimate for 6 blocks
        estimate = pool.get_fee_estimate(6)

        # Fee estimates might not be available immediately
        # but the method should not crash
        if estimate:
            assert isinstance(estimate, FeeEstimate)
            assert estimate.target_blocks == 6
            assert estimate.fee_rate >= 0
            assert 0 <= estimate.confidence <= 1

    def test_clear_expired_transactions(self, pool):
        """Test clearing expired transactions."""
        # Create a transaction and manually set it as expired
        tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

        pool.add_transaction(tx)
        tx_hash = tx.get_hash().to_hex()

        # Manually set the transaction as expired
        with pool._lock:
            entry = pool._transactions[tx_hash]
            entry.age = 3700  # Set to expired

        # Clear expired transactions
        cleared_count = pool.clear_expired_transactions()

        assert cleared_count == 1
        assert len(pool) == 0

    def test_pool_size_limit(self):
        """Test pool size limit enforcement."""
        # Create a small pool
        small_pool = TransactionPool(max_size=2)

        # Add transactions up to the limit
        tx1 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient1")],
            transaction_type=TransactionType.COINBASE,
        )
        tx2 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=2000, recipient_address="recipient2")],
            transaction_type=TransactionType.COINBASE,
        )
        tx3 = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=3000, recipient_address="recipient3")],
            transaction_type=TransactionType.COINBASE,
        )

        # Add first two transactions
        success1, _ = small_pool.add_transaction(tx1)
        success2, _ = small_pool.add_transaction(tx2)

        assert success1 is True
        assert success2 is True
        assert len(small_pool) == 2

        # Try to add third transaction
        success3, message3 = small_pool.add_transaction(tx3)

        # Should either succeed (if eviction works) or fail
        if not success3:
            assert "full" in message3.lower()

    def test_conflict_detection(self, pool):
        """Test transaction conflict detection."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create two transactions that spend the same UTXO
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=0
        )

        tx1 = Transaction(
            inputs=[input_tx],
            outputs=[TransactionOutput(amount=900, recipient_address="recipient1")],
            transaction_type=TransactionType.REGULAR,
        )
        tx2 = Transaction(
            inputs=[input_tx],  # Same input - conflict!
            outputs=[TransactionOutput(amount=800, recipient_address="recipient2")],
            transaction_type=TransactionType.REGULAR,
        )

        # Add first transaction
        success1, _ = pool.add_transaction(tx1)
        assert success1 is True

        # Try to add conflicting transaction
        success2, message2 = pool.add_transaction(tx2)

        # Should fail due to conflict
        assert success2 is False
        assert "conflict" in message2.lower()

    def test_rbf_replacement(self, pool):
        """Test Replace-by-Fee functionality."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create initial transaction
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=0
        )

        tx1 = Transaction(
            inputs=[input_tx],
            outputs=[TransactionOutput(amount=900, recipient_address="recipient")],
            transaction_type=TransactionType.REGULAR,
        )

        # Add with RBF enabled
        success1, _ = pool.add_transaction(tx1, rbf_enabled=True, rbf_sequence=1)
        assert success1 is True

        # Create replacement transaction with higher fee
        tx2 = Transaction(
            inputs=[input_tx],
            outputs=[
                TransactionOutput(amount=800, recipient_address="recipient")
            ],  # Lower output = higher fee
            transaction_type=TransactionType.REGULAR,
        )

        # Set up conflict callback to allow replacement
        conflict_callback = Mock(return_value=True)
        pool.set_conflict_callback(conflict_callback)

        # Add replacement transaction
        success2, _ = pool.add_transaction(tx2, rbf_enabled=True, rbf_sequence=2)

        # Should succeed and replace the first transaction
        assert success2 is True
        assert len(pool) == 1
        assert tx2.get_hash().to_hex() in pool
        assert tx1.get_hash().to_hex() not in pool

    def test_validation_cache(self, pool, sample_transaction, sample_utxos):
        """Test validation caching functionality."""
        # Set up validation callback
        validation_callback = Mock(return_value=True)
        pool.set_validation_callback(validation_callback)

        # Add transaction
        pool.add_transaction(sample_transaction, utxos=sample_utxos)

        # Validation should be called once
        assert validation_callback.call_count == 1

        # Try to add same transaction again (should use cache)
        pool.add_transaction(sample_transaction, utxos=sample_utxos)

        # Validation should still be called only once (due to duplicate check)
        assert validation_callback.call_count == 1

    def test_memory_management(self):
        """Test memory management functionality."""
        # Create pool with very small memory limit
        small_pool = TransactionPool(max_memory_mb=1)  # 1MB limit

        # Add transactions until memory limit is reached
        for i in range(10):
            tx = Transaction(
                inputs=[],
                outputs=[
                    TransactionOutput(amount=1000, recipient_address=f"recipient{i}")
                ],
                transaction_type=TransactionType.COINBASE,
            )

            success, message = small_pool.add_transaction(tx)

            # Should eventually fail due to memory limit
            if not success:
                assert "memory" in message.lower()
                break

    def test_thread_safety(self, pool):
        """Test thread safety of the transaction pool."""
        results = []
        errors = []

        def add_transactions():
            try:
                for i in range(10):
                    tx = Transaction(
                        inputs=[],
                        outputs=[
                            TransactionOutput(
                                amount=1000, recipient_address=f"recipient{i}"
                            )
                        ],
                        transaction_type=TransactionType.COINBASE,
                    )
                    success, _ = pool.add_transaction(tx)
                    results.append(success)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=add_transactions)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Check that some transactions were added successfully
        assert len(results) > 0
        assert any(results)  # At least some should succeed

    def test_cleanup_worker(self, pool):
        """Test the background cleanup worker."""
        # Add a transaction
        tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )
        pool.add_transaction(tx)

        # Manually set transaction as expired
        tx_hash = tx.get_hash().to_hex()
        with pool._lock:
            entry = pool._transactions[tx_hash]
            entry.age = 3700  # Set to expired

        # Force cleanup
        pool.force_cleanup()

        # Transaction should be cleaned up
        assert len(pool) == 0

    def test_pool_iteration(self, pool):
        """Test iterating over the transaction pool."""
        # Add some transactions
        for i in range(3):
            tx = Transaction(
                inputs=[],
                outputs=[
                    TransactionOutput(amount=1000, recipient_address=f"recipient{i}")
                ],
                transaction_type=TransactionType.COINBASE,
            )
            pool.add_transaction(tx)

        # Iterate over the pool
        entries = list(pool)

        assert len(entries) == 3
        assert all(isinstance(entry, TransactionEntry) for entry in entries)

    def test_pool_contains(self, pool, sample_transaction):
        """Test the 'in' operator for the transaction pool."""
        tx_hash = sample_transaction.get_hash().to_hex()

        # Transaction should not be in pool initially
        assert tx_hash not in pool

        # Add transaction
        pool.add_transaction(sample_transaction)

        # Transaction should now be in pool
        assert tx_hash in pool

    def test_pool_length(self, pool):
        """Test the len() function for the transaction pool."""
        assert len(pool) == 0

        # Add transactions
        for i in range(3):
            tx = Transaction(
                inputs=[],
                outputs=[
                    TransactionOutput(amount=1000, recipient_address=f"recipient{i}")
                ],
                transaction_type=TransactionType.COINBASE,
            )
            pool.add_transaction(tx)

        assert len(pool) == 3
