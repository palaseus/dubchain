"""
Comprehensive unit tests for the chain reorganization system.
"""

import logging

logger = logging.getLogger(__name__)
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.core.block import Block, BlockHeader
from dubchain.core.blockchain import BlockchainState
from dubchain.core.chain_reorganization import (
    ChainReorganizationManager,
    ForkPoint,
    ReorganizationPlan,
    ReorganizationResult,
    ReorganizationStatus,
    ReorganizationType,
    TransactionReplayProtection,
)
from dubchain.core.transaction import (
    UTXO,
    Transaction,
    TransactionInput,
    TransactionOutput,
    TransactionType,
)
from dubchain.crypto.hashing import Hash, SHA256Hasher


class TestForkPoint:
    """Test the ForkPoint class."""

    def test_fork_point_creation(self):
        """Test creating a fork point."""
        block_hash = SHA256Hasher.hash("test_block")
        fork_point = ForkPoint(
            block_hash=block_hash,
            block_height=100,
            timestamp=int(time.time()),
            difficulty=5,
            total_difficulty=1000,
            fork_type=ReorganizationType.SOFT_FORK,
            confidence=0.8,
        )

        assert fork_point.block_hash == block_hash
        assert fork_point.block_height == 100
        assert fork_point.difficulty == 5
        assert fork_point.total_difficulty == 1000
        assert fork_point.fork_type == ReorganizationType.SOFT_FORK
        assert fork_point.confidence == 0.8

    def test_fork_point_invalid_confidence(self):
        """Test fork point with invalid confidence."""
        block_hash = SHA256Hasher.hash("test_block")

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ForkPoint(
                block_hash=block_hash,
                block_height=100,
                timestamp=int(time.time()),
                difficulty=5,
                total_difficulty=1000,
                fork_type=ReorganizationType.SOFT_FORK,
                confidence=1.5,  # Invalid confidence
            )


class TestReorganizationPlan:
    """Test the ReorganizationPlan class."""

    def test_reorganization_plan_creation(self):
        """Test creating a reorganization plan."""
        fork_point = ForkPoint(
            block_hash=SHA256Hasher.hash("fork"),
            block_height=100,
            timestamp=int(time.time()),
            difficulty=5,
            total_difficulty=1000,
            fork_type=ReorganizationType.REORG,
        )

        plan = ReorganizationPlan(
            fork_point=fork_point,
            old_chain=[],
            new_chain=[],
            blocks_to_remove=[],
            blocks_to_add=[],
            transactions_to_replay=[],
            utxos_to_remove=set(),
            utxos_to_add={},
            estimated_time=1.0,
            risk_level="medium",
        )

        assert plan.fork_point == fork_point
        assert plan.estimated_time == 1.0
        assert plan.risk_level == "medium"
        assert plan.rollback_plan is None

    def test_reorganization_plan_invalid_risk_level(self):
        """Test reorganization plan with invalid risk level."""
        fork_point = ForkPoint(
            block_hash=SHA256Hasher.hash("fork"),
            block_height=100,
            timestamp=int(time.time()),
            difficulty=5,
            total_difficulty=1000,
            fork_type=ReorganizationType.REORG,
        )

        with pytest.raises(ValueError, match="Risk level must be one of"):
            ReorganizationPlan(
                fork_point=fork_point,
                old_chain=[],
                new_chain=[],
                blocks_to_remove=[],
                blocks_to_add=[],
                transactions_to_replay=[],
                utxos_to_remove=set(),
                utxos_to_add={},
                risk_level="invalid",  # Invalid risk level
            )


class TestReorganizationResult:
    """Test the ReorganizationResult class."""

    def test_reorganization_result_creation(self):
        """Test creating a reorganization result."""
        result = ReorganizationResult(
            status=ReorganizationStatus.COMPLETED,
            success=True,
            blocks_removed=5,
            blocks_added=3,
            transactions_replayed=10,
            utxos_updated=15,
            execution_time=2.5,
        )

        assert result.status == ReorganizationStatus.COMPLETED
        assert result.success is True
        assert result.blocks_removed == 5
        assert result.blocks_added == 3
        assert result.transactions_replayed == 10
        assert result.utxos_updated == 15
        assert result.execution_time == 2.5
        assert result.warnings == []


class TestTransactionReplayProtection:
    """Test the TransactionReplayProtection class."""

    @pytest.fixture
    def replay_protection(self):
        """Create replay protection instance."""
        return TransactionReplayProtection(max_replay_window=100)

    @pytest.fixture
    def sample_transaction(self):
        """Create a sample transaction."""
        return Transaction(
            inputs=[],
            outputs=[TransactionOutput(amount=1000, recipient_address="recipient")],
            transaction_type=TransactionType.COINBASE,
        )

    def test_replay_protection_creation(self, replay_protection):
        """Test creating replay protection."""
        assert replay_protection.max_replay_window == 100
        assert len(replay_protection._replay_cache) == 0
        assert len(replay_protection._replay_history) == 0

    def test_add_transaction(self, replay_protection, sample_transaction):
        """Test adding a transaction to replay protection."""
        success = replay_protection.add_transaction(sample_transaction)

        assert success is True
        assert len(replay_protection._replay_history) == 1

    def test_add_duplicate_transaction(self, replay_protection, sample_transaction):
        """Test adding a duplicate transaction."""
        # Add transaction first time
        replay_protection.add_transaction(sample_transaction)

        # Try to add again
        success = replay_protection.add_transaction(sample_transaction)

        assert success is False  # Should detect replay

    def test_remove_transaction(self, replay_protection, sample_transaction):
        """Test removing a transaction from replay protection."""
        # Add transaction
        replay_protection.add_transaction(sample_transaction)

        # Remove transaction
        replay_protection.remove_transaction(sample_transaction)

        # Should be able to add again
        success = replay_protection.add_transaction(sample_transaction)
        assert success is True

    def test_is_replay(self, replay_protection, sample_transaction):
        """Test checking if transaction is a replay."""
        # Initially not a replay
        assert not replay_protection.is_replay(sample_transaction)

        # Add transaction
        replay_protection.add_transaction(sample_transaction)

        # Now it's a replay
        assert replay_protection.is_replay(sample_transaction)

    def test_cleanup_old_entries(self, replay_protection, sample_transaction):
        """Test cleaning up old entries."""
        # Add transaction
        replay_protection.add_transaction(sample_transaction)

        # Manually set old timestamp
        old_time = time.time() - 7200  # 2 hours ago
        replay_protection._replay_history[0] = (
            sample_transaction.get_hash().to_hex(),
            old_time,
        )

        # Cleanup old entries
        removed_count = replay_protection.cleanup_old_entries(
            max_age=3600
        )  # 1 hour max age

        assert removed_count > 0
        assert len(replay_protection._replay_history) == 0


class TestChainReorganizationManager:
    """Test the ChainReorganizationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a chain reorganization manager."""
        return ChainReorganizationManager(
            max_reorg_depth=50, min_confirmation_blocks=3, reorg_timeout=10.0
        )

    @pytest.fixture
    def genesis_block(self):
        """Create a genesis block."""
        return Block.create_genesis_block(
            coinbase_recipient="genesis_recipient", coinbase_amount=1000000
        )

    @pytest.fixture
    def blockchain_state(self, genesis_block):
        """Create a blockchain state."""
        state = BlockchainState()
        state.blocks = [genesis_block]
        state.block_height = 0
        return state

    @pytest.fixture
    def sample_transaction(self):
        """Create a sample transaction for testing."""
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=0
        )
        output = TransactionOutput(amount=1000, recipient_address="recipient")

        return Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )

    def test_manager_creation(self, manager):
        """Test creating a chain reorganization manager."""
        assert manager.max_reorg_depth == 50
        assert manager.min_confirmation_blocks == 3
        assert manager.reorg_timeout == 10.0
        assert manager.enable_rollback is True
        assert len(manager._active_reorgs) == 0
        assert len(manager._reorg_history) == 0

    def test_detect_fork_no_fork(self, manager, genesis_block):
        """Test detecting no fork."""
        # Create a block that extends the current chain
        coinbase_tx = Transaction.create_coinbase(
            recipient_address="miner", amount=50000000, block_height=1
        )

        new_block = Block.create_block(
            transactions=[coinbase_tx], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block]
        fork_point = manager.detect_fork(current_chain, new_block)

        assert fork_point is None  # No fork detected

    def test_detect_fork_with_fork(self, manager, genesis_block):
        """Test detecting a fork."""
        # Create two different blocks at the same height
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block, block1]
        fork_point = manager.detect_fork(current_chain, block2)

        assert fork_point is not None
        assert fork_point.block_height == 0  # Fork at genesis
        assert fork_point.fork_type == ReorganizationType.SOFT_FORK

    def test_plan_reorganization(self, manager, genesis_block, blockchain_state):
        """Test planning a reorganization."""
        # Create two different chains
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block, block1]
        new_chain = [genesis_block, block2]

        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = manager.plan_reorganization(
            fork_point, current_chain, new_chain, blockchain_state.utxos
        )

        assert plan.fork_point == fork_point
        assert len(plan.blocks_to_remove) == 1
        assert len(plan.blocks_to_add) == 1
        assert plan.risk_level in ["low", "medium", "high", "critical"]
        assert plan.estimated_time >= 0

    def test_execute_reorganization(self, manager, genesis_block, blockchain_state):
        """Test executing a reorganization."""
        # Create reorganization plan
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block, block1]
        new_chain = [genesis_block, block2]

        # Add block1 to the blockchain state (simulating current chain)
        blockchain_state.blocks.append(block1)
        blockchain_state.block_height = 1

        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = manager.plan_reorganization(
            fork_point, current_chain, new_chain, blockchain_state.utxos
        )

        # Execute reorganization
        result = manager.execute_reorganization(plan, blockchain_state)

        assert result.status == ReorganizationStatus.COMPLETED
        assert result.success is True
        assert result.blocks_removed == 1
        assert result.blocks_added == 1
        assert result.execution_time > 0

    def test_execute_reorganization_with_callback(
        self, manager, genesis_block, blockchain_state
    ):
        """Test executing reorganization with callbacks."""
        before_callback = Mock(return_value=True)
        after_callback = Mock()

        manager.set_before_reorg_callback(before_callback)
        manager.set_after_reorg_callback(after_callback)

        # Create reorganization plan
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block, block1]
        new_chain = [genesis_block, block2]

        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = manager.plan_reorganization(
            fork_point, current_chain, new_chain, blockchain_state.utxos
        )

        # Execute reorganization
        result = manager.execute_reorganization(plan, blockchain_state)

        # Check callbacks were called
        before_callback.assert_called_once_with(plan)
        after_callback.assert_called_once_with(result)

    def test_execute_reorganization_cancelled_by_callback(
        self, manager, genesis_block, blockchain_state
    ):
        """Test reorganization cancelled by before callback."""
        before_callback = Mock(return_value=False)  # Cancel reorganization
        manager.set_before_reorg_callback(before_callback)

        # Create reorganization plan
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block, block1]
        new_chain = [genesis_block, block2]

        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = manager.plan_reorganization(
            fork_point, current_chain, new_chain, blockchain_state.utxos
        )

        # Execute reorganization
        result = manager.execute_reorganization(plan, blockchain_state)

        assert result.status == ReorganizationStatus.CANCELLED
        assert result.success is False
        assert "cancelled" in result.error_message.lower()

    def test_rollback_reorganization(self, manager, genesis_block, blockchain_state):
        """Test rolling back a reorganization."""
        # Create reorganization plan with rollback
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2], previous_block=genesis_block, difficulty=1
        )

        current_chain = [genesis_block, block1]
        new_chain = [genesis_block, block2]

        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = manager.plan_reorganization(
            fork_point, current_chain, new_chain, blockchain_state.utxos
        )

        # Rollback reorganization
        result = manager.rollback_reorganization(plan, blockchain_state)

        assert result.status == ReorganizationStatus.COMPLETED
        assert result.success is True

    def test_rollback_reorganization_no_plan(
        self, manager, genesis_block, blockchain_state
    ):
        """Test rolling back reorganization without rollback plan."""
        # Create plan without rollback
        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = ReorganizationPlan(
            fork_point=fork_point,
            old_chain=[],
            new_chain=[],
            blocks_to_remove=[],
            blocks_to_add=[],
            transactions_to_replay=[],
            utxos_to_remove=set(),
            utxos_to_add={},
            rollback_plan=None,  # No rollback plan
        )

        # Try to rollback
        result = manager.rollback_reorganization(plan, blockchain_state)

        assert result.status == ReorganizationStatus.FAILED
        assert result.success is False
        assert "No rollback plan" in result.error_message

    def test_get_reorganization_stats(self, manager):
        """Test getting reorganization statistics."""
        stats = manager.get_reorganization_stats()

        assert "active_reorgs" in stats
        assert "total_reorgs" in stats
        assert "replay_protection_entries" in stats
        assert "max_reorg_depth" in stats
        assert "min_confirmation_blocks" in stats

        assert stats["active_reorgs"] == 0
        assert stats["total_reorgs"] == 0
        assert stats["max_reorg_depth"] == 50
        assert stats["min_confirmation_blocks"] == 3

    def test_cleanup_old_data(self, manager, sample_transaction):
        """Test cleaning up old data."""
        # Add some data
        manager._replay_protection.add_transaction(sample_transaction)

        # Manually set old timestamp
        old_time = time.time() - 7200  # 2 hours ago
        manager._replay_protection._replay_history[0] = (
            sample_transaction.get_hash().to_hex(),
            old_time,
        )

        # Cleanup old data
        removed_count = manager.cleanup_old_data(max_age=3600)  # 1 hour max age

        assert removed_count > 0

    def test_fork_confidence_calculation(self, manager, genesis_block):
        """Test fork confidence calculation."""
        # Create blocks with different difficulties
        coinbase_tx1 = Transaction.create_coinbase(
            recipient_address="miner1", amount=50000000, block_height=1
        )
        coinbase_tx2 = Transaction.create_coinbase(
            recipient_address="miner2", amount=50000000, block_height=1
        )

        block1 = Block.create_block(
            transactions=[coinbase_tx1], previous_block=genesis_block, difficulty=1
        )
        block2 = Block.create_block(
            transactions=[coinbase_tx2],
            previous_block=genesis_block,
            difficulty=5,  # Higher difficulty
        )

        current_chain = [genesis_block, block1]
        fork_point = manager.detect_fork(current_chain, block2)

        assert fork_point is not None
        assert 0.0 <= fork_point.confidence <= 1.0

    def test_risk_level_calculation(self, manager, genesis_block):
        """Test risk level calculation."""
        # Create a deep fork (high risk)
        current_chain = [genesis_block]
        for i in range(60):  # Create 60 blocks (exceeds max_reorg_depth)
            coinbase_tx = Transaction.create_coinbase(
                recipient_address=f"miner{i}", amount=50000000, block_height=i + 1
            )
            block = Block.create_block(
                transactions=[coinbase_tx],
                previous_block=current_chain[-1],
                difficulty=1,
            )
            current_chain.append(block)

        # Create alternative chain
        coinbase_tx = Transaction.create_coinbase(
            recipient_address="alt_miner", amount=50000000, block_height=1
        )
        alt_block = Block.create_block(
            transactions=[coinbase_tx], previous_block=genesis_block, difficulty=1
        )

        fork_point = manager.detect_fork(current_chain, alt_block)

        # Should not detect fork due to depth limit
        assert fork_point is None

    def test_reorganization_with_utxo_updates(
        self, manager, genesis_block, blockchain_state
    ):
        """Test reorganization with UTXO updates."""
        # Create UTXOs
        utxo1 = UTXO(
            tx_hash=SHA256Hasher.hash("tx1"),
            output_index=0,
            amount=1000,
            recipient_address="address1",
        )
        utxo2 = UTXO(
            tx_hash=SHA256Hasher.hash("tx2"),
            output_index=0,
            amount=2000,
            recipient_address="address2",
        )

        blockchain_state.utxos[utxo1.get_key()] = utxo1
        blockchain_state.utxos[utxo2.get_key()] = utxo2

        # Create reorganization plan
        fork_point = ForkPoint(
            block_hash=genesis_block.get_hash(),
            block_height=0,
            timestamp=genesis_block.header.timestamp,
            difficulty=genesis_block.header.difficulty,
            total_difficulty=1,
            fork_type=ReorganizationType.SOFT_FORK,
        )

        plan = ReorganizationPlan(
            fork_point=fork_point,
            old_chain=[genesis_block],
            new_chain=[genesis_block],
            blocks_to_remove=[],
            blocks_to_add=[],
            transactions_to_replay=[],
            utxos_to_remove={utxo1.get_key()},
            utxos_to_add={utxo2.get_key(): utxo2},
            estimated_time=0.1,
            risk_level="low",
        )

        # Execute reorganization
        result = manager.execute_reorganization(plan, blockchain_state)

        assert result.status == ReorganizationStatus.COMPLETED
        assert result.success is True
        assert result.utxos_updated == 2  # 1 removed + 1 added
