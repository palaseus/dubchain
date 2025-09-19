"""Tests for blockchain core module."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.dubchain.core.blockchain import Blockchain, BlockchainState
from src.dubchain.core.block import Block, BlockHeader
from src.dubchain.core.transaction import Transaction, UTXO, TransactionType
from src.dubchain.core.consensus import ConsensusConfig
from src.dubchain.crypto.hashing import Hash, SHA256Hasher
from src.dubchain.crypto.signatures import PrivateKey, PublicKey
from src.dubchain.errors.exceptions import ValidationError


class TestBlockchainState:
    """Test BlockchainState functionality."""

    def test_init(self):
        """Test BlockchainState initialization."""
        state = BlockchainState()
        assert state.blocks == []
        assert state.utxos == {}
        assert state.pending_transactions == []
        assert state.block_height == 0
        assert state.total_difficulty == 0
        assert state.last_block_time == 0

    def test_get_balance(self):
        """Test getting balance for an address."""
        state = BlockchainState()
        
        # Create UTXOs for an address
        utxo1 = UTXO(
            tx_hash=Hash.from_hex("a" * 64),  # 32-byte hex string
            output_index=0,
            amount=100,
            recipient_address="address1"
        )
        utxo2 = UTXO(
            tx_hash=Hash.from_hex("b" * 64),
            output_index=0,
            amount=200,
            recipient_address="address1"
        )
        utxo3 = UTXO(
            tx_hash=Hash.from_hex("c" * 64),
            output_index=0,
            amount=50,
            recipient_address="address2"
        )
        
        state.add_utxo(utxo1)
        state.add_utxo(utxo2)
        state.add_utxo(utxo3)
        
        assert state.get_balance("address1") == 300
        assert state.get_balance("address2") == 50
        assert state.get_balance("nonexistent") == 0

    def test_get_utxos_for_address(self):
        """Test getting UTXOs for an address."""
        state = BlockchainState()
        
        utxo1 = UTXO(
            tx_hash=Hash.from_hex("a" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1"
        )
        utxo2 = UTXO(
            tx_hash=Hash.from_hex("b" * 64),
            output_index=0,
            amount=200,
            recipient_address="address1"
        )
        utxo3 = UTXO(
            tx_hash=Hash.from_hex("c" * 64),
            output_index=0,
            amount=50,
            recipient_address="address2"
        )
        
        state.add_utxo(utxo1)
        state.add_utxo(utxo2)
        state.add_utxo(utxo3)
        
        utxos_address1 = state.get_utxos_for_address("address1")
        assert len(utxos_address1) == 2
        assert utxo1 in utxos_address1
        assert utxo2 in utxos_address1
        
        utxos_address2 = state.get_utxos_for_address("address2")
        assert len(utxos_address2) == 1
        assert utxo3 in utxos_address2

    def test_add_remove_utxo(self):
        """Test adding and removing UTXOs."""
        state = BlockchainState()
        
        utxo = UTXO(
            tx_hash=Hash.from_hex("a" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1"
        )
        
        # Add UTXO
        state.add_utxo(utxo)
        assert len(state.utxos) == 1
        assert utxo.get_key() in state.utxos
        
        # Remove UTXO
        state.remove_utxo(utxo.get_key())
        assert len(state.utxos) == 0

    def test_add_remove_pending_transaction(self):
        """Test adding and removing pending transactions."""
        state = BlockchainState()
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        tx.get_hash.return_value = Hash.from_hex("c" * 64)
        
        # Add pending transaction
        state.add_pending_transaction(tx)
        assert len(state.pending_transactions) == 1
        assert tx in state.pending_transactions
        
        # Add same transaction again (should not duplicate)
        state.add_pending_transaction(tx)
        assert len(state.pending_transactions) == 1
        
        # Remove pending transaction
        state.remove_pending_transaction(tx)
        assert len(state.pending_transactions) == 0

    def test_update_state(self):
        """Test updating state with a new block."""
        state = BlockchainState()
        
        # Create mock block
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 5
        block.header.timestamp = 1234567890
        block.header.difficulty = 10
        
        state.update_state(block)
        
        assert state.block_height == 5
        assert state.last_block_time == 1234567890
        assert state.total_difficulty == 10

    def test_update_utxo_block_height(self):
        """Test updating UTXO block height."""
        state = BlockchainState()
        
        utxo = UTXO(
            tx_hash=Hash.from_hex("a" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1",
            block_height=0
        )
        
        state.add_utxo(utxo)
        state.update_utxo_block_height(utxo.get_key(), 5)
        
        updated_utxo = state.utxos[utxo.get_key()]
        assert updated_utxo.block_height == 5


class TestBlockchain:
    """Test Blockchain functionality."""

    def test_init(self):
        """Test Blockchain initialization."""
        blockchain = Blockchain()
        assert blockchain.config is not None
        assert blockchain.consensus_engine is not None
        assert blockchain.state is not None
        assert blockchain._genesis_created is False
        assert blockchain.genesis_block is None

    def test_init_with_config(self):
        """Test Blockchain initialization with custom config."""
        config = ConsensusConfig()
        blockchain = Blockchain(config)
        assert blockchain.config == config

    def test_create_genesis_block(self):
        """Test creating genesis block."""
        blockchain = Blockchain()
        
        genesis_block = blockchain.create_genesis_block("recipient_address", 1000000)
        
        assert blockchain._genesis_created is True
        assert genesis_block is not None
        assert len(blockchain.state.blocks) == 1
        assert blockchain.state.blocks[0] == genesis_block

    def test_create_genesis_block_already_created(self):
        """Test creating genesis block when already created."""
        blockchain = Blockchain()
        blockchain.create_genesis_block("recipient_address")
        
        with pytest.raises(ValueError, match="Genesis block already created"):
            blockchain.create_genesis_block("another_address")

    def test_initialize_genesis(self):
        """Test initializing with existing genesis block."""
        blockchain = Blockchain()
        
        # Create mock genesis block
        genesis_block = Mock(spec=Block)
        genesis_block.header = Mock(spec=BlockHeader)
        genesis_block.header.block_height = 0
        genesis_block.header.timestamp = 1234567890
        genesis_block.header.difficulty = 0
        
        blockchain.initialize_genesis(genesis_block)
        
        assert blockchain._genesis_created is True
        assert blockchain.genesis_block == genesis_block
        assert blockchain.state.block_height == 0

    def test_add_block_genesis(self):
        """Test adding genesis block."""
        blockchain = Blockchain()
        
        # Create mock genesis block
        genesis_block = Mock(spec=Block)
        genesis_block.header = Mock(spec=BlockHeader)
        genesis_block.header.block_height = 0
        genesis_block.header.timestamp = 1234567890
        genesis_block.header.difficulty = 0
        genesis_block.is_valid.return_value = True
        genesis_block.get_regular_transactions.return_value = []
        genesis_block.transactions = []
        
        result = blockchain.add_block(genesis_block)
        
        assert result is True
        assert len(blockchain.state.blocks) == 1
        assert blockchain.state.blocks[0] == genesis_block

    def test_add_block_invalid(self):
        """Test adding invalid block."""
        blockchain = Blockchain()
        
        # Create mock invalid block
        invalid_block = Mock(spec=Block)
        invalid_block.header = Mock(spec=BlockHeader)
        invalid_block.header.block_height = 1
        invalid_block.header.timestamp = 1234567890
        invalid_block.header.difficulty = 10
        
        # Mock validation to return False
        with patch.object(blockchain, '_validate_block', return_value=False):
            result = blockchain.add_block(invalid_block)
            assert result is False
            assert len(blockchain.state.blocks) == 0

    def test_add_block_exception(self):
        """Test adding block with exception."""
        blockchain = Blockchain()
        
        # Create mock block that raises exception
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 1
        
        with patch.object(blockchain, '_validate_block', side_effect=Exception("Validation error")):
            result = blockchain.add_block(block)
            assert result is False

    def test_validate_block_genesis(self):
        """Test validating genesis block."""
        blockchain = Blockchain()
        
        # Create mock genesis block
        genesis_block = Mock(spec=Block)
        genesis_block.header = Mock(spec=BlockHeader)
        genesis_block.header.block_height = 0
        genesis_block.is_valid.return_value = True
        
        result = blockchain._validate_block(genesis_block)
        assert result is True

    def test_validate_block_genesis_with_existing_blocks(self):
        """Test validating genesis block when blocks already exist."""
        blockchain = Blockchain()
        
        # Add a block first
        blockchain.state.blocks.append(Mock(spec=Block))
        
        # Create mock genesis block
        genesis_block = Mock(spec=Block)
        genesis_block.header = Mock(spec=BlockHeader)
        genesis_block.header.block_height = 0
        
        result = blockchain._validate_block(genesis_block)
        assert result is False

    def test_validate_block_no_previous_blocks(self):
        """Test validating non-genesis block with no previous blocks."""
        blockchain = Blockchain()
        
        # Create mock non-genesis block
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 1
        
        result = blockchain._validate_block(block)
        assert result is False

    def test_validate_block_with_consensus(self):
        """Test validating block with consensus engine."""
        blockchain = Blockchain()
        
        # Add a previous block
        previous_block = Mock(spec=Block)
        blockchain.state.blocks.append(previous_block)
        
        # Create mock block
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 1
        
        # Mock consensus validation
        with patch.object(blockchain.consensus_engine, 'validate_block', return_value=True):
            result = blockchain._validate_block(block)
            assert result is True

    def test_update_utxo_set(self):
        """Test updating UTXO set with block transactions."""
        blockchain = Blockchain()
        
        # Create mock UTXOs
        utxo1 = UTXO(
            tx_hash=Hash.from_hex("a" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1"
        )
        utxo2 = UTXO(
            tx_hash=Hash.from_hex("b" * 64),
            output_index=0,
            amount=200,
            recipient_address="address2"
        )
        
        blockchain.state.add_utxo(utxo1)
        blockchain.state.add_utxo(utxo2)
        
        # Create mock transaction that consumes utxo1
        tx = Mock(spec=Transaction)
        tx.get_utxos_consumed.return_value = [utxo1.get_key()]
        
        # Create mock block
        block = Mock(spec=Block)
        block.get_regular_transactions.return_value = [tx]
        block.transactions = []
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 1
        
        blockchain._update_utxo_set(block)
        
        # utxo1 should be removed, utxo2 should remain
        assert utxo1.get_key() not in blockchain.state.utxos
        assert utxo2.get_key() in blockchain.state.utxos

    def test_remove_pending_transactions(self):
        """Test removing pending transactions that are in a block."""
        blockchain = Blockchain()
        
        # Create mock transactions
        tx1 = Mock(spec=Transaction)
        tx1.get_hash.return_value = Hash.from_hex("a" * 64)
        tx2 = Mock(spec=Transaction)
        tx2.get_hash.return_value = Hash.from_hex("b" * 64)
        tx3 = Mock(spec=Transaction)
        tx3.get_hash.return_value = Hash.from_hex("c" * 64)
        
        blockchain.state.pending_transactions = [tx1, tx2, tx3]
        
        # Create mock block containing tx1 and tx2
        block = Mock(spec=Block)
        block.transactions = [tx1, tx2]
        
        blockchain._remove_pending_transactions(block)
        
        # Only tx3 should remain in pending transactions
        assert len(blockchain.state.pending_transactions) == 1
        assert tx3 in blockchain.state.pending_transactions

    def test_add_transaction_valid(self):
        """Test adding valid transaction."""
        blockchain = Blockchain()
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        tx.is_valid.return_value = True
        tx.get_hash.return_value = Hash.from_hex("c" * 64)
        
        result = blockchain.add_transaction(tx)
        
        assert result is True
        assert len(blockchain.state.pending_transactions) == 1
        assert tx in blockchain.state.pending_transactions

    def test_add_transaction_invalid(self):
        """Test adding invalid transaction."""
        blockchain = Blockchain()
        
        # Create mock invalid transaction
        tx = Mock(spec=Transaction)
        tx.is_valid.return_value = False
        
        result = blockchain.add_transaction(tx)
        
        assert result is False
        assert len(blockchain.state.pending_transactions) == 0

    def test_add_transaction_duplicate(self):
        """Test adding duplicate transaction."""
        blockchain = Blockchain()
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        tx.is_valid.return_value = True
        tx.get_hash.return_value = Hash.from_hex("c" * 64)
        
        # Add transaction first time
        blockchain.add_transaction(tx)
        
        # Try to add same transaction again
        result = blockchain.add_transaction(tx)
        
        assert result is False
        assert len(blockchain.state.pending_transactions) == 1

    def test_add_transaction_exception(self):
        """Test adding transaction with exception."""
        blockchain = Blockchain()
        
        # Create mock transaction that raises exception
        tx = Mock(spec=Transaction)
        tx.is_valid.side_effect = Exception("Validation error")
        
        result = blockchain.add_transaction(tx)
        
        assert result is False

    def test_mine_block_no_blocks(self):
        """Test mining block with no existing blocks."""
        blockchain = Blockchain()
        
        result = blockchain.mine_block("miner_address")
        
        assert result is None

    def test_mine_block_success(self):
        """Test successful block mining."""
        blockchain = Blockchain()
        
        # Create genesis block first
        genesis_block = Mock(spec=Block)
        genesis_block.header = Mock(spec=BlockHeader)
        genesis_block.header.block_height = 0
        genesis_block.header.timestamp = 1234567890
        genesis_block.header.difficulty = 0
        genesis_block.is_valid.return_value = True
        genesis_block.get_regular_transactions.return_value = []
        genesis_block.transactions = []
        
        blockchain.add_block(genesis_block)
        
        # Mock consensus engine
        mined_block = Mock(spec=Block)
        with patch.object(blockchain.consensus_engine, 'mine_block', return_value=mined_block):
            # Mock add_block to return True
            with patch.object(blockchain, 'add_block', return_value=True):
                result = blockchain.mine_block("miner_address")
                assert result == mined_block

    def test_mine_block_mining_fails(self):
        """Test block mining when consensus engine fails."""
        blockchain = Blockchain()
        
        # Create genesis block first
        genesis_block = Mock(spec=Block)
        genesis_block.header = Mock(spec=BlockHeader)
        genesis_block.header.block_height = 0
        genesis_block.header.timestamp = 1234567890
        genesis_block.header.difficulty = 0
        genesis_block.is_valid.return_value = True
        genesis_block.get_regular_transactions.return_value = []
        genesis_block.transactions = []
        
        blockchain.add_block(genesis_block)
        
        # Mock consensus engine to return None
        with patch.object(blockchain.consensus_engine, 'mine_block', return_value=None):
            result = blockchain.mine_block("miner_address")
            assert result is None

    def test_calculate_block_reward(self):
        """Test block reward calculation."""
        blockchain = Blockchain()
        
        # Test initial reward
        blockchain.state.block_height = 0
        reward = blockchain._calculate_block_reward()
        assert reward == 50 * (10**8)  # 50 coins in satoshis
        
        # Test after first halving
        blockchain.state.block_height = 210000
        reward = blockchain._calculate_block_reward()
        assert reward == 25 * (10**8)  # 25 coins in satoshis
        
        # Test after multiple halvings
        blockchain.state.block_height = 420000
        reward = blockchain._calculate_block_reward()
        assert reward == 12.5 * (10**8)  # 12.5 coins in satoshis
        
        # Test after 64 halvings (reward becomes 0)
        blockchain.state.block_height = 64 * 210000
        reward = blockchain._calculate_block_reward()
        assert reward == 0

    def test_get_balance(self):
        """Test getting balance for an address."""
        blockchain = Blockchain()
        
        # Add UTXOs to state
        utxo1 = UTXO(
            tx_hash=Hash.from_hex("a" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1"
        )
        utxo2 = UTXO(
            tx_hash=Hash.from_hex("b" * 64),
            output_index=0,
            amount=200,
            recipient_address="address1"
        )
        
        blockchain.state.add_utxo(utxo1)
        blockchain.state.add_utxo(utxo2)
        
        balance = blockchain.get_balance("address1")
        assert balance == 300

    def test_get_utxos_for_address(self):
        """Test getting UTXOs for an address."""
        blockchain = Blockchain()
        
        utxo = UTXO(
            tx_hash=Hash.from_hex("a" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1"
        )
        
        blockchain.state.add_utxo(utxo)
        
        utxos = blockchain.get_utxos_for_address("address1")
        assert len(utxos) == 1
        assert utxo in utxos

    def test_get_block_by_hash(self):
        """Test getting block by hash."""
        blockchain = Blockchain()
        
        # Create mock block
        block = Mock(spec=Block)
        block_hash = Hash.from_hex("d" * 64)
        block.get_hash.return_value = block_hash
        
        blockchain.state.blocks.append(block)
        
        result = blockchain.get_block_by_hash(block_hash)
        assert result == block
        
        # Test with non-existent hash
        non_existent_hash = Hash.from_hex("1" * 64)
        result = blockchain.get_block_by_hash(non_existent_hash)
        assert result is None

    def test_get_block_by_height(self):
        """Test getting block by height."""
        blockchain = Blockchain()
        
        # Create mock blocks
        block1 = Mock(spec=Block)
        block2 = Mock(spec=Block)
        
        blockchain.state.blocks = [block1, block2]
        
        result = blockchain.get_block_by_height(0)
        assert result == block1
        
        result = blockchain.get_block_by_height(1)
        assert result == block2
        
        # Test with invalid height
        result = blockchain.get_block_by_height(2)
        assert result is None
        
        result = blockchain.get_block_by_height(-1)
        assert result is None

    def test_get_transaction_by_hash(self):
        """Test getting transaction by hash."""
        blockchain = Blockchain()
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        tx_hash = Hash.from_hex("c" * 64)
        tx.get_hash.return_value = tx_hash
        
        # Create mock block
        block = Mock(spec=Block)
        block.transactions = [tx]
        
        blockchain.state.blocks = [block]
        
        result_block, result_tx = blockchain.get_transaction_by_hash(tx_hash)
        assert result_block == block
        assert result_tx == tx
        
        # Test with pending transaction
        pending_tx = Mock(spec=Transaction)
        pending_tx_hash = Hash.from_hex("2" * 64)
        pending_tx.get_hash.return_value = pending_tx_hash
        
        blockchain.state.pending_transactions = [pending_tx]
        
        result_block, result_tx = blockchain.get_transaction_by_hash(pending_tx_hash)
        assert result_block is None
        assert result_tx == pending_tx

    def test_get_chain_info_empty(self):
        """Test getting chain info with empty chain."""
        blockchain = Blockchain()
        
        info = blockchain.get_chain_info()
        
        assert info["block_count"] == 0
        assert info["block_height"] == 0
        assert info["total_difficulty"] == 0
        assert info["pending_transactions"] == 0
        assert info["utxo_count"] == 0

    def test_get_chain_info_with_blocks(self):
        """Test getting chain info with blocks."""
        blockchain = Blockchain()
        
        # Create mock block
        block = Mock(spec=Block)
        block.get_hash.return_value = Hash.from_hex("d" * 64)
        block.header = Mock(spec=BlockHeader)
        block.header.timestamp = 1234567890
        
        blockchain.state.blocks = [block]
        blockchain.state.block_height = 0
        blockchain.state.total_difficulty = 100
        
        # Mock consensus engine
        with patch.object(blockchain.consensus_engine, 'get_consensus_info', return_value={"consensus": "info"}):
            info = blockchain.get_chain_info()
            
            assert info["block_count"] == 1
            assert info["block_height"] == 0
            assert info["total_difficulty"] == 100
            assert info["last_block_hash"] == "d" * 64
            assert info["last_block_time"] == 1234567890
            assert info["consensus"] == "info"

    def test_validate_chain_valid(self):
        """Test validating valid chain."""
        blockchain = Blockchain()
        
        # Create mock blocks
        block1 = Mock(spec=Block)
        block1.header = Mock(spec=BlockHeader)
        block1.header.block_height = 0
        block1.header.previous_hash = Hash.zero()
        block1._verify_merkle_root.return_value = True
        block1.header.meets_difficulty.return_value = True
        block1.get_hash.return_value = Hash.from_hex("e" * 64)

        block2 = Mock(spec=Block)
        block2.header = Mock(spec=BlockHeader)
        block2.header.block_height = 1
        block2.header.previous_hash = Hash.from_hex("e" * 64)
        block2._verify_merkle_root.return_value = True
        block2.header.meets_difficulty.return_value = True
        block2.get_hash.return_value = Hash.from_hex("3" * 64)
        
        blockchain.state.blocks = [block1, block2]
        
        result = blockchain.validate_chain()
        assert result is True

    def test_validate_chain_invalid_merkle(self):
        """Test validating chain with invalid merkle root."""
        blockchain = Blockchain()
        
        # Create mock block with invalid merkle root
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 0
        block._verify_merkle_root.return_value = False
        
        blockchain.state.blocks = [block]
        
        result = blockchain.validate_chain()
        assert result is False

    def test_validate_chain_invalid_difficulty(self):
        """Test validating chain with invalid difficulty."""
        blockchain = Blockchain()
        
        # Create mock block with invalid difficulty
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 0
        block._verify_merkle_root.return_value = True
        block.header.meets_difficulty.return_value = False
        
        blockchain.state.blocks = [block]
        
        result = blockchain.validate_chain()
        assert result is False

    def test_validate_chain_invalid_height(self):
        """Test validating chain with invalid height."""
        blockchain = Blockchain()
        
        # Create mock block with invalid height
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.block_height = 5  # Should be 0
        block._verify_merkle_root.return_value = True
        block.header.meets_difficulty.return_value = True
        
        blockchain.state.blocks = [block]
        
        result = blockchain.validate_chain()
        assert result is False

    def test_validate_chain_invalid_previous_hash(self):
        """Test validating chain with invalid previous hash."""
        blockchain = Blockchain()
        
        # Create mock blocks with invalid previous hash
        block1 = Mock(spec=Block)
        block1.header = Mock(spec=BlockHeader)
        block1.header.block_height = 0
        block1.header.previous_hash = Hash.zero()
        block1._verify_merkle_root.return_value = True
        block1.header.meets_difficulty.return_value = True
        block1.get_hash.return_value = Hash.from_hex("e" * 64)
        
        block2 = Mock(spec=Block)
        block2.header = Mock(spec=BlockHeader)
        block2.header.block_height = 1
        block2.header.previous_hash = Hash.from_hex("4" * 64)  # Wrong previous hash
        block2._verify_merkle_root.return_value = True
        block2.header.meets_difficulty.return_value = True
        
        blockchain.state.blocks = [block1, block2]
        
        result = blockchain.validate_chain()
        assert result is False

    def test_validate_chain_exception(self):
        """Test validating chain with exception."""
        blockchain = Blockchain()
        
        # Create mock block that raises exception
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block._verify_merkle_root.side_effect = Exception("Error")
        
        blockchain.state.blocks = [block]
        
        result = blockchain.validate_chain()
        assert result is False

    def test_get_best_chain(self):
        """Test getting best chain."""
        blockchain = Blockchain()
        
        # Create mock blocks
        block1 = Mock(spec=Block)
        block2 = Mock(spec=Block)
        
        blockchain.state.blocks = [block1, block2]
        
        best_chain = blockchain.get_best_chain()
        assert best_chain == [block1, block2]
        assert best_chain is not blockchain.state.blocks  # Should be a copy

    def test_get_pending_transactions(self):
        """Test getting pending transactions."""
        blockchain = Blockchain()
        
        # Create mock transactions
        tx1 = Mock(spec=Transaction)
        tx2 = Mock(spec=Transaction)
        
        blockchain.state.pending_transactions = [tx1, tx2]
        
        pending_txs = blockchain.get_pending_transactions()
        assert pending_txs == [tx1, tx2]
        assert pending_txs is not blockchain.state.pending_transactions  # Should be a copy

    def test_clear_pending_transactions(self):
        """Test clearing pending transactions."""
        blockchain = Blockchain()
        
        # Add some pending transactions
        tx1 = Mock(spec=Transaction)
        tx2 = Mock(spec=Transaction)
        blockchain.state.pending_transactions = [tx1, tx2]
        
        blockchain.clear_pending_transactions()
        assert len(blockchain.state.pending_transactions) == 0

    def test_export_state(self):
        """Test exporting blockchain state."""
        blockchain = Blockchain()
        
        # Create mock block
        block = Mock(spec=Block)
        block.header = Mock(spec=BlockHeader)
        block.header.version = 1
        block.header.previous_hash = Hash.from_hex("f" * 64)
        block.header.merkle_root = Hash.from_hex("5" * 64)
        block.header.timestamp = 1234567890
        block.header.difficulty = 10
        block.header.nonce = 12345
        block.header.block_height = 0
        block.header.gas_limit = 1000000
        block.header.gas_used = 500000
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        tx.get_hash.return_value = Hash.from_hex("c" * 64)
        tx.transaction_type = TransactionType.REGULAR
        tx.inputs = [Mock()]
        tx.outputs = [Mock()]
        
        block.transactions = [tx]
        blockchain.state.blocks = [block]
        
        # Create mock UTXO
        utxo = UTXO(
            tx_hash=Hash.from_hex("6" * 64),
            output_index=0,
            amount=100,
            recipient_address="address1",
            block_height=0
        )
        blockchain.state.add_utxo(utxo)
        
        # Create mock pending transaction
        pending_tx = Mock(spec=Transaction)
        pending_tx.get_hash.return_value = Hash.from_hex("2" * 64)
        pending_tx.transaction_type = TransactionType.REGULAR
        pending_tx.inputs = [Mock()]
        pending_tx.outputs = [Mock()]
        blockchain.state.pending_transactions = [pending_tx]
        
        blockchain.state.block_height = 0
        blockchain.state.total_difficulty = 100
        
        exported = blockchain.export_state()
        
        assert "blocks" in exported
        assert "utxos" in exported
        assert "pending_transactions" in exported
        assert exported["block_height"] == 0
        assert exported["total_difficulty"] == 100

    def test_process_transaction_valid(self):
        """Test processing valid transaction."""
        blockchain = Blockchain()
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        tx.transaction_id = "tx_id"
        
        with patch.object(blockchain, 'validate_transaction', return_value=True):
            result = blockchain.process_transaction(tx)
            assert result is True
            assert tx in blockchain.state.pending_transactions

    def test_process_transaction_invalid(self):
        """Test processing invalid transaction."""
        blockchain = Blockchain()
        
        # Create mock transaction
        tx = Mock(spec=Transaction)
        
        with patch.object(blockchain, 'validate_transaction', return_value=False):
            with pytest.raises(ValidationError):
                blockchain.process_transaction(tx)

    def test_validate_transaction(self):
        """Test transaction validation."""
        blockchain = Blockchain()
        
        # Test with valid transaction
        tx = Mock(spec=Transaction)
        tx.transaction_id = "tx_id"
        
        result = blockchain.validate_transaction(tx)
        assert result is True
        
        # Test with None transaction
        result = blockchain.validate_transaction(None)
        assert result is False
        
        # Test with transaction without transaction_id
        tx_no_id = Mock()
        del tx_no_id.transaction_id  # Remove the attribute
        result = blockchain.validate_transaction(tx_no_id)
        assert result is False

    def test_str_repr(self):
        """Test string representation."""
        blockchain = Blockchain()
        blockchain.state.block_height = 5
        blockchain.state.blocks = [Mock()] * 3
        blockchain.state.utxos = {"utxo1": Mock(), "utxo2": Mock()}
        
        str_repr = str(blockchain)
        assert "height=5" in str_repr
        assert "blocks=3" in str_repr
        
        repr_str = repr(blockchain)
        assert "height=5" in repr_str
        assert "utxos=2" in repr_str
