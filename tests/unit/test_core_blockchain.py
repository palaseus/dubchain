"""Tests for core blockchain module."""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.core.block import Block
from dubchain.core.blockchain import Blockchain, BlockchainState
from dubchain.core.consensus import ConsensusEngine
from dubchain.core.transaction import UTXO, Transaction, TransactionType
from dubchain.crypto.hashing import Hash
from dubchain.errors.exceptions import ValidationError


class TestBlockchainState:
    """Test BlockchainState functionality."""

    @pytest.fixture
    def blockchain_state(self):
        """Fixture for blockchain state."""
        return BlockchainState()

    def test_blockchain_state_creation(self):
        """Test creating blockchain state."""
        state = BlockchainState()

        assert isinstance(state.blocks, list)
        assert isinstance(state.utxos, dict)
        assert isinstance(state.pending_transactions, list)
        assert state.block_height == 0
        assert state.total_difficulty == 0
        assert state.last_block_time == 0

    def test_get_balance(self, blockchain_state):
        """Test getting balance for address."""
        # Add some UTXOs
        utxo1 = Mock(spec=UTXO)
        utxo1.recipient_address = "address1"
        utxo1.amount = 100
        utxo1.get_key.return_value = "utxo1"

        utxo2 = Mock(spec=UTXO)
        utxo2.recipient_address = "address1"
        utxo2.amount = 50
        utxo2.get_key.return_value = "utxo2"

        utxo3 = Mock(spec=UTXO)
        utxo3.recipient_address = "address2"
        utxo3.amount = 200
        utxo3.get_key.return_value = "utxo3"

        blockchain_state.utxos["utxo1"] = utxo1
        blockchain_state.utxos["utxo2"] = utxo2
        blockchain_state.utxos["utxo3"] = utxo3

        balance1 = blockchain_state.get_balance("address1")
        balance2 = blockchain_state.get_balance("address2")
        balance3 = blockchain_state.get_balance("address3")

        assert balance1 == 150
        assert balance2 == 200
        assert balance3 == 0

    def test_get_utxos_for_address(self, blockchain_state):
        """Test getting UTXOs for address."""
        # Add some UTXOs
        utxo1 = Mock(spec=UTXO)
        utxo1.recipient_address = "address1"
        utxo1.get_key.return_value = "utxo1"

        utxo2 = Mock(spec=UTXO)
        utxo2.recipient_address = "address1"
        utxo2.get_key.return_value = "utxo2"

        utxo3 = Mock(spec=UTXO)
        utxo3.recipient_address = "address2"
        utxo3.get_key.return_value = "utxo3"

        blockchain_state.utxos["utxo1"] = utxo1
        blockchain_state.utxos["utxo2"] = utxo2
        blockchain_state.utxos["utxo3"] = utxo3

        utxos1 = blockchain_state.get_utxos_for_address("address1")
        utxos2 = blockchain_state.get_utxos_for_address("address2")
        utxos3 = blockchain_state.get_utxos_for_address("address3")

        assert len(utxos1) == 2
        assert utxo1 in utxos1
        assert utxo2 in utxos1
        assert len(utxos2) == 1
        assert utxo3 in utxos2
        assert len(utxos3) == 0

    def test_add_utxo(self, blockchain_state):
        """Test adding UTXO."""
        utxo = Mock(spec=UTXO)
        utxo.get_key.return_value = "utxo_key"

        blockchain_state.add_utxo(utxo)

        assert "utxo_key" in blockchain_state.utxos
        assert blockchain_state.utxos["utxo_key"] == utxo

    def test_remove_utxo(self, blockchain_state):
        """Test removing UTXO."""
        utxo = Mock(spec=UTXO)
        utxo.get_key.return_value = "utxo_key"

        blockchain_state.utxos["utxo_key"] = utxo
        assert "utxo_key" in blockchain_state.utxos

        blockchain_state.remove_utxo("utxo_key")
        assert "utxo_key" not in blockchain_state.utxos

    def test_add_pending_transaction(self, blockchain_state):
        """Test adding pending transaction."""
        transaction = Mock(spec=Transaction)

        blockchain_state.add_pending_transaction(transaction)

        assert transaction in blockchain_state.pending_transactions

    def test_remove_pending_transaction(self, blockchain_state):
        """Test removing pending transaction."""
        transaction = Mock(spec=Transaction)

        blockchain_state.pending_transactions.append(transaction)
        assert transaction in blockchain_state.pending_transactions

        blockchain_state.remove_pending_transaction(transaction)
        assert transaction not in blockchain_state.pending_transactions

    def test_update_state(self, blockchain_state):
        """Test updating state."""
        block = Mock(spec=Block)
        block.header = Mock()
        block.header.block_height = 5
        block.header.timestamp = 1234567890
        block.header.difficulty = 100

        blockchain_state.update_state(block)

        assert blockchain_state.block_height == 5
        assert blockchain_state.last_block_time == 1234567890
        assert blockchain_state.total_difficulty == 100


class TestBlockchain:
    """Test Blockchain functionality."""

    @pytest.fixture
    def blockchain(self):
        """Fixture for blockchain."""
        return Blockchain()

    def test_blockchain_creation(self):
        """Test creating blockchain."""
        chain = Blockchain()

        assert isinstance(chain.state, BlockchainState)
        assert isinstance(chain.consensus_engine, ConsensusEngine)
        assert chain.genesis_block is None

    def test_initialize_genesis(self, blockchain):
        """Test initializing genesis block."""
        mock_genesis = Mock(spec=Block)
        mock_genesis.header = Mock()
        mock_genesis.header.block_height = 0
        mock_genesis.header.timestamp = 1234567890
        mock_genesis.header.difficulty = 1

        blockchain.initialize_genesis(mock_genesis)

        assert blockchain.genesis_block == mock_genesis
        assert blockchain.state.block_height == 0
        assert blockchain.state.last_block_time == 1234567890

    def test_add_block(self, blockchain):
        """Test adding block."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)

        # Create new block
        new_block = Mock(spec=Block)
        new_block.header = Mock()
        new_block.header.block_height = 1
        new_block.header.timestamp = 1234567891
        new_block.header.difficulty = 100
        new_block.header.previous_hash = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        new_block.get_hash.return_value = Hash.from_hex(
            "1111111111111111111111111111111111111111111111111111111111111111"
        )
        # Create mock transactions
        mock_tx = Mock(spec=Transaction)
        mock_tx.get_hash.return_value = Hash.from_hex(
            "5555555555555555555555555555555555555555555555555555555555555555"
        )
        mock_tx.get_utxos_consumed.return_value = []
        mock_tx.get_utxos_created.return_value = []
        new_block.transactions = [mock_tx]
        new_block.get_regular_transactions.return_value = []

        # Mock validation
        with patch.object(blockchain, "_validate_block", return_value=True):
            result = blockchain.add_block(new_block)

        assert result is True
        assert new_block in blockchain.state.blocks
        assert blockchain.state.block_height == 1

    def test_add_invalid_block(self, blockchain):
        """Test adding invalid block."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)

        # Create invalid block
        invalid_block = Mock(spec=Block)
        invalid_block.header = Mock()
        invalid_block.header.block_height = 1
        invalid_block.header.timestamp = 1234567891
        invalid_block.header.difficulty = 100
        invalid_block.header.previous_hash = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        invalid_block.get_hash.return_value = Hash.from_hex(
            "2222222222222222222222222222222222222222222222222222222222222222"
        )
        # Create mock transactions
        mock_tx = Mock(spec=Transaction)
        mock_tx.get_hash.return_value = Hash.from_hex(
            "6666666666666666666666666666666666666666666666666666666666666666"
        )
        mock_tx.get_utxos_consumed.return_value = []
        mock_tx.get_utxos_created.return_value = []
        invalid_block.transactions = [mock_tx]
        invalid_block.get_regular_transactions.return_value = []

        # Mock validation failure
        with patch.object(blockchain, "_validate_block", return_value=False):
            result = blockchain.add_block(invalid_block)
            assert result is False

    def test_validate_block(self, blockchain):
        """Test validating block."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)
        blockchain.state.blocks.append(genesis)

        # Create block to validate
        block = Mock(spec=Block)
        block.header = Mock()
        block.header.block_height = 1
        block.header.timestamp = 1234567891
        block.header.difficulty = 100
        block.header.previous_hash = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        block.get_hash.return_value = Hash.from_hex(
            "3333333333333333333333333333333333333333333333333333333333333333"
        )
        # Create mock transactions
        mock_tx = Mock(spec=Transaction)
        mock_tx.get_hash.return_value = Hash.from_hex(
            "7777777777777777777777777777777777777777777777777777777777777777"
        )
        mock_tx.get_utxos_consumed.return_value = []
        mock_tx.get_utxos_created.return_value = []
        block.transactions = [mock_tx]
        block.get_regular_transactions.return_value = []

        # Mock consensus validation
        with patch.object(
            blockchain.consensus_engine, "validate_block", return_value=True
        ):
            result = blockchain._validate_block(block)

        assert result is True

    def test_validate_block_invalid(self, blockchain):
        """Test validating invalid block."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)

        # Create invalid block
        block = Mock(spec=Block)
        block.header = Mock()
        block.header.block_height = 1
        block.header.timestamp = 1234567891
        block.header.difficulty = 100
        block.header.previous_hash = Hash.from_hex(
            "4444444444444444444444444444444444444444444444444444444444444444"
        )  # Wrong previous hash
        block.get_hash.return_value = Hash.from_hex(
            "3333333333333333333333333333333333333333333333333333333333333333"
        )
        # Create mock transactions
        mock_tx = Mock(spec=Transaction)
        mock_tx.get_hash.return_value = Hash.from_hex(
            "7777777777777777777777777777777777777777777777777777777777777777"
        )
        mock_tx.get_utxos_consumed.return_value = []
        mock_tx.get_utxos_created.return_value = []
        block.transactions = [mock_tx]
        block.get_regular_transactions.return_value = []

        result = blockchain._validate_block(block)
        assert result is False

    def test_process_transaction(self, blockchain):
        """Test processing transaction."""
        transaction = Mock(spec=Transaction)
        transaction.transaction_id = "tx1"
        transaction.transaction_type = TransactionType.REGULAR

        # Mock validation
        with patch.object(blockchain, "validate_transaction", return_value=True):
            result = blockchain.process_transaction(transaction)

        assert result is True
        assert transaction in blockchain.state.pending_transactions

    def test_process_invalid_transaction(self, blockchain):
        """Test processing invalid transaction."""
        transaction = Mock(spec=Transaction)
        transaction.transaction_id = "tx1"
        transaction.transaction_type = TransactionType.REGULAR

        # Mock validation failure
        with patch.object(blockchain, "validate_transaction", return_value=False):
            with pytest.raises(ValidationError):
                blockchain.process_transaction(transaction)

    def test_validate_transaction(self, blockchain):
        """Test validating transaction."""
        transaction = Mock(spec=Transaction)
        transaction.transaction_id = "tx1"
        transaction.transaction_type = TransactionType.REGULAR

        # Mock transaction validation
        with patch.object(transaction, "is_valid", return_value=True):
            result = blockchain.validate_transaction(transaction)

        assert result is True

    def test_get_block_by_hash(self, blockchain):
        """Test getting block by hash."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)
        blockchain.state.blocks.append(genesis)

        # Add another block
        block = Mock(spec=Block)
        block.header = Mock()
        block.header.block_height = 1
        block.get_hash.return_value = Hash.from_hex(
            "1111111111111111111111111111111111111111111111111111111111111111"
        )
        blockchain.state.blocks.append(block)

        # Test getting blocks
        genesis_result = blockchain.get_block_by_hash(
            Hash.from_hex(
                "0000000000000000000000000000000000000000000000000000000000000000"
            )
        )
        block_result = blockchain.get_block_by_hash(
            Hash.from_hex(
                "1111111111111111111111111111111111111111111111111111111111111111"
            )
        )
        none_result = blockchain.get_block_by_hash(
            Hash.from_hex(
                "9999999999999999999999999999999999999999999999999999999999999999"
            )
        )

        assert genesis_result == genesis
        assert block_result == block
        assert none_result is None

    def test_get_block_by_height(self, blockchain):
        """Test getting block by height."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)
        blockchain.state.blocks.append(genesis)

        # Add another block
        block = Mock(spec=Block)
        block.header = Mock()
        block.header.block_height = 1
        blockchain.state.blocks.append(block)

        # Test getting blocks
        genesis_result = blockchain.get_block_by_height(0)
        block_result = blockchain.get_block_by_height(1)
        none_result = blockchain.get_block_by_height(2)

        assert genesis_result == genesis
        assert block_result == block
        assert none_result is None

    def test_get_balance(self, blockchain):
        """Test getting balance."""
        # Add some UTXOs to state
        utxo1 = Mock(spec=UTXO)
        utxo1.recipient_address = "address1"
        utxo1.amount = 100
        utxo1.get_key.return_value = "utxo1"

        utxo2 = Mock(spec=UTXO)
        utxo2.recipient_address = "address1"
        utxo2.amount = 50
        utxo2.get_key.return_value = "utxo2"

        blockchain.state.utxos["utxo1"] = utxo1
        blockchain.state.utxos["utxo2"] = utxo2

        balance = blockchain.get_balance("address1")
        assert balance == 150

    def test_get_utxos_for_address(self, blockchain):
        """Test getting UTXOs for address."""
        # Add some UTXOs to state
        utxo1 = Mock(spec=UTXO)
        utxo1.recipient_address = "address1"
        utxo1.get_key.return_value = "utxo1"

        utxo2 = Mock(spec=UTXO)
        utxo2.recipient_address = "address1"
        utxo2.get_key.return_value = "utxo2"

        blockchain.state.utxos["utxo1"] = utxo1
        blockchain.state.utxos["utxo2"] = utxo2

        utxos = blockchain.get_utxos_for_address("address1")
        assert len(utxos) == 2
        assert utxo1 in utxos
        assert utxo2 in utxos

    def test_validate_chain(self, blockchain):
        """Test validating chain."""
        # Setup initial chain
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.header.previous_hash = Hash.zero()
        genesis.header.meets_difficulty.return_value = True
        genesis._verify_merkle_root.return_value = True
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)
        blockchain.state.blocks.append(genesis)

        # Add some blocks
        block1 = Mock(spec=Block)
        block1.header = Mock()
        block1.header.block_height = 1
        block1.header.timestamp = 1234567891
        block1.header.difficulty = 100
        block1.header.previous_hash = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        block1.header.meets_difficulty.return_value = True
        block1._verify_merkle_root.return_value = True
        block1.get_hash.return_value = Hash.from_hex(
            "1111111111111111111111111111111111111111111111111111111111111111"
        )
        blockchain.state.blocks.append(block1)

        # Test chain validation
        result = blockchain.validate_chain()
        assert result is True

    def test_validate_chain_invalid(self, blockchain):
        """Test validating invalid chain."""
        # Setup initial chain
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)

        # Add invalid block
        invalid_block = Mock(spec=Block)
        invalid_block.header = Mock()
        invalid_block.header.block_height = 1
        invalid_block.header.timestamp = 1234567891
        invalid_block.header.difficulty = 100
        blockchain.state.blocks.append(invalid_block)

        # Mock validation failure
        with patch.object(blockchain, "_validate_block", return_value=False):
            result = blockchain.validate_chain()
            assert result is False

    def test_get_chain_info(self, blockchain):
        """Test getting chain information."""
        # Setup genesis block
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        blockchain.initialize_genesis(genesis)
        blockchain.state.blocks.append(genesis)

        # Add another block
        block = Mock(spec=Block)
        block.header = Mock()
        block.header.block_height = 1
        block.header.timestamp = 1234567891
        block.header.difficulty = 100
        block.get_hash.return_value = Hash.from_hex(
            "3333333333333333333333333333333333333333333333333333333333333333"
        )
        blockchain.state.blocks.append(block)
        blockchain.state.total_difficulty += 2**block.header.difficulty

        info = blockchain.get_chain_info()

        assert info["block_height"] == 1
        assert info["total_difficulty"] == 1267650600228229401496703205377
        assert (
            info["last_block_hash"]
            == "3333333333333333333333333333333333333333333333333333333333333333"
        )

    def test_get_pending_transactions(self, blockchain):
        """Test getting pending transactions."""
        # Add some pending transactions
        tx1 = Mock(spec=Transaction)
        tx2 = Mock(spec=Transaction)

        blockchain.state.pending_transactions = [tx1, tx2]

        pending = blockchain.get_pending_transactions()
        assert len(pending) == 2
        assert tx1 in pending
        assert tx2 in pending

    def test_clear_pending_transactions(self, blockchain):
        """Test clearing pending transactions."""
        # Add some pending transactions
        tx1 = Mock(spec=Transaction)
        tx2 = Mock(spec=Transaction)

        blockchain.state.pending_transactions = [tx1, tx2]
        assert len(blockchain.state.pending_transactions) == 2

        blockchain.clear_pending_transactions()
        assert len(blockchain.state.pending_transactions) == 0

    def test_get_state_snapshot(self, blockchain):
        """Test getting state snapshot."""
        # Setup some state
        genesis = Mock(spec=Block)
        genesis.header = Mock()
        genesis.header.block_height = 0
        genesis.header.timestamp = 1234567890
        genesis.header.difficulty = 1
        genesis.get_hash.return_value = Hash.from_hex(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        genesis.transactions = []
        blockchain.initialize_genesis(genesis)
        blockchain.state.blocks.append(genesis)

        utxo = Mock(spec=UTXO)
        utxo.get_key.return_value = "utxo_key"
        utxo.tx_hash = Hash.from_hex(
            "8888888888888888888888888888888888888888888888888888888888888888"
        )
        utxo.output_index = 0
        utxo.amount = 1000
        utxo.recipient_address = "address1"
        utxo.script_pubkey = "script1"
        utxo.contract_address = None
        utxo.data = None
        utxo.block_height = 0
        blockchain.state.utxos["utxo_key"] = utxo

        snapshot = blockchain.export_state()

        assert "blocks" in snapshot
        assert "utxos" in snapshot
        assert "pending_transactions" in snapshot
        assert len(snapshot["blocks"]) == 1
        assert len(snapshot["utxos"]) == 1
