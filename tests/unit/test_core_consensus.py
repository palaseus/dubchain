"""
Unit tests for core consensus module.
"""

import math
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.core.block import Block, BlockHeader
from dubchain.core.consensus import ConsensusConfig, ConsensusEngine, ProofOfWork
from dubchain.core.transaction import Transaction, TransactionType


class TestConsensusConfig:
    """Test ConsensusConfig class."""

    def test_consensus_config_default_values(self):
        """Test default configuration values."""
        config = ConsensusConfig()

        assert config.target_block_time == 10
        assert config.difficulty_adjustment_interval == 2016
        assert config.max_difficulty_change == 4.0
        assert config.min_difficulty == 1
        assert config.max_difficulty == 256
        assert config.max_block_size == 1000000
        assert config.max_transactions_per_block == 10000
        assert config.max_gas_per_block == 10000000
        assert config.max_future_time == 3600
        assert config.min_block_interval == 1

    def test_consensus_config_custom_values(self):
        """Test custom configuration values."""
        config = ConsensusConfig(
            target_block_time=15,
            difficulty_adjustment_interval=1000,
            max_difficulty_change=2.0,
            min_difficulty=2,
            max_difficulty=512,
            max_block_size=2000000,
            max_transactions_per_block=20000,
            max_gas_per_block=20000000,
            max_future_time=1800,
            min_block_interval=2,
        )

        assert config.target_block_time == 15
        assert config.difficulty_adjustment_interval == 1000
        assert config.max_difficulty_change == 2.0
        assert config.min_difficulty == 2
        assert config.max_difficulty == 512
        assert config.max_block_size == 2000000
        assert config.max_transactions_per_block == 20000
        assert config.max_gas_per_block == 20000000
        assert config.max_future_time == 1800
        assert config.min_block_interval == 2


class TestProofOfWork:
    """Test ProofOfWork class."""

    @pytest.fixture
    def config(self):
        """Create a consensus config for testing."""
        return ConsensusConfig()

    @pytest.fixture
    def pow_engine(self, config):
        """Create a ProofOfWork instance for testing."""
        return ProofOfWork(config)

    @pytest.fixture
    def mock_block_header(self):
        """Create a mock block header for testing."""
        header = Mock(spec=BlockHeader)
        header.with_nonce.return_value = header
        header.meets_difficulty.return_value = True
        return header

    def test_proof_of_work_initialization(self, config):
        """Test ProofOfWork initialization."""
        pow_engine = ProofOfWork(config)
        assert pow_engine.config == config

    def test_mine_block_success(self, pow_engine, mock_block_header):
        """Test successful block mining."""
        with patch('time.time', return_value=1000.0):
            result = pow_engine.mine_block(mock_block_header, max_nonce=10)

        assert result is not None
        mock_block_header.with_nonce.assert_called()
        mock_block_header.meets_difficulty.assert_called()

    def test_mine_block_failure_no_valid_nonce(self, pow_engine, mock_block_header):
        """Test block mining failure when no valid nonce is found."""
        mock_block_header.meets_difficulty.return_value = False

        result = pow_engine.mine_block(mock_block_header, max_nonce=5)

        assert result is None

    def test_mine_block_timeout(self, pow_engine, mock_block_header):
        """Test block mining timeout."""
        mock_block_header.meets_difficulty.return_value = False

        with patch('time.time', side_effect=[1000.0, 1000.0, 1400.0]):  # 400 seconds elapsed
            result = pow_engine.mine_block(mock_block_header, max_nonce=1000000)

        assert result is None

    def test_verify_block_success(self, pow_engine):
        """Test successful block verification."""
        mock_block = Mock()
        mock_block.header = Mock()
        mock_block.header.meets_difficulty.return_value = True

        result = pow_engine.verify_block(mock_block)

        assert result is True
        mock_block.header.meets_difficulty.assert_called_once()

    def test_verify_block_failure(self, pow_engine):
        """Test block verification failure."""
        mock_block = Mock()
        mock_block.header = Mock()
        mock_block.header.meets_difficulty.return_value = False

        result = pow_engine.verify_block(mock_block)

        assert result is False

    def test_calculate_difficulty_insufficient_blocks(self, pow_engine):
        """Test difficulty calculation with insufficient blocks."""
        blocks = [Mock()]

        result = pow_engine.calculate_difficulty(blocks)

        assert result == pow_engine.config.min_difficulty

    def test_calculate_difficulty_single_block(self, pow_engine):
        """Test difficulty calculation with single block."""
        mock_block = Mock()
        mock_block.header = Mock()
        mock_block.header.difficulty = 10
        blocks = [mock_block]

        result = pow_engine.calculate_difficulty(blocks)

        # With only 1 block, should return min_difficulty
        assert result == pow_engine.config.min_difficulty

    def test_calculate_difficulty_normal_case(self, pow_engine):
        """Test normal difficulty calculation."""
        # Create mock blocks with timestamps
        blocks = []
        base_time = 1000000
        for i in range(2016):  # Full adjustment interval
            mock_block = Mock()
            mock_block.header = Mock()
            mock_block.header.timestamp = base_time + (i * 10)  # 10 second intervals
            mock_block.header.difficulty = 10
            blocks.append(mock_block)

        result = pow_engine.calculate_difficulty(blocks)

        # Should return a valid difficulty within bounds
        assert pow_engine.config.min_difficulty <= result <= pow_engine.config.max_difficulty

    def test_calculate_difficulty_zero_time_span(self, pow_engine):
        """Test difficulty calculation with zero time span."""
        mock_block = Mock()
        mock_block.header = Mock()
        mock_block.header.timestamp = 1000000
        mock_block.header.difficulty = 10
        blocks = [mock_block, mock_block]  # Same timestamp

        result = pow_engine.calculate_difficulty(blocks)

        assert result == 10

    def test_calculate_difficulty_with_custom_target_time(self, pow_engine):
        """Test difficulty calculation with custom target time."""
        blocks = []
        base_time = 1000000
        for i in range(10):
            mock_block = Mock()
            mock_block.header = Mock()
            mock_block.header.timestamp = base_time + (i * 5)  # 5 second intervals
            mock_block.header.difficulty = 10
            blocks.append(mock_block)

        result = pow_engine.calculate_difficulty(blocks, target_block_time=20)

        assert pow_engine.config.min_difficulty <= result <= pow_engine.config.max_difficulty

    def test_estimate_mining_time(self, pow_engine):
        """Test mining time estimation."""
        difficulty = 10
        hashrate = 1000.0

        result = pow_engine.estimate_mining_time(difficulty, hashrate)

        expected_time = (2**difficulty) / hashrate
        assert result == expected_time

    def test_estimate_mining_time_zero_hashrate(self, pow_engine):
        """Test mining time estimation with zero hashrate."""
        result = pow_engine.estimate_mining_time(10, 0)

        assert result == float("inf")

    def test_estimate_mining_time_negative_hashrate(self, pow_engine):
        """Test mining time estimation with negative hashrate."""
        result = pow_engine.estimate_mining_time(10, -100)

        assert result == float("inf")

    def test_get_difficulty_target(self, pow_engine):
        """Test getting difficulty target."""
        difficulty = 10

        result = pow_engine.get_difficulty_target(difficulty)

        assert result is not None


class TestConsensusEngine:
    """Test ConsensusEngine class."""

    @pytest.fixture
    def config(self):
        """Create a consensus config for testing."""
        return ConsensusConfig()

    @pytest.fixture
    def consensus_engine(self, config):
        """Create a ConsensusEngine instance for testing."""
        return ConsensusEngine(config)

    @pytest.fixture
    def mock_block(self):
        """Create a mock block for testing."""
        block = Mock()
        block.transactions = [Mock()]
        block.transactions[0].transaction_type.value = "coinbase"
        block._verify_merkle_root.return_value = True
        block.to_bytes.return_value = b"test_block_data"
        block.header = Mock()
        block.header.gas_used = 1000000
        block.header.timestamp = int(time.time())
        return block

    @pytest.fixture
    def mock_previous_blocks(self):
        """Create mock previous blocks for testing."""
        blocks = []
        for i in range(3):
            block = Mock()
            block.header = Mock()
            block.header.timestamp = int(time.time()) - (3 - i) * 10
            blocks.append(block)
        return blocks

    @pytest.fixture
    def mock_utxos(self):
        """Create mock UTXO set for testing."""
        return {"utxo1": {"amount": 1000}}

    def test_consensus_engine_initialization(self, config):
        """Test ConsensusEngine initialization."""
        engine = ConsensusEngine(config)

        assert engine.config == config
        assert engine.proof_of_work is not None
        assert isinstance(engine.proof_of_work, ProofOfWork)

    def test_validate_block_success(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test successful block validation."""
        with patch.object(consensus_engine, '_validate_block_structure', return_value=True), \
             patch.object(consensus_engine, '_validate_block_timing', return_value=True), \
             patch.object(consensus_engine, '_validate_block_transactions', return_value=True), \
             patch.object(consensus_engine, '_validate_block_size', return_value=True), \
             patch.object(consensus_engine.proof_of_work, 'verify_block', return_value=True):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is True

    def test_validate_block_structure_failure(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test block validation failure due to structure."""
        with patch.object(consensus_engine, '_validate_block_structure', return_value=False):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is False

    def test_validate_block_proof_of_work_failure(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test block validation failure due to proof of work."""
        with patch.object(consensus_engine, '_validate_block_structure', return_value=True), \
             patch.object(consensus_engine.proof_of_work, 'verify_block', return_value=False):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is False

    def test_validate_block_timing_failure(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test block validation failure due to timing."""
        with patch.object(consensus_engine, '_validate_block_structure', return_value=True), \
             patch.object(consensus_engine.proof_of_work, 'verify_block', return_value=True), \
             patch.object(consensus_engine, '_validate_block_timing', return_value=False):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is False

    def test_validate_block_transactions_failure(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test block validation failure due to transactions."""
        with patch.object(consensus_engine, '_validate_block_structure', return_value=True), \
             patch.object(consensus_engine.proof_of_work, 'verify_block', return_value=True), \
             patch.object(consensus_engine, '_validate_block_timing', return_value=True), \
             patch.object(consensus_engine, '_validate_block_transactions', return_value=False):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is False

    def test_validate_block_size_failure(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test block validation failure due to size."""
        with patch.object(consensus_engine, '_validate_block_structure', return_value=True), \
             patch.object(consensus_engine.proof_of_work, 'verify_block', return_value=True), \
             patch.object(consensus_engine, '_validate_block_timing', return_value=True), \
             patch.object(consensus_engine, '_validate_block_transactions', return_value=True), \
             patch.object(consensus_engine, '_validate_block_size', return_value=False):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is False

    def test_validate_block_exception_handling(self, consensus_engine, mock_block, mock_previous_blocks, mock_utxos):
        """Test block validation exception handling."""
        with patch.object(consensus_engine, '_validate_block_structure', side_effect=Exception("Test error")):

            result = consensus_engine.validate_block(mock_block, mock_previous_blocks, mock_utxos)

            assert result is False

    def test_validate_block_structure_success(self, consensus_engine, mock_block):
        """Test successful block structure validation."""
        result = consensus_engine._validate_block_structure(mock_block)

        assert result is True
        mock_block._verify_merkle_root.assert_called_once()

    def test_validate_block_structure_no_transactions(self, consensus_engine):
        """Test block structure validation with no transactions."""
        mock_block = Mock()
        mock_block.transactions = []

        result = consensus_engine._validate_block_structure(mock_block)

        assert result is False

    def test_validate_block_structure_no_coinbase(self, consensus_engine):
        """Test block structure validation with no coinbase transaction."""
        mock_block = Mock()
        mock_transaction = Mock()
        mock_transaction.transaction_type.value = "transfer"
        mock_block.transactions = [mock_transaction]

        result = consensus_engine._validate_block_structure(mock_block)

        assert result is False

    def test_validate_block_structure_invalid_merkle_root(self, consensus_engine):
        """Test block structure validation with invalid merkle root."""
        mock_block = Mock()
        mock_transaction = Mock()
        mock_transaction.transaction_type.value = "coinbase"
        mock_block.transactions = [mock_transaction]
        mock_block._verify_merkle_root.return_value = False

        result = consensus_engine._validate_block_structure(mock_block)

        assert result is False

    def test_validate_block_timing_success(self, consensus_engine, mock_block, mock_previous_blocks):
        """Test successful block timing validation."""
        result = consensus_engine._validate_block_timing(mock_block, mock_previous_blocks)

        assert result is True

    def test_validate_block_timing_no_previous_blocks(self, consensus_engine, mock_block):
        """Test block timing validation with no previous blocks."""
        result = consensus_engine._validate_block_timing(mock_block, [])

        assert result is True

    def test_validate_block_timing_future_block(self, consensus_engine, mock_block, mock_previous_blocks):
        """Test block timing validation with future block."""
        mock_block.header.timestamp = int(time.time()) + 4000  # Too far in future

        result = consensus_engine._validate_block_timing(mock_block, mock_previous_blocks)

        assert result is False

    def test_validate_block_timing_old_block(self, consensus_engine, mock_block, mock_previous_blocks):
        """Test block timing validation with old block."""
        mock_block.header.timestamp = mock_previous_blocks[-1].header.timestamp - 1

        result = consensus_engine._validate_block_timing(mock_block, mock_previous_blocks)

        assert result is False

    def test_validate_block_timing_too_fast(self, consensus_engine, mock_block, mock_previous_blocks):
        """Test block timing validation with too fast block."""
        mock_block.header.timestamp = mock_previous_blocks[-1].header.timestamp + 0.5  # Less than min interval

        result = consensus_engine._validate_block_timing(mock_block, mock_previous_blocks)

        assert result is False

    def test_validate_block_transactions_success(self, consensus_engine, mock_block, mock_utxos):
        """Test successful block transactions validation."""
        mock_transaction = Mock()
        mock_transaction.is_valid.return_value = True
        mock_block.transactions = [mock_transaction]

        result = consensus_engine._validate_block_transactions(mock_block, mock_utxos)

        assert result is True
        mock_transaction.is_valid.assert_called_once_with(mock_utxos)

    def test_validate_block_transactions_invalid_transaction(self, consensus_engine, mock_block, mock_utxos):
        """Test block transactions validation with invalid transaction."""
        mock_transaction = Mock()
        mock_transaction.is_valid.return_value = False
        mock_block.transactions = [mock_transaction]

        result = consensus_engine._validate_block_transactions(mock_block, mock_utxos)

        assert result is False

    def test_validate_block_size_success(self, consensus_engine, mock_block):
        """Test successful block size validation."""
        result = consensus_engine._validate_block_size(mock_block)

        assert result is True

    def test_validate_block_size_too_large(self, consensus_engine, mock_block):
        """Test block size validation with too large block."""
        mock_block.to_bytes.return_value = b"x" * (consensus_engine.config.max_block_size + 1)

        result = consensus_engine._validate_block_size(mock_block)

        assert result is False

    def test_validate_block_size_too_many_transactions(self, consensus_engine, mock_block):
        """Test block size validation with too many transactions."""
        mock_block.transactions = [Mock()] * (consensus_engine.config.max_transactions_per_block + 1)

        result = consensus_engine._validate_block_size(mock_block)

        assert result is False

    def test_validate_block_size_too_much_gas(self, consensus_engine, mock_block):
        """Test block size validation with too much gas."""
        mock_block.header.gas_used = consensus_engine.config.max_gas_per_block + 1

        result = consensus_engine._validate_block_size(mock_block)

        assert result is False

    def test_calculate_next_difficulty(self, consensus_engine, mock_previous_blocks):
        """Test calculating next difficulty."""
        with patch.object(consensus_engine.proof_of_work, 'calculate_difficulty', return_value=15) as mock_calc:

            result = consensus_engine.calculate_next_difficulty(mock_previous_blocks)

            assert result == 15
            mock_calc.assert_called_once_with(mock_previous_blocks)

    def test_mine_block_success(self, consensus_engine):
        """Test successful block mining."""
        mock_transactions = [Mock()]
        mock_previous_block = Mock()
        mock_utxos = {"utxo1": {"amount": 1000}}

        # Create a mock block with transactions
        mock_block = Mock()
        mock_block.transactions = mock_transactions
        mock_block.header = Mock()

        with patch.object(consensus_engine, 'calculate_next_difficulty', return_value=10), \
             patch.object(Block, 'create_block', return_value=mock_block) as mock_create, \
             patch.object(consensus_engine.proof_of_work, 'mine_block', return_value=Mock()) as mock_mine, \
             patch.object(consensus_engine, 'validate_block', return_value=True) as mock_validate, \
             patch.object(Block, '__init__', return_value=None) as mock_block_init:

            result = consensus_engine.mine_block(mock_transactions, mock_previous_block, mock_utxos)

            assert result is not None
            mock_create.assert_called_once()
            mock_mine.assert_called_once()
            mock_validate.assert_called_once()

    def test_mine_block_mining_failure(self, consensus_engine):
        """Test block mining failure."""
        mock_transactions = [Mock()]
        mock_previous_block = Mock()
        mock_utxos = {"utxo1": {"amount": 1000}}

        with patch.object(consensus_engine, 'calculate_next_difficulty', return_value=10), \
             patch.object(Block, 'create_block', return_value=Mock()), \
             patch.object(consensus_engine.proof_of_work, 'mine_block', return_value=None):

            result = consensus_engine.mine_block(mock_transactions, mock_previous_block, mock_utxos)

            assert result is None

    def test_mine_block_validation_failure(self, consensus_engine):
        """Test block mining with validation failure."""
        mock_transactions = [Mock()]
        mock_previous_block = Mock()
        mock_utxos = {"utxo1": {"amount": 1000}}

        with patch.object(consensus_engine, 'calculate_next_difficulty', return_value=10), \
             patch.object(Block, 'create_block', return_value=Mock()), \
             patch.object(consensus_engine.proof_of_work, 'mine_block', return_value=Mock()), \
             patch.object(consensus_engine, 'validate_block', return_value=False):

            result = consensus_engine.mine_block(mock_transactions, mock_previous_block, mock_utxos)

            assert result is None

    def test_mine_block_exception_handling(self, consensus_engine):
        """Test block mining exception handling."""
        mock_transactions = [Mock()]
        mock_previous_block = Mock()
        mock_utxos = {"utxo1": {"amount": 1000}}

        with patch.object(consensus_engine, 'calculate_next_difficulty', side_effect=Exception("Test error")):

            result = consensus_engine.mine_block(mock_transactions, mock_previous_block, mock_utxos)

            assert result is None

    def test_get_consensus_info_no_blocks(self, consensus_engine):
        """Test getting consensus info with no blocks."""
        result = consensus_engine.get_consensus_info([])

        assert result == {}

    def test_get_consensus_info_single_block(self, consensus_engine):
        """Test getting consensus info with single block."""
        mock_block = Mock()
        mock_block.header = Mock()
        mock_block.header.difficulty = 10
        mock_block.header.block_height = 1
        mock_block.header.timestamp = 1000000

        result = consensus_engine.get_consensus_info([mock_block])

        assert result["current_difficulty"] == 10
        assert result["block_height"] == 1
        assert result["average_block_time"] == 0
        assert result["estimated_hashrate"] == 0

    def test_get_consensus_info_multiple_blocks(self, consensus_engine):
        """Test getting consensus info with multiple blocks."""
        blocks = []
        base_time = 1000000
        for i in range(5):
            mock_block = Mock()
            mock_block.header = Mock()
            mock_block.header.difficulty = 10
            mock_block.header.block_height = i + 1
            mock_block.header.timestamp = base_time + (i * 10)
            blocks.append(mock_block)

        result = consensus_engine.get_consensus_info(blocks)

        assert result["current_difficulty"] == 10
        assert result["block_height"] == 5
        assert result["average_block_time"] == 10.0
        assert result["estimated_hashrate"] > 0
        assert result["target_block_time"] == consensus_engine.config.target_block_time
        assert result["difficulty_adjustment_interval"] == consensus_engine.config.difficulty_adjustment_interval
