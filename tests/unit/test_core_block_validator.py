"""
Comprehensive unit tests for the enhanced block validation system.
"""

import time
from unittest.mock import Mock, patch

import pytest

from dubchain.core.block import Block, BlockHeader
from dubchain.core.block_validator import (
    BlockValidationConfig,
    BlockValidator,
    ValidationError,
    ValidationResult,
)
from dubchain.core.transaction import (
    UTXO,
    Transaction,
    TransactionInput,
    TransactionOutput,
    TransactionType,
)
from dubchain.crypto.hashing import Hash, SHA256Hasher


class TestValidationError:
    """Test the ValidationError enum."""

    def test_validation_error_values(self):
        """Test that validation errors have proper string values."""
        assert ValidationError.INVALID_HEADER.value == "invalid_header"
        assert ValidationError.INVALID_TIMESTAMP.value == "invalid_timestamp"
        assert ValidationError.INVALID_SIZE.value == "invalid_size"
        assert ValidationError.INVALID_DIFFICULTY.value == "invalid_difficulty"
        assert ValidationError.INVALID_REWARD.value == "invalid_reward"
        assert ValidationError.INVALID_GAS.value == "invalid_gas"
        assert ValidationError.INVALID_TRANSACTIONS.value == "invalid_transactions"
        assert ValidationError.INVALID_MERKLE_ROOT.value == "invalid_merkle_root"
        assert ValidationError.INVALID_PROOF_OF_WORK.value == "invalid_proof_of_work"
        assert ValidationError.INVALID_PREVIOUS_HASH.value == "invalid_previous_hash"
        assert ValidationError.INVALID_BLOCK_HEIGHT.value == "invalid_block_height"
        assert ValidationError.INVALID_VERSION.value == "invalid_version"
        assert ValidationError.INVALID_NONCE.value == "invalid_nonce"
        assert ValidationError.INVALID_EXTRA_DATA.value == "invalid_extra_data"


class TestValidationResult:
    """Test the ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.validation_time == 0.0
        assert result.block_size == 0
        assert result.gas_efficiency == 0.0

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        errors = [ValidationError.INVALID_HEADER, ValidationError.INVALID_TIMESTAMP]
        warnings = ["Warning 1", "Warning 2"]

        result = ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            validation_time=0.5,
            block_size=1024,
            gas_efficiency=0.8,
        )

        assert result.is_valid is False
        assert result.errors == errors
        assert result.warnings == warnings
        assert result.validation_time == 0.5
        assert result.block_size == 1024
        assert result.gas_efficiency == 0.8


class TestBlockValidationConfig:
    """Test the BlockValidationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BlockValidationConfig()

        assert config.max_block_size == 32 * 1024 * 1024  # 32MB
        assert config.max_transaction_count == 10000
        assert config.max_extra_data_size == 1024
        assert config.max_future_time == 3600  # 1 hour
        assert config.min_block_interval == 1  # 1 second
        assert config.max_block_interval == 7200  # 2 hours
        assert config.min_difficulty == 1
        assert config.max_difficulty == 256
        assert config.difficulty_adjustment_interval == 2016
        assert config.target_block_time == 600  # 10 minutes
        assert config.min_gas_limit == 1000000  # 1M gas
        assert config.max_gas_limit == 100000000  # 100M gas
        assert config.gas_limit_adjustment_factor == 0.125
        assert config.base_reward == 50 * 10**8  # 50 coins
        assert config.halving_interval == 210000
        assert config.max_halvings == 64
        assert config.min_version == 1
        assert config.max_version == 3
        assert config.max_nonce == 2**64 - 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BlockValidationConfig(
            max_block_size=16 * 1024 * 1024,  # 16MB
            max_transaction_count=5000,
            min_difficulty=2,
            max_difficulty=128,
        )

        assert config.max_block_size == 16 * 1024 * 1024
        assert config.max_transaction_count == 5000
        assert config.min_difficulty == 2
        assert config.max_difficulty == 128


class TestBlockValidator:
    """Test the BlockValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a block validator for testing."""
        return BlockValidator()

    @pytest.fixture
    def custom_validator(self):
        """Create a block validator with custom config."""
        config = BlockValidationConfig(
            max_block_size=1024,  # Very small for testing
            max_transaction_count=2,
            min_difficulty=1,
            max_difficulty=10,
        )
        return BlockValidator(config)

    @pytest.fixture
    def genesis_block(self):
        """Create a genesis block for testing."""
        return Block.create_genesis_block(
            coinbase_recipient="genesis_recipient", coinbase_amount=1000000
        )

    @pytest.fixture
    def valid_block(self, genesis_block):
        """Create a valid block for testing."""
        # Create a coinbase transaction
        coinbase_tx = Transaction.create_coinbase(
            recipient_address="miner", amount=50000000, block_height=1
        )

        # Create block with difficulty 1 and mine it to find a valid nonce
        from dubchain.core.consensus import ConsensusConfig, ProofOfWork

        block = Block.create_block(
            transactions=[coinbase_tx], previous_block=genesis_block, difficulty=1
        )

        # Mine the block to find a valid nonce
        config = ConsensusConfig()
        pow_engine = ProofOfWork(config)
        mined_header = pow_engine.mine_block(
            block.header, max_nonce=1000
        )  # Limit nonce for testing

        if mined_header:
            return Block(header=mined_header, transactions=block.transactions)
        else:
            # If mining fails, create a block with difficulty 0 (for testing only)
            return Block.create_block(
                transactions=[coinbase_tx], previous_block=genesis_block, difficulty=0
            )

    @pytest.fixture
    def sample_utxos(self):
        """Create sample UTXOs for testing."""
        return {}

    def _create_invalid_block(self, valid_block, **header_overrides):
        """Helper to create a block with invalid header fields."""
        # Create new header with overridden fields, bypassing constructor validation
        header_fields = {
            "version": valid_block.header.version,
            "previous_hash": valid_block.header.previous_hash,
            "merkle_root": valid_block.header.merkle_root,
            "timestamp": valid_block.header.timestamp,
            "difficulty": valid_block.header.difficulty,
            "nonce": valid_block.header.nonce,
            "block_height": valid_block.header.block_height,
            "gas_limit": valid_block.header.gas_limit,
            "gas_used": valid_block.header.gas_used,
            "extra_data": valid_block.header.extra_data,
        }

        # Override with the specified values
        header_fields.update(
            {k: v for k, v in header_overrides.items() if v is not None}
        )

        # Create header using object.__new__ to bypass constructor validation
        invalid_header = object.__new__(BlockHeader)
        for field, value in header_fields.items():
            object.__setattr__(invalid_header, field, value)

        # Create block using object.__new__ to bypass constructor validation
        invalid_block = object.__new__(Block)
        object.__setattr__(invalid_block, "header", invalid_header)
        object.__setattr__(invalid_block, "transactions", valid_block.transactions)

        return invalid_block

    def test_validator_creation(self, validator):
        """Test creating a block validator."""
        assert validator.config is not None
        assert isinstance(validator.config, BlockValidationConfig)
        assert len(validator._validation_cache) == 0
        assert validator._cache_ttl == 300.0

    def test_validate_genesis_block(self, validator, genesis_block):
        """Test validating a genesis block."""
        result = validator.validate_block(genesis_block)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.block_size > 0
        assert result.validation_time > 0

    def test_validate_valid_block(self, validator, valid_block, genesis_block):
        """Test validating a valid block."""
        result = validator.validate_block(valid_block, genesis_block)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.block_size > 0
        assert result.validation_time > 0

    def test_validate_invalid_timestamp_future(
        self, validator, valid_block, genesis_block
    ):
        """Test validating block with timestamp too far in future."""
        # Create block with timestamp far in the future
        future_timestamp = int(time.time()) + 7200  # 2 hours in future
        block = self._create_invalid_block(valid_block, timestamp=future_timestamp)

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_TIMESTAMP in result.errors

    def test_validate_invalid_timestamp_past(
        self, validator, valid_block, genesis_block
    ):
        """Test validating block with timestamp too far in past."""
        # Create block with timestamp far in the past and difficulty 0 to avoid proof of work issues
        past_timestamp = int(time.time()) - 14400  # 4 hours ago
        block = self._create_invalid_block(
            valid_block, timestamp=past_timestamp, difficulty=0
        )

        result = validator.validate_block(block, genesis_block)

        # Should be valid but with warning
        assert result.is_valid is True
        assert any("2 hours old" in warning for warning in result.warnings)

    def test_validate_invalid_difficulty(
        self, custom_validator, valid_block, genesis_block
    ):
        """Test validating block with invalid difficulty."""
        # Create block with difficulty too high
        block = self._create_invalid_block(
            valid_block, difficulty=20
        )  # Exceeds max of 10

        result = custom_validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_DIFFICULTY in result.errors

    def test_validate_invalid_gas_limit(
        self, custom_validator, valid_block, genesis_block
    ):
        """Test validating block with invalid gas limit."""
        # Create block with gas limit too high
        block = self._create_invalid_block(
            valid_block, gas_limit=200000000
        )  # Exceeds max

        result = custom_validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_GAS in result.errors

    def test_validate_invalid_gas_used(self, validator, valid_block, genesis_block):
        """Test validating block with gas used exceeding limit."""
        # Create block with gas used exceeding limit
        block = self._create_invalid_block(
            valid_block, gas_used=20000000
        )  # Exceeds limit

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_GAS in result.errors

    def test_validate_invalid_version(self, validator, valid_block, genesis_block):
        """Test validating block with invalid version."""
        # Create block with invalid version
        block = self._create_invalid_block(
            valid_block, version=5
        )  # Exceeds max version of 3

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_VERSION in result.errors

    def test_validate_invalid_nonce(self, validator, valid_block, genesis_block):
        """Test validating block with invalid nonce."""
        # Create block with invalid nonce
        block = self._create_invalid_block(valid_block, nonce=-1)  # Invalid nonce

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_NONCE in result.errors

    def test_validate_invalid_extra_data(self, validator, valid_block, genesis_block):
        """Test validating block with invalid extra data."""
        # Create block with extra data too large
        large_extra_data = b"x" * 2048  # Exceeds max of 1024
        block = self._create_invalid_block(valid_block, extra_data=large_extra_data)

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_EXTRA_DATA in result.errors

    def test_validate_invalid_previous_hash(
        self, validator, valid_block, genesis_block
    ):
        """Test validating block with invalid previous hash."""
        # Create block with wrong previous hash
        block = self._create_invalid_block(
            valid_block, previous_hash=SHA256Hasher.hash("wrong_hash")
        )

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_PREVIOUS_HASH in result.errors

    def test_validate_invalid_block_height(self, validator, valid_block, genesis_block):
        """Test validating block with invalid block height."""
        # Create block with wrong block height
        block = self._create_invalid_block(valid_block, block_height=5)  # Should be 1

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_BLOCK_HEIGHT in result.errors

    def test_validate_genesis_with_nonzero_previous_hash(
        self, validator, genesis_block
    ):
        """Test validating genesis block with non-zero previous hash."""
        # Create an invalid genesis block with non-zero previous hash
        invalid_previous_hash = SHA256Hasher.hash("non_zero")
        block = self._create_invalid_block(
            genesis_block, previous_hash=invalid_previous_hash
        )

        result = validator.validate_block(block)

        assert result.is_valid is False
        assert ValidationError.INVALID_PREVIOUS_HASH in result.errors

    def test_validate_genesis_with_nonzero_height(self, validator, genesis_block):
        """Test validating genesis block with non-zero height."""
        # Create an invalid genesis block with non-zero height
        block = self._create_invalid_block(genesis_block, block_height=1)

        result = validator.validate_block(block)

        assert result.is_valid is False
        assert ValidationError.INVALID_BLOCK_HEIGHT in result.errors

    def test_validate_block_without_coinbase(self, validator, genesis_block):
        """Test validating block without coinbase transaction."""
        # Create regular transaction (not coinbase)
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=0
        )
        output = TransactionOutput(amount=1000, recipient_address="recipient")

        regular_tx = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )

        # Create a valid block first, then modify it to have invalid transactions
        valid_block = Block.create_block(
            transactions=[Transaction.create_coinbase("miner", 50000000, 1)],
            previous_block=genesis_block,
            difficulty=0,
        )

        # Create invalid block with regular transaction instead of coinbase
        invalid_block = object.__new__(Block)
        object.__setattr__(invalid_block, "header", valid_block.header)
        object.__setattr__(invalid_block, "transactions", [regular_tx])

        result = validator.validate_block(invalid_block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_TRANSACTIONS in result.errors

    def test_validate_empty_block(self, validator, genesis_block):
        """Test validating empty block."""
        # Create a valid block first, then modify it to have no transactions
        valid_block = Block.create_block(
            transactions=[Transaction.create_coinbase("miner", 50000000, 1)],
            previous_block=genesis_block,
            difficulty=0,
        )

        # Create invalid block with no transactions
        invalid_block = object.__new__(Block)
        object.__setattr__(invalid_block, "header", valid_block.header)
        object.__setattr__(invalid_block, "transactions", [])

        result = validator.validate_block(invalid_block, genesis_block)

        assert result.is_valid is False
        assert ValidationError.INVALID_SIZE in result.errors

    def test_validation_caching(self, validator, valid_block, genesis_block):
        """Test validation result caching."""
        # First validation
        result1 = validator.validate_block(valid_block, genesis_block, use_cache=True)

        # Second validation should use cache
        result2 = validator.validate_block(valid_block, genesis_block, use_cache=True)

        # Results should be the same
        assert result1.is_valid == result2.is_valid
        assert result1.errors == result2.errors
        assert result1.warnings == result2.warnings

        # Cache should contain the result
        block_hash = valid_block.get_hash().to_hex()
        assert block_hash in validator._validation_cache

    def test_validation_without_cache(self, validator, valid_block, genesis_block):
        """Test validation without caching."""
        # First validation with cache
        result1 = validator.validate_block(valid_block, genesis_block, use_cache=True)

        # Second validation without cache
        result2 = validator.validate_block(valid_block, genesis_block, use_cache=False)

        # Results should be the same but cache should not be used
        assert result1.is_valid == result2.is_valid
        assert result1.errors == result2.errors
        assert result1.warnings == result2.warnings

    def test_clear_cache(self, validator, valid_block, genesis_block):
        """Test clearing validation cache."""
        # Add something to cache
        validator.validate_block(valid_block, genesis_block, use_cache=True)
        assert len(validator._validation_cache) > 0

        # Clear cache
        validator.clear_cache()
        assert len(validator._validation_cache) == 0

    def test_get_cache_stats(self, validator, valid_block, genesis_block):
        """Test getting cache statistics."""
        # Initially empty cache
        stats = validator.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_ttl"] == 300.0

        # Add to cache
        validator.validate_block(valid_block, genesis_block, use_cache=True)

        # Check updated stats
        stats = validator.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["cache_ttl"] == 300.0

    def test_gas_efficiency_calculation(self, validator, genesis_block):
        """Test gas efficiency calculation."""
        # Create a valid block first
        valid_block = Block.create_block(
            transactions=[Transaction.create_coinbase("miner", 50000000, 1)],
            previous_block=genesis_block,
            difficulty=0,
        )

        # Create block with specific gas usage (50% efficiency)
        block = self._create_invalid_block(
            valid_block, gas_used=5000000, gas_limit=10000000
        )

        result = validator.validate_block(block, genesis_block)

        assert result.gas_efficiency == 0.5

    def test_block_size_calculation(self, validator, valid_block, genesis_block):
        """Test block size calculation."""
        result = validator.validate_block(valid_block, genesis_block)

        assert result.block_size > 0
        assert result.block_size == len(valid_block.to_bytes())

    def test_validation_time_measurement(self, validator, valid_block, genesis_block):
        """Test validation time measurement."""
        result = validator.validate_block(valid_block, genesis_block)

        assert result.validation_time > 0
        assert result.validation_time < 1.0  # Should be fast

    def test_multiple_validation_errors(self, validator, valid_block, genesis_block):
        """Test block with multiple validation errors."""
        # Create block with multiple issues
        block = self._create_invalid_block(
            valid_block,
            version=5,  # Invalid version
            previous_hash=SHA256Hasher.hash("wrong_hash"),  # Wrong previous hash
            timestamp=int(time.time()) + 7200,  # Future timestamp
            difficulty=300,  # Invalid difficulty
            nonce=-1,  # Invalid nonce
            block_height=5,  # Wrong height
            gas_limit=200000000,  # Invalid gas limit
            gas_used=30000000,  # Gas used exceeds limit
            extra_data=b"x" * 2048,  # Invalid extra data
        )

        result = validator.validate_block(block, genesis_block)

        assert result.is_valid is False
        assert len(result.errors) > 1
        assert ValidationError.INVALID_VERSION in result.errors
        assert ValidationError.INVALID_PREVIOUS_HASH in result.errors
        assert ValidationError.INVALID_TIMESTAMP in result.errors
        assert ValidationError.INVALID_DIFFICULTY in result.errors
        assert ValidationError.INVALID_NONCE in result.errors
        assert ValidationError.INVALID_BLOCK_HEIGHT in result.errors
        assert ValidationError.INVALID_GAS in result.errors
        assert ValidationError.INVALID_EXTRA_DATA in result.errors
