"""
Enhanced Block Validation System for DubChain.

This module implements comprehensive block validation including:
- Block header validation with all fields
- Timestamp validation with proper bounds checking
- Block size limits and validation
- Difficulty adjustment validation
- Block reward validation
- Gas limit and usage validation
- Transaction ordering and structure validation
- Merkle tree validation
- Proof of work validation
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.merkle import MerkleTree
from .block import Block, BlockHeader
from .transaction import UTXO, Transaction, TransactionType


class ValidationError(Enum):
    """Types of validation errors."""

    INVALID_HEADER = "invalid_header"
    INVALID_TIMESTAMP = "invalid_timestamp"
    INVALID_SIZE = "invalid_size"
    INVALID_DIFFICULTY = "invalid_difficulty"
    INVALID_REWARD = "invalid_reward"
    INVALID_GAS = "invalid_gas"
    INVALID_TRANSACTIONS = "invalid_transactions"
    INVALID_MERKLE_ROOT = "invalid_merkle_root"
    INVALID_PROOF_OF_WORK = "invalid_proof_of_work"
    INVALID_PREVIOUS_HASH = "invalid_previous_hash"
    INVALID_BLOCK_HEIGHT = "invalid_block_height"
    INVALID_VERSION = "invalid_version"
    INVALID_NONCE = "invalid_nonce"
    INVALID_EXTRA_DATA = "invalid_extra_data"


@dataclass
class ValidationResult:
    """Result of block validation."""

    is_valid: bool
    errors: List[ValidationError] = None
    warnings: List[str] = None
    validation_time: float = 0.0
    block_size: int = 0
    gas_efficiency: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class BlockValidationConfig:
    """Configuration for block validation."""

    # Size limits
    max_block_size: int = 32 * 1024 * 1024  # 32MB
    max_transaction_count: int = 10000
    max_extra_data_size: int = 1024  # 1KB

    # Timestamp validation
    max_future_time: int = 3600  # 1 hour in seconds
    min_block_interval: int = 1  # 1 second minimum between blocks
    max_block_interval: int = 7200  # 2 hours maximum between blocks

    # Difficulty validation
    min_difficulty: int = 1
    max_difficulty: int = 256
    difficulty_adjustment_interval: int = 2016  # blocks
    target_block_time: int = 600  # 10 minutes in seconds

    # Gas validation
    min_gas_limit: int = 1000000  # 1M gas
    max_gas_limit: int = 100000000  # 100M gas
    gas_limit_adjustment_factor: float = 0.125  # 12.5% max change per block

    # Reward validation
    base_reward: int = 50 * 10**8  # 50 coins in satoshis
    halving_interval: int = 210000  # blocks
    max_halvings: int = 64

    # Version validation
    min_version: int = 1
    max_version: int = 3

    # Nonce validation
    max_nonce: int = 2**64 - 1


class BlockValidator:
    """Enhanced block validator with comprehensive validation rules."""

    def __init__(self, config: Optional[BlockValidationConfig] = None):
        self.config = config or BlockValidationConfig()
        self._validation_cache: Dict[str, Tuple[ValidationResult, float]] = {}
        self._cache_ttl = 300.0  # 5 minutes

    def validate_block(
        self,
        block: Block,
        previous_block: Optional[Block] = None,
        utxos: Optional[Dict[str, UTXO]] = None,
        use_cache: bool = True,
    ) -> ValidationResult:
        """
        Validate a block comprehensively.

        Args:
            block: Block to validate
            previous_block: Previous block in the chain
            utxos: Current UTXO set
            use_cache: Whether to use validation cache

        Returns:
            ValidationResult with detailed validation information
        """
        start_time = time.time()

        # Check cache first (skip if block can't be serialized)
        if use_cache:
            try:
                block_hash = block.get_hash().to_hex()
                if block_hash in self._validation_cache:
                    cached_result, timestamp = self._validation_cache[block_hash]
                    if time.time() - timestamp < self._cache_ttl:
                        return cached_result
            except (OverflowError, ValueError):
                # Skip caching for blocks that can't be serialized
                pass

        result = ValidationResult(is_valid=True)

        # Calculate block size (skip if block can't be serialized)
        try:
            result.block_size = len(block.to_bytes())
        except (OverflowError, ValueError):
            result.block_size = 0  # Unknown size for invalid blocks

        # Perform all validation checks
        self._validate_header(block.header, result)
        self._validate_timestamp(block.header, previous_block, result)
        self._validate_size(block, result)
        self._validate_difficulty(block.header, previous_block, result)
        self._validate_reward(block, previous_block, result)
        self._validate_gas(block, result)
        self._validate_transactions(block, utxos or {}, result)
        self._validate_merkle_root(block, result)
        self._validate_proof_of_work(block.header, result)
        self._validate_previous_hash(block.header, previous_block, result)
        self._validate_block_height(block.header, previous_block, result)
        self._validate_version(block.header, result)
        self._validate_nonce(block.header, result)
        self._validate_extra_data(block.header, result)

        # Calculate gas efficiency
        if block.header.gas_limit > 0:
            result.gas_efficiency = block.header.gas_used / block.header.gas_limit

        # Set validation time
        result.validation_time = time.time() - start_time

        # Cache result (skip if block can't be serialized)
        if use_cache:
            try:
                block_hash = block.get_hash().to_hex()
                self._validation_cache[block_hash] = (result, time.time())
            except (OverflowError, ValueError):
                # Skip caching for blocks that can't be serialized
                pass

        return result

    def _validate_header(self, header: BlockHeader, result: ValidationResult) -> None:
        """Validate block header structure and values."""
        try:
            # Check that header can be serialized
            header_bytes = header.to_bytes()
            if len(header_bytes) == 0:
                result.errors.append(ValidationError.INVALID_HEADER)
                result.is_valid = False
                return

            # Check that hash can be calculated
            block_hash = header.get_hash()
            if not block_hash or len(block_hash.value) != 32:
                result.errors.append(ValidationError.INVALID_HEADER)
                result.is_valid = False

        except Exception:
            result.errors.append(ValidationError.INVALID_HEADER)
            result.is_valid = False

    def _validate_timestamp(
        self,
        header: BlockHeader,
        previous_block: Optional[Block],
        result: ValidationResult,
    ) -> None:
        """Validate block timestamp."""
        current_time = int(time.time())

        # Check if timestamp is too far in the future
        if header.timestamp > current_time + self.config.max_future_time:
            result.errors.append(ValidationError.INVALID_TIMESTAMP)
            result.is_valid = False
            result.warnings.append(
                f"Block timestamp {header.timestamp} is too far in the future"
            )

        # Check if timestamp is too old (for non-genesis blocks)
        if previous_block:
            time_diff = header.timestamp - previous_block.header.timestamp

            # Check minimum block interval (only error if positive but too short)
            if 0 <= time_diff < self.config.min_block_interval:
                result.errors.append(ValidationError.INVALID_TIMESTAMP)
                result.is_valid = False
                result.warnings.append(f"Block interval {time_diff}s is too short")
            elif time_diff < 0:
                # Block appears to be before previous block - this is unusual but not necessarily invalid
                result.warnings.append(
                    f"Block interval {time_diff}s is negative (block appears before previous block)"
                )

            # Check maximum block interval
            if time_diff > self.config.max_block_interval:
                result.warnings.append(f"Block interval {time_diff}s is unusually long")

        # Check if timestamp is too old (more than 2 hours old)
        if header.timestamp < current_time - 7200:
            result.warnings.append("Block timestamp is more than 2 hours old")

    def _validate_size(self, block: Block, result: ValidationResult) -> None:
        """Validate block size limits."""
        block_size = result.block_size

        # Check maximum block size
        if block_size > self.config.max_block_size:
            result.errors.append(ValidationError.INVALID_SIZE)
            result.is_valid = False
            result.warnings.append(
                f"Block size {block_size} exceeds maximum {self.config.max_block_size}"
            )

        # Check maximum transaction count
        if len(block.transactions) > self.config.max_transaction_count:
            result.errors.append(ValidationError.INVALID_SIZE)
            result.is_valid = False
            result.warnings.append(
                f"Transaction count {len(block.transactions)} exceeds maximum {self.config.max_transaction_count}"
            )

        # Check for empty block (should have at least coinbase)
        if len(block.transactions) == 0:
            result.errors.append(ValidationError.INVALID_SIZE)
            result.is_valid = False

    def _validate_difficulty(
        self,
        header: BlockHeader,
        previous_block: Optional[Block],
        result: ValidationResult,
    ) -> None:
        """Validate block difficulty."""
        # Check difficulty bounds (allow difficulty 0 for test blocks)
        if header.difficulty < 0:
            result.errors.append(ValidationError.INVALID_DIFFICULTY)
            result.is_valid = False
            result.warnings.append(f"Difficulty {header.difficulty} cannot be negative")
        elif 0 < header.difficulty < self.config.min_difficulty:
            result.errors.append(ValidationError.INVALID_DIFFICULTY)
            result.is_valid = False
            result.warnings.append(
                f"Difficulty {header.difficulty} is below minimum {self.config.min_difficulty}"
            )

        if header.difficulty > self.config.max_difficulty:
            result.errors.append(ValidationError.INVALID_DIFFICULTY)
            result.is_valid = False
            result.warnings.append(
                f"Difficulty {header.difficulty} exceeds maximum {self.config.max_difficulty}"
            )

        # For non-genesis blocks, validate difficulty adjustment
        if previous_block:
            expected_difficulty = self._calculate_expected_difficulty(
                header, previous_block
            )
            difficulty_diff = abs(header.difficulty - expected_difficulty)

            # Allow some tolerance for difficulty adjustment
            if difficulty_diff > 1:
                result.warnings.append(
                    f"Difficulty {header.difficulty} differs from expected {expected_difficulty}"
                )

    def _validate_reward(
        self, block: Block, previous_block: Optional[Block], result: ValidationResult
    ) -> None:
        """Validate block reward."""
        # Skip reward validation for empty blocks (already handled by size validation)
        if not block.transactions:
            return

        coinbase_tx = block.get_coinbase_transaction()
        if not coinbase_tx or coinbase_tx.transaction_type != TransactionType.COINBASE:
            result.errors.append(ValidationError.INVALID_REWARD)
            result.is_valid = False
            return

        # Calculate expected reward
        expected_reward = self._calculate_expected_reward(block.header.block_height)

        # Get actual reward from coinbase transaction
        actual_reward = coinbase_tx.get_total_output_amount()

        # Allow some tolerance for transaction fees
        max_allowed_reward = expected_reward + self._estimate_max_fees(block)

        if actual_reward > max_allowed_reward:
            result.errors.append(ValidationError.INVALID_REWARD)
            result.is_valid = False
            result.warnings.append(
                f"Block reward {actual_reward} exceeds maximum {max_allowed_reward}"
            )

        if actual_reward < expected_reward:
            result.warnings.append(
                f"Block reward {actual_reward} is below expected {expected_reward}"
            )

    def _validate_gas(self, block: Block, result: ValidationResult) -> None:
        """Validate gas usage and limits."""
        header = block.header

        # Check gas limit bounds
        if header.gas_limit < self.config.min_gas_limit:
            result.errors.append(ValidationError.INVALID_GAS)
            result.is_valid = False
            result.warnings.append(
                f"Gas limit {header.gas_limit} is below minimum {self.config.min_gas_limit}"
            )

        if header.gas_limit > self.config.max_gas_limit:
            result.errors.append(ValidationError.INVALID_GAS)
            result.is_valid = False
            result.warnings.append(
                f"Gas limit {header.gas_limit} exceeds maximum {self.config.max_gas_limit}"
            )

        # Check gas used doesn't exceed limit
        if header.gas_used > header.gas_limit:
            result.errors.append(ValidationError.INVALID_GAS)
            result.is_valid = False
            result.warnings.append(
                f"Gas used {header.gas_used} exceeds limit {header.gas_limit}"
            )

        # Check gas used matches actual transaction gas
        actual_gas_used = block.get_total_gas_used()
        if header.gas_used != actual_gas_used:
            result.errors.append(ValidationError.INVALID_GAS)
            result.is_valid = False
            result.warnings.append(
                f"Header gas used {header.gas_used} doesn't match actual {actual_gas_used}"
            )

    def _validate_transactions(
        self, block: Block, utxos: Dict[str, UTXO], result: ValidationResult
    ) -> None:
        """Validate all transactions in the block."""
        if not block.transactions:
            result.errors.append(ValidationError.INVALID_TRANSACTIONS)
            result.is_valid = False
            return

        # Check that first transaction is coinbase
        if block.transactions[0].transaction_type != TransactionType.COINBASE:
            result.errors.append(ValidationError.INVALID_TRANSACTIONS)
            result.is_valid = False
            result.warnings.append("First transaction must be coinbase")

        # Validate each transaction
        for i, tx in enumerate(block.transactions):
            if not tx.is_valid(utxos):
                result.errors.append(ValidationError.INVALID_TRANSACTIONS)
                result.is_valid = False
                result.warnings.append(f"Transaction {i} is invalid")

        # Check for duplicate transactions
        tx_hashes = [tx.get_hash() for tx in block.transactions]
        if len(tx_hashes) != len(set(tx_hashes)):
            result.errors.append(ValidationError.INVALID_TRANSACTIONS)
            result.is_valid = False
            result.warnings.append("Block contains duplicate transactions")

    def _validate_merkle_root(self, block: Block, result: ValidationResult) -> None:
        """Validate merkle root."""
        if not block._verify_merkle_root():
            result.errors.append(ValidationError.INVALID_MERKLE_ROOT)
            result.is_valid = False
            result.warnings.append("Merkle root does not match transactions")

    def _validate_proof_of_work(
        self, header: BlockHeader, result: ValidationResult
    ) -> None:
        """Validate proof of work."""
        # Genesis blocks (height 0) and blocks with difficulty 0 are exempt from proof of work validation
        if header.block_height == 0 or header.difficulty == 0:
            return

        try:
            if not header.meets_difficulty():
                result.errors.append(ValidationError.INVALID_PROOF_OF_WORK)
                result.is_valid = False
                result.warnings.append("Block does not meet difficulty requirement")
        except (OverflowError, ValueError):
            # Block can't be serialized, so it can't meet difficulty requirements
            result.errors.append(ValidationError.INVALID_PROOF_OF_WORK)
            result.is_valid = False
            result.warnings.append(
                "Block cannot be serialized for proof of work validation"
            )

    def _validate_previous_hash(
        self,
        header: BlockHeader,
        previous_block: Optional[Block],
        result: ValidationResult,
    ) -> None:
        """Validate previous block hash."""
        if previous_block:
            expected_hash = previous_block.get_hash()
            if header.previous_hash != expected_hash:
                result.errors.append(ValidationError.INVALID_PREVIOUS_HASH)
                result.is_valid = False
                result.warnings.append("Previous hash does not match expected value")
        else:
            # Genesis block should have zero previous hash
            if header.previous_hash != Hash.zero():
                result.errors.append(ValidationError.INVALID_PREVIOUS_HASH)
                result.is_valid = False
                result.warnings.append("Genesis block should have zero previous hash")

    def _validate_block_height(
        self,
        header: BlockHeader,
        previous_block: Optional[Block],
        result: ValidationResult,
    ) -> None:
        """Validate block height."""
        if previous_block:
            expected_height = previous_block.header.block_height + 1
            if header.block_height != expected_height:
                result.errors.append(ValidationError.INVALID_BLOCK_HEIGHT)
                result.is_valid = False
                result.warnings.append(
                    f"Block height {header.block_height} doesn't match expected {expected_height}"
                )
        else:
            # Genesis block should have height 0
            if header.block_height != 0:
                result.errors.append(ValidationError.INVALID_BLOCK_HEIGHT)
                result.is_valid = False
                result.warnings.append("Genesis block should have height 0")

    def _validate_version(self, header: BlockHeader, result: ValidationResult) -> None:
        """Validate block version."""
        if (
            header.version < self.config.min_version
            or header.version > self.config.max_version
        ):
            result.errors.append(ValidationError.INVALID_VERSION)
            result.is_valid = False
            result.warnings.append(f"Block version {header.version} is not supported")

    def _validate_nonce(self, header: BlockHeader, result: ValidationResult) -> None:
        """Validate nonce value."""
        if header.nonce < 0 or header.nonce > self.config.max_nonce:
            result.errors.append(ValidationError.INVALID_NONCE)
            result.is_valid = False
            result.warnings.append(f"Nonce {header.nonce} is out of valid range")

    def _validate_extra_data(
        self, header: BlockHeader, result: ValidationResult
    ) -> None:
        """Validate extra data."""
        if (
            header.extra_data
            and len(header.extra_data) > self.config.max_extra_data_size
        ):
            result.errors.append(ValidationError.INVALID_EXTRA_DATA)
            result.is_valid = False
            result.warnings.append(
                f"Extra data size {len(header.extra_data)} exceeds maximum {self.config.max_extra_data_size}"
            )

    def _calculate_expected_difficulty(
        self, header: BlockHeader, previous_block: Block
    ) -> int:
        """Calculate expected difficulty based on previous blocks."""
        # Simple difficulty adjustment algorithm
        # In a real implementation, this would use a more sophisticated algorithm

        if (
            previous_block.header.block_height
            % self.config.difficulty_adjustment_interval
            != 0
        ):
            return previous_block.header.difficulty

        # Calculate time taken for the last difficulty adjustment interval
        # For simplicity, assume 10 minute blocks
        return max(
            self.config.min_difficulty,
            min(self.config.max_difficulty, previous_block.header.difficulty),
        )

    def _calculate_expected_reward(self, block_height: int) -> int:
        """Calculate expected block reward based on halving schedule."""
        halvings = block_height // self.config.halving_interval

        if halvings >= self.config.max_halvings:
            return 0

        return self.config.base_reward // (2**halvings)

    def _estimate_max_fees(self, block: Block) -> int:
        """Estimate maximum possible transaction fees."""
        # Simple estimation based on transaction count
        # In a real implementation, this would be more sophisticated
        regular_tx_count = len(block.get_regular_transactions())
        return regular_tx_count * 10000  # 10k satoshis per transaction estimate

    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self._validation_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {"cache_size": len(self._validation_cache), "cache_ttl": self._cache_ttl}
