"""
Consensus mechanisms for GodChain.

Implements Proof of Work and other consensus algorithms with
sophisticated difficulty adjustment and validation.
"""

import logging

logger = logging.getLogger(__name__)
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..crypto.hashing import Hash, SHA256Hasher
from .block import Block, BlockHeader
from .transaction import Transaction


@dataclass
class ConsensusConfig:
    """Configuration for consensus mechanisms."""

    # Proof of Work settings
    target_block_time: int = 10  # seconds
    difficulty_adjustment_interval: int = 2016  # blocks
    max_difficulty_change: float = 4.0  # maximum factor change
    min_difficulty: int = 1
    max_difficulty: int = 256

    # Block validation
    max_block_size: int = 1000000  # bytes
    max_transactions_per_block: int = 10000
    max_gas_per_block: int = 10000000

    # Time validation
    max_future_time: int = 3600  # seconds
    min_block_interval: int = 1  # seconds


class ProofOfWork:
    """Proof of Work consensus mechanism."""

    def __init__(self, config: ConsensusConfig):
        self.config = config

    def mine_block(
        self, block_header: BlockHeader, max_nonce: int = 2**32 - 1
    ) -> Optional[BlockHeader]:
        """
        Mine a block by finding a valid nonce.

        Args:
            block_header: Block header to mine
            max_nonce: Maximum nonce to try

        Returns:
            Block header with valid nonce, or None if mining failed
        """
        start_time = time.time()

        for nonce in range(max_nonce + 1):
            # Create header with current nonce
            current_header = block_header.with_nonce(nonce)

            # Check if it meets difficulty
            if current_header.meets_difficulty():
                mining_time = time.time() - start_time
                logger.info(f"Mined block in {mining_time:.2f} seconds with nonce {nonce}")
                return current_header

            # Check for timeout (optional)
            if time.time() - start_time > 300:  # 5 minutes timeout
                break

        return None

    def verify_block(self, block: Block) -> bool:
        """Verify that a block meets the proof of work requirements."""
        return block.header.meets_difficulty()

    def calculate_difficulty(
        self, blocks: List[Block], target_block_time: Optional[int] = None
    ) -> int:
        """
        Calculate the difficulty for the next block.

        Args:
            blocks: List of recent blocks
            target_block_time: Target time between blocks in seconds

        Returns:
            Calculated difficulty
        """
        if target_block_time is None:
            target_block_time = self.config.target_block_time

        if len(blocks) < 2:
            return self.config.min_difficulty

        # Get the last difficulty adjustment interval blocks
        adjustment_blocks = blocks[-self.config.difficulty_adjustment_interval :]

        if len(adjustment_blocks) < 2:
            return blocks[-1].header.difficulty

        # Calculate time span
        first_block = adjustment_blocks[0]
        last_block = adjustment_blocks[-1]
        time_span = last_block.header.timestamp - first_block.header.timestamp

        if time_span <= 0:
            return blocks[-1].header.difficulty

        # Calculate expected time span
        expected_time_span = (len(adjustment_blocks) - 1) * target_block_time

        # Calculate difficulty adjustment
        difficulty_adjustment = expected_time_span / time_span

        # Apply limits
        difficulty_adjustment = max(
            1.0 / self.config.max_difficulty_change,
            min(difficulty_adjustment, self.config.max_difficulty_change),
        )

        # Calculate new difficulty
        old_difficulty = blocks[-1].header.difficulty
        new_difficulty = int(old_difficulty * difficulty_adjustment)

        # Apply bounds
        new_difficulty = max(self.config.min_difficulty, new_difficulty)
        new_difficulty = min(self.config.max_difficulty, new_difficulty)

        return new_difficulty

    def estimate_mining_time(self, difficulty: int, hashrate: float) -> float:
        """
        Estimate the time to mine a block with given difficulty and hashrate.

        Args:
            difficulty: Current difficulty
            hashrate: Hash rate in hashes per second

        Returns:
            Estimated time in seconds
        """
        if hashrate <= 0:
            return float("inf")

        # Calculate target hash rate needed
        target_hashrate = 2**difficulty

        # Calculate time
        time_seconds = target_hashrate / hashrate

        return time_seconds

    def get_difficulty_target(self, difficulty: int) -> Hash:
        """Get the target hash for a given difficulty."""
        return SHA256Hasher.calculate_difficulty_target(difficulty)


class ConsensusEngine:
    """Main consensus engine that coordinates different consensus mechanisms."""

    def __init__(self, config: ConsensusConfig):
        self.config = config
        self.proof_of_work = ProofOfWork(config)

    def validate_block(
        self, block: Block, previous_blocks: List[Block], utxos: Dict[str, Any]
    ) -> bool:
        """
        Validate a block according to consensus rules.

        Args:
            block: Block to validate
            previous_blocks: List of previous blocks
            utxos: Current UTXO set

        Returns:
            True if block is valid, False otherwise
        """
        try:
            # Basic block validation
            if not self._validate_block_structure(block):
                return False

            # Proof of work validation
            if not self.proof_of_work.verify_block(block):
                return False

            # Time validation
            if not self._validate_block_timing(block, previous_blocks):
                return False

            # Transaction validation
            if not self._validate_block_transactions(block, utxos):
                return False

            # Size validation
            if not self._validate_block_size(block):
                return False

            return True

        except Exception:
            return False

    def _validate_block_structure(self, block: Block) -> bool:
        """Validate basic block structure."""
        # Check that block has transactions
        if not block.transactions:
            return False

        # Check that first transaction is coinbase
        if block.transactions[0].transaction_type.value != "coinbase":
            return False

        # Check merkle root
        if not block._verify_merkle_root():
            return False

        return True

    def _validate_block_timing(
        self, block: Block, previous_blocks: List[Block]
    ) -> bool:
        """Validate block timing."""
        current_time = int(time.time())

        # Check not too far in future
        if block.header.timestamp > current_time + self.config.max_future_time:
            return False

        # Check not too old
        if previous_blocks:
            last_block = previous_blocks[-1]
            if block.header.timestamp <= last_block.header.timestamp:
                return False

            # Check minimum block interval
            time_diff = block.header.timestamp - last_block.header.timestamp
            if time_diff < self.config.min_block_interval:
                return False

        return True

    def _validate_block_transactions(self, block: Block, utxos: Dict[str, Any]) -> bool:
        """Validate all transactions in the block."""
        for tx in block.transactions:
            if not tx.is_valid(utxos):
                return False

        return True

    def _validate_block_size(self, block: Block) -> bool:
        """Validate block size limits."""
        block_size = len(block.to_bytes())
        if block_size > self.config.max_block_size:
            return False

        if len(block.transactions) > self.config.max_transactions_per_block:
            return False

        if block.header.gas_used > self.config.max_gas_per_block:
            return False

        return True

    def calculate_next_difficulty(self, blocks: List[Block]) -> int:
        """Calculate difficulty for the next block."""
        return self.proof_of_work.calculate_difficulty(blocks)

    def mine_block(
        self,
        transactions: List[Transaction],
        previous_block: Block,
        utxos: Dict[str, Any],
    ) -> Optional[Block]:
        """
        Mine a new block with the given transactions.

        Args:
            transactions: Transactions to include in the block
            previous_block: Previous block in the chain
            utxos: Current UTXO set

        Returns:
            Mined block, or None if mining failed
        """
        try:
            # Calculate difficulty
            difficulty = self.calculate_next_difficulty([previous_block])

            # Create block
            block = Block.create_block(
                transactions=transactions,
                previous_block=previous_block,
                difficulty=difficulty,
                gas_limit=self.config.max_gas_per_block,
            )

            # Mine the block
            mined_header = self.proof_of_work.mine_block(block.header)
            if mined_header is None:
                return None

            # Create final block with mined header
            final_block = Block(header=mined_header, transactions=block.transactions)

            # Validate the mined block
            if not self.validate_block(final_block, [previous_block], utxos):
                return None

            return final_block

        except Exception:
            return None

    def get_consensus_info(self, blocks: List[Block]) -> Dict[str, Any]:
        """Get information about the current consensus state."""
        if not blocks:
            return {}

        last_block = blocks[-1]

        # Calculate average block time
        if len(blocks) >= 2:
            time_span = last_block.header.timestamp - blocks[0].header.timestamp
            block_count = len(blocks) - 1
            avg_block_time = time_span / block_count if block_count > 0 else 0
        else:
            avg_block_time = 0

        # Calculate hash rate estimate
        if len(blocks) >= 2:
            target_hashrate = 2**last_block.header.difficulty
            estimated_hashrate = target_hashrate / self.config.target_block_time
        else:
            estimated_hashrate = 0

        return {
            "current_difficulty": last_block.header.difficulty,
            "block_height": last_block.header.block_height,
            "average_block_time": avg_block_time,
            "estimated_hashrate": estimated_hashrate,
            "target_block_time": self.config.target_block_time,
            "difficulty_adjustment_interval": self.config.difficulty_adjustment_interval,
        }
