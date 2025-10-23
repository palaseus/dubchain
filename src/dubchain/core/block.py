"""
Block and BlockHeader implementation for GodChain.

Implements the core block structure with advanced features like
merkle trees, difficulty adjustment, and block validation.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.merkle import MerkleTree
from .transaction import Transaction


@dataclass(frozen=True)
class BlockHeader:
    """Header of a blockchain block."""

    version: int = 1
    previous_hash: Hash = field(default_factory=Hash.zero)
    merkle_root: Hash = field(default_factory=Hash.zero)
    timestamp: int = field(default_factory=lambda: int(time.time()))
    difficulty: int = 1
    nonce: int = 0
    block_height: int = 0
    gas_limit: int = 10000000  # For smart contracts
    gas_used: int = 0
    extra_data: Optional[bytes] = None

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("Version must be non-negative")
        if self.difficulty < 0:
            raise ValueError("Difficulty must be non-negative")
        if self.nonce < 0:
            raise ValueError("Nonce must be non-negative")
        if self.block_height < 0:
            raise ValueError("Block height must be non-negative")
        if self.gas_limit < 0 or self.gas_used < 0:
            raise ValueError("Gas limit and used must be non-negative")
        if self.gas_used > self.gas_limit:
            raise ValueError("Gas used cannot exceed gas limit")

    def get_hash(self) -> Hash:
        """Get the hash of this block header."""
        return SHA256Hasher.double_hash(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize block header to bytes."""
        data = (
            self.version.to_bytes(4, byteorder="big")
            + self.previous_hash.value
            + self.merkle_root.value
            + self.timestamp.to_bytes(8, byteorder="big")
            + self.difficulty.to_bytes(4, byteorder="big")
            + self.nonce.to_bytes(8, byteorder="big")
            + self.block_height.to_bytes(8, byteorder="big")
            + self.gas_limit.to_bytes(8, byteorder="big")
            + self.gas_used.to_bytes(8, byteorder="big")
        )

        if self.extra_data:
            data += len(self.extra_data).to_bytes(4, byteorder="big")
            data += self.extra_data

        return data

    def meets_difficulty(self) -> bool:
        """Check if the block header meets the difficulty requirement."""
        block_hash = self.get_hash()
        return SHA256Hasher.verify_proof_of_work(block_hash, self.difficulty)

    def get_difficulty_target(self) -> Hash:
        """Get the target hash for this difficulty."""
        return SHA256Hasher.calculate_difficulty_target(self.difficulty)

    def with_nonce(self, nonce: int) -> "BlockHeader":
        """Create a new block header with a different nonce."""
        return BlockHeader(
            version=self.version,
            previous_hash=self.previous_hash,
            merkle_root=self.merkle_root,
            timestamp=self.timestamp,
            difficulty=self.difficulty,
            nonce=nonce,
            block_height=self.block_height,
            gas_limit=self.gas_limit,
            gas_used=self.gas_used,
            extra_data=self.extra_data,
        )

    def with_merkle_root(self, merkle_root: Hash) -> "BlockHeader":
        """Create a new block header with a different merkle root."""
        return BlockHeader(
            version=self.version,
            previous_hash=self.previous_hash,
            merkle_root=merkle_root,
            timestamp=self.timestamp,
            difficulty=self.difficulty,
            nonce=self.nonce,
            block_height=self.block_height,
            gas_limit=self.gas_limit,
            gas_used=self.gas_used,
            extra_data=self.extra_data,
        )

    def __str__(self) -> str:
        return f"BlockHeader(height={self.block_height}, hash={self.get_hash().to_hex()[:16]}...)"

    def __repr__(self) -> str:
        return f"BlockHeader(height={self.block_height}, difficulty={self.difficulty}, nonce={self.nonce})"


@dataclass(frozen=True)
class Block:
    """A blockchain block containing transactions."""

    header: BlockHeader
    transactions: List[Transaction]

    def __post_init__(self) -> None:
        if not self.transactions:
            raise ValueError("Block must contain at least one transaction")

        # Verify that the first transaction is coinbase
        if self.transactions[0].transaction_type.value != "coinbase":
            raise ValueError("First transaction must be coinbase")

        # Verify merkle root matches transactions
        if not self._verify_merkle_root():
            raise ValueError("Merkle root does not match transactions")

    def _verify_merkle_root(self) -> bool:
        """Verify that the merkle root matches the transactions."""
        if not self.transactions:
            return False

        # Create merkle tree from transaction hashes
        tx_hashes = [tx.get_hash().value for tx in self.transactions]
        merkle_tree = MerkleTree(tx_hashes)

        return merkle_tree.get_root() == self.header.merkle_root

    def get_hash(self) -> Hash:
        """Get the hash of this block."""
        return self.header.get_hash()

    def get_merkle_tree(self) -> MerkleTree:
        """Get the merkle tree of this block's transactions."""
        tx_hashes = [tx.get_hash().value for tx in self.transactions]
        return MerkleTree(tx_hashes)

    def get_transaction_proof(self, transaction: Transaction) -> Optional[Any]:
        """Get a merkle proof for a transaction in this block."""
        merkle_tree = self.get_merkle_tree()
        return merkle_tree.get_proof(transaction.get_hash().value)

    def verify_transaction_proof(self, transaction: Transaction, proof: Any) -> bool:
        """Verify a merkle proof for a transaction."""
        merkle_tree = self.get_merkle_tree()
        return merkle_tree.verify_proof(proof)

    def get_coinbase_transaction(self) -> Transaction:
        """Get the coinbase transaction of this block."""
        return self.transactions[0]

    def get_regular_transactions(self) -> List[Transaction]:
        """Get all non-coinbase transactions."""
        return self.transactions[1:]

    def get_total_transaction_fees(self, utxos: Dict[str, Any]) -> int:
        """Get the total transaction fees in this block."""
        total_fees = 0
        for tx in self.get_regular_transactions():
            total_fees += tx.get_fee(utxos)
        return total_fees

    def get_total_gas_used(self) -> int:
        """Get the total gas used by all transactions."""
        return sum(tx.gas_limit for tx in self.transactions)

    def is_valid(
        self, utxos: Dict[str, Any], previous_block: Optional["Block"] = None
    ) -> bool:
        """Check if this block is valid."""
        try:
            # Check basic structure
            if not self.transactions:
                return False

            # Check that first transaction is coinbase
            if self.transactions[0].transaction_type.value != "coinbase":
                return False

            # Check merkle root
            if not self._verify_merkle_root():
                return False

            # Check that block meets difficulty
            if not self.header.meets_difficulty():
                return False

            # Check timestamp (not too far in future, not too old)
            current_time = int(time.time())
            if self.header.timestamp > current_time + 3600:  # 1 hour in future
                return False

            if (
                previous_block
                and self.header.timestamp <= previous_block.header.timestamp
            ):
                return False

            # Check block height
            if previous_block:
                expected_height = previous_block.header.block_height + 1
                if self.header.block_height != expected_height:
                    return False
            else:
                if self.header.block_height != 0:
                    return False

            # Check previous hash
            if previous_block:
                if self.header.previous_hash != previous_block.get_hash():
                    return False
            else:
                if self.header.previous_hash != Hash.zero():
                    return False

            # Validate all transactions
            for tx in self.transactions:
                if not tx.is_valid(utxos):
                    return False

            # Check gas usage
            if self.header.gas_used != self.get_total_gas_used():
                return False

            return True

        except Exception:
            return False

    def to_bytes(self) -> bytes:
        """Serialize block to bytes."""
        data = self.header.to_bytes()

        # Add transactions
        data += len(self.transactions).to_bytes(4, byteorder="big")
        for tx in self.transactions:
            tx_bytes = tx.to_bytes()
            data += len(tx_bytes).to_bytes(4, byteorder="big")
            data += tx_bytes

        return data

    @classmethod
    def create_genesis_block(
        cls,
        coinbase_recipient: str,
        coinbase_amount: int = 1000000,
        difficulty: int = 1,
    ) -> "Block":
        """Create the genesis block."""
        # Create coinbase transaction
        coinbase_tx = Transaction.create_coinbase(
            recipient_address=coinbase_recipient, amount=coinbase_amount, block_height=0
        )

        # Create merkle tree
        merkle_tree = MerkleTree([coinbase_tx.get_hash().value])

        # Calculate gas used
        gas_used = sum(tx.gas_limit for tx in [coinbase_tx])

        # Create block header
        header = BlockHeader(
            version=1,
            previous_hash=Hash.zero(),
            merkle_root=merkle_tree.get_root(),
            timestamp=int(time.time()),
            difficulty=difficulty,
            nonce=0,
            block_height=0,
            gas_limit=10000000,
            gas_used=gas_used,
        )

        return cls(header=header, transactions=[coinbase_tx])

    @classmethod
    def create_block(
        cls,
        transactions: List[Transaction],
        previous_block: "Block",
        difficulty: int,
        gas_limit: int = 10000000,
        extra_data: Optional[bytes] = None,
    ) -> "Block":
        """Create a new block."""
        if not transactions:
            raise ValueError("Block must contain at least one transaction")

        # Create merkle tree
        tx_hashes = [tx.get_hash().value for tx in transactions]
        merkle_tree = MerkleTree(tx_hashes)

        # Calculate gas used
        gas_used = sum(tx.gas_limit for tx in transactions)

        # Create block header with timestamp at least 1 second after previous block
        current_time = int(time.time())
        min_timestamp = previous_block.header.timestamp + 1
        block_timestamp = max(current_time, min_timestamp)

        header = BlockHeader(
            version=1,
            previous_hash=previous_block.get_hash(),
            merkle_root=merkle_tree.get_root(),
            timestamp=block_timestamp,
            difficulty=difficulty,
            nonce=0,
            block_height=previous_block.header.block_height + 1,
            gas_limit=gas_limit,
            gas_used=gas_used,
            extra_data=extra_data,
        )

        return cls(header=header, transactions=transactions)

    def __str__(self) -> str:
        return f"Block(height={self.header.block_height}, hash={self.get_hash().to_hex()[:16]}...)"

    def __repr__(self) -> str:
        return f"Block(height={self.header.block_height}, tx_count={len(self.transactions)})"
