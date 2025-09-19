"""
Main blockchain implementation for GodChain.

This module provides the core blockchain functionality including:
- Block validation and storage
- UTXO management
- Transaction processing
- Chain reorganization
- State management
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey
from ..errors.exceptions import ValidationError
from .block import Block, BlockHeader
from .consensus import ConsensusConfig, ConsensusEngine
from .transaction import UTXO, Transaction, TransactionType


@dataclass
class BlockchainState:
    """Current state of the blockchain."""

    blocks: List[Block] = field(default_factory=list)
    utxos: Dict[str, UTXO] = field(default_factory=dict)
    pending_transactions: List[Transaction] = field(default_factory=list)
    block_height: int = 0
    total_difficulty: int = 0
    last_block_time: int = 0

    def get_balance(self, address: str) -> int:
        """Get the balance of an address."""
        balance = 0
        for utxo in self.utxos.values():
            if utxo.recipient_address == address:
                balance += utxo.amount
        return balance

    def get_utxos_for_address(self, address: str) -> List[UTXO]:
        """Get all UTXOs for an address."""
        return [
            utxo for utxo in self.utxos.values() if utxo.recipient_address == address
        ]

    def add_utxo(self, utxo: UTXO) -> None:
        """Add a UTXO to the state."""
        key = utxo.get_key()
        self.utxos[key] = utxo

    def remove_utxo(self, utxo_key: str) -> None:
        """Remove a UTXO from the state."""
        if utxo_key in self.utxos:
            del self.utxos[utxo_key]

    def add_pending_transaction(self, transaction: Transaction) -> None:
        """Add a pending transaction."""
        if transaction not in self.pending_transactions:
            self.pending_transactions.append(transaction)

    def remove_pending_transaction(self, transaction: Transaction) -> None:
        """Remove a pending transaction."""
        if transaction in self.pending_transactions:
            self.pending_transactions.remove(transaction)

    def update_state(self, block: Block) -> None:
        """Update state with a new block."""
        self.block_height = block.header.block_height
        self.last_block_time = block.header.timestamp
        self.total_difficulty += block.header.difficulty

    def update_utxo_block_height(self, utxo_key: str, block_height: int) -> None:
        """Update the block height of a UTXO."""
        if utxo_key in self.utxos:
            utxo = self.utxos[utxo_key]
            new_utxo = UTXO(
                tx_hash=utxo.tx_hash,
                output_index=utxo.output_index,
                amount=utxo.amount,
                recipient_address=utxo.recipient_address,
                script_pubkey=utxo.script_pubkey,
                contract_address=utxo.contract_address,
                data=utxo.data,
                block_height=block_height,
            )
            self.utxos[utxo_key] = new_utxo


class Blockchain:
    """Main blockchain implementation."""

    def __init__(self, config: Optional[ConsensusConfig] = None):
        self.config = config or ConsensusConfig()
        self.consensus_engine = ConsensusEngine(self.config)
        self.state = BlockchainState()
        self._genesis_created = False
        self.genesis_block = None

    def create_genesis_block(
        self, coinbase_recipient: str, coinbase_amount: int = 1000000
    ) -> Block:
        """Create and add the genesis block."""
        if self._genesis_created:
            raise ValueError("Genesis block already created")

        genesis_block = Block.create_genesis_block(
            coinbase_recipient=coinbase_recipient,
            coinbase_amount=coinbase_amount,
            difficulty=0,  # Genesis block has no difficulty requirement
        )

        self.add_block(genesis_block)
        self._genesis_created = True

        return genesis_block

    def initialize_genesis(self, genesis_block: Block) -> None:
        """Initialize with a genesis block."""
        self.genesis_block = genesis_block
        self.state.update_state(genesis_block)
        self._genesis_created = True

    def add_block(self, block: Block) -> bool:
        """
        Add a block to the blockchain.

        Args:
            block: Block to add

        Returns:
            True if block was added successfully, False otherwise
        """
        try:
            # Validate the block
            if not self._validate_block(block):
                return False

            # Add the block
            self.state.blocks.append(block)
            self.state.block_height = block.header.block_height
            self.state.total_difficulty += 2**block.header.difficulty
            self.state.last_block_time = block.header.timestamp

            # Update UTXO set
            self._update_utxo_set(block)

            # Remove pending transactions that are now in the block
            self._remove_pending_transactions(block)

            return True

        except Exception as e:
            print(f"Error adding block: {e}")
            return False

    def _validate_block(self, block: Block) -> bool:
        """Validate a block before adding it to the chain."""
        # Check if genesis block
        if block.header.block_height == 0:
            if len(self.state.blocks) > 0:
                return False
            return block.is_valid({})

        # Check if we have previous blocks
        if not self.state.blocks:
            return False

        # Get previous block
        previous_block = self.state.blocks[-1]

        # Validate block
        return self.consensus_engine.validate_block(
            block, self.state.blocks, self.state.utxos
        )

    def _update_utxo_set(self, block: Block) -> None:
        """Update the UTXO set with transactions from a block."""
        # Remove consumed UTXOs
        for tx in block.get_regular_transactions():
            for utxo_key in tx.get_utxos_consumed():
                self.state.remove_utxo(utxo_key)

        # Add new UTXOs
        for tx in block.transactions:
            for utxo in tx.get_utxos_created():
                utxo_with_height = UTXO(
                    tx_hash=utxo.tx_hash,
                    output_index=utxo.output_index,
                    amount=utxo.amount,
                    recipient_address=utxo.recipient_address,
                    script_pubkey=utxo.script_pubkey,
                    contract_address=utxo.contract_address,
                    data=utxo.data,
                    block_height=block.header.block_height,
                )
                self.state.add_utxo(utxo_with_height)

    def _remove_pending_transactions(self, block: Block) -> None:
        """Remove transactions from pending list that are now in a block."""
        block_tx_hashes = {tx.get_hash() for tx in block.transactions}

        self.state.pending_transactions = [
            tx
            for tx in self.state.pending_transactions
            if tx.get_hash() not in block_tx_hashes
        ]

    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a transaction to the pending pool.

        Args:
            transaction: Transaction to add

        Returns:
            True if transaction was added successfully, False otherwise
        """
        try:
            # Validate transaction
            if not transaction.is_valid(self.state.utxos):
                return False

            # Check if transaction already exists
            tx_hash = transaction.get_hash()
            for pending_tx in self.state.pending_transactions:
                if pending_tx.get_hash() == tx_hash:
                    return False

            # Add to pending transactions
            self.state.pending_transactions.append(transaction)
            return True

        except Exception:
            return False

    def mine_block(
        self, miner_address: str, max_transactions: int = 1000
    ) -> Optional[Block]:
        """
        Mine a new block with pending transactions.

        Args:
            miner_address: Address to receive block reward
            max_transactions: Maximum number of transactions to include

        Returns:
            Mined block, or None if mining failed
        """
        try:
            if not self.state.blocks:
                raise ValueError("No blocks in chain. Create genesis block first.")

            # Get pending transactions
            pending_txs = self.state.pending_transactions[:max_transactions]

            # Create coinbase transaction
            block_reward = self._calculate_block_reward()
            coinbase_tx = Transaction.create_coinbase(
                recipient_address=miner_address,
                amount=block_reward,
                block_height=self.state.block_height + 1,
            )

            # Combine coinbase with pending transactions
            all_transactions = [coinbase_tx] + pending_txs

            # Mine the block
            previous_block = self.state.blocks[-1]
            mined_block = self.consensus_engine.mine_block(
                transactions=all_transactions,
                previous_block=previous_block,
                utxos=self.state.utxos,
            )

            if mined_block:
                # Add the mined block
                if self.add_block(mined_block):
                    return mined_block

            return None

        except Exception as e:
            print(f"Error mining block: {e}")
            return None

    def _calculate_block_reward(self) -> int:
        """Calculate the block reward for the current height."""
        # Simple halving every 210,000 blocks (like Bitcoin)
        halving_interval = 210000
        halvings = self.state.block_height // halving_interval

        if halvings >= 64:  # After 64 halvings, reward becomes 0
            return 0

        # Start with 50 coins, halve each time
        base_reward = 50 * (10**8)  # 50 coins in satoshis
        return base_reward // (2**halvings)

    def get_balance(self, address: str) -> int:
        """Get the balance of an address."""
        return self.state.get_balance(address)

    def get_utxos_for_address(self, address: str) -> List[UTXO]:
        """Get all UTXOs for an address."""
        return self.state.get_utxos_for_address(address)

    def create_transfer_transaction(
        self,
        sender_private_key: PrivateKey,
        recipient_address: str,
        amount: int,
        fee: int = 1000,
    ) -> Optional[Transaction]:
        """
        Create a transfer transaction.

        Args:
            sender_private_key: Private key of sender
            recipient_address: Address of recipient
            amount: Amount to transfer
            fee: Transaction fee

        Returns:
            Created transaction, or None if creation failed
        """
        try:
            sender_public_key = sender_private_key.get_public_key()
            sender_address = sender_public_key.to_address()

            # Get UTXOs for sender
            sender_utxos = self.get_utxos_for_address(sender_address)

            if not sender_utxos:
                return None

            # Create transaction
            transaction = Transaction.create_transfer(
                sender_private_key=sender_private_key,
                recipient_address=recipient_address,
                amount=amount,
                utxos=sender_utxos,
                fee=fee,
            )

            return transaction

        except Exception:
            return None

    def get_block_by_hash(self, block_hash: Hash) -> Optional[Block]:
        """Get a block by its hash."""
        for block in self.state.blocks:
            if block.get_hash() == block_hash:
                return block
        return None

    def get_block_by_height(self, height: int) -> Optional[Block]:
        """Get a block by its height."""
        if 0 <= height < len(self.state.blocks):
            return self.state.blocks[height]
        return None

    def get_transaction_by_hash(
        self, tx_hash: Hash
    ) -> Optional[Tuple[Block, Transaction]]:
        """Get a transaction by its hash."""
        for block in self.state.blocks:
            for tx in block.transactions:
                if tx.get_hash() == tx_hash:
                    return block, tx

        # Check pending transactions
        for tx in self.state.pending_transactions:
            if tx.get_hash() == tx_hash:
                return None, tx

        return None, None

    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the blockchain."""
        if not self.state.blocks:
            return {
                "block_count": 0,
                "block_height": 0,
                "total_difficulty": 0,
                "pending_transactions": len(self.state.pending_transactions),
                "utxo_count": len(self.state.utxos),
            }

        last_block = self.state.blocks[-1]
        consensus_info = self.consensus_engine.get_consensus_info(self.state.blocks)

        return {
            "block_count": len(self.state.blocks),
            "block_height": self.state.block_height,
            "total_difficulty": self.state.total_difficulty,
            "last_block_hash": last_block.get_hash().to_hex(),
            "last_block_time": last_block.header.timestamp,
            "pending_transactions": len(self.state.pending_transactions),
            "utxo_count": len(self.state.utxos),
            **consensus_info,
        }

    def validate_chain(self) -> bool:
        """Validate the entire blockchain."""
        try:
            for i, block in enumerate(self.state.blocks):
                # Check block structure
                if not block._verify_merkle_root():
                    return False

                # Check proof of work
                if not block.header.meets_difficulty():
                    return False

                # Check block height
                if block.header.block_height != i:
                    return False

                # Check previous hash
                if i > 0:
                    if (
                        block.header.previous_hash
                        != self.state.blocks[i - 1].get_hash()
                    ):
                        return False
                else:
                    if block.header.previous_hash != Hash.zero():
                        return False

            return True

        except Exception:
            return False

    def get_best_chain(self) -> List[Block]:
        """Get the best (longest) chain."""
        return self.state.blocks.copy()

    def get_pending_transactions(self) -> List[Transaction]:
        """Get all pending transactions."""
        return self.state.pending_transactions.copy()

    def clear_pending_transactions(self) -> None:
        """Clear all pending transactions."""
        self.state.pending_transactions.clear()

    def export_state(self) -> Dict[str, Any]:
        """Export the current blockchain state."""
        return {
            "blocks": [
                {
                    "header": {
                        "version": block.header.version,
                        "previous_hash": block.header.previous_hash.to_hex(),
                        "merkle_root": block.header.merkle_root.to_hex(),
                        "timestamp": block.header.timestamp,
                        "difficulty": block.header.difficulty,
                        "nonce": block.header.nonce,
                        "block_height": block.header.block_height,
                        "gas_limit": block.header.gas_limit,
                        "gas_used": block.header.gas_used,
                    },
                    "transactions": [
                        {
                            "hash": tx.get_hash().to_hex(),
                            "type": tx.transaction_type.value,
                            "inputs": len(tx.inputs),
                            "outputs": len(tx.outputs),
                        }
                        for tx in block.transactions
                    ],
                }
                for block in self.state.blocks
            ],
            "utxos": {
                key: {
                    "tx_hash": utxo.tx_hash.to_hex(),
                    "output_index": utxo.output_index,
                    "amount": utxo.amount,
                    "recipient_address": utxo.recipient_address,
                    "block_height": utxo.block_height,
                }
                for key, utxo in self.state.utxos.items()
            },
            "pending_transactions": [
                {
                    "hash": tx.get_hash().to_hex(),
                    "type": tx.transaction_type.value,
                    "inputs": len(tx.inputs),
                    "outputs": len(tx.outputs),
                }
                for tx in self.state.pending_transactions
            ],
            "block_height": self.state.block_height,
            "total_difficulty": self.state.total_difficulty,
        }

    def process_transaction(self, transaction: Transaction) -> bool:
        """Process a transaction."""
        if self.validate_transaction(transaction):
            self.state.add_pending_transaction(transaction)
            return True
        else:
            from ..errors.exceptions import ValidationError

            raise ValidationError("Transaction validation failed")

    def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate a transaction."""
        # Basic validation - check if transaction is valid
        return transaction is not None and hasattr(transaction, "transaction_id")

    def __str__(self) -> str:
        return f"Blockchain(height={self.state.block_height}, blocks={len(self.state.blocks)})"

    def __repr__(self) -> str:
        return f"Blockchain(height={self.state.block_height}, utxos={len(self.state.utxos)})"
