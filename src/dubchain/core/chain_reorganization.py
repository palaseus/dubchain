"""
Robust Chain Reorganization System for DubChain.

This module implements comprehensive chain reorganization functionality including:
- Fork detection and handling
- Chain reorganization safety checks
- UTXO set updates during reorgs
- Transaction replay protection
- State rollback mechanisms
- Reorganization validation and recovery
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash
from .block import Block
from .blockchain import BlockchainState
from .transaction import UTXO, Transaction, TransactionType


class ReorganizationType(Enum):
    """Types of chain reorganizations."""

    SOFT_FORK = "soft_fork"
    HARD_FORK = "hard_fork"
    REORG = "reorganization"
    ROLLBACK = "rollback"


class ReorganizationStatus(Enum):
    """Status of reorganization operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ForkPoint:
    """Represents a fork point in the blockchain."""

    block_hash: Hash
    block_height: int
    timestamp: int
    difficulty: int
    total_difficulty: int
    fork_type: ReorganizationType
    confidence: float = 0.0  # 0.0 to 1.0

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ReorganizationPlan:
    """Plan for chain reorganization."""

    fork_point: ForkPoint
    old_chain: List[Block]
    new_chain: List[Block]
    blocks_to_remove: List[Block]
    blocks_to_add: List[Block]
    transactions_to_replay: List[Transaction]
    utxos_to_remove: Set[str]
    utxos_to_add: Dict[str, UTXO]
    estimated_time: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    rollback_plan: Optional["ReorganizationPlan"] = None

    def __post_init__(self):
        if self.risk_level not in ["low", "medium", "high", "critical"]:
            raise ValueError("Risk level must be one of: low, medium, high, critical")


@dataclass
class ReorganizationResult:
    """Result of chain reorganization operation."""

    status: ReorganizationStatus
    success: bool
    fork_point: Optional[ForkPoint] = None
    blocks_removed: int = 0
    blocks_added: int = 0
    transactions_replayed: int = 0
    utxos_updated: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    rollback_required: bool = False

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TransactionReplayProtection:
    """Transaction replay protection system."""

    def __init__(self, max_replay_window: int = 1000):
        self.max_replay_window = max_replay_window
        self._replay_cache: Dict[str, Set[str]] = defaultdict(
            set
        )  # address -> tx_hashes
        self._replay_history: deque = deque(maxlen=max_replay_window)

    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to replay protection."""
        tx_hash = transaction.get_hash().to_hex()

        # Get addresses involved in transaction
        addresses = set()
        for inp in transaction.inputs:
            if inp.public_key:
                addresses.add(inp.public_key.to_address())
        for output in transaction.outputs:
            addresses.add(output.recipient_address)

        # Check for replay attacks
        for address in addresses:
            if tx_hash in self._replay_cache[address]:
                return False  # Replay detected

        # Add to replay protection
        for address in addresses:
            self._replay_cache[address].add(tx_hash)

        self._replay_history.append((tx_hash, time.time()))
        return True

    def remove_transaction(self, transaction: Transaction) -> None:
        """Remove a transaction from replay protection."""
        tx_hash = transaction.get_hash().to_hex()

        # Get addresses involved in transaction
        addresses = set()
        for inp in transaction.inputs:
            if inp.public_key:
                addresses.add(inp.public_key.to_address())
        for output in transaction.outputs:
            addresses.add(output.recipient_address)

        # Remove from replay protection
        for address in addresses:
            self._replay_cache[address].discard(tx_hash)

    def is_replay(self, transaction: Transaction) -> bool:
        """Check if transaction is a replay."""
        tx_hash = transaction.get_hash().to_hex()

        # Get addresses involved in transaction
        addresses = set()
        for inp in transaction.inputs:
            if inp.public_key:
                addresses.add(inp.public_key.to_address())
        for output in transaction.outputs:
            addresses.add(output.recipient_address)

        # Check for replay
        for address in addresses:
            if tx_hash in self._replay_cache[address]:
                return True

        return False

    def cleanup_old_entries(self, max_age: float = 3600) -> int:
        """Clean up old replay protection entries."""
        current_time = time.time()
        removed_count = 0

        # Remove old entries from history
        while (
            self._replay_history and current_time - self._replay_history[0][1] > max_age
        ):
            old_tx_hash, _ = self._replay_history.popleft()

            # Remove from cache
            for address_set in self._replay_cache.values():
                if old_tx_hash in address_set:
                    address_set.remove(old_tx_hash)
                    removed_count += 1

        return removed_count


class ChainReorganizationManager:
    """Manages chain reorganizations and fork handling."""

    def __init__(
        self,
        max_reorg_depth: int = 100,
        min_confirmation_blocks: int = 6,
        reorg_timeout: float = 30.0,
        enable_rollback: bool = True,
    ):
        self.max_reorg_depth = max_reorg_depth
        self.min_confirmation_blocks = min_confirmation_blocks
        self.reorg_timeout = reorg_timeout
        self.enable_rollback = enable_rollback

        # State tracking
        self._active_reorgs: Dict[str, ReorganizationPlan] = {}
        self._reorg_history: deque = deque(maxlen=1000)
        self._fork_points: Dict[int, List[ForkPoint]] = defaultdict(list)

        # Replay protection
        self._replay_protection = TransactionReplayProtection()

        # Callbacks
        self._before_reorg_callback: Optional[
            Callable[[ReorganizationPlan], bool]
        ] = None
        self._after_reorg_callback: Optional[
            Callable[[ReorganizationResult], None]
        ] = None
        self._rollback_callback: Optional[Callable[[ReorganizationPlan], bool]] = None

        # Logging
        self._logger = logging.getLogger(__name__)

    def set_before_reorg_callback(
        self, callback: Callable[[ReorganizationPlan], bool]
    ) -> None:
        """Set callback to be called before reorganization."""
        self._before_reorg_callback = callback

    def set_after_reorg_callback(
        self, callback: Callable[[ReorganizationResult], None]
    ) -> None:
        """Set callback to be called after reorganization."""
        self._after_reorg_callback = callback

    def set_rollback_callback(
        self, callback: Callable[[ReorganizationPlan], bool]
    ) -> None:
        """Set callback to be called during rollback."""
        self._rollback_callback = callback

    def detect_fork(
        self, current_chain: List[Block], new_block: Block
    ) -> Optional[ForkPoint]:
        """
        Detect if a new block creates a fork.

        Args:
            current_chain: Current blockchain
            new_block: New block to check

        Returns:
            ForkPoint if fork detected, None otherwise
        """
        if not current_chain:
            return None

        # Check if new block's previous hash matches current chain
        current_tip = current_chain[-1]
        if new_block.header.previous_hash == current_tip.get_hash():
            return None  # No fork, block extends current chain

        # Find common ancestor
        common_ancestor = self._find_common_ancestor(current_chain, new_block)
        if not common_ancestor:
            return None  # No common ancestor found

        # Calculate fork statistics
        fork_depth = (
            current_tip.header.block_height - common_ancestor.header.block_height
        )
        if fork_depth > self.max_reorg_depth:
            return None  # Fork too deep

        # Calculate confidence based on difficulty and time
        confidence = self._calculate_fork_confidence(
            current_chain[common_ancestor.header.block_height :], new_block
        )

        # Determine fork type
        fork_type = self._determine_fork_type(common_ancestor, new_block)

        return ForkPoint(
            block_hash=common_ancestor.get_hash(),
            block_height=common_ancestor.header.block_height,
            timestamp=common_ancestor.header.timestamp,
            difficulty=common_ancestor.header.difficulty,
            total_difficulty=self._calculate_total_difficulty(
                current_chain[: common_ancestor.header.block_height + 1]
            ),
            fork_type=fork_type,
            confidence=confidence,
        )

    def plan_reorganization(
        self,
        fork_point: ForkPoint,
        current_chain: List[Block],
        new_chain: List[Block],
        current_utxos: Dict[str, UTXO],
    ) -> ReorganizationPlan:
        """
        Plan a chain reorganization.

        Args:
            fork_point: Point where the fork occurs
            current_chain: Current blockchain
            new_chain: New blockchain to switch to
            current_utxos: Current UTXO set

        Returns:
            ReorganizationPlan with detailed reorganization steps
        """
        # Find blocks to remove and add
        blocks_to_remove = current_chain[fork_point.block_height + 1 :]
        blocks_to_add = new_chain[fork_point.block_height + 1 :]

        # Calculate transactions to replay
        transactions_to_replay = []
        for block in blocks_to_remove:
            transactions_to_replay.extend(block.get_regular_transactions())

        # Calculate UTXO changes
        utxos_to_remove = set()
        utxos_to_add = {}

        # Remove UTXOs from blocks being removed
        for block in blocks_to_remove:
            for tx in block.get_regular_transactions():
                for utxo_key in tx.get_utxos_consumed():
                    utxos_to_remove.add(utxo_key)

        # Add UTXOs from new blocks
        for block in blocks_to_add:
            for tx in block.transactions:
                for utxo in tx.get_utxos_created():
                    utxo_key = utxo.get_key()
                    utxos_to_add[utxo_key] = utxo

        # Estimate execution time
        estimated_time = self._estimate_reorg_time(
            len(blocks_to_remove), len(blocks_to_add), len(transactions_to_replay)
        )

        # Calculate risk level
        risk_level = self._calculate_risk_level(
            fork_point,
            len(blocks_to_remove),
            len(blocks_to_add),
            len(transactions_to_replay),
        )

        # Create rollback plan if enabled
        rollback_plan = None
        if self.enable_rollback:
            rollback_plan = self._create_rollback_plan(
                fork_point, current_chain, new_chain, current_utxos
            )

        return ReorganizationPlan(
            fork_point=fork_point,
            old_chain=current_chain,
            new_chain=new_chain,
            blocks_to_remove=blocks_to_remove,
            blocks_to_add=blocks_to_add,
            transactions_to_replay=transactions_to_replay,
            utxos_to_remove=utxos_to_remove,
            utxos_to_add=utxos_to_add,
            estimated_time=estimated_time,
            risk_level=risk_level,
            rollback_plan=rollback_plan,
        )

    def execute_reorganization(
        self, plan: ReorganizationPlan, blockchain_state: BlockchainState
    ) -> ReorganizationResult:
        """
        Execute a chain reorganization.

        Args:
            plan: Reorganization plan
            blockchain_state: Current blockchain state

        Returns:
            ReorganizationResult with execution details
        """
        start_time = time.time()
        result = ReorganizationResult(
            status=ReorganizationStatus.IN_PROGRESS, success=False
        )

        try:
            # Call before reorganization callback
            if self._before_reorg_callback and not self._before_reorg_callback(plan):
                result.status = ReorganizationStatus.CANCELLED
                result.error_message = "Reorganization cancelled by callback"
                return result

            # Validate reorganization plan
            if not self._validate_reorg_plan(plan, blockchain_state):
                result.status = ReorganizationStatus.FAILED
                result.error_message = "Invalid reorganization plan"
                return result

            # Execute reorganization steps
            self._logger.info(
                f"Starting reorganization at height {plan.fork_point.block_height}"
            )

            # Step 1: Remove old blocks
            result.blocks_removed = self._remove_blocks(
                plan.blocks_to_remove, blockchain_state
            )

            # Step 2: Update UTXO set
            result.utxos_updated = self._update_utxo_set(
                plan.utxos_to_remove, plan.utxos_to_add, blockchain_state
            )

            # Step 3: Add new blocks
            result.blocks_added = self._add_blocks(plan.blocks_to_add, blockchain_state)

            # Step 4: Replay transactions
            result.transactions_replayed = self._replay_transactions(
                plan.transactions_to_replay, blockchain_state
            )

            # Step 5: Update replay protection
            self._update_replay_protection(plan)

            # Mark as completed
            result.status = ReorganizationStatus.COMPLETED
            result.success = True
            result.fork_point = plan.fork_point
            result.execution_time = time.time() - start_time

            # Record in history
            self._reorg_history.append((plan, result, time.time()))

            # Call after reorganization callback
            if self._after_reorg_callback:
                self._after_reorg_callback(result)

            self._logger.info(
                f"Reorganization completed in {result.execution_time:.2f}s"
            )

        except Exception as e:
            result.status = ReorganizationStatus.FAILED
            result.error_message = str(e)
            result.execution_time = time.time() - start_time

            self._logger.error(f"Reorganization failed: {e}")

            # Attempt rollback if enabled
            if self.enable_rollback and plan.rollback_plan:
                result.rollback_required = True
                self._attempt_rollback(plan.rollback_plan, blockchain_state)

        return result

    def rollback_reorganization(
        self, plan: ReorganizationPlan, blockchain_state: BlockchainState
    ) -> ReorganizationResult:
        """
        Rollback a chain reorganization.

        Args:
            plan: Reorganization plan to rollback
            blockchain_state: Current blockchain state

        Returns:
            ReorganizationResult with rollback details
        """
        if not plan.rollback_plan:
            return ReorganizationResult(
                status=ReorganizationStatus.FAILED,
                success=False,
                error_message="No rollback plan available",
            )

        return self.execute_reorganization(plan.rollback_plan, blockchain_state)

    def _find_common_ancestor(
        self, current_chain: List[Block], new_block: Block
    ) -> Optional[Block]:
        """Find common ancestor between current chain and new block."""
        # Check if new block's previous hash matches any block in current chain
        new_previous_hash = new_block.header.previous_hash

        # Look for the block with the hash that matches new_block's previous_hash
        for block in reversed(current_chain):
            if block.get_hash() == new_previous_hash:
                return block

        # If no match found, check if new block is at height 1 and genesis is the common ancestor
        if new_block.header.block_height == 1 and len(current_chain) > 0:
            genesis_block = current_chain[0]
            if genesis_block.header.block_height == 0:
                return genesis_block

        return None

    def _calculate_fork_confidence(
        self, old_chain_segment: List[Block], new_block: Block
    ) -> float:
        """Calculate confidence in the fork."""
        if not old_chain_segment:
            return 1.0

        # Calculate total difficulty of old chain segment
        old_difficulty = sum(
            2**block.header.difficulty for block in old_chain_segment
        )

        # Calculate difficulty of new block
        new_difficulty = 2**new_block.header.difficulty

        # Confidence based on difficulty ratio
        if old_difficulty == 0:
            return 1.0

        confidence = min(1.0, new_difficulty / old_difficulty)

        # Adjust for time (newer blocks have higher confidence)
        # Use a small time factor to avoid confidence > 1.0
        time_factor = min(
            0.1, (time.time() - new_block.header.timestamp) / 3600
        )  # Max 0.1
        confidence = confidence * (1.0 - time_factor)  # Time penalty

        return min(1.0, max(0.0, confidence))

    def _determine_fork_type(
        self, common_ancestor: Block, new_block: Block
    ) -> ReorganizationType:
        """Determine the type of fork."""
        # Simple heuristic based on block height difference
        height_diff = (
            new_block.header.block_height - common_ancestor.header.block_height
        )

        if height_diff == 1:
            return ReorganizationType.SOFT_FORK
        elif height_diff <= 10:
            return ReorganizationType.REORG
        else:
            return ReorganizationType.HARD_FORK

    def _calculate_total_difficulty(self, blocks: List[Block]) -> int:
        """Calculate total difficulty of a chain segment."""
        return sum(2**block.header.difficulty for block in blocks)

    def _estimate_reorg_time(
        self, blocks_to_remove: int, blocks_to_add: int, transactions_to_replay: int
    ) -> float:
        """Estimate time required for reorganization."""
        # Simple estimation based on operations
        block_time = 0.001  # 1ms per block
        tx_time = 0.0001  # 0.1ms per transaction

        return (
            blocks_to_remove + blocks_to_add
        ) * block_time + transactions_to_replay * tx_time

    def _calculate_risk_level(
        self,
        fork_point: ForkPoint,
        blocks_to_remove: int,
        blocks_to_add: int,
        transactions_to_replay: int,
    ) -> str:
        """Calculate risk level of reorganization."""
        risk_score = 0

        # Risk based on fork depth
        if blocks_to_remove > 50:
            risk_score += 3
        elif blocks_to_remove > 10:
            risk_score += 2
        elif blocks_to_remove > 1:
            risk_score += 1

        # Risk based on transaction count
        if transactions_to_replay > 1000:
            risk_score += 2
        elif transactions_to_replay > 100:
            risk_score += 1

        # Risk based on fork type
        if fork_point.fork_type == ReorganizationType.HARD_FORK:
            risk_score += 2
        elif fork_point.fork_type == ReorganizationType.REORG:
            risk_score += 1

        # Risk based on confidence
        if fork_point.confidence < 0.5:
            risk_score += 2
        elif fork_point.confidence < 0.8:
            risk_score += 1

        # Determine risk level
        if risk_score >= 6:
            return "critical"
        elif risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _create_rollback_plan(
        self,
        fork_point: ForkPoint,
        current_chain: List[Block],
        new_chain: List[Block],
        current_utxos: Dict[str, UTXO],
    ) -> ReorganizationPlan:
        """Create a rollback plan for the reorganization."""
        # Rollback plan is essentially the reverse of the reorganization
        return ReorganizationPlan(
            fork_point=fork_point,
            old_chain=new_chain,
            new_chain=current_chain,
            blocks_to_remove=new_chain[fork_point.block_height + 1 :],
            blocks_to_add=current_chain[fork_point.block_height + 1 :],
            transactions_to_replay=[],
            utxos_to_remove=set(),
            utxos_to_add={},
            estimated_time=0.0,
            risk_level="low",
        )

    def _validate_reorg_plan(
        self, plan: ReorganizationPlan, blockchain_state: BlockchainState
    ) -> bool:
        """Validate reorganization plan before execution."""
        # Check if fork point exists in current chain
        if plan.fork_point.block_height >= len(blockchain_state.blocks):
            return False

        # Check if we have enough confirmations (but allow genesis fork)
        if plan.fork_point.block_height > 0:
            if (
                len(blockchain_state.blocks) - plan.fork_point.block_height
                < self.min_confirmation_blocks
            ):
                return False

        # Check if reorganization is too deep
        if len(plan.blocks_to_remove) > self.max_reorg_depth:
            return False

        # Check if fork point block exists in current chain
        if plan.fork_point.block_height < len(blockchain_state.blocks):
            fork_block = blockchain_state.blocks[plan.fork_point.block_height]
            if fork_block.get_hash() != plan.fork_point.block_hash:
                return False

        return True

    def _remove_blocks(
        self, blocks_to_remove: List[Block], blockchain_state: BlockchainState
    ) -> int:
        """Remove blocks from the blockchain state."""
        removed_count = 0

        for block in reversed(blocks_to_remove):  # Remove in reverse order
            # Find block by hash instead of object identity
            block_hash = block.get_hash()
            for i, existing_block in enumerate(blockchain_state.blocks):
                if existing_block.get_hash() == block_hash:
                    blockchain_state.blocks.pop(i)
                    removed_count += 1
                    break

        # Update block height
        if blockchain_state.blocks:
            blockchain_state.block_height = blockchain_state.blocks[
                -1
            ].header.block_height
        else:
            blockchain_state.block_height = 0

        return removed_count

    def _add_blocks(
        self, blocks_to_add: List[Block], blockchain_state: BlockchainState
    ) -> int:
        """Add blocks to the blockchain state."""
        added_count = 0

        for block in blocks_to_add:
            blockchain_state.blocks.append(block)
            blockchain_state.block_height = block.header.block_height
            added_count += 1

        return added_count

    def _update_utxo_set(
        self,
        utxos_to_remove: Set[str],
        utxos_to_add: Dict[str, UTXO],
        blockchain_state: BlockchainState,
    ) -> int:
        """Update UTXO set during reorganization."""
        updated_count = 0

        # Remove UTXOs
        for utxo_key in utxos_to_remove:
            if utxo_key in blockchain_state.utxos:
                del blockchain_state.utxos[utxo_key]
                updated_count += 1

        # Add UTXOs
        for utxo_key, utxo in utxos_to_add.items():
            blockchain_state.utxos[utxo_key] = utxo
            updated_count += 1

        return updated_count

    def _replay_transactions(
        self,
        transactions_to_replay: List[Transaction],
        blockchain_state: BlockchainState,
    ) -> int:
        """Replay transactions during reorganization."""
        replayed_count = 0

        for tx in transactions_to_replay:
            # Check for replay protection
            if not self._replay_protection.is_replay(tx):
                # Add to pending transactions
                blockchain_state.pending_transactions.append(tx)
                self._replay_protection.add_transaction(tx)
                replayed_count += 1

        return replayed_count

    def _update_replay_protection(self, plan: ReorganizationPlan) -> None:
        """Update replay protection after reorganization."""
        # Remove transactions from removed blocks
        for block in plan.blocks_to_remove:
            for tx in block.get_regular_transactions():
                self._replay_protection.remove_transaction(tx)

        # Add transactions from new blocks
        for block in plan.blocks_to_add:
            for tx in block.get_regular_transactions():
                self._replay_protection.add_transaction(tx)

    def _attempt_rollback(
        self, rollback_plan: ReorganizationPlan, blockchain_state: BlockchainState
    ) -> None:
        """Attempt to rollback a failed reorganization."""
        try:
            if self._rollback_callback and self._rollback_callback(rollback_plan):
                self.execute_reorganization(rollback_plan, blockchain_state)
        except Exception as e:
            self._logger.error(f"Rollback failed: {e}")

    def get_reorganization_stats(self) -> Dict[str, Any]:
        """Get statistics about reorganizations."""
        return {
            "active_reorgs": len(self._active_reorgs),
            "total_reorgs": len(self._reorg_history),
            "replay_protection_entries": len(self._replay_protection._replay_history),
            "max_reorg_depth": self.max_reorg_depth,
            "min_confirmation_blocks": self.min_confirmation_blocks,
        }

    def cleanup_old_data(self, max_age: float = 86400) -> int:
        """Clean up old reorganization data."""
        current_time = time.time()
        removed_count = 0

        # Clean up replay protection
        removed_count += self._replay_protection.cleanup_old_entries(max_age)

        # Clean up old reorganization history
        while (
            self._reorg_history and current_time - self._reorg_history[0][2] > max_age
        ):
            self._reorg_history.popleft()
            removed_count += 1

        return removed_count
