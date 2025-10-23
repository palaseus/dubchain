"""
Atomic swap implementation for DubChain.

This module provides atomic swap functionality for cross-chain transactions
including swap proposals, execution, and validation.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .bridge_types import (
    BridgeAsset,
    BridgeType,
    BridgeValidator,
    CrossChainTransaction,
)


class SwapStatus(Enum):
    """Atomic swap status."""
    
    PENDING = "pending"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class SwapProposal:
    """Atomic swap proposal."""

    proposal_id: str
    initiator: str
    counterparty: str
    source_chain: str
    target_chain: str
    source_asset: str
    target_asset: str
    source_amount: int
    target_amount: int
    secret_hash: str
    timeout: int
    status: str = "pending"  # pending, accepted, completed, expired, cancelled
    created_at: float = field(default_factory=time.time)
    accepted_at: Optional[float] = None
    completed_at: Optional[float] = None
    secret: Optional[str] = None
    initiator_tx_hash: Optional[str] = None
    counterparty_tx_hash: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if proposal is expired."""
        return time.time() - self.created_at > self.timeout

    def calculate_hash(self) -> str:
        """Calculate proposal hash."""
        data_string = f"{self.proposal_id}{self.initiator}{self.counterparty}{self.source_amount}{self.target_amount}{self.created_at}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "initiator": self.initiator,
            "counterparty": self.counterparty,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "source_asset": self.source_asset,
            "target_asset": self.target_asset,
            "source_amount": self.source_amount,
            "target_amount": self.target_amount,
            "secret_hash": self.secret_hash,
            "timeout": self.timeout,
            "status": self.status,
            "created_at": self.created_at,
            "accepted_at": self.accepted_at,
            "completed_at": self.completed_at,
            "secret": self.secret,
            "initiator_tx_hash": self.initiator_tx_hash,
            "counterparty_tx_hash": self.counterparty_tx_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwapProposal":
        """Create from dictionary."""
        return cls(
            proposal_id=data["proposal_id"],
            initiator=data["initiator"],
            counterparty=data["counterparty"],
            source_chain=data["source_chain"],
            target_chain=data["target_chain"],
            source_asset=data["source_asset"],
            target_asset=data["target_asset"],
            source_amount=data["source_amount"],
            target_amount=data["target_amount"],
            secret_hash=data["secret_hash"],
            timeout=data["timeout"],
            status=data["status"],
            created_at=data["created_at"],
            accepted_at=data.get("accepted_at"),
            completed_at=data.get("completed_at"),
            secret=data.get("secret"),
            initiator_tx_hash=data.get("initiator_tx_hash"),
            counterparty_tx_hash=data.get("counterparty_tx_hash"),
        )


@dataclass
class SwapExecution:
    """Atomic swap execution state."""

    proposal_id: str
    execution_phase: str = "initiated"  # initiated, locked, revealed, completed
    initiator_locked: bool = False
    counterparty_locked: bool = False
    secret_revealed: bool = False
    execution_started: float = field(default_factory=time.time)
    lock_timeout: int = 3600  # 1 hour
    reveal_timeout: int = 1800  # 30 minutes

    def is_lock_expired(self) -> bool:
        """Check if lock phase is expired."""
        return time.time() - self.execution_started > self.lock_timeout

    def is_reveal_expired(self) -> bool:
        """Check if reveal phase is expired."""
        return time.time() - self.execution_started > self.reveal_timeout

    def can_proceed_to_reveal(self) -> bool:
        """Check if can proceed to reveal phase."""
        return self.initiator_locked and self.counterparty_locked

    def can_complete(self) -> bool:
        """Check if swap can be completed."""
        return (
            self.secret_revealed and self.initiator_locked and self.counterparty_locked
        )


@dataclass
class SwapValidator:
    """Validates atomic swap operations."""

    validation_rules: Dict[str, Any] = field(default_factory=dict)
    validation_cache: Dict[str, bool] = field(default_factory=dict)

    def validate_proposal(self, proposal: SwapProposal) -> bool:
        """Validate swap proposal."""
        # Check basic requirements
        if proposal.source_amount <= 0 or proposal.target_amount <= 0:
            return False

        if proposal.source_chain == proposal.target_chain:
            return False

        if proposal.initiator == proposal.counterparty:
            return False

        # Check timeout
        if proposal.timeout <= 0:
            return False

        # Check secret hash format
        if not proposal.secret_hash or len(proposal.secret_hash) != 64:
            return False

        return True

    def validate_secret(self, secret: str, secret_hash: str) -> bool:
        """Validate secret against hash."""
        if not secret or not secret_hash:
            return False

        # Calculate hash of secret
        calculated_hash = hashlib.sha256(secret.encode()).hexdigest()
        return calculated_hash == secret_hash

    def validate_execution(
        self, execution: SwapExecution, proposal: SwapProposal
    ) -> bool:
        """Validate swap execution."""
        # Check if execution is not expired
        if execution.is_lock_expired():
            return False

        # Check if both parties have locked funds
        if not execution.initiator_locked or not execution.counterparty_locked:
            return False

        return True


class AtomicSwap:
    """Atomic swap implementation."""

    def __init__(self):
        """Initialize atomic swap system."""
        self.proposals: Dict[str, SwapProposal] = {}
        self.executions: Dict[str, SwapExecution] = {}
        self.validator = SwapValidator()
        self.swap_metrics = {
            "proposals_created": 0,
            "proposals_accepted": 0,
            "swaps_completed": 0,
            "swaps_failed": 0,
        }

    def create_proposal(
        self,
        initiator: str,
        counterparty: str,
        source_chain: str,
        target_chain: str,
        source_asset: str,
        target_asset: str,
        source_amount: int,
        target_amount: int,
        timeout: int = 3600,
    ) -> Optional[SwapProposal]:
        """Create atomic swap proposal."""
        # Generate secret and hash
        secret = secrets.token_hex(32)
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()

        # Create proposal
        proposal_id = self._generate_proposal_id()
        proposal = SwapProposal(
            proposal_id=proposal_id,
            initiator=initiator,
            counterparty=counterparty,
            source_chain=source_chain,
            target_chain=target_chain,
            source_asset=source_asset,
            target_asset=target_asset,
            source_amount=source_amount,
            target_amount=target_amount,
            secret_hash=secret_hash,
            timeout=timeout,
            secret=secret,  # Store secret for initiator
        )

        # Validate proposal
        if not self.validator.validate_proposal(proposal):
            return None

        self.proposals[proposal_id] = proposal
        self.swap_metrics["proposals_created"] += 1

        return proposal

    def accept_proposal(
        self, proposal_id: str, counterparty: str
    ) -> Optional[SwapExecution]:
        """Accept atomic swap proposal."""
        if proposal_id not in self.proposals:
            return None

        proposal = self.proposals[proposal_id]

        # Check if proposal is still valid
        if proposal.status != "pending" or proposal.is_expired():
            return None

        # Check counterparty
        if proposal.counterparty != counterparty:
            return None

        # Update proposal status
        proposal.status = "accepted"
        proposal.accepted_at = time.time()

        # Create execution
        execution = SwapExecution(proposal_id=proposal_id)
        self.executions[proposal_id] = execution

        self.swap_metrics["proposals_accepted"] += 1

        return execution

    def lock_funds(self, proposal_id: str, party: str, tx_hash: str) -> bool:
        """Lock funds for atomic swap."""
        if proposal_id not in self.proposals or proposal_id not in self.executions:
            return False

        proposal = self.proposals[proposal_id]
        execution = self.executions[proposal_id]

        # Check if proposal is accepted
        if proposal.status != "accepted":
            return False

        # Check if execution is not expired
        if execution.is_lock_expired():
            proposal.status = "expired"
            return False

        # Update execution based on party
        if party == proposal.initiator:
            execution.initiator_locked = True
            proposal.initiator_tx_hash = tx_hash
        elif party == proposal.counterparty:
            execution.counterparty_locked = True
            proposal.counterparty_tx_hash = tx_hash
        else:
            return False

        # Check if both parties have locked funds
        if execution.can_proceed_to_reveal():
            execution.execution_phase = "locked"

        return True

    def reveal_secret(self, proposal_id: str, secret: str) -> bool:
        """Reveal secret for atomic swap."""
        if proposal_id not in self.proposals or proposal_id not in self.executions:
            return False

        proposal = self.proposals[proposal_id]
        execution = self.executions[proposal_id]

        # Check if execution is in locked phase
        if execution.execution_phase != "locked":
            return False

        # Check if reveal is not expired
        if execution.is_reveal_expired():
            proposal.status = "expired"
            return False

        # Validate secret
        if not self.validator.validate_secret(secret, proposal.secret_hash):
            return False

        # Update execution
        execution.secret_revealed = True
        execution.execution_phase = "revealed"

        return True

    def complete_swap(self, proposal_id: str) -> bool:
        """Complete atomic swap."""
        if proposal_id not in self.proposals or proposal_id not in self.executions:
            return False

        proposal = self.proposals[proposal_id]
        execution = self.executions[proposal_id]

        # Check if swap can be completed
        if not execution.can_complete():
            return False

        # Update proposal status
        proposal.status = "completed"
        proposal.completed_at = time.time()

        # Update execution
        execution.execution_phase = "completed"

        self.swap_metrics["swaps_completed"] += 1

        return True

    def cancel_proposal(self, proposal_id: str, party: str) -> bool:
        """Cancel atomic swap proposal."""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        # Check if proposal can be cancelled
        if proposal.status not in ["pending", "accepted"]:
            return False

        # Check if party is authorized to cancel
        if party not in [proposal.initiator, proposal.counterparty]:
            return False

        # Update proposal status
        proposal.status = "cancelled"

        return True

    def get_proposal(self, proposal_id: str) -> Optional[SwapProposal]:
        """Get swap proposal by ID."""
        return self.proposals.get(proposal_id)

    def get_execution(self, proposal_id: str) -> Optional[SwapExecution]:
        """Get swap execution by proposal ID."""
        return self.executions.get(proposal_id)

    def get_proposals_by_party(self, party: str) -> List[SwapProposal]:
        """Get proposals by party."""
        return [
            p
            for p in self.proposals.values()
            if p.initiator == party or p.counterparty == party
        ]

    def get_proposals_by_status(self, status: str) -> List[SwapProposal]:
        """Get proposals by status."""
        return [p for p in self.proposals.values() if p.status == status]

    def cleanup_expired_proposals(self) -> int:
        """Clean up expired proposals."""
        expired_count = 0
        current_time = time.time()

        # Collect expired proposal IDs to remove
        expired_proposal_ids = []

        for proposal_id, proposal in self.proposals.items():
            if proposal.is_expired() and proposal.status in ["pending", "accepted"]:
                proposal.status = "expired"
                expired_proposal_ids.append(proposal_id)
                expired_count += 1
                self.swap_metrics["swaps_failed"] += 1

        # Remove expired proposals
        for proposal_id in expired_proposal_ids:
            del self.proposals[proposal_id]
            # Also remove associated execution if exists
            if proposal_id in self.executions:
                del self.executions[proposal_id]

        return expired_count

    def get_swap_metrics(self) -> Dict[str, Any]:
        """Get atomic swap metrics."""
        return {
            "total_proposals": len(self.proposals),
            "active_proposals": len(self.get_proposals_by_status("pending"))
            + len(self.get_proposals_by_status("accepted")),
            "completed_swaps": len(self.get_proposals_by_status("completed")),
            "failed_swaps": len(self.get_proposals_by_status("expired"))
            + len(self.get_proposals_by_status("cancelled")),
            "metrics": self.swap_metrics,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get atomic swap statistics (alias for get_swap_metrics)."""
        return self.get_swap_metrics()

    def _generate_proposal_id(self) -> str:
        """Generate unique proposal ID."""
        timestamp = str(int(time.time() * 1000))
        random_data = secrets.token_hex(4)
        return f"swap_{timestamp}_{random_data}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposals": {k: v.to_dict() for k, v in self.proposals.items()},
            "executions": {
                k: {
                    "proposal_id": v.proposal_id,
                    "execution_phase": v.execution_phase,
                    "initiator_locked": v.initiator_locked,
                    "counterparty_locked": v.counterparty_locked,
                    "secret_revealed": v.secret_revealed,
                    "execution_started": v.execution_started,
                    "lock_timeout": v.lock_timeout,
                    "reveal_timeout": v.reveal_timeout,
                }
                for k, v in self.executions.items()
            },
            "swap_metrics": self.swap_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicSwap":
        """Create from dictionary."""
        atomic_swap = cls()

        # Restore proposals
        atomic_swap.proposals = {
            k: SwapProposal.from_dict(v) for k, v in data["proposals"].items()
        }

        # Restore executions
        atomic_swap.executions = {
            k: SwapExecution(
                proposal_id=v["proposal_id"],
                execution_phase=v["execution_phase"],
                initiator_locked=v["initiator_locked"],
                counterparty_locked=v["counterparty_locked"],
                secret_revealed=v["secret_revealed"],
                execution_started=v["execution_started"],
                lock_timeout=v["lock_timeout"],
                reveal_timeout=v["reveal_timeout"],
            )
            for k, v in data["executions"].items()
        }

        # Restore metrics
        atomic_swap.swap_metrics = data["swap_metrics"]

        return atomic_swap


class AtomicSwapManager:
    """Manager for atomic swap operations."""
    
    def __init__(self):
        """Initialize atomic swap manager."""
        self.swaps: Dict[str, AtomicSwap] = {}
        self.proposals: Dict[str, SwapProposal] = {}
    
    def create_swap(self, swap_id: str) -> AtomicSwap:
        """Create a new atomic swap."""
        swap = AtomicSwap(swap_id)
        self.swaps[swap_id] = swap
        return swap
    
    def get_swap(self, swap_id: str) -> Optional[AtomicSwap]:
        """Get an atomic swap by ID."""
        return self.swaps.get(swap_id)
    
    def create_proposal(
        self,
        proposal_id: str,
        initiator: str,
        counterparty: str,
        source_chain: str,
        target_chain: str,
        source_asset: str,
        target_asset: str,
        source_amount: int,
        target_amount: int,
        timeout: int = 3600
    ) -> SwapProposal:
        """Create a new swap proposal."""
        secret = secrets.token_hex(32)
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()
        
        proposal = SwapProposal(
            proposal_id=proposal_id,
            initiator=initiator,
            counterparty=counterparty,
            source_chain=source_chain,
            target_chain=target_chain,
            source_asset=source_asset,
            target_asset=target_asset,
            source_amount=source_amount,
            target_amount=target_amount,
            secret_hash=secret_hash,
            timeout=timeout,
            secret=secret
        )
        
        self.proposals[proposal_id] = proposal
        return proposal
    
    def get_proposal(self, proposal_id: str) -> Optional[SwapProposal]:
        """Get a proposal by ID."""
        return self.proposals.get(proposal_id)
    
    def accept_proposal(self, proposal_id: str) -> bool:
        """Accept a swap proposal."""
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != SwapStatus.PENDING.value:
            return False
        
        proposal.status = SwapStatus.ACCEPTED.value
        proposal.accepted_at = time.time()
        return True
    
    def complete_proposal(self, proposal_id: str) -> bool:
        """Complete a swap proposal."""
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != SwapStatus.ACCEPTED.value:
            return False
        
        proposal.status = SwapStatus.COMPLETED.value
        proposal.completed_at = time.time()
        return True
    
    def cancel_proposal(self, proposal_id: str) -> bool:
        """Cancel a swap proposal."""
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status not in [SwapStatus.PENDING.value, SwapStatus.ACCEPTED.value]:
            return False
        
        proposal.status = SwapStatus.CANCELLED.value
        return True
    
    def get_all_proposals(self) -> List[SwapProposal]:
        """Get all proposals."""
        return list(self.proposals.values())
    
    def get_proposals_by_status(self, status: SwapStatus) -> List[SwapProposal]:
        """Get proposals by status."""
        return [p for p in self.proposals.values() if p.status == status.value]
