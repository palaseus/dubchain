"""
Treasury management for governance system.

This module handles treasury operations including spending proposals,
multisig controls, and treasury security measures.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.signatures import PrivateKey, PublicKey
from ..errors.exceptions import ValidationError, GovernanceError
from .core import Proposal, ProposalType, ProposalStatus


class TreasuryOperationType(Enum):
    """Types of treasury operations."""
    
    SPENDING = "spending"
    TRANSFER = "transfer"
    INVESTMENT = "investment"
    EMERGENCY_WITHDRAWAL = "emergency_withdrawal"
    BURN = "burn"


class TreasuryStatus(Enum):
    """Status of treasury operations."""
    
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class TreasuryProposal:
    """A treasury spending proposal."""
    
    proposal_id: str
    proposer_address: str
    operation_type: TreasuryOperationType
    recipient_address: str
    amount: int
    token_address: Optional[str] = None
    description: str = ""
    justification: str = ""
    
    # Approval requirements
    requires_multisig: bool = True
    multisig_threshold: int = 3
    multisig_signatures: List[str] = field(default_factory=list)
    
    # Status and metadata
    status: TreasuryStatus = TreasuryStatus.PENDING
    created_at: float = field(default_factory=time.time)
    approved_at: Optional[float] = None
    executed_at: Optional[float] = None
    
    # Security and validation
    max_amount: Optional[int] = None
    daily_limit: Optional[int] = None
    monthly_limit: Optional[int] = None
    
    def __post_init__(self):
        """Validate treasury proposal after initialization."""
        if self.amount <= 0:
            raise ValidationError("Treasury amount must be positive")
        
        if not self.recipient_address:
            raise ValidationError("Treasury proposal must have a recipient")
        
        if self.requires_multisig and self.multisig_threshold <= 0:
            raise ValidationError("Multisig threshold must be positive")
    
    def add_multisig_signature(self, signature: str) -> None:
        """Add a multisig signature."""
        if signature not in self.multisig_signatures:
            self.multisig_signatures.append(signature)
    
    def is_multisig_approved(self) -> bool:
        """Check if multisig approval is complete."""
        if not self.requires_multisig:
            return True
        
        return len(self.multisig_signatures) >= self.multisig_threshold
    
    def can_execute(self) -> bool:
        """Check if proposal can be executed."""
        return (
            self.status == TreasuryStatus.APPROVED and
            self.is_multisig_approved()
        )


@dataclass
class TreasuryBalance:
    """Treasury balance for a specific token."""
    
    token_address: str
    token_symbol: str
    balance: int
    last_updated: float = field(default_factory=time.time)
    
    def add_balance(self, amount: int) -> None:
        """Add to treasury balance."""
        self.balance += amount
        self.last_updated = time.time()
    
    def subtract_balance(self, amount: int) -> bool:
        """Subtract from treasury balance."""
        if self.balance >= amount:
            self.balance -= amount
            self.last_updated = time.time()
            return True
        return False


@dataclass
class TreasuryLimits:
    """Treasury spending limits."""
    
    max_single_transaction: int = 1000000
    max_daily_spending: int = 5000000
    max_monthly_spending: int = 50000000
    max_emergency_withdrawal: int = 10000000
    
    # Time windows
    daily_window: int = 86400  # 24 hours
    monthly_window: int = 2592000  # 30 days
    
    def validate_amount(self, amount: int, operation_type: TreasuryOperationType) -> bool:
        """Validate spending amount against limits."""
        if operation_type == TreasuryOperationType.EMERGENCY_WITHDRAWAL:
            return amount <= self.max_emergency_withdrawal
        
        return amount <= self.max_single_transaction


class TreasuryManager:
    """Manages treasury operations and security."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize treasury manager."""
        self.config = config or {}
        self.balances: Dict[str, TreasuryBalance] = {}
        self.proposals: Dict[str, TreasuryProposal] = {}
        self.limits = TreasuryLimits(**self.config.get("limits", {}))
        
        # Spending tracking
        self.daily_spending: Dict[str, List[Tuple[float, int]]] = {}  # token -> [(timestamp, amount)]
        self.monthly_spending: Dict[str, List[Tuple[float, int]]] = {}  # token -> [(timestamp, amount)]
        
        # Multisig management
        self.multisig_signers: Set[str] = set()
        self.multisig_threshold = self.config.get("multisig_threshold", 3)
        
        # Security
        self.blocked_addresses: Set[str] = set()
        self.whitelisted_addresses: Set[str] = set()
    
    def add_multisig_signer(self, address: str) -> None:
        """Add a multisig signer."""
        self.multisig_signers.add(address)
    
    def remove_multisig_signer(self, address: str) -> None:
        """Remove a multisig signer."""
        self.multisig_signers.discard(address)
    
    def create_treasury_proposal(
        self,
        proposer_address: str,
        operation_type: TreasuryOperationType,
        recipient_address: str,
        amount: int,
        token_address: Optional[str] = None,
        description: str = "",
        justification: str = "",
        requires_multisig: bool = True
    ) -> TreasuryProposal:
        """Create a new treasury proposal."""
        # Validate amount against limits
        if not self.limits.validate_amount(amount, operation_type):
            raise ValidationError(f"Amount {amount} exceeds treasury limits")
        
        # Check if recipient is blocked
        if recipient_address in self.blocked_addresses:
            raise ValidationError("Recipient address is blocked")
        
        # Check if token has sufficient balance
        if token_address and token_address in self.balances:
            if self.balances[token_address].balance < amount:
                raise ValidationError("Insufficient treasury balance")
        
        # Create proposal
        proposal = TreasuryProposal(
            proposal_id=f"treasury_{int(time.time())}_{proposer_address[:8]}",
            proposer_address=proposer_address,
            operation_type=operation_type,
            recipient_address=recipient_address,
            amount=amount,
            token_address=token_address,
            description=description,
            justification=justification,
            requires_multisig=requires_multisig,
            multisig_threshold=self.multisig_threshold,
        )
        
        # Add to proposals
        self.proposals[proposal.proposal_id] = proposal
        
        return proposal
    
    def approve_treasury_proposal(
        self,
        proposal_id: str,
        approver_address: str,
        signature: str
    ) -> bool:
        """Approve a treasury proposal."""
        if proposal_id not in self.proposals:
            raise ValidationError(f"Treasury proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        # Check if approver is a valid multisig signer
        if approver_address not in self.multisig_signers:
            raise ValidationError("Approver is not a valid multisig signer")
        
        # Add signature
        proposal.add_multisig_signature(signature)
        
        # Check if fully approved
        if proposal.is_multisig_approved():
            proposal.status = TreasuryStatus.APPROVED
            proposal.approved_at = time.time()
            return True
        
        return False
    
    def execute_treasury_proposal(
        self,
        proposal_id: str,
        executor_address: str
    ) -> bool:
        """Execute a treasury proposal."""
        if proposal_id not in self.proposals:
            raise ValidationError(f"Treasury proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        # Check if proposal can be executed
        if not proposal.can_execute():
            raise ValidationError("Proposal cannot be executed")
        
        # Check spending limits
        if not self._check_spending_limits(proposal):
            raise ValidationError("Proposal exceeds spending limits")
        
        # Execute the operation
        success = self._execute_treasury_operation(proposal)
        
        if success:
            proposal.status = TreasuryStatus.EXECUTED
            proposal.executed_at = time.time()
            
            # Update spending tracking
            self._update_spending_tracking(proposal)
        
        return success
    
    def _check_spending_limits(self, proposal: TreasuryProposal) -> bool:
        """Check if proposal meets spending limits."""
        token_address = proposal.token_address or "default"
        current_time = time.time()
        
        # Check daily limit
        daily_spending = self._get_spending_in_window(
            self.daily_spending.get(token_address, []),
            current_time,
            self.limits.daily_window
        )
        
        if daily_spending + proposal.amount > self.limits.max_daily_spending:
            return False
        
        # Check monthly limit
        monthly_spending = self._get_spending_in_window(
            self.monthly_spending.get(token_address, []),
            current_time,
            self.limits.monthly_window
        )
        
        if monthly_spending + proposal.amount > self.limits.max_monthly_spending:
            return False
        
        return True
    
    def _get_spending_in_window(
        self,
        spending_history: List[Tuple[float, int]],
        current_time: float,
        window: int
    ) -> int:
        """Get total spending in a time window."""
        cutoff_time = current_time - window
        return sum(
            amount for timestamp, amount in spending_history
            if timestamp >= cutoff_time
        )
    
    def _execute_treasury_operation(self, proposal: TreasuryProposal) -> bool:
        """Execute a treasury operation."""
        token_address = proposal.token_address or "default"
        
        # Ensure balance exists
        if token_address not in self.balances:
            self.balances[token_address] = TreasuryBalance(
                token_address=token_address,
                token_symbol=token_address,
                balance=0
            )
        
        # Execute based on operation type
        if proposal.operation_type == TreasuryOperationType.SPENDING:
            return self.balances[token_address].subtract_balance(proposal.amount)
        elif proposal.operation_type == TreasuryOperationType.TRANSFER:
            return self.balances[token_address].subtract_balance(proposal.amount)
        elif proposal.operation_type == TreasuryOperationType.BURN:
            return self.balances[token_address].subtract_balance(proposal.amount)
        elif proposal.operation_type == TreasuryOperationType.EMERGENCY_WITHDRAWAL:
            return self.balances[token_address].subtract_balance(proposal.amount)
        
        return False
    
    def _update_spending_tracking(self, proposal: TreasuryProposal) -> None:
        """Update spending tracking after execution."""
        token_address = proposal.token_address or "default"
        current_time = time.time()
        
        # Update daily spending
        if token_address not in self.daily_spending:
            self.daily_spending[token_address] = []
        self.daily_spending[token_address].append((current_time, proposal.amount))
        
        # Update monthly spending
        if token_address not in self.monthly_spending:
            self.monthly_spending[token_address] = []
        self.monthly_spending[token_address].append((current_time, proposal.amount))
    
    def add_treasury_balance(
        self,
        token_address: str,
        amount: int,
        token_symbol: str = None
    ) -> None:
        """Add balance to treasury."""
        if token_address not in self.balances:
            self.balances[token_address] = TreasuryBalance(
                token_address=token_address,
                token_symbol=token_symbol or token_address,
                balance=0
            )
        
        self.balances[token_address].add_balance(amount)
    
    def get_treasury_balance(self, token_address: str) -> int:
        """Get treasury balance for a token."""
        if token_address not in self.balances:
            return 0
        return self.balances[token_address].balance
    
    def get_all_balances(self) -> Dict[str, TreasuryBalance]:
        """Get all treasury balances."""
        return self.balances.copy()
    
    def block_address(self, address: str) -> None:
        """Block an address from receiving treasury funds."""
        self.blocked_addresses.add(address)
    
    def unblock_address(self, address: str) -> None:
        """Unblock an address."""
        self.blocked_addresses.discard(address)
    
    def whitelist_address(self, address: str) -> None:
        """Whitelist an address for treasury operations."""
        self.whitelisted_addresses.add(address)
    
    def get_treasury_statistics(self) -> Dict[str, Any]:
        """Get treasury statistics."""
        total_balance = sum(balance.balance for balance in self.balances.values())
        total_proposals = len(self.proposals)
        executed_proposals = len([p for p in self.proposals.values() if p.status == TreasuryStatus.EXECUTED])
        
        return {
            "total_balance": total_balance,
            "total_proposals": total_proposals,
            "executed_proposals": executed_proposals,
            "multisig_signers": len(self.multisig_signers),
            "blocked_addresses": len(self.blocked_addresses),
            "whitelisted_addresses": len(self.whitelisted_addresses),
            "token_count": len(self.balances),
        }
    
    def get_spending_summary(self, token_address: str = None) -> Dict[str, Any]:
        """Get spending summary for a token."""
        if token_address is None:
            token_address = "default"
        
        current_time = time.time()
        
        daily_spending = self._get_spending_in_window(
            self.daily_spending.get(token_address, []),
            current_time,
            self.limits.daily_window
        )
        
        monthly_spending = self._get_spending_in_window(
            self.monthly_spending.get(token_address, []),
            current_time,
            self.limits.monthly_window
        )
        
        return {
            "token_address": token_address,
            "daily_spending": daily_spending,
            "monthly_spending": monthly_spending,
            "daily_limit": self.limits.max_daily_spending,
            "monthly_limit": self.limits.max_monthly_spending,
            "daily_remaining": self.limits.max_daily_spending - daily_spending,
            "monthly_remaining": self.limits.max_monthly_spending - monthly_spending,
        }
