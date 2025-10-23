"""
Observability and audit trail system for governance.

This module provides comprehensive observability including events, audit trails,
merkle proofs for off-chain votes, and governance metrics.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..errors.exceptions import ValidationError
from .core import Proposal, Vote, ProposalStatus, VoteChoice


class EventType(Enum):
    """Types of governance events."""
    
    PROPOSAL_CREATED = "proposal_created"
    PROPOSAL_ACTIVATED = "proposal_activated"
    PROPOSAL_QUEUED = "proposal_queued"
    PROPOSAL_EXECUTED = "proposal_executed"
    PROPOSAL_CANCELLED = "proposal_cancelled"
    PROPOSAL_EXPIRED = "proposal_expired"
    
    VOTE_CAST = "vote_cast"
    VOTE_CHANGED = "vote_changed"
    VOTE_REVOKED = "vote_revoked"
    
    DELEGATION_CREATED = "delegation_created"
    DELEGATION_REVOKED = "delegation_revoked"
    
    TREASURY_SPENDING = "treasury_spending"
    TREASURY_APPROVAL = "treasury_approval"
    
    EMERGENCY_PAUSE = "emergency_pause"
    EMERGENCY_RESUME = "emergency_resume"
    
    SECURITY_ALERT = "security_alert"
    ADDRESS_BLOCKED = "address_blocked"
    ADDRESS_UNBLOCKED = "address_unblocked"


@dataclass
class GovernanceEvent:
    """A governance event for audit trail."""
    
    event_id: str
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    block_height: Optional[int] = None
    transaction_hash: Optional[str] = None
    
    # Event data
    proposal_id: Optional[str] = None
    voter_address: Optional[str] = None
    delegatee_address: Optional[str] = None
    delegator_address: Optional[str] = None
    
    # Event metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cryptographic integrity
    event_hash: Optional[str] = None
    previous_event_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate event hash after initialization."""
        self.event_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash of this event."""
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "block_height": self.block_height,
            "transaction_hash": self.transaction_hash,
            "proposal_id": self.proposal_id,
            "voter_address": self.voter_address,
            "delegatee_address": self.delegatee_address,
            "delegator_address": self.delegator_address,
            "metadata": self.metadata,
            "previous_event_hash": self.previous_event_hash,
        }
        
        event_json = json.dumps(event_data, sort_keys=True)
        return str(SHA256Hasher.hash(event_json.encode()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "block_height": self.block_height,
            "transaction_hash": self.transaction_hash,
            "proposal_id": self.proposal_id,
            "voter_address": self.voter_address,
            "delegatee_address": self.delegatee_address,
            "delegator_address": self.delegator_address,
            "metadata": self.metadata,
            "event_hash": self.event_hash,
            "previous_event_hash": self.previous_event_hash,
        }


class AuditTrail:
    """Maintains an immutable audit trail of governance events."""
    
    def __init__(self):
        """Initialize audit trail."""
        self.events: List[GovernanceEvent] = []
        self.event_index: Dict[str, int] = {}  # event_id -> index
        self.proposal_events: Dict[str, List[GovernanceEvent]] = {}  # proposal_id -> events
        self.voter_events: Dict[str, List[GovernanceEvent]] = {}  # voter_address -> events
    
    def add_event(self, event: GovernanceEvent) -> None:
        """Add an event to the audit trail."""
        # Set previous event hash for chain integrity
        if self.events:
            event.previous_event_hash = self.events[-1].event_hash
        
        # Recalculate hash with previous event hash
        event.event_hash = event._calculate_hash()
        
        # Add to events list
        self.events.append(event)
        self.event_index[event.event_id] = len(self.events) - 1
        
        # Index by proposal
        if event.proposal_id:
            if event.proposal_id not in self.proposal_events:
                self.proposal_events[event.proposal_id] = []
            self.proposal_events[event.proposal_id].append(event)
        
        # Index by voter
        if event.voter_address:
            if event.voter_address not in self.voter_events:
                self.voter_events[event.voter_address] = []
            self.voter_events[event.voter_address].append(event)
    
    def get_event(self, event_id: str) -> Optional[GovernanceEvent]:
        """Get an event by ID."""
        if event_id in self.event_index:
            return self.events[self.event_index[event_id]]
        return None
    
    def get_proposal_events(self, proposal_id: str) -> List[GovernanceEvent]:
        """Get all events for a proposal."""
        return self.proposal_events.get(proposal_id, [])
    
    def get_voter_events(self, voter_address: str) -> List[GovernanceEvent]:
        """Get all events for a voter."""
        return self.voter_events.get(voter_address, [])
    
    def get_events_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[GovernanceEvent]:
        """Get events within a time range."""
        return [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
        ]
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit trail."""
        for i, event in enumerate(self.events):
            # Verify event hash
            if event.event_hash != event._calculate_hash():
                return False
            
            # Verify chain integrity
            if i > 0:
                if event.previous_event_hash != self.events[i-1].event_hash:
                    return False
        
        return True
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit trail summary."""
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "unique_proposals": len(self.proposal_events),
            "unique_voters": len(self.voter_events),
            "integrity_verified": self.verify_integrity(),
        }


class MerkleProofManager:
    """Manages Merkle proofs for off-chain voting and governance data."""
    
    def __init__(self):
        """Initialize Merkle proof manager."""
        self.merkle_trees: Dict[str, str] = {}  # tree_id -> merkle_root
        self.tree_data: Dict[str, List[str]] = {}  # tree_id -> data_leaves
    
    def create_merkle_tree(
        self,
        tree_id: str,
        data: List[str]
    ) -> str:
        """Create a Merkle tree from data."""
        if not data:
            return SHA256Hasher.hash(b"empty")
        
        # Sort data for deterministic ordering
        sorted_data = sorted(data)
        self.tree_data[tree_id] = sorted_data
        
        # Create leaf hashes
        leaves = [SHA256Hasher.hash(item.encode()) for item in sorted_data]
        
        # Build Merkle tree
        merkle_root = self._build_merkle_tree(leaves)
        self.merkle_trees[tree_id] = merkle_root
        
        return merkle_root
    
    def _build_merkle_tree(self, leaves: List[str]) -> str:
        """Build Merkle tree from leaves."""
        if not leaves:
            return SHA256Hasher.hash(b"empty")
        
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = SHA256Hasher.hash(left + right)
                next_level.append(combined)
            current_level = next_level
        
        return current_level[0]
    
    def generate_merkle_proof(
        self,
        tree_id: str,
        data_item: str
    ) -> Optional[Dict[str, Any]]:
        """Generate Merkle proof for a data item."""
        if tree_id not in self.tree_data:
            return None
        
        data_list = self.tree_data[tree_id]
        if data_item not in data_list:
            return None
        
        # Find item index
        item_index = data_list.index(data_item)
        
        # Create leaf hashes
        leaves = [SHA256Hasher.hash(item.encode()) for item in data_list]
        
        # Generate proof path
        proof_path = []
        current_level = leaves
        current_index = item_index
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = SHA256Hasher.hash(left + right)
                next_level.append(combined)
                
                # Add to proof if this is our path
                if i == current_index or i + 1 == current_index:
                    sibling = right if i == current_index else left
                    proof_path.append(sibling)
            
            current_index = current_index // 2
            current_level = next_level
        
        return {
            "tree_id": tree_id,
            "data_item": data_item,
            "merkle_root": self.merkle_trees[tree_id],
            "proof_path": proof_path,
            "leaf_index": item_index,
        }
    
    def verify_merkle_proof(self, proof: Dict[str, Any]) -> bool:
        """Verify a Merkle proof."""
        data_item = proof["data_item"]
        merkle_root = proof["merkle_root"]
        proof_path = proof["proof_path"]
        leaf_index = proof["leaf_index"]
        
        # Recreate leaf hash
        leaf_hash = SHA256Hasher.hash(data_item.encode())
        
        # Verify proof path
        current_hash = leaf_hash
        current_index = leaf_index
        
        for sibling_hash in proof_path:
            if current_index % 2 == 0:
                # Current is left, sibling is right
                current_hash = SHA256Hasher.hash(current_hash + sibling_hash)
            else:
                # Current is right, sibling is left
                current_hash = SHA256Hasher.hash(sibling_hash + current_hash)
            current_index = current_index // 2
        
        return current_hash == merkle_root
    
    def create_vote_merkle_tree(
        self,
        proposal_id: str,
        votes: List[Vote]
    ) -> str:
        """Create Merkle tree for proposal votes."""
        # Create vote data strings
        vote_data = []
        for vote in votes:
            vote_string = f"{vote.voter_address}:{vote.choice.value}:{vote.voting_power.total_power()}"
            vote_data.append(vote_string)
        
        tree_id = f"votes_{proposal_id}"
        return self.create_merkle_tree(tree_id, vote_data)
    
    def generate_vote_proof(
        self,
        proposal_id: str,
        vote: Vote
    ) -> Optional[Dict[str, Any]]:
        """Generate Merkle proof for a vote."""
        tree_id = f"votes_{proposal_id}"
        vote_string = f"{vote.voter_address}:{vote.choice.value}:{vote.voting_power.total_power()}"
        return self.generate_merkle_proof(tree_id, vote_string)


@dataclass
class GovernanceMetrics:
    """Governance system metrics."""
    
    # Proposal metrics
    total_proposals: int = 0
    active_proposals: int = 0
    executed_proposals: int = 0
    failed_proposals: int = 0
    cancelled_proposals: int = 0
    
    # Voting metrics
    total_votes: int = 0
    unique_voters: int = 0
    total_voting_power: int = 0
    
    # Delegation metrics
    total_delegations: int = 0
    active_delegations: int = 0
    total_delegated_power: int = 0
    
    # Treasury metrics
    treasury_balance: int = 0
    treasury_spending: int = 0
    treasury_proposals: int = 0
    
    # Security metrics
    security_alerts: int = 0
    blocked_addresses: int = 0
    suspicious_addresses: int = 0
    
    # Performance metrics
    average_proposal_duration: float = 0.0
    average_voting_participation: float = 0.0
    execution_success_rate: float = 0.0
    
    # Timestamps
    last_updated: float = field(default_factory=time.time)
    
    def update_proposal_metrics(
        self,
        total: int,
        active: int,
        executed: int,
        failed: int,
        cancelled: int
    ) -> None:
        """Update proposal metrics."""
        self.total_proposals = total
        self.active_proposals = active
        self.executed_proposals = executed
        self.failed_proposals = failed
        self.cancelled_proposals = cancelled
        self.last_updated = time.time()
    
    def update_voting_metrics(
        self,
        total_votes: int,
        unique_voters: int,
        total_voting_power: int
    ) -> None:
        """Update voting metrics."""
        self.total_votes = total_votes
        self.unique_voters = unique_voters
        self.total_voting_power = total_voting_power
        self.last_updated = time.time()
    
    def update_delegation_metrics(
        self,
        total_delegations: int,
        active_delegations: int,
        total_delegated_power: int
    ) -> None:
        """Update delegation metrics."""
        self.total_delegations = total_delegations
        self.active_delegations = active_delegations
        self.total_delegated_power = total_delegated_power
        self.last_updated = time.time()
    
    def update_treasury_metrics(
        self,
        balance: int,
        spending: int,
        proposals: int
    ) -> None:
        """Update treasury metrics."""
        self.treasury_balance = balance
        self.treasury_spending = spending
        self.treasury_proposals = proposals
        self.last_updated = time.time()
    
    def update_security_metrics(
        self,
        alerts: int,
        blocked: int,
        suspicious: int
    ) -> None:
        """Update security metrics."""
        self.security_alerts = alerts
        self.blocked_addresses = blocked
        self.suspicious_addresses = suspicious
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_proposals": self.total_proposals,
            "active_proposals": self.active_proposals,
            "executed_proposals": self.executed_proposals,
            "failed_proposals": self.failed_proposals,
            "cancelled_proposals": self.cancelled_proposals,
            "total_votes": self.total_votes,
            "unique_voters": self.unique_voters,
            "total_voting_power": self.total_voting_power,
            "total_delegations": self.total_delegations,
            "active_delegations": self.active_delegations,
            "total_delegated_power": self.total_delegated_power,
            "treasury_balance": self.treasury_balance,
            "treasury_spending": self.treasury_spending,
            "treasury_proposals": self.treasury_proposals,
            "security_alerts": self.security_alerts,
            "blocked_addresses": self.blocked_addresses,
            "suspicious_addresses": self.suspicious_addresses,
            "average_proposal_duration": self.average_proposal_duration,
            "average_voting_participation": self.average_voting_participation,
            "execution_success_rate": self.execution_success_rate,
            "last_updated": self.last_updated,
        }


class GovernanceEvents:
    """Event system for governance observability."""
    
    def __init__(self):
        """Initialize governance events system."""
        self.audit_trail = AuditTrail()
        self.merkle_proof_manager = MerkleProofManager()
        self.metrics = GovernanceMetrics()
        
        # Event listeners
        self.event_listeners: Dict[EventType, List[callable]] = {}
    
    def add_event_listener(self, event_type: EventType, listener: callable) -> None:
        """Add an event listener."""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
    
    def emit_event(
        self,
        event_type: EventType,
        proposal_id: Optional[str] = None,
        voter_address: Optional[str] = None,
        delegatee_address: Optional[str] = None,
        delegator_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        block_height: Optional[int] = None,
        transaction_hash: Optional[str] = None
    ) -> GovernanceEvent:
        """Emit a governance event."""
        event = GovernanceEvent(
            event_id=f"{event_type.value}_{int(time.time())}_{hash(str(metadata))}",
            event_type=event_type,
            proposal_id=proposal_id,
            voter_address=voter_address,
            delegatee_address=delegatee_address,
            delegator_address=delegator_address,
            metadata=metadata or {},
            block_height=block_height,
            transaction_hash=transaction_hash,
        )
        
        # Add to audit trail
        self.audit_trail.add_event(event)
        
        # Notify listeners
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    listener(event)
                except Exception as e:
                    logger.info(f"Error in event listener: {e}")
        
        return event
    
    def get_audit_trail(self) -> AuditTrail:
        """Get the audit trail."""
        return self.audit_trail
    
    def get_merkle_proof_manager(self) -> MerkleProofManager:
        """Get the Merkle proof manager."""
        return self.merkle_proof_manager
    
    def get_metrics(self) -> GovernanceMetrics:
        """Get governance metrics."""
        return self.metrics
    
    def export_audit_trail(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Export audit trail for a time range."""
        events = self.audit_trail.get_events_in_range(start_time, end_time)
        return [event.to_dict() for event in events]
    
    def verify_audit_integrity(self) -> bool:
        """Verify the integrity of the audit trail."""
        return self.audit_trail.verify_integrity()
