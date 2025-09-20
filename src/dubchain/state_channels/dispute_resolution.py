"""
Dispute Resolution System for State Channels

This module implements the on-chain dispute resolution mechanism including:
- Smart contract for enforcing final state during disputes
- Evidence collection and validation
- Timeout-based resolution
- Fraud proof mechanisms
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PublicKey, Signature
from ..vm import SmartContract, ExecutionContext, ExecutionResult
from .channel_protocol import (
    ChannelConfig,
    ChannelId,
    ChannelState,
    StateUpdate,
    ChannelCloseReason,
)


class DisputeStatus(Enum):
    """Status of a dispute resolution process."""
    PENDING = "pending"           # Dispute submitted, waiting for evidence
    EVIDENCE_PERIOD = "evidence_period"  # Collecting evidence
    CHALLENGE_PERIOD = "challenge_period"  # Challenge period active
    RESOLVED = "resolved"         # Dispute resolved
    EXPIRED = "expired"           # Dispute expired without resolution
    FRAUD_DETECTED = "fraud_detected"  # Fraud detected and penalized


@dataclass
class DisputeEvidence:
    """Evidence submitted for dispute resolution."""
    evidence_id: str
    channel_id: ChannelId
    submitter: str
    evidence_type: str
    evidence_data: Dict[str, Any]
    timestamp: int
    signature: Optional[Signature] = None
    
    def get_hash(self) -> Hash:
        """Get hash of this evidence."""
        data = {
            "evidence_id": self.evidence_id,
            "channel_id": self.channel_id.value,
            "submitter": self.submitter,
            "evidence_type": self.evidence_type,
            "evidence_data": self.evidence_data,
            "timestamp": self.timestamp,
        }
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return SHA256Hasher.hash(serialized)
    
    def verify_signature(self, public_key: PublicKey) -> bool:
        """Verify the signature on this evidence."""
        if not self.signature:
            return False
        
        evidence_hash = self.get_hash()
        return public_key.verify(self.signature, evidence_hash)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "channel_id": self.channel_id.value,
            "submitter": self.submitter,
            "evidence_type": self.evidence_type,
            "evidence_data": self.evidence_data,
            "timestamp": self.timestamp,
            "signature": self.signature.to_hex() if self.signature else None,
        }


@dataclass
class DisputeResolution:
    """Represents a dispute resolution process."""
    dispute_id: str
    channel_id: ChannelId
    initiator: str
    reason: str
    status: DisputeStatus
    created_at: int
    evidence_period_end: int
    challenge_period_end: int
    
    # Evidence and challenges
    evidence: List[DisputeEvidence] = field(default_factory=list)
    challenges: List[DisputeEvidence] = field(default_factory=list)
    
    # Resolution
    resolved_state: Optional[ChannelState] = None
    resolution_timestamp: Optional[int] = None
    resolution_reason: Optional[str] = None
    
    def is_evidence_period_active(self) -> bool:
        """Check if evidence period is still active."""
        return time.time() < self.evidence_period_end
    
    def is_challenge_period_active(self) -> bool:
        """Check if challenge period is still active."""
        return time.time() < self.challenge_period_end
    
    def can_submit_evidence(self) -> bool:
        """Check if evidence can still be submitted."""
        return self.status == DisputeStatus.EVIDENCE_PERIOD and self.is_evidence_period_active()
    
    def can_submit_challenge(self) -> bool:
        """Check if challenges can still be submitted."""
        return self.status == DisputeStatus.CHALLENGE_PERIOD and self.is_challenge_period_active()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dispute_id": self.dispute_id,
            "channel_id": self.channel_id.value,
            "initiator": self.initiator,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at,
            "evidence_period_end": self.evidence_period_end,
            "challenge_period_end": self.challenge_period_end,
            "evidence": [e.to_dict() for e in self.evidence],
            "challenges": [c.to_dict() for c in self.challenges],
            "resolved_state": self.resolved_state.to_dict() if self.resolved_state else None,
            "resolution_timestamp": self.resolution_timestamp,
            "resolution_reason": self.resolution_reason,
        }


class OnChainContract:
    """On-chain smart contract for state channel dispute resolution."""
    
    def __init__(self, contract_address: str):
        self.contract_address = contract_address
        self.disputes: Dict[str, DisputeResolution] = {}
        self.channel_registry: Dict[str, Dict[str, Any]] = {}
        self.fraud_penalties: Dict[str, int] = {}
        
        # Contract state
        self.total_deposits: int = 0
        self.total_disputes: int = 0
        self.resolved_disputes: int = 0
    
    def register_channel(
        self,
        channel_id: ChannelId,
        participants: List[str],
        deposits: Dict[str, int],
        config: ChannelConfig
    ) -> bool:
        """Register a new state channel on-chain."""
        try:
            # Validate inputs
            if len(participants) < 2:
                return False
            
            if len(participants) > config.max_participants:
                return False
            
            # Check minimum deposits
            for participant, deposit in deposits.items():
                if deposit < config.min_deposit:
                    return False
            
            # Register channel
            self.channel_registry[channel_id.value] = {
                "participants": participants,
                "deposits": deposits,
                "config": config.to_dict(),
                "registered_at": int(time.time()),
                "status": "active",
                "total_deposits": sum(deposits.values()),
            }
            
            # Update total deposits
            self.total_deposits += sum(deposits.values())
            
            return True
        
        except Exception:
            return False
    
    def initiate_dispute(
        self,
        channel_id: ChannelId,
        initiator: str,
        reason: str,
        evidence_period_blocks: int = 100,
        challenge_period_blocks: int = 50
    ) -> Optional[str]:
        """Initiate a dispute resolution process."""
        try:
            # Check if channel is registered
            if channel_id.value not in self.channel_registry:
                return None
            
            # Check if dispute already exists
            for dispute in self.disputes.values():
                if dispute.channel_id == channel_id and dispute.status in [
                    DisputeStatus.PENDING,
                    DisputeStatus.EVIDENCE_PERIOD,
                    DisputeStatus.CHALLENGE_PERIOD
                ]:
                    return None  # Dispute already exists
            
            # Create dispute
            dispute_id = f"dispute_{channel_id.value}_{int(time.time())}"
            current_time = int(time.time())
            
            dispute = DisputeResolution(
                dispute_id=dispute_id,
                channel_id=channel_id,
                initiator=initiator,
                reason=reason,
                status=DisputeStatus.EVIDENCE_PERIOD,
                created_at=current_time,
                evidence_period_end=current_time + (evidence_period_blocks * 12),  # Assume 12s per block
                challenge_period_end=current_time + ((evidence_period_blocks + challenge_period_blocks) * 12),
            )
            
            self.disputes[dispute_id] = dispute
            self.total_disputes += 1
            
            return dispute_id
        
        except Exception:
            return None
    
    def submit_evidence(
        self,
        dispute_id: str,
        evidence: DisputeEvidence,
        submitter_public_key: PublicKey
    ) -> bool:
        """Submit evidence for a dispute."""
        try:
            # Check if dispute exists
            if dispute_id not in self.disputes:
                return False
            
            dispute = self.disputes[dispute_id]
            
            # Check if evidence can be submitted
            if not dispute.can_submit_evidence():
                return False
            
            # Verify evidence signature
            if not evidence.verify_signature(submitter_public_key):
                return False
            
            # Validate evidence based on type
            if not self._validate_evidence(evidence, dispute):
                return False
            
            # Add evidence
            dispute.evidence.append(evidence)
            
            return True
        
        except Exception:
            return False
    
    def submit_challenge(
        self,
        dispute_id: str,
        challenge: DisputeEvidence,
        challenger_public_key: PublicKey
    ) -> bool:
        """Submit a challenge to evidence."""
        try:
            # Check if dispute exists
            if dispute_id not in self.disputes:
                return False
            
            dispute = self.disputes[dispute_id]
            
            # Check if challenge can be submitted
            if not dispute.can_submit_challenge():
                return False
            
            # Verify challenge signature
            if not challenge.verify_signature(challenger_public_key):
                return False
            
            # Validate challenge
            if not self._validate_challenge(challenge, dispute):
                return False
            
            # Add challenge
            dispute.challenges.append(challenge)
            
            # Move to challenge period if not already there
            if dispute.status == DisputeStatus.EVIDENCE_PERIOD:
                dispute.status = DisputeStatus.CHALLENGE_PERIOD
            
            return True
        
        except Exception:
            return False
    
    def resolve_dispute(
        self,
        dispute_id: str,
        final_state: ChannelState,
        resolution_reason: str
    ) -> bool:
        """Resolve a dispute with a final state."""
        try:
            # Check if dispute exists
            if dispute_id not in self.disputes:
                return False
            
            dispute = self.disputes[dispute_id]
            
            # Check if dispute can be resolved
            if dispute.status not in [DisputeStatus.EVIDENCE_PERIOD, DisputeStatus.CHALLENGE_PERIOD]:
                return False
            
            # Validate final state
            if not self._validate_final_state(final_state, dispute):
                return False
            
            # Resolve dispute
            dispute.status = DisputeStatus.RESOLVED
            dispute.resolved_state = final_state
            dispute.resolution_timestamp = int(time.time())
            dispute.resolution_reason = resolution_reason
            
            # Update channel registry
            channel_id = dispute.channel_id
            if channel_id.value in self.channel_registry:
                self.channel_registry[channel_id.value]["status"] = "resolved"
                self.channel_registry[channel_id.value]["resolved_at"] = int(time.time())
                self.channel_registry[channel_id.value]["final_state"] = final_state.to_dict()
            
            self.resolved_disputes += 1
            
            return True
        
        except Exception:
            return False
    
    def expire_dispute(self, dispute_id: str) -> bool:
        """Expire a dispute due to timeout."""
        try:
            if dispute_id not in self.disputes:
                return False
            
            dispute = self.disputes[dispute_id]
            
            # Check if dispute has expired
            if dispute.status in [DisputeStatus.RESOLVED, DisputeStatus.EXPIRED]:
                return False
            
            current_time = time.time()
            if current_time > dispute.challenge_period_end:
                dispute.status = DisputeStatus.EXPIRED
                dispute.resolution_timestamp = int(current_time)
                dispute.resolution_reason = "Dispute expired due to timeout"
                
                # Apply timeout resolution (e.g., return to last valid state)
                self._apply_timeout_resolution(dispute)
                
                return True
            
            return False
        
        except Exception:
            return False
    
    def detect_fraud(
        self,
        dispute_id: str,
        fraud_evidence: DisputeEvidence,
        fraudster: str
    ) -> bool:
        """Detect and penalize fraud."""
        try:
            if dispute_id not in self.disputes:
                return False
            
            dispute = self.disputes[dispute_id]
            
            # Validate fraud evidence
            if not self._validate_fraud_evidence(fraud_evidence, dispute):
                return False
            
            # Apply fraud penalty
            penalty_amount = self._calculate_fraud_penalty(dispute)
            self.fraud_penalties[fraudster] = self.fraud_penalties.get(fraudster, 0) + penalty_amount
            
            # Update dispute status
            dispute.status = DisputeStatus.FRAUD_DETECTED
            dispute.resolution_timestamp = int(time.time())
            dispute.resolution_reason = f"Fraud detected from {fraudster}"
            
            return True
        
        except Exception:
            return False
    
    def _validate_evidence(self, evidence: DisputeEvidence, dispute: DisputeResolution) -> bool:
        """Validate submitted evidence."""
        evidence_type = evidence.evidence_type
        
        if evidence_type == "state_update":
            return self._validate_state_update_evidence(evidence, dispute)
        elif evidence_type == "signature_proof":
            return self._validate_signature_proof_evidence(evidence, dispute)
        elif evidence_type == "timeout_proof":
            return self._validate_timeout_proof_evidence(evidence, dispute)
        elif evidence_type == "fraud_proof":
            return self._validate_fraud_proof_evidence(evidence, dispute)
        
        return False
    
    def _validate_state_update_evidence(
        self, 
        evidence: DisputeEvidence, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate state update evidence."""
        evidence_data = evidence.evidence_data
        
        # Check required fields
        required_fields = ["state_update", "participants", "signatures"]
        for field in required_fields:
            if field not in evidence_data:
                return False
        
        # Validate state update structure
        state_update_data = evidence_data["state_update"]
        if not isinstance(state_update_data, dict):
            return False
        
        # Validate participants match channel
        channel_info = self.channel_registry.get(dispute.channel_id.value)
        if not channel_info:
            return False
        
        evidence_participants = set(evidence_data["participants"])
        channel_participants = set(channel_info["participants"])
        
        if evidence_participants != channel_participants:
            return False
        
        return True
    
    def _validate_signature_proof_evidence(
        self, 
        evidence: DisputeEvidence, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate signature proof evidence."""
        evidence_data = evidence.evidence_data
        
        # Check required fields
        required_fields = ["message_hash", "signature", "public_key"]
        for field in required_fields:
            if field not in evidence_data:
                return False
        
        # Validate signature format
        signature_hex = evidence_data["signature"]
        if not isinstance(signature_hex, str) or len(signature_hex) != 128:  # 64 bytes in hex
            return False
        
        return True
    
    def _validate_timeout_proof_evidence(
        self, 
        evidence: DisputeEvidence, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate timeout proof evidence."""
        evidence_data = evidence.evidence_data
        
        # Check required fields
        required_fields = ["timeout_timestamp", "block_number", "proof"]
        for field in required_fields:
            if field not in evidence_data:
                return False
        
        # Validate timeout timestamp
        timeout_timestamp = evidence_data["timeout_timestamp"]
        if not isinstance(timeout_timestamp, (int, float)):
            return False
        
        # Check if timeout has actually occurred
        if timeout_timestamp > time.time():
            return False
        
        return True
    
    def _validate_fraud_proof_evidence(
        self, 
        evidence: DisputeEvidence, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate fraud proof evidence."""
        evidence_data = evidence.evidence_data
        
        # Check required fields
        required_fields = ["fraud_type", "proof_data", "violation_details"]
        for field in required_fields:
            if field not in evidence_data:
                return False
        
        # Validate fraud type
        fraud_type = evidence_data["fraud_type"]
        valid_fraud_types = ["double_spend", "invalid_signature", "state_manipulation", "replay_attack"]
        if fraud_type not in valid_fraud_types:
            return False
        
        return True
    
    def _validate_challenge(
        self, 
        challenge: DisputeEvidence, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate a challenge to evidence."""
        # Basic validation - challenges should be similar to evidence
        return self._validate_evidence(challenge, dispute)
    
    def _validate_final_state(
        self, 
        final_state: ChannelState, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate the final state for dispute resolution."""
        # Check channel ID matches
        if final_state.channel_id != dispute.channel_id:
            return False
        
        # Check participants match
        channel_info = self.channel_registry.get(dispute.channel_id.value)
        if not channel_info:
            return False
        
        if set(final_state.participants) != set(channel_info["participants"]):
            return False
        
        # Check balance conservation
        if not final_state.validate_balances():
            return False
        
        # Check deposits match
        if final_state.deposits != channel_info["deposits"]:
            return False
        
        return True
    
    def _validate_fraud_evidence(
        self, 
        fraud_evidence: DisputeEvidence, 
        dispute: DisputeResolution
    ) -> bool:
        """Validate fraud evidence."""
        return self._validate_fraud_proof_evidence(fraud_evidence, dispute)
    
    def _calculate_fraud_penalty(self, dispute: DisputeResolution) -> int:
        """Calculate penalty for fraud."""
        channel_info = self.channel_registry.get(dispute.channel_id.value)
        if not channel_info:
            return 0
        
        # Penalty is 10% of total deposits
        total_deposits = channel_info["total_deposits"]
        return total_deposits // 10
    
    def _apply_timeout_resolution(self, dispute: DisputeResolution) -> None:
        """Apply timeout-based resolution."""
        # In a timeout, we might return to the last valid state
        # or apply some default resolution logic
        channel_info = self.channel_registry.get(dispute.channel_id.value)
        if channel_info:
            channel_info["status"] = "timeout_resolved"
            channel_info["resolved_at"] = int(time.time())
    
    def get_dispute(self, dispute_id: str) -> Optional[DisputeResolution]:
        """Get a dispute by ID."""
        return self.disputes.get(dispute_id)
    
    def get_channel_disputes(self, channel_id: ChannelId) -> List[DisputeResolution]:
        """Get all disputes for a channel."""
        return [dispute for dispute in self.disputes.values() 
                if dispute.channel_id == channel_id]
    
    def get_contract_state(self) -> Dict[str, Any]:
        """Get the current state of the contract."""
        return {
            "contract_address": self.contract_address,
            "total_deposits": self.total_deposits,
            "total_disputes": self.total_disputes,
            "resolved_disputes": self.resolved_disputes,
            "active_disputes": len([d for d in self.disputes.values() 
                                  if d.status in [DisputeStatus.EVIDENCE_PERIOD, DisputeStatus.CHALLENGE_PERIOD]]),
            "registered_channels": len(self.channel_registry),
            "fraud_penalties": self.fraud_penalties,
        }


class DisputeManager:
    """Manages dispute resolution processes."""
    
    def __init__(self, on_chain_contract: OnChainContract):
        self.contract = on_chain_contract
        self.active_disputes: Dict[str, DisputeResolution] = {}
        self.dispute_callbacks: List[callable] = []
        
        # Dispute monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 60  # seconds
    
    def initiate_dispute(
        self,
        channel_id: ChannelId,
        initiator: str,
        reason: str,
        initial_evidence: Optional[DisputeEvidence] = None
    ) -> Optional[str]:
        """Initiate a new dispute."""
        # Create dispute on-chain
        dispute_id = self.contract.initiate_dispute(channel_id, initiator, reason)
        if not dispute_id:
            return None
        
        # Get the created dispute
        dispute = self.contract.get_dispute(dispute_id)
        if not dispute:
            return None
        
        # Add initial evidence if provided
        if initial_evidence:
            # Note: In practice, you'd need the submitter's public key
            # self.contract.submit_evidence(dispute_id, initial_evidence, submitter_public_key)
            pass
        
        # Track active dispute
        self.active_disputes[dispute_id] = dispute
        
        # Notify callbacks
        self._notify_dispute_callbacks("dispute_initiated", dispute)
        
        return dispute_id
    
    def submit_evidence(
        self,
        dispute_id: str,
        evidence: DisputeEvidence,
        submitter_public_key: PublicKey
    ) -> bool:
        """Submit evidence for a dispute."""
        success = self.contract.submit_evidence(dispute_id, evidence, submitter_public_key)
        
        if success:
            # Update local dispute tracking
            dispute = self.contract.get_dispute(dispute_id)
            if dispute:
                self.active_disputes[dispute_id] = dispute
                self._notify_dispute_callbacks("evidence_submitted", dispute)
        
        return success
    
    def submit_challenge(
        self,
        dispute_id: str,
        challenge: DisputeEvidence,
        challenger_public_key: PublicKey
    ) -> bool:
        """Submit a challenge to evidence."""
        success = self.contract.submit_challenge(dispute_id, challenge, challenger_public_key)
        
        if success:
            # Update local dispute tracking
            dispute = self.contract.get_dispute(dispute_id)
            if dispute:
                self.active_disputes[dispute_id] = dispute
                self._notify_dispute_callbacks("challenge_submitted", dispute)
        
        return success
    
    def resolve_dispute(
        self,
        dispute_id: str,
        final_state: ChannelState,
        resolution_reason: str
    ) -> bool:
        """Resolve a dispute."""
        success = self.contract.resolve_dispute(dispute_id, final_state, resolution_reason)
        
        if success:
            # Update local dispute tracking
            dispute = self.contract.get_dispute(dispute_id)
            if dispute:
                self.active_disputes[dispute_id] = dispute
                self._notify_dispute_callbacks("dispute_resolved", dispute)
        
        return success
    
    def monitor_disputes(self) -> None:
        """Monitor active disputes for timeouts and updates."""
        if not self.monitoring_enabled:
            return
        
        current_time = time.time()
        disputes_to_remove = []
        
        for dispute_id, dispute in self.active_disputes.items():
            # Check for timeout
            if dispute.status in [DisputeStatus.EVIDENCE_PERIOD, DisputeStatus.CHALLENGE_PERIOD]:
                if current_time > dispute.challenge_period_end:
                    # Dispute has expired
                    self.contract.expire_dispute(dispute_id)
                    dispute.status = DisputeStatus.EXPIRED
                    self._notify_dispute_callbacks("dispute_expired", dispute)
                    disputes_to_remove.append(dispute_id)
            
            # Check for resolution
            elif dispute.status == DisputeStatus.RESOLVED:
                disputes_to_remove.append(dispute_id)
        
        # Remove resolved/expired disputes
        for dispute_id in disputes_to_remove:
            del self.active_disputes[dispute_id]
    
    def add_dispute_callback(self, callback: callable) -> None:
        """Add a callback for dispute events."""
        self.dispute_callbacks.append(callback)
    
    def remove_dispute_callback(self, callback: callable) -> None:
        """Remove a dispute callback."""
        if callback in self.dispute_callbacks:
            self.dispute_callbacks.remove(callback)
    
    def _notify_dispute_callbacks(self, event_type: str, dispute: DisputeResolution) -> None:
        """Notify all dispute callbacks of an event."""
        for callback in self.dispute_callbacks:
            try:
                callback(event_type, dispute)
            except Exception as e:
                print(f"Error in dispute callback: {e}")
    
    def get_active_disputes(self) -> List[DisputeResolution]:
        """Get all active disputes."""
        return list(self.active_disputes.values())
    
    def get_dispute_status(self, dispute_id: str) -> Optional[DisputeStatus]:
        """Get the status of a dispute."""
        dispute = self.active_disputes.get(dispute_id)
        if dispute:
            return dispute.status
        
        # Check on-chain contract
        dispute = self.contract.get_dispute(dispute_id)
        if dispute:
            return dispute.status
        
        return None
    
    def get_dispute_statistics(self) -> Dict[str, Any]:
        """Get dispute resolution statistics."""
        return {
            "active_disputes": len(self.active_disputes),
            "total_disputes": self.contract.total_disputes,
            "resolved_disputes": self.contract.resolved_disputes,
            "dispute_resolution_rate": (
                self.contract.resolved_disputes / self.contract.total_disputes 
                if self.contract.total_disputes > 0 else 0
            ),
            "contract_state": self.contract.get_contract_state(),
        }
