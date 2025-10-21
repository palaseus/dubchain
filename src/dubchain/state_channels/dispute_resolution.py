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

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

class DisputeStatus(Enum):
    """Status of a dispute resolution process."""
    PENDING = "pending"
    EVIDENCE_PERIOD = "evidence_period"
    CHALLENGE_PERIOD = "challenge_period"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    FRAUD_DETECTED = "fraud_detected"

class EvidenceType(Enum):
    """Types of evidence."""
    STATE_UPDATE = "state_update"
    SIGNATURE = "signature"
    TIMESTAMP = "timestamp"
    TRANSACTION_HASH = "transaction_hash"
    BLOCK_PROOF = "block_proof"
    FRAUD_PROOF = "fraud_proof"

@dataclass
class DisputeEvidence:
    """Evidence submitted for dispute resolution."""
    evidence_id: str
    channel_id: str
    submitter: str
    evidence_type: EvidenceType
    evidence_data: Dict[str, Any]
    timestamp: int
    signature: Optional[str] = None
    block_height: Optional[int] = None

@dataclass
class DisputeResolution:
    """Dispute resolution process."""
    dispute_id: str
    channel_id: str
    initiator: str
    status: DisputeStatus
    created_at: int
    evidence_period_end: int
    challenge_period_end: int
    evidence: List[DisputeEvidence] = field(default_factory=list)
    resolution_data: Optional[Dict[str, Any]] = None
    fraud_detected: bool = False
    penalty_amount: int = 0

@dataclass
class FraudProof:
    """Fraud proof for dispute resolution."""
    proof_id: str
    channel_id: str
    fraudulent_state: Dict[str, Any]
    correct_state: Dict[str, Any]
    evidence: List[DisputeEvidence]
    submitter: str
    timestamp: int
    signature: Optional[str] = None

@dataclass
class DisputeConfig:
    """Configuration for dispute resolution."""
    evidence_period_duration: int = 86400  # 24 hours
    challenge_period_duration: int = 172800  # 48 hours
    max_evidence_per_dispute: int = 10
    fraud_penalty_multiplier: float = 2.0
    min_stake_for_dispute: int = 1000
    enable_fraud_proofs: bool = True

class EvidenceValidator:
    """Validates evidence submitted for disputes."""
    
    def __init__(self, config: DisputeConfig):
        """Initialize evidence validator."""
        self.config = config
        logger.info("Initialized evidence validator")
    
    def validate_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate evidence."""
        try:
            # Check evidence type
            if not isinstance(evidence.evidence_type, EvidenceType):
                logger.error(f"Invalid evidence type: {evidence.evidence_type}")
                return False
            
            # Validate evidence data based on type
            if evidence.evidence_type == EvidenceType.STATE_UPDATE:
                return self._validate_state_update_evidence(evidence)
            elif evidence.evidence_type == EvidenceType.SIGNATURE:
                return self._validate_signature_evidence(evidence)
            elif evidence.evidence_type == EvidenceType.TIMESTAMP:
                return self._validate_timestamp_evidence(evidence)
            elif evidence.evidence_type == EvidenceType.TRANSACTION_HASH:
                return self._validate_transaction_hash_evidence(evidence)
            elif evidence.evidence_type == EvidenceType.BLOCK_PROOF:
                return self._validate_block_proof_evidence(evidence)
            elif evidence.evidence_type == EvidenceType.FRAUD_PROOF:
                return self._validate_fraud_proof_evidence(evidence)
            else:
                logger.error(f"Unknown evidence type: {evidence.evidence_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating evidence: {e}")
            return False
    
    def _validate_state_update_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate state update evidence."""
        try:
            data = evidence.evidence_data
            
            # Check required fields
            required_fields = ['state_hash', 'nonce', 'participants']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in state update evidence")
                    return False
            
            # Validate state hash format
            state_hash = data['state_hash']
            if not isinstance(state_hash, str) or len(state_hash) != 64:
                logger.error("Invalid state hash format")
                return False
            
            # Validate nonce
            nonce = data['nonce']
            if not isinstance(nonce, int) or nonce < 0:
                logger.error("Invalid nonce")
                return False
            
            # Validate participants
            participants = data['participants']
            if not isinstance(participants, list) or len(participants) < 2:
                logger.error("Invalid participants list")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating state update evidence: {e}")
            return False
    
    def _validate_signature_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate signature evidence."""
        try:
            data = evidence.evidence_data
            
            # Check required fields
            required_fields = ['signature', 'public_key', 'message_hash']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in signature evidence")
                return False
            
            # Validate signature format
            signature = data['signature']
            if not isinstance(signature, str) or len(signature) != 128:
                logger.error("Invalid signature format")
                return False
            
            # Validate public key format
            public_key = data['public_key']
            if not isinstance(public_key, str) or len(public_key) != 66:
                logger.error("Invalid public key format")
                return False
            
            # Validate message hash format
            message_hash = data['message_hash']
            if not isinstance(message_hash, str) or len(message_hash) != 64:
                logger.error("Invalid message hash format")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating signature evidence: {e}")
            return False
    
    def _validate_timestamp_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate timestamp evidence."""
        try:
            data = evidence.evidence_data
            
            # Check required fields
            if 'timestamp' not in data:
                logger.error("Missing timestamp in timestamp evidence")
                return False
            
            # Validate timestamp
            timestamp = data['timestamp']
            if not isinstance(timestamp, int) or timestamp <= 0:
                logger.error("Invalid timestamp")
                return False
            
            # Check if timestamp is reasonable (not too far in past or future)
            current_time = int(time.time())
            if abs(timestamp - current_time) > 31536000:  # 1 year
                logger.error("Timestamp too far from current time")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating timestamp evidence: {e}")
            return False
    
    def _validate_transaction_hash_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate transaction hash evidence."""
        try:
            data = evidence.evidence_data
            
            # Check required fields
            if 'transaction_hash' not in data:
                logger.error("Missing transaction hash in transaction hash evidence")
                return False
            
            # Validate transaction hash format
            tx_hash = data['transaction_hash']
            if not isinstance(tx_hash, str) or len(tx_hash) != 64:
                logger.error("Invalid transaction hash format")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating transaction hash evidence: {e}")
            return False
    
    def _validate_block_proof_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate block proof evidence."""
        try:
            data = evidence.evidence_data
            
            # Check required fields
            required_fields = ['block_hash', 'merkle_proof', 'block_height']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in block proof evidence")
                    return False
            
            # Validate block hash format
            block_hash = data['block_hash']
            if not isinstance(block_hash, str) or len(block_hash) != 64:
                logger.error("Invalid block hash format")
                return False
            
            # Validate merkle proof
            merkle_proof = data['merkle_proof']
            if not isinstance(merkle_proof, list):
                logger.error("Invalid merkle proof format")
                return False
            
            # Validate block height
            block_height = data['block_height']
            if not isinstance(block_height, int) or block_height < 0:
                logger.error("Invalid block height")
                return False
                
                return True
            
        except Exception as e:
            logger.error(f"Error validating block proof evidence: {e}")
            return False
    
    def _validate_fraud_proof_evidence(self, evidence: DisputeEvidence) -> bool:
        """Validate fraud proof evidence."""
        try:
            data = evidence.evidence_data
            
            # Check required fields
            required_fields = ['fraudulent_state', 'correct_state', 'proof_data']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in fraud proof evidence")
                return False
            
            # Validate states
            fraudulent_state = data['fraudulent_state']
            correct_state = data['correct_state']
            
            if not isinstance(fraudulent_state, dict) or not isinstance(correct_state, dict):
                logger.error("Invalid state format")
                return False
            
            # Validate proof data
            proof_data = data['proof_data']
            if not isinstance(proof_data, dict):
                logger.error("Invalid proof data format")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating fraud proof evidence: {e}")
            return False

class FraudDetector:
    """Detects fraud in state channel disputes."""
    
    def __init__(self, config: DisputeConfig):
        """Initialize fraud detector."""
        self.config = config
        logger.info("Initialized fraud detector")
    
    def detect_fraud(self, dispute: DisputeResolution) -> Optional[FraudProof]:
        """Detect fraud in dispute evidence."""
        try:
            if not self.config.enable_fraud_proofs:
                return None
            
            # Analyze evidence for inconsistencies
            inconsistencies = self._find_inconsistencies(dispute.evidence)
            
            if inconsistencies:
                # Create fraud proof
                fraud_proof = self._create_fraud_proof(dispute, inconsistencies)
                return fraud_proof
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            return None
    
    def _find_inconsistencies(self, evidence: List[DisputeEvidence]) -> List[Dict[str, Any]]:
        """Find inconsistencies in evidence."""
        inconsistencies = []
        
        try:
            # Group evidence by type
            evidence_by_type = {}
            for ev in evidence:
                if ev.evidence_type not in evidence_by_type:
                    evidence_by_type[ev.evidence_type] = []
                evidence_by_type[ev.evidence_type].append(ev)
            
            # Check for conflicting state updates
            if EvidenceType.STATE_UPDATE in evidence_by_type:
                state_evidence = evidence_by_type[EvidenceType.STATE_UPDATE]
                for i, ev1 in enumerate(state_evidence):
                    for ev2 in state_evidence[i+1:]:
                        if self._states_conflict(ev1.evidence_data, ev2.evidence_data):
                            inconsistencies.append({
                                'type': 'conflicting_states',
                                'evidence1': ev1.evidence_id,
                                'evidence2': ev2.evidence_id,
                                'description': 'Conflicting state updates found'
                            })
            
            # Check for signature inconsistencies
            if EvidenceType.SIGNATURE in evidence_by_type:
                signature_evidence = evidence_by_type[EvidenceType.SIGNATURE]
                for ev in signature_evidence:
                    if not self._validate_signature_consistency(ev):
                        inconsistencies.append({
                            'type': 'invalid_signature',
                            'evidence': ev.evidence_id,
                            'description': 'Invalid signature found'
                        })
            
            # Check for timestamp inconsistencies
            if EvidenceType.TIMESTAMP in evidence_by_type:
                timestamp_evidence = evidence_by_type[EvidenceType.TIMESTAMP]
                timestamps = [ev.evidence_data['timestamp'] for ev in timestamp_evidence]
                if len(set(timestamps)) != len(timestamps):
                    inconsistencies.append({
                        'type': 'duplicate_timestamps',
                        'description': 'Duplicate timestamps found'
                    })
            
            return inconsistencies
            
        except Exception as e:
            logger.error(f"Error finding inconsistencies: {e}")
            return []
    
    def _states_conflict(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Check if two states conflict."""
        try:
            # Compare state hashes
            hash1 = state1.get('state_hash')
            hash2 = state2.get('state_hash')
            
            if hash1 and hash2 and hash1 != hash2:
                # Check if nonces are different
                nonce1 = state1.get('nonce', 0)
                nonce2 = state2.get('nonce', 0)
                
                if nonce1 != nonce2:
                    return True
    
            return False
        
        except Exception as e:
            logger.error(f"Error checking state conflicts: {e}")
            return False
        
    def _validate_signature_consistency(self, evidence: DisputeEvidence) -> bool:
        """Validate signature consistency."""
        try:
            data = evidence.evidence_data
            
            # Basic format validation
            signature = data.get('signature')
            public_key = data.get('public_key')
            message_hash = data.get('message_hash')
            
            if not all([signature, public_key, message_hash]):
                return False
        
            # In a real implementation, would verify the signature
            # For now, just check format
            return (len(signature) == 128 and 
                   len(public_key) == 66 and 
                   len(message_hash) == 64)
            
        except Exception as e:
            logger.error(f"Error validating signature consistency: {e}")
            return False
        
    def _create_fraud_proof(self, dispute: DisputeResolution, inconsistencies: List[Dict[str, Any]]) -> FraudProof:
        """Create fraud proof from inconsistencies."""
        try:
            # Find conflicting states
            conflicting_states = []
            for inconsistency in inconsistencies:
                if inconsistency['type'] == 'conflicting_states':
                    conflicting_states.append(inconsistency)
            
            if not conflicting_states:
                # Create generic fraud proof
                fraudulent_state = {'type': 'generic_fraud', 'inconsistencies': inconsistencies}
                correct_state = {'type': 'correct_state', 'resolved': True}
            else:
                # Use first conflicting state
                first_conflict = conflicting_states[0]
                fraudulent_state = {'type': 'conflicting_state', 'conflict': first_conflict}
                correct_state = {'type': 'resolved_state', 'resolution': 'fraud_detected'}
            
            fraud_proof = FraudProof(
                proof_id=f"fraud_proof_{dispute.dispute_id}_{int(time.time())}",
                channel_id=dispute.channel_id,
                fraudulent_state=fraudulent_state,
                correct_state=correct_state,
                evidence=dispute.evidence,
                submitter="fraud_detector",
                timestamp=int(time.time())
            )
            
            return fraud_proof
            
        except Exception as e:
            logger.error(f"Error creating fraud proof: {e}")
            raise ClientError(f"Failed to create fraud proof: {e}")

class DisputeResolver:
    """Resolves disputes based on evidence."""
    
    def __init__(self, config: DisputeConfig):
        """Initialize dispute resolver."""
        self.config = config
        self.evidence_validator = EvidenceValidator(config)
        self.fraud_detector = FraudDetector(config)
        logger.info("Initialized dispute resolver")
    
    def resolve_dispute(self, dispute: DisputeResolution) -> Dict[str, Any]:
        """Resolve a dispute based on evidence."""
        try:
            logger.info(f"Resolving dispute {dispute.dispute_id}")
            
            # Validate all evidence
            valid_evidence = []
            for evidence in dispute.evidence:
                if self.evidence_validator.validate_evidence(evidence):
                    valid_evidence.append(evidence)
                else:
                    logger.warning(f"Invalid evidence {evidence.evidence_id}")
            
            if not valid_evidence:
                logger.error(f"No valid evidence for dispute {dispute.dispute_id}")
                return {
                    'status': 'failed',
                    'reason': 'no_valid_evidence',
                    'resolution': None
                }
            
            # Detect fraud
            fraud_proof = self.fraud_detector.detect_fraud(dispute)
            if fraud_proof:
                logger.warning(f"Fraud detected in dispute {dispute.dispute_id}")
                return self._resolve_fraud_dispute(dispute, fraud_proof)
            
            # Resolve based on evidence
            resolution = self._resolve_based_on_evidence(dispute, valid_evidence)
            
            logger.info(f"Dispute {dispute.dispute_id} resolved: {resolution['status']}")
            return resolution
            
        except Exception as e:
            logger.error(f"Error resolving dispute: {e}")
            return {
                'status': 'failed',
                'reason': 'resolution_error',
                'error': str(e),
                'resolution': None
            }
    
    def _resolve_fraud_dispute(self, dispute: DisputeResolution, fraud_proof: FraudProof) -> Dict[str, Any]:
        """Resolve dispute with fraud detected."""
        try:
            # Calculate penalty
            penalty_amount = int(self.config.fraud_penalty_multiplier * self.config.min_stake_for_dispute)
            
            resolution = {
                'status': 'fraud_detected',
                'fraud_proof': fraud_proof,
                'penalty_amount': penalty_amount,
                'resolution': {
                    'type': 'fraud_penalty',
                    'amount': penalty_amount,
                    'reason': 'fraud_detected'
                }
            }
            
            return resolution
            
        except Exception as e:
            logger.error(f"Error resolving fraud dispute: {e}")
            return {
                'status': 'failed',
                'reason': 'fraud_resolution_error',
                'error': str(e)
            }
    
    def _resolve_based_on_evidence(self, dispute: DisputeResolution, valid_evidence: List[DisputeEvidence]) -> Dict[str, Any]:
        """Resolve dispute based on valid evidence."""
        try:
            # Find the most recent valid state update
            state_updates = [ev for ev in valid_evidence if ev.evidence_type == EvidenceType.STATE_UPDATE]
            
            if not state_updates:
                logger.error(f"No state updates in dispute {dispute.dispute_id}")
                return {
                    'status': 'failed',
                    'reason': 'no_state_updates',
                    'resolution': None
                }
            
            # Sort by nonce (highest first)
            state_updates.sort(key=lambda x: x.evidence_data.get('nonce', 0), reverse=True)
            latest_state = state_updates[0]
            
            # Create resolution
            resolution = {
                'status': 'resolved',
                'resolution': {
                    'type': 'state_update',
                    'state': latest_state.evidence_data,
                    'evidence_id': latest_state.evidence_id,
                    'nonce': latest_state.evidence_data.get('nonce', 0)
                }
            }
            
            return resolution
            
        except Exception as e:
            logger.error(f"Error resolving based on evidence: {e}")
            return {
                'status': 'failed',
                'reason': 'evidence_resolution_error',
                'error': str(e)
            }

class DisputeManager:
    """Main dispute management system."""
    
    def __init__(self, config: DisputeConfig):
        """Initialize dispute manager."""
        self.config = config
        self.disputes: Dict[str, DisputeResolution] = {}
        self.resolver = DisputeResolver(config)
        logger.info("Initialized dispute manager")
    
    def create_dispute(self, channel_id: str, initiator: str) -> str:
        """Create a new dispute."""
        try:
            dispute_id = f"dispute_{channel_id}_{int(time.time())}"
            current_time = int(time.time())
            
            dispute = DisputeResolution(
                dispute_id=dispute_id,
                channel_id=channel_id,
                initiator=initiator,
                status=DisputeStatus.PENDING,
                created_at=current_time,
                evidence_period_end=current_time + self.config.evidence_period_duration,
                challenge_period_end=current_time + self.config.evidence_period_duration + self.config.challenge_period_duration
            )
            
            self.disputes[dispute_id] = dispute
            
            logger.info(f"Created dispute {dispute_id} for channel {channel_id}")
            return dispute_id
            
        except Exception as e:
            logger.error(f"Error creating dispute: {e}")
            raise ClientError(f"Failed to create dispute: {e}")
    
    def submit_evidence(self, dispute_id: str, evidence: DisputeEvidence) -> bool:
        """Submit evidence for a dispute."""
        try:
            if dispute_id not in self.disputes:
                logger.error(f"Dispute {dispute_id} not found")
                return False
            
            dispute = self.disputes[dispute_id]
            
            # Check if evidence period is still active
            current_time = int(time.time())
            if current_time > dispute.evidence_period_end:
                logger.error(f"Evidence period ended for dispute {dispute_id}")
                return False
            
            # Check evidence limit
            if len(dispute.evidence) >= self.config.max_evidence_per_dispute:
                logger.error(f"Evidence limit reached for dispute {dispute_id}")
                return False
            
            # Add evidence
            dispute.evidence.append(evidence)
            
            # Update status
            if dispute.status == DisputeStatus.PENDING:
                dispute.status = DisputeStatus.EVIDENCE_PERIOD
            
            logger.info(f"Submitted evidence {evidence.evidence_id} for dispute {dispute_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting evidence: {e}")
            return False
    
    def resolve_dispute(self, dispute_id: str) -> Dict[str, Any]:
        """Resolve a dispute."""
        try:
            if dispute_id not in self.disputes:
                logger.error(f"Dispute {dispute_id} not found")
                return {
                    'status': 'failed',
                    'reason': 'dispute_not_found'
                }
            
            dispute = self.disputes[dispute_id]
            
            # Check if dispute can be resolved
            current_time = int(time.time())
            if current_time < dispute.challenge_period_end:
                logger.error(f"Challenge period still active for dispute {dispute_id}")
                return {
                    'status': 'failed',
                    'reason': 'challenge_period_active'
                }
            
            # Resolve dispute
            resolution = self.resolver.resolve_dispute(dispute)
            
            # Update dispute status
            if resolution['status'] == 'resolved':
                dispute.status = DisputeStatus.RESOLVED
            elif resolution['status'] == 'fraud_detected':
                dispute.status = DisputeStatus.FRAUD_DETECTED
                dispute.fraud_detected = True
                dispute.penalty_amount = resolution.get('penalty_amount', 0)
            else:
                    dispute.status = DisputeStatus.EXPIRED
            
            dispute.resolution_data = resolution
            
            logger.info(f"Resolved dispute {dispute_id}: {resolution['status']}")
            return resolution
            
        except Exception as e:
            logger.error(f"Error resolving dispute: {e}")
            return {
                'status': 'failed',
                'reason': 'resolution_error',
                'error': str(e)
            }
    
    def get_dispute(self, dispute_id: str) -> Optional[DisputeResolution]:
        """Get dispute information."""
        return self.disputes.get(dispute_id)
    
    def list_disputes(self, channel_id: Optional[str] = None) -> List[DisputeResolution]:
        """List disputes."""
        disputes = list(self.disputes.values())
        
        if channel_id:
            disputes = [d for d in disputes if d.channel_id == channel_id]
        
        return disputes
    
    def cleanup_expired_disputes(self) -> int:
        """Clean up expired disputes."""
        try:
            current_time = int(time.time())
            expired_disputes = []
            
            for dispute_id, dispute in self.disputes.items():
                if (dispute.status == DisputeStatus.PENDING and 
                    current_time > dispute.challenge_period_end):
                    dispute.status = DisputeStatus.EXPIRED
                    expired_disputes.append(dispute_id)
            
            logger.info(f"Cleaned up {len(expired_disputes)} expired disputes")
            return len(expired_disputes)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired disputes: {e}")
            return 0

class OnChainContract:
    """On-chain smart contract for dispute resolution."""
    
    def __init__(self, contract_address: str):
        """Initialize on-chain contract."""
        self.contract_address = contract_address
        logger.info(f"Initialized on-chain contract at {contract_address}")
    
    def submit_evidence(self, dispute_id: str, evidence: DisputeEvidence) -> bool:
        """Submit evidence to on-chain contract."""
        try:
            logger.info(f"Submitting evidence {evidence.evidence_id} for dispute {dispute_id}")
            # In a real implementation, would interact with smart contract
            return True
        except Exception as e:
            logger.error(f"Error submitting evidence: {e}")
            return False
    
    def resolve_dispute(self, dispute_id: str) -> Dict[str, Any]:
        """Resolve dispute on-chain."""
        try:
            logger.info(f"Resolving dispute {dispute_id} on-chain")
            # In a real implementation, would call smart contract
            return {
                'status': 'resolved',
                'resolution': 'on_chain_resolution'
            }
        except Exception as e:
            logger.error(f"Error resolving dispute on-chain: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

__all__ = [
    "DisputeManager",
    "DisputeResolver",
    "EvidenceValidator",
    "FraudDetector",
    "DisputeResolution",
    "DisputeEvidence",
    "FraudProof",
    "DisputeConfig",
    "DisputeStatus",
    "EvidenceType",
    "OnChainContract",
]