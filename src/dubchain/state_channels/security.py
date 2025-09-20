"""
Security Module for State Channels

This module implements comprehensive security measures for state channels including:
- Fraud detection and prevention
- Cryptographic proof systems
- Timeout mechanisms
- Byzantine fault tolerance
- Replay attack prevention
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PublicKey, Signature
from .channel_protocol import (
    ChannelConfig,
    ChannelId,
    ChannelState,
    StateUpdate,
    ChannelCloseReason,
)


class SecurityThreat(Enum):
    """Types of security threats."""
    REPLAY_ATTACK = "replay_attack"
    DOUBLE_SPEND = "double_spend"
    INVALID_SIGNATURE = "invalid_signature"
    STATE_MANIPULATION = "state_manipulation"
    TIMEOUT_VIOLATION = "timeout_violation"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    FRAUD_PROOF_VIOLATION = "fraud_proof_violation"
    NONCE_REUSE = "nonce_reuse"
    SEQUENCE_VIOLATION = "sequence_violation"


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CryptographicProof:
    """Represents a cryptographic proof for security validation."""
    proof_type: str
    proof_data: Dict[str, Any]
    timestamp: int
    nonce: int
    signature: Optional[Signature] = None
    
    def get_hash(self) -> Hash:
        """Get hash of this proof."""
        data = {
            "proof_type": self.proof_type,
            "proof_data": self.proof_data,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return SHA256Hasher.hash(serialized)
    
    def verify(self, public_key: PublicKey) -> bool:
        """Verify this cryptographic proof."""
        if not self.signature:
            return False
        
        proof_hash = self.get_hash()
        return public_key.verify(self.signature, proof_hash)


@dataclass
class FraudProof:
    """Represents a fraud proof for detecting malicious behavior."""
    fraud_type: SecurityThreat
    evidence: Dict[str, Any]
    violator: str
    timestamp: int
    proof_data: Dict[str, Any] = field(default_factory=dict)
    severity: SecurityLevel = SecurityLevel.HIGH
    
    def get_hash(self) -> Hash:
        """Get hash of this fraud proof."""
        data = {
            "fraud_type": self.fraud_type.value,
            "evidence": self.evidence,
            "violator": self.violator,
            "timestamp": self.timestamp,
            "proof_data": self.proof_data,
            "severity": self.severity.value,
        }
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return SHA256Hasher.hash(serialized)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fraud_type": self.fraud_type.value,
            "evidence": self.evidence,
            "violator": self.violator,
            "timestamp": self.timestamp,
            "proof_data": self.proof_data,
            "severity": self.severity.value,
        }


@dataclass
class SecurityEvent:
    """Represents a security event or violation."""
    event_id: str
    threat_type: SecurityThreat
    channel_id: ChannelId
    participant: str
    timestamp: int
    severity: SecurityLevel
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.value,
            "channel_id": self.channel_id.value,
            "participant": self.participant,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "details": self.details,
            "resolved": self.resolved,
        }


class TimeoutManager:
    """Manages timeouts and deadline enforcement."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.active_timeouts: Dict[str, Dict[str, Any]] = {}
        self.timeout_callbacks: List[callable] = []
    
    def set_timeout(
        self,
        channel_id: ChannelId,
        timeout_type: str,
        duration: int,
        callback: Optional[callable] = None
    ) -> str:
        """Set a timeout for a channel operation."""
        timeout_id = f"{channel_id.value}_{timeout_type}_{int(time.time())}"
        
        timeout_data = {
            "timeout_id": timeout_id,
            "channel_id": channel_id,
            "timeout_type": timeout_type,
            "expires_at": int(time.time()) + duration,
            "callback": callback,
            "active": True,
        }
        
        self.active_timeouts[timeout_id] = timeout_data
        return timeout_id
    
    def check_timeouts(self) -> List[str]:
        """Check for expired timeouts and return expired timeout IDs."""
        current_time = int(time.time())
        expired_timeouts = []
        
        for timeout_id, timeout_data in self.active_timeouts.items():
            if timeout_data["active"] and current_time >= timeout_data["expires_at"]:
                expired_timeouts.append(timeout_id)
                
                # Execute callback if provided
                if timeout_data["callback"]:
                    try:
                        timeout_data["callback"](timeout_data)
                    except Exception as e:
                        print(f"Error in timeout callback: {e}")
                
                # Notify timeout callbacks
                self._notify_timeout_callbacks(timeout_data)
                
                # Mark as inactive
                timeout_data["active"] = False
        
        return expired_timeouts
    
    def cancel_timeout(self, timeout_id: str) -> bool:
        """Cancel an active timeout."""
        if timeout_id in self.active_timeouts:
            self.active_timeouts[timeout_id]["active"] = False
            return True
        return False
    
    def get_remaining_time(self, timeout_id: str) -> Optional[int]:
        """Get remaining time for a timeout."""
        if timeout_id not in self.active_timeouts:
            return None
        
        timeout_data = self.active_timeouts[timeout_id]
        if not timeout_data["active"]:
            return None
        
        remaining = timeout_data["expires_at"] - int(time.time())
        return max(0, remaining)
    
    def add_timeout_callback(self, callback: callable) -> None:
        """Add a callback for timeout events."""
        self.timeout_callbacks.append(callback)
    
    def _notify_timeout_callbacks(self, timeout_data: Dict[str, Any]) -> None:
        """Notify all timeout callbacks."""
        for callback in self.timeout_callbacks:
            try:
                callback(timeout_data)
            except Exception as e:
                print(f"Error in timeout callback: {e}")


class ChannelSecurity:
    """Core security manager for state channels."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.timeout_manager = TimeoutManager(config)
        self.security_events: List[SecurityEvent] = []
        self.fraud_proofs: List[FraudProof] = []
        self.nonce_tracker: Dict[str, Set[int]] = {}  # participant -> set of used nonces
        self.sequence_tracker: Dict[str, int] = {}  # participant -> last sequence number
        self.replay_protection: Dict[str, Set[str]] = {}  # channel -> set of seen hashes
        
        # Security policies
        self.security_policies: Dict[str, callable] = {}
        self._register_default_policies()
    
    def _register_default_policies(self) -> None:
        """Register default security policies."""
        self.security_policies["replay_protection"] = self._check_replay_attack
        self.security_policies["nonce_validation"] = self._check_nonce_reuse
        self.security_policies["sequence_validation"] = self._check_sequence_violation
        self.security_policies["signature_validation"] = self._check_signature_validity
        self.security_policies["timeout_validation"] = self._check_timeout_violation
        self.security_policies["balance_validation"] = self._check_balance_manipulation
    
    def validate_state_update(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate a state update for security threats."""
        security_events = []
        
        # Check all security policies
        for policy_name, policy_func in self.security_policies.items():
            try:
                is_secure, events = policy_func(update, channel_state, public_keys)
                if not is_secure:
                    security_events.extend(events)
            except Exception as e:
                # Log policy error but don't fail validation
                print(f"Error in security policy {policy_name}: {e}")
        
        # Check for critical security violations
        critical_violations = [e for e in security_events if e.severity == SecurityLevel.CRITICAL]
        if critical_violations:
            return False, security_events
        
        # Record security events
        self.security_events.extend(security_events)
        
        return len(security_events) == 0, security_events
    
    def _check_replay_attack(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check for replay attacks."""
        events = []
        update_hash = update.get_hash().to_hex()
        
        # Check if we've seen this update before
        if channel_state.channel_id.value in self.replay_protection:
            if update_hash in self.replay_protection[channel_state.channel_id.value]:
                event = SecurityEvent(
                    event_id=f"replay_{int(time.time())}",
                    threat_type=SecurityThreat.REPLAY_ATTACK,
                    channel_id=channel_state.channel_id,
                    participant=update.participants[0],  # Assume first participant
                    timestamp=int(time.time()),
                    severity=SecurityLevel.CRITICAL,
                    details={"update_hash": update_hash, "sequence_number": update.sequence_number}
                )
                events.append(event)
                return False, events
        
        # Add to replay protection
        if channel_state.channel_id.value not in self.replay_protection:
            self.replay_protection[channel_state.channel_id.value] = set()
        self.replay_protection[channel_state.channel_id.value].add(update_hash)
        
        return True, events
    
    def _check_nonce_reuse(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check for nonce reuse."""
        events = []
        
        for participant in update.participants:
            if participant in self.nonce_tracker:
                if update.nonce in self.nonce_tracker[participant]:
                    event = SecurityEvent(
                        event_id=f"nonce_reuse_{int(time.time())}",
                        threat_type=SecurityThreat.NONCE_REUSE,
                        channel_id=channel_state.channel_id,
                        participant=participant,
                        timestamp=int(time.time()),
                        severity=SecurityLevel.HIGH,
                        details={"nonce": update.nonce, "sequence_number": update.sequence_number}
                    )
                    events.append(event)
                    return False, events
        
        # Track nonce usage
        for participant in update.participants:
            if participant not in self.nonce_tracker:
                self.nonce_tracker[participant] = set()
            self.nonce_tracker[participant].add(update.nonce)
        
        return True, events
    
    def _check_sequence_violation(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check for sequence number violations."""
        events = []
        
        # Check if sequence number is valid
        expected_sequence = channel_state.sequence_number + 1
        if update.sequence_number != expected_sequence:
            event = SecurityEvent(
                event_id=f"sequence_violation_{int(time.time())}",
                threat_type=SecurityThreat.SEQUENCE_VIOLATION,
                channel_id=channel_state.channel_id,
                participant=update.participants[0],
                timestamp=int(time.time()),
                severity=SecurityLevel.HIGH,
                details={
                    "expected_sequence": expected_sequence,
                    "actual_sequence": update.sequence_number
                }
            )
            events.append(event)
            return False, events
        
        return True, events
    
    def _check_signature_validity(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check signature validity."""
        events = []
        
        # Verify all signatures
        for participant, signature in update.signatures.items():
            if participant not in public_keys:
                event = SecurityEvent(
                    event_id=f"invalid_signature_{int(time.time())}",
                    threat_type=SecurityThreat.INVALID_SIGNATURE,
                    channel_id=channel_state.channel_id,
                    participant=participant,
                    timestamp=int(time.time()),
                    severity=SecurityLevel.CRITICAL,
                    details={"error": "Unknown participant"}
                )
                events.append(event)
                return False, events
            
            public_key = public_keys[participant]
            update_hash = update.get_hash()
            
            if not public_key.verify(signature, update_hash):
                event = SecurityEvent(
                    event_id=f"invalid_signature_{int(time.time())}",
                    threat_type=SecurityThreat.INVALID_SIGNATURE,
                    channel_id=channel_state.channel_id,
                    participant=participant,
                    timestamp=int(time.time()),
                    severity=SecurityLevel.CRITICAL,
                    details={"error": "Signature verification failed"}
                )
                events.append(event)
                return False, events
        
        return True, events
    
    def _check_timeout_violation(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check for timeout violations."""
        events = []
        
        if self.config.enable_timeout_mechanism:
            current_time = int(time.time())
            time_since_last = current_time - channel_state.last_update_timestamp
            
            if time_since_last > self.config.state_update_timeout:
                event = SecurityEvent(
                    event_id=f"timeout_violation_{int(time.time())}",
                    threat_type=SecurityThreat.TIMEOUT_VIOLATION,
                    channel_id=channel_state.channel_id,
                    participant=update.participants[0],
                    timestamp=int(time.time()),
                    severity=SecurityLevel.MEDIUM,
                    details={
                        "time_since_last": time_since_last,
                        "timeout_limit": self.config.state_update_timeout
                    }
                )
                events.append(event)
                return False, events
        
        return True, events
    
    def _check_balance_manipulation(
        self,
        update: StateUpdate,
        channel_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check for balance manipulation."""
        events = []
        
        # Simulate the update to check for balance violations
        if update.update_type.value == "transfer":
            sender = update.state_data.get("sender")
            amount = update.state_data.get("amount", 0)
            
            if sender and sender in channel_state.balances:
                if channel_state.balances[sender] < amount:
                    event = SecurityEvent(
                        event_id=f"balance_manipulation_{int(time.time())}",
                        threat_type=SecurityThreat.DOUBLE_SPEND,
                        channel_id=channel_state.channel_id,
                        participant=sender,
                        timestamp=int(time.time()),
                        severity=SecurityLevel.CRITICAL,
                        details={
                            "available_balance": channel_state.balances[sender],
                            "requested_amount": amount
                        }
                    )
                    events.append(event)
                    return False, events
        
        return True, events
    
    def detect_byzantine_behavior(
        self,
        participant: str,
        behavior_pattern: Dict[str, Any]
    ) -> Optional[FraudProof]:
        """Detect Byzantine behavior patterns."""
        # Analyze behavior patterns for suspicious activity
        suspicious_indicators = []
        
        # Check for rapid state updates (potential spam)
        if "rapid_updates" in behavior_pattern:
            if behavior_pattern["rapid_updates"] > 10:  # More than 10 updates per minute
                suspicious_indicators.append("rapid_updates")
        
        # Check for conflicting state submissions
        if "conflicting_states" in behavior_pattern:
            if behavior_pattern["conflicting_states"] > 0:
                suspicious_indicators.append("conflicting_states")
        
        # Check for signature failures
        if "signature_failures" in behavior_pattern:
            if behavior_pattern["signature_failures"] > 3:
                suspicious_indicators.append("signature_failures")
        
        # If multiple indicators, create fraud proof
        if len(suspicious_indicators) >= 2:
            fraud_proof = FraudProof(
                fraud_type=SecurityThreat.BYZANTINE_BEHAVIOR,
                evidence=behavior_pattern,
                violator=participant,
                timestamp=int(time.time()),
                proof_data={"indicators": suspicious_indicators},
                severity=SecurityLevel.HIGH
            )
            
            self.fraud_proofs.append(fraud_proof)
            return fraud_proof
        
        return None
    
    def create_fraud_proof(
        self,
        fraud_type: SecurityThreat,
        evidence: Dict[str, Any],
        violator: str,
        severity: SecurityLevel = SecurityLevel.HIGH
    ) -> FraudProof:
        """Create a fraud proof for detected malicious behavior."""
        fraud_proof = FraudProof(
            fraud_type=fraud_type,
            evidence=evidence,
            violator=violator,
            timestamp=int(time.time()),
            severity=severity
        )
        
        self.fraud_proofs.append(fraud_proof)
        return fraud_proof
    
    def get_security_events(
        self,
        channel_id: Optional[ChannelId] = None,
        threat_type: Optional[SecurityThreat] = None,
        severity: Optional[SecurityLevel] = None
    ) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        events = self.security_events
        
        if channel_id:
            events = [e for e in events if e.channel_id == channel_id]
        
        if threat_type:
            events = [e for e in events if e.threat_type == threat_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return events
    
    def get_fraud_proofs(
        self,
        fraud_type: Optional[SecurityThreat] = None,
        violator: Optional[str] = None
    ) -> List[FraudProof]:
        """Get fraud proofs with optional filtering."""
        proofs = self.fraud_proofs
        
        if fraud_type:
            proofs = [p for p in proofs if p.fraud_type == fraud_type]
        
        if violator:
            proofs = [p for p in proofs if p.violator == violator]
        
        return proofs
    
    def add_security_policy(self, name: str, policy_func: callable) -> None:
        """Add a custom security policy."""
        self.security_policies[name] = policy_func
    
    def remove_security_policy(self, name: str) -> None:
        """Remove a security policy."""
        if name in self.security_policies:
            del self.security_policies[name]
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        total_events = len(self.security_events)
        critical_events = len([e for e in self.security_events if e.severity == SecurityLevel.CRITICAL])
        resolved_events = len([e for e in self.security_events if e.resolved])
        
        threat_counts = {}
        for event in self.security_events:
            threat_type = event.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            "total_security_events": total_events,
            "critical_events": critical_events,
            "resolved_events": resolved_events,
            "unresolved_events": total_events - resolved_events,
            "threat_type_counts": threat_counts,
            "total_fraud_proofs": len(self.fraud_proofs),
            "active_policies": len(self.security_policies),
        }


class SecurityManager:
    """High-level security manager for state channels."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.channel_security: Dict[ChannelId, ChannelSecurity] = {}
        self.global_security = ChannelSecurity(config)
        self.security_callbacks: List[callable] = []
        
        # Security monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 30  # seconds
    
    def get_channel_security(self, channel_id: ChannelId) -> ChannelSecurity:
        """Get or create security manager for a channel."""
        if channel_id not in self.channel_security:
            self.channel_security[channel_id] = ChannelSecurity(self.config)
        return self.channel_security[channel_id]
    
    def validate_channel_operation(
        self,
        channel_id: ChannelId,
        operation_type: str,
        operation_data: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate a channel operation for security threats."""
        security = self.get_channel_security(channel_id)
        
        # Perform security validation based on operation type
        if operation_type == "state_update":
            return self._validate_state_update_operation(security, operation_data)
        elif operation_type == "channel_creation":
            return self._validate_channel_creation_operation(security, operation_data)
        elif operation_type == "channel_closure":
            return self._validate_channel_closure_operation(security, operation_data)
        else:
            return True, []  # Unknown operation type, allow by default
    
    def _validate_state_update_operation(
        self,
        security: ChannelSecurity,
        operation_data: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate a state update operation."""
        # Extract data from operation
        update = operation_data.get("update")
        channel_state = operation_data.get("channel_state")
        public_keys = operation_data.get("public_keys", {})
        
        if not all([update, channel_state]):
            return False, []
        
        return security.validate_state_update(update, channel_state, public_keys)
    
    def _validate_channel_creation_operation(
        self,
        security: ChannelSecurity,
        operation_data: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate a channel creation operation."""
        # Basic validation for channel creation
        participants = operation_data.get("participants", [])
        deposits = operation_data.get("deposits", {})
        
        events = []
        
        # Check minimum participants
        if len(participants) < 2:
            event = SecurityEvent(
                event_id=f"insufficient_participants_{int(time.time())}",
                threat_type=SecurityThreat.BYZANTINE_BEHAVIOR,
                channel_id=operation_data.get("channel_id"),
                participant=participants[0] if participants else "unknown",
                timestamp=int(time.time()),
                severity=SecurityLevel.MEDIUM,
                details={"participant_count": len(participants)}
            )
            events.append(event)
        
        # Check deposit amounts
        for participant, deposit in deposits.items():
            if deposit < self.config.min_deposit:
                event = SecurityEvent(
                    event_id=f"insufficient_deposit_{int(time.time())}",
                    threat_type=SecurityThreat.BYZANTINE_BEHAVIOR,
                    channel_id=operation_data.get("channel_id"),
                    participant=participant,
                    timestamp=int(time.time()),
                    severity=SecurityLevel.MEDIUM,
                    details={"deposit": deposit, "minimum": self.config.min_deposit}
                )
                events.append(event)
        
        return len(events) == 0, events
    
    def _validate_channel_closure_operation(
        self,
        security: ChannelSecurity,
        operation_data: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate a channel closure operation."""
        # Basic validation for channel closure
        return True, []  # Channel closure is generally safe
    
    def monitor_security(self) -> None:
        """Monitor security across all channels."""
        if not self.monitoring_enabled:
            return
        
        # Check timeouts
        for security in self.channel_security.values():
            expired_timeouts = security.timeout_manager.check_timeouts()
            if expired_timeouts:
                self._notify_security_callbacks("timeout_expired", {
                    "expired_timeouts": expired_timeouts
                })
        
        # Check for suspicious patterns
        for channel_id, security in self.channel_security.items():
            # Analyze recent security events
            recent_events = [e for e in security.security_events 
                           if time.time() - e.timestamp < 300]  # Last 5 minutes
            
            if len(recent_events) > 10:  # More than 10 events in 5 minutes
                self._notify_security_callbacks("suspicious_activity", {
                    "channel_id": channel_id,
                    "event_count": len(recent_events)
                })
    
    def add_security_callback(self, callback: callable) -> None:
        """Add a security event callback."""
        self.security_callbacks.append(callback)
    
    def remove_security_callback(self, callback: callable) -> None:
        """Remove a security callback."""
        if callback in self.security_callbacks:
            self.security_callbacks.remove(callback)
    
    def _notify_security_callbacks(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Notify all security callbacks."""
        for callback in self.security_callbacks:
            try:
                callback(event_type, event_data)
            except Exception as e:
                print(f"Error in security callback: {e}")
    
    def get_global_security_statistics(self) -> Dict[str, Any]:
        """Get global security statistics."""
        total_events = 0
        total_fraud_proofs = 0
        channel_count = len(self.channel_security)
        
        for security in self.channel_security.values():
            total_events += len(security.security_events)
            total_fraud_proofs += len(security.fraud_proofs)
        
        return {
            "total_channels": channel_count,
            "total_security_events": total_events,
            "total_fraud_proofs": total_fraud_proofs,
            "monitoring_enabled": self.monitoring_enabled,
            "global_security_events": len(self.global_security.security_events),
        }
