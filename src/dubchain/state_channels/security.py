"""
Security Module for State Channels

This module implements security features including:
- Cryptographic security measures
- Access control and permissions
- Security monitoring and alerts
- Threat detection and prevention
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of threats."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INVALID_SIGNATURE = "invalid_signature"
    BALANCE_MANIPULATION = "balance_manipulation"
    REPLAY_ATTACK = "replay_attack"
    DOUBLE_SPENDING = "double_spending"
    FRAUD = "fraud"
    MALICIOUS_BEHAVIOR = "malicious_behavior"

@dataclass
class SecurityThreat:
    """Represents a security threat."""
    threat_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    detected_at: float
    source: str
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[float] = None

class AccessLevel(Enum):
    """Access levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"

@dataclass
class SecurityEvent:
    """Security event."""
    event_id: str
    channel_id: str
    threat_type: ThreatType
    security_level: SecurityLevel
    description: str
    timestamp: int
    source_address: Optional[str] = None
    target_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessControl:
    """Access control rules."""
    channel_id: str
    address: str
    access_level: AccessLevel
    permissions: Set[str] = field(default_factory=set)
    expires_at: Optional[int] = None
    created_at: int = field(default_factory=lambda: int(time.time()))

@dataclass
class SecurityConfig:
    """Configuration for security."""
    enable_access_control: bool = True
    enable_threat_detection: bool = True
    enable_security_monitoring: bool = True
    max_failed_attempts: int = 5
    lockout_duration: int = 3600  # 1 hour
    enable_rate_limiting: bool = True
    rate_limit_window: int = 60  # 1 minute
    max_requests_per_window: int = 100
    enable_signature_validation: bool = True
    enable_replay_protection: bool = True

class SignatureValidator:
    """Validates cryptographic signatures."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize signature validator."""
        self.config = config
        self.used_nonces: Set[str] = set()
        logger.info("Initialized signature validator")
    
    def validate_signature(self, message: str, signature: str, public_key: str, nonce: Optional[str] = None) -> bool:
        """Validate cryptographic signature."""
        try:
            if not self.config.enable_signature_validation:
                return True
            
            # Check replay protection
            if self.config.enable_replay_protection and nonce:
                if nonce in self.used_nonces:
                    logger.warning(f"Replay attack detected with nonce {nonce}")
                    return False
                self.used_nonces.add(nonce)
            
            # Basic signature format validation
            if not self._validate_signature_format(signature):
                logger.error("Invalid signature format")
                return False
            
            if not self._validate_public_key_format(public_key):
                logger.error("Invalid public key format")
                return False
            
            # In a real implementation, would verify the signature using cryptography
            # For now, just check format and nonce
            return True
            
        except Exception as e:
            logger.error(f"Error validating signature: {e}")
            return False
    
    def _validate_signature_format(self, signature: str) -> bool:
        """Validate signature format."""
        try:
            # Check if signature is hex string
            if not signature or len(signature) != 128:
                return False
            
            # Check if it's valid hex
            int(signature, 16)
            return True
            
        except ValueError:
            return False
    
    def _validate_public_key_format(self, public_key: str) -> bool:
        """Validate public key format."""
        try:
            # Check if public key is hex string
            if not public_key or len(public_key) != 66:
                return False
            
            # Check if it starts with 0x
            if not public_key.startswith('0x'):
                return False
            
            # Check if it's valid hex
            int(public_key[2:], 16)
            return True
            
        except ValueError:
            return False

class AccessController:
    """Manages access control for state channels."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize access controller."""
        self.config = config
        self.access_rules: Dict[str, List[AccessControl]] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_addresses: Set[str] = set()
        logger.info("Initialized access controller")
    
    def grant_access(self, channel_id: str, address: str, access_level: AccessLevel, permissions: Set[str] = None, expires_at: Optional[int] = None) -> bool:
        """Grant access to a channel."""
        try:
            if not self.config.enable_access_control:
                return True
            
            # Check if address is locked
            if address in self.locked_addresses:
                logger.error(f"Address {address} is locked")
                return False
            
            # Create access control rule
            access_control = AccessControl(
                channel_id=channel_id,
                address=address,
                access_level=access_level,
                permissions=permissions or set(),
                expires_at=expires_at
            )
            
            # Store access rule
            if channel_id not in self.access_rules:
                self.access_rules[channel_id] = []
            
            self.access_rules[channel_id].append(access_control)
            
            logger.info(f"Granted {access_level.value} access to {address} for channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error granting access: {e}")
            return False
    
    def revoke_access(self, channel_id: str, address: str) -> bool:
        """Revoke access to a channel."""
        try:
            if channel_id not in self.access_rules:
                return True
            
            # Remove access rules for address
            self.access_rules[channel_id] = [
                rule for rule in self.access_rules[channel_id] 
                if rule.address != address
            ]
            
            logger.info(f"Revoked access for {address} from channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking access: {e}")
            return False
    
    def check_access(self, channel_id: str, address: str, required_level: AccessLevel) -> bool:
        """Check if address has required access level."""
        try:
            if not self.config.enable_access_control:
                return True
            
            # Check if address is locked
            if address in self.locked_addresses:
                logger.error(f"Address {address} is locked")
                return False
            
            if channel_id not in self.access_rules:
                logger.error(f"No access rules for channel {channel_id}")
                return False
            
            # Find access rules for address
            current_time = int(time.time())
            access_rules = [
                rule for rule in self.access_rules[channel_id]
                if rule.address == address and 
                   (rule.expires_at is None or rule.expires_at > current_time)
            ]
            
            if not access_rules:
                logger.error(f"No valid access rules for {address}")
                return False
            
            # Check if any rule has required level or higher
            access_levels = [AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN, AccessLevel.OWNER]
            required_index = access_levels.index(required_level)
            
            for rule in access_rules:
                rule_index = access_levels.index(rule.access_level)
                if rule_index >= required_index:
                    return True
            
            logger.error(f"Insufficient access level for {address}")
            return False
            
        except Exception as e:
            logger.error(f"Error checking access: {e}")
            return False
    
    def record_failed_attempt(self, address: str) -> None:
        """Record a failed access attempt."""
        try:
            if address not in self.failed_attempts:
                self.failed_attempts[address] = 0
            
            self.failed_attempts[address] += 1
            
            # Lock address if too many failed attempts
            if self.failed_attempts[address] >= self.config.max_failed_attempts:
                self.locked_addresses.add(address)
                logger.warning(f"Locked address {address} due to {self.failed_attempts[address]} failed attempts")
            
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
    
    def unlock_address(self, address: str) -> bool:
        """Unlock a locked address."""
        try:
            if address in self.locked_addresses:
                self.locked_addresses.remove(address)
                self.failed_attempts[address] = 0
                logger.info(f"Unlocked address {address}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unlocking address: {e}")
            return False

class ThreatDetector:
    """Detects security threats."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize threat detector."""
        self.config = config
        self.security_events: List[SecurityEvent] = []
        logger.info("Initialized threat detector")
    
    def detect_threat(self, channel_id: str, event_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect security threats."""
        try:
            if not self.config.enable_threat_detection:
                return None
            
            # Analyze event data for threats
            threat_type = self._analyze_event(event_data)
            if not threat_type:
                return None
            
            # Determine security level
            security_level = self._determine_security_level(threat_type, event_data)
            
            # Create security event
            event = SecurityEvent(
                event_id=f"security_event_{int(time.time())}",
                channel_id=channel_id,
                threat_type=threat_type,
                security_level=security_level,
                description=self._create_description(threat_type, event_data),
                timestamp=int(time.time()),
                source_address=event_data.get('source_address'),
                target_address=event_data.get('target_address'),
                metadata=event_data
            )
            
            # Store event
            self.security_events.append(event)
            
            logger.warning(f"Security threat detected: {threat_type.value} in channel {channel_id}")
            return event
            
        except Exception as e:
            logger.error(f"Error detecting threat: {e}")
            return None
    
    def _analyze_event(self, event_data: Dict[str, Any]) -> Optional[ThreatType]:
        """Analyze event data for threats."""
        try:
            # Check for invalid signature
            if 'signature' in event_data and 'public_key' in event_data:
                if not self._validate_signature_format(event_data['signature']):
                    return ThreatType.INVALID_SIGNATURE
            
            # Check for balance manipulation
            if 'balance_change' in event_data:
                balance_change = event_data['balance_change']
                if abs(balance_change) > 1000000:  # Large balance change
                    return ThreatType.BALANCE_MANIPULATION
            
            # Check for replay attack
            if 'nonce' in event_data:
                nonce = event_data['nonce']
                if isinstance(nonce, str) and len(nonce) < 8:
                    return ThreatType.REPLAY_ATTACK
            
            # Check for double spending
            if 'payment_amount' in event_data and 'balance' in event_data:
                payment_amount = event_data['payment_amount']
                balance = event_data['balance']
                if payment_amount > balance:
                    return ThreatType.DOUBLE_SPENDING
            
            # Check for unauthorized access
            if 'access_denied' in event_data:
                return ThreatType.UNAUTHORIZED_ACCESS
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            return None
    
    def _determine_security_level(self, threat_type: ThreatType, event_data: Dict[str, Any]) -> SecurityLevel:
        """Determine security level based on threat type."""
        try:
            if threat_type == ThreatType.FRAUD:
                return SecurityLevel.CRITICAL
            elif threat_type in [ThreatType.BALANCE_MANIPULATION, ThreatType.DOUBLE_SPENDING]:
                return SecurityLevel.HIGH
            elif threat_type in [ThreatType.INVALID_SIGNATURE, ThreatType.UNAUTHORIZED_ACCESS]:
                return SecurityLevel.MEDIUM
            else:
                return SecurityLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining security level: {e}")
            return SecurityLevel.LOW
    
    def _create_description(self, threat_type: ThreatType, event_data: Dict[str, Any]) -> str:
        """Create description for security event."""
        try:
            descriptions = {
                ThreatType.UNAUTHORIZED_ACCESS: "Unauthorized access attempt detected",
                ThreatType.INVALID_SIGNATURE: "Invalid signature detected",
                ThreatType.BALANCE_MANIPULATION: "Suspicious balance manipulation detected",
                ThreatType.REPLAY_ATTACK: "Potential replay attack detected",
                ThreatType.DOUBLE_SPENDING: "Double spending attempt detected",
                ThreatType.FRAUD: "Fraudulent activity detected",
                ThreatType.MALICIOUS_BEHAVIOR: "Malicious behavior detected"
            }
            
            base_description = descriptions.get(threat_type, "Security threat detected")
            
            # Add specific details
            if 'source_address' in event_data:
                base_description += f" from {event_data['source_address']}"
            
            if 'amount' in event_data:
                base_description += f" involving {event_data['amount']} tokens"
            
            return base_description
            
        except Exception as e:
            logger.error(f"Error creating description: {e}")
            return "Security threat detected"
    
    def _validate_signature_format(self, signature: str) -> bool:
        """Validate signature format."""
        try:
            return signature and len(signature) == 128 and all(c in '0123456789abcdef' for c in signature.lower())
        except:
            return False

class SecurityMonitor:
    """Monitors security events and generates alerts."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize security monitor."""
        self.config = config
        self.threat_detector = ThreatDetector(config)
        self.alerts: List[Dict[str, Any]] = []
        logger.info("Initialized security monitor")
    
    def monitor_event(self, channel_id: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Monitor an event for security threats."""
        try:
            if not self.config.enable_security_monitoring:
                return None
            
            # Detect threats
            security_event = self.threat_detector.detect_threat(channel_id, event_data)
            if not security_event:
                return None
            
            # Generate alert
            alert = self._generate_alert(security_event)
            if alert:
                self.alerts.append(alert)
                logger.warning(f"Security alert generated: {alert['alert_type']}")
            
            return alert
            
        except Exception as e:
            logger.error(f"Error monitoring event: {e}")
            return None
    
    def _generate_alert(self, security_event: SecurityEvent) -> Optional[Dict[str, Any]]:
        """Generate security alert."""
        try:
            alert = {
                'alert_id': f"alert_{security_event.event_id}",
                'channel_id': security_event.channel_id,
                'alert_type': security_event.threat_type.value,
                'security_level': security_event.security_level.value,
                'description': security_event.description,
                'timestamp': security_event.timestamp,
                'source_address': security_event.source_address,
                'target_address': security_event.target_address,
                'metadata': security_event.metadata
            }
            
            return alert
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            return None
    
    def get_alerts(self, channel_id: Optional[str] = None, security_level: Optional[SecurityLevel] = None) -> List[Dict[str, Any]]:
        """Get security alerts."""
        try:
            alerts = self.alerts.copy()
            
            if channel_id:
                alerts = [a for a in alerts if a['channel_id'] == channel_id]
            
            if security_level:
                alerts = [a for a in alerts if a['security_level'] == security_level.value]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def cleanup_old_alerts(self, max_age: int = 86400) -> int:
        """Clean up old alerts."""
        try:
            current_time = int(time.time())
            old_alerts = [a for a in self.alerts if current_time - a['timestamp'] > max_age]
            
            for alert in old_alerts:
                self.alerts.remove(alert)
            
            logger.info(f"Cleaned up {len(old_alerts)} old alerts")
            return len(old_alerts)
            
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")
            return 0

class SecurityManager:
    """Main security management system."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize security manager."""
        self.config = config
        self.signature_validator = SignatureValidator(config)
        self.access_controller = AccessController(config)
        self.security_monitor = SecurityMonitor(config)
        logger.info("Initialized security manager")
    
    def validate_access(self, channel_id: str, address: str, required_level: AccessLevel) -> bool:
        """Validate access to a channel."""
        try:
            # Check access control
            if not self.access_controller.check_access(channel_id, address, required_level):
                # Record failed attempt
                self.access_controller.record_failed_attempt(address)
                
                # Monitor for security threats
                self.security_monitor.monitor_event(channel_id, {
                    'access_denied': True,
                    'source_address': address,
                    'required_level': required_level.value
                })
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating access: {e}")
            return False
    
    def validate_signature(self, message: str, signature: str, public_key: str, nonce: Optional[str] = None) -> bool:
        """Validate cryptographic signature."""
        try:
            # Validate signature
            if not self.signature_validator.validate_signature(message, signature, public_key, nonce):
                # Monitor for security threats
                self.security_monitor.monitor_event("unknown", {
                    'invalid_signature': True,
                    'signature': signature,
                    'public_key': public_key
                })
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signature: {e}")
            return False
    
    def monitor_security_event(self, channel_id: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Monitor a security event."""
        return self.security_monitor.monitor_event(channel_id, event_data)
    
    def grant_access(self, channel_id: str, address: str, access_level: AccessLevel, permissions: Set[str] = None, expires_at: Optional[int] = None) -> bool:
        """Grant access to a channel."""
        return self.access_controller.grant_access(channel_id, address, access_level, permissions, expires_at)
    
    def revoke_access(self, channel_id: str, address: str) -> bool:
        """Revoke access to a channel."""
        return self.access_controller.revoke_access(channel_id, address)
    
    def get_security_alerts(self, channel_id: Optional[str] = None, security_level: Optional[SecurityLevel] = None) -> List[Dict[str, Any]]:
        """Get security alerts."""
        return self.security_monitor.get_alerts(channel_id, security_level)
    
    def cleanup_security_data(self) -> int:
        """Clean up old security data."""
        try:
            cleaned_count = 0
            
            # Clean up old alerts
            cleaned_count += self.security_monitor.cleanup_old_alerts()
            
            # Clean up old nonces (keep last 1000)
            if len(self.signature_validator.used_nonces) > 1000:
                old_nonces = list(self.signature_validator.used_nonces)[:-1000]
                for nonce in old_nonces:
                    self.signature_validator.used_nonces.discard(nonce)
                cleaned_count += len(old_nonces)
            
            logger.info(f"Cleaned up {cleaned_count} security data entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up security data: {e}")
            return 0

@dataclass
class FraudProof:
    """Fraud proof for security events."""
    proof_id: str
    channel_id: str
    fraudulent_activity: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    submitter: str
    timestamp: int
    signature: Optional[str] = None

class ChannelSecurity:
    """Channel security manager."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize channel security."""
        self.config = config
        self.security_manager = SecurityManager(config)
        logger.info("Initialized channel security")
    
    def validate_channel_access(self, channel_id: str, address: str) -> bool:
        """Validate access to a channel."""
        return self.security_manager.validate_access(channel_id, address, AccessLevel.READ)
    
    def validate_channel_signature(self, message: str, signature: str, public_key: str) -> bool:
        """Validate channel signature."""
        return self.security_manager.validate_signature(message, signature, public_key)

class TimeoutManager:
    """Manages timeouts for state channels."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize timeout manager."""
        self.config = config
        self.timeouts: Dict[str, int] = {}
        logger.info("Initialized timeout manager")
    
    def set_timeout(self, channel_id: str, timeout_duration: int) -> None:
        """Set timeout for a channel."""
        self.timeouts[channel_id] = int(time.time()) + timeout_duration
    
    def check_timeout(self, channel_id: str) -> bool:
        """Check if channel has timed out."""
        if channel_id not in self.timeouts:
            return False
        
        return int(time.time()) > self.timeouts[channel_id]
    
    def clear_timeout(self, channel_id: str) -> None:
        """Clear timeout for a channel."""
        if channel_id in self.timeouts:
            del self.timeouts[channel_id]

@dataclass
class CryptographicProof:
    """Cryptographic proof for security validation."""
    proof_id: str
    proof_type: str
    data: Dict[str, Any]
    signature: str
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)

__all__ = [
    "SecurityManager",
    "SignatureValidator",
    "AccessController",
    "ThreatDetector",
    "SecurityMonitor",
    "SecurityEvent",
    "AccessControl",
    "SecurityConfig",
    "SecurityLevel",
    "ThreatType",
    "SecurityThreat",
    "AccessLevel",
    "FraudProof",
    "ChannelSecurity",
    "TimeoutManager",
    "CryptographicProof",
]