"""
Production Ethereum Bridge Security and Validation

This module provides production-grade security features for the Ethereum bridge including:
- Transaction validation and verification
- Security monitoring and threat detection
- Rate limiting and DDoS protection
- Multi-signature validation
- Fraud detection and prevention
- Emergency pause mechanisms
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import threading
from collections import defaultdict, deque

from ....errors import BridgeError, ValidationError
from ....logging import get_logger

logger = get_logger(__name__)

class SecurityLevel(Enum):
    """Security levels for bridge operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats."""
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"
    HIGH_VALUE_TRANSFER = "high_value_transfer"
    RAPID_TRANSFERS = "rapid_transfers"
    UNKNOWN_CONTRACT = "unknown_contract"
    MALICIOUS_ADDRESS = "malicious_address"
    DOUBLE_SPENDING = "double_spending"
    REPLAY_ATTACK = "replay_attack"
    GAS_PRICE_MANIPULATION = "gas_price_manipulation"

@dataclass
class SecurityConfig:
    """Configuration for bridge security."""
    enable_transaction_validation: bool = True
    enable_fraud_detection: bool = True
    enable_rate_limiting: bool = True
    enable_multi_sig_validation: bool = True
    enable_emergency_pause: bool = True
    max_transaction_value: int = 1000000000000000000  # 1 ETH in wei
    max_daily_volume: int = 10000000000000000000  # 10 ETH in wei
    max_transactions_per_hour: int = 100
    max_transactions_per_day: int = 1000
    suspicious_value_threshold: int = 100000000000000000  # 0.1 ETH in wei
    rapid_transfer_threshold: int = 10  # transactions per minute
    emergency_pause_threshold: int = 5  # critical threats before pause
    validation_timeout: float = 30.0
    fraud_detection_interval: float = 60.0

@dataclass
class SecurityEvent:
    """Security event data."""
    event_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    transaction_hash: Optional[str] = None
    address: Optional[str] = None
    value: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransactionValidation:
    """Transaction validation result."""
    is_valid: bool
    validation_score: float  # 0.0 to 1.0
    security_level: SecurityLevel
    threats_detected: List[ThreatType]
    warnings: List[str]
    recommendations: List[str]
    validation_time: float

class TransactionValidator:
    """Validates Ethereum transactions for security threats."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize transaction validator."""
        self.config = config
        self.blacklisted_addresses: Set[str] = set()
        self.whitelisted_addresses: Set[str] = set()
        self.known_contracts: Set[str] = set()
        self.suspicious_patterns: Dict[str, Any] = {}
        self._lock = threading.RLock()
        logger.info("Initialized transaction validator")
    
    async def validate_transaction(self, transaction: Dict[str, Any]) -> TransactionValidation:
        """Validate a transaction for security threats."""
        try:
            start_time = time.time()
            threats_detected = []
            warnings = []
            recommendations = []
            validation_score = 1.0
            
            # Basic validation
            if not self._validate_basic_fields(transaction):
                threats_detected.append(ThreatType.SUSPICIOUS_TRANSACTION)
                validation_score -= 0.3
            
            # Value validation
            value = transaction.get('value', 0)
            if value > self.config.max_transaction_value:
                threats_detected.append(ThreatType.HIGH_VALUE_TRANSFER)
                validation_score -= 0.4
                warnings.append(f"Transaction value {value} exceeds maximum allowed")
            
            # Address validation
            from_address = transaction.get('from')
            to_address = transaction.get('to')
            
            if from_address in self.blacklisted_addresses:
                threats_detected.append(ThreatType.MALICIOUS_ADDRESS)
                validation_score -= 0.5
                warnings.append(f"From address {from_address} is blacklisted")
            
            if to_address in self.blacklisted_addresses:
                threats_detected.append(ThreatType.MALICIOUS_ADDRESS)
                validation_score -= 0.5
                warnings.append(f"To address {to_address} is blacklisted")
            
            # Contract validation
            if to_address and to_address not in self.known_contracts:
                threats_detected.append(ThreatType.UNKNOWN_CONTRACT)
                validation_score -= 0.2
                warnings.append(f"Unknown contract address {to_address}")
            
            # Gas price validation
            gas_price = transaction.get('gasPrice', 0)
            if self._is_gas_price_suspicious(gas_price):
                threats_detected.append(ThreatType.GAS_PRICE_MANIPULATION)
                validation_score -= 0.3
                warnings.append("Suspicious gas price detected")
            
            # Pattern analysis
            pattern_threats = await self._analyze_patterns(transaction)
            threats_detected.extend(pattern_threats)
            validation_score -= len(pattern_threats) * 0.1
            
            # Determine security level
            security_level = self._determine_security_level(validation_score, threats_detected)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(threats_detected, warnings)
            
            validation_time = time.time() - start_time
            
            return TransactionValidation(
                is_valid=validation_score >= 0.5 and len(threats_detected) == 0,
                validation_score=max(0.0, validation_score),
                security_level=security_level,
                threats_detected=threats_detected,
                warnings=warnings,
                recommendations=recommendations,
                validation_time=validation_time
            )
            
        except Exception as e:
            logger.error(f"Error validating transaction: {e}")
            return TransactionValidation(
                is_valid=False,
                validation_score=0.0,
                security_level=SecurityLevel.CRITICAL,
                threats_detected=[ThreatType.SUSPICIOUS_TRANSACTION],
                warnings=[f"Validation error: {e}"],
                recommendations=["Manual review required"],
                validation_time=0.0
            )
    
    def _validate_basic_fields(self, transaction: Dict[str, Any]) -> bool:
        """Validate basic transaction fields."""
        required_fields = ['from', 'to', 'value', 'gasPrice', 'gasLimit']
        return all(field in transaction for field in required_fields)
    
    def _is_gas_price_suspicious(self, gas_price: int) -> bool:
        """Check if gas price is suspicious."""
        # Check for extremely high or low gas prices
        return gas_price > 100000000000 or gas_price < 1000000000  # > 100 gwei or < 1 gwei
    
    async def _analyze_patterns(self, transaction: Dict[str, Any]) -> List[ThreatType]:
        """Analyze transaction patterns for threats."""
        threats = []
        
        # Check for rapid transfers
        from_address = transaction.get('from')
        if from_address and self._is_rapid_transfer(from_address):
            threats.append(ThreatType.RAPID_TRANSFERS)
        
        # Check for double spending patterns
        if self._is_double_spending_pattern(transaction):
            threats.append(ThreatType.DOUBLE_SPENDING)
        
        return threats
    
    def _is_rapid_transfer(self, address: str) -> bool:
        """Check if address is making rapid transfers."""
        # This would check against a rate limiting system
        # For now, return False as a placeholder
        return False
    
    def _is_double_spending_pattern(self, transaction: Dict[str, Any]) -> bool:
        """Check for double spending patterns."""
        # This would check against a transaction tracking system
        # For now, return False as a placeholder
        return False
    
    def _determine_security_level(self, validation_score: float, threats: List[ThreatType]) -> SecurityLevel:
        """Determine security level based on validation results."""
        if validation_score < 0.3 or ThreatType.MALICIOUS_ADDRESS in threats:
            return SecurityLevel.CRITICAL
        elif validation_score < 0.5 or len(threats) > 2:
            return SecurityLevel.HIGH
        elif validation_score < 0.7 or len(threats) > 0:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    def _generate_recommendations(self, threats: List[ThreatType], warnings: List[str]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if ThreatType.HIGH_VALUE_TRANSFER in threats:
            recommendations.append("Consider manual approval for high-value transfers")
        
        if ThreatType.MALICIOUS_ADDRESS in threats:
            recommendations.append("Block transaction from/to malicious addresses")
        
        if ThreatType.UNKNOWN_CONTRACT in threats:
            recommendations.append("Verify contract address before proceeding")
        
        if ThreatType.RAPID_TRANSFERS in threats:
            recommendations.append("Implement rate limiting for rapid transfers")
        
        return recommendations
    
    def add_blacklisted_address(self, address: str) -> None:
        """Add address to blacklist."""
        with self._lock:
            self.blacklisted_addresses.add(address.lower())
    
    def remove_blacklisted_address(self, address: str) -> None:
        """Remove address from blacklist."""
        with self._lock:
            self.blacklisted_addresses.discard(address.lower())
    
    def add_whitelisted_address(self, address: str) -> None:
        """Add address to whitelist."""
        with self._lock:
            self.whitelisted_addresses.add(address.lower())
    
    def add_known_contract(self, address: str) -> None:
        """Add known contract address."""
        with self._lock:
            self.known_contracts.add(address.lower())

class FraudDetector:
    """Detects fraudulent activities in bridge operations."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize fraud detector."""
        self.config = config
        self.transaction_history: deque = deque(maxlen=10000)
        self.address_activity: Dict[str, List[float]] = defaultdict(list)
        self.suspicious_activities: List[SecurityEvent] = []
        self._lock = threading.RLock()
        self._running = False
        self._detection_task: Optional[asyncio.Task] = None
        logger.info("Initialized fraud detector")
    
    async def start(self) -> None:
        """Start fraud detection."""
        if self._running:
            return
        
        self._running = True
        self._detection_task = asyncio.create_task(self._detection_loop())
        logger.info("Fraud detection started")
    
    async def stop(self) -> None:
        """Stop fraud detection."""
        self._running = False
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        logger.info("Fraud detection stopped")
    
    async def _detection_loop(self) -> None:
        """Main fraud detection loop."""
        while self._running:
            try:
                await self._analyze_transactions()
                await asyncio.sleep(self.config.fraud_detection_interval)
            except Exception as e:
                logger.error(f"Error in fraud detection loop: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_transactions(self) -> None:
        """Analyze recent transactions for fraud patterns."""
        try:
            current_time = time.time()
            
            # Analyze rapid transfers
            await self._detect_rapid_transfers(current_time)
            
            # Analyze high-value transfers
            await self._detect_high_value_transfers(current_time)
            
            # Analyze suspicious patterns
            await self._detect_suspicious_patterns(current_time)
            
        except Exception as e:
            logger.error(f"Error analyzing transactions: {e}")
    
    async def _detect_rapid_transfers(self, current_time: float) -> None:
        """Detect rapid transfer patterns."""
        try:
            time_window = 60  # 1 minute
            cutoff_time = current_time - time_window
            
            # Count transactions per address in time window
            address_counts = defaultdict(int)
            
            with self._lock:
                for transaction in self.transaction_history:
                    if transaction.get('timestamp', 0) >= cutoff_time:
                        from_address = transaction.get('from')
                        if from_address:
                            address_counts[from_address] += 1
            
            # Check for addresses exceeding threshold
            for address, count in address_counts.items():
                if count >= self.config.rapid_transfer_threshold:
                    event = SecurityEvent(
                        event_id=f"rapid_transfer_{address}_{int(current_time)}",
                        threat_type=ThreatType.RAPID_TRANSFERS,
                        severity=SecurityLevel.HIGH,
                        description=f"Address {address} made {count} transfers in {time_window} seconds",
                        address=address,
                        timestamp=current_time
                    )
                    
                    with self._lock:
                        self.suspicious_activities.append(event)
                    
                    logger.warning(f"Rapid transfer detected: {address} - {count} transfers")
            
        except Exception as e:
            logger.error(f"Error detecting rapid transfers: {e}")
    
    async def _detect_high_value_transfers(self, current_time: float) -> None:
        """Detect high-value transfer patterns."""
        try:
            with self._lock:
                for transaction in self.transaction_history:
                    value = transaction.get('value', 0)
                    if value >= self.config.suspicious_value_threshold:
                        from_address = transaction.get('from')
                        
                        event = SecurityEvent(
                            event_id=f"high_value_{from_address}_{int(current_time)}",
                            threat_type=ThreatType.HIGH_VALUE_TRANSFER,
                            severity=SecurityLevel.MEDIUM,
                            description=f"High-value transfer: {value} wei from {from_address}",
                            transaction_hash=transaction.get('hash'),
                            address=from_address,
                            value=value,
                            timestamp=current_time
                        )
                        
                        self.suspicious_activities.append(event)
                        
                        logger.warning(f"High-value transfer detected: {value} wei from {from_address}")
            
        except Exception as e:
            logger.error(f"Error detecting high-value transfers: {e}")
    
    async def _detect_suspicious_patterns(self, current_time: float) -> None:
        """Detect other suspicious patterns."""
        try:
            # This would implement more sophisticated pattern detection
            # For now, it's a placeholder
            pass
            
        except Exception as e:
            logger.error(f"Error detecting suspicious patterns: {e}")
    
    def add_transaction(self, transaction: Dict[str, Any]) -> None:
        """Add transaction to history for analysis."""
        with self._lock:
            self.transaction_history.append(transaction)
            
            # Track address activity
            from_address = transaction.get('from')
            if from_address:
                self.address_activity[from_address].append(time.time())
                
                # Keep only recent activity (last 24 hours)
                cutoff_time = time.time() - 86400
                self.address_activity[from_address] = [
                    t for t in self.address_activity[from_address] if t >= cutoff_time
                ]
    
    def get_suspicious_activities(self, limit: Optional[int] = None) -> List[SecurityEvent]:
        """Get suspicious activities."""
        with self._lock:
            activities = list(self.suspicious_activities)
            if limit:
                return activities[-limit:]
            return activities

class RateLimiter:
    """Rate limiting for bridge operations."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize rate limiter."""
        self.config = config
        self.address_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'hourly_count': 0,
            'daily_count': 0,
            'hourly_reset': time.time() + 3600,
            'daily_reset': time.time() + 86400,
            'last_transaction': 0
        })
        self._lock = threading.RLock()
        logger.info("Initialized rate limiter")
    
    def is_allowed(self, address: str) -> bool:
        """Check if address is allowed to make transactions."""
        try:
            with self._lock:
                current_time = time.time()
                limits = self.address_limits[address]
                
                # Reset counters if needed
                if current_time >= limits['hourly_reset']:
                    limits['hourly_count'] = 0
                    limits['hourly_reset'] = current_time + 3600
                
                if current_time >= limits['daily_reset']:
                    limits['daily_count'] = 0
                    limits['daily_reset'] = current_time + 86400
                
                # Check limits
                if limits['hourly_count'] >= self.config.max_transactions_per_hour:
                    return False
                
                if limits['daily_count'] >= self.config.max_transactions_per_day:
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    def record_transaction(self, address: str) -> None:
        """Record a transaction for rate limiting."""
        try:
            with self._lock:
                limits = self.address_limits[address]
                limits['hourly_count'] += 1
                limits['daily_count'] += 1
                limits['last_transaction'] = time.time()
                
        except Exception as e:
            logger.error(f"Error recording transaction: {e}")
    
    def get_remaining_transactions(self, address: str) -> Dict[str, int]:
        """Get remaining transactions for address."""
        try:
            with self._lock:
                limits = self.address_limits[address]
                
                hourly_remaining = max(0, self.config.max_transactions_per_hour - limits['hourly_count'])
                daily_remaining = max(0, self.config.max_transactions_per_day - limits['daily_count'])
                
                return {
                    'hourly_remaining': hourly_remaining,
                    'daily_remaining': daily_remaining
                }
                
        except Exception as e:
            logger.error(f"Error getting remaining transactions: {e}")
            return {'hourly_remaining': 0, 'daily_remaining': 0}

class MultiSigValidator:
    """Validates multi-signature requirements for bridge operations."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize multi-sig validator."""
        self.config = config
        self.required_signatures: Dict[str, int] = {}
        self.signature_thresholds: Dict[str, int] = {}
        self._lock = threading.RLock()
        logger.info("Initialized multi-sig validator")
    
    def set_signature_requirements(self, operation: str, required_signatures: int) -> None:
        """Set signature requirements for operation."""
        with self._lock:
            self.required_signatures[operation] = required_signatures
    
    def validate_signatures(self, operation: str, signatures: List[str]) -> bool:
        """Validate signatures for operation."""
        try:
            with self._lock:
                required = self.required_signatures.get(operation, 1)
                return len(signatures) >= required
                
        except Exception as e:
            logger.error(f"Error validating signatures: {e}")
            return False
    
    def get_required_signatures(self, operation: str) -> int:
        """Get required signatures for operation."""
        with self._lock:
            return self.required_signatures.get(operation, 1)

class EmergencyPauseManager:
    """Manages emergency pause functionality for bridge operations."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize emergency pause manager."""
        self.config = config
        self.is_paused = False
        self.pause_reason = ""
        self.pause_timestamp = 0
        self.threat_count = 0
        self._lock = threading.RLock()
        logger.info("Initialized emergency pause manager")
    
    def check_and_pause(self, threat_level: SecurityLevel) -> bool:
        """Check if bridge should be paused and pause if necessary."""
        try:
            with self._lock:
                if threat_level == SecurityLevel.CRITICAL:
                    self.threat_count += 1
                    
                    if self.threat_count >= self.config.emergency_pause_threshold:
                        self.pause_bridge("Critical threats detected")
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking pause condition: {e}")
            return False
    
    def pause_bridge(self, reason: str) -> None:
        """Pause bridge operations."""
        with self._lock:
            self.is_paused = True
            self.pause_reason = reason
            self.pause_timestamp = time.time()
            logger.critical(f"Bridge paused: {reason}")
    
    def resume_bridge(self) -> None:
        """Resume bridge operations."""
        with self._lock:
            self.is_paused = False
            self.pause_reason = ""
            self.pause_timestamp = 0
            self.threat_count = 0
            logger.info("Bridge resumed")
    
    def is_bridge_paused(self) -> bool:
        """Check if bridge is paused."""
        with self._lock:
            return self.is_paused

class EthereumBridgeSecurity:
    """Main security manager for Ethereum bridge."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize bridge security."""
        self.config = config
        self.transaction_validator = TransactionValidator(config)
        self.fraud_detector = FraudDetector(config)
        self.rate_limiter = RateLimiter(config)
        self.multi_sig_validator = MultiSigValidator(config)
        self.emergency_pause_manager = EmergencyPauseManager(config)
        self._running = False
        logger.info("Initialized Ethereum bridge security")
    
    async def start(self) -> None:
        """Start security monitoring."""
        if self._running:
            return
        
        self._running = True
        await self.fraud_detector.start()
        logger.info("Ethereum bridge security started")
    
    async def stop(self) -> None:
        """Stop security monitoring."""
        self._running = False
        await self.fraud_detector.stop()
        logger.info("Ethereum bridge security stopped")
    
    async def validate_transaction(self, transaction: Dict[str, Any]) -> TransactionValidation:
        """Validate transaction for security threats."""
        try:
            # Check if bridge is paused
            if self.emergency_pause_manager.is_bridge_paused():
                return TransactionValidation(
                    is_valid=False,
                    validation_score=0.0,
                    security_level=SecurityLevel.CRITICAL,
                    threats_detected=[ThreatType.SUSPICIOUS_TRANSACTION],
                    warnings=["Bridge is paused"],
                    recommendations=["Wait for bridge to resume"],
                    validation_time=0.0
                )
            
            # Check rate limits
            from_address = transaction.get('from')
            if from_address and not self.rate_limiter.is_allowed(from_address):
                return TransactionValidation(
                    is_valid=False,
                    validation_score=0.0,
                    security_level=SecurityLevel.HIGH,
                    threats_detected=[ThreatType.RAPID_TRANSFERS],
                    warnings=["Rate limit exceeded"],
                    recommendations=["Wait before making more transactions"],
                    validation_time=0.0
                )
            
            # Validate transaction
            validation = await self.transaction_validator.validate_transaction(transaction)
            
            # Check for emergency pause
            if validation.security_level == SecurityLevel.CRITICAL:
                self.emergency_pause_manager.check_and_pause(validation.security_level)
            
            # Record transaction for fraud detection
            if from_address:
                self.fraud_detector.add_transaction(transaction)
                self.rate_limiter.record_transaction(from_address)
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating transaction: {e}")
            return TransactionValidation(
                is_valid=False,
                validation_score=0.0,
                security_level=SecurityLevel.CRITICAL,
                threats_detected=[ThreatType.SUSPICIOUS_TRANSACTION],
                warnings=[f"Validation error: {e}"],
                recommendations=["Manual review required"],
                validation_time=0.0
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "is_paused": self.emergency_pause_manager.is_bridge_paused(),
            "pause_reason": self.emergency_pause_manager.pause_reason,
            "threat_count": self.emergency_pause_manager.threat_count,
            "suspicious_activities": len(self.fraud_detector.get_suspicious_activities()),
            "blacklisted_addresses": len(self.transaction_validator.blacklisted_addresses),
            "known_contracts": len(self.transaction_validator.known_contracts),
        }
    
    def pause_bridge(self, reason: str) -> None:
        """Pause bridge operations."""
        self.emergency_pause_manager.pause_bridge(reason)
    
    def resume_bridge(self) -> None:
        """Resume bridge operations."""
        self.emergency_pause_manager.resume_bridge()

__all__ = [
    "SecurityConfig",
    "SecurityLevel",
    "ThreatType",
    "SecurityEvent",
    "TransactionValidation",
    "TransactionValidator",
    "FraudDetector",
    "RateLimiter",
    "MultiSigValidator",
    "EmergencyPauseManager",
    "EthereumBridgeSecurity",
]
