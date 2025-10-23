"""
Bridge Core Enhancements: Validator Network, Fraud Detection, Relayer System, Analytics

This module provides comprehensive bridge core enhancements including:
- Validator network for decentralized bridge validation
- Fraud detection system for security monitoring
- Relayer system for transaction processing
- Analytics system for bridge performance monitoring
- Comprehensive security and monitoring
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import threading
from collections import defaultdict, deque
import secrets
import statistics

from ...errors import BridgeError, ValidationError
from ...logging import get_logger

logger = get_logger(__name__)

class ValidatorStatus(Enum):
    """Validator status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"

class FraudLevel(Enum):
    """Fraud detection levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RelayerStatus(Enum):
    """Relayer status states."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"

class TransactionStatus(Enum):
    """Transaction status states."""
    PENDING = "pending"
    PROCESSING = "processing"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BridgeCoreConfig:
    """Bridge core configuration."""
    validator_threshold: int = 3
    validator_timeout: int = 30
    fraud_detection_enabled: bool = True
    fraud_threshold: float = 0.8
    relayer_pool_size: int = 10
    relayer_timeout: int = 60
    analytics_enabled: bool = True
    analytics_interval: int = 60
    max_transaction_age: int = 3600  # 1 hour
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 10
    circuit_breaker_timeout: int = 300  # 5 minutes

@dataclass
class Validator:
    """Bridge validator."""
    validator_id: str
    address: str
    public_key: str
    stake: int
    status: ValidatorStatus = ValidatorStatus.ACTIVE
    reputation_score: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0

@dataclass
class FraudDetectionResult:
    """Fraud detection result."""
    detection_id: str
    transaction_id: str
    fraud_level: FraudLevel
    confidence: float
    reasons: List[str]
    detected_at: float = field(default_factory=time.time)
    action_taken: Optional[str] = None

@dataclass
class Relayer:
    """Bridge relayer."""
    relayer_id: str
    address: str
    status: RelayerStatus = RelayerStatus.ONLINE
    capacity: int = 100
    current_load: int = 0
    success_rate: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0

@dataclass
class BridgeTransaction:
    """Bridge transaction."""
    transaction_id: str
    source_chain: str
    target_chain: str
    amount: int
    token_address: str
    user_address: str
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    confirmed_at: Optional[float] = None
    failed_at: Optional[float] = None
    validator_signatures: List[str] = field(default_factory=list)
    relayer_id: Optional[str] = None
    fraud_detection_result: Optional[FraudDetectionResult] = None

@dataclass
class BridgeMetrics:
    """Bridge performance metrics."""
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    pending_transactions: int = 0
    average_processing_time: float = 0.0
    average_confirmation_time: float = 0.0
    fraud_detections: int = 0
    validator_uptime: float = 1.0
    relayer_uptime: float = 1.0
    last_updated: float = field(default_factory=time.time)

class ValidatorNetwork:
    """Decentralized validator network for bridge validation."""
    
    def __init__(self, config: BridgeCoreConfig):
        """Initialize validator network."""
        self.config = config
        self.validators: Dict[str, Validator] = {}
        self._lock = threading.RLock()
        logger.info("Initialized validator network")
    
    def add_validator(self, validator_id: str, address: str, public_key: str, stake: int) -> Validator:
        """Add a new validator to the network."""
        try:
            validator = Validator(
                validator_id=validator_id,
                address=address,
                public_key=public_key,
                stake=stake
            )
            
            with self._lock:
                self.validators[validator_id] = validator
            
            logger.info(f"Added validator {validator_id}")
            return validator
            
        except Exception as e:
            logger.error(f"Error adding validator: {e}")
            raise BridgeError(f"Failed to add validator: {e}")
    
    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator from the network."""
        try:
            with self._lock:
                if validator_id not in self.validators:
                    return False
                
                del self.validators[validator_id]
            
            logger.info(f"Removed validator {validator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing validator: {e}")
            return False
    
    def update_validator_status(self, validator_id: str, status: ValidatorStatus) -> bool:
        """Update validator status."""
        try:
            with self._lock:
                if validator_id not in self.validators:
                    return False
                
                self.validators[validator_id].status = status
                self.validators[validator_id].last_activity = time.time()
            
            logger.info(f"Updated validator {validator_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating validator status: {e}")
            return False
    
    def get_active_validators(self) -> List[Validator]:
        """Get list of active validators."""
        with self._lock:
            return [v for v in self.validators.values() if v.status == ValidatorStatus.ACTIVE]
    
    def select_validators(self, count: Optional[int] = None) -> List[Validator]:
        """Select validators for transaction validation."""
        try:
            active_validators = self.get_active_validators()
            
            if not active_validators:
                raise BridgeError("No active validators available")
            
            # Sort by reputation score and stake
            sorted_validators = sorted(
                active_validators,
                key=lambda v: (v.reputation_score, v.stake),
                reverse=True
            )
            
            count = count or self.config.validator_threshold
            selected = sorted_validators[:count]
            
            logger.info(f"Selected {len(selected)} validators for validation")
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting validators: {e}")
            raise BridgeError(f"Failed to select validators: {e}")
    
    def record_validation_result(self, validator_id: str, success: bool) -> None:
        """Record validation result for validator."""
        try:
            with self._lock:
                if validator_id not in self.validators:
                    return
                
                validator = self.validators[validator_id]
                validator.total_validations += 1
                
                if success:
                    validator.successful_validations += 1
                else:
                    validator.failed_validations += 1
                
                # Update reputation score
                success_rate = validator.successful_validations / validator.total_validations
                validator.reputation_score = success_rate
                
                validator.last_activity = time.time()
            
            logger.debug(f"Recorded validation result for validator {validator_id}: {success}")
            
        except Exception as e:
            logger.error(f"Error recording validation result: {e}")
    
    def get_validator(self, validator_id: str) -> Optional[Validator]:
        """Get validator by ID."""
        with self._lock:
            return self.validators.get(validator_id)
    
    def list_validators(self) -> List[Validator]:
        """List all validators."""
        with self._lock:
            return list(self.validators.values())

class FraudDetector:
    """Fraud detection system for bridge security."""
    
    def __init__(self, config: BridgeCoreConfig):
        """Initialize fraud detector."""
        self.config = config
        self.detection_history: Dict[str, FraudDetectionResult] = {}
        self.suspicious_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.RLock()
        logger.info("Initialized fraud detector")
    
    def detect_fraud(self, transaction: BridgeTransaction) -> Optional[FraudDetectionResult]:
        """Detect fraud in transaction."""
        try:
            if not self.config.fraud_detection_enabled:
                return None
            
            detection_id = f"fraud_{transaction.transaction_id}_{secrets.token_hex(8)}"
            reasons = []
            confidence = 0.0
            
            # Check for suspicious patterns
            if self._check_suspicious_amount(transaction):
                reasons.append("Suspicious transaction amount")
                confidence += 0.3
            
            if self._check_suspicious_frequency(transaction):
                reasons.append("Suspicious transaction frequency")
                confidence += 0.2
            
            if self._check_suspicious_address(transaction):
                reasons.append("Suspicious address pattern")
                confidence += 0.2
            
            if self._check_suspicious_timing(transaction):
                reasons.append("Suspicious transaction timing")
                confidence += 0.1
            
            if self._check_suspicious_chain_combination(transaction):
                reasons.append("Suspicious chain combination")
                confidence += 0.2
            
            # Determine fraud level
            if confidence >= 0.8:
                fraud_level = FraudLevel.CRITICAL
            elif confidence >= 0.6:
                fraud_level = FraudLevel.HIGH
            elif confidence >= 0.4:
                fraud_level = FraudLevel.MEDIUM
            elif confidence >= 0.2:
                fraud_level = FraudLevel.LOW
            else:
                return None  # No fraud detected
            
            # Create fraud detection result
            result = FraudDetectionResult(
                detection_id=detection_id,
                transaction_id=transaction.transaction_id,
                fraud_level=fraud_level,
                confidence=confidence,
                reasons=reasons
            )
            
            with self._lock:
                self.detection_history[detection_id] = result
            
            logger.warning(f"Fraud detected in transaction {transaction.transaction_id}: {fraud_level.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            return None
    
    def _check_suspicious_amount(self, transaction: BridgeTransaction) -> bool:
        """Check for suspicious transaction amounts."""
        # Check for unusually large amounts
        if transaction.amount > 1000000000000:  # 1M tokens
            return True
        
        # Check for round numbers (potential test transactions)
        if transaction.amount % 1000000 == 0 and transaction.amount > 1000000:
            return True
        
        return False
    
    def _check_suspicious_frequency(self, transaction: BridgeTransaction) -> bool:
        """Check for suspicious transaction frequency."""
        user_key = f"{transaction.user_address}_{transaction.source_chain}_{transaction.target_chain}"
        
        with self._lock:
            user_transactions = self.suspicious_patterns[user_key]
            
            # Check for rapid transactions
            recent_transactions = [
                tx for tx in user_transactions
                if time.time() - tx.get("timestamp", 0) < 300  # 5 minutes
            ]
            
            if len(recent_transactions) > 5:
                return True
        
        return False
    
    def _check_suspicious_address(self, transaction: BridgeTransaction) -> bool:
        """Check for suspicious address patterns."""
        # Check for known malicious addresses (simplified)
        malicious_addresses = {
            "0x0000000000000000000000000000000000000000",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222"
        }
        
        if transaction.user_address.lower() in malicious_addresses:
            return True
        
        return False
    
    def _check_suspicious_timing(self, transaction: BridgeTransaction) -> bool:
        """Check for suspicious transaction timing."""
        # Check for transactions at unusual hours (simplified)
        current_hour = time.localtime().tm_hour
        
        if current_hour < 6 or current_hour > 22:  # Night time
            return True
        
        return False
    
    def _check_suspicious_chain_combination(self, transaction: BridgeTransaction) -> bool:
        """Check for suspicious chain combinations."""
        # Check for unusual chain combinations
        suspicious_combinations = [
            ("bitcoin", "ethereum"),
            ("ethereum", "bitcoin"),
            ("polygon", "bitcoin")
        ]
        
        combination = (transaction.source_chain.lower(), transaction.target_chain.lower())
        return combination in suspicious_combinations
    
    def record_transaction(self, transaction: BridgeTransaction) -> None:
        """Record transaction for pattern analysis."""
        try:
            user_key = f"{transaction.user_address}_{transaction.source_chain}_{transaction.target_chain}"
            
            with self._lock:
                self.suspicious_patterns[user_key].append({
                    "transaction_id": transaction.transaction_id,
                    "amount": transaction.amount,
                    "timestamp": time.time()
                })
                
                # Keep only recent transactions
                cutoff_time = time.time() - 3600  # 1 hour
                self.suspicious_patterns[user_key] = [
                    tx for tx in self.suspicious_patterns[user_key]
                    if tx.get("timestamp", 0) > cutoff_time
                ]
            
            logger.debug(f"Recorded transaction {transaction.transaction_id} for pattern analysis")
            
        except Exception as e:
            logger.error(f"Error recording transaction: {e}")
    
    def get_detection_history(self) -> List[FraudDetectionResult]:
        """Get fraud detection history."""
        with self._lock:
            return list(self.detection_history.values())

class RelayerSystem:
    """Relayer system for transaction processing."""
    
    def __init__(self, config: BridgeCoreConfig):
        """Initialize relayer system."""
        self.config = config
        self.relayers: Dict[str, Relayer] = {}
        self.transaction_queue: deque = deque()
        self._lock = threading.RLock()
        self._running = False
        logger.info("Initialized relayer system")
    
    async def start(self) -> None:
        """Start relayer system."""
        try:
            if self._running:
                logger.warning("Relayer system is already running")
                return
            
            self._running = True
            
            # Start transaction processing loop
            asyncio.create_task(self._process_transactions())
            
            logger.info("Relayer system started")
            
        except Exception as e:
            logger.error(f"Error starting relayer system: {e}")
            raise BridgeError(f"Failed to start relayer system: {e}")
    
    async def stop(self) -> None:
        """Stop relayer system."""
        try:
            if not self._running:
                logger.warning("Relayer system is not running")
                return
            
            self._running = False
            logger.info("Relayer system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping relayer system: {e}")
    
    def add_relayer(self, relayer_id: str, address: str, capacity: int = 100) -> Relayer:
        """Add a new relayer."""
        try:
            relayer = Relayer(
                relayer_id=relayer_id,
                address=address,
                capacity=capacity
            )
            
            with self._lock:
                self.relayers[relayer_id] = relayer
            
            logger.info(f"Added relayer {relayer_id}")
            return relayer
            
        except Exception as e:
            logger.error(f"Error adding relayer: {e}")
            raise BridgeError(f"Failed to add relayer: {e}")
    
    def remove_relayer(self, relayer_id: str) -> bool:
        """Remove relayer."""
        try:
            with self._lock:
                if relayer_id not in self.relayers:
                    return False
                
                del self.relayers[relayer_id]
            
            logger.info(f"Removed relayer {relayer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing relayer: {e}")
            return False
    
    def update_relayer_status(self, relayer_id: str, status: RelayerStatus) -> bool:
        """Update relayer status."""
        try:
            with self._lock:
                if relayer_id not in self.relayers:
                    return False
                
                self.relayers[relayer_id].status = status
                self.relayers[relayer_id].last_activity = time.time()
            
            logger.info(f"Updated relayer {relayer_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating relayer status: {e}")
            return False
    
    def queue_transaction(self, transaction: BridgeTransaction) -> bool:
        """Queue transaction for processing."""
        try:
            with self._lock:
                self.transaction_queue.append(transaction)
            
            logger.info(f"Queued transaction {transaction.transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error queuing transaction: {e}")
            return False
    
    def select_relayer(self) -> Optional[Relayer]:
        """Select best relayer for transaction processing."""
        try:
            with self._lock:
                available_relayers = [
                    r for r in self.relayers.values()
                    if r.status == RelayerStatus.ONLINE and r.current_load < r.capacity
                ]
                
                if not available_relayers:
                    return None
                
                # Select relayer with lowest load and highest success rate
                best_relayer = min(
                    available_relayers,
                    key=lambda r: (r.current_load, -r.success_rate)
                )
                
                return best_relayer
            
        except Exception as e:
            logger.error(f"Error selecting relayer: {e}")
            return None
    
    async def _process_transactions(self) -> None:
        """Process queued transactions."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Process every second
                
                if not self._running:
                    break
                
                # Get next transaction
                with self._lock:
                    if not self.transaction_queue:
                        continue
                    
                    transaction = self.transaction_queue.popleft()
                
                # Select relayer
                relayer = self.select_relayer()
                if not relayer:
                    # No available relayer, requeue transaction
                    with self._lock:
                        self.transaction_queue.appendleft(transaction)
                    continue
                
                # Process transaction
                await self._process_transaction(transaction, relayer)
                
            except Exception as e:
                logger.error(f"Error in transaction processing loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_transaction(self, transaction: BridgeTransaction, relayer: Relayer) -> None:
        """Process transaction with relayer."""
        try:
            # Update transaction status
            transaction.status = TransactionStatus.PROCESSING
            transaction.relayer_id = relayer.relayer_id
            
            # Update relayer load
            with self._lock:
                relayer.current_load += 1
            
            # Simulate transaction processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Update transaction status
            transaction.status = TransactionStatus.CONFIRMED
            transaction.processed_at = time.time()
            transaction.confirmed_at = time.time()
            
            # Update relayer metrics
            with self._lock:
                relayer.current_load -= 1
                relayer.total_transactions += 1
                relayer.successful_transactions += 1
                relayer.success_rate = relayer.successful_transactions / relayer.total_transactions
                relayer.last_activity = time.time()
            
            logger.info(f"Processed transaction {transaction.transaction_id} with relayer {relayer.relayer_id}")
            
        except Exception as e:
            logger.error(f"Error processing transaction {transaction.transaction_id}: {e}")
            
            # Update transaction status
            transaction.status = TransactionStatus.FAILED
            transaction.failed_at = time.time()
            
            # Update relayer metrics
            with self._lock:
                relayer.current_load -= 1
                relayer.total_transactions += 1
                relayer.failed_transactions += 1
                relayer.success_rate = relayer.successful_transactions / relayer.total_transactions
    
    def get_relayer(self, relayer_id: str) -> Optional[Relayer]:
        """Get relayer by ID."""
        with self._lock:
            return self.relayers.get(relayer_id)
    
    def list_relayers(self) -> List[Relayer]:
        """List all relayers."""
        with self._lock:
            return list(self.relayers.values())

class BridgeAnalytics:
    """Analytics system for bridge performance monitoring."""
    
    def __init__(self, config: BridgeCoreConfig):
        """Initialize bridge analytics."""
        self.config = config
        self.metrics = BridgeMetrics()
        self.transaction_history: List[BridgeTransaction] = []
        self._lock = threading.RLock()
        self._running = False
        logger.info("Initialized bridge analytics")
    
    async def start(self) -> None:
        """Start analytics system."""
        try:
            if self._running:
                logger.warning("Analytics system is already running")
                return
            
            self._running = True
            
            # Start metrics collection loop
            asyncio.create_task(self._collect_metrics())
            
            logger.info("Analytics system started")
            
        except Exception as e:
            logger.error(f"Error starting analytics system: {e}")
            raise BridgeError(f"Failed to start analytics system: {e}")
    
    async def stop(self) -> None:
        """Stop analytics system."""
        try:
            if not self._running:
                logger.warning("Analytics system is not running")
                return
            
            self._running = False
            logger.info("Analytics system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping analytics system: {e}")
    
    def record_transaction(self, transaction: BridgeTransaction) -> None:
        """Record transaction for analytics."""
        try:
            with self._lock:
                self.transaction_history.append(transaction)
                
                # Keep only recent transactions
                cutoff_time = time.time() - 86400  # 24 hours
                self.transaction_history = [
                    tx for tx in self.transaction_history
                    if tx.created_at > cutoff_time
                ]
            
            logger.debug(f"Recorded transaction {transaction.transaction_id} for analytics")
            
        except Exception as e:
            logger.error(f"Error recording transaction: {e}")
    
    async def _collect_metrics(self) -> None:
        """Collect bridge metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.config.analytics_interval)
                
                if not self._running:
                    break
                
                await self._update_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_metrics(self) -> None:
        """Update bridge metrics."""
        try:
            with self._lock:
                # Count transactions by status
                total_transactions = len(self.transaction_history)
                successful_transactions = len([
                    tx for tx in self.transaction_history
                    if tx.status == TransactionStatus.CONFIRMED
                ])
                failed_transactions = len([
                    tx for tx in self.transaction_history
                    if tx.status == TransactionStatus.FAILED
                ])
                pending_transactions = len([
                    tx for tx in self.transaction_history
                    if tx.status == TransactionStatus.PENDING
                ])
                
                # Calculate average processing times
                confirmed_transactions = [
                    tx for tx in self.transaction_history
                    if tx.status == TransactionStatus.CONFIRMED and tx.processed_at
                ]
                
                if confirmed_transactions:
                    processing_times = [
                        tx.processed_at - tx.created_at
                        for tx in confirmed_transactions
                    ]
                    confirmation_times = [
                        tx.confirmed_at - tx.created_at
                        for tx in confirmed_transactions
                        if tx.confirmed_at
                    ]
                    
                    self.metrics.average_processing_time = statistics.mean(processing_times)
                    self.metrics.average_confirmation_time = statistics.mean(confirmation_times)
                
                # Update metrics
                self.metrics.total_transactions = total_transactions
                self.metrics.successful_transactions = successful_transactions
                self.metrics.failed_transactions = failed_transactions
                self.metrics.pending_transactions = pending_transactions
                self.metrics.last_updated = time.time()
            
            logger.debug("Updated bridge metrics")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_metrics(self) -> BridgeMetrics:
        """Get current bridge metrics."""
        with self._lock:
            return self.metrics
    
    def get_transaction_history(self) -> List[BridgeTransaction]:
        """Get transaction history."""
        with self._lock:
            return list(self.transaction_history)

class ProductionBridgeCore:
    """Production bridge core with validator network, fraud detection, relayer system, and analytics."""
    
    def __init__(self, config: BridgeCoreConfig):
        """Initialize production bridge core."""
        self.config = config
        self.validator_network = ValidatorNetwork(config)
        self.fraud_detector = FraudDetector(config)
        self.relayer_system = RelayerSystem(config)
        self.analytics = BridgeAnalytics(config)
        self._running = False
        logger.info("Initialized production bridge core")
    
    async def start(self) -> None:
        """Start bridge core."""
        try:
            if self._running:
                logger.warning("Bridge core is already running")
                return
            
            # Start subsystems
            await self.relayer_system.start()
            await self.analytics.start()
            
            self._running = True
            logger.info("Production bridge core started")
            
        except Exception as e:
            logger.error(f"Error starting bridge core: {e}")
            raise BridgeError(f"Failed to start bridge core: {e}")
    
    async def stop(self) -> None:
        """Stop bridge core."""
        try:
            if not self._running:
                logger.warning("Bridge core is not running")
                return
            
            # Stop subsystems
            await self.relayer_system.stop()
            await self.analytics.stop()
            
            self._running = False
            logger.info("Production bridge core stopped")
            
        except Exception as e:
            logger.error(f"Error stopping bridge core: {e}")
    
    async def process_transaction(self, transaction: BridgeTransaction) -> bool:
        """Process bridge transaction."""
        try:
            # Record transaction for analytics
            self.analytics.record_transaction(transaction)
            
            # Record transaction for fraud detection
            self.fraud_detector.record_transaction(transaction)
            
            # Detect fraud
            fraud_result = self.fraud_detector.detect_fraud(transaction)
            if fraud_result:
                transaction.fraud_detection_result = fraud_result
                
                if fraud_result.fraud_level in [FraudLevel.HIGH, FraudLevel.CRITICAL]:
                    logger.warning(f"High fraud risk detected for transaction {transaction.transaction_id}")
                    return False
            
            # Select validators
            validators = self.validator_network.select_validators()
            if not validators:
                logger.error("No validators available for transaction validation")
                return False
            
            # Queue transaction for processing
            success = self.relayer_system.queue_transaction(transaction)
            if not success:
                logger.error(f"Failed to queue transaction {transaction.transaction_id}")
                return False
            
            logger.info(f"Processing transaction {transaction.transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return False
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "running": self._running,
            "validators": len(self.validator_network.validators),
            "active_validators": len(self.validator_network.get_active_validators()),
            "relayers": len(self.relayer_system.relayers),
            "online_relayers": len([
                r for r in self.relayer_system.relayers.values()
                if r.status == RelayerStatus.ONLINE
            ]),
            "queued_transactions": len(self.relayer_system.transaction_queue),
            "fraud_detections": len(self.fraud_detector.detection_history),
            "metrics": self.analytics.get_metrics(),
            "config": {
                "validator_threshold": self.config.validator_threshold,
                "fraud_detection_enabled": self.config.fraud_detection_enabled,
                "relayer_pool_size": self.config.relayer_pool_size,
                "analytics_enabled": self.config.analytics_enabled,
            }
        }

__all__ = [
    "BridgeCoreConfig",
    "ValidatorStatus",
    "FraudLevel",
    "RelayerStatus",
    "TransactionStatus",
    "Validator",
    "FraudDetectionResult",
    "Relayer",
    "BridgeTransaction",
    "BridgeMetrics",
    "ValidatorNetwork",
    "FraudDetector",
    "RelayerSystem",
    "BridgeAnalytics",
    "ProductionBridgeCore",
]
