"""
Production Ethereum Bridge Manager

This module provides a production-ready Ethereum bridge manager that integrates
all security features, monitoring, and validation for enterprise-grade operations.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import threading
from collections import defaultdict, deque

from .client import EthereumClient, EthereumConfig, EthereumTransaction
from .monitoring import EthereumMonitoringService, MonitoringConfig
from .security import (
    EthereumBridgeSecurity,
    SecurityConfig,
    SecurityLevel,
    ThreatType,
    TransactionValidation,
)
from .contracts import ERC20Contract, ERC721Contract, BridgeContract
from ....errors import BridgeError, ValidationError
from ....logging import get_logger

logger = get_logger(__name__)

class BridgeStatus(Enum):
    """Bridge status states."""
    ACTIVE = "active"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class TransactionStatus(Enum):
    """Transaction status states."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"

@dataclass
class BridgeTransaction:
    """Bridge transaction data."""
    transaction_id: str
    source_chain: str
    target_chain: str
    from_address: str
    to_address: str
    amount: int
    token_address: Optional[str] = None
    transaction_hash: Optional[str] = None
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    confirmed_at: Optional[float] = None
    failed_at: Optional[float] = None
    failure_reason: Optional[str] = None
    gas_used: Optional[int] = None
    gas_price: Optional[int] = None
    block_number: Optional[int] = None
    validation_result: Optional[TransactionValidation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BridgeMetrics:
    """Bridge performance metrics."""
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    total_volume: int = 0
    average_gas_price: float = 0.0
    average_confirmation_time: float = 0.0
    security_events: int = 0
    last_updated: float = field(default_factory=time.time)

@dataclass
class ProductionBridgeConfig:
    """Production bridge configuration."""
    ethereum_config: EthereumConfig
    security_config: SecurityConfig
    monitoring_config: MonitoringConfig
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 5.0
    confirmation_blocks: int = 12
    max_pending_transactions: int = 1000
    transaction_timeout: float = 3600.0  # 1 hour
    enable_metrics_collection: bool = True
    metrics_update_interval: float = 60.0

class ProductionEthereumBridge:
    """Production-ready Ethereum bridge manager."""
    
    def __init__(self, config: ProductionBridgeConfig):
        """Initialize production bridge."""
        self.config = config
        self.ethereum_client = EthereumClient(config.ethereum_config)
        self.security_manager = EthereumBridgeSecurity(config.security_config)
        self.monitoring_service: Optional[EthereumMonitoringService] = None
        
        # Transaction management
        self.pending_transactions: Dict[str, BridgeTransaction] = {}
        self.completed_transactions: Dict[str, BridgeTransaction] = {}
        self.failed_transactions: Dict[str, BridgeTransaction] = {}
        
        # Metrics and monitoring
        self.metrics = BridgeMetrics()
        self.status = BridgeStatus.ACTIVE
        self._running = False
        self._lock = threading.RLock()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized production Ethereum bridge")
    
    async def start(self) -> None:
        """Start the bridge."""
        try:
            if self._running:
                logger.warning("Bridge is already running")
                return
            
            # Start security manager
            await self.security_manager.start()
            
            # Initialize monitoring service
            if self.ethereum_client.web3:
                self.monitoring_service = EthereumMonitoringService(
                    self.ethereum_client.web3,
                    self.config.monitoring_config
                )
                await self.monitoring_service.start()
            
            # Start background tasks
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.status = BridgeStatus.ACTIVE
            logger.info("Production Ethereum bridge started")
            
        except Exception as e:
            logger.error(f"Error starting bridge: {e}")
            self.status = BridgeStatus.ERROR
            raise BridgeError(f"Failed to start bridge: {e}")
    
    async def stop(self) -> None:
        """Stop the bridge."""
        try:
            if not self._running:
                logger.warning("Bridge is not running")
                return
            
            self._running = False
            
            # Stop background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self._metrics_task:
                self._metrics_task.cancel()
                try:
                    await self._metrics_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Stop services
            await self.security_manager.stop()
            if self.monitoring_service:
                await self.monitoring_service.stop()
            
            self.status = BridgeStatus.PAUSED
            logger.info("Production Ethereum bridge stopped")
            
        except Exception as e:
            logger.error(f"Error stopping bridge: {e}")
            self.status = BridgeStatus.ERROR
    
    async def create_transaction(
        self,
        source_chain: str,
        target_chain: str,
        from_address: str,
        to_address: str,
        amount: int,
        token_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new bridge transaction."""
        try:
            # Check bridge status
            if self.status != BridgeStatus.ACTIVE:
                raise BridgeError(f"Bridge is not active. Status: {self.status.value}")
            
            # Generate transaction ID
            transaction_id = f"bridge_tx_{int(time.time())}_{hash(f'{from_address}{to_address}{amount}') % 10000}"
            
            # Create bridge transaction
            bridge_tx = BridgeTransaction(
                transaction_id=transaction_id,
                source_chain=source_chain,
                target_chain=target_chain,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                token_address=token_address,
                metadata=metadata or {}
            )
            
            # Validate transaction
            validation = await self.security_manager.validate_transaction({
                'from': from_address,
                'to': to_address,
                'value': amount,
                'gasPrice': 0,  # Will be set later
                'gasLimit': 21000
            })
            
            bridge_tx.validation_result = validation
            
            if not validation.is_valid:
                bridge_tx.status = TransactionStatus.REJECTED
                bridge_tx.failed_at = time.time()
                bridge_tx.failure_reason = f"Validation failed: {', '.join(validation.warnings)}"
                
                with self._lock:
                    self.failed_transactions[transaction_id] = bridge_tx
                
                logger.warning(f"Transaction {transaction_id} rejected: {bridge_tx.failure_reason}")
                raise ValidationError(f"Transaction validation failed: {bridge_tx.failure_reason}")
            
            # Add to pending transactions
            with self._lock:
                if len(self.pending_transactions) >= self.config.max_pending_transactions:
                    raise BridgeError("Too many pending transactions")
                
                self.pending_transactions[transaction_id] = bridge_tx
            
            logger.info(f"Created bridge transaction {transaction_id}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            raise BridgeError(f"Failed to create transaction: {e}")
    
    async def execute_transaction(self, transaction_id: str, private_key: str) -> str:
        """Execute a bridge transaction."""
        try:
            # Get transaction
            with self._lock:
                if transaction_id not in self.pending_transactions:
                    raise BridgeError(f"Transaction {transaction_id} not found")
                
                bridge_tx = self.pending_transactions[transaction_id]
            
            # Check if bridge is paused
            if self.security_manager.emergency_pause_manager.is_bridge_paused():
                raise BridgeError("Bridge is paused due to security concerns")
            
            # Execute transaction
            if bridge_tx.token_address:
                # ERC20 token transfer
                tx_hash = await self._execute_token_transfer(bridge_tx, private_key)
            else:
                # Native ETH transfer
                tx_hash = await self._execute_native_transfer(bridge_tx, private_key)
            
            # Update transaction
            bridge_tx.transaction_hash = tx_hash
            bridge_tx.status = TransactionStatus.PENDING
            
            logger.info(f"Executed transaction {transaction_id} with hash {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error executing transaction {transaction_id}: {e}")
            
            # Mark transaction as failed
            with self._lock:
                if transaction_id in self.pending_transactions:
                    bridge_tx = self.pending_transactions[transaction_id]
                    bridge_tx.status = TransactionStatus.FAILED
                    bridge_tx.failed_at = time.time()
                    bridge_tx.failure_reason = str(e)
                    
                    self.failed_transactions[transaction_id] = bridge_tx
                    del self.pending_transactions[transaction_id]
            
            raise BridgeError(f"Failed to execute transaction: {e}")
    
    async def _execute_native_transfer(self, bridge_tx: BridgeTransaction, private_key: str) -> str:
        """Execute native ETH transfer."""
        try:
            # Build transaction
            transaction = {
                'from': bridge_tx.from_address,
                'to': bridge_tx.to_address,
                'value': bridge_tx.amount,
                'gas': 21000,
                'gasPrice': self.ethereum_client.get_optimized_gas_price(),
                'nonce': self.ethereum_client.get_nonce(bridge_tx.from_address),
            }
            
            # Sign and send transaction
            tx_hash = self.ethereum_client.send_transaction(transaction, private_key)
            
            # Update transaction with gas info
            bridge_tx.gas_price = transaction['gasPrice']
            bridge_tx.gas_used = 21000
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error executing native transfer: {e}")
            raise
    
    async def _execute_token_transfer(self, bridge_tx: BridgeTransaction, private_key: str) -> str:
        """Execute ERC20 token transfer."""
        try:
            # Create ERC20 contract instance
            contract = ERC20Contract(
                self.ethereum_client,
                bridge_tx.token_address
            )
            
            # Execute transfer
            tx_hash = contract.transfer(
                bridge_tx.to_address,
                bridge_tx.amount,
                bridge_tx.from_address,
                private_key
            )
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error executing token transfer: {e}")
            raise
    
    async def _monitoring_loop(self) -> None:
        """Monitor pending transactions."""
        while self._running:
            try:
                await self._check_transaction_status()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_transaction_status(self) -> None:
        """Check status of pending transactions."""
        try:
            current_time = time.time()
            confirmed_transactions = []
            
            with self._lock:
                for transaction_id, bridge_tx in list(self.pending_transactions.items()):
                    if not bridge_tx.transaction_hash:
                        continue
                    
                    try:
                        # Get transaction receipt
                        receipt = self.ethereum_client.get_transaction_receipt(bridge_tx.transaction_hash)
                        
                        if receipt:
                            if receipt.status == 1:  # Success
                                bridge_tx.status = TransactionStatus.CONFIRMED
                                bridge_tx.confirmed_at = current_time
                                bridge_tx.block_number = receipt.blockNumber
                                bridge_tx.gas_used = receipt.gasUsed
                                
                                confirmed_transactions.append(transaction_id)
                                
                                # Move to completed
                                self.completed_transactions[transaction_id] = bridge_tx
                                del self.pending_transactions[transaction_id]
                                
                                logger.info(f"Transaction {transaction_id} confirmed")
                                
                            else:  # Failed
                                bridge_tx.status = TransactionStatus.FAILED
                                bridge_tx.failed_at = current_time
                                bridge_tx.failure_reason = "Transaction failed on-chain"
                                
                                self.failed_transactions[transaction_id] = bridge_tx
                                del self.pending_transactions[transaction_id]
                                
                                logger.warning(f"Transaction {transaction_id} failed")
                        
                        # Check timeout
                        elif current_time - bridge_tx.created_at > self.config.transaction_timeout:
                            bridge_tx.status = TransactionStatus.FAILED
                            bridge_tx.failed_at = current_time
                            bridge_tx.failure_reason = "Transaction timeout"
                            
                            self.failed_transactions[transaction_id] = bridge_tx
                            del self.pending_transactions[transaction_id]
                            
                            logger.warning(f"Transaction {transaction_id} timed out")
                    
                    except Exception as e:
                        logger.error(f"Error checking transaction {transaction_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error checking transaction status: {e}")
    
    async def _metrics_loop(self) -> None:
        """Update bridge metrics."""
        while self._running:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.config.metrics_update_interval)
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self) -> None:
        """Update bridge metrics."""
        try:
            with self._lock:
                # Count transactions
                total_transactions = len(self.completed_transactions) + len(self.failed_transactions)
                successful_transactions = len(self.completed_transactions)
                failed_transactions = len(self.failed_transactions)
                
                # Calculate total volume
                total_volume = sum(tx.amount for tx in self.completed_transactions.values())
                
                # Calculate average gas price
                gas_prices = [tx.gas_price for tx in self.completed_transactions.values() if tx.gas_price]
                average_gas_price = sum(gas_prices) / len(gas_prices) if gas_prices else 0.0
                
                # Calculate average confirmation time
                confirmation_times = []
                for tx in self.completed_transactions.values():
                    if tx.confirmed_at and tx.created_at:
                        confirmation_times.append(tx.confirmed_at - tx.created_at)
                
                average_confirmation_time = sum(confirmation_times) / len(confirmation_times) if confirmation_times else 0.0
                
                # Get security events count
                security_events = len(self.security_manager.fraud_detector.get_suspicious_activities())
                
                # Update metrics
                self.metrics = BridgeMetrics(
                    total_transactions=total_transactions,
                    successful_transactions=successful_transactions,
                    failed_transactions=failed_transactions,
                    total_volume=total_volume,
                    average_gas_price=average_gas_price,
                    average_confirmation_time=average_confirmation_time,
                    security_events=security_events,
                    last_updated=time.time()
                )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old transactions."""
        while self._running:
            try:
                await self._cleanup_old_transactions()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_transactions(self) -> None:
        """Cleanup old completed and failed transactions."""
        try:
            current_time = time.time()
            cleanup_threshold = current_time - 86400 * 7  # 7 days
            
            with self._lock:
                # Cleanup completed transactions
                old_completed = [
                    tx_id for tx_id, tx in self.completed_transactions.items()
                    if tx.created_at < cleanup_threshold
                ]
                
                for tx_id in old_completed:
                    del self.completed_transactions[tx_id]
                
                # Cleanup failed transactions
                old_failed = [
                    tx_id for tx_id, tx in self.failed_transactions.items()
                    if tx.created_at < cleanup_threshold
                ]
                
                for tx_id in old_failed:
                    del self.failed_transactions[tx_id]
                
                if old_completed or old_failed:
                    logger.info(f"Cleaned up {len(old_completed)} completed and {len(old_failed)} failed transactions")
            
        except Exception as e:
            logger.error(f"Error cleaning up transactions: {e}")
    
    def get_transaction(self, transaction_id: str) -> Optional[BridgeTransaction]:
        """Get transaction by ID."""
        with self._lock:
            if transaction_id in self.pending_transactions:
                return self.pending_transactions[transaction_id]
            elif transaction_id in self.completed_transactions:
                return self.completed_transactions[transaction_id]
            elif transaction_id in self.failed_transactions:
                return self.failed_transactions[transaction_id]
            else:
                return None
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        with self._lock:
            return {
                "status": self.status.value,
                "pending_transactions": len(self.pending_transactions),
                "completed_transactions": len(self.completed_transactions),
                "failed_transactions": len(self.failed_transactions),
                "metrics": {
                    "total_transactions": self.metrics.total_transactions,
                    "successful_transactions": self.metrics.successful_transactions,
                    "failed_transactions": self.metrics.failed_transactions,
                    "total_volume": self.metrics.total_volume,
                    "average_gas_price": self.metrics.average_gas_price,
                    "average_confirmation_time": self.metrics.average_confirmation_time,
                    "security_events": self.metrics.security_events,
                    "last_updated": self.metrics.last_updated,
                },
                "security_status": self.security_manager.get_security_status(),
            }
    
    def pause_bridge(self, reason: str) -> None:
        """Pause bridge operations."""
        self.security_manager.pause_bridge(reason)
        self.status = BridgeStatus.PAUSED
        logger.critical(f"Bridge paused: {reason}")
    
    def resume_bridge(self) -> None:
        """Resume bridge operations."""
        self.security_manager.resume_bridge()
        self.status = BridgeStatus.ACTIVE
        logger.info("Bridge resumed")

__all__ = [
    "ProductionBridgeConfig",
    "BridgeTransaction",
    "BridgeMetrics",
    "BridgeStatus",
    "TransactionStatus",
    "ProductionEthereumBridge",
]
