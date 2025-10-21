"""
Ethereum Event Monitoring and Gas Price Oracle

This module provides comprehensive monitoring of Ethereum events and real-time gas price estimation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import threading
from collections import defaultdict, deque
import statistics

from web3 import Web3
from web3.types import FilterParams, LogReceipt, BlockData, TxReceipt
from web3.exceptions import BlockNotFound, TransactionNotFound

from ....errors import BridgeError, MonitoringError
from dubchain.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GasPriceData:
    """Gas price data structure"""
    slow: int
    standard: int
    fast: int
    instant: int
    timestamp: datetime
    block_number: int
    base_fee: Optional[int] = None  # EIP-1559 base fee
    priority_fee: Optional[int] = None  # EIP-1559 priority fee


@dataclass
class EventData:
    """Ethereum event data structure"""
    address: str
    topics: List[str]
    data: str
    block_number: int
    transaction_hash: str
    log_index: int
    timestamp: datetime
    removed: bool = False


@dataclass
class MonitoringConfig:
    """Configuration for event monitoring"""
    poll_interval: float = 2.0  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    gas_price_update_interval: float = 10.0  # seconds
    event_history_limit: int = 10000
    gas_price_history_limit: int = 1000
    enable_eip1559: bool = True
    enable_event_indexing: bool = True


class GasPriceOracle:
    """Real-time gas price oracle with EIP-1559 support"""
    
    def __init__(self, w3: Web3, config: MonitoringConfig):
        self.w3 = w3
        self.config = config
        self._gas_price_history: deque = deque(maxlen=config.gas_price_history_limit)
        self._current_gas_price: Optional[GasPriceData] = None
        self._lock = threading.Lock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the gas price oracle"""
        if self._running:
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Gas price oracle started")
        
    async def stop(self) -> None:
        """Stop the gas price oracle"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Gas price oracle stopped")
        
    async def _update_loop(self) -> None:
        """Main update loop for gas price oracle"""
        while self._running:
            try:
                await self._update_gas_prices()
                await asyncio.sleep(self.config.gas_price_update_interval)
            except Exception as e:
                logger.error(f"Error in gas price update loop: {e}")
                await asyncio.sleep(self.config.retry_delay)
                
    async def _update_gas_prices(self) -> None:
        """Update gas prices from the network"""
        try:
            # Get current block
            block = self.w3.eth.get_block('latest')
            block_number = block.number
            
            # Get gas prices
            gas_prices = await self._get_gas_prices(block_number)
            
            # Store in history
            with self._lock:
                self._gas_price_history.append(gas_prices)
                self._current_gas_price = gas_prices
                
            logger.debug(f"Updated gas prices: {gas_prices}")
            
        except Exception as e:
            logger.error(f"Failed to update gas prices: {e}")
            raise
            
    async def _get_gas_prices(self, block_number: int) -> GasPriceData:
        """Get gas prices for a specific block"""
        try:
            # Get block data
            block = self.w3.eth.get_block(block_number)
            timestamp = datetime.fromtimestamp(block.timestamp)
            
            # Get base fee for EIP-1559
            base_fee = None
            priority_fee = None
            
            if self.config.enable_eip1559 and hasattr(block, 'baseFeePerGas'):
                base_fee = block.baseFeePerGas
                
            # Get gas price from pending transactions
            pending_txs = self.w3.eth.get_block('pending').transactions
            gas_prices = []
            
            for tx in pending_txs:
                if hasattr(tx, 'gasPrice') and tx.gasPrice:
                    gas_prices.append(tx.gasPrice)
                elif hasattr(tx, 'maxFeePerGas') and tx.maxFeePerGas:
                    gas_prices.append(tx.maxFeePerGas)
                    
            if gas_prices:
                # Calculate percentiles
                gas_prices.sort()
                slow = gas_prices[int(len(gas_prices) * 0.1)]
                standard = gas_prices[int(len(gas_prices) * 0.5)]
                fast = gas_prices[int(len(gas_prices) * 0.8)]
                instant = gas_prices[int(len(gas_prices) * 0.95)]
            else:
                # Fallback to network gas price
                network_gas_price = self.w3.eth.gas_price
                slow = int(network_gas_price * 0.8)
                standard = network_gas_price
                fast = int(network_gas_price * 1.2)
                instant = int(network_gas_price * 1.5)
                
            return GasPriceData(
                slow=slow,
                standard=standard,
                fast=fast,
                instant=instant,
                timestamp=timestamp,
                block_number=block_number,
                base_fee=base_fee,
                priority_fee=priority_fee
            )
            
        except Exception as e:
            logger.error(f"Failed to get gas prices for block {block_number}: {e}")
            raise
            
    def get_current_gas_price(self) -> Optional[GasPriceData]:
        """Get current gas price data"""
        with self._lock:
            return self._current_gas_price
            
    def get_gas_price_history(self, limit: Optional[int] = None) -> List[GasPriceData]:
        """Get gas price history"""
        with self._lock:
            history = list(self._gas_price_history)
            if limit:
                return history[-limit:]
            return history
            
    def get_recommended_gas_price(self, urgency: str = 'standard') -> Optional[int]:
        """Get recommended gas price based on urgency"""
        current = self.get_current_gas_price()
        if not current:
            return None
            
        urgency_map = {
            'slow': current.slow,
            'standard': current.standard,
            'fast': current.fast,
            'instant': current.instant
        }
        
        return urgency_map.get(urgency, current.standard)
        
    def get_gas_price_trend(self, hours: int = 1) -> Dict[str, float]:
        """Get gas price trend over time"""
        history = self.get_gas_price_history()
        if not history:
            return {}
            
        # Filter by time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [gp for gp in history if gp.timestamp >= cutoff_time]
        
        if len(recent_history) < 2:
            return {}
            
        # Calculate trends
        trends = {}
        for field in ['slow', 'standard', 'fast', 'instant']:
            values = [getattr(gp, field) for gp in recent_history]
            if len(values) > 1:
                trend = (values[-1] - values[0]) / values[0] * 100
                trends[field] = trend
                
        return trends


class EventMonitor:
    """Ethereum event monitor with real-time indexing"""
    
    def __init__(self, w3: Web3, config: MonitoringConfig):
        self.w3 = w3
        self.config = config
        self._event_history: deque = deque(maxlen=config.event_history_limit)
        self._event_filters: Dict[str, FilterParams] = {}
        self._event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
    async def start(self) -> None:
        """Start the event monitor"""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Event monitor started")
        
    async def stop(self) -> None:
        """Stop the event monitor"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Event monitor stopped")
        
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_events()
                await asyncio.sleep(self.config.poll_interval)
            except Exception as e:
                logger.error(f"Error in event monitoring loop: {e}")
                await asyncio.sleep(self.config.retry_delay)
                
    async def _check_events(self) -> None:
        """Check for new events"""
        for filter_id, filter_params in self._event_filters.items():
            try:
                # Get new logs
                logs = self.w3.eth.get_logs(filter_params)
                
                for log in logs:
                    event_data = EventData(
                        address=log.address,
                        topics=log.topics,
                        data=log.data,
                        block_number=log.blockNumber,
                        transaction_hash=log.transactionHash.hex(),
                        log_index=log.logIndex,
                        timestamp=datetime.now(),
                        removed=log.removed
                    )
                    
                    # Store in history
                    with self._lock:
                        self._event_history.append(event_data)
                        
                    # Trigger callbacks
                    for callback in self._event_callbacks[filter_id]:
                        try:
                            await callback(event_data)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
                            
            except Exception as e:
                logger.error(f"Error checking events for filter {filter_id}: {e}")
                
    def add_event_filter(self, filter_id: str, filter_params: FilterParams) -> None:
        """Add an event filter"""
        self._event_filters[filter_id] = filter_params
        logger.info(f"Added event filter: {filter_id}")
        
    def remove_event_filter(self, filter_id: str) -> None:
        """Remove an event filter"""
        if filter_id in self._event_filters:
            del self._event_filters[filter_id]
            if filter_id in self._event_callbacks:
                del self._event_callbacks[filter_id]
            logger.info(f"Removed event filter: {filter_id}")
            
    def add_event_callback(self, filter_id: str, callback: Callable[[EventData], None]) -> None:
        """Add a callback for events"""
        self._event_callbacks[filter_id].append(callback)
        logger.info(f"Added event callback for filter: {filter_id}")
        
    def remove_event_callback(self, filter_id: str, callback: Callable[[EventData], None]) -> None:
        """Remove an event callback"""
        if filter_id in self._event_callbacks:
            try:
                self._event_callbacks[filter_id].remove(callback)
                logger.info(f"Removed event callback for filter: {filter_id}")
            except ValueError:
                pass
                
    def get_event_history(self, limit: Optional[int] = None) -> List[EventData]:
        """Get event history"""
        with self._lock:
            history = list(self._event_history)
            if limit:
                return history[-limit:]
            return history
            
    def get_events_by_address(self, address: str, limit: Optional[int] = None) -> List[EventData]:
        """Get events for a specific address"""
        with self._lock:
            events = [e for e in self._event_history if e.address.lower() == address.lower()]
            if limit:
                return events[-limit:]
            return events
            
    def get_events_by_topic(self, topic: str, limit: Optional[int] = None) -> List[EventData]:
        """Get events for a specific topic"""
        with self._lock:
            events = [e for e in self._event_history if topic in e.topics]
            if limit:
                return events[-limit:]
            return events


class EthereumMonitoringService:
    """Main Ethereum monitoring service"""
    
    def __init__(self, w3: Web3, config: Optional[MonitoringConfig] = None):
        self.w3 = w3
        self.config = config or MonitoringConfig()
        self.gas_price_oracle = GasPriceOracle(w3, self.config)
        self.event_monitor = EventMonitor(w3, self.config)
        self._running = False
        
    async def start(self) -> None:
        """Start the monitoring service"""
        if self._running:
            return
            
        self._running = True
        await self.gas_price_oracle.start()
        await self.event_monitor.start()
        logger.info("Ethereum monitoring service started")
        
    async def stop(self) -> None:
        """Stop the monitoring service"""
        self._running = False
        await self.gas_price_oracle.stop()
        await self.event_monitor.stop()
        logger.info("Ethereum monitoring service stopped")
        
    def get_gas_price(self, urgency: str = 'standard') -> Optional[int]:
        """Get current gas price"""
        return self.gas_price_oracle.get_recommended_gas_price(urgency)
        
    def get_gas_price_data(self) -> Optional[GasPriceData]:
        """Get current gas price data"""
        return self.gas_price_oracle.get_current_gas_price()
        
    def get_gas_price_trend(self, hours: int = 1) -> Dict[str, float]:
        """Get gas price trend"""
        return self.gas_price_oracle.get_gas_price_trend(hours)
        
    def add_event_filter(self, filter_id: str, filter_params: FilterParams) -> None:
        """Add event filter"""
        self.event_monitor.add_event_filter(filter_id, filter_params)
        
    def remove_event_filter(self, filter_id: str) -> None:
        """Remove event filter"""
        self.event_monitor.remove_event_filter(filter_id)
        
    def add_event_callback(self, filter_id: str, callback: Callable[[EventData], None]) -> None:
        """Add event callback"""
        self.event_monitor.add_event_callback(filter_id, callback)
        
    def remove_event_callback(self, filter_id: str, callback: Callable[[EventData], None]) -> None:
        """Remove event callback"""
        self.event_monitor.remove_event_callback(filter_id, callback)
        
    def get_event_history(self, limit: Optional[int] = None) -> List[EventData]:
        """Get event history"""
        return self.event_monitor.get_event_history(limit)
        
    def get_events_by_address(self, address: str, limit: Optional[int] = None) -> List[EventData]:
        """Get events by address"""
        return self.event_monitor.get_events_by_address(address, limit)
        
    def get_events_by_topic(self, topic: str, limit: Optional[int] = None) -> List[EventData]:
        """Get events by topic"""
        return self.event_monitor.get_events_by_topic(topic, limit)
        
    async def wait_for_event(self, filter_id: str, timeout: float = 30.0) -> Optional[EventData]:
        """Wait for a specific event"""
        event_received = asyncio.Event()
        received_event = None
        
        def callback(event: EventData) -> None:
            nonlocal received_event
            received_event = event
            event_received.set()
            
        self.add_event_callback(filter_id, callback)
        
        try:
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event
        except asyncio.TimeoutError:
            return None
        finally:
            self.remove_event_callback(filter_id, callback)
            
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        gas_price_history = self.gas_price_oracle.get_gas_price_history()
        event_history = self.event_monitor.get_event_history()
        
        return {
            'gas_price_updates': len(gas_price_history),
            'events_monitored': len(event_history),
            'active_filters': len(self.event_monitor._event_filters),
            'active_callbacks': sum(len(callbacks) for callbacks in self.event_monitor._event_callbacks.values()),
            'running': self._running
        }
