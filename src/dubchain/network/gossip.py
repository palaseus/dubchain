"""
Gossip Protocol Module

This module implements an efficient gossip protocol for message propagation including:
- Epidemic-style message spreading
- Message deduplication and filtering
- Priority-based message routing
- Network topology-aware propagation
- Anti-spam and rate limiting mechanisms
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
from collections import defaultdict, deque

from ..errors import NetworkError, ValidationError
from ..logging import get_logger

logger = get_logger(__name__)

class MessageType(Enum):
    """Types of gossip messages."""
    BLOCK = "block"
    TRANSACTION = "transaction"
    CONSENSUS = "consensus"
    PEER_INFO = "peer_info"
    HEARTBEAT = "heartbeat"
    CUSTOM = "custom"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class PropagationStrategy(Enum):
    """Message propagation strategies."""
    FLOOD = "flood"
    PUSH = "push"
    PULL = "pull"
    PUSH_PULL = "push_pull"
    ADAPTIVE = "adaptive"

@dataclass
class GossipConfig:
    """Configuration for gossip protocol."""
    max_peers: int = 50
    message_timeout: int = 30
    propagation_timeout: int = 60
    max_message_size: int = 1024 * 1024  # 1MB
    duplicate_window: int = 300  # 5 minutes
    rate_limit_per_peer: int = 100  # messages per minute
    enable_anti_spam: bool = True
    enable_priority_routing: bool = True
    enable_topology_awareness: bool = True
    heartbeat_interval: int = 30
    peer_discovery_interval: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class GossipMessage:
    """Gossip protocol message."""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    data: Dict[str, Any]
    sender: str
    timestamp: float
    ttl: int = 10  # Time to live (hops)
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageFilter:
    """Message filtering criteria."""
    message_types: Set[MessageType] = field(default_factory=set)
    priority_threshold: MessagePriority = MessagePriority.LOW
    sender_whitelist: Set[str] = field(default_factory=set)
    sender_blacklist: Set[str] = field(default_factory=set)
    content_filters: List[str] = field(default_factory=list)
    max_message_size: int = 1024 * 1024  # 1MB

@dataclass
class PropagationMetrics:
    """Metrics for message propagation."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    duplicate_messages: int = 0
    propagation_time: float = 0.0
    network_coverage: float = 0.0
    bandwidth_used: int = 0

class MessageDeduplicator:
    """Handles message deduplication."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize message deduplicator."""
        self.config = config
        self.seen_messages: Dict[str, float] = {}
        self.message_cache_size = config.get("cache_size", 10000)
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour
        logger.info("Initialized message deduplicator")
    
    def is_duplicate(self, message: GossipMessage) -> bool:
        """Check if message is a duplicate."""
        try:
            current_time = time.time()
            message_id = message.message_id
            
            # Check if we've seen this message recently
            if message_id in self.seen_messages:
                last_seen = self.seen_messages[message_id]
                if current_time - last_seen < self.cache_ttl:
                    return True
            
            # Add to seen messages
            self.seen_messages[message_id] = current_time
            
            # Clean up old entries
            self._cleanup_cache()
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking message duplication: {e}")
            return False
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        try:
            current_time = time.time()
            
            # Remove expired entries
            expired_keys = [
                key for key, timestamp in self.seen_messages.items()
                if current_time - timestamp > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.seen_messages[key]
            
            # If still too large, remove oldest entries
            if len(self.seen_messages) > self.message_cache_size:
                sorted_items = sorted(self.seen_messages.items(), key=lambda x: x[1])
                items_to_remove = len(self.seen_messages) - self.message_cache_size
                
                for key, _ in sorted_items[:items_to_remove]:
                    del self.seen_messages[key]
            
        except Exception as e:
            logger.error(f"Error cleaning up message cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.seen_messages),
            "max_cache_size": self.message_cache_size,
            "cache_ttl": self.cache_ttl
        }

class MessageRouter:
    """Routes messages based on network topology and priority."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize message router."""
        self.config = config
        self.peer_weights: Dict[str, float] = {}
        self.message_queues: Dict[MessagePriority, deque] = {
            priority: deque() for priority in MessagePriority
        }
        self.routing_strategy = PropagationStrategy(config.get("strategy", "adaptive"))
        logger.info("Initialized message router")
    
    def add_peer(self, peer_id: str, weight: float = 1.0) -> None:
        """Add peer to routing table."""
        self.peer_weights[peer_id] = weight
        logger.debug(f"Added peer {peer_id} with weight {weight}")
    
    def remove_peer(self, peer_id: str) -> None:
        """Remove peer from routing table."""
        if peer_id in self.peer_weights:
            del self.peer_weights[peer_id]
            logger.debug(f"Removed peer {peer_id}")
    
    def select_peers_for_propagation(self, message: GossipMessage, available_peers: List[str], 
                                   max_peers: int = 3) -> List[str]:
        """Select peers for message propagation."""
        try:
            if not available_peers:
                return []
            
            # Filter peers based on message type and priority
            eligible_peers = self._filter_eligible_peers(message, available_peers)
            
            if not eligible_peers:
                return []
            
            # Select peers based on strategy
            if self.routing_strategy == PropagationStrategy.FLOOD:
                return eligible_peers[:max_peers]
            elif self.routing_strategy == PropagationStrategy.PUSH:
                return self._select_push_peers(eligible_peers, max_peers)
            elif self.routing_strategy == PropagationStrategy.PULL:
                return self._select_pull_peers(eligible_peers, max_peers)
            elif self.routing_strategy == PropagationStrategy.PUSH_PULL:
                return self._select_push_pull_peers(eligible_peers, max_peers)
            elif self.routing_strategy == PropagationStrategy.ADAPTIVE:
                return self._select_adaptive_peers(message, eligible_peers, max_peers)
            else:
                return eligible_peers[:max_peers]
                
        except Exception as e:
            logger.error(f"Error selecting peers for propagation: {e}")
            return []
    
    def _filter_eligible_peers(self, message: GossipMessage, peers: List[str]) -> List[str]:
        """Filter peers eligible for receiving the message."""
        try:
            eligible = []
            
            for peer_id in peers:
                # Check if peer has weight (is active)
                if peer_id not in self.peer_weights:
                    continue
                
                # Check message-specific filters
                if self._is_peer_eligible_for_message(message, peer_id):
                    eligible.append(peer_id)
            
            return eligible
            
        except Exception as e:
            logger.error(f"Error filtering eligible peers: {e}")
            return []
    
    def _is_peer_eligible_for_message(self, message: GossipMessage, peer_id: str) -> bool:
        """Check if peer is eligible for specific message."""
        try:
            # Check message type restrictions
            if message.message_type == MessageType.CONSENSUS:
                # Only send consensus messages to validators
                return self.peer_weights.get(peer_id, 0) > 0.5
            
            # Check priority-based routing
            if message.priority == MessagePriority.CRITICAL:
                # Send critical messages to high-weight peers
                return self.peer_weights.get(peer_id, 0) > 0.3
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking peer eligibility: {e}")
            return False
    
    def _select_push_peers(self, peers: List[str], max_peers: int) -> List[str]:
        """Select peers for push-based propagation."""
        try:
            # Sort by weight (descending)
            sorted_peers = sorted(peers, key=lambda p: self.peer_weights.get(p, 0), reverse=True)
            return sorted_peers[:max_peers]
            
        except Exception as e:
            logger.error(f"Error selecting push peers: {e}")
            return peers[:max_peers]
    
    def _select_pull_peers(self, peers: List[str], max_peers: int) -> List[str]:
        """Select peers for pull-based propagation."""
        try:
            # Select peers with lower weights (to balance load)
            sorted_peers = sorted(peers, key=lambda p: self.peer_weights.get(p, 0))
            return sorted_peers[:max_peers]
            
        except Exception as e:
            logger.error(f"Error selecting pull peers: {e}")
            return peers[:max_peers]
    
    def _select_push_pull_peers(self, peers: List[str], max_peers: int) -> List[str]:
        """Select peers for push-pull propagation."""
        try:
            # Mix of high and low weight peers
            sorted_peers = sorted(peers, key=lambda p: self.peer_weights.get(p, 0), reverse=True)
            
            push_count = max_peers // 2
            pull_count = max_peers - push_count
            
            selected = sorted_peers[:push_count]  # High weight peers
            selected.extend(sorted_peers[-pull_count:])  # Low weight peers
            
            return selected[:max_peers]
            
        except Exception as e:
            logger.error(f"Error selecting push-pull peers: {e}")
            return peers[:max_peers]
    
    def _select_adaptive_peers(self, message: GossipMessage, peers: List[str], max_peers: int) -> List[str]:
        """Select peers using adaptive strategy."""
        try:
            # Adaptive strategy based on message characteristics
            if message.priority == MessagePriority.CRITICAL:
                return self._select_push_peers(peers, max_peers)
            elif message.message_type == MessageType.HEARTBEAT:
                return self._select_pull_peers(peers, max_peers)
            else:
                return self._select_push_pull_peers(peers, max_peers)
                
        except Exception as e:
            logger.error(f"Error selecting adaptive peers: {e}")
            return peers[:max_peers]
    
    def queue_message(self, message: GossipMessage) -> None:
        """Queue message for propagation."""
        try:
            self.message_queues[message.priority].append(message)
            logger.debug(f"Queued message {message.message_id} with priority {message.priority}")
            
        except Exception as e:
            logger.error(f"Error queuing message: {e}")
    
    def get_next_message(self) -> Optional[GossipMessage]:
        """Get next message from queue (priority order)."""
        try:
            # Process messages in priority order
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                if self.message_queues[priority]:
                    return self.message_queues[priority].popleft()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next message: {e}")
            return None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get message queue statistics."""
        return {
            priority.name: len(queue) 
            for priority, queue in self.message_queues.items()
        }

class AntiSpamFilter:
    """Anti-spam and rate limiting for gossip messages."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize anti-spam filter."""
        self.config = config
        self.sender_rates: Dict[str, deque] = defaultdict(deque)
        self.message_rates: Dict[str, deque] = defaultdict(deque)
        self.blocked_senders: Set[str] = set()
        
        # Rate limiting parameters
        self.max_messages_per_second = config.get("max_messages_per_second", 10)
        self.max_messages_per_minute = config.get("max_messages_per_minute", 100)
        self.block_duration = config.get("block_duration", 300)  # 5 minutes
        
        logger.info("Initialized anti-spam filter")
    
    def is_allowed(self, message: GossipMessage) -> Tuple[bool, str]:
        """Check if message is allowed (not spam)."""
        try:
            sender = message.sender
            current_time = time.time()
            
            # Check if sender is blocked
            if sender in self.blocked_senders:
                return False, "Sender is blocked"
            
            # Check sender rate limits
            if not self._check_sender_rate_limit(sender, current_time):
                self._block_sender(sender)
                return False, "Sender rate limit exceeded"
            
            # Check message type rate limits
            message_type = message.message_type.value
            if not self._check_message_type_rate_limit(message_type, current_time):
                return False, "Message type rate limit exceeded"
            
            # Check message size
            message_size = len(json.dumps(message.data))
            max_size = self.config.get("max_message_size", 1024 * 1024)
            if message_size > max_size:
                return False, "Message too large"
            
            return True, "Allowed"
            
        except Exception as e:
            logger.error(f"Error checking message allowance: {e}")
            return False, "Error checking message"
    
    def _check_sender_rate_limit(self, sender: str, current_time: float) -> bool:
        """Check sender rate limits."""
        try:
            # Clean old timestamps
            self._cleanup_timestamps(self.sender_rates[sender], current_time)
            
            # Check per-second limit
            recent_messages = [t for t in self.sender_rates[sender] if current_time - t < 1.0]
            if len(recent_messages) >= self.max_messages_per_second:
                return False
            
            # Check per-minute limit
            recent_messages = [t for t in self.sender_rates[sender] if current_time - t < 60.0]
            if len(recent_messages) >= self.max_messages_per_minute:
                return False
            
            # Add current timestamp
            self.sender_rates[sender].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking sender rate limit: {e}")
            return False
    
    def _check_message_type_rate_limit(self, message_type: str, current_time: float) -> bool:
        """Check message type rate limits."""
        try:
            # Clean old timestamps
            self._cleanup_timestamps(self.message_rates[message_type], current_time)
            
            # Check per-minute limit for message type
            max_per_minute = self.config.get("max_per_message_type_per_minute", 50)
            recent_messages = [t for t in self.message_rates[message_type] if current_time - t < 60.0]
            
            if len(recent_messages) >= max_per_minute:
                return False
            
            # Add current timestamp
            self.message_rates[message_type].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking message type rate limit: {e}")
            return False
    
    def _cleanup_timestamps(self, timestamps: deque, current_time: float) -> None:
        """Clean up old timestamps."""
        try:
            cutoff_time = current_time - 300  # Keep last 5 minutes
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
                
        except Exception as e:
            logger.error(f"Error cleaning up timestamps: {e}")
    
    def _block_sender(self, sender: str) -> None:
        """Block a sender temporarily."""
        try:
            self.blocked_senders.add(sender)
            logger.warning(f"Blocked sender {sender} for spam")
            
            # Schedule unblock
            asyncio.create_task(self._unblock_sender_after_delay(sender))
            
        except Exception as e:
            logger.error(f"Error blocking sender: {e}")
    
    async def _unblock_sender_after_delay(self, sender: str) -> None:
        """Unblock sender after delay."""
        try:
            await asyncio.sleep(self.block_duration)
            self.blocked_senders.discard(sender)
            logger.info(f"Unblocked sender {sender}")
            
        except Exception as e:
            logger.error(f"Error unblocking sender: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get anti-spam filter statistics."""
        return {
            "blocked_senders": len(self.blocked_senders),
            "active_senders": len(self.sender_rates),
            "active_message_types": len(self.message_rates)
        }

class GossipProtocol:
    """Main gossip protocol implementation."""
    
    def __init__(self, config: Dict[str, Any], peer_manager):
        """Initialize gossip protocol."""
        self.config = config
        self.peer_manager = peer_manager
        
        # Initialize components
        self.deduplicator = MessageDeduplicator(config.get("deduplication", {}))
        self.router = MessageRouter(config.get("routing", {}))
        self.anti_spam = AntiSpamFilter(config.get("anti_spam", {}))
        
        # Message filters
        self.message_filter = MessageFilter()
        
        # Metrics
        self.metrics = PropagationMetrics()
        
        # Propagation settings
        self.max_propagation_hops = config.get("max_propagation_hops", 10)
        self.propagation_timeout = config.get("propagation_timeout", 30)
        
        logger.info("Initialized gossip protocol")
    
    async def propagate_message(self, message: GossipMessage, exclude_peers: Set[str] = None) -> bool:
        """Propagate a message through the network."""
        try:
            exclude_peers = exclude_peers or set()
            
            # Check if message is allowed
            allowed, reason = self.anti_spam.is_allowed(message)
            if not allowed:
                logger.warning(f"Message {message.message_id} blocked: {reason}")
                self.metrics.messages_dropped += 1
                return False
            
            # Check if message is duplicate
            if self.deduplicator.is_duplicate(message):
                logger.debug(f"Duplicate message {message.message_id} dropped")
                self.metrics.duplicate_messages += 1
                return False
            
            # Apply message filter
            if not self._passes_filter(message):
                logger.debug(f"Message {message.message_id} filtered out")
                self.metrics.messages_dropped += 1
                return False
            
            # Get available peers
            available_peers = [
                peer_id for peer_id in self.peer_manager.get_connected_peers()
                if peer_id not in exclude_peers and peer_id != message.sender
            ]
            
            if not available_peers:
                logger.debug(f"No peers available for message {message.message_id}")
                return False
            
            # Select peers for propagation
            selected_peers = self.router.select_peers_for_propagation(
                message, available_peers, max_peers=self.config.get("max_propagation_peers", 3)
            )
            
            if not selected_peers:
                logger.debug(f"No peers selected for message {message.message_id}")
                return False
            
            # Propagate to selected peers
            propagation_tasks = []
            for peer_id in selected_peers:
                task = self._send_to_peer(peer_id, message)
                propagation_tasks.append(task)
            
            # Wait for propagation to complete
            results = await asyncio.gather(*propagation_tasks, return_exceptions=True)
            
            # Count successful propagations
            successful = sum(1 for result in results if result is True)
            
            # Update metrics
            self.metrics.messages_sent += len(selected_peers)
            self.metrics.network_coverage = successful / len(selected_peers) if selected_peers else 0
            
            logger.info(f"Propagated message {message.message_id} to {successful}/{len(selected_peers)} peers")
            return successful > 0
            
        except Exception as e:
            logger.error(f"Error propagating message {message.message_id}: {e}")
            return False
    
    async def _send_to_peer(self, peer_id: str, message: GossipMessage) -> bool:
        """Send message to a specific peer."""
        try:
            # Create gossip message
            gossip_data = {
                "type": "gossip_message",
                "message": {
                    "message_id": message.message_id,
                    "message_type": message.message_type.value,
                    "priority": message.priority.value,
                    "data": message.data,
                    "sender": message.sender,
                    "timestamp": message.timestamp,
                    "ttl": message.ttl,
                    "signature": message.signature,
                    "metadata": message.metadata
                }
            }
            
            # Send message
            success = await self.peer_manager.send_message(peer_id, gossip_data)
            
            if success:
                logger.debug(f"Sent gossip message {message.message_id} to peer {peer_id}")
            else:
                logger.warning(f"Failed to send gossip message {message.message_id} to peer {peer_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message to peer {peer_id}: {e}")
            return False
    
    async def handle_received_message(self, peer_id: str, message_data: Dict[str, Any]) -> bool:
        """Handle received gossip message."""
        try:
            if message_data.get("type") != "gossip_message":
                return False
            
            message_info = message_data["message"]
            
            # Create GossipMessage object
            message = GossipMessage(
                message_id=message_info["message_id"],
                message_type=MessageType(message_info["message_type"]),
                priority=MessagePriority(message_info["priority"]),
                data=message_info["data"],
                sender=message_info["sender"],
                timestamp=message_info["timestamp"],
                ttl=message_info["ttl"],
                signature=message_info.get("signature"),
                metadata=message_info.get("metadata", {})
            )
            
            # Check TTL
            if message.ttl <= 0:
                logger.debug(f"Message {message.message_id} TTL expired")
                return False
            
            # Decrement TTL
            message.ttl -= 1
            
            # Update metrics
            self.metrics.messages_received += 1
            
            # Check if we should propagate further
            if message.ttl > 0:
                # Add current peer to exclude list
                exclude_peers = {peer_id, message.sender}
                
                # Propagate message
                await self.propagate_message(message, exclude_peers)
            
            logger.debug(f"Handled gossip message {message.message_id} from peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling received message from peer {peer_id}: {e}")
            return False
    
    def _passes_filter(self, message: GossipMessage) -> bool:
        """Check if message passes the filter."""
        try:
            # Check message type
            if self.message_filter.message_types and message.message_type not in self.message_filter.message_types:
                return False
            
            # Check priority threshold
            if message.priority.value < self.message_filter.priority_threshold.value:
                return False
            
            # Check sender whitelist
            if self.message_filter.sender_whitelist and message.sender not in self.message_filter.sender_whitelist:
                return False
            
            # Check sender blacklist
            if message.sender in self.message_filter.sender_blacklist:
                return False
            
            # Check message size
            message_size = len(json.dumps(message.data))
            if message_size > self.message_filter.max_message_size:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking message filter: {e}")
            return False
    
    def set_message_filter(self, filter_config: MessageFilter) -> None:
        """Set message filter configuration."""
        self.message_filter = filter_config
        logger.info("Updated message filter configuration")
    
    def get_propagation_metrics(self) -> PropagationMetrics:
        """Get propagation metrics."""
        return self.metrics
    
    def get_deduplicator_stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        return self.deduplicator.get_cache_stats()
    
    def get_router_stats(self) -> Dict[str, int]:
        """Get router statistics."""
        return self.router.get_queue_stats()
    
    def get_anti_spam_stats(self) -> Dict[str, Any]:
        """Get anti-spam filter statistics."""
        return self.anti_spam.get_stats()

__all__ = [
    "GossipProtocol",
    "MessageDeduplicator",
    "MessageRouter",
    "AntiSpamFilter",
    "GossipMessage",
    "MessageFilter",
    "PropagationMetrics",
    "MessageType",
    "MessagePriority",
    "PropagationStrategy",
]