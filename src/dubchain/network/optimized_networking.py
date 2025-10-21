"""
Optimized networking layer for DubChain.

This module provides performance-optimized networking implementations with:
- Async I/O for non-blocking operations
- Message batching and coalescing
- Zero-copy serialization
- Adaptive backpressure and peer prioritization
"""

import asyncio
import json
import struct
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import orjson

    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

from ..performance.optimizations import OptimizationManager
from .connection_manager import ConnectionManager
from .gossip import GossipProtocol
from .message_router import MessageRouter
from .peer import Peer


@dataclass
class OptimizedNetworkConfig:
    """Configuration for optimized networking."""

    # Async I/O configuration
    enable_async_io: bool = True
    max_concurrent_connections: int = 1000
    connection_timeout: float = 30.0
    read_timeout: float = 10.0
    write_timeout: float = 10.0

    # Batching configuration
    enable_message_batching: bool = True
    batch_size: int = 50
    batch_timeout: float = 0.05  # 50ms
    max_batch_size: int = 1000

    # Zero-copy configuration
    enable_zero_copy: bool = True
    use_binary_protocol: bool = True
    buffer_pool_size: int = 1000

    # Backpressure configuration
    enable_adaptive_backpressure: bool = True
    max_queue_size: int = 10000
    backpressure_threshold: float = 0.8
    peer_priority_enabled: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: float = 1.0


@dataclass
class MessageBatch:
    """Batch of messages for efficient transmission."""

    messages: List[Dict[str, Any]]
    batch_id: str
    timestamp: float
    total_size: int
    compression_enabled: bool = False


@dataclass
class PeerMetrics:
    """Metrics for peer performance."""

    peer_id: str
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    latency_ms: float = 0.0
    error_count: int = 0
    last_seen: float = field(default_factory=time.time)
    priority_score: float = 1.0


class ZeroCopySerializer:
    """Zero-copy serialization for network messages."""

    def __init__(self, use_binary: bool = True):
        self.use_binary = use_binary
        self.buffer_pool = deque(maxlen=1000)
        self._lock = threading.Lock()

    def serialize_message(self, message: Dict[str, Any]) -> bytes:
        """Serialize message with zero-copy optimization."""
        if self.use_binary and MSGPACK_AVAILABLE:
            return self._serialize_binary(message)
        elif self.use_binary and ORJSON_AVAILABLE:
            return self._serialize_orjson(message)
        else:
            return self._serialize_json(message)

    def _serialize_binary(self, message: Dict[str, Any]) -> bytes:
        """Serialize using msgpack for binary format."""
        try:
            return msgpack.packb(message)
        except Exception:
            # Fallback to JSON
            return self._serialize_json(message)

    def _serialize_orjson(self, message: Dict[str, Any]) -> bytes:
        """Serialize using orjson for fast JSON."""
        try:
            return orjson.dumps(message)
        except Exception:
            # Fallback to standard JSON
            return self._serialize_json(message)

    def _serialize_json(self, message: Dict[str, Any]) -> bytes:
        """Serialize using standard JSON."""
        return json.dumps(message).encode("utf-8")

    def deserialize_message(self, data: bytes) -> Dict[str, Any]:
        """Deserialize message with zero-copy optimization."""
        if self.use_binary and MSGPACK_AVAILABLE:
            return self._deserialize_binary(data)
        elif self.use_binary and ORJSON_AVAILABLE:
            return self._deserialize_orjson(data)
        else:
            return self._deserialize_json(data)

    def _deserialize_binary(self, data: bytes) -> Dict[str, Any]:
        """Deserialize using msgpack."""
        try:
            return msgpack.unpackb(data)
        except Exception:
            # Fallback to JSON
            return self._deserialize_json(data)

    def _deserialize_orjson(self, data: bytes) -> Dict[str, Any]:
        """Deserialize using orjson."""
        try:
            return orjson.loads(data)
        except Exception:
            # Fallback to standard JSON
            return self._deserialize_json(data)

    def _deserialize_json(self, data: bytes) -> Dict[str, Any]:
        """Deserialize using standard JSON."""
        return json.loads(data.decode("utf-8"))

    def get_buffer(self, size: int) -> bytearray:
        """Get buffer from pool for zero-copy operations."""
        with self._lock:
            # Try to find suitable buffer in pool
            for buffer in self.buffer_pool:
                if len(buffer) >= size:
                    self.buffer_pool.remove(buffer)
                    buffer[:size] = b"\x00" * size  # Clear buffer
                    return buffer[:size]

            # Create new buffer if none available
            return bytearray(size)

    def return_buffer(self, buffer: bytearray) -> None:
        """Return buffer to pool for reuse."""
        with self._lock:
            if len(self.buffer_pool) < self.buffer_pool.maxlen:
                self.buffer_pool.append(buffer)


class MessageBatcher:
    """Batches messages for efficient network transmission."""

    def __init__(self, config: OptimizedNetworkConfig):
        self.config = config
        self.pending_messages: Dict[str, deque] = defaultdict(lambda: deque())
        self.batch_timers: Dict[str, float] = {}
        self.batch_lock = threading.Lock()
        self.serializer = ZeroCopySerializer(config.use_binary_protocol)

    def add_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Add message to batch for a peer."""
        if not self.config.enable_message_batching:
            return False

        with self.batch_lock:
            peer_queue = self.pending_messages[peer_id]
            peer_queue.append(message)

            # Check if batch is ready
            if len(
                peer_queue
            ) >= self.config.batch_size or self._is_batch_timeout_reached(peer_id):
                # Process batch
                messages_to_send = list(peer_queue)
                peer_queue.clear()
                self.batch_timers.pop(peer_id, None)

                # Create batch message
                batch = self._create_batch(messages_to_send)
                return self._send_batch(peer_id, batch)

            # Set timer if this is the first message
            if peer_id not in self.batch_timers:
                self.batch_timers[peer_id] = time.time()

        return True

    def _is_batch_timeout_reached(self, peer_id: str) -> bool:
        """Check if batch timeout has been reached."""
        if peer_id not in self.batch_timers:
            return False
        return time.time() - self.batch_timers[peer_id] >= self.config.batch_timeout

    def _create_batch(self, messages: List[Dict[str, Any]]) -> MessageBatch:
        """Create a message batch."""
        batch_id = f"batch_{int(time.time() * 1000)}"
        timestamp = time.time()

        # Calculate total size
        total_size = sum(
            len(self.serializer.serialize_message(msg)) for msg in messages
        )

        return MessageBatch(
            messages=messages,
            batch_id=batch_id,
            timestamp=timestamp,
            total_size=total_size,
        )

    def _send_batch(self, peer_id: str, batch: MessageBatch) -> bool:
        """
        Send batch to peer.

        TODO: Implement actual network sending mechanism
        This would involve:
        1. Serializing the message batch
        2. Establishing connection to peer
        3. Sending data over network protocol
        4. Handling network errors and retries
        5. Updating delivery metrics
        """
        try:
            # TODO: Implement actual sending logic
            # For now, simulate successful sending
            logger.debug(
                f"Simulated sending batch of {len(batch.messages)} messages to peer {peer_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error sending batch to peer {peer_id}: {e}")
            return False

    def flush_pending_messages(self, peer_id: Optional[str] = None) -> int:
        """Flush pending messages for a peer or all peers."""
        with self.batch_lock:
            if peer_id:
                # Flush specific peer
                if peer_id in self.pending_messages:
                    messages = list(self.pending_messages[peer_id])
                    self.pending_messages[peer_id].clear()
                    self.batch_timers.pop(peer_id, None)

                    if messages:
                        batch = self._create_batch(messages)
                        self._send_batch(peer_id, batch)
                        return len(messages)
                return 0
            else:
                # Flush all peers
                total_flushed = 0
                for pid, queue in self.pending_messages.items():
                    if queue:
                        messages = list(queue)
                        queue.clear()
                        batch = self._create_batch(messages)
                        self._send_batch(pid, batch)
                        total_flushed += len(messages)

                self.batch_timers.clear()
                return total_flushed


class AdaptiveBackpressure:
    """Adaptive backpressure management for network operations."""

    def __init__(self, config: OptimizedNetworkConfig):
        self.config = config
        self.peer_metrics: Dict[str, PeerMetrics] = {}
        self.queue_sizes: Dict[str, int] = defaultdict(int)
        self.backpressure_states: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def update_peer_metrics(self, peer_id: str, metrics_update: Dict[str, Any]) -> None:
        """Update metrics for a peer."""
        with self._lock:
            if peer_id not in self.peer_metrics:
                self.peer_metrics[peer_id] = PeerMetrics(peer_id=peer_id)

            metrics = self.peer_metrics[peer_id]

            # Update metrics
            if "message_count" in metrics_update:
                metrics.message_count += metrics_update["message_count"]
            if "bytes_sent" in metrics_update:
                metrics.bytes_sent += metrics_update["bytes_sent"]
            if "bytes_received" in metrics_update:
                metrics.bytes_received += metrics_update["bytes_received"]
            if "latency_ms" in metrics_update:
                metrics.latency_ms = metrics_update["latency_ms"]
            if "error_count" in metrics_update:
                metrics.error_count += metrics_update["error_count"]

            metrics.last_seen = time.time()

            # Update priority score
            self._update_priority_score(metrics)

    def _update_priority_score(self, metrics: PeerMetrics) -> None:
        """Update priority score for a peer."""
        # Calculate priority based on performance metrics
        latency_factor = max(
            0, 1.0 - (metrics.latency_ms / 1000.0)
        )  # Lower latency = higher priority
        error_factor = max(
            0, 1.0 - (metrics.error_count / max(1, metrics.message_count))
        )
        activity_factor = min(
            1.0, metrics.message_count / 1000.0
        )  # More activity = higher priority

        metrics.priority_score = (
            latency_factor * 0.4 + error_factor * 0.4 + activity_factor * 0.2
        )

    def should_apply_backpressure(self, peer_id: str) -> bool:
        """Check if backpressure should be applied to a peer."""
        if not self.config.enable_adaptive_backpressure:
            return False

        with self._lock:
            queue_size = self.queue_sizes.get(peer_id, 0)
            max_queue = self.config.max_queue_size

            # Check queue size threshold
            if queue_size >= max_queue * self.config.backpressure_threshold:
                return True

            # Check peer-specific backpressure
            if peer_id in self.peer_metrics:
                metrics = self.peer_metrics[peer_id]

                # Apply backpressure for high-error peers
                if metrics.error_count > metrics.message_count * 0.1:  # 10% error rate
                    return True

                # Apply backpressure for high-latency peers
                if metrics.latency_ms > 5000:  # 5 second latency
                    return True

            return False

    def update_queue_size(self, peer_id: str, size: int) -> None:
        """Update queue size for a peer."""
        with self._lock:
            self.queue_sizes[peer_id] = size

    def get_peer_priority(self, peer_id: str) -> float:
        """Get priority score for a peer."""
        with self._lock:
            if peer_id in self.peer_metrics:
                return self.peer_metrics[peer_id].priority_score
            return 1.0  # Default priority

    def get_top_peers(self, count: int = 10) -> List[str]:
        """Get top priority peers."""
        with self._lock:
            sorted_peers = sorted(
                self.peer_metrics.items(),
                key=lambda x: x[1].priority_score,
                reverse=True,
            )
            return [peer_id for peer_id, _ in sorted_peers[:count]]


class OptimizedConnectionManager(ConnectionManager):
    """Optimized connection manager with async I/O and performance optimizations."""

    def __init__(self, config: OptimizedNetworkConfig):
        super().__init__()
        self.config = config
        self.optimization_manager = OptimizationManager()

        # Async I/O components
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_tasks: Set[asyncio.Task] = set()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Optimization components
        self.message_batcher = MessageBatcher(config)
        self.backpressure = AdaptiveBackpressure(config)
        self.serializer = ZeroCopySerializer(config.use_binary_protocol)

        # Performance monitoring
        self.performance_metrics = {
            "connections_created": 0,
            "connections_closed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "batch_count": 0,
            "backpressure_applied": 0,
        }

    async def start_async(self) -> None:
        """Start async I/O operations."""
        if not self.config.enable_async_io:
            return

        self.event_loop = asyncio.get_event_loop()

        # Start async tasks
        self.async_tasks.add(asyncio.create_task(self._batch_processor()))
        self.async_tasks.add(asyncio.create_task(self._backpressure_monitor()))
        self.async_tasks.add(asyncio.create_task(self._metrics_collector()))

    async def stop_async(self) -> None:
        """Stop async I/O operations."""
        # Cancel all async tasks
        for task in self.async_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.async_tasks:
            await asyncio.gather(*self.async_tasks, return_exceptions=True)

        self.async_tasks.clear()

    async def _batch_processor(self) -> None:
        """Process message batches asynchronously."""
        while True:
            try:
                # Flush pending messages periodically
                flushed_count = self.message_batcher.flush_pending_messages()
                if flushed_count > 0:
                    self.performance_metrics["batch_count"] += 1

                await asyncio.sleep(self.config.batch_timeout)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Batch processor error: {e}")
                await asyncio.sleep(1.0)

    async def _backpressure_monitor(self) -> None:
        """Monitor and apply backpressure asynchronously."""
        while True:
            try:
                # Check backpressure for all peers
                for peer_id in list(self.peers.keys()):
                    if self.backpressure.should_apply_backpressure(peer_id):
                        await self._apply_backpressure(peer_id)

                await asyncio.sleep(0.1)  # Check every 100ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Backpressure monitor error: {e}")
                await asyncio.sleep(1.0)

    async def _metrics_collector(self) -> None:
        """Collect performance metrics asynchronously."""
        while True:
            try:
                # Update peer metrics
                for peer_id, peer in self.peers.items():
                    metrics_update = {
                        "message_count": getattr(peer, "message_count", 0),
                        "bytes_sent": getattr(peer, "bytes_sent", 0),
                        "bytes_received": getattr(peer, "bytes_received", 0),
                        "latency_ms": getattr(peer, "latency_ms", 0.0),
                        "error_count": getattr(peer, "error_count", 0),
                    }
                    self.backpressure.update_peer_metrics(peer_id, metrics_update)

                await asyncio.sleep(self.config.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics collector error: {e}")
                await asyncio.sleep(1.0)

    async def _apply_backpressure(self, peer_id: str) -> None:
        """Apply backpressure to a peer."""
        if peer_id in self.peers:
            peer = self.peers[peer_id]

            # Reduce connection priority or throttle
            if hasattr(peer, "throttle"):
                peer.throttle = True

            self.performance_metrics["backpressure_applied"] += 1

    def send_message_optimized(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Send message with optimizations."""
        if not self.config.enable_async_io:
            return self._send_message_sync(peer_id, message)

        # Check backpressure
        if self.backpressure.should_apply_backpressure(peer_id):
            return False

        # Try to batch message
        if self.message_batcher.add_message(peer_id, message):
            return True

        # Send immediately if batching failed
        return self._send_message_sync(peer_id, message)

    def _send_message_sync(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Send message synchronously."""
        if peer_id not in self.peers:
            return False

        try:
            # Serialize message
            serialized = self.serializer.serialize_message(message)

            # TODO: Implement actual peer sending mechanism
            # This would involve:
            # 1. Checking peer connection status
            # 2. Sending data over established connection
            # 3. Handling network errors and timeouts
            # 4. Implementing retry logic for failed sends
            peer = self.peers[peer_id]
            # peer.send(serialized)  # TODO: Implement actual sending

            # Update metrics
            self.performance_metrics["messages_sent"] += 1
            self.performance_metrics["bytes_sent"] += len(serialized)

            # Update peer metrics
            self.backpressure.update_peer_metrics(
                peer_id, {"message_count": 1, "bytes_sent": len(serialized)}
            )

            return True

        except Exception as e:
            print(f"Error sending message to {peer_id}: {e}")

            # Update error metrics
            self.backpressure.update_peer_metrics(peer_id, {"error_count": 1})

            return False

    def receive_message_optimized(
        self, peer_id: str, data: bytes
    ) -> Optional[Dict[str, Any]]:
        """Receive and deserialize message with optimizations."""
        try:
            # Deserialize message
            message = self.serializer.deserialize_message(data)

            # Update metrics
            self.performance_metrics["messages_received"] += 1
            self.performance_metrics["bytes_received"] += len(data)

            # Update peer metrics
            self.backpressure.update_peer_metrics(
                peer_id, {"message_count": 1, "bytes_received": len(data)}
            )

            return message

        except Exception as e:
            print(f"Error receiving message from {peer_id}: {e}")

            # Update error metrics
            self.backpressure.update_peer_metrics(peer_id, {"error_count": 1})

            return None

    def add_peer_optimized(self, peer: Peer) -> bool:
        """Add peer with optimizations."""
        success = super().add_peer(peer)

        if success:
            self.performance_metrics["connections_created"] += 1

            # Initialize peer metrics
            self.backpressure.update_peer_metrics(peer.peer_id, {})

        return success

    def remove_peer_optimized(self, peer_id: str) -> bool:
        """Remove peer with optimizations."""
        success = super().remove_peer(peer_id)

        if success:
            self.performance_metrics["connections_closed"] += 1

            # Clean up peer data
            self.message_batcher.flush_pending_messages(peer_id)

        return success

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()

        # Add peer-specific metrics
        metrics["peer_count"] = len(self.peers)
        metrics["top_peers"] = self.backpressure.get_top_peers(5)

        # Add backpressure metrics
        backpressure_peers = sum(
            1
            for peer_id in self.peers.keys()
            if self.backpressure.should_apply_backpressure(peer_id)
        )
        metrics["backpressure_peers"] = backpressure_peers

        return metrics


class OptimizedGossipProtocol(GossipProtocol):
    """Optimized gossip protocol with batching and prioritization."""

    def __init__(self, config: OptimizedNetworkConfig):
        super().__init__()
        self.config = config
        self.optimization_manager = OptimizationManager()

        # Gossip optimizations
        self.message_cache: Dict[str, float] = {}  # message_hash -> timestamp
        self.cache_ttl = 300  # 5 minutes
        self.gossip_batcher = MessageBatcher(config)
        self.peer_priorities: Dict[str, float] = {}

    def gossip_message_optimized(
        self, message: Dict[str, Any], exclude_peers: Optional[Set[str]] = None
    ) -> int:
        """Gossip message with optimizations."""
        message_hash = self._get_message_hash(message)

        # Check cache to avoid duplicate gossip
        if message_hash in self.message_cache:
            cache_time = self.message_cache[message_hash]
            if time.time() - cache_time < self.cache_ttl:
                return 0  # Already gossiped recently

        # Add to cache
        self.message_cache[message_hash] = time.time()

        # Clean old cache entries
        self._clean_message_cache()

        # Get prioritized peers
        peers_to_gossip = self._get_prioritized_peers(exclude_peers)

        # Gossip to peers
        gossip_count = 0
        for peer_id in peers_to_gossip:
            if self.gossip_batcher.add_message(peer_id, message):
                gossip_count += 1

        return gossip_count

    def _get_message_hash(self, message: Dict[str, Any]) -> str:
        """Get hash for message deduplication."""
        import hashlib

        message_str = json.dumps(message, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()

    def _clean_message_cache(self) -> None:
        """Clean old entries from message cache."""
        current_time = time.time()
        expired_keys = [
            key
            for key, timestamp in self.message_cache.items()
            if current_time - timestamp >= self.cache_ttl
        ]

        for key in expired_keys:
            del self.message_cache[key]

    def _get_prioritized_peers(
        self, exclude_peers: Optional[Set[str]] = None
    ) -> List[str]:
        """Get prioritized peers for gossip."""
        if not self.config.peer_priority_enabled:
            # Return all peers except excluded ones
            all_peers = set(self.peers.keys())
            if exclude_peers:
                all_peers -= exclude_peers
            return list(all_peers)

        # Sort peers by priority
        peer_priorities = [
            (peer_id, self.peer_priorities.get(peer_id, 1.0))
            for peer_id in self.peers.keys()
            if exclude_peers is None or peer_id not in exclude_peers
        ]

        # Sort by priority (descending)
        peer_priorities.sort(key=lambda x: x[1], reverse=True)

        # Return top peers (limit to avoid overwhelming network)
        max_peers = min(len(peer_priorities), 20)  # Limit to top 20 peers
        return [peer_id for peer_id, _ in peer_priorities[:max_peers]]

    def update_peer_priority(self, peer_id: str, priority: float) -> None:
        """Update priority for a peer."""
        self.peer_priorities[peer_id] = priority


class OptimizedMessageRouter(MessageRouter):
    """Optimized message router with performance optimizations."""

    def __init__(self, config: OptimizedNetworkConfig):
        super().__init__()
        self.config = config
        self.optimization_manager = OptimizationManager()

        # Router optimizations
        self.route_cache: Dict[str, str] = {}  # message_type -> handler
        self.handler_metrics: Dict[str, Dict[str, Any]] = {}
        self.async_handlers: Dict[str, Callable] = {}

    def register_handler_optimized(
        self, message_type: str, handler: Callable, async_handler: bool = False
    ) -> None:
        """Register message handler with optimizations."""
        super().register_handler(message_type, handler)

        # Cache route for fast lookup
        self.route_cache[message_type] = message_type

        # Initialize metrics
        self.handler_metrics[message_type] = {
            "call_count": 0,
            "total_time": 0.0,
            "error_count": 0,
            "last_called": 0.0,
        }

        # Store async handler if provided
        if async_handler:
            self.async_handlers[message_type] = handler

    def route_message_optimized(self, message: Dict[str, Any], peer_id: str) -> bool:
        """Route message with optimizations."""
        message_type = message.get("type", "unknown")

        # Fast route lookup
        if message_type not in self.route_cache:
            return False

        # Get handler
        handler = self.handlers.get(message_type)
        if not handler:
            return False

        # Update metrics
        start_time = time.time()
        self.handler_metrics[message_type]["call_count"] += 1
        self.handler_metrics[message_type]["last_called"] = start_time

        try:
            # Route message
            if message_type in self.async_handlers:
                # Handle async message
                asyncio.create_task(
                    self._handle_async_message(handler, message, peer_id)
                )
                success = True
            else:
                # Handle sync message
                success = handler(message, peer_id)

            # Update metrics
            end_time = time.time()
            self.handler_metrics[message_type]["total_time"] += end_time - start_time

            return success

        except Exception as e:
            # Update error metrics
            self.handler_metrics[message_type]["error_count"] += 1
            print(f"Error routing message {message_type}: {e}")
            return False

    async def _handle_async_message(
        self, handler: Callable, message: Dict[str, Any], peer_id: str
    ) -> None:
        """Handle async message."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message, peer_id)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, message, peer_id)
        except Exception as e:
            print(f"Error in async message handler: {e}")

    def get_handler_metrics(self) -> Dict[str, Any]:
        """Get handler performance metrics."""
        metrics = {}

        for message_type, handler_metrics in self.handler_metrics.items():
            call_count = handler_metrics["call_count"]
            total_time = handler_metrics["total_time"]

            metrics[message_type] = {
                "call_count": call_count,
                "total_time": total_time,
                "avg_time": total_time / max(1, call_count),
                "error_count": handler_metrics["error_count"],
                "error_rate": handler_metrics["error_count"] / max(1, call_count),
                "last_called": handler_metrics["last_called"],
            }

        return metrics


class OptimizedNetworkManager:
    """Main optimized network manager."""

    def __init__(self, config: OptimizedNetworkConfig):
        self.config = config
        self.optimization_manager = OptimizationManager()

        # Network components
        self.connection_manager = OptimizedConnectionManager(config)
        self.gossip_protocol = OptimizedGossipProtocol(config)
        self.message_router = OptimizedMessageRouter(config)

        # Performance monitoring
        self.performance_monitor = None
        if config.enable_performance_monitoring:
            from ..performance.monitoring import PerformanceMonitor

            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_monitoring()

    async def start(self) -> None:
        """Start optimized network manager."""
        # Start async components
        await self.connection_manager.start_async()

        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()

    async def stop(self) -> None:
        """Stop optimized network manager."""
        # Stop async components
        await self.connection_manager.stop_async()

        # Stop performance monitoring
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "config": {
                "async_io_enabled": self.config.enable_async_io,
                "batching_enabled": self.config.enable_message_batching,
                "zero_copy_enabled": self.config.enable_zero_copy,
                "backpressure_enabled": self.config.enable_adaptive_backpressure,
            }
        }

        # Connection manager metrics
        metrics[
            "connection_manager"
        ] = self.connection_manager.get_performance_metrics()

        # Message router metrics
        metrics["message_router"] = self.message_router.get_handler_metrics()

        # Gossip protocol metrics
        metrics["gossip_protocol"] = {
            "cached_messages": len(self.gossip_protocol.message_cache),
            "prioritized_peers": len(self.gossip_protocol.peer_priorities),
        }

        return metrics
