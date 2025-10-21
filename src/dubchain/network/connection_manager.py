"""
Connection Manager Module

This module provides connection management including:
- Connection pooling and lifecycle management
- Connection health monitoring and recovery
- Load balancing across connections
- Connection quality assessment
- Automatic failover and reconnection
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
from collections import defaultdict, deque

from ..errors import NetworkError, ValidationError
from ..logging import get_logger

logger = get_logger(__name__)

class ConnectionState(Enum):
    """Connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"

class ConnectionType(Enum):
    """Types of connections."""
    PEER = "peer"
    CLIENT = "client"
    SERVER = "server"
    BRIDGE = "bridge"
    RELAY = "relay"

class ConnectionPriority(Enum):
    """Connection priorities."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ConnectionStrategy(Enum):
    """Connection strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    STICKY = "sticky"

@dataclass
class ConnectionMetrics:
    """Connection metrics."""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_time: float = 0.0
    last_activity: float = field(default_factory=time.time)
    latency: float = 0.0
    bandwidth: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0

@dataclass
class ConnectionConfig:
    """Connection configuration."""
    max_connections: int = 100
    connection_timeout: float = 30.0
    keepalive_interval: float = 60.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 5
    health_check_interval: float = 30.0
    idle_timeout: float = 300.0
    enable_compression: bool = True
    enable_encryption: bool = True
    buffer_size: int = 65536

@dataclass
class Connection:
    """Represents a network connection."""
    connection_id: str
    connection_type: ConnectionType
    remote_address: str
    remote_port: int
    local_address: str
    local_port: int
    state: ConnectionState = ConnectionState.DISCONNECTED
    priority: ConnectionPriority = ConnectionPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConnectionPool:
    """Manages a pool of connections."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize connection pool."""
        self.config = config
        self.connections: Dict[str, Connection] = {}
        self.connection_queue: deque = deque()
        self.active_connections: Set[str] = set()
        self.failed_connections: Set[str] = set()
        
        # Connection limits by type
        self.type_limits: Dict[ConnectionType, int] = {
            ConnectionType.PEER: config.max_connections // 2,
            ConnectionType.CLIENT: config.max_connections // 4,
            ConnectionType.SERVER: config.max_connections // 4,
            ConnectionType.BRIDGE: 10,
            ConnectionType.RELAY: 5
        }
        
        logger.info("Initialized connection pool")
    
    def add_connection(self, connection: Connection) -> bool:
        """Add a connection to the pool."""
        try:
            if connection.connection_id in self.connections:
                logger.warning(f"Connection {connection.connection_id} already exists")
                return False
            
            # Check type limits
            type_count = sum(1 for conn in self.connections.values() 
                           if conn.connection_type == connection.connection_type)
            
            if type_count >= self.type_limits.get(connection.connection_type, 0):
                logger.warning(f"Connection type {connection.connection_type.value} limit reached")
                return False
            
            self.connections[connection.connection_id] = connection
            self.connection_queue.append(connection.connection_id)
            
            logger.debug(f"Added connection {connection.connection_id} to pool")
            return True
            
        except Exception as e:
            logger.error(f"Error adding connection to pool: {e}")
            return False
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove a connection from the pool."""
        try:
            if connection_id not in self.connections:
                return False
            
            del self.connections[connection_id]
            
            # Remove from other collections
            self.active_connections.discard(connection_id)
            self.failed_connections.discard(connection_id)
            
            # Remove from queue
            if connection_id in self.connection_queue:
                self.connection_queue.remove(connection_id)
            
            logger.debug(f"Removed connection {connection_id} from pool")
            return True
            
        except Exception as e:
            logger.error(f"Error removing connection from pool: {e}")
            return False
    
    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID."""
        return self.connections.get(connection_id)
    
    def get_connections_by_type(self, connection_type: ConnectionType) -> List[Connection]:
        """Get all connections of a specific type."""
        return [conn for conn in self.connections.values() 
                if conn.connection_type == connection_type]
    
    def get_active_connections(self) -> List[Connection]:
        """Get all active connections."""
        return [conn for conn in self.connections.values() 
                if conn.connection_id in self.active_connections]
    
    def get_failed_connections(self) -> List[Connection]:
        """Get all failed connections."""
        return [conn for conn in self.connections.values() 
                if conn.connection_id in self.failed_connections]
    
    def mark_connection_active(self, connection_id: str) -> None:
        """Mark a connection as active."""
        if connection_id in self.connections:
            self.active_connections.add(connection_id)
            self.failed_connections.discard(connection_id)
    
    def mark_connection_failed(self, connection_id: str) -> None:
        """Mark a connection as failed."""
        if connection_id in self.connections:
            self.failed_connections.add(connection_id)
            self.active_connections.discard(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        try:
            total_connections = len(self.connections)
            active_connections = len(self.active_connections)
            failed_connections = len(self.failed_connections)
            
            type_counts = defaultdict(int)
            for conn in self.connections.values():
                type_counts[conn.connection_type.value] += 1
            
            return {
                "total_connections": total_connections,
                "active_connections": active_connections,
                "failed_connections": failed_connections,
                "type_counts": dict(type_counts),
                "pool_utilization": total_connections / self.config.max_connections if self.config.max_connections > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}

class ConnectionHealthMonitor:
    """Monitors connection health and quality."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize health monitor."""
        self.config = config
        self.health_checks: Dict[str, float] = {}
        self.quality_scores: Dict[str, float] = {}
        logger.info("Initialized connection health monitor")
    
    def check_connection_health(self, connection: Connection) -> Dict[str, Any]:
        """Check the health of a connection."""
        try:
            current_time = time.time()
            
            # Basic health indicators
            is_active = connection.state == ConnectionState.CONNECTED
            is_recently_active = current_time - connection.last_activity < self.config.idle_timeout
            has_low_error_rate = connection.metrics.error_count < 10
            has_good_success_rate = connection.metrics.success_rate > 0.8
            
            # Calculate health score
            health_score = 0
            if is_active:
                health_score += 40
            if is_recently_active:
                health_score += 30
            if has_low_error_rate:
                health_score += 20
            if has_good_success_rate:
                health_score += 10
            
            # Store health check timestamp
            self.health_checks[connection.connection_id] = current_time
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(connection)
            self.quality_scores[connection.connection_id] = quality_score
            
            return {
                "connection_id": connection.connection_id,
                "is_healthy": health_score >= 70,
                "health_score": health_score,
                "quality_score": quality_score,
                "is_active": is_active,
                "is_recently_active": is_recently_active,
                "has_low_error_rate": has_low_error_rate,
                "has_good_success_rate": has_good_success_rate,
                "last_activity": connection.last_activity,
                "error_count": connection.metrics.error_count,
                "success_rate": connection.metrics.success_rate
            }
            
        except Exception as e:
            logger.error(f"Error checking connection health: {e}")
            return {
                "connection_id": connection.connection_id,
                "is_healthy": False,
                "health_score": 0,
                "quality_score": 0,
                "error": str(e)
            }
    
    def _calculate_quality_score(self, connection: Connection) -> float:
        """Calculate connection quality score."""
        try:
            score = 0.0
            
            # Latency score (lower is better)
            if connection.metrics.latency > 0:
                latency_score = max(0, 100 - connection.metrics.latency / 10)
                score += latency_score * 0.3
            
            # Bandwidth score (higher is better)
            if connection.metrics.bandwidth > 0:
                bandwidth_score = min(100, connection.metrics.bandwidth / 10)
                score += bandwidth_score * 0.3
            
            # Success rate score
            score += connection.metrics.success_rate * 100 * 0.2
            
            # Activity score
            current_time = time.time()
            time_since_activity = current_time - connection.last_activity
            activity_score = max(0, 100 - time_since_activity / 10)
            score += activity_score * 0.2
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def get_health_summary(self, connections: List[Connection]) -> Dict[str, Any]:
        """Get health summary for multiple connections."""
        try:
            total_connections = len(connections)
            healthy_connections = 0
            total_health_score = 0
            total_quality_score = 0
            
            for connection in connections:
                health_info = self.check_connection_health(connection)
                
                if health_info.get("is_healthy", False):
                    healthy_connections += 1
                
                total_health_score += health_info.get("health_score", 0)
                total_quality_score += health_info.get("quality_score", 0)
            
            avg_health_score = total_health_score / total_connections if total_connections > 0 else 0
            avg_quality_score = total_quality_score / total_connections if total_connections > 0 else 0
            
            return {
                "total_connections": total_connections,
                "healthy_connections": healthy_connections,
                "unhealthy_connections": total_connections - healthy_connections,
                "health_percentage": (healthy_connections / total_connections * 100) if total_connections > 0 else 0,
                "average_health_score": avg_health_score,
                "average_quality_score": avg_quality_score
            }
            
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {}

class ConnectionLoadBalancer:
    """Load balances connections across available endpoints."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize load balancer."""
        self.config = config
        self.connection_weights: Dict[str, float] = {}
        self.connection_usage: Dict[str, int] = defaultdict(int)
        logger.info("Initialized connection load balancer")
    
    def select_connection(self, connections: List[Connection], 
                        connection_type: Optional[ConnectionType] = None) -> Optional[Connection]:
        """Select the best connection for use."""
        try:
            if not connections:
                return None
            
            # Filter by connection type if specified
            if connection_type:
                connections = [conn for conn in connections 
                             if conn.connection_type == connection_type]
            
            if not connections:
                return None
            
            # Filter to only active connections
            active_connections = [conn for conn in connections 
                                if conn.state == ConnectionState.CONNECTED]
            
            if not active_connections:
                return None
            
            # Use weighted round-robin selection
            return self._weighted_round_robin(active_connections)
            
        except Exception as e:
            logger.error(f"Error selecting connection: {e}")
            return None
    
    def _weighted_round_robin(self, connections: List[Connection]) -> Connection:
        """Select connection using weighted round-robin."""
        try:
            if not connections:
                return None
            
            # Calculate weights based on connection quality
            weights = []
            for conn in connections:
                weight = self._calculate_connection_weight(conn)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                # Equal weights if all are zero
                weights = [1.0] * len(connections)
                total_weight = len(connections)
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Select based on weighted probability
            import random
            rand = random.random()
            cumulative = 0
            
            for i, weight in enumerate(normalized_weights):
                cumulative += weight
                if rand <= cumulative:
                    return connections[i]
            
            # Fallback to first connection
            return connections[0]
            
        except Exception as e:
            logger.error(f"Error in weighted round-robin: {e}")
            return connections[0] if connections else None
    
    def _calculate_connection_weight(self, connection: Connection) -> float:
        """Calculate weight for a connection."""
        try:
            weight = 1.0
            
            # Priority multiplier
            priority_multipliers = {
                ConnectionPriority.LOW: 0.5,
                ConnectionPriority.NORMAL: 1.0,
                ConnectionPriority.HIGH: 1.5,
                ConnectionPriority.CRITICAL: 2.0
            }
            weight *= priority_multipliers.get(connection.priority, 1.0)
            
            # Success rate multiplier
            weight *= connection.metrics.success_rate
            
            # Latency penalty (lower latency = higher weight)
            if connection.metrics.latency > 0:
                latency_penalty = max(0.1, 1.0 - connection.metrics.latency / 1000)
                weight *= latency_penalty
            
            # Bandwidth bonus
            if connection.metrics.bandwidth > 0:
                bandwidth_bonus = min(2.0, 1.0 + connection.metrics.bandwidth / 1000)
                weight *= bandwidth_bonus
            
            # Usage penalty (less used = higher weight)
            usage_count = self.connection_usage.get(connection.connection_id, 0)
            usage_penalty = max(0.1, 1.0 - usage_count / 100)
            weight *= usage_penalty
            
            return max(0.1, weight)
            
        except Exception as e:
            logger.error(f"Error calculating connection weight: {e}")
            return 1.0
    
    def record_connection_usage(self, connection_id: str) -> None:
        """Record usage of a connection."""
        self.connection_usage[connection_id] += 1
    
    def reset_connection_usage(self, connection_id: str) -> None:
        """Reset usage count for a connection."""
        self.connection_usage[connection_id] = 0

class ConnectionManager:
    """Main connection management system."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize connection manager."""
        self.config = config
        self.pool = ConnectionPool(config)
        self.health_monitor = ConnectionHealthMonitor(config)
        self.load_balancer = ConnectionLoadBalancer(config)
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized connection manager")
    
    async def start(self) -> None:
        """Start connection management."""
        try:
            # Start health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Started connection management")
            
        except Exception as e:
            logger.error(f"Error starting connection management: {e}")
    
    async def stop(self) -> None:
        """Stop connection management."""
        try:
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Stopped connection management")
            
        except Exception as e:
            logger.error(f"Error stopping connection management: {e}")
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        try:
            while True:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
                
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop."""
        try:
            while True:
                await asyncio.sleep(self.config.health_check_interval * 2)
                await self._perform_cleanup()
                
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all connections."""
        try:
            connections = list(self.pool.connections.values())
            
            for connection in connections:
                health_info = self.health_monitor.check_connection_health(connection)
                
                if health_info.get("is_healthy", False):
                    self.pool.mark_connection_active(connection.connection_id)
                else:
                    self.pool.mark_connection_failed(connection.connection_id)
                    logger.warning(f"Connection {connection.connection_id} is unhealthy")
            
            logger.debug("Completed health checks")
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform cleanup of failed connections."""
        try:
            current_time = time.time()
            failed_connections = self.pool.get_failed_connections()
            
            for connection in failed_connections:
                # Remove connections that have been failed for too long
                if current_time - connection.last_activity > self.config.idle_timeout * 2:
                    self.pool.remove_connection(connection.connection_id)
                    logger.info(f"Removed failed connection {connection.connection_id}")
            
            logger.debug("Completed cleanup")
            
        except Exception as e:
            logger.error(f"Error performing cleanup: {e}")
    
    def create_connection(self, connection_type: ConnectionType, 
                        remote_address: str, remote_port: int,
                        local_address: str = "0.0.0.0", local_port: int = 0,
                        priority: ConnectionPriority = ConnectionPriority.NORMAL,
                        config: Optional[Dict[str, Any]] = None) -> Connection:
        """Create a new connection."""
        try:
            connection_id = str(uuid.uuid4())
            
            connection = Connection(
                connection_id=connection_id,
                connection_type=connection_type,
                remote_address=remote_address,
                remote_port=remote_port,
                local_address=local_address,
                local_port=local_port,
                priority=priority,
                config=config or {}
            )
            
            self.pool.add_connection(connection)
            logger.info(f"Created connection {connection_id}")
            
            return connection
            
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            raise NetworkError(f"Failed to create connection: {e}")
    
    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get a connection by ID."""
        return self.pool.get_connection(connection_id)
    
    def get_best_connection(self, connection_type: Optional[ConnectionType] = None) -> Optional[Connection]:
        """Get the best available connection."""
        try:
            connections = list(self.pool.connections.values())
            return self.load_balancer.select_connection(connections, connection_type)
            
        except Exception as e:
            logger.error(f"Error getting best connection: {e}")
            return None
    
    def update_connection_metrics(self, connection_id: str, 
                                 bytes_sent: int = 0, bytes_received: int = 0,
                                 messages_sent: int = 0, messages_received: int = 0,
                                 latency: float = 0.0, bandwidth: float = 0.0,
                                 success: bool = True) -> None:
        """Update connection metrics."""
        try:
            connection = self.pool.get_connection(connection_id)
            if not connection:
                return
            
            # Update metrics
            connection.metrics.bytes_sent += bytes_sent
            connection.metrics.bytes_received += bytes_received
            connection.metrics.messages_sent += messages_sent
            connection.metrics.messages_received += messages_received
            connection.last_activity = time.time()
            
            if latency > 0:
                connection.metrics.latency = latency
            if bandwidth > 0:
                connection.metrics.bandwidth = bandwidth
            
            # Update success rate
            total_operations = connection.metrics.messages_sent + connection.metrics.messages_received
            if success:
                connection.metrics.success_rate = (connection.metrics.success_rate * (total_operations - 1) + 1) / total_operations
            else:
                connection.metrics.error_count += 1
                connection.metrics.success_rate = (connection.metrics.success_rate * (total_operations - 1)) / total_operations
            
            # Record usage for load balancing
            self.load_balancer.record_connection_usage(connection_id)
            
        except Exception as e:
            logger.error(f"Error updating connection metrics: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        try:
            pool_stats = self.pool.get_connection_stats()
            connections = list(self.pool.connections.values())
            health_summary = self.health_monitor.get_health_summary(connections)
            
            return {
                "pool_stats": pool_stats,
                "health_summary": health_summary,
                "config": {
                    "max_connections": self.config.max_connections,
                    "connection_timeout": self.config.connection_timeout,
                    "keepalive_interval": self.config.keepalive_interval,
                    "health_check_interval": self.config.health_check_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}

__all__ = [
    "ConnectionManager",
    "ConnectionPool",
    "ConnectionHealthMonitor",
    "ConnectionLoadBalancer",
    "Connection",
    "ConnectionConfig",
    "ConnectionMetrics",
    "ConnectionState",
    "ConnectionType",
    "ConnectionPriority",
    "ConnectionStrategy",
]