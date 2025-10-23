"""
Peer Management Module

This module provides comprehensive peer lifecycle management including:
- Peer discovery and connection establishment
- Peer authentication and key exchange
- Peer health monitoring and maintenance
- Peer reputation and trust management
- Peer communication protocols
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

from ..errors import NetworkError, ValidationError
from ..logging import get_logger
from ..crypto.signatures import PublicKey, PrivateKey, Signature

logger = get_logger(__name__)

class PeerStatus(Enum):
    """Peer connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    SYNCING = "syncing"
    READY = "ready"
    ERROR = "error"
    SUSPENDED = "suspended"
    BANNED = "banned"

class PeerConnectionStatus(Enum):
    """Peer connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUSPENDED = "suspended"
    BANNED = "banned"

class PeerRole(Enum):
    """Peer roles in the network."""
    VALIDATOR = "validator"
    FULL_NODE = "full_node"
    LIGHT_CLIENT = "light_client"
    BRIDGE_NODE = "bridge_node"
    RELAYER = "relayer"
    ARCHIVE_NODE = "archive_node"

class ConnectionType(Enum):
    """Types of peer connections."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    RELAY = "relay"
    SEED = "seed"
    PERSISTENT = "persistent"
    TEMPORARY = "temporary"

@dataclass
class PeerConfig:
    """Configuration for peer management."""
    max_connections: int = 100
    connection_timeout: int = 30
    keepalive_interval: int = 60
    inactive_timeout: int = 600  # 10 minutes
    max_message_size: int = 1024 * 1024  # 1MB
    enable_compression: bool = True
    enable_encryption: bool = True
    reputation_threshold: float = 0.5
    ban_threshold: float = 0.1
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class PeerInfo:
    """Information about a peer."""
    peer_id: str
    address: str
    port: int
    public_key: str
    connection_type: ConnectionType = ConnectionType.OUTBOUND
    status: PeerStatus = PeerStatus.DISCONNECTED
    roles: Set[PeerRole] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    user_agent: str = "DubChain/1.0.0"
    last_seen: float = field(default_factory=time.time)
    successful_connections: int = 0
    failed_connections: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.peer_id:
            raise ValueError("Peer ID cannot be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError("Port must be between 1 and 65535")
    
    def update_last_seen(self) -> None:
        """Update the last seen timestamp."""
        self.last_seen = time.time()
    
    def record_successful_connection(self) -> None:
        """Record a successful connection."""
        self.successful_connections += 1
    
    def record_failed_connection(self) -> None:
        """Record a failed connection."""
        self.failed_connections += 1
    
    def add_capability(self, capability: str) -> None:
        """Add a capability."""
        self.capabilities.add(capability)
    
    def has_capability(self, capability: str) -> bool:
        """Check if peer has a capability."""
        return capability in self.capabilities
    
    def get_connection_success_rate(self) -> float:
        """Get the connection success rate."""
        total_connections = self.successful_connections + self.failed_connections
        if total_connections == 0:
            return 0.0
        return self.successful_connections / total_connections
    
    def is_healthy(self) -> bool:
        """Check if peer is healthy."""
        # Consider healthy if connected and has good success rate
        if self.status != PeerStatus.CONNECTED:
            return False
        
        success_rate = self.get_connection_success_rate()
        return success_rate >= 0.5  # At least 50% success rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'peer_id': self.peer_id,
            'address': self.address,
            'port': self.port,
            'public_key': str(self.public_key),
            'connection_type': self.connection_type.value,
            'status': self.status.value,
            'roles': [role.value for role in self.roles],
            'capabilities': list(self.capabilities),
            'version': self.version,
            'user_agent': self.user_agent,
            'last_seen': self.last_seen,
            'successful_connections': self.successful_connections,
            'failed_connections': self.failed_connections,
            'metadata': self.metadata,
        }

@dataclass
class PeerMetrics:
    """Peer performance metrics."""
    peer_id: str
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_time: float = field(default_factory=time.time)
    last_message_time: float = field(default_factory=time.time)
    latency: float = 0.0
    uptime: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0

@dataclass
class PeerReputation:
    """Peer reputation and trust score."""
    peer_id: str
    trust_score: float = 0.5  # 0.0 to 1.0
    reputation_score: float = 0.5  # 0.0 to 1.0
    violation_count: int = 0
    last_violation: Optional[float] = None
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_updated: float = field(default_factory=time.time)

@dataclass
class PeerConnection:
    """Active peer connection."""
    peer_id: str
    connection_type: ConnectionType
    status: PeerStatus
    transport: Any  # asyncio transport
    protocol: Any  # protocol instance
    peer_info: PeerInfo
    metrics: PeerMetrics
    reputation: PeerReputation
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

class PeerAuthenticator:
    """Handles peer authentication and key exchange."""
    
    def __init__(self, private_key: PrivateKey):
        """Initialize peer authenticator."""
        self.private_key = private_key
        self.public_key = private_key.get_public_key()
        self.peer_keys: Dict[str, PublicKey] = {}
        logger.info("Initialized peer authenticator")
    
    async def authenticate_peer(self, peer_id: str, challenge: bytes) -> Tuple[bool, Optional[bytes]]:
        """Authenticate a peer using challenge-response."""
        try:
            # Sign the challenge with our private key
            signature = self.private_key.sign(challenge)
            
            # Create response with our public key and signature
            response = {
                "public_key": self.public_key.to_hex(),
                "signature": signature.to_hex(),
                "timestamp": int(time.time())
            }
            
            response_bytes = json.dumps(response).encode()
            
            logger.info(f"Authenticated peer {peer_id}")
            return True, response_bytes
            
        except Exception as e:
            logger.error(f"Failed to authenticate peer {peer_id}: {e}")
            return False, None
    
    async def verify_peer_authentication(self, peer_id: str, response: bytes) -> bool:
        """Verify peer authentication response."""
        try:
            response_data = json.loads(response.decode())
            
            # Extract peer's public key and signature
            peer_public_key_hex = response_data["public_key"]
            signature_hex = response_data["signature"]
            timestamp = response_data["timestamp"]
            
            # Check timestamp (within 5 minutes)
            if abs(time.time() - timestamp) > 300:
                logger.error(f"Authentication response too old for peer {peer_id}")
                return False
            
            # Create peer public key
            peer_public_key = PublicKey.from_hex(peer_public_key_hex)
            
            # Verify signature
            challenge = f"auth_challenge_{peer_id}_{timestamp}".encode()
            signature = Signature.from_hex(signature_hex)
            
            if not peer_public_key.verify(challenge, signature):
                logger.error(f"Invalid signature for peer {peer_id}")
                return False
            
            # Store peer's public key
            self.peer_keys[peer_id] = peer_public_key
            
            logger.info(f"Verified authentication for peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify authentication for peer {peer_id}: {e}")
            return False
    
    def get_peer_public_key(self, peer_id: str) -> Optional[PublicKey]:
        """Get peer's public key."""
        return self.peer_keys.get(peer_id)

class PeerHealthMonitor:
    """Monitors peer health and connection quality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize peer health monitor."""
        self.config = config
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.unhealthy_peers: Set[str] = set()
        logger.info("Initialized peer health monitor")
    
    async def check_peer_health(self, peer_id: str, connection: PeerConnection) -> bool:
        """Check if a peer is healthy."""
        try:
            current_time = time.time()
            
            # Check connection status
            if connection.status not in [PeerStatus.CONNECTED, PeerStatus.AUTHENTICATED]:
                return False
            
            # Check last activity
            if current_time - connection.last_activity > self.config.get("max_idle_time", 300):
                logger.warning(f"Peer {peer_id} has been idle too long")
                return False
            
            # Check error rate
            if connection.metrics.error_count > self.config.get("max_errors", 10):
                logger.warning(f"Peer {peer_id} has too many errors")
                return False
            
            # Check success rate
            if connection.metrics.success_rate < self.config.get("min_success_rate", 0.8):
                logger.warning(f"Peer {peer_id} has low success rate")
                return False
            
            # Check latency
            if connection.metrics.latency > self.config.get("max_latency", 5.0):
                logger.warning(f"Peer {peer_id} has high latency")
                return False
            
            # Perform ping test
            ping_success = await self._ping_peer(connection)
            if not ping_success:
                logger.warning(f"Ping test failed for peer {peer_id}")
                return False
            
            # Update health status
            self.health_checks[peer_id] = {
                "last_check": current_time,
                "healthy": True,
                "latency": connection.metrics.latency,
                "success_rate": connection.metrics.success_rate
            }
            
            # Remove from unhealthy set if it was there
            self.unhealthy_peers.discard(peer_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for peer {peer_id}: {e}")
            self.unhealthy_peers.add(peer_id)
            return False
    
    async def _ping_peer(self, connection: PeerConnection) -> bool:
        """Send ping to peer and measure latency."""
        try:
            start_time = time.time()
            
            # Send ping message
            ping_message = {
                "type": "ping",
                "timestamp": start_time,
                "nonce": str(uuid.uuid4())
            }
            
            # In a real implementation, would send through the connection
            # For now, simulate ping success
            await asyncio.sleep(0.001)  # Simulate network delay
            
            # Calculate latency
            latency = time.time() - start_time
            connection.metrics.latency = latency
            connection.last_activity = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False
    
    def get_unhealthy_peers(self) -> Set[str]:
        """Get list of unhealthy peers."""
        return self.unhealthy_peers.copy()
    
    def get_peer_health(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Get peer health information."""
        return self.health_checks.get(peer_id)

class PeerReputationManager:
    """Manages peer reputation and trust scores."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reputation manager."""
        self.config = config
        self.reputations: Dict[str, PeerReputation] = {}
        logger.info("Initialized peer reputation manager")
    
    def update_reputation(self, peer_id: str, interaction_type: str, success: bool, severity: float = 1.0) -> None:
        """Update peer reputation based on interaction."""
        try:
            if peer_id not in self.reputations:
                self.reputations[peer_id] = PeerReputation(peer_id=peer_id)
            
            reputation = self.reputations[peer_id]
            current_time = time.time()
            
            if success:
                # Positive interaction
                reputation.positive_interactions += 1
                
                # Increase trust score
                trust_increase = self.config.get("trust_increase_rate", 0.01) * severity
                reputation.trust_score = min(1.0, reputation.trust_score + trust_increase)
                
                # Increase reputation score
                rep_increase = self.config.get("reputation_increase_rate", 0.005) * severity
                reputation.reputation_score = min(1.0, reputation.reputation_score + rep_increase)
                
            else:
                # Negative interaction
                reputation.negative_interactions += 1
                reputation.violation_count += 1
                reputation.last_violation = current_time
                
                # Decrease trust score
                trust_decrease = self.config.get("trust_decrease_rate", 0.05) * severity
                reputation.trust_score = max(0.0, reputation.trust_score - trust_decrease)
                
                # Decrease reputation score
                rep_decrease = self.config.get("reputation_decrease_rate", 0.02) * severity
                reputation.reputation_score = max(0.0, reputation.reputation_score - rep_decrease)
            
            reputation.last_updated = current_time
            
            logger.debug(f"Updated reputation for peer {peer_id}: trust={reputation.trust_score:.3f}, rep={reputation.reputation_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update reputation for peer {peer_id}: {e}")
    
    def get_reputation(self, peer_id: str) -> Optional[PeerReputation]:
        """Get peer reputation."""
        return self.reputations.get(peer_id)
    
    def is_trusted_peer(self, peer_id: str) -> bool:
        """Check if peer is trusted."""
        reputation = self.reputations.get(peer_id)
        if not reputation:
            return False
        
        min_trust = self.config.get("min_trust_score", 0.3)
        min_reputation = self.config.get("min_reputation_score", 0.2)
        
        return (reputation.trust_score >= min_trust and 
                reputation.reputation_score >= min_reputation)
    
    def get_top_peers(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top peers by reputation score."""
        peer_scores = [
            (peer_id, rep.reputation_score)
            for peer_id, rep in self.reputations.items()
        ]
        
        # Sort by reputation score (descending)
        peer_scores.sort(key=lambda x: x[1], reverse=True)
        
        return peer_scores[:limit]

class Peer:
    """Represents a peer in the network."""
    
    def __init__(self, peer_info: PeerInfo, private_key: Optional[PrivateKey] = None):
        """Initialize peer."""
        self.peer_info = peer_info
        self.private_key = private_key
        self.connection: Optional[PeerConnection] = None
        self.metrics = PeerMetrics()
        self.reputation = PeerReputation()
        self.last_activity = time.time()
        self.created_at = time.time()
    
    def get_id(self) -> str:
        """Get peer ID."""
        return self.peer_info.node_id
    
    def get_address(self) -> str:
        """Get peer address."""
        return self.peer_info.address
    
    def get_public_key(self) -> Optional[PublicKey]:
        """Get peer public key."""
        return self.peer_info.public_key
    
    def is_connected(self) -> bool:
        """Check if peer is connected."""
        return self.connection is not None and self.connection.is_connected()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def __str__(self) -> str:
        """String representation."""
        return f"Peer(id={self.get_id()}, address={self.get_address()})"
    
    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()

class PeerNode:
    """Represents a peer node in the network."""
    
    def __init__(self, peer_info: PeerInfo, private_key: Optional[PrivateKey] = None):
        """Initialize peer node."""
        self.peer_info = peer_info
        self.private_key = private_key
        self.connection: Optional[PeerConnection] = None
        self.metrics = PeerMetrics()
        self.reputation = PeerReputation()
        self.last_activity = time.time()
        self.created_at = time.time()
    
    def get_id(self) -> str:
        """Get peer ID."""
        return self.peer_info.node_id
    
    def get_address(self) -> str:
        """Get peer address."""
        return self.peer_info.address
    
    def get_public_key(self) -> Optional[PublicKey]:
        """Get peer public key."""
        return self.peer_info.public_key
    
    def is_connected(self) -> bool:
        """Check if peer is connected."""
        return self.connection is not None and self.connection.is_connected()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def __str__(self) -> str:
        """String representation."""
        return f"PeerNode(id={self.get_id()}, address={self.get_address()})"
    
    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()


class PeerManager:
    """Main peer management system."""
    
    def __init__(self, config: Dict[str, Any], private_key: PrivateKey):
        """Initialize peer manager."""
        self.config = config
        self.private_key = private_key
        self.peer_id = self._generate_peer_id()
        
        # Initialize components
        self.authenticator = PeerAuthenticator(private_key)
        self.health_monitor = PeerHealthMonitor(config.get("health", {}))
        self.reputation_manager = PeerReputationManager(config.get("reputation", {}))
        
        # Peer storage
        self.connections: Dict[str, PeerConnection] = {}
        self.peer_info_cache: Dict[str, PeerInfo] = {}
        
        # Connection limits
        self.max_connections = config.get("max_connections", 100)
        self.max_inbound = config.get("max_inbound", 50)
        self.max_outbound = config.get("max_outbound", 50)
        
        logger.info(f"Initialized peer manager with ID: {self.peer_id}")
    
    def _generate_peer_id(self) -> str:
        """Generate unique peer ID."""
        public_key = self.private_key.get_public_key()
        peer_id_hash = hashlib.sha256(public_key.to_bytes()).hexdigest()[:16]
        return f"peer_{peer_id_hash}"
    
    async def connect_to_peer(self, address: str, port: int, peer_info: Optional[PeerInfo] = None) -> Optional[str]:
        """Connect to a peer."""
        try:
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                logger.warning("Maximum connections reached")
                return None
            
            outbound_count = sum(1 for conn in self.connections.values() 
                               if conn.connection_type == ConnectionType.OUTBOUND)
            if outbound_count >= self.max_outbound:
                logger.warning("Maximum outbound connections reached")
                return None
            
            # Generate peer ID if not provided
            if not peer_info:
                peer_id = f"peer_{address}_{port}_{int(time.time())}"
                peer_info = PeerInfo(
                    peer_id=peer_id,
                    address=address,
                    port=port,
                    public_key="",  # Will be set during authentication
                    roles=set(),
                    capabilities=set()
                )
            else:
                peer_id = peer_info.peer_id
            
            # Check if already connected
            if peer_id in self.connections:
                logger.warning(f"Already connected to peer {peer_id}")
                return peer_id
            
            # Create connection
            connection = PeerConnection(
                peer_id=peer_id,
                connection_type=ConnectionType.OUTBOUND,
                status=PeerStatus.CONNECTING,
                transport=None,  # Will be set by transport
                protocol=None,   # Will be set by protocol
                peer_info=peer_info,
                metrics=PeerMetrics(peer_id=peer_id),
                reputation=PeerReputation(peer_id=peer_id)
            )
            
            # Store connection
            self.connections[peer_id] = connection
            
            # In a real implementation, would establish actual network connection
            # For now, simulate connection
            await asyncio.sleep(0.1)
            connection.status = PeerStatus.CONNECTED
            
            logger.info(f"Connected to peer {peer_id} at {address}:{port}")
            return peer_id
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {address}:{port}: {e}")
            return None
    
    async def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect from a peer."""
        try:
            if peer_id not in self.connections:
                logger.warning(f"Peer {peer_id} not connected")
                return False
            
            connection = self.connections[peer_id]
            
            # Close transport if available
            if connection.transport and not connection.transport.is_closing():
                connection.transport.close()
            
            # Update status
            connection.status = PeerStatus.DISCONNECTED
            
            # Remove from connections
            del self.connections[peer_id]
            
            logger.info(f"Disconnected from peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from peer {peer_id}: {e}")
            return False
    
    async def authenticate_peer(self, peer_id: str) -> bool:
        """Authenticate a peer."""
        try:
            if peer_id not in self.connections:
                logger.error(f"Peer {peer_id} not connected")
                return False
            
            connection = self.connections[peer_id]
            
            # Generate challenge
            challenge = f"auth_challenge_{peer_id}_{int(time.time())}".encode()
            
            # Authenticate peer
            success, response = await self.authenticator.authenticate_peer(peer_id, challenge)
            if not success:
                logger.error(f"Failed to authenticate peer {peer_id}")
                return False
            
            # Verify peer authentication
            verified = await self.authenticator.verify_peer_authentication(peer_id, response)
            if not verified:
                logger.error(f"Peer {peer_id} authentication verification failed")
                return False
            
            # Update connection status
            connection.status = PeerStatus.AUTHENTICATED
            
            logger.info(f"Successfully authenticated peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed for peer {peer_id}: {e}")
            return False
    
    async def send_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Send message to peer."""
        try:
            if peer_id not in self.connections:
                logger.error(f"Peer {peer_id} not connected")
                return False
            
            connection = self.connections[peer_id]
            
            if connection.status not in [PeerStatus.CONNECTED, PeerStatus.AUTHENTICATED]:
                logger.error(f"Peer {peer_id} not ready for messages")
                return False
            
            # Serialize message
            message_bytes = json.dumps(message).encode()
            
            # Update metrics
            connection.metrics.messages_sent += 1
            connection.metrics.bytes_sent += len(message_bytes)
            connection.last_activity = time.time()
            
            # In a real implementation, would send through transport
            logger.debug(f"Sent message to peer {peer_id}: {message.get('type', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to peer {peer_id}: {e}")
            # Update error metrics
            if peer_id in self.connections:
                self.connections[peer_id].metrics.error_count += 1
            return False
    
    async def receive_message(self, peer_id: str, message_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Receive message from peer."""
        try:
            if peer_id not in self.connections:
                logger.error(f"Peer {peer_id} not connected")
                return None
            
            connection = self.connections[peer_id]
            
            # Deserialize message
            message = json.loads(message_bytes.decode())
            
            # Update metrics
            connection.metrics.messages_received += 1
            connection.metrics.bytes_received += len(message_bytes)
            connection.last_activity = time.time()
            
            logger.debug(f"Received message from peer {peer_id}: {message.get('type', 'unknown')}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive message from peer {peer_id}: {e}")
            # Update error metrics
            if peer_id in self.connections:
                self.connections[peer_id].metrics.error_count += 1
            return None
    
    async def health_check_all_peers(self) -> Dict[str, bool]:
        """Perform health check on all connected peers."""
        results = {}
        
        for peer_id, connection in self.connections.items():
            try:
                healthy = await self.health_monitor.check_peer_health(peer_id, connection)
                results[peer_id] = healthy
                
                # Update reputation based on health check
                self.reputation_manager.update_reputation(
                    peer_id, "health_check", healthy, severity=0.5
                )
                
            except Exception as e:
                logger.error(f"Health check failed for peer {peer_id}: {e}")
                results[peer_id] = False
        
        return results
    
    def get_peer_info(self, peer_id: str) -> Optional[PeerInfo]:
        """Get peer information."""
        if peer_id in self.connections:
            return self.connections[peer_id].peer_info
        return self.peer_info_cache.get(peer_id)
    
    def get_peer_metrics(self, peer_id: str) -> Optional[PeerMetrics]:
        """Get peer metrics."""
        if peer_id in self.connections:
            return self.connections[peer_id].metrics
        return None
    
    def get_peer_reputation(self, peer_id: str) -> Optional[PeerReputation]:
        """Get peer reputation."""
        return self.reputation_manager.get_reputation(peer_id)
    
    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        return list(self.connections.keys())
    
    def get_peer_count(self) -> int:
        """Get total number of connected peers."""
        return len(self.connections)
    
    async def cleanup_inactive_peers(self) -> int:
        """Clean up inactive peers."""
        try:
            current_time = time.time()
            inactive_timeout = self.config.get("inactive_timeout", 600)  # 10 minutes
            
            peers_to_remove = []
            
            for peer_id, connection in self.connections.items():
                if current_time - connection.last_activity > inactive_timeout:
                    peers_to_remove.append(peer_id)
            
            # Disconnect inactive peers
            for peer_id in peers_to_remove:
                await self.disconnect_peer(peer_id)
            
            logger.info(f"Cleaned up {len(peers_to_remove)} inactive peers")
            return len(peers_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup inactive peers: {e}")
            return 0

__all__ = [
    "PeerManager",
    "PeerAuthenticator",
    "PeerHealthMonitor",
    "PeerReputationManager",
    "Peer",
    "PeerNode",
    "PeerConfig",
    "PeerInfo",
    "PeerMetrics",
    "PeerReputation",
    "PeerConnection",
    "PeerStatus",
    "PeerConnectionStatus",
    "PeerRole",
    "ConnectionType",
]