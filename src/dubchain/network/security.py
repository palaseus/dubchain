"""
Security mechanisms for GodChain P2P network.
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .peer import Peer, PeerInfo


class SecurityLevel(Enum):
    """Security levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class SecurityConfig:
    """Configuration for network security."""

    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_peer_authentication: bool = True
    enable_message_encryption: bool = True
    enable_ddos_protection: bool = True
    max_connections_per_ip: int = 10
    connection_rate_limit: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PeerAuthenticator:
    """Peer authentication system."""

    def __init__(self, config: SecurityConfig):
        """Initialize peer authenticator."""
        self.config = config
        self.authenticated_peers: Dict[str, Dict[str, Any]] = {}

    def authenticate_peer(self, peer_info: PeerInfo, challenge: bytes) -> bool:
        """Authenticate peer with challenge."""
        # Simple authentication for demo
        return True

    def is_peer_authenticated(self, peer_id: str) -> bool:
        """Check if peer is authenticated."""
        return peer_id in self.authenticated_peers


class MessageEncryption:
    """Message encryption system."""

    def __init__(self, config: SecurityConfig):
        """Initialize message encryption."""
        self.config = config

    def encrypt_message(self, message: bytes, key: bytes) -> bytes:
        """Encrypt message."""
        # Simple XOR encryption for demo
        return bytes(
            a ^ b for a, b in zip(message, key * (len(message) // len(key) + 1))
        )

    def decrypt_message(self, encrypted_message: bytes, key: bytes) -> bytes:
        """Decrypt message."""
        # Simple XOR decryption for demo
        return bytes(
            a ^ b
            for a, b in zip(
                encrypted_message, key * (len(encrypted_message) // len(key) + 1)
            )
        )


class DDoSProtection:
    """DDoS protection system."""

    def __init__(self, config: SecurityConfig):
        """Initialize DDoS protection."""
        self.config = config
        self.connection_counts: Dict[str, int] = {}
        self.rate_limits: Dict[str, List[float]] = {}

    def check_connection_limit(self, ip_address: str) -> bool:
        """Check if IP has exceeded connection limit."""
        current_count = self.connection_counts.get(ip_address, 0)
        return current_count < self.config.max_connections_per_ip

    def check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP has exceeded rate limit."""
        current_time = time.time()
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []

        # Remove old timestamps
        self.rate_limits[ip_address] = [
            timestamp
            for timestamp in self.rate_limits[ip_address]
            if current_time - timestamp < 60.0
        ]

        return (
            len(self.rate_limits[ip_address]) < self.config.connection_rate_limit * 60
        )


class NetworkSecurity:
    """Network security manager."""

    def __init__(self, config: SecurityConfig):
        """Initialize network security."""
        self.config = config
        self.authenticator = PeerAuthenticator(config)
        self.encryption = MessageEncryption(config)
        self.ddos_protection = DDoSProtection(config)

    def authenticate_peer(self, peer_info: PeerInfo, challenge: bytes) -> bool:
        """Authenticate peer."""
        return self.authenticator.authenticate_peer(peer_info, challenge)

    def encrypt_message(self, message: bytes, key: bytes) -> bytes:
        """Encrypt message."""
        return self.encryption.encrypt_message(message, key)

    def decrypt_message(self, encrypted_message: bytes, key: bytes) -> bytes:
        """Decrypt message."""
        return self.encryption.decrypt_message(encrypted_message, key)

    def check_security(self, ip_address: str) -> bool:
        """Check security constraints."""
        return self.ddos_protection.check_connection_limit(
            ip_address
        ) and self.ddos_protection.check_rate_limit(ip_address)
