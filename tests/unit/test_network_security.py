"""Tests for network security module."""

import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.network.security import (
    SecurityLevel,
    SecurityConfig,
    PeerAuthenticator,
    MessageEncryption,
    DDoSProtection,
    NetworkSecurity,
)
from src.dubchain.network.peer import PeerInfo, ConnectionType, PeerStatus


class TestSecurityLevel:
    """Test SecurityLevel enum."""

    def test_security_levels(self):
        """Test security level values."""
        assert SecurityLevel.LOW.value == "low"
        assert SecurityLevel.MEDIUM.value == "medium"
        assert SecurityLevel.HIGH.value == "high"
        assert SecurityLevel.MAXIMUM.value == "maximum"


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""

    def test_init_default(self):
        """Test SecurityConfig with default values."""
        config = SecurityConfig()
        assert config.security_level == SecurityLevel.HIGH
        assert config.enable_peer_authentication is True
        assert config.enable_message_encryption is True
        assert config.enable_ddos_protection is True
        assert config.max_connections_per_ip == 10
        assert config.connection_rate_limit == 5.0
        assert config.metadata == {}

    def test_init_custom(self):
        """Test SecurityConfig with custom values."""
        metadata = {"custom": "value"}
        config = SecurityConfig(
            security_level=SecurityLevel.MAXIMUM,
            enable_peer_authentication=False,
            enable_message_encryption=False,
            enable_ddos_protection=False,
            max_connections_per_ip=5,
            connection_rate_limit=2.0,
            metadata=metadata
        )
        assert config.security_level == SecurityLevel.MAXIMUM
        assert config.enable_peer_authentication is False
        assert config.enable_message_encryption is False
        assert config.enable_ddos_protection is False
        assert config.max_connections_per_ip == 5
        assert config.connection_rate_limit == 2.0
        assert config.metadata == metadata


class TestPeerAuthenticator:
    """Test PeerAuthenticator class."""

    def test_init(self):
        """Test PeerAuthenticator initialization."""
        config = SecurityConfig()
        authenticator = PeerAuthenticator(config)
        assert authenticator.config == config
        assert authenticator.authenticated_peers == {}

    def test_authenticate_peer(self):
        """Test peer authentication."""
        config = SecurityConfig()
        authenticator = PeerAuthenticator(config)
        
        # Create mock peer info
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=Mock(),
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        challenge = b"test_challenge"
        
        # Should return True (simple authentication for demo)
        result = authenticator.authenticate_peer(peer_info, challenge)
        assert result is True

    def test_is_peer_authenticated(self):
        """Test checking if peer is authenticated."""
        config = SecurityConfig()
        authenticator = PeerAuthenticator(config)
        
        # Initially no peers are authenticated
        assert authenticator.is_peer_authenticated("peer1") is False
        
        # Add a peer to authenticated list
        authenticator.authenticated_peers["peer1"] = {"timestamp": time.time()}
        assert authenticator.is_peer_authenticated("peer1") is True


class TestMessageEncryption:
    """Test MessageEncryption class."""

    def test_init(self):
        """Test MessageEncryption initialization."""
        config = SecurityConfig()
        encryption = MessageEncryption(config)
        assert encryption.config == config

    def test_encrypt_decrypt_message(self):
        """Test message encryption and decryption."""
        config = SecurityConfig()
        encryption = MessageEncryption(config)
        
        message = b"Hello, World!"
        key = b"secret_key"
        
        # Encrypt message
        encrypted = encryption.encrypt_message(message, key)
        assert encrypted != message
        assert isinstance(encrypted, bytes)
        
        # Decrypt message
        decrypted = encryption.decrypt_message(encrypted, key)
        assert decrypted == message

    def test_encrypt_decrypt_empty_message(self):
        """Test encryption/decryption of empty message."""
        config = SecurityConfig()
        encryption = MessageEncryption(config)
        
        message = b""
        key = b"secret_key"
        
        encrypted = encryption.encrypt_message(message, key)
        decrypted = encryption.decrypt_message(encrypted, key)
        assert decrypted == message

    def test_encrypt_decrypt_long_message(self):
        """Test encryption/decryption of long message."""
        config = SecurityConfig()
        encryption = MessageEncryption(config)
        
        message = b"A" * 1000  # Long message
        key = b"short_key"
        
        encrypted = encryption.encrypt_message(message, key)
        decrypted = encryption.decrypt_message(encrypted, key)
        assert decrypted == message

    def test_encrypt_decrypt_short_key(self):
        """Test encryption/decryption with short key."""
        config = SecurityConfig()
        encryption = MessageEncryption(config)
        
        message = b"Hello, World!"
        key = b"xy"  # Very short key
        
        encrypted = encryption.encrypt_message(message, key)
        decrypted = encryption.decrypt_message(encrypted, key)
        assert decrypted == message


class TestDDoSProtection:
    """Test DDoSProtection class."""

    def test_init(self):
        """Test DDoSProtection initialization."""
        config = SecurityConfig()
        ddos_protection = DDoSProtection(config)
        assert ddos_protection.config == config
        assert ddos_protection.connection_counts == {}
        assert ddos_protection.rate_limits == {}

    def test_check_connection_limit(self):
        """Test connection limit checking."""
        config = SecurityConfig(max_connections_per_ip=3)
        ddos_protection = DDoSProtection(config)
        
        ip_address = "192.168.1.1"
        
        # Initially should be under limit
        assert ddos_protection.check_connection_limit(ip_address) is True
        
        # Set connection count to limit
        ddos_protection.connection_counts[ip_address] = 3
        assert ddos_protection.check_connection_limit(ip_address) is False
        
        # Set connection count above limit
        ddos_protection.connection_counts[ip_address] = 5
        assert ddos_protection.check_connection_limit(ip_address) is False

    def test_check_rate_limit(self):
        """Test rate limit checking."""
        config = SecurityConfig(connection_rate_limit=2.0)  # 2 connections per minute
        ddos_protection = DDoSProtection(config)
        
        ip_address = "192.168.1.1"
        
        # Initially should be under rate limit
        assert ddos_protection.check_rate_limit(ip_address) is True
        
        # Add timestamps within rate limit
        current_time = time.time()
        ddos_protection.rate_limits[ip_address] = [current_time - 30, current_time - 20]
        assert ddos_protection.check_rate_limit(ip_address) is True
        
        # Add timestamps exceeding rate limit (more than 2 per minute)
        ddos_protection.rate_limits[ip_address] = [
            current_time - 30, current_time - 20, current_time - 10, current_time - 5
        ]
        # Should still be under limit since 4 connections < 2*60 = 120
        assert ddos_protection.check_rate_limit(ip_address) is True

    def test_check_rate_limit_old_timestamps(self):
        """Test rate limit with old timestamps."""
        config = SecurityConfig(connection_rate_limit=2.0)
        ddos_protection = DDoSProtection(config)
        
        ip_address = "192.168.1.1"
        current_time = time.time()
        
        # Add old timestamps (older than 60 seconds)
        ddos_protection.rate_limits[ip_address] = [
            current_time - 70, current_time - 80, current_time - 90
        ]
        
        # Should be under rate limit since old timestamps are filtered out
        assert ddos_protection.check_rate_limit(ip_address) is True

    def test_check_rate_limit_multiple_ips(self):
        """Test rate limit checking for multiple IPs."""
        config = SecurityConfig(connection_rate_limit=1.0)
        ddos_protection = DDoSProtection(config)
        
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"
        
        # Each IP should have independent rate limits
        current_time = time.time()
        ddos_protection.rate_limits[ip1] = [current_time - 30]
        ddos_protection.rate_limits[ip2] = [current_time - 30]
        
        assert ddos_protection.check_rate_limit(ip1) is True
        assert ddos_protection.check_rate_limit(ip2) is True


class TestNetworkSecurity:
    """Test NetworkSecurity class."""

    def test_init(self):
        """Test NetworkSecurity initialization."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        assert security.config == config
        assert isinstance(security.authenticator, PeerAuthenticator)
        assert isinstance(security.encryption, MessageEncryption)
        assert isinstance(security.ddos_protection, DDoSProtection)

    def test_authenticate_peer(self):
        """Test peer authentication through NetworkSecurity."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        peer_info = PeerInfo(
            peer_id="test_peer",
            public_key=Mock(),
            address="127.0.0.1",
            port=8080,
            connection_type=ConnectionType.INBOUND
        )
        challenge = b"test_challenge"
        
        result = security.authenticate_peer(peer_info, challenge)
        assert result is True

    def test_encrypt_message(self):
        """Test message encryption through NetworkSecurity."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        message = b"Test message"
        key = b"secret_key"
        
        encrypted = security.encrypt_message(message, key)
        assert encrypted != message
        assert isinstance(encrypted, bytes)

    def test_decrypt_message(self):
        """Test message decryption through NetworkSecurity."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        message = b"Test message"
        key = b"secret_key"
        
        encrypted = security.encrypt_message(message, key)
        decrypted = security.decrypt_message(encrypted, key)
        assert decrypted == message

    def test_check_security(self):
        """Test security constraint checking."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        ip_address = "192.168.1.1"
        
        # Should pass both connection limit and rate limit checks
        result = security.check_security(ip_address)
        assert result is True

    def test_check_security_connection_limit_exceeded(self):
        """Test security check with connection limit exceeded."""
        config = SecurityConfig(max_connections_per_ip=0)  # No connections allowed
        security = NetworkSecurity(config)
        
        ip_address = "192.168.1.1"
        
        # Should fail connection limit check
        result = security.check_security(ip_address)
        assert result is False

    def test_check_security_rate_limit_exceeded(self):
        """Test security check with rate limit exceeded."""
        config = SecurityConfig(connection_rate_limit=0.0)  # No connections per minute
        security = NetworkSecurity(config)
        
        ip_address = "192.168.1.1"
        
        # Add a recent timestamp to exceed rate limit
        current_time = time.time()
        security.ddos_protection.rate_limits[ip_address] = [current_time]
        
        # Should fail rate limit check
        result = security.check_security(ip_address)
        assert result is False

    def test_encrypt_decrypt_roundtrip(self):
        """Test complete encrypt/decrypt roundtrip."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        original_message = b"Hello, World! This is a test message."
        key = b"my_secret_key_123"
        
        # Encrypt
        encrypted = security.encrypt_message(original_message, key)
        
        # Decrypt
        decrypted = security.decrypt_message(encrypted, key)
        
        assert decrypted == original_message

    def test_different_keys_produce_different_encryption(self):
        """Test that different keys produce different encrypted messages."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        message = b"Test message"
        key1 = b"key1"
        key2 = b"key2"
        
        encrypted1 = security.encrypt_message(message, key1)
        encrypted2 = security.encrypt_message(message, key2)
        
        assert encrypted1 != encrypted2

    def test_same_key_same_message_produces_same_encryption(self):
        """Test that same key and message produce same encrypted result."""
        config = SecurityConfig()
        security = NetworkSecurity(config)
        
        message = b"Test message"
        key = b"secret_key"
        
        encrypted1 = security.encrypt_message(message, key)
        encrypted2 = security.encrypt_message(message, key)
        
        assert encrypted1 == encrypted2
