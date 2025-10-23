"""
Common Authentication Infrastructure for DubChain API.

This module provides unified authentication across all API protocols
(GraphQL, REST, gRPC) with JWT tokens, signature verification, and
role-based access control.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from ...crypto.signatures import PrivateKey, PublicKey, Signature
from ...wallet.wallet_manager import WalletManager

class AuthError(Exception):
    """Authentication error."""
    pass

class TokenError(AuthError):
    """Token-related error."""
    pass

class SignatureError(AuthError):
    """Signature verification error."""
    pass

class JWTAuth:
    """JWT-based authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """Initialize JWT auth."""
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
        self.refresh_expiry = timedelta(days=7)
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token."""
        payload = {
            "user_id": user_data.get("user_id"),
            "address": user_data.get("address"),
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", []),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.token_expiry,
            "jti": secrets.token_urlsafe(32)  # JWT ID
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token."""
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.refresh_expiry,
            "jti": secrets.token_urlsafe(32)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenError("Token has expired")
        except jwt.InvalidTokenError:
            raise TokenError("Invalid token")
    
    def refresh_token(self, refresh_token: str) -> str:
        """Refresh access token."""
        try:
            payload = self.verify_token(refresh_token)
            if payload.get("type") != "refresh":
                raise TokenError("Invalid refresh token")
            
            # Create new access token
            user_data = {
                "user_id": payload["user_id"],
                "address": payload.get("address"),
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", [])
            }
            
            return self.create_token(user_data)
        except Exception as e:
            raise TokenError(f"Token refresh failed: {str(e)}")

class SignatureAuth:
    """Signature-based authentication."""
    
    def __init__(self):
        """Initialize signature auth."""
        self.nonce_cache: Dict[str, float] = {}
        self.nonce_ttl = 300  # 5 minutes
    
    def generate_nonce(self, address: str) -> str:
        """Generate nonce for signature challenge."""
        nonce = secrets.token_urlsafe(32)
        self.nonce_cache[nonce] = {
            "address": address,
            "timestamp": time.time()
        }
        return nonce
    
    def verify_signature(self, address: str, signature: str, message: str) -> bool:
        """Verify signature for authentication."""
        try:
            # Parse signature
            sig_bytes = bytes.fromhex(signature)
            if len(sig_bytes) != 64:
                return False
            
            # Create signature object
            sig = Signature.from_bytes(sig_bytes)
            
            # Get public key from address (simplified)
            public_key = self._get_public_key_from_address(address)
            if not public_key:
                return False
            
            # Verify signature
            return public_key.verify_signature(message.encode(), sig)
        except Exception:
            return False
    
    def _get_public_key_from_address(self, address: str) -> Optional[PublicKey]:
        """Get public key from address."""
        # This would integrate with the wallet system
        # For now, return a placeholder
        return None
    
    def cleanup_expired_nonces(self):
        """Clean up expired nonces."""
        current_time = time.time()
        expired_nonces = [
            nonce for nonce, data in self.nonce_cache.items()
            if current_time - data["timestamp"] > self.nonce_ttl
        ]
        for nonce in expired_nonces:
            del self.nonce_cache[nonce]

class RoleBasedAuth:
    """Role-based access control."""
    
    def __init__(self):
        """Initialize RBAC."""
        self.roles = {
            "admin": {
                "permissions": ["*"],  # All permissions
                "description": "Administrator with full access"
            },
            "validator": {
                "permissions": [
                    "consensus.participate",
                    "consensus.propose",
                    "consensus.vote",
                    "network.peer",
                    "metrics.read"
                ],
                "description": "Validator with consensus participation rights"
            },
            "governor": {
                "permissions": [
                    "governance.propose",
                    "governance.vote",
                    "governance.execute",
                    "metrics.read"
                ],
                "description": "Governance participant"
            },
            "user": {
                "permissions": [
                    "wallet.create",
                    "wallet.read",
                    "transaction.create",
                    "contract.deploy",
                    "contract.call",
                    "bridge.transfer",
                    "metrics.read"
                ],
                "description": "Regular user with basic blockchain access"
            },
            "readonly": {
                "permissions": [
                    "blockchain.read",
                    "transaction.read",
                    "contract.read",
                    "metrics.read"
                ],
                "description": "Read-only access"
            }
        }
    
    def has_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        for role in user_roles:
            if role in self.roles:
                permissions = self.roles[role]["permissions"]
                if "*" in permissions or required_permission in permissions:
                    return True
        return False
    
    def get_user_permissions(self, user_roles: List[str]) -> List[str]:
        """Get all permissions for user roles."""
        permissions = set()
        for role in user_roles:
            if role in self.roles:
                permissions.update(self.roles[role]["permissions"])
        return list(permissions)
    
    def add_role(self, role_name: str, permissions: List[str], description: str = ""):
        """Add custom role."""
        self.roles[role_name] = {
            "permissions": permissions,
            "description": description
        }
    
    def remove_role(self, role_name: str):
        """Remove role."""
        if role_name in self.roles:
            del self.roles[role_name]

class AuthManager:
    """Unified authentication manager."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize auth manager."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.jwt_auth = JWTAuth(self.secret_key)
        self.signature_auth = SignatureAuth()
        self.rbac = RoleBasedAuth()
        self.wallet_manager = WalletManager()
        
        # User sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_ttl = 3600  # 1 hour
    
    async def authenticate_user(self, address: str, signature: str, message: str) -> Dict[str, Any]:
        """Authenticate user with signature."""
        try:
            # Verify signature
            if not self.signature_auth.verify_signature(address, signature, message):
                raise AuthError("Invalid signature")
            
            # Get user data
            user_data = await self._get_user_data(address)
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            self.active_sessions[session_id] = {
                "user_data": user_data,
                "created_at": time.time(),
                "last_activity": time.time()
            }
            
            # Create tokens
            access_token = self.jwt_auth.create_token(user_data)
            refresh_token = self.jwt_auth.create_refresh_token(user_data["user_id"])
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "session_id": session_id,
                "user": user_data,
                "expires_in": int(self.jwt_auth.token_expiry.total_seconds())
            }
        except Exception as e:
            raise AuthError(f"Authentication failed: {str(e)}")
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = self.jwt_auth.verify_token(token)
            
            # Check if session is still active
            user_id = payload.get("user_id")
            if user_id and user_id in self.active_sessions:
                session = self.active_sessions[user_id]
                if time.time() - session["last_activity"] > self.session_ttl:
                    del self.active_sessions[user_id]
                    raise TokenError("Session expired")
                
                # Update last activity
                session["last_activity"] = time.time()
            
            return payload
        except Exception as e:
            raise TokenError(f"Token verification failed: {str(e)}")
    
    async def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh access token."""
        return self.jwt_auth.refresh_token(refresh_token)
    
    async def logout(self, user_id: str):
        """Logout user."""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
    
    def has_permission(self, user_roles: List[str], permission: str) -> bool:
        """Check if user has permission."""
        return self.rbac.has_permission(user_roles, permission)
    
    def get_user_permissions(self, user_roles: List[str]) -> List[str]:
        """Get user permissions."""
        return self.rbac.get_user_permissions(user_roles)
    
    async def _get_user_data(self, address: str) -> Dict[str, Any]:
        """Get user data from address."""
        # This would integrate with the wallet system
        # For now, return basic user data
        return {
            "user_id": address,
            "address": address,
            "roles": ["user"],  # Default role
            "permissions": self.rbac.get_user_permissions(["user"]),
            "created_at": time.time(),
            "last_login": time.time()
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = [
            user_id for user_id, session in self.active_sessions.items()
            if current_time - session["last_activity"] > self.session_ttl
        ]
        for user_id in expired_sessions:
            del self.active_sessions[user_id]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "active_sessions": len(self.active_sessions),
            "session_ttl": self.session_ttl,
            "roles_defined": len(self.rbac.roles),
            "permissions_defined": sum(len(role["permissions"]) for role in self.rbac.roles.values())
        }

class APIKeyAuth:
    """API key authentication."""
    
    def __init__(self):
        """Initialize API key auth."""
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_prefix = "dub_"
    
    def generate_api_key(self, user_id: str, permissions: List[str], expires_at: Optional[datetime] = None) -> str:
        """Generate API key."""
        key_id = secrets.token_urlsafe(16)
        api_key = f"{self.key_prefix}{key_id}"
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0
        }
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key."""
        if api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        
        # Check expiration
        if key_data["expires_at"] and datetime.utcnow() > key_data["expires_at"]:
            del self.api_keys[api_key]
            return None
        
        # Update usage stats
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False
    
    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List API keys for user."""
        return [
            {"key": key, **data}
            for key, data in self.api_keys.items()
            if data["user_id"] == user_id
        ]

# Global auth manager instance
auth_manager = AuthManager()
