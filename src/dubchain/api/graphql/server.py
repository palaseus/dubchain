"""
GraphQL Server for DubChain API.

This module provides the GraphQL server implementation with authentication,
rate limiting, caching, and WebSocket support.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_WS_PROTOCOL
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .schema import schema
from .resolvers import QueryResolvers, MutationResolvers, SubscriptionResolvers
from ..common.auth import AuthManager, JWTAuth
from ..common.rate_limit import RateLimiter
from ..common.cache import CacheManager
from ..common.monitoring import MetricsCollector

# Security
security = HTTPBearer()

# Global instances
auth_manager = AuthManager()
rate_limiter = RateLimiter()
cache_manager = CacheManager()
metrics_collector = MetricsCollector()

class GraphQLServer:
    """GraphQL server implementation."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize GraphQL server."""
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="DubChain GraphQL API",
            description="Advanced blockchain API with GraphQL support",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup GraphQL router
        self.graphql_app = GraphQLRouter(
            schema=schema,
            context_getter=self.get_context,
            subscription_protocols=[GRAPHQL_WS_PROTOCOL]
        )
        
        self.app.include_router(self.graphql_app, prefix="/graphql")
        
        # Add health check endpoint
        self.app.add_api_route("/health", self.health_check, methods=["GET"])
        
        # Add metrics endpoint
        self.app.add_api_route("/metrics", self.metrics_endpoint, methods=["GET"])
    
    async def get_context(self, request: Request) -> Dict[str, Any]:
        """Get GraphQL context."""
        # Extract authentication
        auth_token = None
        if "authorization" in request.headers:
            auth_header = request.headers["authorization"]
            if auth_header.startswith("Bearer "):
                auth_token = auth_header[7:]
        
        # Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        await rate_limiter.check_rate_limit(client_ip)
        
        # Get user from token
        user = None
        if auth_token:
            try:
                user = await auth_manager.verify_token(auth_token)
            except Exception:
                pass  # Invalid token, user remains None
        
        return {
            "request": request,
            "user": user,
            "cache": cache_manager,
            "metrics": metrics_collector,
            "auth_manager": auth_manager,
            "rate_limiter": rate_limiter
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "services": {
                "blockchain": "running",
                "wallet": "running",
                "bridge": "running",
                "sharding": "running",
                "consensus": "running",
                "governance": "running"
            }
        }
    
    async def metrics_endpoint(self) -> Dict[str, Any]:
        """Metrics endpoint."""
        return await metrics_collector.get_metrics()
    
    def run(self, debug: bool = False):
        """Run the GraphQL server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            log_level="info" if not debug else "debug"
        )

# Authentication middleware
class AuthMiddleware:
    """Authentication middleware for GraphQL."""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    async def authenticate(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate request."""
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header[7:]
        try:
            return await self.auth_manager.verify_token(token)
        except Exception:
            return None

# Rate limiting middleware
class RateLimitMiddleware:
    """Rate limiting middleware."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def check_rate_limit(self, client_ip: str, operation: str = "query") -> bool:
        """Check rate limit for client."""
        return await self.rate_limiter.check_rate_limit(
            client_ip, 
            operation=operation
        )

# Cache middleware
class CacheMiddleware:
    """Cache middleware for GraphQL queries."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    async def get_cached_result(self, query: str, variables: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self._generate_cache_key(query, variables)
        return await self.cache_manager.get(cache_key)
    
    async def cache_result(self, query: str, variables: Dict[str, Any], result: Any, ttl: int = 300):
        """Cache query result."""
        cache_key = self._generate_cache_key(query, variables)
        await self.cache_manager.set(cache_key, result, ttl=ttl)
    
    def _generate_cache_key(self, query: str, variables: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        import hashlib
        key_data = f"{query}:{json.dumps(variables, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

# Error handling
class GraphQLErrorHandler:
    """GraphQL error handler."""
    
    @staticmethod
    def format_error(error: Exception) -> Dict[str, Any]:
        """Format GraphQL error."""
        return {
            "message": str(error),
            "locations": [],
            "path": [],
            "extensions": {
                "code": "INTERNAL_ERROR",
                "timestamp": time.time()
            }
        }

# Subscription manager
class SubscriptionManager:
    """Manages GraphQL subscriptions."""
    
    def __init__(self):
        self.active_subscriptions: Dict[str, Any] = {}
        self.subscription_handlers: Dict[str, Any] = {}
    
    async def add_subscription(self, subscription_id: str, handler: Any):
        """Add active subscription."""
        self.active_subscriptions[subscription_id] = handler
        self.subscription_handlers[subscription_id] = handler
    
    async def remove_subscription(self, subscription_id: str):
        """Remove subscription."""
        if subscription_id in self.active_subscriptions:
            del self.active_subscriptions[subscription_id]
        if subscription_id in self.subscription_handlers:
            del self.subscription_handlers[subscription_id]
    
    async def broadcast_to_subscribers(self, event_type: str, data: Any):
        """Broadcast data to subscribers."""
        for subscription_id, handler in self.subscription_handlers.items():
            if event_type in handler.subscribed_events:
                try:
                    await handler.send_data(data)
                except Exception as e:
                    logger.info(f"Error broadcasting to subscription {subscription_id}: {e}")

# WebSocket connection manager
class WebSocketManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.connection_count = 0
    
    async def add_connection(self, connection_id: str, websocket: Any):
        """Add WebSocket connection."""
        self.connections[connection_id] = websocket
        self.connection_count += 1
    
    async def remove_connection(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.connections:
            del self.connections[connection_id]
            self.connection_count -= 1
    
    async def send_to_connection(self, connection_id: str, data: Any):
        """Send data to specific connection."""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send_text(json.dumps(data))
            except Exception as e:
                logger.info(f"Error sending to connection {connection_id}: {e}")
    
    async def broadcast(self, data: Any):
        """Broadcast data to all connections."""
        for connection_id, websocket in self.connections.items():
            try:
                await websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.info(f"Error broadcasting to connection {connection_id}: {e}")

# Main server instance
def create_graphql_server(host: str = "0.0.0.0", port: int = 8000) -> GraphQLServer:
    """Create GraphQL server instance."""
    return GraphQLServer(host=host, port=port)

# CLI entry point
if __name__ == "__main__":
    server = create_graphql_server()
    server.run(debug=True)
