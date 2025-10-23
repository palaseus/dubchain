"""
gRPC Interceptors for DubChain API

This module provides gRPC interceptors for authentication, logging,
rate limiting, monitoring, and error handling.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import grpc
from grpc import aio
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from collections import defaultdict, deque

from ..common.auth import APIAuth, RoleBasedAuth
from ..common.rate_limit import RateLimiter, RateLimitConfig
from ...logging import get_logger

logger = get_logger(__name__)

@dataclass
class InterceptorConfig:
    """Configuration for gRPC interceptors."""
    enable_auth: bool = True
    enable_rate_limiting: bool = True
    enable_logging: bool = True
    enable_metrics: bool = True
    rate_limit_config: Optional[RateLimitConfig] = None
    auth_config: Optional[Dict[str, Any]] = None

class AuthInterceptor:
    """Authentication interceptor for gRPC services."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize auth interceptor."""
        self.config = config
        self.auth = APIAuth(config.auth_config or {})
        self.rbac = RoleBasedAuth()
        logger.info("Initialized gRPC auth interceptor")
    
    def __call__(self, request, context, next_handler):
        """Intercept gRPC requests for authentication."""
        try:
            # Extract metadata
            metadata = dict(context.invocation_metadata())
            
            # Get API key from metadata
            api_key = metadata.get('api-key')
            if not api_key:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "API key required")
                return
            
            # Authenticate API key
            if not self.auth.validate_api_key(api_key):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")
                return
            
            # Get user info
            user_info = self.auth.get_user_info(api_key)
            if not user_info:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found")
                return
            
            # Check permissions for the service method
            service_method = context.method()
            required_permission = self._get_required_permission(service_method)
            
            if required_permission and not self.rbac.has_permission(
                user_info.get('roles', []), required_permission
            ):
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "Insufficient permissions")
                return
            
            # Add user info to context
            context.set_details(f"user_id:{user_info.get('user_id')}")
            
            # Continue to next handler
            return next_handler(request, context)
            
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.error(f"Error in auth interceptor: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Authentication error")
    
    def _get_required_permission(self, service_method: str) -> Optional[str]:
        """Get required permission for service method."""
        # Map service methods to permissions
        permission_map = {
            'BlockchainService/GetBlock': 'blockchain.read',
            'BlockchainService/GetBlocks': 'blockchain.read',
            'BlockchainService/StreamBlocks': 'blockchain.read',
            'BlockchainService/GetTransaction': 'blockchain.read',
            'BlockchainService/CreateTransaction': 'transaction.create',
            'BlockchainService/StreamTransactions': 'blockchain.read',
            'BlockchainService/GetAccount': 'blockchain.read',
            'BlockchainService/GetContract': 'contract.read',
            'BlockchainService/DeployContract': 'contract.deploy',
            'BlockchainService/CallContract': 'contract.call',
            'BlockchainService/StreamContractEvents': 'contract.read',
            'WalletService/CreateWallet': 'wallet.create',
            'WalletService/GetWallet': 'wallet.read',
            'WalletService/ListWallets': 'wallet.read',
            'BridgeService/CreateBridgeTransfer': 'bridge.transfer',
            'BridgeService/GetBridgeTransfer': 'bridge.read',
            'BridgeService/StreamBridgeTransfers': 'bridge.read',
            'GovernanceService/CreateProposal': 'governance.propose',
            'GovernanceService/GetProposal': 'governance.read',
            'GovernanceService/Vote': 'governance.vote',
            'GovernanceService/StreamGovernanceUpdates': 'governance.read',
            'NetworkService/GetNetworkStats': 'metrics.read',
            'NetworkService/GetPerformanceMetrics': 'metrics.read',
            'NetworkService/GetNodeInfo': 'metrics.read',
            'NetworkService/ListPeers': 'network.read',
            'ConsensusService/GetValidators': 'consensus.read',
            'ConsensusService/GetShards': 'consensus.read',
            'ConsensusService/GetShard': 'consensus.read',
        }
        
        return permission_map.get(service_method)

class RateLimitInterceptor:
    """Rate limiting interceptor for gRPC services."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize rate limit interceptor."""
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_config or RateLimitConfig())
        logger.info("Initialized gRPC rate limit interceptor")
    
    def __call__(self, request, context, next_handler):
        """Intercept gRPC requests for rate limiting."""
        try:
            # Extract client IP from context
            client_ip = self._get_client_ip(context)
            
            # Check rate limit
            if not self.rate_limiter.is_allowed(client_ip):
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Rate limit exceeded")
                return
            
            # Continue to next handler
            return next_handler(request, context)
            
        except grpc.RpcError:
            raise
        except Exception as e:
            logger.error(f"Error in rate limit interceptor: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Rate limiting error")
    
    def _get_client_ip(self, context) -> str:
        """Get client IP from gRPC context."""
        try:
            # Try to get IP from peer
            peer = context.peer()
            if peer:
                # Extract IP from peer string (format: "ipv4:127.0.0.1:port")
                if ':' in peer:
                    return peer.split(':')[1]
            return "unknown"
        except Exception:
            return "unknown"

class LoggingInterceptor:
    """Logging interceptor for gRPC services."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize logging interceptor."""
        self.config = config
        logger.info("Initialized gRPC logging interceptor")
    
    def __call__(self, request, context, next_handler):
        """Intercept gRPC requests for logging."""
        start_time = time.time()
        
        try:
            # Log request
            logger.info(f"gRPC request: {context.method()}")
            
            # Continue to next handler
            response = next_handler(request, context)
            
            # Log response
            duration = time.time() - start_time
            logger.info(f"gRPC response: {context.method()} - {duration:.3f}s")
            
            return response
            
        except grpc.RpcError as e:
            # Log error
            duration = time.time() - start_time
            logger.error(f"gRPC error: {context.method()} - {e.code()} - {duration:.3f}s")
            raise
        except Exception as e:
            # Log unexpected error
            duration = time.time() - start_time
            logger.error(f"gRPC unexpected error: {context.method()} - {e} - {duration:.3f}s")
            context.abort(grpc.StatusCode.INTERNAL, "Internal server error")

class MetricsInterceptor:
    """Metrics interceptor for gRPC services."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize metrics interceptor."""
        self.config = config
        self.request_counts = defaultdict(int)
        self.request_durations = defaultdict(list)
        self.error_counts = defaultdict(int)
        logger.info("Initialized gRPC metrics interceptor")
    
    def __call__(self, request, context, next_handler):
        """Intercept gRPC requests for metrics collection."""
        start_time = time.time()
        method = context.method()
        
        try:
            # Increment request count
            self.request_counts[method] += 1
            
            # Continue to next handler
            response = next_handler(request, context)
            
            # Record duration
            duration = time.time() - start_time
            self.request_durations[method].append(duration)
            
            # Keep only last 1000 durations per method
            if len(self.request_durations[method]) > 1000:
                self.request_durations[method] = self.request_durations[method][-1000:]
            
            return response
            
        except grpc.RpcError as e:
            # Record error
            duration = time.time() - start_time
            self.error_counts[f"{method}:{e.code()}"] += 1
            self.request_durations[method].append(duration)
            raise
        except Exception as e:
            # Record unexpected error
            duration = time.time() - start_time
            self.error_counts[f"{method}:INTERNAL"] += 1
            self.request_durations[method].append(duration)
            context.abort(grpc.StatusCode.INTERNAL, "Internal server error")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        metrics = {
            "request_counts": dict(self.request_counts),
            "error_counts": dict(self.error_counts),
            "average_durations": {},
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
        }
        
        # Calculate average durations
        for method, durations in self.request_durations.items():
            if durations:
                metrics["average_durations"][method] = sum(durations) / len(durations)
        
        return metrics

class ErrorHandlingInterceptor:
    """Error handling interceptor for gRPC services."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize error handling interceptor."""
        self.config = config
        logger.info("Initialized gRPC error handling interceptor")
    
    def __call__(self, request, context, next_handler):
        """Intercept gRPC requests for error handling."""
        try:
            # Continue to next handler
            return next_handler(request, context)
            
        except grpc.RpcError:
            # Re-raise gRPC errors as-is
            raise
        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Validation error in {context.method()}: {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except PermissionError as e:
            # Handle permission errors
            logger.warning(f"Permission error in {context.method()}: {e}")
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))
        except FileNotFoundError as e:
            # Handle not found errors
            logger.warning(f"Not found error in {context.method()}: {e}")
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except TimeoutError as e:
            # Handle timeout errors
            logger.warning(f"Timeout error in {context.method()}: {e}")
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(e))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in {context.method()}: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal server error")

class InterceptorChain:
    """Chain of gRPC interceptors."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize interceptor chain."""
        self.config = config
        self.interceptors = []
        
        # Add interceptors based on configuration
        if config.enable_auth:
            self.interceptors.append(AuthInterceptor(config))
        
        if config.enable_rate_limiting:
            self.interceptors.append(RateLimitInterceptor(config))
        
        if config.enable_logging:
            self.interceptors.append(LoggingInterceptor(config))
        
        if config.enable_metrics:
            self.interceptors.append(MetricsInterceptor(config))
        
        # Always add error handling
        self.interceptors.append(ErrorHandlingInterceptor(config))
        
        logger.info(f"Initialized interceptor chain with {len(self.interceptors)} interceptors")
    
    def __call__(self, request, context, next_handler):
        """Execute interceptor chain."""
        def create_handler(index: int):
            """Create handler for interceptor at given index."""
            if index >= len(self.interceptors):
                return next_handler
            
            interceptor = self.interceptors[index]
            return lambda req, ctx: interceptor(req, ctx, create_handler(index + 1))
        
        return create_handler(0)(request, context)

class AsyncInterceptorChain:
    """Async chain of gRPC interceptors."""
    
    def __init__(self, config: InterceptorConfig):
        """Initialize async interceptor chain."""
        self.config = config
        self.interceptors = []
        
        # Add interceptors based on configuration
        if config.enable_auth:
            self.interceptors.append(AuthInterceptor(config))
        
        if config.enable_rate_limiting:
            self.interceptors.append(RateLimitInterceptor(config))
        
        if config.enable_logging:
            self.interceptors.append(LoggingInterceptor(config))
        
        if config.enable_metrics:
            self.interceptors.append(MetricsInterceptor(config))
        
        # Always add error handling
        self.interceptors.append(ErrorHandlingInterceptor(config))
        
        logger.info(f"Initialized async interceptor chain with {len(self.interceptors)} interceptors")
    
    async def __call__(self, request, context, next_handler):
        """Execute async interceptor chain."""
        async def create_handler(index: int):
            """Create async handler for interceptor at given index."""
            if index >= len(self.interceptors):
                return await next_handler(request, context)
            
            interceptor = self.interceptors[index]
            return await interceptor(request, context, create_handler(index + 1))
        
        return await create_handler(0)

def create_interceptor_chain(config: InterceptorConfig) -> InterceptorChain:
    """Create interceptor chain with given configuration."""
    return InterceptorChain(config)

def create_async_interceptor_chain(config: InterceptorConfig) -> AsyncInterceptorChain:
    """Create async interceptor chain with given configuration."""
    return AsyncInterceptorChain(config)

__all__ = [
    "InterceptorConfig",
    "AuthInterceptor",
    "RateLimitInterceptor", 
    "LoggingInterceptor",
    "MetricsInterceptor",
    "ErrorHandlingInterceptor",
    "InterceptorChain",
    "AsyncInterceptorChain",
    "create_interceptor_chain",
    "create_async_interceptor_chain",
]
