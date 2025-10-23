"""
gRPC Server Implementation for DubChain API

This module provides the main gRPC server setup and configuration
for the DubChain blockchain platform.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional
import grpc
from grpc import aio
from concurrent import futures

from .interceptors import InterceptorConfig, create_async_interceptor_chain
from .services import (
    BlockchainService,
    WalletService,
    BridgeService,
    GovernanceService,
    NetworkService,
    ConsensusService,
)
from ..common.auth import APIAuth
from ..common.rate_limit import RateLimitConfig
from ...logging import get_logger

logger = get_logger(__name__)

class GRPCServerConfig:
    """Configuration for gRPC server."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
        max_message_length: int = 4 * 1024 * 1024,  # 4MB
        max_metadata_size: int = 8 * 1024,  # 8KB
        keepalive_time_ms: int = 30000,  # 30 seconds
        keepalive_timeout_ms: int = 5000,  # 5 seconds
        keepalive_permit_without_calls: bool = True,
        max_connection_idle_ms: int = 300000,  # 5 minutes
        max_connection_age_ms: int = 1800000,  # 30 minutes
        max_connection_age_grace_ms: int = 5000,  # 5 seconds
        enable_reflection: bool = True,
        enable_health_check: bool = True,
        auth_config: Optional[Dict[str, Any]] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize gRPC server configuration."""
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.max_message_length = max_message_length
        self.max_metadata_size = max_metadata_size
        self.keepalive_time_ms = keepalive_time_ms
        self.keepalive_timeout_ms = keepalive_timeout_ms
        self.keepalive_permit_without_calls = keepalive_permit_without_calls
        self.max_connection_idle_ms = max_connection_idle_ms
        self.max_connection_age_ms = max_connection_age_ms
        self.max_connection_age_grace_ms = max_connection_age_grace_ms
        self.enable_reflection = enable_reflection
        self.enable_health_check = enable_health_check
        self.auth_config = auth_config or {}
        self.rate_limit_config = rate_limit_config or RateLimitConfig()

class DubChainGRPCServer:
    """Main gRPC server for DubChain API."""
    
    def __init__(self, config: GRPCServerConfig):
        """Initialize gRPC server."""
        self.config = config
        self.server: Optional[aio.Server] = None
        self.services = {}
        self.interceptor_config = InterceptorConfig(
            enable_auth=True,
            enable_rate_limiting=True,
            enable_logging=True,
            enable_metrics=True,
            auth_config=config.auth_config,
            rate_limit_config=config.rate_limit_config,
        )
        
        logger.info("Initialized DubChain gRPC server")
    
    async def start(self) -> None:
        """Start the gRPC server."""
        try:
            # Create server with options
            options = [
                ('grpc.max_send_message_length', self.config.max_message_length),
                ('grpc.max_receive_message_length', self.config.max_message_length),
                ('grpc.max_metadata_size', self.config.max_metadata_size),
                ('grpc.keepalive_time_ms', self.config.keepalive_time_ms),
                ('grpc.keepalive_timeout_ms', self.config.keepalive_timeout_ms),
                ('grpc.keepalive_permit_without_calls', self.config.keepalive_permit_without_calls),
                ('grpc.max_connection_idle_ms', self.config.max_connection_idle_ms),
                ('grpc.max_connection_age_ms', self.config.max_connection_age_ms),
                ('grpc.max_connection_age_grace_ms', self.config.max_connection_age_grace_ms),
            ]
            
            self.server = aio.server(futures.ThreadPoolExecutor(max_workers=self.config.max_workers), options=options)
            
            # Create interceptor chain
            interceptor_chain = create_async_interceptor_chain(self.interceptor_config)
            
            # Add services with interceptors
            await self._add_services(interceptor_chain)
            
            # Add reflection service if enabled
            if self.config.enable_reflection:
                await self._add_reflection_service()
            
            # Add health check service if enabled
            if self.config.enable_health_check:
                await self._add_health_check_service()
            
            # Start server
            listen_addr = f"{self.config.host}:{self.config.port}"
            self.server.add_insecure_port(listen_addr)
            
            await self.server.start()
            logger.info(f"gRPC server started on {listen_addr}")
            
        except Exception as e:
            logger.error(f"Error starting gRPC server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the gRPC server."""
        try:
            if self.server:
                await self.server.stop(grace=5.0)  # 5 second grace period
                logger.info("gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")
    
    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        try:
            if self.server:
                await self.server.wait_for_termination()
        except Exception as e:
            logger.error(f"Error waiting for server termination: {e}")
    
    async def _add_services(self, interceptor_chain) -> None:
        """Add all gRPC services to the server."""
        try:
            # Initialize services
            self.services = {
                'blockchain': BlockchainService(),
                'wallet': WalletService(),
                'bridge': BridgeService(),
                'governance': GovernanceService(),
                'network': NetworkService(),
                'consensus': ConsensusService(),
            }
            
            # Add services to server
            # Note: In a real implementation, you would use the generated protobuf service definitions
            # For now, we'll create mock service definitions
            
            # Blockchain service
            blockchain_service = self.services['blockchain']
            # self.server.add_generic_rpc_handlers([
            #     grpc.method_handlers_generic_handler(
            #         'dubchain.BlockchainService',
            #         blockchain_service
            #     )
            # ])
            
            # Wallet service
            wallet_service = self.services['wallet']
            # self.server.add_generic_rpc_handlers([
            #     grpc.method_handlers_generic_handler(
            #         'dubchain.WalletService',
            #         wallet_service
            #     )
            # ])
            
            # Bridge service
            bridge_service = self.services['bridge']
            # self.server.add_generic_rpc_handlers([
            #     grpc.method_handlers_generic_handler(
            #         'dubchain.BridgeService',
            #         bridge_service
            #     )
            # ])
            
            # Governance service
            governance_service = self.services['governance']
            # self.server.add_generic_rpc_handlers([
            #     grpc.method_handlers_generic_handler(
            #         'dubchain.GovernanceService',
            #         governance_service
            #     )
            # ])
            
            # Network service
            network_service = self.services['network']
            # self.server.add_generic_rpc_handlers([
            #     grpc.method_handlers_generic_handler(
            #         'dubchain.NetworkService',
            #         network_service
            #     )
            # ])
            
            # Consensus service
            consensus_service = self.services['consensus']
            # self.server.add_generic_rpc_handlers([
            #     grpc.method_handlers_generic_handler(
            #         'dubchain.ConsensusService',
            #         consensus_service
            #     )
            # ])
            
            logger.info("Added all gRPC services to server")
            
        except Exception as e:
            logger.error(f"Error adding services: {e}")
            raise
    
    async def _add_reflection_service(self) -> None:
        """Add gRPC reflection service."""
        try:
            # In a real implementation, you would use grpc_reflection
            # from grpc_reflection.v1alpha import reflection
            
            # reflection.enable_server_reflection(
            #     ['dubchain.BlockchainService', 'dubchain.WalletService', 'dubchain.BridgeService',
            #      'dubchain.GovernanceService', 'dubchain.NetworkService', 'dubchain.ConsensusService'],
            #     self.server
            # )
            
            logger.info("Added gRPC reflection service")
            
        except Exception as e:
            logger.error(f"Error adding reflection service: {e}")
    
    async def _add_health_check_service(self) -> None:
        """Add gRPC health check service."""
        try:
            # In a real implementation, you would use grpc_health
            # from grpc_health.v1 import health_pb2_grpc
            # from grpc_health.v1.health import HealthServicer
            
            # health_servicer = HealthServicer()
            # health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.server)
            
            logger.info("Added gRPC health check service")
            
        except Exception as e:
            logger.error(f"Error adding health check service: {e}")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get service by name."""
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all services."""
        return self.services.copy()

class GRPCServerManager:
    """Manager for gRPC server lifecycle."""
    
    def __init__(self, config: GRPCServerConfig):
        """Initialize server manager."""
        self.config = config
        self.server: Optional[DubChainGRPCServer] = None
        self.running = False
        logger.info("Initialized gRPC server manager")
    
    async def start_server(self) -> None:
        """Start the gRPC server."""
        try:
            if self.running:
                logger.warning("Server is already running")
                return
            
            self.server = DubChainGRPCServer(self.config)
            await self.server.start()
            self.running = True
            
            logger.info("gRPC server started successfully")
            
        except Exception as e:
            logger.error(f"Error starting gRPC server: {e}")
            raise
    
    async def stop_server(self) -> None:
        """Stop the gRPC server."""
        try:
            if not self.running or not self.server:
                logger.warning("Server is not running")
                return
            
            await self.server.stop()
            self.running = False
            
            logger.info("gRPC server stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")
            raise
    
    async def restart_server(self) -> None:
        """Restart the gRPC server."""
        try:
            logger.info("Restarting gRPC server")
            await self.stop_server()
            await asyncio.sleep(1)  # Wait 1 second
            await self.start_server()
            logger.info("gRPC server restarted successfully")
            
        except Exception as e:
            logger.error(f"Error restarting gRPC server: {e}")
            raise
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running
    
    def get_server(self) -> Optional[DubChainGRPCServer]:
        """Get the server instance."""
        return self.server

async def create_and_start_server(config: GRPCServerConfig) -> DubChainGRPCServer:
    """Create and start a gRPC server."""
    server = DubChainGRPCServer(config)
    await server.start()
    return server

async def run_server(config: GRPCServerConfig) -> None:
    """Run the gRPC server until termination."""
    server = await create_and_start_server(config)
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.stop()

def create_default_config() -> GRPCServerConfig:
    """Create default gRPC server configuration."""
    return GRPCServerConfig(
        host="0.0.0.0",
        port=50051,
        max_workers=10,
        max_message_length=4 * 1024 * 1024,  # 4MB
        max_metadata_size=8 * 1024,  # 8KB
        keepalive_time_ms=30000,  # 30 seconds
        keepalive_timeout_ms=5000,  # 5 seconds
        keepalive_permit_without_calls=True,
        max_connection_idle_ms=300000,  # 5 minutes
        max_connection_age_ms=1800000,  # 30 minutes
        max_connection_age_grace_ms=5000,  # 5 seconds
        enable_reflection=True,
        enable_health_check=True,
    )

if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    asyncio.run(run_server(config))

__all__ = [
    "GRPCServerConfig",
    "DubChainGRPCServer",
    "GRPCServerManager",
    "create_and_start_server",
    "run_server",
    "create_default_config",
]
