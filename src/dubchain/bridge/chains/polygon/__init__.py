"""
Polygon Bridge Module

This module provides comprehensive Polygon network integration including:
- Polygon RPC client with Web3 integration
- PoS bridge functionality
- Fast finality optimizations
- Gas optimization for Polygon network
- Mumbai testnet support
"""

from .client import (
    PolygonClient,
    PolygonConfig,
    PolygonTransaction,
    PolygonBlock,
    PoSBridgeConfig,
    PoSBridge,
)

from .zkevm import (
    ZkEVMClient,
    ZkEVMConfig,
    ZkEVMTransaction,
    ZkEVMBatch,
    ZkProof,
    ZkEVMBridge,
)

from .production_bridge import (
    ProductionPolygonBridge,
    PolygonNetwork,
    CheckpointStatus,
    ExitStatus,
    zkEVMStatus,
    Checkpoint,
    ExitRequest,
    zkEVMProof,
    PlasmaBlock,
    PolygonRPCClient,
    CheckpointManager,
    ExitManager,
    zkEVMManager,
    PlasmaManager,
)

__all__ = [
    "PolygonClient",
    "PolygonConfig",
    "PolygonTransaction", 
    "PolygonBlock",
    "PoSBridgeConfig",
    "PoSBridge",
    "ZkEVMClient",
    "ZkEVMConfig",
    "ZkEVMTransaction",
    "ZkEVMBatch",
    "ZkProof",
    "ZkEVMBridge",
    # Production Bridge
    "ProductionPolygonBridge",
    "PolygonNetwork",
    "CheckpointStatus",
    "ExitStatus",
    "zkEVMStatus",
    "Checkpoint",
    "ExitRequest",
    "zkEVMProof",
    "PlasmaBlock",
    "PolygonRPCClient",
    "CheckpointManager",
    "ExitManager",
    "zkEVMManager",
    "PlasmaManager",
]