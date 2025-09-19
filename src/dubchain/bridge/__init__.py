"""
Cross-Chain Bridge System for DubChain.

This module provides sophisticated cross-chain interoperability including:
- Multi-chain asset transfers
- Atomic swaps between chains
- Cross-chain message passing
- Universal asset management
- Bridge security and validation
"""

from .atomic_swap import AtomicSwap, SwapExecution, SwapProposal, SwapValidator
from .bridge_manager import AssetManager, BridgeManager, BridgeValidator, ChainManager
from .bridge_security import (
    BridgeMonitoring,
    BridgeSecurity,
    FraudDetection,
    SecurityValidator,
)
from .bridge_types import (
    AssetType,
    BridgeAsset,
    BridgeConfig,
    BridgeMetrics,
    BridgeStatus,
    BridgeType,
    BridgeValidator,
    ChainType,
    CrossChainTransaction,
)
from .cross_chain_messaging import (
    ChainRouter,
    CrossChainMessaging,
    MessageRelay,
    MessageValidator,
)
from .universal_assets import (
    AssetConverter,
    AssetRegistry,
    AssetValidator,
    UniversalAsset,
)

__all__ = [
    # Types
    "BridgeType",
    "BridgeStatus",
    "ChainType",
    "AssetType",
    "BridgeConfig",
    "BridgeMetrics",
    "CrossChainTransaction",
    "BridgeAsset",
    "BridgeValidator",
    # Management
    "BridgeManager",
    "ChainManager",
    "AssetManager",
    "BridgeValidator",
    # Atomic Swaps
    "AtomicSwap",
    "SwapProposal",
    "SwapExecution",
    "SwapValidator",
    # Cross-Chain Messaging
    "CrossChainMessaging",
    "MessageRelay",
    "ChainRouter",
    "MessageValidator",
    # Security
    "BridgeSecurity",
    "SecurityValidator",
    "FraudDetection",
    "BridgeMonitoring",
    # Universal Assets
    "UniversalAsset",
    "AssetRegistry",
    "AssetConverter",
    "AssetValidator",
]
