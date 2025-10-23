"""
Avalanche Bridge Module

This module provides comprehensive Avalanche bridging capabilities including:
- Avalanche RPC client integration
- C-Chain (EVM) support
- X-Chain (UTXO) support
- P-Chain (Platform) support
- Transaction batching and optimization
- Bridge security and validation
"""

import logging

logger = logging.getLogger(__name__)
from .client import (
    AvalancheClient,
    AvalancheConfig,
    AvalancheTransaction,
    AvalancheUTXO,
    AvalancheBlock,
    AVAXToken,
)

from .bridge import (
    AvalancheBridge,
    BridgeConfig,
    BridgeTransaction,
    CChainConfig,
    XChainConfig,
    UTXOManager,
    EVMManager,
)

__all__ = [
    # Client
    "AvalancheClient",
    "AvalancheConfig",
    "AvalancheTransaction",
    "AvalancheUTXO",
    "AvalancheBlock",
    "AVAXToken",
    # Bridge
    "AvalancheBridge",
    "BridgeConfig",
    "BridgeTransaction",
    "CChainConfig",
    "XChainConfig",
    "UTXOManager",
    "EVMManager",
]
