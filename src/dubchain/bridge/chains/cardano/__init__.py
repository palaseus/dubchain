"""
Cardano Bridge Module

This module provides comprehensive Cardano bridging capabilities including:
- Cardano CLI integration
- Native token support
- UTXO management
- Transaction batching and optimization
- Bridge security and validation
"""

from .client import (
    CardanoClient,
    CardanoConfig,
    CardanoTransaction,
    CardanoUTXO,
    CardanoBlock,
    NativeToken,
)

from .bridge import (
    CardanoBridge,
    BridgeConfig,
    BridgeTransaction,
    NativeTokenConfig,
    UTXOManager,
    TokenManager,
)

__all__ = [
    # Client
    "CardanoClient",
    "CardanoConfig",
    "CardanoTransaction",
    "CardanoUTXO",
    "CardanoBlock",
    "NativeToken",
    # Bridge
    "CardanoBridge",
    "BridgeConfig",
    "BridgeTransaction",
    "NativeTokenConfig",
    "UTXOManager",
    "TokenManager",
]
