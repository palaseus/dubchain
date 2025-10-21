"""
Solana Bridge Module

This module provides comprehensive Solana bridging capabilities including:
- Solana RPC client integration
- SPL token support (SPL-20, SPL-721)
- Program-derived address (PDA) management
- Transaction batching and optimization
- Bridge security and validation
"""

from .client import (
    SolanaClient,
    SolanaConfig,
    SolanaTransaction,
    SolanaAccount,
    SolanaBlock,
    SPLToken,
)

from .bridge import (
    SolanaBridge,
    BridgeConfig,
    BridgeTransaction,
    SPLTokenConfig,
    PDAManager,
    TokenManager,
)

__all__ = [
    # Client
    "SolanaClient",
    "SolanaConfig",
    "SolanaTransaction",
    "SolanaAccount",
    "SolanaBlock",
    "SPLToken",
    # Bridge
    "SolanaBridge",
    "BridgeConfig",
    "BridgeTransaction",
    "SPLTokenConfig",
    "PDAManager",
    "TokenManager",
]
