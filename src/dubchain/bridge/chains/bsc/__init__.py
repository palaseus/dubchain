"""
Binance Smart Chain Bridge Module

This module provides comprehensive BSC integration including:
- BSC RPC client with Web3 integration
- Fast block confirmation
- BEP token support
- Gas optimization for BSC
- Testnet support
"""

from .client import (
    BSCClient,
    BSCConfig,
    BSCTransaction,
    BSCBlock,
    BSCGasPriceInfo,
    BSCGasPriceOracle,
)

from .bridge import (
    BSCBridge,
    BridgeConfig,
    BridgeTransaction,
    BEP20Token,
    BEP721Token,
    BEP20Manager,
    BEP721Manager,
)

__all__ = [
    "BSCClient",
    "BSCConfig",
    "BSCTransaction",
    "BSCBlock", 
    "BSCGasPriceInfo",
    "BSCGasPriceOracle",
    "BSCBridge",
    "BridgeConfig",
    "BridgeTransaction",
    "BEP20Token",
    "BEP721Token",
    "BEP20Manager",
    "BEP721Manager",
]
