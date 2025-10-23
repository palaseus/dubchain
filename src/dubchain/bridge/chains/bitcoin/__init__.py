"""
Bitcoin Bridge Module

This module provides comprehensive Bitcoin bridging capabilities including:
- SegWit transaction support
- Multi-signature wallet support
- UTXO management and tracking
- Transaction batching and optimization
- Bridge security and validation
"""

import logging

logger = logging.getLogger(__name__)
from .client import (
    BitcoinClient,
    BitcoinRPCClient,
    BitcoinConfig,
    BitcoinTransaction,
    BitcoinBlock,
    UTXO,
)

from .bridge import (
    BitcoinBridge,
    BridgeConfig,
    BridgeTransaction,
    SegWitConfig,
    MultiSigConfig,
    SegWitManager,
    MultiSigManager,
)

from .lightning import (
    LightningBridge,
    LightningManager,
    LightningConfig,
    LightningChannel,
    LightningInvoice,
    LightningPayment,
    LightningNode,
)

from .production_bridge import (
    ProductionBitcoinBridge,
    BitcoinNetwork,
    TransactionType,
    HTLCStatus,
    HTLCContract,
    MultiSigWallet,
    SPVProof,
    BitcoinRPCClient as ProductionBitcoinRPCClient,
    HTLCManager,
    MultiSigManager as ProductionMultiSigManager,
    SPVVerifier,
)

__all__ = [
    # Client
    "BitcoinClient",
    "BitcoinRPCClient",
    "BitcoinConfig",
    "BitcoinTransaction",
    "BitcoinBlock",
    "UTXO",
    # Bridge
    "BitcoinBridge",
    "BridgeConfig", 
    "BridgeTransaction",
    "SegWitConfig",
    "MultiSigConfig",
    "SegWitManager",
    "MultiSigManager",
    # Lightning
    "LightningBridge",
    "LightningManager",
    "LightningConfig",
    "LightningChannel",
    "LightningInvoice",
    "LightningPayment",
    "LightningNode",
    # Production Bridge
    "ProductionBitcoinBridge",
    "BitcoinNetwork",
    "TransactionType",
    "HTLCStatus",
    "HTLCContract",
    "MultiSigWallet",
    "SPVProof",
    "ProductionBitcoinRPCClient",
    "HTLCManager",
    "ProductionMultiSigManager",
    "SPVVerifier",
]