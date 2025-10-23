"""
Polkadot Bridge Module

This module provides comprehensive Polkadot bridging capabilities including:
- Polkadot RPC client integration
- Substrate framework support
- Parachain integration
- Cross-chain message passing (XCMP)
- Transaction batching and optimization
- Bridge security and validation
"""

import logging

logger = logging.getLogger(__name__)
from .client import (
    PolkadotClient,
    PolkadotConfig,
    PolkadotTransaction,
    PolkadotBlock,
    DOTToken,
    ParachainInfo,
)

from .bridge import (
    PolkadotBridge,
    BridgeConfig,
    BridgeTransaction,
    SubstrateConfig,
    ParachainConfig,
    XCMPManager,
    RelayChainManager,
)

__all__ = [
    # Client
    "PolkadotClient",
    "PolkadotConfig",
    "PolkadotTransaction",
    "PolkadotBlock",
    "DOTToken",
    "ParachainInfo",
    # Bridge
    "PolkadotBridge",
    "BridgeConfig",
    "BridgeTransaction",
    "SubstrateConfig",
    "ParachainConfig",
    "XCMPManager",
    "RelayChainManager",
]
