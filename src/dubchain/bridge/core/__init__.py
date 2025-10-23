"""
Bridge Core Module

This module provides the core bridge functionality including:
- Validator network management
- Fraud detection system
- Relayer system for transaction processing
- Analytics and monitoring
- Production bridge core
"""

import logging

logger = logging.getLogger(__name__)
from .production_core import (
    ProductionBridgeCore,
    BridgeCoreConfig,
    ValidatorStatus,
    FraudLevel,
    RelayerStatus,
    TransactionStatus,
    Validator,
    FraudDetectionResult,
    Relayer,
    BridgeTransaction,
    BridgeMetrics,
    ValidatorNetwork,
    FraudDetector,
    RelayerSystem,
    BridgeAnalytics,
)

__all__ = [
    "ProductionBridgeCore",
    "BridgeCoreConfig",
    "ValidatorStatus",
    "FraudLevel",
    "RelayerStatus",
    "TransactionStatus",
    "Validator",
    "FraudDetectionResult",
    "Relayer",
    "BridgeTransaction",
    "BridgeMetrics",
    "ValidatorNetwork",
    "FraudDetector",
    "RelayerSystem",
    "BridgeAnalytics",
]
