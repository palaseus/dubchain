"""
State Channels - Layer-2 Scaling Solution

This module provides a comprehensive state channel implementation for off-chain
transactions with on-chain dispute resolution. Features include:

- Multi-party state channels with flexible update logic
- Off-chain state management with cryptographic security
- On-chain smart contracts for dispute resolution
- Adversarial condition handling and Byzantine fault tolerance
- Comprehensive testing suite with property-based and adversarial tests
"""

from .channel_manager import ChannelManager
from .channel_protocol import (
    ChannelCloseReason,
    ChannelConfig,
    ChannelError,
    ChannelEvent,
    ChannelId,
    ChannelState,
    ChannelStatus,
    StateChannel,
    StateUpdate,
    StateUpdateType,
)
from .dispute_resolution import (
    DisputeEvidence,
    DisputeManager,
    DisputeResolution,
    DisputeStatus,
    OnChainContract,
)
from .off_chain_state import (
    OffChainStateManager,
    StateSignature,
    StateTransition,
    StateValidator,
)
from .security import (
    ChannelSecurity,
    CryptographicProof,
    FraudProof,
    SecurityManager,
    TimeoutManager,
)

__all__ = [
    # Core Protocol
    "StateChannel",
    "ChannelManager",
    "ChannelConfig",
    "ChannelState",
    "ChannelStatus",
    "ChannelId",
    "ChannelEvent",
    "ChannelError",
    "ChannelCloseReason",
    "StateUpdate",
    "StateUpdateType",
    # Off-chain State Management
    "OffChainStateManager",
    "StateTransition",
    "StateSignature",
    "StateValidator",
    # Dispute Resolution
    "DisputeManager",
    "DisputeResolution",
    "DisputeEvidence",
    "DisputeStatus",
    "OnChainContract",
    # Security
    "SecurityManager",
    "ChannelSecurity",
    "TimeoutManager",
    "CryptographicProof",
    "FraudProof",
]

__version__ = "1.0.0"
