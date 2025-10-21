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

# Import new comprehensive modules
from .channel_protocol import (
    ChannelManager as NewChannelManager,
    StateValidator as NewStateValidator,
    PaymentProcessor,
    ChannelState as NewChannelState,
    ChannelParticipant,
    Payment,
    ChannelConfig as NewChannelConfig,
    ChannelStatus as NewChannelStatus,
    PaymentType,
)
from .dispute_resolution import (
    DisputeManager as NewDisputeManager,
    DisputeResolver,
    EvidenceValidator,
    FraudDetector,
    DisputeResolution as NewDisputeResolution,
    DisputeEvidence as NewDisputeEvidence,
    FraudProof as NewFraudProof,
    DisputeConfig,
    DisputeStatus as NewDisputeStatus,
    EvidenceType,
)
from .off_chain_state import (
    OffChainStateManager as NewOffChainStateManager,
    StateCompressor,
    StateVersioner,
    StateValidator as OffChainStateValidator,
    StateSnapshot,
    StateDiff,
    StateConfig,
    StateVersion,
    CompressionType,
)
from .security import (
    SecurityManager as NewSecurityManager,
    SignatureValidator,
    AccessController,
    ThreatDetector,
    SecurityMonitor,
    SecurityEvent,
    AccessControl,
    SecurityConfig,
    SecurityLevel,
    ThreatType,
    AccessLevel,
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
    # New comprehensive modules
    "NewChannelManager",
    "NewStateValidator",
    "PaymentProcessor",
    "NewChannelState",
    "ChannelParticipant",
    "Payment",
    "NewChannelConfig",
    "NewChannelStatus",
    "PaymentType",
    "NewDisputeManager",
    "DisputeResolver",
    "EvidenceValidator",
    "FraudDetector",
    "NewDisputeResolution",
    "NewDisputeEvidence",
    "NewFraudProof",
    "DisputeConfig",
    "NewDisputeStatus",
    "EvidenceType",
    "NewOffChainStateManager",
    "StateCompressor",
    "StateVersioner",
    "OffChainStateValidator",
    "StateSnapshot",
    "StateDiff",
    "StateConfig",
    "StateVersion",
    "CompressionType",
    "NewSecurityManager",
    "SignatureValidator",
    "AccessController",
    "ThreatDetector",
    "SecurityMonitor",
    "SecurityEvent",
    "AccessControl",
    "SecurityConfig",
    "SecurityLevel",
    "ThreatType",
    "AccessLevel",
]

__version__ = "1.0.0"
