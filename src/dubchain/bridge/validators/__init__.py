"""
Bridge Validator Network Module

This module provides a Byzantine Fault Tolerant validator network for bridge operations including:
- Validator registration and management
- BFT consensus for bridge decisions
- Slashing mechanisms for malicious validators
- Emergency pause functionality
- Governance integration
"""

from .network import (
    BridgeValidatorNetwork,
    ValidatorManager,
    BFTConsensus,
    EmergencyManager,
    ValidatorConfig,
    Validator,
    ValidatorStatus,
    ConsensusPhase,
    ConsensusMessage,
    BridgeProposal,
)

__all__ = [
    "BridgeValidatorNetwork",
    "ValidatorManager",
    "BFTConsensus",
    "EmergencyManager",
    "ValidatorConfig",
    "Validator",
    "ValidatorStatus",
    "ConsensusPhase",
    "ConsensusMessage",
    "BridgeProposal",
]