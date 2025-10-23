"""
Advanced consensus mechanisms for DubChain.

This module provides sophisticated consensus algorithms including:
- Proof of Stake (PoS)
- Delegated Proof of Stake (DPoS) 
- Practical Byzantine Fault Tolerance (PBFT)
- Hybrid consensus mechanisms
"""

import logging

logger = logging.getLogger(__name__)
from .consensus_engine import ConsensusConfig, ConsensusEngine, ConsensusState
from .consensus_types import (
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    ValidatorRole,
    ValidatorStatus,
)

# CUDA-accelerated consensus imports
from .cuda_consensus import (
    CUDAConsensusAccelerator,
    CUDAConsensusConfig,
    get_global_cuda_consensus_accelerator,
)
from .delegated_proof_of_stake import (
    DelegatedProofOfStake,
    DelegateInfo,
    ElectionManager,
    VotingPower,
)
from .hybrid_consensus import ConsensusSelector, ConsensusSwitcher, HybridConsensus
from .pbft import (
    PBFTMessage,
    PBFTPhase,
    PBFTValidator,
    PracticalByzantineFaultTolerance,
)
from .proof_of_stake import ProofOfStake, RewardCalculator, StakingInfo, StakingPool
from .validator import Validator, ValidatorInfo, ValidatorManager, ValidatorSet

__all__ = [
    # Types
    "ConsensusType",
    "ValidatorStatus",
    "ValidatorRole",
    "ConsensusResult",
    "ConsensusMetrics",
    # Validator Management
    "Validator",
    "ValidatorInfo",
    "ValidatorSet",
    "ValidatorManager",
    # Proof of Stake
    "ProofOfStake",
    "StakingInfo",
    "StakingPool",
    "RewardCalculator",
    # Delegated Proof of Stake
    "DelegatedProofOfStake",
    "DelegateInfo",
    "VotingPower",
    "ElectionManager",
    # PBFT
    "PracticalByzantineFaultTolerance",
    "PBFTMessage",
    "PBFTPhase",
    "PBFTValidator",
    # Hybrid Consensus
    "HybridConsensus",
    "ConsensusSelector",
    "ConsensusSwitcher",
    # Main Engine
    "ConsensusEngine",
    "ConsensusConfig",
    "ConsensusState",
    # CUDA Consensus
    "CUDAConsensusAccelerator",
    "CUDAConsensusConfig",
    "get_global_cuda_consensus_accelerator",
]
