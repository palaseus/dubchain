"""
On-Chain Governance System for DubChain.

This module provides a comprehensive, production-grade governance system with:
- Proposal lifecycle management
- Multiple voting strategies
- Vote delegation and delegation chains
- Quorum and threshold management
- Proposal execution engine
- Timelock mechanisms
- Emergency pause/fast-track capabilities
- Treasury management integration
- Security defenses against various attacks
- Comprehensive audit trails and observability
"""

from .core import (
    GovernanceEngine,
    GovernanceConfig,
    GovernanceState,
    Proposal,
    ProposalStatus,
    ProposalType,
    Vote,
    VoteChoice,
    VotingPower,
)
from .delegation import (
    DelegationManager,
    DelegationChain,
    DelegationStrategy,
)
from .execution import (
    ExecutionEngine,
    ExecutionResult,
    TimelockManager,
    EmergencyManager,
)
from .strategies import (
    VotingStrategy,
    TokenWeightedStrategy,
    QuadraticVotingStrategy,
    ConvictionVotingStrategy,
    SnapshotVotingStrategy,
    StrategyFactory,
)
from .treasury import (
    TreasuryManager,
    TreasuryProposal,
)
from .security import (
    SecurityManager,
    AttackDetector,
    SybilDetector,
    VoteBuyingDetector,
    FlashLoanDetector,
)
from .observability import (
    GovernanceEvents,
    AuditTrail,
    MerkleProofManager,
    GovernanceMetrics,
)
from .upgrades import (
    UpgradeManager,
    ProxyGovernance,
    UpgradeProposal,
    EmergencyEscapeHatch,
)

__all__ = [
    # Core
    "GovernanceEngine",
    "GovernanceConfig", 
    "GovernanceState",
    "Proposal",
    "ProposalStatus",
    "ProposalType",
    "Vote",
    "VoteChoice",
    "VotingPower",
    
    # Delegation
    "DelegationManager",
    "DelegationChain",
    "DelegationStrategy",
    
    # Execution
    "ExecutionEngine",
    "ExecutionResult",
    "TimelockManager",
    "EmergencyManager",
    
    # Voting Strategies
    "VotingStrategy",
    "TokenWeightedStrategy",
    "QuadraticVotingStrategy", 
    "ConvictionVotingStrategy",
    "SnapshotVotingStrategy",
    "StrategyFactory",
    
    # Treasury
    "TreasuryManager",
    "TreasuryProposal",
    "SpendingProposal",
    
    # Security
    "SecurityManager",
    "AttackDetector",
    "SybilDetector",
    "VoteBuyingDetector",
    "FlashLoanDetector",
    
    # Observability
    "GovernanceEvents",
    "AuditTrail",
    "MerkleProofManager",
    "GovernanceMetrics",
    
    # Upgrades
    "UpgradeManager",
    "ProxyGovernance",
    "UpgradeProposal",
    "EmergencyEscapeHatch",
]
