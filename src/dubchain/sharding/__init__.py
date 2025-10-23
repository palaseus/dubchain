"""
Blockchain Sharding System for DubChain.

This module provides sophisticated sharding capabilities including:
- Network sharding for horizontal scaling
- Cross-shard transactions and communication
- Dynamic shard rebalancing
- Shard consensus coordination
- Shard state management
"""

import logging

logger = logging.getLogger(__name__)
from .cross_shard_communication import (
    CrossShardMessage,
    CrossShardMessaging,
    CrossShardValidator,
    MessageRelay,
    MessageType,
    ShardRouter,
)
from .shard_consensus import (
    ShardCommittee,
    ShardConsensus,
    ShardProposer,
    ShardValidator,
)
from .shard_manager import ShardAllocator, ShardBalancer, ShardCoordinator, ShardManager
from .shard_network import ShardDiscovery, ShardNetwork, ShardRouting, ShardTopology
from .shard_state_manager import (
    ShardStateManager,
    StateSnapshot,
    StateSync,
    StateValidator,
)
from .shard_types import (
    CrossShardTransaction,
    ShardConfig,
    ShardId,
    ShardMetrics,
    ShardState,
    ShardStatus,
    ShardType,
)

# Enhanced sharding components
from .enhanced_sharding import (
    LoadBalancingStrategy,
    ReshardingStrategy,
    ShardHealthStatus,
    ShardLoadMetrics,
    ShardHealthInfo,
    ReshardingPlan,
    ShardOperation,
    ShardLoadBalancer,
    ConsistentHashBalancer,
    LeastLoadedBalancer,
    AdaptiveBalancer,
    ShardReshardingManager,
    ShardHealthMonitor,
    EnhancedShardManager,
)

__all__ = [
    # Types
    "ShardId",
    "ShardStatus",
    "ShardType",
    "ShardConfig",
    "ShardMetrics",
    "CrossShardTransaction",
    "ShardState",
    # Management
    "ShardManager",
    "ShardAllocator",
    "ShardBalancer",
    "ShardCoordinator",
    # Consensus
    "ShardConsensus",
    "ShardValidator",
    "ShardProposer",
    "ShardCommittee",
    # Cross-shard Communication
    "CrossShardMessaging",
    "ShardRouter",
    "MessageRelay",
    "CrossShardValidator",
    "MessageType",
    "CrossShardMessage",
    # State Management
    "ShardStateManager",
    "StateSync",
    "StateValidator",
    "StateSnapshot",
    # Network
    "ShardNetwork",
    "ShardTopology",
    "ShardDiscovery",
    "ShardRouting",
    # Enhanced Sharding
    "LoadBalancingStrategy",
    "ReshardingStrategy",
    "ShardHealthStatus",
    "ShardLoadMetrics",
    "ShardHealthInfo",
    "ReshardingPlan",
    "ShardOperation",
    "ShardLoadBalancer",
    "ConsistentHashBalancer",
    "LeastLoadedBalancer",
    "AdaptiveBalancer",
    "ShardReshardingManager",
    "ShardHealthMonitor",
    "EnhancedShardManager",
]
