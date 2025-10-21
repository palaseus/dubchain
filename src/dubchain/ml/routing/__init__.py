"""
ML Routing Optimization Module

This module provides reinforcement learning-based routing optimization including:
- PPO (Proximal Policy Optimization) for routing decisions
- Q-Learning for peer selection
- Multi-agent reinforcement learning for distributed routing
- Reward function design for network optimization
- Experience replay and target networks
- Policy gradient methods for continuous action spaces
"""

from .optimizer import (
    RoutingOptimizer,
    RoutingConfig,
    QLearningAgent,
    PPOAgent,
    RoutingEnvironment,
    QNetwork,
    PolicyNetwork,
    ValueNetwork,
    ExperienceReplayBuffer,
    RoutingState,
    RoutingAction,
    RoutingReward,
    Experience,
)

__all__ = [
    "RoutingOptimizer",
    "RoutingConfig",
    "QLearningAgent",
    "PPOAgent",
    "RoutingEnvironment",
    "QNetwork",
    "PolicyNetwork",
    "ValueNetwork",
    "ExperienceReplayBuffer",
    "RoutingState",
    "RoutingAction",
    "RoutingReward",
    "Experience",
]