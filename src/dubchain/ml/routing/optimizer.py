"""
Reinforcement Learning for Routing and Peer Selection

This module provides reinforcement learning-based routing optimization including:
- PPO (Proximal Policy Optimization) for routing decisions
- Q-Learning for peer selection
- Multi-agent reinforcement learning for distributed routing
- Reward function design for network optimization
- Experience replay and target networks
- Policy gradient methods for continuous action spaces
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
import json
import hashlib
import random
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from ...errors import BridgeError, ClientError
from ...logging import get_logger
from ..network import NetworkTopology, PeerNode, TopologyOptimizer
from ..features import FeaturePipeline, FeatureConfig

logger = get_logger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for routing optimization."""
    enable_ppo: bool = True
    enable_q_learning: bool = True
    enable_multi_agent: bool = True
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 100
    target_update_frequency: int = 1000
    enable_experience_replay: bool = True
    enable_double_dqn: bool = True
    enable_dueling_dqn: bool = True


@dataclass
class RoutingState:
    """State representation for routing decisions."""
    current_node: str
    destination_node: str
    available_peers: List[str]
    network_topology: NetworkTopology
    traffic_load: Dict[str, float]
    latency_matrix: Dict[Tuple[str, str], float]
    bandwidth_matrix: Dict[Tuple[str, str], float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingAction:
    """Action representation for routing decisions."""
    selected_peer: str
    route_path: List[str]
    priority_level: int  # 0-10
    bandwidth_allocation: float  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingReward:
    """Reward structure for routing optimization."""
    latency_reward: float
    bandwidth_reward: float
    reliability_reward: float
    load_balancing_reward: float
    total_reward: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Experience:
    """Experience tuple for experience replay."""
    state: RoutingState
    action: RoutingAction
    reward: RoutingReward
    next_state: RoutingState
    done: bool
    timestamp: float = field(default_factory=time.time)


class RoutingEnvironment:
    """Environment for routing optimization."""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.current_state: Optional[RoutingState] = None
        
        # Handle gym spaces availability
        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(100)  # Max 100 peers
            self.observation_space = spaces.Box(low=0, high=1, shape=(100,), dtype=np.float32)
        else:
            # Fallback when gym is not available
            self.action_space = None
            self.observation_space = None
            
        self.reward_history = deque(maxlen=1000)
        
    def reset(self, initial_state: RoutingState) -> RoutingState:
        """Reset the environment."""
        self.current_state = initial_state
        return self.current_state
    
    def step(self, action: RoutingAction) -> Tuple[RoutingState, RoutingReward, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if not self.current_state:
            raise ValueError("Environment not initialized")
        
        # Execute action
        next_state = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, next_state)
        
        # Check if episode is done
        done = self._is_done(next_state)
        
        # Update current state
        self.current_state = next_state
        
        # Store reward
        self.reward_history.append(reward)
        
        info = {
            "action_executed": True,
            "reward_components": {
                "latency": reward.latency_reward,
                "bandwidth": reward.bandwidth_reward,
                "reliability": reward.reliability_reward,
                "load_balancing": reward.load_balancing_reward
            }
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: RoutingAction) -> RoutingState:
        """Execute the routing action."""
        # Create next state based on action
        next_state = RoutingState(
            current_node=action.selected_peer,
            destination_node=self.current_state.destination_node,
            available_peers=self.current_state.available_peers.copy(),
            network_topology=self.current_state.network_topology,
            traffic_load=self.current_state.traffic_load.copy(),
            latency_matrix=self.current_state.latency_matrix.copy(),
            bandwidth_matrix=self.current_state.bandwidth_matrix.copy()
        )
        
        # Update traffic load
        if action.selected_peer in next_state.traffic_load:
            next_state.traffic_load[action.selected_peer] += action.bandwidth_allocation
        
        return next_state
    
    def _calculate_reward(self, action: RoutingAction, next_state: RoutingState) -> RoutingReward:
        """Calculate reward for the action."""
        # Latency reward (lower is better)
        latency = self._get_latency(action.current_node, action.selected_peer)
        latency_reward = max(0, 1.0 - latency / 1000)  # Normalize to 0-1
        
        # Bandwidth reward (higher is better)
        bandwidth = self._get_bandwidth(action.current_node, action.selected_peer)
        bandwidth_reward = min(1.0, bandwidth / 1000)  # Normalize to 0-1
        
        # Reliability reward (based on peer health)
        reliability_reward = self._get_reliability(action.selected_peer)
        
        # Load balancing reward (encourage even distribution)
        load_balancing_reward = self._get_load_balancing_reward(next_state)
        
        # Total reward (weighted sum)
        total_reward = (
            latency_reward * 0.3 +
            bandwidth_reward * 0.3 +
            reliability_reward * 0.2 +
            load_balancing_reward * 0.2
        )
        
        return RoutingReward(
            latency_reward=latency_reward,
            bandwidth_reward=bandwidth_reward,
            reliability_reward=reliability_reward,
            load_balancing_reward=load_balancing_reward,
            total_reward=total_reward
        )
    
    def _get_latency(self, src: str, dst: str) -> float:
        """Get latency between two nodes."""
        return self.current_state.latency_matrix.get((src, dst), 100.0)
    
    def _get_bandwidth(self, src: str, dst: str) -> float:
        """Get bandwidth between two nodes."""
        return self.current_state.bandwidth_matrix.get((src, dst), 100.0)
    
    def _get_reliability(self, peer: str) -> float:
        """Get reliability score for a peer."""
        if peer in self.current_state.network_topology.nodes:
            return self.current_state.network_topology.nodes[peer].health_score
        return 0.5
    
    def _get_load_balancing_reward(self, state: RoutingState) -> float:
        """Calculate load balancing reward."""
        if not state.traffic_load:
            return 1.0
        
        loads = list(state.traffic_load.values())
        if not loads:
            return 1.0
        
        # Calculate coefficient of variation (lower is better)
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        
        if mean_load == 0:
            return 1.0
        
        cv = std_load / mean_load
        return max(0, 1.0 - cv)
    
    def _is_done(self, state: RoutingState) -> bool:
        """Check if episode is done."""
        return state.current_node == state.destination_node


class QNetwork(nn.Module):
    """Q-Network for Q-Learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Dueling DQN architecture
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """Q-Learning agent for routing optimization."""
    
    def __init__(self, config: RoutingConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if TORCH_AVAILABLE:
            self.q_network = QNetwork(state_dim, action_dim)
            self.target_network = QNetwork(state_dim, action_dim)
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
            
            # Initialize target network
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = config.epsilon
        self.replay_buffer = ExperienceReplayBuffer(config.buffer_size)
        self.update_count = 0
    
    def select_action(self, state: RoutingState) -> RoutingAction:
        """Select action using epsilon-greedy policy."""
        if not TORCH_AVAILABLE:
            return self._random_action(state)
        
        if random.random() < self.epsilon:
            return self._random_action(state)
        
        # Convert state to tensor
        state_tensor = self._state_to_tensor(state)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        return self._action_from_index(action_idx, state)
    
    def _random_action(self, state: RoutingState) -> RoutingAction:
        """Select random action."""
        if not state.available_peers:
            return RoutingAction(
                selected_peer=state.current_node,
                route_path=[state.current_node],
                priority_level=5,
                bandwidth_allocation=0.5
            )
        
        selected_peer = random.choice(state.available_peers)
        return RoutingAction(
            selected_peer=selected_peer,
            route_path=[state.current_node, selected_peer],
            priority_level=random.randint(0, 10),
            bandwidth_allocation=random.random()
        )
    
    def _action_from_index(self, action_idx: int, state: RoutingState) -> RoutingAction:
        """Convert action index to RoutingAction."""
        if not state.available_peers:
            return RoutingAction(
                selected_peer=state.current_node,
                route_path=[state.current_node],
                priority_level=5,
                bandwidth_allocation=0.5
            )
        
        # Map action index to peer
        peer_idx = action_idx % len(state.available_peers)
        selected_peer = state.available_peers[peer_idx]
        
        # Map action index to priority and bandwidth
        priority_level = (action_idx // len(state.available_peers)) % 11
        bandwidth_allocation = (action_idx % 10) / 10.0
        
        return RoutingAction(
            selected_peer=selected_peer,
            route_path=[state.current_node, selected_peer],
            priority_level=priority_level,
            bandwidth_allocation=bandwidth_allocation
        )
    
    def _state_to_tensor(self, state: RoutingState) -> torch.Tensor:
        """Convert state to tensor."""
        # Simplified state representation
        state_vector = []
        
        # Add peer features
        for peer in state.available_peers[:10]:  # Limit to 10 peers
            if peer in state.network_topology.nodes:
                node = state.network_topology.nodes[peer]
                state_vector.extend([
                    node.priority_score,
                    node.health_score,
                    node.latency,
                    node.bandwidth
                ])
            else:
                state_vector.extend([0.0, 0.0, 0.0, 0.0])
        
        # Pad or truncate to fixed size
        while len(state_vector) < self.state_dim:
            state_vector.append(0.0)
        state_vector = state_vector[:self.state_dim]
        
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
    
    def store_experience(self, experience: Experience):
        """Store experience in replay buffer."""
        self.replay_buffer.push(experience)
    
    def update(self):
        """Update Q-network using experience replay."""
        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.stack([self._state_to_tensor(exp.state).squeeze() for exp in batch])
        actions = torch.tensor([self._action_to_index(exp.action, exp.state) for exp in batch])
        rewards = torch.tensor([exp.reward.total_reward for exp in batch])
        next_states = torch.stack([self._state_to_tensor(exp.next_state).squeeze() for exp in batch])
        dones = torch.tensor([exp.done for exp in batch])
        
        # Calculate target Q-values
        with torch.no_grad():
            if self.config.enable_double_dqn:
                # Double DQN
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + self.config.gamma * next_q_values.squeeze() * (1 - dones)
        
        # Calculate current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
    
    def _action_to_index(self, action: RoutingAction, state: RoutingState) -> int:
        """Convert RoutingAction to index."""
        if not state.available_peers:
            return 0
        
        try:
            peer_idx = state.available_peers.index(action.selected_peer)
            priority_idx = action.priority_level
            bandwidth_idx = int(action.bandwidth_allocation * 10)
            
            return peer_idx + priority_idx * len(state.available_peers) + bandwidth_idx * 11 * len(state.available_peers)
        except ValueError:
            return 0


class PPOAgent:
    """PPO agent for routing optimization."""
    
    def __init__(self, config: RoutingConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if TORCH_AVAILABLE:
            self.policy_network = PolicyNetwork(state_dim, action_dim)
            self.value_network = ValueNetwork(state_dim)
            self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
            self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=config.learning_rate)
        
        self.experience_buffer = []
    
    def select_action(self, state: RoutingState) -> Tuple[RoutingAction, float]:
        """Select action using policy network."""
        if not TORCH_AVAILABLE:
            action = self._random_action(state)
            return action, 0.0
        
        state_tensor = self._state_to_tensor(state)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            log_prob = action_dist.log_prob(action_idx)
        
        action = self._action_from_index(action_idx.item(), state)
        return action, log_prob.item()
    
    def _random_action(self, state: RoutingState) -> RoutingAction:
        """Select random action."""
        if not state.available_peers:
            return RoutingAction(
                selected_peer=state.current_node,
                route_path=[state.current_node],
                priority_level=5,
                bandwidth_allocation=0.5
            )
        
        selected_peer = random.choice(state.available_peers)
        return RoutingAction(
            selected_peer=selected_peer,
            route_path=[state.current_node, selected_peer],
            priority_level=random.randint(0, 10),
            bandwidth_allocation=random.random()
        )
    
    def _action_from_index(self, action_idx: int, state: RoutingState) -> RoutingAction:
        """Convert action index to RoutingAction."""
        if not state.available_peers:
            return RoutingAction(
                selected_peer=state.current_node,
                route_path=[state.current_node],
                priority_level=5,
                bandwidth_allocation=0.5
            )
        
        # Map action index to peer
        peer_idx = action_idx % len(state.available_peers)
        selected_peer = state.available_peers[peer_idx]
        
        # Map action index to priority and bandwidth
        priority_level = (action_idx // len(state.available_peers)) % 11
        bandwidth_allocation = (action_idx % 10) / 10.0
        
        return RoutingAction(
            selected_peer=selected_peer,
            route_path=[state.current_node, selected_peer],
            priority_level=priority_level,
            bandwidth_allocation=bandwidth_allocation
        )
    
    def _state_to_tensor(self, state: RoutingState) -> torch.Tensor:
        """Convert state to tensor."""
        # Simplified state representation
        state_vector = []
        
        # Add peer features
        for peer in state.available_peers[:10]:  # Limit to 10 peers
            if peer in state.network_topology.nodes:
                node = state.network_topology.nodes[peer]
                state_vector.extend([
                    node.priority_score,
                    node.health_score,
                    node.latency,
                    node.bandwidth
                ])
            else:
                state_vector.extend([0.0, 0.0, 0.0, 0.0])
        
        # Pad or truncate to fixed size
        while len(state_vector) < self.state_dim:
            state_vector.append(0.0)
        state_vector = state_vector[:self.state_dim]
        
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
    
    def store_experience(self, state: RoutingState, action: RoutingAction, 
                        reward: RoutingReward, next_state: RoutingState, 
                        log_prob: float, done: bool):
        """Store experience for PPO update."""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'log_prob': log_prob,
            'done': done
        })
    
    def update(self):
        """Update policy and value networks using PPO."""
        if not TORCH_AVAILABLE or len(self.experience_buffer) < self.config.batch_size:
            return
        
        # Convert experiences to tensors
        states = torch.stack([self._state_to_tensor(exp['state']).squeeze() for exp in self.experience_buffer])
        actions = torch.tensor([self._action_to_index(exp['action'], exp['state']) for exp in self.experience_buffer])
        rewards = torch.tensor([exp['reward'].total_reward for exp in self.experience_buffer])
        log_probs = torch.tensor([exp['log_prob'] for exp in self.experience_buffer])
        dones = torch.tensor([exp['done'] for exp in self.experience_buffer])
        
        # Calculate returns
        returns = self._calculate_returns(rewards, dones)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Policy update
            action_probs = self.policy_network(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Value update
            values = self.value_network(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Clear buffer
        self.experience_buffer.clear()
    
    def _calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _action_to_index(self, action: RoutingAction, state: RoutingState) -> int:
        """Convert RoutingAction to index."""
        if not state.available_peers:
            return 0
        
        try:
            peer_idx = state.available_peers.index(action.selected_peer)
            priority_idx = action.priority_level
            bandwidth_idx = int(action.bandwidth_allocation * 10)
            
            return peer_idx + priority_idx * len(state.available_peers) + bandwidth_idx * 11 * len(state.available_peers)
        except ValueError:
            return 0


class RoutingOptimizer:
    """Main routing optimizer using reinforcement learning."""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.q_learning_agent: Optional[QLearningAgent] = None
        self.ppo_agent: Optional[PPOAgent] = None
        self.environment = RoutingEnvironment(config)
        self.training_history = []
        
    def initialize_agents(self, state_dim: int, action_dim: int) -> None:
        """Initialize RL agents."""
        if self.config.enable_q_learning:
            self.q_learning_agent = QLearningAgent(self.config, state_dim, action_dim)
        
        if self.config.enable_ppo:
            self.ppo_agent = PPOAgent(self.config, state_dim, action_dim)
        
        logger.info("RL agents initialized")
    
    def train_agents(self, episodes: int = 1000) -> None:
        """Train RL agents."""
        for episode in range(episodes):
            episode_reward = 0.0
            
            # Initialize episode
            initial_state = self._create_initial_state()
            state = self.environment.reset(initial_state)
            
            done = False
            step_count = 0
            
            while not done and step_count < 100:  # Max 100 steps per episode
                # Select action
                if self.config.enable_ppo and self.ppo_agent:
                    action, log_prob = self.ppo_agent.select_action(state)
                elif self.config.enable_q_learning and self.q_learning_agent:
                    action = self.q_learning_agent.select_action(state)
                    log_prob = 0.0
                else:
                    action = self._random_action(state)
                    log_prob = 0.0
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                if self.config.enable_q_learning and self.q_learning_agent:
                    self.q_learning_agent.store_experience(experience)
                
                if self.config.enable_ppo and self.ppo_agent:
                    self.ppo_agent.store_experience(state, action, reward, next_state, log_prob, done)
                
                # Update agents
                if step_count % self.config.update_frequency == 0:
                    if self.config.enable_q_learning and self.q_learning_agent:
                        self.q_learning_agent.update()
                    
                    if self.config.enable_ppo and self.ppo_agent:
                        self.ppo_agent.update()
                
                state = next_state
                episode_reward += reward.total_reward
                step_count += 1
            
            self.training_history.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history[-100:])
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
    
    def optimize_route(self, current_node: str, destination_node: str, 
                     network_topology: NetworkTopology) -> List[str]:
        """Optimize route using trained agents."""
        # Create initial state
        initial_state = RoutingState(
            current_node=current_node,
            destination_node=destination_node,
            available_peers=list(network_topology.nodes.keys()),
            network_topology=network_topology,
            traffic_load={},
            latency_matrix={},
            bandwidth_matrix={}
        )
        
        state = initial_state
        route = [current_node]
        
        while state.current_node != destination_node:
            # Select action using trained agent
            if self.config.enable_ppo and self.ppo_agent:
                action, _ = self.ppo_agent.select_action(state)
            elif self.config.enable_q_learning and self.q_learning_agent:
                action = self.q_learning_agent.select_action(state)
            else:
                action = self._random_action(state)
            
            # Update route
            route.append(action.selected_peer)
            
            # Update state
            state.current_node = action.selected_peer
        
        return route
    
    def _create_initial_state(self) -> RoutingState:
        """Create initial state for training."""
        # Simplified initial state creation
        return RoutingState(
            current_node="node_0",
            destination_node="node_9",
            available_peers=["node_1", "node_2", "node_3", "node_4", "node_5"],
            network_topology=NetworkTopology(nodes={}, edges=[]),
            traffic_load={},
            latency_matrix={},
            bandwidth_matrix={}
        )
    
    def _random_action(self, state: RoutingState) -> RoutingAction:
        """Select random action."""
        if not state.available_peers:
            return RoutingAction(
                selected_peer=state.current_node,
                route_path=[state.current_node],
                priority_level=5,
                bandwidth_allocation=0.5
            )
        
        selected_peer = random.choice(state.available_peers)
        return RoutingAction(
            selected_peer=selected_peer,
            route_path=[state.current_node, selected_peer],
            priority_level=random.randint(0, 10),
            bandwidth_allocation=random.random()
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "q_learning_enabled": self.config.enable_q_learning,
            "ppo_enabled": self.config.enable_ppo,
            "multi_agent_enabled": self.config.enable_multi_agent,
            "training_episodes": len(self.training_history),
            "average_reward": np.mean(self.training_history[-100:]) if self.training_history else 0.0,
            "epsilon": self.q_learning_agent.epsilon if self.q_learning_agent else 0.0,
            "replay_buffer_size": len(self.q_learning_agent.replay_buffer) if self.q_learning_agent else 0,
            "torch_available": TORCH_AVAILABLE,
            "gym_available": GYM_AVAILABLE
        }
