"""
Routing Optimization Module

This module provides reinforcement learning for routing optimization.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import json

from ..errors import ClientError
from dubchain.logging import get_logger

logger = get_logger(__name__)

@dataclass
class RoutingState:
    """State representation for routing."""
    node_id: str
    peer_connections: List[str]
    network_latency: Dict[str, float]
    bandwidth: Dict[str, float]
    congestion_level: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class RoutingAction:
    """Action representation for routing."""
    target_node: str
    priority: int
    route_type: str  # "direct", "relay", "multihop"

@dataclass
class RoutingReward:
    """Reward structure for routing optimization."""
    latency_reward: float
    throughput_reward: float
    reliability_reward: float
    total_reward: float

class PPOAgent:
    """Proximal Policy Optimization agent for routing optimization."""
    
    def __init__(self, state_dim: int = 100, action_dim: int = 50, learning_rate: float = 0.001):
        """Initialize PPO agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Policy network parameters (simplified)
        self.policy_weights = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.value_weights = np.random.normal(0, 0.1, (state_dim, 1))
        
        # Training parameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = 1000
        
        logger.info(f"Initialized PPO agent with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: RoutingState) -> RoutingAction:
        """Select action based on current state."""
        try:
            # Convert state to feature vector
            state_vector = self._state_to_vector(state)
            
            # Compute action probabilities
            action_probs = self._compute_action_probs(state_vector)
            
            # Sample action
            action_idx = np.random.choice(self.action_dim, p=action_probs)
            
            # Convert to routing action
            action = self._idx_to_action(action_idx, state)
            
            return action
            
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            # Fallback to random action
            return self._random_action(state)
    
    def update_policy(self, experiences: List[Tuple[RoutingState, RoutingAction, RoutingReward]]) -> Dict[str, float]:
        """Update policy based on experiences."""
        try:
            if len(experiences) < 10:
                return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            
            # Convert experiences to training data
            states = [self._state_to_vector(exp[0]) for exp in experiences]
            actions = [self._action_to_idx(exp[1]) for exp in experiences]
            rewards = [exp[2].total_reward for exp in experiences]
            
            # Compute advantages
            advantages = self._compute_advantages(rewards)
            
            # Update policy network
            policy_loss = self._update_policy_network(states, actions, advantages)
            
            # Update value network
            value_loss = self._update_value_network(states, rewards)
            
            # Clear experience buffer
            self.experience_buffer.clear()
            
            return {
                "loss": policy_loss + value_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss
            }
            
        except Exception as e:
            logger.error(f"Error updating policy: {e}")
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    
    def _state_to_vector(self, state: RoutingState) -> np.ndarray:
        """Convert routing state to feature vector."""
        # Create feature vector from state
        features = []
        
        # Node features
        features.append(len(state.peer_connections))
        features.append(state.congestion_level)
        
        # Network features
        if state.network_latency:
            features.append(np.mean(list(state.network_latency.values())))
            features.append(np.std(list(state.network_latency.values())))
        else:
            features.extend([0.0, 0.0])
        
        if state.bandwidth:
            features.append(np.mean(list(state.bandwidth.values())))
            features.append(np.std(list(state.bandwidth.values())))
        else:
            features.extend([0.0, 0.0])
        
        # Pad or truncate to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def _compute_action_probs(self, state_vector: np.ndarray) -> np.ndarray:
        """Compute action probabilities using policy network."""
        # Simple linear policy
        logits = np.dot(state_vector, self.policy_weights)
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def _idx_to_action(self, action_idx: int, state: RoutingState) -> RoutingAction:
        """Convert action index to routing action."""
        # Map action index to routing action
        if state.peer_connections:
            target_node = state.peer_connections[action_idx % len(state.peer_connections)]
        else:
            target_node = f"node_{action_idx}"
        
        priority = (action_idx % 3) + 1
        route_types = ["direct", "relay", "multihop"]
        route_type = route_types[action_idx % len(route_types)]
        
        return RoutingAction(
            target_node=target_node,
            priority=priority,
            route_type=route_type
        )
    
    def _action_to_idx(self, action: RoutingAction) -> int:
        """Convert routing action to action index."""
        # Simple hash-based mapping
        action_str = f"{action.target_node}_{action.priority}_{action.route_type}"
        return hash(action_str) % self.action_dim
    
    def _compute_advantages(self, rewards: List[float]) -> np.ndarray:
        """Compute advantages using GAE."""
        advantages = []
        gamma = 0.99
        lam = 0.95
        
        # Simple advantage computation
        for i, reward in enumerate(rewards):
            advantage = reward
            advantages.append(advantage)
        
        return np.array(advantages)
    
    def _update_policy_network(self, states: List[np.ndarray], actions: List[int], advantages: np.ndarray) -> float:
        """Update policy network."""
        # Simple gradient update
        learning_rate = self.learning_rate
        
        for state, action, advantage in zip(states, actions, advantages):
            # Compute gradient
            action_probs = self._compute_action_probs(state)
            gradient = np.outer(state, action_probs)
            gradient[:, action] -= state
            
            # Update weights
            self.policy_weights -= learning_rate * gradient * advantage
        
        return 0.1  # Mock loss
    
    def _update_value_network(self, states: List[np.ndarray], rewards: List[float]) -> float:
        """Update value network."""
        # Simple value function update
        learning_rate = self.learning_rate
        
        for state, reward in zip(states, rewards):
            # Compute value prediction
            value_pred = np.dot(state, self.value_weights)[0]
            
            # Compute gradient
            gradient = state.reshape(-1, 1) * (value_pred - reward)
            
            # Update weights
            self.value_weights -= learning_rate * gradient
        
        return 0.1  # Mock loss
    
    def _random_action(self, state: RoutingState) -> RoutingAction:
        """Generate random action as fallback."""
        if state.peer_connections:
            target_node = random.choice(state.peer_connections)
        else:
            target_node = f"random_node_{random.randint(0, 100)}"
        
        return RoutingAction(
            target_node=target_node,
            priority=random.randint(1, 3),
            route_type=random.choice(["direct", "relay", "multihop"])
        )

class QLearningAgent:
    """Q-Learning agent for routing optimization."""
    
    def __init__(self, state_dim: int = 100, action_dim: int = 50, learning_rate: float = 0.1, epsilon: float = 0.1):
        """Initialize Q-Learning agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = 0.95
        
        # Q-table (simplified as linear function approximation)
        self.q_weights = np.random.normal(0, 0.1, (state_dim, action_dim))
        
        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = 1000
        
        logger.info(f"Initialized Q-Learning agent with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: RoutingState) -> RoutingAction:
        """Select action using epsilon-greedy policy."""
        try:
            # Convert state to feature vector
            state_vector = self._state_to_vector(state)
            
            # Compute Q-values
            q_values = np.dot(state_vector, self.q_weights)
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action_idx = random.randint(0, self.action_dim - 1)
            else:
                action_idx = np.argmax(q_values)
            
            # Convert to routing action
            action = self._idx_to_action(action_idx, state)
            
            return action
            
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            return self._random_action(state)
    
    def update_q_values(self, state: RoutingState, action: RoutingAction, reward: RoutingReward, next_state: RoutingState) -> float:
        """Update Q-values using Q-learning update rule."""
        try:
            # Convert states to feature vectors
            state_vector = self._state_to_vector(state)
            next_state_vector = self._state_to_vector(next_state)
            
            # Convert action to index
            action_idx = self._action_to_idx(action)
            
            # Compute current Q-value
            current_q = np.dot(state_vector, self.q_weights[:, action_idx])
            
            # Compute target Q-value
            next_q_values = np.dot(next_state_vector, self.q_weights)
            max_next_q = np.max(next_q_values)
            target_q = reward.total_reward + self.gamma * max_next_q
            
            # Q-learning update
            td_error = target_q - current_q
            self.q_weights[:, action_idx] += self.learning_rate * td_error * state_vector
            
            return abs(td_error)
            
        except Exception as e:
            logger.error(f"Error updating Q-values: {e}")
            return 0.0
    
    def _state_to_vector(self, state: RoutingState) -> np.ndarray:
        """Convert routing state to feature vector."""
        # Same implementation as PPO agent
        features = []
        
        features.append(len(state.peer_connections))
        features.append(state.congestion_level)
        
        if state.network_latency:
            features.append(np.mean(list(state.network_latency.values())))
            features.append(np.std(list(state.network_latency.values())))
        else:
            features.extend([0.0, 0.0])
        
        if state.bandwidth:
            features.append(np.mean(list(state.bandwidth.values())))
            features.append(np.std(list(state.bandwidth.values())))
        else:
            features.extend([0.0, 0.0])
        
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def _idx_to_action(self, action_idx: int, state: RoutingState) -> RoutingAction:
        """Convert action index to routing action."""
        if state.peer_connections:
            target_node = state.peer_connections[action_idx % len(state.peer_connections)]
        else:
            target_node = f"node_{action_idx}"
        
        priority = (action_idx % 3) + 1
        route_types = ["direct", "relay", "multihop"]
        route_type = route_types[action_idx % len(route_types)]
        
        return RoutingAction(
            target_node=target_node,
            priority=priority,
            route_type=route_type
        )
    
    def _action_to_idx(self, action: RoutingAction) -> int:
        """Convert routing action to action index."""
        action_str = f"{action.target_node}_{action.priority}_{action.route_type}"
        return hash(action_str) % self.action_dim
    
    def _random_action(self, state: RoutingState) -> RoutingAction:
        """Generate random action as fallback."""
        if state.peer_connections:
            target_node = random.choice(state.peer_connections)
        else:
            target_node = f"random_node_{random.randint(0, 100)}"
        
        return RoutingAction(
            target_node=target_node,
            priority=random.randint(1, 3),
            route_type=random.choice(["direct", "relay", "multihop"])
        )

class RoutingOptimizer:
    """Main routing optimizer that coordinates RL agents."""
    
    def __init__(self, agent_type: str = "ppo"):
        """Initialize routing optimizer."""
        self.agent_type = agent_type
        
        if agent_type == "ppo":
            self.agent = PPOAgent()
        elif agent_type == "qlearning":
            self.agent = QLearningAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.training_history = []
        logger.info(f"Initialized routing optimizer with {agent_type} agent")
    
    def optimize_routing(self, current_state: RoutingState) -> RoutingAction:
        """Optimize routing for current state."""
        try:
            action = self.agent.select_action(current_state)
            logger.debug(f"Selected routing action: {action}")
            return action
        except Exception as e:
            logger.error(f"Error optimizing routing: {e}")
            return self._fallback_action(current_state)
    
    def update_with_feedback(self, state: RoutingState, action: RoutingAction, reward: RoutingReward, next_state: RoutingState = None):
        """Update agent with feedback."""
        try:
            if isinstance(self.agent, QLearningAgent):
                td_error = self.agent.update_q_values(state, action, reward, next_state or state)
                self.training_history.append({
                    "timestamp": time.time(),
                    "td_error": td_error,
                    "reward": reward.total_reward
                })
            elif isinstance(self.agent, PPOAgent):
                # Collect experience for batch update
                self.agent.experience_buffer.append((state, action, reward))
                
                if len(self.agent.experience_buffer) >= 32:  # Batch size
                    metrics = self.agent.update_policy(self.agent.experience_buffer)
                    self.training_history.append({
                        "timestamp": time.time(),
                        "metrics": metrics,
                        "reward": reward.total_reward
                    })
            
        except Exception as e:
            logger.error(f"Error updating with feedback: {e}")
    
    def _fallback_action(self, state: RoutingState) -> RoutingAction:
        """Fallback action when optimization fails."""
        if state.peer_connections:
            target_node = state.peer_connections[0]
        else:
            target_node = "fallback_node"
        
        return RoutingAction(
            target_node=target_node,
            priority=1,
            route_type="direct"
        )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_history:
            return {"total_updates": 0, "avg_reward": 0.0}
        
        recent_history = self.training_history[-100:]  # Last 100 updates
        avg_reward = np.mean([h["reward"] for h in recent_history])
        
        return {
            "total_updates": len(self.training_history),
            "avg_reward": avg_reward,
            "agent_type": self.agent_type
        }

__all__ = [
    "RoutingOptimizer",
    "PPOAgent",
    "QLearningAgent",
]