"""
Hybrid consensus system for DubChain.

This module implements a sophisticated hybrid consensus that can:
- Switch between different consensus mechanisms
- Combine multiple consensus algorithms
- Optimize for different network conditions
- Provide fault tolerance and adaptability
"""

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusState,
    ConsensusType,
)
from .delegated_proof_of_stake import DelegatedProofOfStake
from .pbft import PracticalByzantineFaultTolerance
from .proof_of_stake import ProofOfStake
from .validator import Validator, ValidatorInfo


@dataclass
class ConsensusSelector:
    """Selects appropriate consensus mechanism based on conditions."""

    network_size_threshold: int = 50
    latency_threshold: float = 100.0  # milliseconds
    fault_tolerance_requirement: float = 0.33  # 33% fault tolerance
    energy_efficiency_weight: float = 0.3
    security_weight: float = 0.4
    performance_weight: float = 0.3

    def select_consensus(self, network_conditions: Dict[str, Any]) -> ConsensusType:
        """Select best consensus mechanism for current conditions."""
        network_size = network_conditions.get("network_size", 0)
        average_latency = network_conditions.get("average_latency", 0.0)
        fault_tolerance_needed = network_conditions.get("fault_tolerance", 0.0)

        # Decision matrix based on network conditions
        if network_size < 10:
            # Small network - use PBFT for strong consistency
            return ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        elif network_size < 50:
            # Medium network - use DPoS for efficiency
            return ConsensusType.DELEGATED_PROOF_OF_STAKE
        elif network_size < 200:
            # Large network - use PoS for decentralization
            return ConsensusType.PROOF_OF_STAKE
        else:
            # Very large network - use hybrid approach
            return ConsensusType.HYBRID

    def calculate_consensus_score(
        self, consensus_type: ConsensusType, network_conditions: Dict[str, Any]
    ) -> float:
        """Calculate score for a consensus mechanism."""
        scores = {
            ConsensusType.PROOF_OF_STAKE: self._score_pos(network_conditions),
            ConsensusType.DELEGATED_PROOF_OF_STAKE: self._score_dpos(
                network_conditions
            ),
            ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE: self._score_pbft(
                network_conditions
            ),
        }

        return scores.get(consensus_type, 0.0)

    def _score_pos(self, conditions: Dict[str, Any]) -> float:
        """Score for Proof of Stake."""
        network_size = conditions.get("network_size", 0)
        latency = conditions.get("average_latency", 0.0)

        # PoS is good for large networks with moderate latency
        size_score = min(network_size / 100, 1.0)
        latency_score = max(0, 1.0 - latency / 200.0)

        return (
            size_score * 0.4 + latency_score * 0.3 + 0.3
        ) * self.energy_efficiency_weight

    def _score_dpos(self, conditions: Dict[str, Any]) -> float:
        """Score for Delegated Proof of Stake."""
        network_size = conditions.get("network_size", 0)
        latency = conditions.get("average_latency", 0.0)

        # DPoS is good for medium networks with low latency
        size_score = 1.0 - abs(network_size - 25) / 50.0
        latency_score = max(0, 1.0 - latency / 100.0)

        return (size_score * 0.4 + latency_score * 0.4 + 0.2) * self.performance_weight

    def _score_pbft(self, conditions: Dict[str, Any]) -> float:
        """Score for PBFT."""
        network_size = conditions.get("network_size", 0)
        fault_tolerance = conditions.get("fault_tolerance", 0.0)

        # PBFT is good for small networks requiring high fault tolerance
        size_score = max(0, 1.0 - network_size / 20.0)
        fault_score = min(fault_tolerance / 0.5, 1.0)

        return (size_score * 0.3 + fault_score * 0.5 + 0.2) * self.security_weight


@dataclass
class ConsensusSwitcher:
    """Manages switching between consensus mechanisms."""

    switch_cooldown: float = 300.0  # 5 minutes
    performance_window: int = 100  # blocks
    last_switch_time: float = field(default_factory=time.time)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)

    def can_switch(self) -> bool:
        """Check if consensus can be switched."""
        return time.time() - self.last_switch_time >= self.switch_cooldown

    def record_performance(
        self,
        consensus_type: ConsensusType,
        block_time: float,
        success: bool,
        gas_used: int,
    ) -> None:
        """Record performance metrics."""
        performance_data = {
            "consensus_type": consensus_type,
            "block_time": block_time,
            "success": success,
            "gas_used": gas_used,
            "timestamp": time.time(),
        }

        self.performance_history.append(performance_data)

        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history = self.performance_history[
                -self.performance_window :
            ]

    def should_switch_consensus(
        self, current_consensus: ConsensusType, network_conditions: Dict[str, Any]
    ) -> Tuple[bool, ConsensusType]:
        """Determine if consensus should be switched."""
        if not self.can_switch():
            return False, current_consensus

        # Calculate performance metrics
        recent_performance = (
            self.performance_history[-50:]
            if len(self.performance_history) >= 50
            else self.performance_history
        )

        if not recent_performance:
            return False, current_consensus

        # Calculate success rate and average block time
        success_rate = sum(1 for p in recent_performance if p["success"]) / len(
            recent_performance
        )
        avg_block_time = sum(p["block_time"] for p in recent_performance) / len(
            recent_performance
        )

        # Check if performance is below threshold
        if success_rate < 0.8 or avg_block_time > 10.0:
            # Select better consensus mechanism
            selector = ConsensusSelector()
            new_consensus = selector.select_consensus(network_conditions)

            if new_consensus != current_consensus:
                self.last_switch_time = time.time()
                return True, new_consensus

        return False, current_consensus


class HybridConsensus:
    """Hybrid consensus system that can switch between mechanisms."""

    def __init__(self, config: ConsensusConfig):
        """Initialize hybrid consensus."""
        self.config = config
        self.current_consensus_type = config.consensus_type
        self.consensus_state = ConsensusState(
            current_consensus=self.current_consensus_type, config=config
        )

        # Initialize consensus mechanisms
        self.consensus_mechanisms: Dict[ConsensusType, Any] = {}
        self._initialize_consensus_mechanisms()

        # Consensus management
        self.selector = ConsensusSelector()
        self.switcher = ConsensusSwitcher()

        # Network monitoring
        self.network_conditions: Dict[str, Any] = {}
        self.last_condition_update = time.time()

        # Metrics
        self.metrics = ConsensusMetrics(consensus_type=ConsensusType.HYBRID)
        self.switch_count = 0

    def _initialize_consensus_mechanisms(self) -> None:
        """Initialize all consensus mechanisms."""
        # Create PoS instance
        pos_config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            block_time=self.config.block_time,
            max_validators=self.config.max_validators,
            min_stake=self.config.min_stake,
            unbonding_period=self.config.unbonding_period,
            slashing_percentage=self.config.slashing_percentage,
            reward_rate=self.config.reward_rate,
            max_gas_per_block=self.config.max_gas_per_block,
        )
        self.consensus_mechanisms[ConsensusType.PROOF_OF_STAKE] = ProofOfStake(
            pos_config
        )

        # Create DPoS instance
        dpos_config = ConsensusConfig(
            consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            block_time=self.config.block_time,
            max_validators=self.config.max_validators,
            min_stake=self.config.min_stake,
            unbonding_period=self.config.unbonding_period,
            slashing_percentage=self.config.slashing_percentage,
            reward_rate=self.config.reward_rate,
            max_gas_per_block=self.config.max_gas_per_block,
        )
        self.consensus_mechanisms[
            ConsensusType.DELEGATED_PROOF_OF_STAKE
        ] = DelegatedProofOfStake(dpos_config)

        # Create PBFT instance
        pbft_config = ConsensusConfig(
            consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            block_time=self.config.block_time,
            max_validators=self.config.max_validators,
            min_stake=self.config.min_stake,
            unbonding_period=self.config.unbonding_period,
            slashing_percentage=self.config.slashing_percentage,
            reward_rate=self.config.reward_rate,
            max_gas_per_block=self.config.max_gas_per_block,
            pbft_fault_tolerance=self.config.pbft_fault_tolerance,
        )
        self.consensus_mechanisms[
            ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        ] = PracticalByzantineFaultTolerance(pbft_config)

    def update_network_conditions(self, conditions: Dict[str, Any]) -> None:
        """Update network conditions for consensus selection."""
        self.network_conditions.update(conditions)
        self.last_condition_update = time.time()

        # Check if consensus should be switched
        if self.config.enable_hybrid:
            should_switch, new_consensus = self.switcher.should_switch_consensus(
                self.current_consensus_type, self.network_conditions
            )

            if should_switch:
                self.switch_consensus(new_consensus)

    def switch_consensus(self, new_consensus_type: ConsensusType) -> bool:
        """Switch to a different consensus mechanism."""
        if new_consensus_type not in self.consensus_mechanisms:
            return False

        if new_consensus_type == self.current_consensus_type:
            return True

        # Migrate validators to new consensus
        success = self._migrate_validators(new_consensus_type)
        if not success:
            return False

        # Update state
        old_consensus = self.current_consensus_type
        self.current_consensus_type = new_consensus_type
        self.consensus_state.current_consensus = new_consensus_type
        self.switch_count += 1

        # Record switch event
        switch_event = {
            "from_consensus": old_consensus.value,
            "to_consensus": new_consensus_type.value,
            "timestamp": time.time(),
            "network_conditions": self.network_conditions.copy(),
        }

        return True

    def _migrate_validators(self, new_consensus_type: ConsensusType) -> bool:
        """Migrate validators between consensus mechanisms."""
        try:
            current_consensus = self.consensus_mechanisms[self.current_consensus_type]
            new_consensus = self.consensus_mechanisms[new_consensus_type]

            # Get validators from current consensus
            if hasattr(current_consensus, "validator_set"):
                validators = current_consensus.validator_set.validators
            elif hasattr(current_consensus, "validators"):
                validators = current_consensus.validators
            else:
                return False

            # Add validators to new consensus
            for validator_info in validators.values():
                if new_consensus_type == ConsensusType.PROOF_OF_STAKE:
                    # Create validator object for PoS
                    from .validator import Validator

                    validator = Validator(
                        validator_info.validator_id, None
                    )  # Simplified
                    new_consensus.register_validator(
                        validator, validator_info.total_stake
                    )
                elif new_consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE:
                    # Add to DPoS
                    from .validator import Validator

                    validator = Validator(
                        validator_info.validator_id, None
                    )  # Simplified
                    new_consensus.register_delegate(
                        validator, validator_info.total_stake
                    )
                elif (
                    new_consensus_type
                    == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
                ):
                    # Add to PBFT
                    from .validator import Validator

                    validator = Validator(
                        validator_info.validator_id, None
                    )  # Simplified
                    new_consensus.add_validator(validator)

            return True
        except Exception:
            return False

    def propose_block(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """Propose a block using current consensus mechanism."""
        start_time = time.time()

        # Get current consensus mechanism
        if self.current_consensus_type == ConsensusType.HYBRID:
            # For hybrid, use the first available mechanism
            if self.consensus_mechanisms:
                current_consensus = list(self.consensus_mechanisms.values())[0]
            else:
                return ConsensusResult(
                    success=False,
                    error_message="No consensus mechanisms available",
                    consensus_type=self.current_consensus_type,
                )
        else:
            current_consensus = self.consensus_mechanisms[self.current_consensus_type]

        # Propose block
        if self.current_consensus_type == ConsensusType.PROOF_OF_STAKE:
            # Select proposer and finalize block
            proposer = current_consensus.select_proposer(
                block_data.get("block_number", 0)
            )
            if proposer:
                result = current_consensus.finalize_block(block_data, proposer)
            else:
                result = ConsensusResult(
                    success=False,
                    error_message="No proposer available",
                    consensus_type=self.current_consensus_type,
                )
        elif self.current_consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE:
            result = current_consensus.produce_block(block_data)
        elif (
            self.current_consensus_type
            == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        ):
            result = current_consensus.start_consensus(block_data)
        else:
            result = ConsensusResult(
                success=False,
                error_message="Unknown consensus type",
                consensus_type=self.current_consensus_type,
            )

        # Record performance
        block_time = time.time() - start_time
        self.switcher.record_performance(
            self.current_consensus_type, block_time, result.success, result.gas_used
        )

        # Update metrics
        self.consensus_state.update_metrics(result.success, block_time, result.gas_used)

        return result

    def get_consensus_info(self) -> Dict[str, Any]:
        """Get information about current consensus."""
        # For hybrid consensus, get info from the current active mechanism
        if self.current_consensus_type == ConsensusType.HYBRID:
            # Get the first available consensus mechanism
            if self.consensus_mechanisms:
                current_consensus = list(self.consensus_mechanisms.values())[0]
            else:
                return {
                    "current_consensus": "hybrid",
                    "error": "No consensus mechanisms available",
                }
        else:
            current_consensus = self.consensus_mechanisms[self.current_consensus_type]

        info = {
            "current_consensus": self.current_consensus_type.value,
            "switch_count": self.switch_count,
            "last_switch_time": self.switcher.last_switch_time,
            "can_switch": self.switcher.can_switch(),
            "network_conditions": self.network_conditions,
            "performance_history_length": len(self.switcher.performance_history),
        }

        # Add consensus-specific info
        if hasattr(current_consensus, "get_consensus_metrics"):
            info["consensus_metrics"] = current_consensus.get_consensus_metrics()

        return info

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get hybrid consensus metrics."""
        # Aggregate metrics from all consensus mechanisms
        total_blocks = 0
        successful_blocks = 0
        total_validators = 0
        active_validators = 0

        for consensus_type, mechanism in self.consensus_mechanisms.items():
            if hasattr(mechanism, "get_consensus_metrics"):
                metrics = mechanism.get_consensus_metrics()
                total_blocks += metrics.total_blocks
                successful_blocks += metrics.successful_blocks
                total_validators += metrics.validator_count
                active_validators += metrics.active_validators

        self.metrics.total_blocks = total_blocks
        self.metrics.successful_blocks = successful_blocks
        self.metrics.validator_count = total_validators
        self.metrics.active_validators = active_validators
        self.metrics.consensus_type = ConsensusType.HYBRID

        return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "current_consensus_type": self.current_consensus_type.value,
            "consensus_state": {
                "current_consensus": self.consensus_state.current_consensus.value,
                "active_validators": self.consensus_state.active_validators,
                "current_proposer": self.consensus_state.current_proposer,
                "current_view": self.consensus_state.current_view,
                "last_block_time": self.consensus_state.last_block_time,
                "metrics": {
                    "total_blocks": self.consensus_state.metrics.total_blocks,
                    "successful_blocks": self.consensus_state.metrics.successful_blocks,
                    "failed_blocks": self.consensus_state.metrics.failed_blocks,
                    "average_block_time": self.consensus_state.metrics.average_block_time,
                    "average_gas_used": self.consensus_state.metrics.average_gas_used,
                    "validator_count": self.consensus_state.metrics.validator_count,
                    "active_validators": self.consensus_state.metrics.active_validators,
                    "consensus_type": self.consensus_state.metrics.consensus_type.value,
                    "last_updated": self.consensus_state.metrics.last_updated,
                },
            },
            "switcher": {
                "switch_cooldown": self.switcher.switch_cooldown,
                "performance_window": self.switcher.performance_window,
                "last_switch_time": self.switcher.last_switch_time,
                "performance_history": self.switcher.performance_history,
            },
            "network_conditions": self.network_conditions,
            "last_condition_update": self.last_condition_update,
            "switch_count": self.switch_count,
            "metrics": {
                "total_blocks": self.metrics.total_blocks,
                "successful_blocks": self.metrics.successful_blocks,
                "failed_blocks": self.metrics.failed_blocks,
                "average_block_time": self.metrics.average_block_time,
                "average_gas_used": self.metrics.average_gas_used,
                "validator_count": self.metrics.validator_count,
                "active_validators": self.metrics.active_validators,
                "consensus_type": self.metrics.consensus_type.value,
                "last_updated": self.metrics.last_updated,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridConsensus":
        """Create from dictionary."""
        config = ConsensusConfig.from_dict(data["config"])
        hybrid = cls(config)

        # Restore state
        hybrid.current_consensus_type = ConsensusType(data["current_consensus_type"])

        # Restore switcher
        switcher_data = data["switcher"]
        hybrid.switcher = ConsensusSwitcher(
            switch_cooldown=switcher_data["switch_cooldown"],
            performance_window=switcher_data["performance_window"],
        )
        hybrid.switcher.last_switch_time = switcher_data["last_switch_time"]
        hybrid.switcher.performance_history = switcher_data["performance_history"]

        # Restore other state
        hybrid.network_conditions = data["network_conditions"]
        hybrid.last_condition_update = data["last_condition_update"]
        hybrid.switch_count = data["switch_count"]

        return hybrid
