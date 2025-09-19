"""
Main consensus engine for DubChain.

This module provides the main consensus engine that orchestrates all consensus mechanisms
and provides a unified interface for blockchain consensus operations.
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusState,
    ConsensusType,
)
from .delegated_proof_of_stake import DelegatedProofOfStake
from .hybrid_consensus import HybridConsensus
from .pbft import PracticalByzantineFaultTolerance
from .proof_of_stake import ProofOfStake
from .validator import Validator, ValidatorInfo, ValidatorManager, ValidatorSet


class ConsensusEngine:
    """Main consensus engine for DubChain."""

    def __init__(self, config: ConsensusConfig):
        """Initialize consensus engine."""
        self.config = config
        self.consensus_state = ConsensusState(
            current_consensus=config.consensus_type, config=config
        )

        # Initialize consensus mechanism
        self.consensus_mechanism = self._create_consensus_mechanism(
            config.consensus_type
        )

        # Validator management
        self.validator_manager = ValidatorManager(
            ValidatorSet(max_validators=config.max_validators)
        )

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Async support
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Metrics
        self.metrics = ConsensusMetrics(consensus_type=config.consensus_type)

    def _create_consensus_mechanism(
        self, consensus_type: ConsensusType
    ) -> Union[
        ProofOfStake,
        DelegatedProofOfStake,
        PracticalByzantineFaultTolerance,
        HybridConsensus,
    ]:
        """Create consensus mechanism based on type."""
        if consensus_type == ConsensusType.PROOF_OF_STAKE:
            return ProofOfStake(self.config)
        elif consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE:
            return DelegatedProofOfStake(self.config)
        elif consensus_type == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE:
            return PracticalByzantineFaultTolerance(self.config)
        elif consensus_type == ConsensusType.HYBRID:
            return HybridConsensus(self.config)
        else:
            # Default to Proof of Stake
            return ProofOfStake(self.config)

    def register_validator(self, validator: Validator, initial_stake: int = 0) -> bool:
        """Register a new validator."""
        # Register with validator manager
        success = self.validator_manager.register_validator(validator, initial_stake)
        if not success:
            return False

        # Register with consensus mechanism
        if hasattr(self.consensus_mechanism, "register_validator"):
            return self.consensus_mechanism.register_validator(validator, initial_stake)
        elif hasattr(self.consensus_mechanism, "register_delegate"):
            return self.consensus_mechanism.register_delegate(validator, initial_stake)
        elif hasattr(self.consensus_mechanism, "add_validator"):
            return self.consensus_mechanism.add_validator(validator)

        return True

    def stake_to_validator(
        self, validator_id: str, delegator_id: str, amount: int
    ) -> bool:
        """Stake tokens to a validator."""
        # Stake with validator manager
        success = self.validator_manager.stake(validator_id, delegator_id, amount)
        if not success:
            return False

        # Stake with consensus mechanism
        if hasattr(self.consensus_mechanism, "stake_to_validator"):
            return self.consensus_mechanism.stake_to_validator(
                validator_id, delegator_id, amount
            )
        elif hasattr(self.consensus_mechanism, "vote_for_delegate"):
            return self.consensus_mechanism.vote_for_delegate(
                delegator_id, validator_id, amount
            )

        return True

    def propose_block(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """Propose a new block through consensus."""
        start_time = time.time()

        # Validate block data
        if not self._validate_block_data(block_data):
            return ConsensusResult(
                success=False,
                error_message="Invalid block data",
                consensus_type=self.config.consensus_type,
            )

        # Propose block through consensus mechanism
        if hasattr(self.consensus_mechanism, "propose_block"):
            result = self.consensus_mechanism.propose_block(block_data)
        elif hasattr(self.consensus_mechanism, "finalize_block"):
            # For PoS, select proposer first
            proposer = self.consensus_mechanism.select_proposer(
                block_data.get("block_number", 0)
            )
            if proposer:
                result = self.consensus_mechanism.finalize_block(block_data, proposer)
            else:
                result = ConsensusResult(
                    success=False,
                    error_message="No proposer available",
                    consensus_type=self.config.consensus_type,
                )
        elif hasattr(self.consensus_mechanism, "produce_block"):
            result = self.consensus_mechanism.produce_block(block_data)
        elif hasattr(self.consensus_mechanism, "start_consensus"):
            result = self.consensus_mechanism.start_consensus(block_data)
        else:
            result = ConsensusResult(
                success=False,
                error_message="Unsupported consensus mechanism",
                consensus_type=self.config.consensus_type,
            )

        # Record performance
        block_time = time.time() - start_time
        self._record_performance(result, block_time)

        # Update consensus state
        self.consensus_state.update_metrics(result.success, block_time, result.gas_used)

        return result

    async def propose_block_async(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """Asynchronously propose a new block."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.propose_block, block_data)

    def _validate_block_data(self, block_data: Dict[str, Any]) -> bool:
        """Validate block data before consensus."""
        required_fields = ["block_number", "timestamp", "transactions", "previous_hash"]

        for field in required_fields:
            if field not in block_data:
                return False

        # Validate timestamp
        current_time = time.time()
        block_timestamp = block_data["timestamp"]
        if abs(current_time - block_timestamp) > 300:  # 5 minutes tolerance
            return False

        # Validate gas usage
        gas_used = block_data.get("gas_used", 0)
        if gas_used > self.config.max_gas_per_block:
            return False

        return True

    def _record_performance(self, result: ConsensusResult, block_time: float) -> None:
        """Record performance metrics."""
        performance_data = {
            "timestamp": time.time(),
            "success": result.success,
            "block_time": block_time,
            "gas_used": result.gas_used,
            "consensus_type": result.consensus_type.value
            if result.consensus_type
            else self.config.consensus_type.value,
            "validator_id": result.validator_id,
            "error_message": result.error_message,
        }

        self.performance_history.append(performance_data)

        # Keep history size manageable
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[
                -self.max_history_size :
            ]

    def get_validator_info(self, validator_id: str) -> Optional[ValidatorInfo]:
        """Get information about a validator."""
        return self.validator_manager.validator_set.validators.get(validator_id)

    def get_active_validators(self) -> List[str]:
        """Get list of active validators."""
        return list(self.validator_manager.validator_set.active_validators)

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        # Get metrics from consensus mechanism
        if hasattr(self.consensus_mechanism, "get_consensus_metrics"):
            mechanism_metrics = self.consensus_mechanism.get_consensus_metrics()
            self.metrics = mechanism_metrics

        # Update with validator manager metrics
        validator_metrics = self.validator_manager.get_validator_metrics()
        self.metrics.validator_count = validator_metrics.validator_count
        self.metrics.active_validators = validator_metrics.active_validators

        return self.metrics

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}

        recent_history = self.performance_history[-100:]  # Last 100 blocks

        total_blocks = len(recent_history)
        successful_blocks = sum(1 for p in recent_history if p["success"])
        failed_blocks = total_blocks - successful_blocks

        avg_block_time = sum(p["block_time"] for p in recent_history) / total_blocks
        avg_gas_used = sum(p["gas_used"] for p in recent_history) / total_blocks

        return {
            "total_blocks": total_blocks,
            "successful_blocks": successful_blocks,
            "failed_blocks": failed_blocks,
            "success_rate": successful_blocks / total_blocks if total_blocks > 0 else 0,
            "average_block_time": avg_block_time,
            "average_gas_used": avg_gas_used,
            "consensus_type": self.config.consensus_type.value,
            "active_validators": len(self.get_active_validators()),
            "total_validators": len(self.validator_manager.validator_set.validators),
        }

    def switch_consensus(self, new_consensus_type: ConsensusType) -> bool:
        """Switch to a different consensus mechanism."""
        if new_consensus_type == self.config.consensus_type:
            return True

        # Create new consensus mechanism
        new_consensus = self._create_consensus_mechanism(new_consensus_type)

        # Migrate validators
        success = self._migrate_validators(new_consensus)
        if not success:
            return False

        # Update configuration and state
        self.config.consensus_type = new_consensus_type
        self.consensus_mechanism = new_consensus
        self.consensus_state.current_consensus = new_consensus_type
        self.metrics.consensus_type = new_consensus_type

        return True

    def _migrate_validators(self, new_consensus) -> bool:
        """Migrate validators to new consensus mechanism."""
        try:
            validators = self.validator_manager.validator_set.validators

            for validator_info in validators.values():
                # Create validator object (simplified)
                validator = Validator(validator_info.validator_id, None)

                if hasattr(new_consensus, "register_validator"):
                    new_consensus.register_validator(
                        validator, validator_info.total_stake
                    )
                elif hasattr(new_consensus, "register_delegate"):
                    new_consensus.register_delegate(
                        validator, validator_info.total_stake
                    )
                elif hasattr(new_consensus, "add_validator"):
                    new_consensus.add_validator(validator)

            return True
        except Exception:
            return False

    def get_consensus_info(self) -> Dict[str, Any]:
        """Get comprehensive consensus information."""
        return {
            "consensus_type": self.config.consensus_type.value,
            "config": self.config.to_dict(),
            "state": {
                "current_consensus": self.consensus_state.current_consensus.value,
                "active_validators": self.consensus_state.active_validators,
                "current_proposer": self.consensus_state.current_proposer,
                "current_view": self.consensus_state.current_view,
                "last_block_time": self.consensus_state.last_block_time,
            },
            "metrics": self.get_consensus_metrics(),
            "performance": self.get_performance_statistics(),
            "validators": {
                "total": len(self.validator_manager.validator_set.validators),
                "active": len(self.get_active_validators()),
                "list": self.get_active_validators(),
            },
        }

    def shutdown(self) -> None:
        """Shutdown consensus engine."""
        self.executor.shutdown(wait=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
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
            "validator_manager": self.validator_manager.to_dict(),
            "performance_history": self.performance_history[-100:],  # Keep last 100
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
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusEngine":
        """Create from dictionary."""
        config = ConsensusConfig.from_dict(data["config"])
        engine = cls(config)

        # Restore validator manager
        engine.validator_manager = ValidatorManager.from_dict(data["validator_manager"])

        # Restore performance history
        engine.performance_history = data.get("performance_history", [])

        # Restore metrics
        metrics_data = data["metrics"]
        engine.metrics = ConsensusMetrics(
            total_blocks=metrics_data["total_blocks"],
            successful_blocks=metrics_data["successful_blocks"],
            failed_blocks=metrics_data["failed_blocks"],
            average_block_time=metrics_data["average_block_time"],
            average_gas_used=metrics_data["average_gas_used"],
            validator_count=metrics_data["validator_count"],
            active_validators=metrics_data["active_validators"],
            consensus_type=ConsensusType(metrics_data["consensus_type"]),
            last_updated=metrics_data["last_updated"],
        )

        return engine
