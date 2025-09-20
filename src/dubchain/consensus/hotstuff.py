"""
HotStuff consensus mechanism implementation.

This module implements the HotStuff consensus algorithm, a modern BFT protocol
that provides linear communication complexity and optimistic responsiveness.
HotStuff is designed for high-throughput blockchain systems.

Key features:
- Linear communication complexity
- Optimistic responsiveness
- View-change protocol
- Safety and liveness guarantees
- Leader rotation
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    HotStuffMessage,
    HotStuffPhase,
    ValidatorStatus,
)
from .validator import Validator


@dataclass
class HotStuffState:
    """State for HotStuff consensus."""

    current_view: int = 0
    current_leader: Optional[str] = None
    leader_index: int = 0
    validators: List[str] = field(default_factory=list)
    messages: Dict[str, List[HotStuffMessage]] = field(default_factory=dict)
    prepared_blocks: Dict[str, str] = field(default_factory=dict)  # block_hash -> parent_hash
    committed_blocks: Set[str] = field(default_factory=set)
    view_change_timeout: float = 0.0
    view_change_counter: int = 0
    metrics: ConsensusMetrics = field(default_factory=ConsensusMetrics)


class HotStuffConsensus:
    """
    HotStuff consensus mechanism.

    HotStuff is a modern BFT consensus algorithm that provides:
    - Linear communication complexity (O(n) messages per decision)
    - Optimistic responsiveness (fast path when network is synchronous)
    - Safety and liveness guarantees
    - Leader rotation for fairness
    """

    def __init__(self, config: ConsensusConfig):
        """Initialize HotStuff consensus."""
        self.config = config
        self.state = HotStuffState()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def add_validator(self, validator_id: str) -> bool:
        """
        Add a validator to the consensus.

        Args:
            validator_id: Unique identifier for the validator

        Returns:
            True if addition successful, False otherwise
        """
        if validator_id in self.state.validators:
            return False

        self.state.validators.append(validator_id)
        
        # Set first leader if none exists
        if not self.state.current_leader:
            self.state.current_leader = validator_id

        return True

    def remove_validator(self, validator_id: str) -> bool:
        """
        Remove a validator from the consensus.

        Args:
            validator_id: ID of the validator to remove

        Returns:
            True if removal successful, False otherwise
        """
        if validator_id not in self.state.validators:
            return False

        self.state.validators.remove(validator_id)
        
        # Update leader if necessary
        if self.state.current_leader == validator_id:
            self._rotate_leader()

        return True

    def _rotate_leader(self) -> None:
        """Rotate to the next leader."""
        if not self.state.validators:
            self.state.current_leader = None
            return

        self.state.leader_index = (self.state.leader_index + 1) % len(self.state.validators)
        self.state.current_leader = self.state.validators[self.state.leader_index]
        self.state.current_view += 1

    def _get_leader(self) -> Optional[str]:
        """Get the current leader."""
        return self.state.current_leader

    def _is_leader(self, validator_id: str) -> bool:
        """Check if a validator is the current leader."""
        return validator_id == self.state.current_leader

    def _get_safety_threshold(self) -> int:
        """Get the safety threshold (2f+1)."""
        n = len(self.state.validators)
        f = (n - 1) // 3  # Maximum Byzantine validators
        return 2 * f + 1

    def propose_block(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """
        Propose a new block through HotStuff consensus.

        Args:
            block_data: Block data to propose

        Returns:
            ConsensusResult indicating success or failure
        """
        start_time = time.time()

        # Get proposer
        proposer_id = block_data.get("proposer_id")
        if not proposer_id:
            return ConsensusResult(
                success=False,
                error_message="No proposer specified",
                consensus_type=ConsensusType.HOTSTUFF,
            )

        # Check if proposer is current leader
        if not self._is_leader(proposer_id):
            return ConsensusResult(
                success=False,
                error_message="Not current leader",
                consensus_type=ConsensusType.HOTSTUFF,
            )

        # Validate block data
        if not self._validate_block_data(block_data):
            return ConsensusResult(
                success=False,
                error_message="Invalid block data",
                consensus_type=ConsensusType.HOTSTUFF,
            )

        # Generate block hash
        block_hash = self._generate_block_hash(block_data)

        # Start HotStuff protocol
        result = self._execute_hotstuff_protocol(block_data, block_hash, proposer_id)

        # Calculate gas used
        gas_used = len(str(block_data)) * 100

        return ConsensusResult(
            success=result["success"],
            block_hash=block_hash if result["success"] else None,
            validator_id=proposer_id,
            consensus_type=ConsensusType.HOTSTUFF,
            gas_used=gas_used,
            error_message=result.get("error_message"),
            metadata={
                "view": self.state.current_view,
                "leader": self.state.current_leader,
                "phase": result.get("phase"),
                "messages_count": len(self.state.messages.get(block_hash, [])),
            },
        )

    def _execute_hotstuff_protocol(self, block_data: Dict[str, Any], block_hash: str, proposer_id: str) -> Dict[str, Any]:
        """Execute the HotStuff consensus protocol."""
        try:
            # Phase 1: Prepare
            prepare_result = self._prepare_phase(block_hash, proposer_id)
            if not prepare_result["success"]:
                return prepare_result

            # Phase 2: Pre-commit
            precommit_result = self._precommit_phase(block_hash, proposer_id)
            if not precommit_result["success"]:
                return precommit_result

            # Phase 3: Commit
            commit_result = self._commit_phase(block_hash, proposer_id)
            if not commit_result["success"]:
                return commit_result

            # Phase 4: Decide
            decide_result = self._decide_phase(block_hash, proposer_id)
            if not decide_result["success"]:
                return decide_result

            # Block is committed
            self.state.committed_blocks.add(block_hash)
            self._rotate_leader()

            return {"success": True, "phase": "decide"}

        except Exception as e:
            return {"success": False, "error_message": f"HotStuff protocol error: {e}"}

    def _prepare_phase(self, block_hash: str, proposer_id: str) -> Dict[str, Any]:
        """Execute the prepare phase."""
        # Leader sends prepare message
        prepare_message = HotStuffMessage(
            message_type=HotStuffPhase.PREPARE,
            view_number=self.state.current_view,
            block_hash=block_hash,
            parent_hash=self._get_parent_hash(block_hash),
            validator_id=proposer_id,
            signature=self._sign_message(proposer_id, block_hash),
        )

        # Collect prepare votes
        prepare_votes = self._collect_votes(prepare_message, HotStuffPhase.PREPARE)
        
        if len(prepare_votes) < self._get_safety_threshold():
            return {"success": False, "error_message": "Insufficient prepare votes"}

        return {"success": True, "phase": "prepare"}

    def _precommit_phase(self, block_hash: str, proposer_id: str) -> Dict[str, Any]:
        """Execute the pre-commit phase."""
        # Leader sends pre-commit message
        precommit_message = HotStuffMessage(
            message_type=HotStuffPhase.PRE_COMMIT,
            view_number=self.state.current_view,
            block_hash=block_hash,
            parent_hash=self._get_parent_hash(block_hash),
            validator_id=proposer_id,
            signature=self._sign_message(proposer_id, block_hash),
        )

        # Collect pre-commit votes
        precommit_votes = self._collect_votes(precommit_message, HotStuffPhase.PRE_COMMIT)
        
        if len(precommit_votes) < self._get_safety_threshold():
            return {"success": False, "error_message": "Insufficient pre-commit votes"}

        return {"success": True, "phase": "precommit"}

    def _commit_phase(self, block_hash: str, proposer_id: str) -> Dict[str, Any]:
        """Execute the commit phase."""
        # Leader sends commit message
        commit_message = HotStuffMessage(
            message_type=HotStuffPhase.COMMIT,
            view_number=self.state.current_view,
            block_hash=block_hash,
            parent_hash=self._get_parent_hash(block_hash),
            validator_id=proposer_id,
            signature=self._sign_message(proposer_id, block_hash),
        )

        # Collect commit votes
        commit_votes = self._collect_votes(commit_message, HotStuffPhase.COMMIT)
        
        if len(commit_votes) < self._get_safety_threshold():
            return {"success": False, "error_message": "Insufficient commit votes"}

        return {"success": True, "phase": "commit"}

    def _decide_phase(self, block_hash: str, proposer_id: str) -> Dict[str, Any]:
        """Execute the decide phase."""
        # Leader sends decide message
        decide_message = HotStuffMessage(
            message_type=HotStuffPhase.DECIDE,
            view_number=self.state.current_view,
            block_hash=block_hash,
            parent_hash=self._get_parent_hash(block_hash),
            validator_id=proposer_id,
            signature=self._sign_message(proposer_id, block_hash),
        )

        # Collect decide votes
        decide_votes = self._collect_votes(decide_message, HotStuffPhase.DECIDE)
        
        if len(decide_votes) < self._get_safety_threshold():
            return {"success": False, "error_message": "Insufficient decide votes"}

        return {"success": True, "phase": "decide"}

    def _collect_votes(self, message: HotStuffMessage, phase: HotStuffPhase) -> List[HotStuffMessage]:
        """Collect votes for a specific phase."""
        votes = []
        
        # Simulate collecting votes from validators
        for validator_id in self.state.validators:
            if validator_id != message.validator_id:  # Don't vote for own message
                vote = HotStuffMessage(
                    message_type=phase,
                    view_number=message.view_number,
                    block_hash=message.block_hash,
                    parent_hash=message.parent_hash,
                    validator_id=validator_id,
                    signature=self._sign_message(validator_id, message.block_hash),
                )
                votes.append(vote)

        # Store messages
        if message.block_hash not in self.state.messages:
            self.state.messages[message.block_hash] = []
        self.state.messages[message.block_hash].extend(votes)

        return votes

    def _get_parent_hash(self, block_hash: str) -> str:
        """Get parent hash for a block."""
        # In a real implementation, this would look up the actual parent
        return hashlib.sha256(f"parent_{block_hash}".encode()).hexdigest()

    def _sign_message(self, validator_id: str, data: str) -> str:
        """Sign a message (simplified for demo)."""
        # In a real implementation, this would use actual cryptographic signatures
        return hashlib.sha256(f"{validator_id}_{data}".encode()).hexdigest()

    def _validate_block_data(self, block_data: Dict[str, Any]) -> bool:
        """Validate block data."""
        required_fields = ["block_number", "timestamp", "transactions", "previous_hash"]
        
        for field in required_fields:
            if field not in block_data:
                return False

        # Validate timestamp
        current_time = time.time()
        block_timestamp = block_data["timestamp"]
        if abs(current_time - block_timestamp) > 300:  # 5 minutes tolerance
            return False

        return True

    def _generate_block_hash(self, block_data: Dict[str, Any]) -> str:
        """Generate block hash."""
        data = f"{block_data['block_number']}{block_data['timestamp']}{block_data['previous_hash']}"
        return hashlib.sha256(data.encode()).hexdigest()

    def start_view_change(self) -> bool:
        """Start a view change process."""
        if self.state.view_change_counter >= self.config.hotstuff_max_view_changes:
            return False

        self.state.view_change_counter += 1
        self.state.view_change_timeout = time.time() + self.config.hotstuff_view_timeout
        self._rotate_leader()
        return True

    def get_current_view(self) -> int:
        """Get the current view number."""
        return self.state.current_view

    def get_current_leader(self) -> Optional[str]:
        """Get the current leader."""
        return self.state.current_leader

    def get_validators(self) -> List[str]:
        """Get list of validators."""
        return self.state.validators.copy()

    def get_committed_blocks(self) -> Set[str]:
        """Get set of committed blocks."""
        return self.state.committed_blocks.copy()

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        self.state.metrics.validator_count = len(self.state.validators)
        self.state.metrics.active_validators = len(self.state.validators)
        self.state.metrics.consensus_type = ConsensusType.HOTSTUFF

        return self.state.metrics

    def update_metrics(self, success: bool, block_time: float, gas_used: int) -> None:
        """Update consensus metrics."""
        self.state.metrics.total_blocks += 1
        if success:
            self.state.metrics.successful_blocks += 1
        else:
            self.state.metrics.failed_blocks += 1

        # Update average block time
        total_time = self.state.metrics.average_block_time * (self.state.metrics.total_blocks - 1)
        self.state.metrics.average_block_time = (
            total_time + block_time
        ) / self.state.metrics.total_blocks

        # Update average gas used
        total_gas = self.state.metrics.average_gas_used * (self.state.metrics.total_blocks - 1)
        self.state.metrics.average_gas_used = (
            total_gas + gas_used
        ) / self.state.metrics.total_blocks

        self.state.metrics.last_updated = time.time()

    def get_hotstuff_statistics(self) -> Dict[str, Any]:
        """Get HotStuff-specific statistics."""
        return {
            "current_view": self.state.current_view,
            "current_leader": self.state.current_leader,
            "leader_index": self.state.leader_index,
            "view_change_counter": self.state.view_change_counter,
            "committed_blocks": len(self.state.committed_blocks),
            "total_messages": sum(len(msgs) for msgs in self.state.messages.values()),
            "safety_threshold": self._get_safety_threshold(),
            "validator_count": len(self.state.validators),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_view": self.state.current_view,
            "current_leader": self.state.current_leader,
            "leader_index": self.state.leader_index,
            "validators": self.state.validators,
            "messages": {
                block_hash: [
                    {
                        "message_type": msg.message_type.value,
                        "view_number": msg.view_number,
                        "block_hash": msg.block_hash,
                        "parent_hash": msg.parent_hash,
                        "validator_id": msg.validator_id,
                        "signature": msg.signature,
                        "timestamp": msg.timestamp,
                    }
                    for msg in messages
                ]
                for block_hash, messages in self.state.messages.items()
            },
            "prepared_blocks": self.state.prepared_blocks,
            "committed_blocks": list(self.state.committed_blocks),
            "view_change_timeout": self.state.view_change_timeout,
            "view_change_counter": self.state.view_change_counter,
            "metrics": {
                "total_blocks": self.state.metrics.total_blocks,
                "successful_blocks": self.state.metrics.successful_blocks,
                "failed_blocks": self.state.metrics.failed_blocks,
                "average_block_time": self.state.metrics.average_block_time,
                "average_gas_used": self.state.metrics.average_gas_used,
                "validator_count": self.state.metrics.validator_count,
                "active_validators": self.state.metrics.active_validators,
                "consensus_type": self.state.metrics.consensus_type.value,
                "last_updated": self.state.metrics.last_updated,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: ConsensusConfig) -> "HotStuffConsensus":
        """Create from dictionary."""
        hotstuff = cls(config)

        # Restore state
        hotstuff.state.current_view = data["current_view"]
        hotstuff.state.current_leader = data["current_leader"]
        hotstuff.state.leader_index = data["leader_index"]
        hotstuff.state.validators = data["validators"]
        hotstuff.state.prepared_blocks = data["prepared_blocks"]
        hotstuff.state.committed_blocks = set(data["committed_blocks"])
        hotstuff.state.view_change_timeout = data["view_change_timeout"]
        hotstuff.state.view_change_counter = data["view_change_counter"]

        # Restore messages
        for block_hash, messages_data in data["messages"].items():
            messages = []
            for msg_data in messages_data:
                message = HotStuffMessage(
                    message_type=HotStuffPhase(msg_data["message_type"]),
                    view_number=msg_data["view_number"],
                    block_hash=msg_data["block_hash"],
                    parent_hash=msg_data["parent_hash"],
                    validator_id=msg_data["validator_id"],
                    signature=msg_data["signature"],
                    timestamp=msg_data["timestamp"],
                )
                messages.append(message)
            hotstuff.state.messages[block_hash] = messages

        # Restore metrics
        metrics_data = data["metrics"]
        hotstuff.state.metrics = ConsensusMetrics(
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

        return hotstuff

    def __del__(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
