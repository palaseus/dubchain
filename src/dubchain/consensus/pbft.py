"""
Practical Byzantine Fault Tolerance (PBFT) consensus implementation for DubChain.

This module implements PBFT consensus with:
- Three-phase consensus protocol
- Byzantine fault tolerance
- View changes and recovery
- Message authentication and ordering
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature
from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    PBFTMessage,
    PBFTPhase,
)
from .validator import Validator, ValidatorInfo, ValidatorSet


@dataclass
class PBFTValidator:
    """PBFT-specific validator information."""

    validator_id: str
    public_key: PublicKey
    is_primary: bool = False
    view_number: int = 0
    sequence_number: int = 0
    prepared: bool = False
    committed: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    message_log: List[PBFTMessage] = field(default_factory=list)

    def add_message(self, message: PBFTMessage) -> None:
        """Add message to validator's log."""
        self.message_log.append(message)
        self.last_heartbeat = time.time()

    def is_online(self, timeout: float = 30.0) -> bool:
        """Check if validator is online."""
        return time.time() - self.last_heartbeat < timeout


class PracticalByzantineFaultTolerance:
    """PBFT consensus implementation."""

    def __init__(self, config: ConsensusConfig):
        """Initialize PBFT consensus."""
        self.config = config
        self.validators: Dict[str, PBFTValidator] = {}
        self.primary_validator: Optional[str] = None
        self.current_view = 0
        self.sequence_number = 0
        self.metrics = ConsensusMetrics(
            consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        )

        # Message tracking
        self.pre_prepare_messages: Dict[int, Dict[str, PBFTMessage]] = defaultdict(dict)
        self.prepare_messages: Dict[int, Dict[str, PBFTMessage]] = defaultdict(dict)
        self.commit_messages: Dict[int, Dict[str, PBFTMessage]] = defaultdict(dict)

        # Consensus state
        self.prepared_requests: Set[int] = set()
        self.committed_requests: Set[int] = set()
        self.checkpoint_interval = 100
        self.last_checkpoint = 0

    def add_validator(self, validator: Validator) -> bool:
        """Add validator to PBFT network."""
        if len(self.validators) >= self.config.max_validators:
            return False

        pbft_validator = PBFTValidator(
            validator_id=validator.validator_id, public_key=validator.public_key
        )

        self.validators[validator.validator_id] = pbft_validator

        # Set primary if this is the first validator
        if self.primary_validator is None:
            self.primary_validator = validator.validator_id
            pbft_validator.is_primary = True

        return True

    def remove_validator(self, validator_id: str) -> bool:
        """Remove validator from PBFT network."""
        if validator_id not in self.validators:
            return False

        # If removing primary, select new primary
        if self.primary_validator == validator_id:
            self._select_new_primary()

        del self.validators[validator_id]
        return True

    def _select_new_primary(self) -> None:
        """Select new primary validator."""
        if not self.validators:
            self.primary_validator = None
            return

        # Select validator with lowest ID as primary
        self.primary_validator = min(self.validators.keys())
        for validator in self.validators.values():
            validator.is_primary = validator.validator_id == self.primary_validator

    def start_consensus(self, request_data: Dict[str, Any]) -> ConsensusResult:
        """Start PBFT consensus process."""
        if not self.validators or not self.primary_validator:
            return ConsensusResult(
                success=False,
                error_message="No validators available",
                consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            )

        # Check if we have enough validators for fault tolerance
        if len(self.validators) < 3 * self.config.pbft_fault_tolerance + 1:
            return ConsensusResult(
                success=False,
                error_message="Insufficient validators for fault tolerance",
                consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            )

        self.sequence_number += 1
        request_hash = self._hash_request(request_data)

        # Phase 1: Pre-prepare
        pre_prepare_result = self._pre_prepare_phase(request_data, request_hash)
        if not pre_prepare_result:
            return ConsensusResult(
                success=False,
                error_message="Pre-prepare phase failed",
                consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            )

        # Phase 2: Prepare
        prepare_result = self._prepare_phase(request_hash)
        if not prepare_result:
            return ConsensusResult(
                success=False,
                error_message="Prepare phase failed",
                consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            )

        # Phase 3: Commit
        commit_result = self._commit_phase(request_hash)
        if not commit_result:
            return ConsensusResult(
                success=False,
                error_message="Commit phase failed",
                consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            )

        # Update metrics
        self.metrics.total_blocks += 1
        self.metrics.successful_blocks += 1
        self.metrics.last_updated = time.time()

        return ConsensusResult(
            success=True,
            block_hash=request_hash,
            validator_id=self.primary_validator,
            consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            timestamp=time.time(),
        )

    def _pre_prepare_phase(
        self, request_data: Dict[str, Any], request_hash: str
    ) -> bool:
        """Execute pre-prepare phase."""
        if not self.primary_validator:
            return False

        # Primary creates pre-prepare message
        pre_prepare_msg = PBFTMessage(
            message_type=PBFTPhase.PRE_PREPARE,
            view_number=self.current_view,
            sequence_number=self.sequence_number,
            block_hash=request_hash,
            validator_id=self.primary_validator,
            signature="",  # Would be signed in real implementation
            payload=request_data,
        )

        # Broadcast to all validators
        for validator_id in self.validators:
            if validator_id != self.primary_validator:
                self.validators[validator_id].add_message(pre_prepare_msg)

        # Store pre-prepare message
        self.pre_prepare_messages[self.sequence_number][
            self.primary_validator
        ] = pre_prepare_msg

        return True

    def _prepare_phase(self, request_hash: str) -> bool:
        """Execute prepare phase."""
        prepare_count = 0
        required_prepares = 2 * self.config.pbft_fault_tolerance  # 2f

        # Each validator sends prepare message
        for validator_id, validator in self.validators.items():
            if validator_id == self.primary_validator:
                continue  # Primary doesn't send prepare

            prepare_msg = PBFTMessage(
                message_type=PBFTPhase.PREPARE,
                view_number=self.current_view,
                sequence_number=self.sequence_number,
                block_hash=request_hash,
                validator_id=validator_id,
                signature="",  # Would be signed in real implementation
                payload={"request_hash": request_hash},
            )

            # Broadcast to all validators
            for other_validator_id in self.validators:
                self.validators[other_validator_id].add_message(prepare_msg)

            # Store prepare message
            self.prepare_messages[self.sequence_number][validator_id] = prepare_msg
            prepare_count += 1

        # Check if we have enough prepare messages
        return prepare_count >= required_prepares

    def _commit_phase(self, request_hash: str) -> bool:
        """Execute commit phase."""
        commit_count = 0
        required_commits = 2 * self.config.pbft_fault_tolerance + 1  # 2f + 1

        # Each validator sends commit message
        for validator_id, validator in self.validators.items():
            commit_msg = PBFTMessage(
                message_type=PBFTPhase.COMMIT,
                view_number=self.current_view,
                sequence_number=self.sequence_number,
                block_hash=request_hash,
                validator_id=validator_id,
                signature="",  # Would be signed in real implementation
                payload={"request_hash": request_hash},
            )

            # Broadcast to all validators
            for other_validator_id in self.validators:
                self.validators[other_validator_id].add_message(commit_msg)

            # Store commit message
            self.commit_messages[self.sequence_number][validator_id] = commit_msg
            commit_count += 1

        # Check if we have enough commit messages
        if commit_count >= required_commits:
            self.prepared_requests.add(self.sequence_number)
            self.committed_requests.add(self.sequence_number)
            return True

        return False

    def _hash_request(self, request_data: Dict[str, Any]) -> str:
        """Hash request data."""
        request_string = json.dumps(request_data, sort_keys=True)
        return SHA256Hasher.hash(request_string.encode()).to_hex()

    def handle_view_change(self, new_view: int, validator_id: str) -> bool:
        """Handle view change request."""
        if validator_id not in self.validators:
            return False

        # Check if view change is valid
        if new_view <= self.current_view:
            return False

        # Update view
        self.current_view = new_view

        # Select new primary
        self._select_new_primary()

        # Reset sequence number for new view
        self.sequence_number = 0

        return True

    def detect_byzantine_fault(self, validator_id: str) -> bool:
        """Detect Byzantine fault in validator."""
        if validator_id not in self.validators:
            return False

        validator = self.validators[validator_id]

        # Check for conflicting messages
        conflicting_messages = self._find_conflicting_messages(validator_id)
        if conflicting_messages:
            return True

        # Check for timeout
        if not validator.is_online():
            return True

        return False

    def _find_conflicting_messages(self, validator_id: str) -> List[PBFTMessage]:
        """Find conflicting messages from a validator."""
        validator = self.validators[validator_id]
        conflicting = []

        # Check for duplicate sequence numbers with different hashes
        sequence_hashes = {}
        for message in validator.message_log:
            if message.sequence_number in sequence_hashes:
                if sequence_hashes[message.sequence_number] != message.block_hash:
                    conflicting.append(message)
            else:
                sequence_hashes[message.sequence_number] = message.block_hash

        return conflicting

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        self.metrics.validator_count = len(self.validators)
        self.metrics.active_validators = len(
            [v for v in self.validators.values() if v.is_online()]
        )
        return self.metrics

    def get_validator_status(self, validator_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a validator."""
        if validator_id not in self.validators:
            return None

        validator = self.validators[validator_id]
        return {
            "validator_id": validator_id,
            "is_primary": validator.is_primary,
            "view_number": validator.view_number,
            "sequence_number": validator.sequence_number,
            "is_online": validator.is_online(),
            "message_count": len(validator.message_log),
            "last_heartbeat": validator.last_heartbeat,
        }

    def get_network_status(self) -> Dict[str, Any]:
        """Get overall network status."""
        online_validators = [v for v in self.validators.values() if v.is_online()]

        return {
            "total_validators": len(self.validators),
            "online_validators": len(online_validators),
            "primary_validator": self.primary_validator,
            "current_view": self.current_view,
            "sequence_number": self.sequence_number,
            "fault_tolerance": self.config.pbft_fault_tolerance,
            "max_faults": self.config.pbft_fault_tolerance,
            "prepared_requests": len(self.prepared_requests),
            "committed_requests": len(self.committed_requests),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "validators": {
                k: {
                    "validator_id": v.validator_id,
                    "public_key": v.public_key.to_hex(),
                    "is_primary": v.is_primary,
                    "view_number": v.view_number,
                    "sequence_number": v.sequence_number,
                    "prepared": v.prepared,
                    "committed": v.committed,
                    "last_heartbeat": v.last_heartbeat,
                    "message_count": len(v.message_log),
                }
                for k, v in self.validators.items()
            },
            "primary_validator": self.primary_validator,
            "current_view": self.current_view,
            "sequence_number": self.sequence_number,
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
            "prepared_requests": list(self.prepared_requests),
            "committed_requests": list(self.committed_requests),
            "checkpoint_interval": self.checkpoint_interval,
            "last_checkpoint": self.last_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PracticalByzantineFaultTolerance":
        """Create from dictionary."""
        config = ConsensusConfig.from_dict(data["config"])
        pbft = cls(config)

        # Restore validators (simplified - would need to restore full validator objects)
        for validator_id, validator_data in data["validators"].items():
            pbft_validator = PBFTValidator(
                validator_id=validator_data["validator_id"],
                public_key=PublicKey.from_hex(validator_data["public_key"]),
                is_primary=validator_data["is_primary"],
                view_number=validator_data["view_number"],
                sequence_number=validator_data["sequence_number"],
                prepared=validator_data["prepared"],
                committed=validator_data["committed"],
                last_heartbeat=validator_data["last_heartbeat"],
            )
            pbft.validators[validator_id] = pbft_validator

        pbft.primary_validator = data["primary_validator"]
        pbft.current_view = data["current_view"]
        pbft.sequence_number = data["sequence_number"]

        # Restore metrics
        metrics_data = data["metrics"]
        pbft.metrics = ConsensusMetrics(
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

        pbft.prepared_requests = set(data["prepared_requests"])
        pbft.committed_requests = set(data["committed_requests"])
        pbft.checkpoint_interval = data["checkpoint_interval"]
        pbft.last_checkpoint = data["last_checkpoint"]

        return pbft
