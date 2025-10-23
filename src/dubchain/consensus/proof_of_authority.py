"""
Proof-of-Authority consensus mechanism implementation.

This module implements a Proof-of-Authority (PoA) consensus mechanism where
pre-approved authorities take turns proposing blocks. Authorities are selected
based on reputation scores and rotation schedules.

Key features:
- Pre-approved authority set
- Reputation-based authority selection
- Authority rotation and slashing
- Fast block finality
- Low energy consumption
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    PoAAuthority,
    PoAStatus,
    ValidatorStatus,
)
from .validator import Validator


@dataclass
class PoAState:
    """State for Proof-of-Authority consensus."""

    authorities: Dict[str, PoAAuthority] = field(default_factory=dict)
    current_authority_index: int = 0
    last_block_time: float = field(default_factory=time.time)
    rotation_counter: int = 0
    slashed_authorities: Set[str] = field(default_factory=set)
    metrics: ConsensusMetrics = field(default_factory=ConsensusMetrics)


class ProofOfAuthority:
    """
    Proof-of-Authority consensus mechanism.

    In PoA, a predefined set of authorities take turns proposing blocks.
    Authorities are selected based on reputation scores and rotation schedules.
    Misbehaving authorities can be slashed and removed from the authority set.
    """

    def __init__(self, config: ConsensusConfig):
        """Initialize Proof-of-Authority consensus."""
        self.config = config
        self.state = PoAState()
        self._initialize_authorities()

    def _initialize_authorities(self) -> None:
        """Initialize authority set from configuration."""
        for authority_id in self.config.poa_authority_set:
            authority = PoAAuthority(
                authority_id=authority_id,
                public_key=f"pubkey_{authority_id}",  # Simplified for demo
                status=PoAStatus.AUTHORITY,
                reputation_score=100.0,
            )
            self.state.authorities[authority_id] = authority

    def register_authority(self, authority_id: str, public_key: str) -> bool:
        """
        Register a new authority.

        Args:
            authority_id: Unique identifier for the authority
            public_key: Public key of the authority

        Returns:
            True if registration successful, False otherwise
        """
        if authority_id in self.state.authorities:
            return False

        authority = PoAAuthority(
            authority_id=authority_id,
            public_key=public_key,
            status=PoAStatus.CANDIDATE,
            reputation_score=50.0,  # Start with lower reputation
        )
        self.state.authorities[authority_id] = authority
        return True

    def revoke_authority(self, authority_id: str, reason: str = "misbehavior") -> bool:
        """
        Revoke an authority's status.

        Args:
            authority_id: ID of the authority to revoke
            reason: Reason for revocation

        Returns:
            True if revocation successful, False otherwise
        """
        if authority_id not in self.state.authorities:
            return False

        authority = self.state.authorities[authority_id]
        authority.status = PoAStatus.REVOKED
        authority.is_active = False
        self.state.slashed_authorities.add(authority_id)
        return True

    def slash_authority(self, authority_id: str, penalty: float) -> bool:
        """
        Slash an authority's reputation.

        Args:
            authority_id: ID of the authority to slash
            penalty: Reputation penalty to apply

        Returns:
            True if slashing successful, False otherwise
        """
        if authority_id not in self.state.authorities:
            return False

        authority = self.state.authorities[authority_id]
        authority.reputation_score = max(0.0, authority.reputation_score - penalty)

        # Revoke if reputation falls below threshold
        if authority.reputation_score < self.config.poa_reputation_threshold:
            return self.revoke_authority(authority_id, "low_reputation")

        return True

    def get_next_authority(self) -> Optional[str]:
        """
        Get the next authority to propose a block.

        Returns:
            Authority ID of the next proposer, or None if no valid authority
        """
        active_authorities = [
            auth
            for auth in self.state.authorities.values()
            if auth.is_active
            and auth.status == PoAStatus.AUTHORITY
            and auth.reputation_score >= self.config.poa_reputation_threshold
        ]

        if not active_authorities:
            return None

        # Simple round-robin selection
        authority = active_authorities[
            self.state.current_authority_index % len(active_authorities)
        ]
        self.state.current_authority_index = (
            self.state.current_authority_index + 1
        ) % len(active_authorities)
        return authority.authority_id

    def propose_block(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """
        Propose a new block through PoA consensus.

        Args:
            block_data: Block data to propose

        Returns:
            ConsensusResult indicating success or failure
        """
        start_time = time.time()

        # Get next authority
        authority_id = self.get_next_authority()
        if not authority_id:
            return ConsensusResult(
                success=False,
                error_message="No valid authority available",
                consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            )

        # Validate block data
        if not self._validate_block_data(block_data):
            # Slash authority for proposing invalid block
            self.slash_authority(authority_id, self.config.poa_slashing_threshold * 100)
            return ConsensusResult(
                success=False,
                error_message="Invalid block data",
                consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            )

        # Check if authority is within time window
        current_time = time.time()
        if current_time - self.state.last_block_time < self.config.block_time:
            return ConsensusResult(
                success=False,
                error_message="Block proposed too early",
                consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            )

        # Generate block hash
        block_hash = self._generate_block_hash(block_data, authority_id)

        # Update authority metrics
        authority = self.state.authorities[authority_id]
        authority.blocks_proposed += 1
        authority.last_activity = current_time

        # Update state
        self.state.last_block_time = current_time
        self.state.rotation_counter += 1

        # Check for authority rotation
        if self.state.rotation_counter >= self.config.poa_rotation_period:
            self._rotate_authorities()

        # Calculate gas used (simplified)
        gas_used = len(str(block_data)) * 100  # Rough estimate

        return ConsensusResult(
            success=True,
            block_hash=block_hash,
            validator_id=authority_id,
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            gas_used=gas_used,
            metadata={
                "authority_id": authority_id,
                "reputation_score": authority.reputation_score,
                "blocks_proposed": authority.blocks_proposed,
                "rotation_counter": self.state.rotation_counter,
            },
        )

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

        # Validate gas usage
        gas_used = block_data.get("gas_used", 0)
        if gas_used > self.config.max_gas_per_block:
            return False

        return True

    def _generate_block_hash(
        self, block_data: Dict[str, Any], authority_id: str
    ) -> str:
        """Generate block hash."""
        data = f"{block_data['block_number']}{block_data['timestamp']}{authority_id}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _rotate_authorities(self) -> None:
        """Rotate authority set based on performance."""
        # Simple rotation: promote candidates with high reputation
        candidates = [
            auth
            for auth in self.state.authorities.values()
            if auth.status == PoAStatus.CANDIDATE
            and auth.reputation_score >= self.config.poa_reputation_threshold
        ]

        # Promote best candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: x.reputation_score)
            best_candidate.status = PoAStatus.AUTHORITY

        # Reset rotation counter
        self.state.rotation_counter = 0

    def get_authority_info(self, authority_id: str) -> Optional[PoAAuthority]:
        """Get information about an authority."""
        return self.state.authorities.get(authority_id)

    def get_active_authorities(self) -> List[str]:
        """Get list of active authorities."""
        return [
            auth.authority_id
            for auth in self.state.authorities.values()
            if auth.is_active and auth.status == PoAStatus.AUTHORITY
        ]

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        active_count = len(self.get_active_authorities())
        total_count = len(self.state.authorities)

        self.state.metrics.validator_count = total_count
        self.state.metrics.active_validators = active_count
        self.state.metrics.consensus_type = ConsensusType.PROOF_OF_AUTHORITY

        return self.state.metrics

    def update_metrics(self, success: bool, block_time: float, gas_used: int) -> None:
        """Update consensus metrics."""
        self.state.metrics.total_blocks += 1
        if success:
            self.state.metrics.successful_blocks += 1
        else:
            self.state.metrics.failed_blocks += 1

        # Update average block time
        total_time = self.state.metrics.average_block_time * (
            self.state.metrics.total_blocks - 1
        )
        self.state.metrics.average_block_time = (
            total_time + block_time
        ) / self.state.metrics.total_blocks

        # Update average gas used
        total_gas = self.state.metrics.average_gas_used * (
            self.state.metrics.total_blocks - 1
        )
        self.state.metrics.average_gas_used = (
            total_gas + gas_used
        ) / self.state.metrics.total_blocks

        self.state.metrics.last_updated = time.time()

    def get_authority_rankings(self) -> List[Dict[str, Any]]:
        """Get authority rankings by reputation."""
        authorities = list(self.state.authorities.values())
        authorities.sort(key=lambda x: x.reputation_score, reverse=True)

        return [
            {
                "authority_id": auth.authority_id,
                "reputation_score": auth.reputation_score,
                "blocks_proposed": auth.blocks_proposed,
                "status": auth.status.value,
                "is_active": auth.is_active,
            }
            for auth in authorities
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "authorities": {
                auth_id: {
                    "authority_id": auth.authority_id,
                    "public_key": auth.public_key,
                    "status": auth.status.value,
                    "reputation_score": auth.reputation_score,
                    "blocks_proposed": auth.blocks_proposed,
                    "last_activity": auth.last_activity,
                    "is_active": auth.is_active,
                }
                for auth_id, auth in self.state.authorities.items()
            },
            "current_authority_index": self.state.current_authority_index,
            "last_block_time": self.state.last_block_time,
            "rotation_counter": self.state.rotation_counter,
            "slashed_authorities": list(self.state.slashed_authorities),
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
    def from_dict(
        cls, data: Dict[str, Any], config: ConsensusConfig
    ) -> "ProofOfAuthority":
        """Create from dictionary."""
        poa = cls(config)

        # Restore authorities
        for auth_id, auth_data in data["authorities"].items():
            authority = PoAAuthority(
                authority_id=auth_data["authority_id"],
                public_key=auth_data["public_key"],
                status=PoAStatus(auth_data["status"]),
                reputation_score=auth_data["reputation_score"],
                blocks_proposed=auth_data["blocks_proposed"],
                last_activity=auth_data["last_activity"],
                is_active=auth_data["is_active"],
            )
            poa.state.authorities[auth_id] = authority

        # Restore state
        poa.state.current_authority_index = data["current_authority_index"]
        poa.state.last_block_time = data["last_block_time"]
        poa.state.rotation_counter = data["rotation_counter"]
        poa.state.slashed_authorities = set(data["slashed_authorities"])

        # Restore metrics
        metrics_data = data["metrics"]
        poa.state.metrics = ConsensusMetrics(
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

        return poa
