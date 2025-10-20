"""
Proof-of-History consensus mechanism implementation.

This module implements a Proof-of-History (PoH) consensus mechanism where
a verifiable delay function (VDF) creates a cryptographically secure
sequence of historical events that can be verified by any observer.

Key features:
- Verifiable delay function for time ordering
- Leader rotation based on PoH entries
- Fast verification of historical events
- Resistance to time manipulation attacks
- Deterministic block ordering
"""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusType,
    PoHEntry,
    PoHStatus,
    ValidatorStatus,
)
from .validator import Validator


@dataclass
class PoHState:
    """State for Proof-of-History consensus."""

    entries: List[PoHEntry] = field(default_factory=list)
    current_leader: Optional[str] = None
    leader_rotation_counter: int = 0
    last_entry_time: float = field(default_factory=time.time)
    poh_clock_running: bool = False
    metrics: ConsensusMetrics = field(default_factory=ConsensusMetrics)
    validators: Set[str] = field(default_factory=set)


class VerifiableDelayFunction:
    """
    Simple verifiable delay function implementation.

    In a real implementation, this would use a proper VDF like Wesolowski's
    or Pietrzak's VDF. This is a simplified version for demonstration.
    """

    def __init__(self, difficulty: int = 1000):
        """Initialize VDF with difficulty parameter."""
        self.difficulty = difficulty

    def compute(self, input_data: bytes) -> tuple[bytes, str]:
        """
        Compute VDF output and proof.

        Args:
            input_data: Input data for VDF computation

        Returns:
            Tuple of (output, proof)
        """
        # Simplified VDF: hash the input multiple times
        current_hash = hashlib.sha256(input_data).digest()

        for _ in range(self.difficulty):
            current_hash = hashlib.sha256(current_hash).digest()

        # Generate proof (simplified)
        proof = hashlib.sha256(current_hash + input_data).hexdigest()

        return current_hash, proof

    def verify(self, input_data: bytes, output: bytes, proof: str) -> bool:
        """
        Verify VDF computation.

        Args:
            input_data: Original input data
            output: VDF output
            proof: VDF proof

        Returns:
            True if verification successful, False otherwise
        """
        # Verify proof
        expected_proof = hashlib.sha256(output + input_data).hexdigest()
        if expected_proof != proof:
            return False

        # Verify output (simplified)
        current_hash = hashlib.sha256(input_data).digest()
        for _ in range(self.difficulty):
            current_hash = hashlib.sha256(current_hash).digest()

        return current_hash == output


class ProofOfHistory:
    """
    Proof-of-History consensus mechanism.

    PoH uses a verifiable delay function to create a cryptographically secure
    sequence of historical events. This provides a global clock that all
    validators can agree on, enabling fast consensus on block ordering.
    """

    def __init__(self, config: ConsensusConfig):
        """Initialize Proof-of-History consensus."""
        self.config = config
        self.state = PoHState()
        self.vdf = VerifiableDelayFunction(difficulty=1000)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._poh_thread: Optional[threading.Thread] = None
        self._stop_poh = threading.Event()

    def start_poh_clock(self) -> None:
        """Start the PoH clock generation."""
        if self.state.poh_clock_running:
            return

        self.state.poh_clock_running = True
        self._stop_poh.clear()
        self._poh_thread = threading.Thread(target=self._generate_poh_entries)
        self._poh_thread.daemon = True
        self._poh_thread.start()

    def stop_poh_clock(self) -> None:
        """Stop the PoH clock generation."""
        if not self.state.poh_clock_running:
            return

        self.state.poh_clock_running = False
        self._stop_poh.set()
        if self._poh_thread:
            self._poh_thread.join(timeout=5.0)

    def _generate_poh_entries(self) -> None:
        """Generate PoH entries continuously."""
        while not self._stop_poh.is_set():
            try:
                # Generate new PoH entry
                self._create_poh_entry()

                # Sleep based on clock frequency
                sleep_time = 1.0 / self.config.poh_clock_frequency
                time.sleep(sleep_time)
            except Exception as e:
                print(f"Error generating PoH entry: {e}")
                time.sleep(1.0)

    def _create_poh_entry(self) -> None:
        """Create a new PoH entry."""
        current_time = time.time()

        # Get previous hash
        previous_hash = ""
        if self.state.entries:
            previous_hash = self.state.entries[-1].hash

        # Create entry data
        entry_data = f"{current_time}{previous_hash}{len(self.state.entries)}".encode()

        # Compute VDF
        output, proof = self.vdf.compute(entry_data)
        entry_hash = output.hex()  # Use VDF output directly, not SHA256 of output

        # Create PoH entry
        entry = PoHEntry(
            entry_id=f"poh_{len(self.state.entries)}",
            timestamp=current_time,
            hash=entry_hash,
            previous_hash=previous_hash,
            data=entry_data,
            proof=proof,
            validator_id=self.state.current_leader or "system",
        )

        self.state.entries.append(entry)
        self.state.last_entry_time = current_time

        # Check for leader rotation
        if len(self.state.entries) % self.config.poh_leader_rotation == 0:
            self._rotate_leader()

    def _rotate_leader(self) -> None:
        """Rotate the current leader."""
        if not self.state.validators:
            return

        validator_list = list(self.state.validators)
        self.state.leader_rotation_counter += 1
        leader_index = self.state.leader_rotation_counter % len(validator_list)
        self.state.current_leader = validator_list[leader_index]

    def register_validator(self, validator_id: str) -> bool:
        """
        Register a validator for PoH consensus.

        Args:
            validator_id: Unique identifier for the validator

        Returns:
            True if registration successful, False otherwise
        """
        if validator_id in self.state.validators:
            return False

        self.state.validators.add(validator_id)

        # Set first leader if none exists
        if not self.state.current_leader:
            self.state.current_leader = validator_id

        return True

    def unregister_validator(self, validator_id: str) -> bool:
        """
        Unregister a validator from PoH consensus.

        Args:
            validator_id: Unique identifier for the validator

        Returns:
            True if unregistration successful, False otherwise
        """
        if validator_id not in self.state.validators:
            return False

        self.state.validators.remove(validator_id)

        # If the removed validator was the current leader, rotate to a new leader
        if self.state.current_leader == validator_id:
            self._rotate_leader()

        return True

    def propose_block(self, block_data: Dict[str, Any]) -> ConsensusResult:
        """
        Propose a new block through PoH consensus.

        Args:
            block_data: Block data to propose

        Returns:
            ConsensusResult indicating success or failure
        """
        start_time = time.time()

        # Check if proposer is current leader
        proposer_id = block_data.get("proposer_id")
        if proposer_id != self.state.current_leader:
            return ConsensusResult(
                success=False,
                error_message="Not current leader",
                consensus_type=ConsensusType.PROOF_OF_HISTORY,
            )

        # Validate block data
        if not self._validate_block_data(block_data):
            return ConsensusResult(
                success=False,
                error_message="Invalid block data",
                consensus_type=ConsensusType.PROOF_OF_HISTORY,
            )

        # Get latest PoH entry
        if not self.state.entries:
            return ConsensusResult(
                success=False,
                error_message="No PoH entries available",
                consensus_type=ConsensusType.PROOF_OF_HISTORY,
            )

        latest_entry = self.state.entries[-1]

        # Validate PoH entry
        if not self._validate_poh_entry(latest_entry):
            return ConsensusResult(
                success=False,
                error_message="Invalid PoH entry",
                consensus_type=ConsensusType.PROOF_OF_HISTORY,
            )

        # Generate block hash
        block_hash = self._generate_block_hash(block_data, latest_entry)

        # Calculate gas used
        gas_used = len(str(block_data)) * 100

        return ConsensusResult(
            success=True,
            block_hash=block_hash,
            validator_id=proposer_id,
            consensus_type=ConsensusType.PROOF_OF_HISTORY,
            gas_used=gas_used,
            metadata={
                "poh_entry_id": latest_entry.entry_id,
                "poh_timestamp": latest_entry.timestamp,
                "leader_rotation_counter": self.state.leader_rotation_counter,
                "total_entries": len(self.state.entries),
            },
        )

    def _validate_block_data(self, block_data: Dict[str, Any]) -> bool:
        """Validate block data."""
        required_fields = ["block_number", "timestamp", "transactions", "previous_hash"]

        for field in required_fields:
            if field not in block_data:
                return False

        # Validate timestamp against PoH
        block_timestamp = block_data["timestamp"]
        current_time = time.time()

        # Check if timestamp is within acceptable skew
        if abs(current_time - block_timestamp) > self.config.poh_max_skew:
            return False

        return True

    def _validate_poh_entry(self, entry: PoHEntry) -> bool:
        """Validate a PoH entry."""
        # Verify VDF computation
        return self.vdf.verify(entry.data, bytes.fromhex(entry.hash), entry.proof)

    def _generate_block_hash(
        self, block_data: Dict[str, Any], poh_entry: PoHEntry
    ) -> str:
        """Generate block hash including PoH entry."""
        data = f"{block_data['block_number']}{block_data['timestamp']}{poh_entry.hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_poh_sequence(self, start_index: int, end_index: int) -> bool:
        """
        Verify a sequence of PoH entries.

        Args:
            start_index: Starting index of entries to verify
            end_index: Ending index of entries to verify

        Returns:
            True if sequence is valid, False otherwise
        """
        if start_index < 0 or end_index >= len(self.state.entries):
            return False

        for i in range(start_index, end_index + 1):
            entry = self.state.entries[i]

            # Verify VDF
            if not self._validate_poh_entry(entry):
                return False

            # Verify hash chain
            if i > 0:
                previous_entry = self.state.entries[i - 1]
                if entry.previous_hash != previous_entry.hash:
                    return False

        return True

    def get_poh_entries(self, start_index: int = 0, count: int = 100) -> List[PoHEntry]:
        """
        Get PoH entries for verification.

        Args:
            start_index: Starting index
            count: Number of entries to return

        Returns:
            List of PoH entries
        """
        end_index = min(start_index + count, len(self.state.entries))
        return self.state.entries[start_index:end_index]

    def get_current_leader(self) -> Optional[str]:
        """Get current leader."""
        return self.state.current_leader

    def get_consensus_metrics(self) -> ConsensusMetrics:
        """Get consensus metrics."""
        self.state.metrics.validator_count = len(self.state.validators)
        self.state.metrics.active_validators = len(self.state.validators)
        self.state.metrics.consensus_type = ConsensusType.PROOF_OF_HISTORY

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp,
                    "hash": entry.hash,
                    "previous_hash": entry.previous_hash,
                    "data": entry.data.hex(),
                    "proof": entry.proof,
                    "validator_id": entry.validator_id,
                }
                for entry in self.state.entries
            ],
            "current_leader": self.state.current_leader,
            "leader_rotation_counter": self.state.leader_rotation_counter,
            "last_entry_time": self.state.last_entry_time,
            "poh_clock_running": self.state.poh_clock_running,
            "validators": list(self.state.validators),
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
    ) -> "ProofOfHistory":
        """Create from dictionary."""
        poh = cls(config)

        # Restore entries
        for entry_data in data["entries"]:
            entry = PoHEntry(
                entry_id=entry_data["entry_id"],
                timestamp=entry_data["timestamp"],
                hash=entry_data["hash"],
                previous_hash=entry_data["previous_hash"],
                data=bytes.fromhex(entry_data["data"]),
                proof=entry_data["proof"],
                validator_id=entry_data["validator_id"],
            )
            poh.state.entries.append(entry)

        # Restore state
        poh.state.current_leader = data["current_leader"]
        poh.state.leader_rotation_counter = data["leader_rotation_counter"]
        poh.state.last_entry_time = data["last_entry_time"]
        poh.state.poh_clock_running = data["poh_clock_running"]
        poh.state.validators = set(data["validators"])

        # Restore metrics
        metrics_data = data["metrics"]
        poh.state.metrics = ConsensusMetrics(
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

        return poh

    def __del__(self):
        """Cleanup resources."""
        self.stop_poh_clock()
        self.executor.shutdown(wait=False)
