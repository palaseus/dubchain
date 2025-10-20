"""
Proposal execution engine with timelock and emergency management.

This module handles the execution of approved governance proposals with
timelock mechanisms, emergency pause capabilities, and execution validation.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..errors.exceptions import GovernanceError, ValidationError
from .core import GovernanceState, Proposal, ProposalStatus


class ExecutionStatus(Enum):
    """Status of proposal execution."""

    PENDING = "pending"
    QUEUED = "queued"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of proposal execution."""

    proposal_id: str
    status: ExecutionStatus
    executed_at: Optional[float] = None
    execution_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    gas_used: Optional[int] = None
    transaction_hash: Optional[str] = None
    block_height: Optional[int] = None

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.EXECUTED

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == ExecutionStatus.FAILED


@dataclass
class TimelockEntry:
    """Entry in the timelock queue."""

    proposal_id: str
    execution_block: int
    execution_data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    is_emergency: bool = False

    def can_execute(self, current_block: int) -> bool:
        """Check if this entry can be executed at the current block."""
        return current_block >= self.execution_block


class TimelockManager:
    """Manages timelock delays for proposal execution."""

    def __init__(self, default_delay: int = 100):
        """Initialize timelock manager."""
        self.default_delay = default_delay
        self.timelock_queue: Dict[str, TimelockEntry] = {}
        self.execution_history: List[ExecutionResult] = []

    def queue_proposal(
        self,
        proposal: Proposal,
        current_block: int,
        execution_data: Dict[str, Any],
        is_emergency: bool = False,
    ) -> TimelockEntry:
        """Queue a proposal for execution after timelock delay."""
        if proposal.proposal_id in self.timelock_queue:
            raise ValidationError(f"Proposal {proposal.proposal_id} already queued")

        # Calculate execution block
        delay = (
            proposal.execution_delay
            if proposal.execution_delay > 0
            else self.default_delay
        )
        if is_emergency:
            delay = min(delay, 10)  # Emergency proposals have shorter delay

        execution_block = current_block + delay

        # Create timelock entry
        entry = TimelockEntry(
            proposal_id=proposal.proposal_id,
            execution_block=execution_block,
            execution_data=execution_data,
            is_emergency=is_emergency,
        )

        self.timelock_queue[proposal.proposal_id] = entry
        return entry

    def get_ready_proposals(self, current_block: int) -> List[TimelockEntry]:
        """Get proposals ready for execution."""
        ready_proposals = []
        for entry in self.timelock_queue.values():
            if entry.can_execute(current_block):
                ready_proposals.append(entry)
        return ready_proposals

    def remove_proposal(self, proposal_id: str) -> bool:
        """Remove a proposal from the timelock queue."""
        if proposal_id in self.timelock_queue:
            del self.timelock_queue[proposal_id]
            return True
        return False

    def get_timelock_status(
        self, proposal_id: str, current_block: int
    ) -> Optional[Dict[str, Any]]:
        """Get timelock status for a proposal."""
        if proposal_id not in self.timelock_queue:
            return None

        entry = self.timelock_queue[proposal_id]
        blocks_remaining = max(0, entry.execution_block - current_block)

        return {
            "proposal_id": proposal_id,
            "execution_block": entry.execution_block,
            "current_block": current_block,
            "blocks_remaining": blocks_remaining,
            "can_execute": entry.can_execute(current_block),
            "is_emergency": entry.is_emergency,
            "created_at": entry.created_at,
        }


class EmergencyManager:
    """Manages emergency pause and fast-track capabilities."""

    def __init__(self, emergency_threshold: float = 0.8):
        """Initialize emergency manager."""
        self.emergency_threshold = emergency_threshold
        self.is_paused = False
        self.pause_reason = None
        self.pause_block = None
        self.pause_initiator = None
        self.emergency_proposals: Set[str] = set()

    def pause_governance(self, reason: str, block_height: int, initiator: str) -> None:
        """Pause governance due to emergency."""
        self.is_paused = True
        self.pause_reason = reason
        self.pause_block = block_height
        self.pause_initiator = initiator

    def resume_governance(self) -> None:
        """Resume governance after emergency."""
        self.is_paused = False
        self.pause_reason = None
        self.pause_block = None
        self.pause_initiator = None

    def is_emergency_proposal(self, proposal: Proposal) -> bool:
        """Check if a proposal qualifies as emergency."""
        return proposal.proposal_id in self.emergency_proposals

    def mark_emergency_proposal(self, proposal_id: str) -> None:
        """Mark a proposal as emergency."""
        self.emergency_proposals.add(proposal_id)

    def can_execute_emergency_proposal(self, proposal: Proposal) -> bool:
        """Check if emergency proposal can be executed during pause."""
        return (
            self.is_paused
            and self.is_emergency_proposal(proposal)
            and proposal.proposal_type.value == "emergency"
        )


class ExecutionEngine:
    """Main execution engine for governance proposals."""

    def __init__(self, governance_state: GovernanceState):
        """Initialize execution engine."""
        self.governance_state = governance_state
        self.timelock_manager = TimelockManager()
        self.emergency_manager = EmergencyManager()

        # Execution handlers for different proposal types
        self.execution_handlers: Dict[str, Callable] = {
            "parameter_change": self._execute_parameter_change,
            "treasury_spending": self._execute_treasury_spending,
            "upgrade": self._execute_upgrade,
            "emergency": self._execute_emergency,
            "custom": self._execute_custom,
        }

        # Execution history
        self.execution_history: List[ExecutionResult] = []
        self.failed_executions: List[ExecutionResult] = []

    def queue_proposal_for_execution(
        self,
        proposal: Proposal,
        current_block: int,
        execution_data: Optional[Dict[str, Any]] = None,
    ) -> TimelockEntry:
        """Queue a proposal for execution after timelock delay."""
        if proposal.status != ProposalStatus.QUEUED:
            raise ValidationError("Only queued proposals can be executed")

        if not proposal.can_execute():
            raise ValidationError("Proposal does not meet execution criteria")

        # Check if governance is paused
        if (
            self.emergency_manager.is_paused
            and not self.emergency_manager.can_execute_emergency_proposal(proposal)
        ):
            raise GovernanceError("Governance is paused and proposal is not emergency")

        # Prepare execution data
        if execution_data is None:
            execution_data = proposal.execution_data or {}

        # Check if it's an emergency proposal
        is_emergency = self.emergency_manager.is_emergency_proposal(proposal)

        # Queue for execution
        entry = self.timelock_manager.queue_proposal(
            proposal, current_block, execution_data, is_emergency
        )

        return entry

    def execute_proposal(
        self,
        proposal: Proposal,
        current_block: int,
        execution_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a governance proposal."""
        if proposal.status != ProposalStatus.QUEUED:
            raise ValidationError("Only queued proposals can be executed")

        if not proposal.can_execute():
            raise ValidationError("Proposal does not meet execution criteria")

        # Check timelock
        if proposal.execution_delay > 0:
            # Calculate blocks since proposal was queued
            # If no block numbers are set, assume it was queued at block 0
            queue_block = proposal.start_block or proposal.end_block or 0
            blocks_since_queued = current_block - queue_block
            if blocks_since_queued < proposal.execution_delay:
                raise ValidationError(
                    f"Proposal is still in timelock. {proposal.execution_delay - blocks_since_queued} blocks remaining"
                )

        # Check if governance is paused
        if (
            self.emergency_manager.is_paused
            and not self.emergency_manager.can_execute_emergency_proposal(proposal)
        ):
            raise GovernanceError("Governance is paused and proposal is not emergency")

        # Prepare execution data
        if execution_data is None:
            execution_data = proposal.execution_data or {}

        # Create execution result
        result = ExecutionResult(
            proposal_id=proposal.proposal_id,
            status=ExecutionStatus.EXECUTING,
            execution_data=execution_data,
        )

        try:
            # Execute based on proposal type
            handler = self.execution_handlers.get(proposal.proposal_type.value)
            if handler is None:
                raise ValidationError(
                    f"No execution handler for proposal type: {proposal.proposal_type.value}"
                )

            # Execute the proposal
            execution_result = handler(proposal, execution_data, current_block)

            # Update result
            result.status = ExecutionStatus.EXECUTED
            result.executed_at = time.time()
            result.execution_data = execution_result
            result.block_height = current_block

            # Update proposal status
            self.governance_state.update_proposal_status(
                proposal.proposal_id, ProposalStatus.EXECUTED
            )

        except Exception as e:
            # Execution failed
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            result.executed_at = time.time()
            result.block_height = current_block

            # Update proposal status
            self.governance_state.update_proposal_status(
                proposal.proposal_id, ProposalStatus.FAILED
            )

            # Add to failed executions
            self.failed_executions.append(result)

        # Add to execution history
        self.execution_history.append(result)

        return result

    def process_timelock_queue(self, current_block: int) -> List[ExecutionResult]:
        """Process all ready proposals in the timelock queue."""
        ready_proposals = self.timelock_manager.get_ready_proposals(current_block)
        results = []

        for entry in ready_proposals:
            proposal = self.governance_state.get_proposal(entry.proposal_id)
            if proposal is None:
                continue

            # Execute the proposal
            result = self.execute_proposal(
                proposal, current_block, entry.execution_data
            )
            results.append(result)

            # Remove from timelock queue
            self.timelock_manager.remove_proposal(entry.proposal_id)

        return results

    def _execute_parameter_change(
        self, proposal: Proposal, execution_data: Dict[str, Any], current_block: int
    ) -> Dict[str, Any]:
        """Execute a parameter change proposal."""
        parameter_name = execution_data.get("parameter_name")
        new_value = execution_data.get("new_value")

        if not parameter_name or new_value is None:
            raise ValidationError(
                "Parameter change requires parameter_name and new_value"
            )

        # TODO: Implement actual parameter change logic
        # This would involve:
        # 1. Retrieving current parameter value from blockchain state
        # 2. Validating the new value against governance rules
        # 3. Updating the blockchain state with the new parameter
        # 4. Emitting events for parameter changes

        # For now, return a structured response indicating the change would be made
        return {
            "parameter_name": parameter_name,
            "old_value": f"current_{parameter_name}_value",  # Would be retrieved from current state
            "new_value": new_value,
            "executed_at": time.time(),
            "block_height": current_block,
            "status": "pending_implementation",
        }

    def _execute_treasury_spending(
        self, proposal: Proposal, execution_data: Dict[str, Any], current_block: int
    ) -> Dict[str, Any]:
        """Execute a treasury spending proposal."""
        recipient = execution_data.get("recipient")
        amount = execution_data.get("amount")

        if not recipient or amount is None:
            raise ValidationError("Treasury spending requires recipient and amount")

        if amount <= 0:
            raise ValidationError("Treasury spending amount must be positive")

        # TODO: Implement actual treasury spending logic
        # This would involve:
        # 1. Validating treasury has sufficient funds
        # 2. Creating a treasury transaction
        # 3. Updating treasury balance
        # 4. Emitting treasury spending events

        # For now, return a structured response indicating the spending would be made
        return {
            "recipient": recipient,
            "amount": amount,
            "executed_at": time.time(),
            "block_height": current_block,
            "status": "pending_implementation",
            "treasury_balance": "unknown",  # Would be retrieved from current state
        }

    def _execute_upgrade(
        self, proposal: Proposal, execution_data: Dict[str, Any], current_block: int
    ) -> Dict[str, Any]:
        """Execute an upgrade proposal."""
        upgrade_type = execution_data.get("upgrade_type")
        upgrade_data = execution_data.get("upgrade_data")

        if not upgrade_type:
            raise ValidationError("Upgrade requires upgrade_type")

        # TODO: Implement actual upgrade logic
        # This would involve:
        # 1. Validating upgrade compatibility
        # 2. Scheduling upgrade activation
        # 3. Coordinating with validators
        # 4. Managing upgrade state transitions

        # For now, return a structured response indicating the upgrade would be scheduled
        return {
            "upgrade_type": upgrade_type,
            "upgrade_data": upgrade_data,
            "executed_at": time.time(),
            "block_height": current_block,
            "status": "pending_implementation",
            "activation_height": current_block + 100,  # Example delay
        }

    def _execute_emergency(
        self, proposal: Proposal, execution_data: Dict[str, Any], current_block: int
    ) -> Dict[str, Any]:
        """Execute an emergency proposal."""
        emergency_action = execution_data.get("emergency_action")

        if not emergency_action:
            raise ValidationError("Emergency proposal requires emergency_action")

        # TODO: Implement actual emergency action logic
        # This would involve:
        # 1. Validating emergency conditions
        # 2. Executing emergency protocols
        # 3. Coordinating with validators
        # 4. Managing emergency state

        # For now, return a structured response indicating the emergency action would be taken
        return {
            "emergency_action": emergency_action,
            "executed_at": time.time(),
            "block_height": current_block,
            "status": "pending_implementation",
            "severity": "high",  # Emergency actions are typically high severity
        }

    def _execute_custom(
        self, proposal: Proposal, execution_data: Dict[str, Any], current_block: int
    ) -> Dict[str, Any]:
        """Execute a custom proposal."""
        custom_action = execution_data.get("custom_action")

        if not custom_action:
            raise ValidationError("Custom proposal requires custom_action")

        # TODO: Implement actual custom action logic
        # This would involve:
        # 1. Validating custom action parameters
        # 2. Executing custom business logic
        # 3. Managing custom state changes
        # 4. Emitting custom events

        # For now, return a structured response indicating the custom action would be executed
        return {
            "custom_action": custom_action,
            "executed_at": time.time(),
            "block_height": current_block,
            "status": "pending_implementation",
            "action_type": "custom",
        }

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_executions = len(self.execution_history)
        successful_executions = len(
            [r for r in self.execution_history if r.is_successful()]
        )
        failed_executions = len(self.failed_executions)

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / max(total_executions, 1),
            "pending_timelock": len(self.timelock_manager.timelock_queue),
            "is_paused": self.emergency_manager.is_paused,
        }

    def emergency_pause(self, reason: str, block_height: int, initiator: str) -> None:
        """Pause governance due to emergency."""
        self.emergency_manager.pause_governance(reason, block_height, initiator)

    def emergency_resume(self) -> None:
        """Resume governance after emergency."""
        self.emergency_manager.resume_governance()
