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

        # Implement actual parameter change logic
        try:
            # Retrieve current parameter value from blockchain state
            old_value = self._get_current_parameter_value(parameter_name)
            
            # Validate the new value against governance rules
            if not self._validate_parameter_change(parameter_name, new_value, old_value):
                raise ValidationError(f"Invalid parameter change: {parameter_name}")

            # Update the blockchain state with the new parameter
            success = self._update_governance_parameter(parameter_name, new_value)
            
            if not success:
                raise ValidationError(f"Failed to update parameter: {parameter_name}")

            # Emit events for parameter changes
            self._emit_parameter_change_event(parameter_name, old_value, new_value, current_block)

            # Notify all affected components
            self._notify_parameter_change(parameter_name, new_value)

            return {
                "parameter_name": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "executed_at": time.time(),
                "block_height": current_block,
                "status": "executed",
                "success": True,
            }
            
        except Exception as e:
            return {
                "parameter_name": parameter_name,
                "old_value": old_value if 'old_value' in locals() else None,
                "new_value": new_value,
                "executed_at": time.time(),
                "block_height": current_block,
                "status": "failed",
                "error": str(e),
                "success": False,
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

        # Implement actual treasury spending logic
        try:
            # Validate treasury has sufficient funds
            treasury_balance = self._get_treasury_balance()
            if treasury_balance < amount:
                raise ValidationError(f"Insufficient treasury funds: {treasury_balance} < {amount}")

            # Validate recipient address
            if not self._validate_recipient_address(recipient):
                raise ValidationError(f"Invalid recipient address: {recipient}")

            # Create a treasury transaction
            transaction_id = self._create_treasury_transaction(recipient, amount, current_block)
            
            if not transaction_id:
                raise ValidationError("Failed to create treasury transaction")

            # Update treasury balance
            success = self._update_treasury_balance(treasury_balance - amount)
            
            if not success:
                raise ValidationError("Failed to update treasury balance")

            # Emit treasury spending events
            self._emit_treasury_spending_event(recipient, amount, transaction_id, current_block)

            return {
                "recipient": recipient,
                "amount": amount,
                "transaction_id": transaction_id,
                "treasury_balance_before": treasury_balance,
                "treasury_balance_after": treasury_balance - amount,
                "executed_at": time.time(),
                "block_height": current_block,
                "status": "executed",
                "success": True,
            }
            
        except Exception as e:
            return {
                "recipient": recipient,
                "amount": amount,
                "executed_at": time.time(),
                "block_height": current_block,
                "status": "failed",
                "error": str(e),
                "success": False,
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

    def _get_current_parameter_value(self, parameter_name: str) -> Any:
        """Get current parameter value from blockchain state."""
        # In a real implementation, this would query the blockchain state
        # For now, return a mock value based on parameter name
        mock_values = {
            "block_size_limit": 1000000,
            "gas_limit": 30000000,
            "consensus_timeout": 30,
            "validator_count": 100,
            "staking_minimum": 1000,
            "reward_rate": 0.05,
        }
        return mock_values.get(parameter_name, None)

    def _validate_parameter_change(self, parameter_name: str, new_value: Any, old_value: Any) -> bool:
        """Validate parameter change against governance rules."""
        # Basic validation rules
        if parameter_name == "block_size_limit":
            return isinstance(new_value, int) and 100000 <= new_value <= 10000000
        elif parameter_name == "gas_limit":
            return isinstance(new_value, int) and 1000000 <= new_value <= 100000000
        elif parameter_name == "consensus_timeout":
            return isinstance(new_value, int) and 5 <= new_value <= 300
        elif parameter_name == "validator_count":
            return isinstance(new_value, int) and 3 <= new_value <= 1000
        elif parameter_name == "staking_minimum":
            return isinstance(new_value, int) and new_value > 0
        elif parameter_name == "reward_rate":
            return isinstance(new_value, (int, float)) and 0 <= new_value <= 1
        
        # Default validation
        return new_value is not None and new_value != old_value

    def _update_governance_parameter(self, parameter_name: str, new_value: Any) -> bool:
        """Update governance parameter in blockchain state."""
        try:
            # In a real implementation, this would update the blockchain state
            # For now, simulate successful update
            print(f"Updating parameter {parameter_name} to {new_value}")
            return True
        except Exception as e:
            print(f"Failed to update parameter {parameter_name}: {e}")
            return False

    def _emit_parameter_change_event(self, parameter_name: str, old_value: Any, new_value: Any, block_height: int) -> None:
        """Emit parameter change event."""
        event = {
            "type": "parameter_change",
            "parameter_name": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            "block_height": block_height,
            "timestamp": time.time(),
        }
        print(f"Emitting parameter change event: {event}")

    def _notify_parameter_change(self, parameter_name: str, new_value: Any) -> None:
        """Notify all affected components of parameter change."""
        # In a real implementation, this would notify consensus, VM, network, etc.
        print(f"Notifying components of parameter change: {parameter_name} = {new_value}")

    def _get_treasury_balance(self) -> int:
        """Get current treasury balance."""
        # In a real implementation, this would query the blockchain state
        # For now, return a mock balance
        return 1000000  # 1M tokens

    def _validate_recipient_address(self, address: str) -> bool:
        """Validate recipient address format."""
        # Basic address validation
        if not address or len(address) < 20:
            return False
        
        # Check if it's a valid hex address
        try:
            int(address, 16)
            return True
        except ValueError:
            return False

    def _create_treasury_transaction(self, recipient: str, amount: int, block_height: int) -> str:
        """Create a treasury transaction."""
        try:
            # In a real implementation, this would create an actual transaction
            transaction_id = f"treasury_tx_{int(time.time())}_{recipient[:8]}"
            print(f"Created treasury transaction {transaction_id}: {amount} to {recipient}")
            return transaction_id
        except Exception as e:
            print(f"Failed to create treasury transaction: {e}")
            return None

    def _update_treasury_balance(self, new_balance: int) -> bool:
        """Update treasury balance."""
        try:
            # In a real implementation, this would update the blockchain state
            print(f"Updated treasury balance to {new_balance}")
            return True
        except Exception as e:
            print(f"Failed to update treasury balance: {e}")
            return False

    def _emit_treasury_spending_event(self, recipient: str, amount: int, transaction_id: str, block_height: int) -> None:
        """Emit treasury spending event."""
        event = {
            "type": "treasury_spending",
            "recipient": recipient,
            "amount": amount,
            "transaction_id": transaction_id,
            "block_height": block_height,
            "timestamp": time.time(),
        }
        print(f"Emitting treasury spending event: {event}")
