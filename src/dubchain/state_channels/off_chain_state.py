"""
Off-Chain State Management for State Channels

This module handles off-chain state management including:
- State transition validation and execution
- Cryptographic signature management
- State synchronization between participants
- Off-chain consensus mechanisms
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature
from .channel_protocol import (
    ChannelConfig,
    ChannelId,
    ChannelState,
    ChannelStatus,
    StateUpdate,
    StateUpdateType,
)


@dataclass(frozen=True)
class StateSignature:
    """Represents a signature on a state update."""
    participant: str
    signature: Signature
    timestamp: int
    nonce: int = 0
    
    def verify(self, public_key: PublicKey, message_hash: Hash) -> bool:
        """Verify this signature."""
        return public_key.verify(self.signature, message_hash)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "participant": self.participant,
            "signature": self.signature.to_hex(),
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], message_hash: Hash) -> "StateSignature":
        """Create from dictionary."""
        # Note: This is simplified - in practice you'd need to reconstruct the signature
        return cls(
            participant=data["participant"],
            signature=Signature.from_hex(data["signature"], message_hash.value),
            timestamp=data["timestamp"],
            nonce=data.get("nonce", 0),
        )


@dataclass
class StateTransition:
    """Represents a state transition with validation rules."""
    from_state: Dict[str, Any]
    to_state: Dict[str, Any]
    transition_type: StateUpdateType
    validation_rules: List[str] = field(default_factory=list)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    
    def validate_preconditions(self, current_state: ChannelState) -> bool:
        """Validate preconditions for this transition."""
        for condition, expected_value in self.preconditions.items():
            if condition == "min_balance":
                participant = expected_value.get("participant")
                min_balance = expected_value.get("amount")
                if current_state.balances.get(participant, 0) < min_balance:
                    return False
            elif condition == "sequence_number":
                if current_state.sequence_number != expected_value:
                    return False
            elif condition == "participant_exists":
                participant = expected_value
                if participant not in current_state.participants:
                    return False
        
        return True
    
    def validate_postconditions(self, new_state: ChannelState) -> bool:
        """Validate postconditions for this transition."""
        for condition, expected_value in self.postconditions.items():
            if condition == "balance_conservation":
                # Total balances should be conserved
                total_before = sum(self.from_state.get("balances", {}).values())
                total_after = sum(new_state.balances.values())
                if total_before != total_after:
                    return False
            elif condition == "sequence_increment":
                if new_state.sequence_number != self.from_state.get("sequence_number", 0) + 1:
                    return False
        
        return True
    
    def get_transition_hash(self) -> Hash:
        """Get hash of this state transition."""
        data = {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "transition_type": self.transition_type.value,
            "validation_rules": self.validation_rules,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
        }
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return SHA256Hasher.hash(serialized)


class StateValidator:
    """Validates state transitions and updates."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.validation_rules: Dict[str, callable] = {}
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        self.validation_rules["balance_conservation"] = self._validate_balance_conservation
        self.validation_rules["sequence_number"] = self._validate_sequence_number
        self.validation_rules["participant_authorization"] = self._validate_participant_authorization
        self.validation_rules["signature_verification"] = self._validate_signature_verification
        self.validation_rules["timeout_check"] = self._validate_timeout
        self.validation_rules["deposit_sufficiency"] = self._validate_deposit_sufficiency
    
    def validate_state_update(
        self, 
        update: StateUpdate, 
        current_state: ChannelState,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[str]]:
        """Validate a state update."""
        errors = []
        
        # Check basic structure
        if not self._validate_basic_structure(update, errors):
            return False, errors
        
        # Check custom validation rules first (before signature validation)
        for rule_name, rule_func in self.validation_rules.items():
            if rule_name.startswith("custom_"):
                is_valid, rule_errors = rule_func(update, current_state, public_keys)
                if not is_valid:
                    errors.extend(rule_errors)
                    return False, errors
        
        # Check sequence number
        if not self._validate_sequence_number(update, current_state, errors):
            return False, errors
        
        # Check participants
        if not self._validate_participant_authorization(update, current_state, errors):
            return False, errors
        
        # Check signatures
        if not self._validate_signature_verification(update, public_keys, errors):
            return False, errors
        
        # Check timeout
        if not self._validate_timeout(update, current_state, errors):
            return False, errors
        
        # Check state-specific rules
        if not self._validate_state_specific_rules(update, current_state, errors):
            return False, errors
        
        return len(errors) == 0, errors
    
    def _validate_basic_structure(self, update: StateUpdate, errors: List[str]) -> bool:
        """Validate basic structure of the update."""
        if not update.update_id:
            errors.append("Missing update ID")
            return False
        
        if not update.participants:
            errors.append("No participants specified")
            return False
        
        if not update.state_data:
            errors.append("No state data provided")
            return False
        
        return True
    
    def _validate_sequence_number(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate sequence number."""
        expected_sequence = current_state.sequence_number + 1
        if update.sequence_number != expected_sequence:
            errors.append(f"Invalid sequence number: expected {expected_sequence}, got {update.sequence_number}")
            return False
        
        return True
    
    def _validate_participant_authorization(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate participant authorization."""
        if set(update.participants) != set(current_state.participants):
            errors.append("Participant set mismatch")
            return False
        
        return True
    
    def _validate_signature_verification(
        self, 
        update: StateUpdate, 
        public_keys: Dict[str, PublicKey], 
        errors: List[str]
    ) -> bool:
        """Validate signature verification."""
        if not update.verify_signatures(public_keys):
            errors.append("Invalid signatures")
            return False
        
        if not update.has_required_signatures(self.config):
            errors.append("Insufficient signatures")
            return False
        
        return True
    
    def _validate_timeout(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate timeout constraints."""
        if self.config.enable_timeout_mechanism:
            current_time = int(time.time())
            time_since_last = current_time - current_state.last_update_timestamp
            
            if time_since_last > self.config.state_update_timeout:
                errors.append(f"Update timeout: {time_since_last}s > {self.config.state_update_timeout}s")
                return False
        
        return True
    
    def _validate_state_specific_rules(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate state-specific rules based on update type."""
        if update.update_type == StateUpdateType.TRANSFER:
            return self._validate_transfer_update(update, current_state, errors)
        elif update.update_type == StateUpdateType.CONDITIONAL:
            return self._validate_conditional_update(update, current_state, errors)
        elif update.update_type == StateUpdateType.MULTI_PARTY:
            return self._validate_multi_party_update(update, current_state, errors)
        elif update.update_type == StateUpdateType.CUSTOM:
            return self._validate_custom_update(update, current_state, errors)
        
        return True
    
    def _validate_transfer_update(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate a transfer update."""
        state_data = update.state_data
        
        sender = state_data.get("sender")
        recipient = state_data.get("recipient")
        amount = state_data.get("amount", 0)
        
        if not sender or not recipient:
            errors.append("Missing sender or recipient")
            return False
        
        if sender not in current_state.participants:
            errors.append(f"Sender {sender} not in channel")
            return False
        
        if recipient not in current_state.participants:
            errors.append(f"Recipient {recipient} not in channel")
            return False
        
        if amount <= 0:
            errors.append("Transfer amount must be positive")
            return False
        
        if current_state.balances.get(sender, 0) < amount:
            errors.append(f"Insufficient balance: {current_state.balances.get(sender, 0)} < {amount}")
            return False
        
        return True
    
    def _validate_conditional_update(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate a conditional update."""
        state_data = update.state_data
        
        condition = state_data.get("condition")
        if not condition:
            errors.append("Missing condition")
            return False
        
        # Validate the condition structure
        condition_type = condition.get("type")
        if not condition_type:
            errors.append("Missing condition type")
            return False
        
        # Validate based on condition type
        if condition_type == "time_based":
            target_time = condition.get("target_time")
            if not target_time or target_time <= int(time.time()):
                errors.append("Invalid time-based condition")
                return False
        elif condition_type == "balance_based":
            participant = condition.get("participant")
            min_balance = condition.get("min_balance")
            if not participant or min_balance is None:
                errors.append("Invalid balance-based condition")
                return False
            if participant not in current_state.participants:
                errors.append(f"Condition participant {participant} not in channel")
                return False
        
        # Validate the underlying transfer
        return self._validate_transfer_update(update, current_state, errors)
    
    def _validate_multi_party_update(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate a multi-party update."""
        state_data = update.state_data
        
        transfers = state_data.get("transfers", [])
        if not transfers:
            errors.append("No transfers specified")
            return False
        
        # Validate each transfer
        for i, transfer in enumerate(transfers):
            if not self._validate_single_transfer(transfer, current_state, errors, f"transfer[{i}]"):
                return False
        
        return True
    
    def _validate_single_transfer(
        self, 
        transfer: Dict[str, Any], 
        current_state: ChannelState, 
        errors: List[str],
        context: str = ""
    ) -> bool:
        """Validate a single transfer."""
        sender = transfer.get("sender")
        recipient = transfer.get("recipient")
        amount = transfer.get("amount", 0)
        
        if not sender or not recipient:
            errors.append(f"{context}: Missing sender or recipient")
            return False
        
        if sender not in current_state.participants:
            errors.append(f"{context}: Sender {sender} not in channel")
            return False
        
        if recipient not in current_state.participants:
            errors.append(f"{context}: Recipient {recipient} not in channel")
            return False
        
        if amount <= 0:
            errors.append(f"{context}: Transfer amount must be positive")
            return False
        
        if current_state.balances.get(sender, 0) < amount:
            errors.append(f"{context}: Insufficient balance: {current_state.balances.get(sender, 0)} < {amount}")
            return False
        
        return True
    
    def _validate_custom_update(
        self, 
        update: StateUpdate, 
        current_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate a custom update."""
        state_data = update.state_data
        
        if not isinstance(state_data, dict):
            errors.append("Custom update state data must be a dictionary")
            return False
        
        # Basic validation - custom logic would be application-specific
        return True
    
    def _validate_balance_conservation(
        self, 
        before_state: ChannelState, 
        after_state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate that balances are conserved."""
        total_before = before_state.get_total_balances()
        total_after = after_state.get_total_balances()
        
        if total_before != total_after:
            errors.append(f"Balance conservation violated: {total_before} != {total_after}")
            return False
        
        return True
    
    def _validate_deposit_sufficiency(
        self, 
        state: ChannelState, 
        errors: List[str]
    ) -> bool:
        """Validate that deposits are sufficient."""
        for participant, deposit in state.deposits.items():
            if deposit < self.config.min_deposit:
                errors.append(f"Deposit too small for {participant}: {deposit} < {self.config.min_deposit}")
                return False
        
        return True
    
    def add_validation_rule(self, name: str, rule_func: callable) -> None:
        """Add a custom validation rule."""
        self.validation_rules[name] = rule_func
    
    def remove_validation_rule(self, name: str) -> None:
        """Remove a validation rule."""
        if name in self.validation_rules:
            del self.validation_rules[name]


class OffChainStateManager:
    """Manages off-chain state for state channels."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.validator = StateValidator(config)
        self.state_cache: Dict[ChannelId, ChannelState] = {}
        self.pending_updates: Dict[ChannelId, List[StateUpdate]] = {}
        self.signature_cache: Dict[str, List[StateSignature]] = {}
        
        # State synchronization
        self.sync_callbacks: List[callable] = []
        self.conflict_resolution_strategy = "latest_wins"  # or "consensus_based"
    
    def create_channel_state(
        self,
        channel_id: ChannelId,
        participants: List[str],
        deposits: Dict[str, int],
        initial_balances: Optional[Dict[str, int]] = None
    ) -> ChannelState:
        """Create a new channel state."""
        if initial_balances is None:
            initial_balances = deposits.copy()
        
        state = ChannelState(
            channel_id=channel_id,
            participants=participants,
            deposits=deposits,
            balances=initial_balances,
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.PENDING,
            config=self.config
        )
        
        # Cache the state
        self.state_cache[channel_id] = state
        self.pending_updates[channel_id] = []
        
        return state
    
    def get_channel_state(self, channel_id: ChannelId) -> Optional[ChannelState]:
        """Get the current state of a channel."""
        return self.state_cache.get(channel_id)
    
    def update_channel_state(
        self,
        channel_id: ChannelId,
        update: StateUpdate,
        public_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[str]]:
        """Update a channel state with validation."""
        # Get current state
        current_state = self.get_channel_state(channel_id)
        if not current_state:
            return False, ["Channel not found"]
        
        # Validate the update
        is_valid, errors = self.validator.validate_state_update(
            update, current_state, public_keys
        )
        
        if not is_valid:
            return False, errors
        
        # Apply the update
        try:
            if current_state.apply_state_update(update):
                # Update cache
                self.state_cache[channel_id] = current_state
                
                # Add to pending updates for synchronization
                self.pending_updates[channel_id].append(update)
                
                # Notify sync callbacks
                self._notify_sync_callbacks(channel_id, update)
                
                return True, []
            else:
                return False, ["Failed to apply state update"]
        
        except Exception as e:
            return False, [f"Error applying update: {str(e)}"]
    
    def sign_state_update(
        self,
        update: StateUpdate,
        participant: str,
        private_key: PrivateKey
    ) -> StateSignature:
        """Sign a state update."""
        update_hash = update.get_hash()
        signature = private_key.sign(update_hash)
        
        state_signature = StateSignature(
            participant=participant,
            signature=signature,
            timestamp=int(time.time()),
            nonce=0
        )
        
        # Cache the signature
        signature_key = f"{update.channel_id.value}:{update.update_id}"
        if signature_key not in self.signature_cache:
            self.signature_cache[signature_key] = []
        self.signature_cache[signature_key].append(state_signature)
        
        return state_signature
    
    def collect_signatures(
        self,
        update: StateUpdate,
        participants: List[str],
        private_keys: Dict[str, PrivateKey]
    ) -> Dict[str, StateSignature]:
        """Collect signatures from multiple participants."""
        signatures = {}
        
        for participant in participants:
            if participant in private_keys:
                signature = self.sign_state_update(update, participant, private_keys[participant])
                signatures[participant] = signature
        
        return signatures
    
    def verify_state_consistency(self, channel_id: ChannelId) -> Tuple[bool, List[str]]:
        """Verify state consistency for a channel."""
        state = self.get_channel_state(channel_id)
        if not state:
            return False, ["Channel not found"]
        
        errors = []
        
        # Check balance conservation
        if not state.validate_balances():
            errors.append("Balance conservation violated")
        
        # Check sequence number consistency
        if state.sequence_number != len(state.state_history):
            errors.append("Sequence number inconsistent with history")
        
        # Check deposit sufficiency
        for participant, deposit in state.deposits.items():
            if deposit < self.config.min_deposit:
                errors.append(f"Deposit too small for {participant}")
        
        # Check state history integrity
        for i, update in enumerate(state.state_history):
            if update.sequence_number != i + 1:
                errors.append(f"Invalid sequence number in history at index {i}")
        
        return len(errors) == 0, errors
    
    def resolve_state_conflicts(
        self,
        channel_id: ChannelId,
        conflicting_states: List[ChannelState]
    ) -> Optional[ChannelState]:
        """Resolve conflicts between multiple channel states."""
        if not conflicting_states:
            return None
        
        if self.conflict_resolution_strategy == "latest_wins":
            # Choose the state with the highest sequence number
            return max(conflicting_states, key=lambda s: s.sequence_number)
        
        elif self.conflict_resolution_strategy == "consensus_based":
            # Choose the state that appears in the most participants' views
            state_counts = {}
            for state in conflicting_states:
                state_hash = state.get_latest_state_hash().to_hex()
                state_counts[state_hash] = state_counts.get(state_hash, 0) + 1
            
            # Find the most common state
            most_common_hash = max(state_counts, key=state_counts.get)
            for state in conflicting_states:
                if state.get_latest_state_hash().to_hex() == most_common_hash:
                    return state
        
        # Default: return the first state
        return conflicting_states[0]
    
    def synchronize_states(
        self,
        channel_id: ChannelId,
        remote_states: List[ChannelState]
    ) -> bool:
        """Synchronize local state with remote states."""
        local_state = self.get_channel_state(channel_id)
        if not local_state:
            return False
        
        # Check for conflicts
        all_states = [local_state] + remote_states
        # Remove duplicates by comparing channel_id and sequence_number
        unique_states = []
        seen = set()
        for state in all_states:
            key = (state.channel_id.value, state.sequence_number)
            if key not in seen:
                seen.add(key)
                unique_states.append(state)
        
        if len(unique_states) > 1:
            # Conflict detected, resolve it
            resolved_state = self.resolve_state_conflicts(channel_id, unique_states)
            if resolved_state:
                self.state_cache[channel_id] = resolved_state
                return True
            else:
                return False
        
        return True
    
    def get_pending_updates(self, channel_id: ChannelId) -> List[StateUpdate]:
        """Get pending updates for a channel."""
        return self.pending_updates.get(channel_id, [])
    
    def clear_pending_updates(self, channel_id: ChannelId) -> None:
        """Clear pending updates for a channel."""
        if channel_id in self.pending_updates:
            self.pending_updates[channel_id] = []
    
    def add_sync_callback(self, callback: callable) -> None:
        """Add a callback for state synchronization events."""
        self.sync_callbacks.append(callback)
    
    def remove_sync_callback(self, callback: callable) -> None:
        """Remove a sync callback."""
        if callback in self.sync_callbacks:
            self.sync_callbacks.remove(callback)
    
    def _notify_sync_callbacks(self, channel_id: ChannelId, update: StateUpdate) -> None:
        """Notify all sync callbacks of a state update."""
        for callback in self.sync_callbacks:
            try:
                callback(channel_id, update)
            except Exception as e:
                print(f"Error in sync callback: {e}")
    
    def export_state(self, channel_id: ChannelId) -> Optional[Dict[str, Any]]:
        """Export channel state for backup or transfer."""
        state = self.get_channel_state(channel_id)
        if not state:
            return None
        
        return {
            "channel_id": channel_id.value,
            "state": state.to_dict(),
            "pending_updates": [update.to_dict() for update in self.pending_updates.get(channel_id, [])],
            "export_timestamp": int(time.time()),
        }
    
    def import_state(self, state_data: Dict[str, Any]) -> bool:
        """Import channel state from backup."""
        try:
            channel_id = ChannelId(state_data["channel_id"])
            
            # Reconstruct state
            state_dict = state_data["state"]
            state = ChannelState(
                channel_id=channel_id,
                participants=state_dict["participants"],
                deposits=state_dict["deposits"],
                balances=state_dict["balances"],
                sequence_number=state_dict["sequence_number"],
                last_update_timestamp=state_dict["last_update_timestamp"],
                status=ChannelStatus(state_dict["status"]),
                config=ChannelConfig.from_dict(state_dict["config"])
            )
            
            # Restore state history
            for update_dict in state_dict.get("state_history", []):
                update = StateUpdate.from_dict(update_dict)
                state.state_history.append(update)
            
            # Cache the state
            self.state_cache[channel_id] = state
            
            # Restore pending updates
            pending_updates = []
            for update_dict in state_data.get("pending_updates", []):
                update = StateUpdate.from_dict(update_dict)
                pending_updates.append(update)
            self.pending_updates[channel_id] = pending_updates
            
            return True
        
        except Exception as e:
            print(f"Error importing state: {e}")
            return False
    
    def cleanup_channel(self, channel_id: ChannelId) -> None:
        """Clean up resources for a closed channel."""
        # Keep the final state for historical purposes, just clean up pending updates
        if channel_id in self.pending_updates:
            del self.pending_updates[channel_id]
        
        # Clean up signature cache
        keys_to_remove = [key for key in self.signature_cache.keys() 
                         if key.startswith(channel_id.value)]
        for key in keys_to_remove:
            del self.signature_cache[key]
