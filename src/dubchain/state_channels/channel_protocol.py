"""
Core State Channel Protocol Implementation

This module defines the fundamental state channel protocol including:
- Channel lifecycle management (open, update, close)
- State update mechanisms with cryptographic verification
- Multi-party coordination and consensus
- Event handling and error management
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature
from ..errors.exceptions import ValidationError


class ChannelStatus(Enum):
    """State channel status enumeration."""
    PENDING = "pending"           # Channel creation pending
    OPEN = "open"                # Channel is active and accepting updates
    CLOSING = "closing"          # Channel is in the process of closing
    CLOSED = "closed"            # Channel is closed cooperatively
    DISPUTED = "disputed"        # Channel is in dispute resolution
    EXPIRED = "expired"          # Channel has expired due to timeout
    FROZEN = "frozen"            # Channel is frozen due to security issues


class ChannelCloseReason(Enum):
    """Reason for channel closure."""
    COOPERATIVE = "cooperative"   # All parties agreed to close
    TIMEOUT = "timeout"          # Channel expired due to timeout
    DISPUTE = "dispute"          # Dispute resolution triggered
    FRAUD = "fraud"              # Fraud detected
    INSUFFICIENT_FUNDS = "insufficient_funds"  # Not enough funds
    SECURITY_VIOLATION = "security_violation"  # Security breach detected


class StateUpdateType(Enum):
    """Type of state update."""
    TRANSFER = "transfer"         # Simple token transfer
    CONDITIONAL = "conditional"   # Conditional payment
    MULTI_PARTY = "multi_party"   # Multi-party operation
    CUSTOM = "custom"            # Custom application logic


class ChannelEvent(Enum):
    """Channel events for monitoring."""
    CREATED = "created"
    OPENED = "opened"
    STATE_UPDATED = "state_updated"
    DISPUTE_INITIATED = "dispute_initiated"
    DISPUTE_RESOLVED = "dispute_resolved"
    CLOSING = "closing"
    CLOSED = "closed"
    EXPIRED = "expired"
    FROZEN = "frozen"


@dataclass(frozen=True)
class ChannelId:
    """Unique identifier for a state channel."""
    value: str
    
    @classmethod
    def generate(cls) -> "ChannelId":
        """Generate a new unique channel ID."""
        return cls(str(uuid.uuid4()))
    
    def to_hex(self) -> str:
        """Convert to hexadecimal string."""
        return self.value
    
    def __str__(self) -> str:
        return self.value


@dataclass
class ChannelConfig:
    """Configuration for a state channel."""
    # Channel parameters
    timeout_blocks: int = 1000      # Channel timeout in blocks
    dispute_period_blocks: int = 100  # Dispute period in blocks
    max_participants: int = 10      # Maximum number of participants
    min_deposit: int = 1000         # Minimum deposit per participant
    
    # Security parameters
    require_all_signatures: bool = True  # Require all participants to sign updates
    enable_fraud_proofs: bool = True     # Enable fraud proof mechanisms
    enable_timeout_mechanism: bool = True  # Enable timeout-based closure
    
    # Performance parameters
    max_state_updates: int = 10000  # Maximum number of state updates
    state_update_timeout: int = 300  # Timeout for state updates (seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeout_blocks": self.timeout_blocks,
            "dispute_period_blocks": self.dispute_period_blocks,
            "max_participants": self.max_participants,
            "min_deposit": self.min_deposit,
            "require_all_signatures": self.require_all_signatures,
            "enable_fraud_proofs": self.enable_fraud_proofs,
            "enable_timeout_mechanism": self.enable_timeout_mechanism,
            "max_state_updates": self.max_state_updates,
            "state_update_timeout": self.state_update_timeout,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StateUpdate:
    """Represents a state update in the channel."""
    update_id: str
    channel_id: ChannelId
    sequence_number: int
    update_type: StateUpdateType
    participants: List[str]  # List of participant addresses
    state_data: Dict[str, Any]
    timestamp: int
    nonce: int = 0
    
    # Cryptographic proof
    signatures: Dict[str, Signature] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate state update after initialization."""
        if self.sequence_number < 0:
            raise ValueError("Sequence number must be non-negative")
        if not self.participants:
            raise ValueError("At least one participant required")
        if not self.state_data:
            raise ValueError("State data cannot be empty")
    
    def add_signature(self, participant: str, signature: Signature) -> None:
        """Add a signature from a participant."""
        if participant not in self.participants:
            raise ValueError(f"Participant {participant} not in channel")
        self.signatures[participant] = signature
    
    def get_hash(self) -> Hash:
        """Get the hash of this state update."""
        data = {
            "update_id": self.update_id,
            "channel_id": self.channel_id.value,
            "sequence_number": self.sequence_number,
            "update_type": self.update_type.value,
            "participants": sorted(self.participants),
            "state_data": self.state_data,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return SHA256Hasher.hash(serialized)
    
    def verify_signatures(self, public_keys: Dict[str, PublicKey]) -> bool:
        """Verify all signatures on this state update."""
        if not self.signatures:
            return False
        
        update_hash = self.get_hash()
        
        for participant, signature in self.signatures.items():
            if participant not in public_keys:
                return False
            
            public_key = public_keys[participant]
            if not public_key.verify(signature, update_hash):
                return False
        
        return True
    
    def has_required_signatures(self, config: ChannelConfig) -> bool:
        """Check if update has required signatures based on config."""
        if config.require_all_signatures:
            return len(self.signatures) == len(self.participants)
        else:
            # Require majority signatures
            required = (len(self.participants) // 2) + 1
            return len(self.signatures) >= required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "update_id": self.update_id,
            "channel_id": self.channel_id.value,
            "sequence_number": self.sequence_number,
            "update_type": self.update_type.value,
            "participants": self.participants,
            "state_data": self.state_data,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signatures": {
                participant: sig.to_hex() 
                for participant, sig in self.signatures.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateUpdate":
        """Create from dictionary."""
        # Note: Signatures would need to be reconstructed from hex
        # This is a simplified version for serialization
        return cls(
            update_id=data["update_id"],
            channel_id=ChannelId(data["channel_id"]),
            sequence_number=data["sequence_number"],
            update_type=StateUpdateType(data["update_type"]),
            participants=data["participants"],
            state_data=data["state_data"],
            timestamp=data["timestamp"],
            nonce=data.get("nonce", 0),
        )


@dataclass
class ChannelState:
    """Current state of a state channel."""
    channel_id: ChannelId
    participants: List[str]
    deposits: Dict[str, int]  # Participant -> deposit amount
    balances: Dict[str, int]  # Participant -> current balance
    sequence_number: int
    last_update_timestamp: int
    status: ChannelStatus
    config: ChannelConfig
    
    # State history for dispute resolution
    state_history: List[StateUpdate] = field(default_factory=list)
    
    # Metadata
    created_at: int = field(default_factory=lambda: int(time.time()))
    opened_at: Optional[int] = None
    closed_at: Optional[int] = None
    close_reason: Optional[ChannelCloseReason] = None
    
    def __post_init__(self):
        """Validate channel state after initialization."""
        if not self.participants:
            raise ValueError("At least one participant required")
        if len(self.participants) > self.config.max_participants:
            raise ValueError(f"Too many participants: {len(self.participants)} > {self.config.max_participants}")
        
        # Validate deposits
        for participant in self.participants:
            if participant not in self.deposits:
                raise ValueError(f"Missing deposit for participant {participant}")
            if self.deposits[participant] < self.config.min_deposit:
                raise ValueError(f"Deposit too small for {participant}: {self.deposits[participant]} < {self.config.min_deposit}")
    
    def get_total_deposits(self) -> int:
        """Get total deposits in the channel."""
        return sum(self.deposits.values())
    
    def get_total_balances(self) -> int:
        """Get total balances in the channel."""
        return sum(self.balances.values())
    
    def validate_balances(self) -> bool:
        """Validate that balances are consistent with deposits."""
        return self.get_total_balances() == self.get_total_deposits()
    
    def can_update_state(self, update: StateUpdate) -> bool:
        """Check if a state update is valid for this channel state."""
        # Check sequence number
        if update.sequence_number != self.sequence_number + 1:
            return False
        
        # Check participants match
        if set(update.participants) != set(self.participants):
            return False
        
        # Check channel is open
        if self.status != ChannelStatus.OPEN:
            return False
        
        # Check timeout
        if self.config.enable_timeout_mechanism:
            current_time = int(time.time())
            if current_time - self.last_update_timestamp > self.config.state_update_timeout:
                return False
        
        return True
    
    def apply_state_update(self, update: StateUpdate) -> bool:
        """Apply a state update to the channel state."""
        if not self.can_update_state(update):
            return False
        
        # Update sequence number
        self.sequence_number = update.sequence_number
        self.last_update_timestamp = update.timestamp
        
        # Apply state changes based on update type
        if update.update_type == StateUpdateType.TRANSFER:
            self._apply_transfer_update(update)
        elif update.update_type == StateUpdateType.CONDITIONAL:
            self._apply_conditional_update(update)
        elif update.update_type == StateUpdateType.MULTI_PARTY:
            self._apply_multi_party_update(update)
        elif update.update_type == StateUpdateType.CUSTOM:
            self._apply_custom_update(update)
        
        # Add to history
        self.state_history.append(update)
        
        return True
    
    def _apply_transfer_update(self, update: StateUpdate) -> None:
        """Apply a transfer state update."""
        state_data = update.state_data
        
        # Extract transfer information
        sender = state_data.get("sender")
        recipient = state_data.get("recipient")
        amount = state_data.get("amount", 0)
        
        if sender not in self.balances or recipient not in self.balances:
            raise ValueError("Invalid participants in transfer")
        
        if self.balances[sender] < amount:
            raise ValueError("Insufficient balance for transfer")
        
        # Apply transfer
        self.balances[sender] -= amount
        self.balances[recipient] += amount
    
    def _apply_conditional_update(self, update: StateUpdate) -> None:
        """Apply a conditional state update."""
        state_data = update.state_data
        
        # Check condition
        condition = state_data.get("condition")
        if not self._evaluate_condition(condition):
            return  # Don't apply update if condition not met
        
        # Apply the update
        self._apply_transfer_update(update)
    
    def _apply_multi_party_update(self, update: StateUpdate) -> None:
        """Apply a multi-party state update."""
        state_data = update.state_data
        
        # Apply multiple transfers atomically
        transfers = state_data.get("transfers", [])
        for transfer in transfers:
            sender = transfer["sender"]
            recipient = transfer["recipient"]
            amount = transfer["amount"]
            
            if self.balances[sender] < amount:
                raise ValueError(f"Insufficient balance for {sender}")
            
            self.balances[sender] -= amount
            self.balances[recipient] += amount
    
    def _apply_custom_update(self, update: StateUpdate) -> None:
        """Apply a custom state update."""
        # This would be implemented based on specific application logic
        # For now, we'll just validate the update
        state_data = update.state_data
        if not isinstance(state_data, dict):
            raise ValueError("Custom update must have dict state data")
    
    def _evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate a condition for conditional updates."""
        # Simplified condition evaluation
        # In practice, this would be more sophisticated
        condition_type = condition.get("type")
        
        if condition_type == "time_based":
            target_time = condition.get("target_time")
            return int(time.time()) >= target_time
        elif condition_type == "balance_based":
            participant = condition.get("participant")
            min_balance = condition.get("min_balance")
            return self.balances.get(participant, 0) >= min_balance
        elif condition_type == "external":
            # External condition would require oracle or external data
            return condition.get("result", False)
        
        return False
    
    def get_latest_state_hash(self) -> Hash:
        """Get hash of the latest state."""
        state_data = {
            "channel_id": self.channel_id.value,
            "participants": sorted(self.participants),
            "balances": self.balances,
            "sequence_number": self.sequence_number,
            "timestamp": self.last_update_timestamp,
        }
        serialized = json.dumps(state_data, sort_keys=True).encode('utf-8')
        return SHA256Hasher.hash(serialized)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id.value,
            "participants": self.participants,
            "deposits": self.deposits,
            "balances": self.balances,
            "sequence_number": self.sequence_number,
            "last_update_timestamp": self.last_update_timestamp,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "state_history": [update.to_dict() for update in self.state_history],
            "created_at": self.created_at,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason.value if self.close_reason else None,
        }


class ChannelError(Exception):
    """Base exception for channel-related errors."""
    pass


class InvalidStateUpdateError(ChannelError):
    """Raised when a state update is invalid."""
    pass


class InsufficientSignaturesError(ChannelError):
    """Raised when insufficient signatures are provided."""
    pass


class ChannelTimeoutError(ChannelError):
    """Raised when channel operations timeout."""
    pass


class ChannelSecurityError(ChannelError):
    """Raised when security violations are detected."""
    pass


class StateChannel:
    """Main state channel implementation."""
    
    def __init__(self, channel_id: ChannelId, config: ChannelConfig):
        self.channel_id = channel_id
        self.config = config
        self.state: Optional[ChannelState] = None
        self.participant_keys: Dict[str, PublicKey] = {}
        self.event_handlers: Dict[ChannelEvent, List[callable]] = {}
        
        # Initialize event handlers
        for event in ChannelEvent:
            self.event_handlers[event] = []
    
    def create_channel(
        self, 
        participants: List[str], 
        deposits: Dict[str, int],
        participant_keys: Dict[str, PublicKey]
    ) -> bool:
        """Create a new state channel."""
        try:
            # Validate inputs
            if len(participants) < 2:
                return False
            
            if len(participants) > self.config.max_participants:
                return False
            
            # Validate deposits
            for participant in participants:
                if participant not in deposits:
                    return False
                if deposits[participant] < self.config.min_deposit:
                    return False
            
            # Initialize balances equal to deposits
            balances = deposits.copy()
            
            # Create channel state
            self.state = ChannelState(
                channel_id=self.channel_id,
                participants=participants,
                deposits=deposits,
                balances=balances,
                sequence_number=0,
                last_update_timestamp=int(time.time()),
                status=ChannelStatus.PENDING,
                config=self.config
            )
            
            # Store participant keys
            self.participant_keys = participant_keys.copy()
            
            # Emit event
            self._emit_event(ChannelEvent.CREATED)
            
            return True
            
        except Exception as e:
            return False
    
    def open_channel(self) -> bool:
        """Open the channel for state updates."""
        if not self.state:
            raise ChannelError("Channel not created")
        
        if self.state.status != ChannelStatus.PENDING:
            raise ChannelError(f"Channel not in pending state: {self.state.status}")
        
        # Validate initial state
        if not self.state.validate_balances():
            raise ChannelError("Invalid initial balances")
        
        # Update status
        self.state.status = ChannelStatus.OPEN
        self.state.opened_at = int(time.time())
        
        # Emit event
        self._emit_event(ChannelEvent.OPENED)
        
        return True
    
    def update_state(self, update: StateUpdate) -> bool:
        """Update the channel state."""
        if not self.state:
            raise ChannelError("Channel not created")
        
        if self.state.status != ChannelStatus.OPEN:
            raise ChannelError(f"Channel not open: {self.state.status}")
        
        # Validate update
        if not self._validate_state_update(update):
            raise InvalidStateUpdateError("Invalid state update")
        
        # Apply update
        if not self.state.apply_state_update(update):
            raise InvalidStateUpdateError("Failed to apply state update")
        
        # Emit event
        self._emit_event(ChannelEvent.STATE_UPDATED)
        
        return True
    
    def _validate_state_update(self, update: StateUpdate) -> bool:
        """Validate a state update."""
        # Check channel ID matches
        if update.channel_id != self.channel_id:
            return False
        
        # Check participants match
        if set(update.participants) != set(self.state.participants):
            return False
        
        # Check sequence number
        if update.sequence_number != self.state.sequence_number + 1:
            return False
        
        # Verify signatures
        if not update.verify_signatures(self.participant_keys):
            return False
        
        # Check required signatures
        if not update.has_required_signatures(self.config):
            return False
        
        return True
    
    def initiate_dispute(self, evidence: Dict[str, Any]) -> bool:
        """Initiate a dispute resolution process."""
        if not self.state:
            raise ChannelError("Channel not created")
        
        if self.state.status not in [ChannelStatus.OPEN, ChannelStatus.CLOSING]:
            raise ChannelError(f"Cannot dispute channel in state: {self.state.status}")
        
        # Update status
        self.state.status = ChannelStatus.DISPUTED
        
        # Emit event
        self._emit_event(ChannelEvent.DISPUTE_INITIATED)
        
        return True
    
    def close_channel(self, reason: ChannelCloseReason = ChannelCloseReason.COOPERATIVE) -> bool:
        """Close the channel."""
        if not self.state:
            raise ChannelError("Channel not created")
        
        if self.state.status in [ChannelStatus.CLOSED, ChannelStatus.EXPIRED]:
            return True  # Already closed
        
        # Update status
        self.state.status = ChannelStatus.CLOSED
        self.state.closed_at = int(time.time())
        self.state.close_reason = reason
        
        # Emit event
        self._emit_event(ChannelEvent.CLOSING)
        self._emit_event(ChannelEvent.CLOSED)
        
        return True
    
    def expire_channel(self) -> bool:
        """Expire the channel due to timeout."""
        if not self.state:
            raise ChannelError("Channel not created")
        
        # Update status
        self.state.status = ChannelStatus.EXPIRED
        self.state.closed_at = int(time.time())
        self.state.close_reason = ChannelCloseReason.TIMEOUT
        
        # Emit event
        self._emit_event(ChannelEvent.EXPIRED)
        
        return True
    
    def freeze_channel(self, reason: str = "Security violation") -> bool:
        """Freeze the channel due to security issues."""
        if not self.state:
            raise ChannelError("Channel not created")
        
        # Update status
        self.state.status = ChannelStatus.FROZEN
        self.state.close_reason = ChannelCloseReason.SECURITY_VIOLATION
        
        # Emit event
        self._emit_event(ChannelEvent.FROZEN)
        
        return True
    
    def add_event_handler(self, event: ChannelEvent, handler: callable) -> None:
        """Add an event handler."""
        self.event_handlers[event].append(handler)
    
    def remove_event_handler(self, event: ChannelEvent, handler: callable) -> None:
        """Remove an event handler."""
        if handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)
    
    def _emit_event(self, event: ChannelEvent) -> None:
        """Emit an event to all registered handlers."""
        for handler in self.event_handlers[event]:
            try:
                handler(event, self.state)
            except Exception as e:
                # Log error but don't fail the operation
                print(f"Error in event handler for {event}: {e}")
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get comprehensive channel information."""
        if not self.state:
            return {"error": "Channel not created"}
        
        return {
            "channel_id": self.channel_id.value,
            "status": self.state.status.value,
            "participants": self.state.participants,
            "total_deposits": self.state.get_total_deposits(),
            "total_balances": self.state.get_total_balances(),
            "sequence_number": self.state.sequence_number,
            "created_at": self.state.created_at,
            "opened_at": self.state.opened_at,
            "closed_at": self.state.closed_at,
            "close_reason": self.state.close_reason.value if self.state.close_reason else None,
            "config": self.state.config.to_dict(),
        }
    
    def get_latest_state(self) -> Optional[ChannelState]:
        """Get the latest channel state."""
        return self.state
    
    def is_active(self) -> bool:
        """Check if the channel is active."""
        if not self.state:
            return False
        return self.state.status == ChannelStatus.OPEN
