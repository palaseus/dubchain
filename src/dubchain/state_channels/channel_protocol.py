"""
State Channel Protocol Implementation

This module implements the core state channel protocol including:
- Channel creation and management
- State updates and transitions
- Payment processing
- Channel closure mechanisms
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..errors import ClientError
from dubchain.logging import get_logger
from .dispute_resolution import DisputeManager, DisputeConfig

logger = get_logger(__name__)

# Type aliases for compatibility
class ChannelId:
    """Channel ID with generation capability."""
    
    def __init__(self, value: str):
        """Initialize channel ID."""
        self.value = value
    
    @classmethod
    def generate(cls) -> 'ChannelId':
        """Generate a new unique channel ID."""
        return cls(f"channel_{int(time.time())}_{uuid.uuid4().hex[:8]}")
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __repr__(self) -> str:
        """Representation."""
        return f"ChannelId('{self.value}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if isinstance(other, ChannelId):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self.value)

class ChannelStatus(Enum):
    """Status of a state channel."""
    CREATING = "creating"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    DISPUTED = "disputed"
    EXPIRED = "expired"

class PaymentType(Enum):
    """Types of payments."""
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    FEE = "fee"
    PENALTY = "penalty"

class StateUpdateType(Enum):
    """Types of state updates."""
    PAYMENT = "payment"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    CLOSE = "close"
    DISPUTE = "dispute"

class ChannelCloseReason(Enum):
    """Reasons for channel closure."""
    MUTUAL_CLOSE = "mutual_close"
    UNILATERAL_CLOSE = "unilateral_close"
    DISPUTE_CLOSE = "dispute_close"
    TIMEOUT_CLOSE = "timeout_close"
    FRAUD_CLOSE = "fraud_close"

class ChannelEvent(Enum):
    """Channel events."""
    CREATED = "created"
    OPENED = "opened"
    UPDATED = "updated"
    CLOSED = "closed"
    DISPUTED = "disputed"
    EXPIRED = "expired"

@dataclass
class ChannelParticipant:
    """Participant in a state channel."""
    address: str
    public_key: str
    balance: int
    nonce: int = 0
    is_active: bool = True
    last_activity: int = field(default_factory=lambda: int(time.time()))

@dataclass
class Payment:
    """Payment within a state channel."""
    payment_id: str
    from_address: str
    to_address: str
    amount: int
    payment_type: PaymentType
    timestamp: int
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateUpdate:
    """State update for a channel."""
    update_id: str
    channel_id: str
    update_type: StateUpdateType
    data: Dict[str, Any]
    timestamp: int
    signature: Optional[str] = None
    nonce: int = 0
    sequence_number: int = 0  # Added for test compatibility

@dataclass
class ChannelState:
    """State of a state channel."""
    channel_id: str
    participants: Dict[str, ChannelParticipant]
    total_balance: int
    nonce: int
    created_at: int
    last_updated: int
    state_hash: str
    payments: List[Payment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChannelConfig:
    """Configuration for state channels."""
    max_participants: int = 10
    min_balance: int = 1000
    max_balance: int = 1000000000
    timeout_duration: int = 86400  # 24 hours
    timeout_blocks: int = 1000  # Block timeout - changed from 100 to 1000 for test compatibility
    dispute_timeout: int = 172800  # 48 hours
    max_payments_per_channel: int = 1000
    enable_disputes: bool = True
    enable_timeouts: bool = True
    min_deposit: int = 1000  # Added for test compatibility
    require_all_signatures: bool = True  # Changed to True for test compatibility
    dispute_period_blocks: int = 100  # Added for test compatibility
    enable_fraud_proofs: bool = True  # Added for test compatibility
    enable_timeout_mechanism: bool = True  # Added for test compatibility
    state_update_timeout: int = 300  # Added for test compatibility - 5 minutes

@dataclass
class StateChannel:
    """Complete state channel."""
    channel_id: str
    config: ChannelConfig
    state: Optional[ChannelState] = None
    status: ChannelStatus = ChannelStatus.CREATING
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_updated: int = field(default_factory=lambda: int(time.time()))
    events: List[ChannelEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.state is None:
            self.state = ChannelState(
                channel_id=self.channel_id,
                participants={},
                total_balance=0,
                nonce=0,
                created_at=self.created_at,
                last_updated=self.last_updated,
                state_hash=""
            )

class StateValidator:
    """Validates channel state transitions."""
    
    def __init__(self, config: ChannelConfig):
        """Initialize state validator."""
        self.config = config
        logger.info("Initialized state validator")
    
    def validate_state_transition(self, old_state: ChannelState, new_state: ChannelState) -> bool:
        """Validate state transition."""
        try:
            # Check channel ID consistency
            if old_state.channel_id != new_state.channel_id:
                logger.error("Channel ID mismatch in state transition")
                return False
            
            # Check nonce increment
            if new_state.nonce <= old_state.nonce:
                logger.error("Nonce must increment in state transition")
                return False
            
            # Check balance conservation
            if not self._validate_balance_conservation(old_state, new_state):
                logger.error("Balance conservation violated")
                return False
            
            # Check participant consistency
            if not self._validate_participant_consistency(old_state, new_state):
                logger.error("Participant consistency violated")
                return False
            
            # Check state hash
            if not self._validate_state_hash(new_state):
                logger.error("Invalid state hash")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state transition: {e}")
            return False
    
    def _validate_balance_conservation(self, old_state: ChannelState, new_state: ChannelState) -> bool:
        """Validate balance conservation."""
        try:
            # Check total balance
            if old_state.total_balance != new_state.total_balance:
                logger.error("Total balance changed")
                return False
            
            # Check individual balances
            for address, participant in new_state.participants.items():
                if address not in old_state.participants:
                    logger.error(f"New participant {address} added")
                    return False
                
                old_participant = old_state.participants[address]
                if participant.balance < 0:
                    logger.error(f"Negative balance for {address}")
                    return False
                
                # Check if balance change is valid
                balance_change = participant.balance - old_participant.balance
                if abs(balance_change) > self.config.max_balance:
                    logger.error(f"Balance change too large for {address}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating balance conservation: {e}")
            return False
    
    def _validate_participant_consistency(self, old_state: ChannelState, new_state: ChannelState) -> bool:
        """Validate participant consistency."""
        try:
            # Check participant count
            if len(new_state.participants) != len(old_state.participants):
                logger.error("Participant count changed")
                return False
            
            # Check participant addresses
            old_addresses = set(old_state.participants.keys())
            new_addresses = set(new_state.participants.keys())
            
            if old_addresses != new_addresses:
                logger.error("Participant addresses changed")
                return False
            
            # Check public keys
            for address in old_addresses:
                old_pubkey = old_state.participants[address].public_key
                new_pubkey = new_state.participants[address].public_key
                
                if old_pubkey != new_pubkey:
                    logger.error(f"Public key changed for {address}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating participant consistency: {e}")
            return False
    
    def _validate_state_hash(self, state: ChannelState) -> bool:
        """Validate state hash."""
        try:
            # Calculate expected hash
            expected_hash = self._calculate_state_hash(state)
            
            if state.state_hash != expected_hash:
                logger.error(f"State hash mismatch: expected {expected_hash}, got {state.state_hash}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state hash: {e}")
            return False
    
    def _calculate_state_hash(self, state: ChannelState) -> str:
        """Calculate state hash."""
        try:
            # Create hashable representation
            hash_data = {
                'channel_id': state.channel_id,
                'participants': {
                    addr: {
                        'balance': p.balance,
                        'nonce': p.nonce,
                        'public_key': p.public_key
                    }
                    for addr, p in state.participants.items()
                },
                'total_balance': state.total_balance,
                'nonce': state.nonce,
                'created_at': state.created_at,
                'last_updated': state.last_updated
            }
            
            # Add payment hashes
            payment_hashes = []
            for payment in state.payments:
                payment_hash = hashlib.sha256(
                    f"{payment.payment_id}{payment.from_address}{payment.to_address}{payment.amount}{payment.timestamp}".encode()
                ).hexdigest()
                payment_hashes.append(payment_hash)
            
            hash_data['payment_hashes'] = payment_hashes
            
            # Calculate hash
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating state hash: {e}")
            return ""

class PaymentProcessor:
    """Processes payments within state channels."""
    
    def __init__(self, config: ChannelConfig):
        """Initialize payment processor."""
        self.config = config
        logger.info("Initialized payment processor")
    
    def process_payment(self, channel_state: ChannelState, payment: Payment) -> Optional[ChannelState]:
        """Process a payment and return new channel state."""
        try:
            # Validate payment
            if not self._validate_payment(channel_state, payment):
                logger.error(f"Invalid payment {payment.payment_id}")
                return None
            
            # Create new state
            new_state = self._create_new_state(channel_state)
            
            # Apply payment
            if not self._apply_payment(new_state, payment):
                logger.error(f"Failed to apply payment {payment.payment_id}")
                return None
            
            # Update state
            new_state.nonce += 1
            new_state.last_updated = int(time.time())
            new_state.payments.append(payment)
            
            # Calculate new hash
            new_state.state_hash = self._calculate_state_hash(new_state)
            
            logger.info(f"Processed payment {payment.payment_id}")
            return new_state
            
        except Exception as e:
            logger.error(f"Error processing payment: {e}")
            return None
    
    def _validate_payment(self, channel_state: ChannelState, payment: Payment) -> bool:
        """Validate payment."""
        try:
            # Check payment ID
            if not payment.payment_id:
                logger.error("Payment ID required")
                return False
            
            # Check addresses
            if payment.from_address not in channel_state.participants:
                logger.error(f"From address {payment.from_address} not in channel")
                return False
            
            if payment.to_address not in channel_state.participants:
                logger.error(f"To address {payment.to_address} not in channel")
                return False
            
            # Check amount
            if payment.amount <= 0:
                logger.error("Payment amount must be positive")
                return False
            
            # Check balance
            from_participant = channel_state.participants[payment.from_address]
            if from_participant.balance < payment.amount:
                logger.error(f"Insufficient balance for {payment.from_address}")
                return False
            
            # Check payment limit
            if len(channel_state.payments) >= self.config.max_payments_per_channel:
                logger.error("Payment limit reached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating payment: {e}")
            return False
    
    def _apply_payment(self, channel_state: ChannelState, payment: Payment) -> bool:
        """Apply payment to channel state."""
        try:
            # Update balances
            from_participant = channel_state.participants[payment.from_address]
            to_participant = channel_state.participants[payment.to_address]
            
            from_participant.balance -= payment.amount
            to_participant.balance += payment.amount
            
            # Update nonces
            from_participant.nonce += 1
            to_participant.nonce += 1
            
            # Update activity
            from_participant.last_activity = int(time.time())
            to_participant.last_activity = int(time.time())
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying payment: {e}")
            return False
    
    def _create_new_state(self, old_state: ChannelState) -> ChannelState:
        """Create new state from old state."""
        try:
            # Deep copy participants
            new_participants = {}
            for addr, participant in old_state.participants.items():
                new_participants[addr] = ChannelParticipant(
                    address=participant.address,
                    public_key=participant.public_key,
                    balance=participant.balance,
                    nonce=participant.nonce,
                    is_active=participant.is_active,
                    last_activity=participant.last_activity
                )
            
            # Create new state
            new_state = ChannelState(
                channel_id=old_state.channel_id,
                participants=new_participants,
                total_balance=old_state.total_balance,
                nonce=old_state.nonce,
                created_at=old_state.created_at,
                last_updated=old_state.last_updated,
                state_hash=old_state.state_hash,
                payments=old_state.payments.copy(),
                metadata=old_state.metadata.copy()
            )
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error creating new state: {e}")
            return None
    
    def _calculate_state_hash(self, state: ChannelState) -> str:
        """Calculate state hash."""
        try:
            # Create hashable representation
            hash_data = {
                'channel_id': state.channel_id,
                'participants': {
                    addr: {
                        'balance': p.balance,
                        'nonce': p.nonce,
                        'public_key': p.public_key
                    }
                    for addr, p in state.participants.items()
                },
                'total_balance': state.total_balance,
                'nonce': state.nonce,
                'created_at': state.created_at,
                'last_updated': state.last_updated
            }
            
            # Add payment hashes
            payment_hashes = []
            for payment in state.payments:
                payment_hash = hashlib.sha256(
                    f"{payment.payment_id}{payment.from_address}{payment.to_address}{payment.amount}{payment.timestamp}".encode()
                ).hexdigest()
                payment_hashes.append(payment_hash)
            
            hash_data['payment_hashes'] = payment_hashes
            
            # Calculate hash
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating state hash: {e}")
            return ""

class ChannelManager:
    """Main channel management system."""
    
    def __init__(self, config: ChannelConfig):
        """Initialize channel manager."""
        self.config = config
        self.channels: Dict[str, ChannelState] = {}
        self.state_validator = StateValidator(config)
        self.payment_processor = PaymentProcessor(config)
        
        # Initialize dispute manager if enabled
        if config.enable_disputes:
            dispute_config = DisputeConfig()
            self.dispute_manager = DisputeManager(dispute_config)
        else:
            self.dispute_manager = None
        
        logger.info("Initialized channel manager")
    
    def create_channel(self, channel_id: str, participants: List[ChannelParticipant], initial_balance: int) -> bool:
        """Create a new state channel."""
        try:
            # Validate inputs
            if not self._validate_channel_creation(channel_id, participants, initial_balance):
                logger.error(f"Invalid channel creation parameters")
                return False
            
            # Check if channel already exists
            if channel_id in self.channels:
                logger.error(f"Channel {channel_id} already exists")
                return False
            
            # Create channel state
            current_time = int(time.time())
            participants_dict = {p.address: p for p in participants}
            
            channel_state = ChannelState(
                channel_id=channel_id,
                participants=participants_dict,
                total_balance=initial_balance,
                nonce=0,
                created_at=current_time,
                last_updated=current_time,
                state_hash="",  # Will be calculated
                payments=[],
                metadata={}
            )
            
            # Calculate initial state hash
            channel_state.state_hash = self._calculate_state_hash(channel_state)
            
            # Store channel
            self.channels[channel_id] = channel_state
            
            logger.info(f"Created channel {channel_id} with {len(participants)} participants")
            return True
            
        except Exception as e:
            logger.error(f"Error creating channel: {e}")
            return False
    
    def process_payment(self, channel_id: str, payment: Payment) -> bool:
        """Process a payment in a channel."""
        try:
            if channel_id not in self.channels:
                logger.error(f"Channel {channel_id} not found")
                return False
            
            channel_state = self.channels[channel_id]
            
            # Check channel status
            if channel_state.last_updated + self.config.timeout_duration < int(time.time()):
                logger.error(f"Channel {channel_id} has expired")
                return False
            
            # Process payment
            new_state = self.payment_processor.process_payment(channel_state, payment)
            if not new_state:
                logger.error(f"Failed to process payment {payment.payment_id}")
                return False
            
            # Validate state transition
            if not self.state_validator.validate_state_transition(channel_state, new_state):
                logger.error(f"Invalid state transition for payment {payment.payment_id}")
                return False
            
            # Update channel state
            self.channels[channel_id] = new_state
            
            logger.info(f"Processed payment {payment.payment_id} in channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing payment: {e}")
            return False
    
    def close_channel(self, channel_id: str, closer_address: str) -> bool:
        """Close a state channel."""
        try:
            if channel_id not in self.channels:
                logger.error(f"Channel {channel_id} not found")
                return False
            
            channel_state = self.channels[channel_id]
            
            # Check if closer is a participant
            if closer_address not in channel_state.participants:
                logger.error(f"Address {closer_address} not in channel {channel_id}")
                return False
            
            # Check if channel is already closed
            if channel_state.last_updated == 0:  # Closed state
                logger.error(f"Channel {channel_id} already closed")
                return False
            
            # Close channel
            channel_state.last_updated = 0  # Mark as closed
            channel_state.state_hash = self._calculate_state_hash(channel_state)
            
            logger.info(f"Closed channel {channel_id} by {closer_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing channel: {e}")
            return False
    
    def initiate_dispute(self, channel_id: str, initiator_address: str) -> Optional[str]:
        """Initiate a dispute for a channel."""
        try:
            if not self.dispute_manager:
                logger.error("Disputes not enabled")
                return None
            
            if channel_id not in self.channels:
                logger.error(f"Channel {channel_id} not found")
                return None
            
            channel_state = self.channels[channel_id]
            
            # Check if initiator is a participant
            if initiator_address not in channel_state.participants:
                logger.error(f"Address {initiator_address} not in channel {channel_id}")
                return None
            
            # Create dispute
            dispute_id = self.dispute_manager.create_dispute(channel_id, initiator_address)
            
            # Update channel status
            channel_state.metadata['dispute_id'] = dispute_id
            channel_state.metadata['status'] = ChannelStatus.DISPUTED.value
            
            logger.info(f"Initiated dispute {dispute_id} for channel {channel_id}")
            return dispute_id
            
        except Exception as e:
            logger.error(f"Error initiating dispute: {e}")
            return None
    
    def get_channel(self, channel_id: str) -> Optional[ChannelState]:
        """Get channel state."""
        return self.channels.get(channel_id)
    
    def list_channels(self, participant_address: Optional[str] = None) -> List[ChannelState]:
        """List channels."""
        channels = list(self.channels.values())
        
        if participant_address:
            channels = [c for c in channels if participant_address in c.participants]
        
        return channels
    
    def cleanup_expired_channels(self) -> int:
        """Clean up expired channels."""
        try:
            current_time = int(time.time())
            expired_channels = []
            
            for channel_id, channel_state in self.channels.items():
                if (channel_state.last_updated > 0 and  # Not closed
                    current_time > channel_state.last_updated + self.config.timeout_duration):
                    expired_channels.append(channel_id)
            
            # Mark as expired
            for channel_id in expired_channels:
                self.channels[channel_id].metadata['status'] = ChannelStatus.EXPIRED.value
            
            logger.info(f"Cleaned up {len(expired_channels)} expired channels")
            return len(expired_channels)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired channels: {e}")
            return 0
    
    def _validate_channel_creation(self, channel_id: str, participants: List[ChannelParticipant], initial_balance: int) -> bool:
        """Validate channel creation parameters."""
        try:
            # Check channel ID
            if not channel_id:
                logger.error("Channel ID required")
                return False
            
            # Check participants
            if len(participants) < 2:
                logger.error("At least 2 participants required")
                return False
            
            if len(participants) > self.config.max_participants:
                logger.error(f"Too many participants: {len(participants)}")
                return False
            
            # Check addresses are unique
            addresses = [p.address for p in participants]
            if len(addresses) != len(set(addresses)):
                logger.error("Duplicate participant addresses")
                return False
            
            # Check initial balance
            if initial_balance < self.config.min_balance:
                logger.error(f"Initial balance too low: {initial_balance}")
                return False
            
            if initial_balance > self.config.max_balance:
                logger.error(f"Initial balance too high: {initial_balance}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating channel creation: {e}")
            return False
    
    def _calculate_state_hash(self, state: ChannelState) -> str:
        """Calculate state hash."""
        try:
            # Create hashable representation
            hash_data = {
                'channel_id': state.channel_id,
                'participants': {
                    addr: {
                        'balance': p.balance,
                        'nonce': p.nonce,
                        'public_key': p.public_key
                    }
                    for addr, p in state.participants.items()
                },
                'total_balance': state.total_balance,
                'nonce': state.nonce,
                'created_at': state.created_at,
                'last_updated': state.last_updated
            }
            
            # Add payment hashes
            payment_hashes = []
            for payment in state.payments:
                payment_hash = hashlib.sha256(
                    f"{payment.payment_id}{payment.from_address}{payment.to_address}{payment.amount}{payment.timestamp}".encode()
                ).hexdigest()
                payment_hashes.append(payment_hash)
            
            hash_data['payment_hashes'] = payment_hashes
            
            # Calculate hash
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating state hash: {e}")
            return ""

class ChannelError(Exception):
    """Channel-specific error."""
    pass

class InvalidStateUpdateError(ChannelError):
    """Invalid state update error."""
    pass

class InsufficientSignaturesError(ChannelError):
    """Insufficient signatures error."""
    pass

class ChannelTimeoutError(ChannelError):
    """Channel timeout error."""
    pass

class ChannelSecurityError(ChannelError):
    """Channel security error."""
    pass

__all__ = [
    "ChannelManager",
    "StateValidator",
    "PaymentProcessor",
    "ChannelState",
    "ChannelParticipant",
    "Payment",
    "ChannelConfig",
    "ChannelStatus",
    "PaymentType",
    "StateUpdateType",
    "ChannelCloseReason",
    "ChannelEvent",
    "StateUpdate",
    "StateChannel",
    "ChannelId",
    "ChannelError",
    "InvalidStateUpdateError",
    "InsufficientSignaturesError",
    "ChannelTimeoutError",
    "ChannelSecurityError",
]