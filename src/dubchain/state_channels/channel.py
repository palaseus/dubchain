"""
State channels implementation for DubChain.

This module provides off-chain state channel functionality for
fast, low-cost transactions between parties.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature


class ChannelStatus(Enum):
    """Status of a state channel."""

    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    DISPUTED = "disputed"
    FINALIZED = "finalized"


class ChannelState(Enum):
    """State of a channel."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class ChannelParticipant:
    """A participant in a state channel."""

    address: str
    public_key: PublicKey
    balance: int = 0
    nonce: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "public_key": self.public_key.to_hex(),
            "balance": self.balance,
            "nonce": self.nonce,
        }


@dataclass
class ChannelUpdate:
    """An update to a state channel."""

    channel_id: str
    participants: List[ChannelParticipant]
    nonce: int
    timestamp: int = 0
    signatures: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time())

    def add_signature(self, participant_address: str, signature: str) -> None:
        """Add a signature to the update."""
        self.signatures[participant_address] = signature

    def is_fully_signed(self, required_participants: Set[str]) -> bool:
        """Check if the update is fully signed."""
        return all(addr in self.signatures for addr in required_participants)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id,
            "participants": [p.to_dict() for p in self.participants],
            "nonce": self.nonce,
            "timestamp": self.timestamp,
            "signatures": self.signatures,
        }


@dataclass
class StateChannel:
    """A state channel between multiple parties."""

    channel_id: str
    participants: List[ChannelParticipant]
    status: ChannelStatus = ChannelStatus.OPENING
    state: ChannelState = ChannelState.PENDING
    created_at: int = 0
    expires_at: int = 0
    total_deposit: int = 0
    current_update: Optional[ChannelUpdate] = None
    dispute_period: int = 7 * 24 * 3600  # 7 days
    challenge_period: int = 24 * 3600  # 1 day

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = int(time.time())
        if self.expires_at == 0:
            self.expires_at = self.created_at + (30 * 24 * 3600)  # 30 days

    def add_participant(self, participant: ChannelParticipant) -> None:
        """Add a participant to the channel."""
        self.participants.append(participant)
        self.total_deposit += participant.balance

    def get_participant(self, address: str) -> Optional[ChannelParticipant]:
        """Get a participant by address."""
        for participant in self.participants:
            if participant.address == address:
                return participant
        return None

    def update_participant_balance(self, address: str, new_balance: int) -> bool:
        """Update a participant's balance."""
        participant = self.get_participant(address)
        if not participant:
            return False

        old_balance = participant.balance
        participant.balance = new_balance
        self.total_deposit = self.total_deposit - old_balance + new_balance
        return True

    def create_update(self, new_balances: Dict[str, int]) -> Optional[ChannelUpdate]:
        """Create a new channel update."""
        if self.status != ChannelStatus.OPEN:
            return None

        # Update participant balances
        for address, balance in new_balances.items():
            if not self.update_participant_balance(address, balance):
                return None

        # Create update
        update = ChannelUpdate(
            channel_id=self.channel_id,
            participants=self.participants.copy(),
            nonce=self.current_update.nonce + 1 if self.current_update else 1,
        )

        self.current_update = update
        return update

    def sign_update(self, participant_address: str, private_key: PrivateKey) -> bool:
        """Sign the current update."""
        if not self.current_update:
            return False

        participant = self.get_participant(participant_address)
        if not participant:
            return False

        # Create signature data
        update_data = self.current_update.to_dict()
        signature_data = f"{self.channel_id}_{participant_address}_{self.current_update.nonce}_{self.current_update.timestamp}"
        signature_hash = SHA256Hasher.hash(signature_data.encode())
        signature = private_key.sign(signature_hash)

        # Add signature
        self.current_update.add_signature(participant_address, signature.to_hex())
        return True

    def can_close(self) -> bool:
        """Check if the channel can be closed."""
        if not self.current_update:
            return False

        required_participants = {p.address for p in self.participants}
        return self.current_update.is_fully_signed(required_participants)

    def close_channel(self) -> bool:
        """Close the channel."""
        if not self.can_close():
            return False

        self.status = ChannelStatus.CLOSING
        return True

    def finalize_channel(self) -> bool:
        """Finalize the channel."""
        if self.status != ChannelStatus.CLOSING:
            return False

        self.status = ChannelStatus.FINALIZED
        self.state = ChannelState.INACTIVE
        return True

    def create_channel(self, participants: List[ChannelParticipant], deposits: List[int], public_keys: List[PublicKey]) -> bool:
        """Create/initialize the channel with participants and deposits."""
        try:
            if len(participants) < 2:
                return False
            
            if len(participants) != len(deposits) or len(participants) != len(public_keys):
                return False
            
            # Set participants with deposits
            self.participants = []
            self.total_deposit = 0
            
            for i, participant in enumerate(participants):
                participant.balance = deposits[i]
                participant.public_key = public_keys[i]
                self.participants.append(participant)
                self.total_deposit += deposits[i]
            
            self.status = ChannelStatus.OPENING
            self.state = ChannelState.PENDING
            return True
            
        except Exception:
            return False

    def get_latest_state(self) -> Optional[ChannelUpdate]:
        """Get the latest state update."""
        return self.current_update

    def is_expired(self) -> bool:
        """Check if the channel is expired."""
        return int(time.time()) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id,
            "participants": [p.to_dict() for p in self.participants],
            "status": self.status.value,
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "total_deposit": self.total_deposit,
            "current_update": self.current_update.to_dict()
            if self.current_update
            else None,
            "dispute_period": self.dispute_period,
            "challenge_period": self.challenge_period,
        }


class StateChannelManager:
    """Manages state channels."""

    def __init__(self):
        self.channels: Dict[str, StateChannel] = {}
        self.participant_channels: Dict[str, Set[str]] = {}

    def create_channel(
        self,
        channel_id: str,
        participants: List[ChannelParticipant],
        dispute_period: int = 7 * 24 * 3600,
        challenge_period: int = 24 * 3600,
    ) -> StateChannel:
        """Create a new state channel."""
        channel = StateChannel(
            channel_id=channel_id,
            participants=participants,
            dispute_period=dispute_period,
            challenge_period=challenge_period,
        )

        self.channels[channel_id] = channel

        # Track channels per participant
        for participant in participants:
            if participant.address not in self.participant_channels:
                self.participant_channels[participant.address] = set()
            self.participant_channels[participant.address].add(channel_id)

        return channel

    def get_channel(self, channel_id: str) -> Optional[StateChannel]:
        """Get a channel by ID."""
        return self.channels.get(channel_id)

    def get_participant_channels(self, address: str) -> List[StateChannel]:
        """Get all channels for a participant."""
        channel_ids = self.participant_channels.get(address, set())
        return [self.channels[cid] for cid in channel_ids if cid in self.channels]

    def update_channel(
        self,
        channel_id: str,
        new_balances: Dict[str, int],
        participant_address: str,
        private_key: PrivateKey,
    ) -> bool:
        """Update a channel with new balances."""
        channel = self.get_channel(channel_id)
        if not channel:
            return False

        # Create update
        update = channel.create_update(new_balances)
        if not update:
            return False

        # Sign update
        return channel.sign_update(participant_address, private_key)

    def close_channel(self, channel_id: str) -> bool:
        """Close a channel."""
        channel = self.get_channel(channel_id)
        if not channel:
            return False

        return channel.close_channel()

    def finalize_channel(self, channel_id: str) -> bool:
        """Finalize a channel."""
        channel = self.get_channel(channel_id)
        if not channel:
            return False

        return channel.finalize_channel()

    def get_channel_stats(self) -> Dict[str, Any]:
        """Get channel statistics."""
        total_channels = len(self.channels)
        open_channels = len(
            [c for c in self.channels.values() if c.status == ChannelStatus.OPEN]
        )
        closed_channels = len(
            [c for c in self.channels.values() if c.status == ChannelStatus.CLOSED]
        )
        disputed_channels = len(
            [c for c in self.channels.values() if c.status == ChannelStatus.DISPUTED]
        )

        total_participants = len(self.participant_channels)
        total_deposits = sum(c.total_deposit for c in self.channels.values())

        return {
            "total_channels": total_channels,
            "open_channels": open_channels,
            "closed_channels": closed_channels,
            "disputed_channels": disputed_channels,
            "total_participants": total_participants,
            "total_deposits": total_deposits,
        }

    def cleanup_expired_channels(self) -> int:
        """Clean up expired channels."""
        current_time = int(time.time())
        expired_channels = []

        for channel_id, channel in self.channels.items():
            if channel.is_expired() and channel.status not in [
                ChannelStatus.CLOSED,
                ChannelStatus.FINALIZED,
            ]:
                expired_channels.append(channel_id)

        for channel_id in expired_channels:
            channel = self.channels[channel_id]
            channel.status = ChannelStatus.CLOSED
            channel.state = ChannelState.EXPIRED

            # Remove from participant tracking
            for participant in channel.participants:
                if participant.address in self.participant_channels:
                    self.participant_channels[participant.address].discard(channel_id)
        return len(expired_channels)
