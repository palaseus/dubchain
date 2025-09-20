"""
Unit tests for State Channel Protocol

Tests the core state channel protocol including:
- Channel lifecycle management
- State update validation
- Multi-party coordination
- Event handling
"""

import json
import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.state_channels.channel_protocol import (
    ChannelConfig,
    ChannelId,
    ChannelState,
    StateChannel,
    StateUpdate,
    StateUpdateType,
    ChannelStatus,
    ChannelEvent,
    ChannelCloseReason,
    InvalidStateUpdateError,
    InsufficientSignaturesError,
    ChannelTimeoutError,
    ChannelSecurityError,
)
from src.dubchain.crypto.signatures import PrivateKey, PublicKey, Signature


class TestChannelId:
    """Test ChannelId functionality."""
    
    def test_generate_unique_ids(self):
        """Test that generated channel IDs are unique."""
        ids = [ChannelId.generate() for _ in range(100)]
        unique_ids = set(id.value for id in ids)
        assert len(unique_ids) == 100
    
    def test_id_equality(self):
        """Test channel ID equality."""
        id1 = ChannelId("test-id")
        id2 = ChannelId("test-id")
        id3 = ChannelId("different-id")
        
        assert id1 == id2
        assert id1 != id3
        assert hash(id1) == hash(id2)
        assert hash(id1) != hash(id3)


class TestChannelConfig:
    """Test ChannelConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChannelConfig()
        
        assert config.timeout_blocks == 1000
        assert config.dispute_period_blocks == 100
        assert config.max_participants == 10
        assert config.min_deposit == 1000
        assert config.require_all_signatures is True
        assert config.enable_fraud_proofs is True
        assert config.enable_timeout_mechanism is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChannelConfig(
            timeout_blocks=500,
            max_participants=5,
            min_deposit=5000,
            require_all_signatures=False
        )
        
        assert config.timeout_blocks == 500
        assert config.max_participants == 5
        assert config.min_deposit == 5000
        assert config.require_all_signatures is False
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = ChannelConfig(
            timeout_blocks=2000,
            max_participants=8,
            min_deposit=2000
        )
        
        config_dict = config.to_dict()
        restored_config = ChannelConfig.from_dict(config_dict)
        
        assert restored_config.timeout_blocks == config.timeout_blocks
        assert restored_config.max_participants == config.max_participants
        assert restored_config.min_deposit == config.min_deposit


class TestStateUpdate:
    """Test StateUpdate functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.channel_id = ChannelId.generate()
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
    
    def test_create_transfer_update(self):
        """Test creating a transfer state update."""
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={
                "sender": "alice",
                "recipient": "bob",
                "amount": 1000
            },
            timestamp=int(time.time())
        )
        
        assert update.update_id == "update-1"
        assert update.channel_id == self.channel_id
        assert update.sequence_number == 1
        assert update.update_type == StateUpdateType.TRANSFER
        assert update.participants == self.participants
    
    def test_add_signatures(self):
        """Test adding signatures to state update."""
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # Add signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        assert len(update.signatures) == 2
        assert "alice" in update.signatures
        assert "bob" in update.signatures
    
    def test_verify_signatures(self):
        """Test signature verification."""
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # Add valid signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        # Verify signatures
        assert update.verify_signatures(self.public_keys) is True
    
    def test_verify_invalid_signatures(self):
        """Test verification of invalid signatures."""
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # Add invalid signature (wrong private key)
        wrong_key = PrivateKey.generate()
        signature = wrong_key.sign(update.get_hash())
        update.add_signature("alice", signature)
        
        # Verification should fail
        assert update.verify_signatures(self.public_keys) is False
    
    def test_has_required_signatures(self):
        """Test checking for required signatures."""
        config = ChannelConfig(require_all_signatures=True)
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # No signatures
        assert update.has_required_signatures(config) is False
        
        # One signature (not enough)
        alice_key = self.private_keys["alice"]
        signature = alice_key.sign(update.get_hash())
        update.add_signature("alice", signature)
        assert update.has_required_signatures(config) is False
        
        # All signatures
        bob_key = self.private_keys["bob"]
        signature = bob_key.sign(update.get_hash())
        update.add_signature("bob", signature)
        assert update.has_required_signatures(config) is True
    
    def test_has_required_signatures_majority(self):
        """Test checking for majority signatures."""
        config = ChannelConfig(require_all_signatures=False)
        participants = ["alice", "bob", "charlie"]
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # 2 out of 3 signatures (majority)
        alice_key = PrivateKey.generate()
        bob_key = PrivateKey.generate()
        
        alice_sig = alice_key.sign(update.get_hash())
        bob_sig = bob_key.sign(update.get_hash())
        
        update.add_signature("alice", alice_sig)
        update.add_signature("bob", bob_sig)
        
        assert update.has_required_signatures(config) is True
    
    def test_get_hash_consistency(self):
        """Test that hash is consistent for same data."""
        update1 = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=1234567890
        )
        
        update2 = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=1234567890
        )
        
        assert update1.get_hash() == update2.get_hash()
    
    def test_get_hash_different_data(self):
        """Test that hash is different for different data."""
        update1 = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        update2 = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 2000},  # Different amount
            timestamp=int(time.time())
        )
        
        assert update1.get_hash() != update2.get_hash()


class TestChannelState:
    """Test ChannelState functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.channel_id = ChannelId.generate()
        self.participants = ["alice", "bob"]
        self.deposits = {"alice": 5000, "bob": 3000}
        self.config = ChannelConfig(min_deposit=1000)
    
    def test_create_channel_state(self):
        """Test creating a channel state."""
        state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=self.deposits,
            balances=self.deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.PENDING,
            config=self.config
        )
        
        assert state.channel_id == self.channel_id
        assert state.participants == self.participants
        assert state.deposits == self.deposits
        assert state.balances == self.deposits
        assert state.sequence_number == 0
        assert state.status == ChannelStatus.PENDING
    
    def test_validate_balances(self):
        """Test balance validation."""
        state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=self.deposits,
            balances=self.deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.PENDING,
            config=self.config
        )
        
        assert state.validate_balances() is True
        
        # Modify balances to be inconsistent
        state.balances["alice"] = 6000
        assert state.validate_balances() is False
    
    def test_get_total_deposits(self):
        """Test getting total deposits."""
        state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=self.deposits,
            balances=self.deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.PENDING,
            config=self.config
        )
        
        assert state.get_total_deposits() == 8000
    
    def test_can_update_state(self):
        """Test checking if state can be updated."""
        state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=self.deposits,
            balances=self.deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        # Create valid update
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        assert state.can_update_state(update) is True
        
        # Test with wrong sequence number
        update.sequence_number = 2
        assert state.can_update_state(update) is False
        
        # Test with wrong participants
        update.sequence_number = 1
        update.participants = ["alice", "charlie"]
        assert state.can_update_state(update) is False
    
    def test_apply_transfer_update(self):
        """Test applying a transfer state update."""
        state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=self.deposits,
            balances=self.deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # Apply update
        success = state.apply_state_update(update)
        assert success is True
        
        # Check balances
        assert state.balances["alice"] == 4000  # 5000 - 1000
        assert state.balances["bob"] == 4000    # 3000 + 1000
        assert state.sequence_number == 1
    
    def test_apply_insufficient_balance_transfer(self):
        """Test applying transfer with insufficient balance."""
        state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=self.deposits,
            balances=self.deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 10000},  # More than alice has
            timestamp=int(time.time())
        )
        
        # Apply update should fail
        with pytest.raises(ValueError, match="Insufficient balance"):
            state.apply_state_update(update)
    
    def test_apply_multi_party_update(self):
        """Test applying a multi-party state update."""
        participants = ["alice", "bob", "charlie"]
        deposits = {"alice": 5000, "bob": 3000, "charlie": 2000}
        
        state = ChannelState(
            channel_id=self.channel_id,
            participants=participants,
            deposits=deposits,
            balances=deposits.copy(),
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.MULTI_PARTY,
            participants=participants,
            state_data={
                "transfers": [
                    {"sender": "alice", "recipient": "bob", "amount": 1000},
                    {"sender": "bob", "recipient": "charlie", "amount": 500}
                ]
            },
            timestamp=int(time.time())
        )
        
        # Apply update
        success = state.apply_state_update(update)
        assert success is True
        
        # Check balances
        assert state.balances["alice"] == 4000  # 5000 - 1000
        assert state.balances["bob"] == 3500    # 3000 + 1000 - 500
        assert state.balances["charlie"] == 2500  # 2000 + 500
        assert state.sequence_number == 1


class TestStateChannel:
    """Test StateChannel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.channel_id = ChannelId.generate()
        self.config = ChannelConfig()
        self.participants = ["alice", "bob"]
        self.deposits = {"alice": 5000, "bob": 3000}
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
    
    def test_create_channel(self):
        """Test creating a state channel."""
        channel = StateChannel(self.channel_id, self.config)
        
        success = channel.create_channel(self.participants, self.deposits, self.public_keys)
        assert success is True
        
        state = channel.get_latest_state()
        assert state is not None
        assert state.participants == self.participants
        assert state.deposits == self.deposits
        assert state.status == ChannelStatus.PENDING
    
    def test_create_channel_insufficient_participants(self):
        """Test creating channel with insufficient participants."""
        channel = StateChannel(self.channel_id, self.config)
        
        success = channel.create_channel(["alice"], self.deposits, self.public_keys)
        assert success is False
    
    def test_create_channel_insufficient_deposits(self):
        """Test creating channel with insufficient deposits."""
        channel = StateChannel(self.channel_id, self.config)
        
        small_deposits = {"alice": 100, "bob": 200}  # Below minimum
        success = channel.create_channel(self.participants, small_deposits, self.public_keys)
        assert success is False
    
    def test_open_channel(self):
        """Test opening a channel."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        
        success = channel.open_channel()
        assert success is True
        
        state = channel.get_latest_state()
        assert state.status == ChannelStatus.OPEN
        assert state.opened_at is not None
    
    def test_update_state(self):
        """Test updating channel state."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        # Create and sign update
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # Add signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        # Update state
        success = channel.update_state(update)
        assert success is True
        
        state = channel.get_latest_state()
        assert state.sequence_number == 1
        assert state.balances["alice"] == 4000
        assert state.balances["bob"] == 4000
    
    def test_update_state_invalid_signatures(self):
        """Test updating state with invalid signatures."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        # Create update with invalid signature
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        # Add invalid signature
        wrong_key = PrivateKey.generate()
        signature = wrong_key.sign(update.get_hash())
        update.add_signature("alice", signature)
        
        # Update should fail
        with pytest.raises(InvalidStateUpdateError):
            channel.update_state(update)
    
    def test_close_channel(self):
        """Test closing a channel."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        success = channel.close_channel()
        assert success is True
        
        state = channel.get_latest_state()
        assert state.status == ChannelStatus.CLOSED
        assert state.closed_at is not None
        assert state.close_reason == ChannelCloseReason.COOPERATIVE
    
    def test_expire_channel(self):
        """Test expiring a channel."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        success = channel.expire_channel()
        assert success is True
        
        state = channel.get_latest_state()
        assert state.status == ChannelStatus.EXPIRED
        assert state.close_reason == ChannelCloseReason.TIMEOUT
    
    def test_freeze_channel(self):
        """Test freezing a channel."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        success = channel.freeze_channel("Security violation detected")
        assert success is True
        
        state = channel.get_latest_state()
        assert state.status == ChannelStatus.FROZEN
        assert state.close_reason == ChannelCloseReason.SECURITY_VIOLATION
    
    def test_event_handlers(self):
        """Test event handler functionality."""
        channel = StateChannel(self.channel_id, self.config)
        
        # Track events
        events_received = []
        
        def event_handler(event, state):
            events_received.append(event)
        
        # Add event handler
        channel.add_event_handler(ChannelEvent.CREATED, event_handler)
        channel.add_event_handler(ChannelEvent.OPENED, event_handler)
        
        # Create and open channel
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        # Check events were received
        assert ChannelEvent.CREATED in events_received
        assert ChannelEvent.OPENED in events_received
    
    def test_get_channel_info(self):
        """Test getting channel information."""
        channel = StateChannel(self.channel_id, self.config)
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        channel.open_channel()
        
        info = channel.get_channel_info()
        
        assert info["channel_id"] == self.channel_id.value
        assert info["status"] == ChannelStatus.OPEN.value
        assert info["participants"] == self.participants
        assert info["total_deposits"] == 8000
        assert info["total_balances"] == 8000
        assert info["sequence_number"] == 0
    
    def test_is_active(self):
        """Test checking if channel is active."""
        channel = StateChannel(self.channel_id, self.config)
        
        # Not created yet
        assert channel.is_active() is False
        
        # Created but not opened
        channel.create_channel(self.participants, self.deposits, self.public_keys)
        assert channel.is_active() is False
        
        # Opened
        channel.open_channel()
        assert channel.is_active() is True
        
        # Closed
        channel.close_channel()
        assert channel.is_active() is False


if __name__ == "__main__":
    pytest.main([__file__])
