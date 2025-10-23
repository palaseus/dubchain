"""
Unit tests for Off-Chain State Management

Tests the off-chain state management including:
- State transition validation
- Signature management
- State synchronization
- Conflict resolution
"""

import logging

logger = logging.getLogger(__name__)
import json
import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.state_channels.off_chain_state import (
    OffChainStateManager,
    StateValidator,
    StateTransition,
    StateSignature,
)
from src.dubchain.state_channels.channel_protocol import (
    ChannelConfig,
    ChannelId,
    ChannelState,
    StateUpdate,
    StateUpdateType,
    ChannelStatus,
)
from src.dubchain.crypto.signatures import PrivateKey, PublicKey, Signature


class TestStateSignature:
    """Test StateSignature functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.private_key = PrivateKey.generate()
        self.public_key = self.private_key.get_public_key()
        self.message_hash = self.private_key.sign(b"test message").message
    
    def test_create_state_signature(self):
        """Test creating a state signature."""
        signature = self.private_key.sign(self.message_hash)
        
        state_sig = StateSignature(
            participant="alice",
            signature=signature,
            timestamp=int(time.time()),
            nonce=123
        )
        
        assert state_sig.participant == "alice"
        assert state_sig.signature == signature
        assert state_sig.nonce == 123
    
    def test_verify_signature(self):
        """Test signature verification."""
        signature = self.private_key.sign(self.message_hash)
        
        state_sig = StateSignature(
            participant="alice",
            signature=signature,
            timestamp=int(time.time())
        )
        
        assert state_sig.verify(self.public_key, self.message_hash) is True
    
    def test_verify_invalid_signature(self):
        """Test verification of invalid signature."""
        wrong_key = PrivateKey.generate()
        signature = wrong_key.sign(self.message_hash)
        
        state_sig = StateSignature(
            participant="alice",
            signature=signature,
            timestamp=int(time.time())
        )
        
        assert state_sig.verify(self.public_key, self.message_hash) is False
    
    def test_serialization(self):
        """Test signature serialization."""
        signature = self.private_key.sign(self.message_hash)
        
        state_sig = StateSignature(
            participant="alice",
            signature=signature,
            timestamp=1234567890,
            nonce=456
        )
        
        data = state_sig.to_dict()
        
        assert data["participant"] == "alice"
        assert data["timestamp"] == 1234567890
        assert data["nonce"] == 456
        assert "signature" in data


class TestStateTransition:
    """Test StateTransition functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.from_state = {
            "balances": {"alice": 5000, "bob": 3000},
            "sequence_number": 0
        }
        self.to_state = {
            "balances": {"alice": 4000, "bob": 4000},
            "sequence_number": 1
        }
    
    def test_create_state_transition(self):
        """Test creating a state transition."""
        transition = StateTransition(
            from_state=self.from_state,
            to_state=self.to_state,
            transition_type=StateUpdateType.TRANSFER,
            validation_rules=["balance_conservation"],
            preconditions={"min_balance": {"participant": "alice", "amount": 1000}},
            postconditions={"balance_conservation": True}
        )
        
        assert transition.from_state == self.from_state
        assert transition.to_state == self.to_state
        assert transition.transition_type == StateUpdateType.TRANSFER
        assert "balance_conservation" in transition.validation_rules
    
    def test_validate_preconditions(self):
        """Test precondition validation."""
        transition = StateTransition(
            from_state=self.from_state,
            to_state=self.to_state,
            transition_type=StateUpdateType.TRANSFER,
            preconditions={"min_balance": {"participant": "alice", "amount": 1000}}
        )
        
        # Create mock channel state
        channel_state = Mock()
        channel_state.balances = {"alice": 5000, "bob": 3000}
        channel_state.sequence_number = 0
        channel_state.participants = ["alice", "bob"]
        
        assert transition.validate_preconditions(channel_state) is True
        
        # Test with insufficient balance
        channel_state.balances = {"alice": 500, "bob": 3000}
        assert transition.validate_preconditions(channel_state) is False
    
    def test_validate_postconditions(self):
        """Test postcondition validation."""
        transition = StateTransition(
            from_state=self.from_state,
            to_state=self.to_state,
            transition_type=StateUpdateType.TRANSFER,
            postconditions={"balance_conservation": True}
        )
        
        # Create mock channel state
        new_state = Mock()
        new_state.balances = {"alice": 4000, "bob": 4000}
        new_state.sequence_number = 1
        
        assert transition.validate_postconditions(new_state) is True
        
        # Test with balance conservation violation
        new_state.balances = {"alice": 3000, "bob": 3000}  # Total changed
        assert transition.validate_postconditions(new_state) is False
    
    def test_get_transition_hash(self):
        """Test getting transition hash."""
        transition1 = StateTransition(
            from_state=self.from_state,
            to_state=self.to_state,
            transition_type=StateUpdateType.TRANSFER
        )
        
        transition2 = StateTransition(
            from_state=self.from_state,
            to_state=self.to_state,
            transition_type=StateUpdateType.TRANSFER
        )
        
        # Same transitions should have same hash
        assert transition1.get_transition_hash() == transition2.get_transition_hash()
        
        # Different transitions should have different hashes
        transition3 = StateTransition(
            from_state=self.from_state,
            to_state={"balances": {"alice": 3000, "bob": 5000}},  # Different to_state
            transition_type=StateUpdateType.TRANSFER
        )
        
        assert transition1.get_transition_hash() != transition3.get_transition_hash()


class TestStateValidator:
    """Test StateValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.validator = StateValidator(self.config)
        self.channel_id = ChannelId.generate()
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
    
    def create_channel_state(self, sequence_number=0):
        """Create a test channel state."""
        return ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits={"alice": 5000, "bob": 3000},
            balances={"alice": 5000, "bob": 3000},
            sequence_number=sequence_number,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
    
    def create_signed_update(self, sequence_number=1, amount=1000):
        """Create a signed state update."""
        update = StateUpdate(
            update_id=f"update-{sequence_number}",
            channel_id=self.channel_id,
            sequence_number=sequence_number,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": amount},
            timestamp=int(time.time())
        )
        
        # Add signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        return update
    
    def test_validate_state_update_success(self):
        """Test successful state update validation."""
        channel_state = self.create_channel_state()
        update = self.create_signed_update()
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_state_update_wrong_sequence(self):
        """Test validation with wrong sequence number."""
        channel_state = self.create_channel_state(sequence_number=0)
        update = self.create_signed_update(sequence_number=3)  # Wrong sequence
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Invalid sequence number" in error for error in errors)
    
    def test_validate_state_update_wrong_participants(self):
        """Test validation with wrong participants."""
        channel_state = self.create_channel_state()
        update = self.create_signed_update()
        update.participants = ["alice", "charlie"]  # Wrong participants
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Participant set mismatch" in error for error in errors)
    
    def test_validate_state_update_invalid_signatures(self):
        """Test validation with invalid signatures."""
        channel_state = self.create_channel_state()
        update = self.create_signed_update()
        
        # Replace with invalid signature
        wrong_key = PrivateKey.generate()
        invalid_sig = wrong_key.sign(update.get_hash())
        update.signatures["alice"] = invalid_sig
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Invalid signatures" in error for error in errors)
    
    def test_validate_state_update_insufficient_signatures(self):
        """Test validation with insufficient signatures."""
        config = ChannelConfig(require_all_signatures=True)
        validator = StateValidator(config)
        channel_state = self.create_channel_state()
        update = self.create_signed_update()
        
        # Remove one signature
        del update.signatures["bob"]
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Insufficient signatures" in error for error in errors)
    
    def test_validate_transfer_update_insufficient_balance(self):
        """Test validation of transfer with insufficient balance."""
        channel_state = self.create_channel_state()
        update = self.create_signed_update(amount=10000)  # More than alice has
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Insufficient balance" in error for error in errors)
    
    def test_validate_conditional_update(self):
        """Test validation of conditional update."""
        channel_state = self.create_channel_state()
        
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.CONDITIONAL,
            participants=self.participants,
            state_data={
                "sender": "alice",
                "recipient": "bob",
                "amount": 1000,
                "condition": {
                    "type": "time_based",
                    "target_time": int(time.time()) + 3600  # 1 hour from now
                }
            },
            timestamp=int(time.time())
        )
        
        # Add signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_multi_party_update(self):
        """Test validation of multi-party update."""
        channel_state = self.create_channel_state()
        
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.MULTI_PARTY,
            participants=self.participants,
            state_data={
                "transfers": [
                    {"sender": "alice", "recipient": "bob", "amount": 1000},
                    {"sender": "bob", "recipient": "alice", "amount": 500}
                ]
            },
            timestamp=int(time.time())
        )
        
        # Add signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_multi_party_update_insufficient_balance(self):
        """Test validation of multi-party update with insufficient balance."""
        channel_state = self.create_channel_state()
        
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.MULTI_PARTY,
            participants=self.participants,
            state_data={
                "transfers": [
                    {"sender": "alice", "recipient": "bob", "amount": 10000},  # Too much
                    {"sender": "bob", "recipient": "alice", "amount": 500}
                ]
            },
            timestamp=int(time.time())
        )
        
        # Add signatures
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Insufficient balance" in error for error in errors)
    
    def test_add_custom_validation_rule(self):
        """Test adding custom validation rules."""
        def custom_rule(update, channel_state, public_keys):
            if update.update_id == "forbidden":
                return False, ["Forbidden update ID"]
            return True, []
        
        self.validator.add_validation_rule("custom_rule", custom_rule)
        
        # Test with forbidden update ID
        channel_state = self.create_channel_state()
        update = self.create_signed_update()
        update.update_id = "forbidden"
        
        is_valid, errors = self.validator.validate_state_update(
            update, channel_state, self.public_keys
        )
        
        assert is_valid is False
        assert any("Forbidden update ID" in error for error in errors)


class TestOffChainStateManager:
    """Test OffChainStateManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.manager = OffChainStateManager(self.config)
        self.channel_id = ChannelId.generate()
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
    
    def test_create_channel_state(self):
        """Test creating a channel state."""
        deposits = {"alice": 5000, "bob": 3000}
        
        state = self.manager.create_channel_state(
            self.channel_id, self.participants, deposits
        )
        
        assert state.channel_id == self.channel_id
        assert state.participants == self.participants
        assert state.deposits == deposits
        assert state.balances == deposits
        assert state.sequence_number == 0
        assert state.status == ChannelStatus.PENDING
    
    def test_get_channel_state(self):
        """Test getting a channel state."""
        deposits = {"alice": 5000, "bob": 3000}
        self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        state = self.manager.get_channel_state(self.channel_id)
        assert state is not None
        assert state.channel_id == self.channel_id
    
    def test_get_nonexistent_channel_state(self):
        """Test getting a non-existent channel state."""
        nonexistent_id = ChannelId.generate()
        state = self.manager.get_channel_state(nonexistent_id)
        assert state is None
    
    def test_update_channel_state_success(self):
        """Test successful channel state update."""
        deposits = {"alice": 5000, "bob": 3000}
        state = self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        # Open the channel for updates
        state.status = ChannelStatus.OPEN
        
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
        
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        # Update state
        success, errors = self.manager.update_channel_state(
            self.channel_id, update, self.public_keys
        )
        
        assert success is True
        assert len(errors) == 0
        
        # Check state was updated
        state = self.manager.get_channel_state(self.channel_id)
        assert state.sequence_number == 1
        assert state.balances["alice"] == 4000
        assert state.balances["bob"] == 4000
    
    def test_update_channel_state_validation_failure(self):
        """Test channel state update with validation failure."""
        deposits = {"alice": 5000, "bob": 3000}
        self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        # Create update with insufficient balance
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 10000},
            timestamp=int(time.time())
        )
        
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        # Update should fail
        success, errors = self.manager.update_channel_state(
            self.channel_id, update, self.public_keys
        )
        
        assert success is False
        assert len(errors) > 0
        assert any("Insufficient balance" in error for error in errors)
    
    def test_sign_state_update(self):
        """Test signing a state update."""
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        alice_key = self.private_keys["alice"]
        state_sig = self.manager.sign_state_update(update, "alice", alice_key)
        
        assert state_sig.participant == "alice"
        assert state_sig.verify(self.public_keys["alice"], update.get_hash()) is True
    
    def test_collect_signatures(self):
        """Test collecting signatures from multiple participants."""
        update = StateUpdate(
            update_id="update-1",
            channel_id=self.channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
            timestamp=int(time.time())
        )
        
        signatures = self.manager.collect_signatures(
            update, self.participants, self.private_keys
        )
        
        assert len(signatures) == 2
        assert "alice" in signatures
        assert "bob" in signatures
        
        # Verify signatures
        for participant, state_sig in signatures.items():
            assert state_sig.verify(self.public_keys[participant], update.get_hash()) is True
    
    def test_verify_state_consistency(self):
        """Test state consistency verification."""
        deposits = {"alice": 5000, "bob": 3000}
        self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        is_consistent, errors = self.manager.verify_state_consistency(self.channel_id)
        
        assert is_consistent is True
        assert len(errors) == 0
    
    def test_verify_state_consistency_balance_violation(self):
        """Test state consistency verification with balance violation."""
        deposits = {"alice": 5000, "bob": 3000}
        state = self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        # Manually corrupt balances
        state.balances["alice"] = 6000  # This violates conservation
        
        is_consistent, errors = self.manager.verify_state_consistency(self.channel_id)
        
        assert is_consistent is False
        assert any("Balance conservation violated" in error for error in errors)
    
    def test_resolve_state_conflicts_latest_wins(self):
        """Test state conflict resolution with latest wins strategy."""
        # Create conflicting states
        state1 = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits={"alice": 5000, "bob": 3000},
            balances={"alice": 5000, "bob": 3000},
            sequence_number=1,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        state2 = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits={"alice": 5000, "bob": 3000},
            balances={"alice": 4000, "bob": 4000},
            sequence_number=2,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        resolved_state = self.manager.resolve_state_conflicts(
            self.channel_id, [state1, state2]
        )
        
        assert resolved_state.sequence_number == 2  # Latest wins
    
    def test_synchronize_states(self):
        """Test state synchronization."""
        deposits = {"alice": 5000, "bob": 3000}
        local_state = self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        # Create identical remote state
        remote_state = ChannelState(
            channel_id=self.channel_id,
            participants=self.participants,
            deposits=deposits,
            balances=deposits,
            sequence_number=0,
            last_update_timestamp=int(time.time()),
            status=ChannelStatus.OPEN,
            config=self.config
        )
        
        success = self.manager.synchronize_states(self.channel_id, [remote_state])
        assert success is True
    
    def test_export_import_state(self):
        """Test state export and import."""
        deposits = {"alice": 5000, "bob": 3000}
        self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        # Export state
        exported_data = self.manager.export_state(self.channel_id)
        assert exported_data is not None
        assert exported_data["channel_id"] == self.channel_id.value
        
        # Create new manager and import state
        new_manager = OffChainStateManager(self.config)
        success = new_manager.import_state(exported_data)
        
        assert success is True
        
        # Verify imported state
        imported_state = new_manager.get_channel_state(self.channel_id)
        assert imported_state is not None
        assert imported_state.participants == self.participants
        assert imported_state.deposits == deposits
    
    def test_cleanup_channel(self):
        """Test channel cleanup."""
        deposits = {"alice": 5000, "bob": 3000}
        self.manager.create_channel_state(self.channel_id, self.participants, deposits)
        
        # Verify channel exists
        assert self.manager.get_channel_state(self.channel_id) is not None
        
        # Cleanup channel
        self.manager.cleanup_channel(self.channel_id)
        
        # Verify channel state is preserved but pending updates are removed
        assert self.manager.get_channel_state(self.channel_id) is not None
        assert self.channel_id not in self.manager.pending_updates


if __name__ == "__main__":
    pytest.main([__file__])
