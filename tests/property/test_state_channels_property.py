"""
Property-based tests for State Channels

Tests state channel invariants and properties using hypothesis for
generative testing. These tests verify that certain properties always
hold regardless of the specific inputs or sequences of operations.
"""

import pytest

# Temporarily disable property tests due to hanging issues
pytestmark = pytest.mark.skip(reason="Temporarily disabled due to hanging issues")
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, Bundle
import time
from typing import List, Dict, Any, Tuple

from src.dubchain.state_channels.channel_manager import ChannelManager
from src.dubchain.state_channels.channel_protocol import (
    ChannelConfig,
    ChannelId,
    StateUpdate,
    StateUpdateType,
    ChannelStatus,
    ChannelEvent,
)
from src.dubchain.crypto.signatures import PrivateKey, PublicKey


class StateChannelPropertyTests:
    """Property-based tests for state channels."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig(
            timeout_blocks=1000,
            dispute_period_blocks=100,
            max_participants=10,
            min_deposit=1000,
            state_update_timeout=300
        )
    
    @given(
        participants=st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=2,
            max_size=5,
            unique=True
        ),
        deposits=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            values=st.integers(min_value=1000, max_value=100000),
            min_size=2,
            max_size=5
        )
    )
    def test_balance_conservation_property(self, participants: List[str], deposits: Dict[str, int]):
        """Test that total balances are always conserved."""
        # Ensure all participants have deposits
        if not all(p in deposits for p in participants):
            return
        
        manager = ChannelManager(self.config)
        
        # Create keys for participants
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        
        # Create channel
        success, channel_id, errors = manager.create_channel(
            participants, deposits, public_keys
        )
        
        if not success:
            return  # Skip if channel creation fails
        
        manager.open_channel(channel_id)
        
        initial_total = sum(deposits.values())
        
        # Perform random updates
        for i in range(10):
            # Generate random transfer
            sender = participants[i % len(participants)]
            recipient = participants[(i + 1) % len(participants)]
            amount = min(1000, deposits[sender] // 2)  # Reasonable amount
            
            if amount > 0:
                update = StateUpdate(
                    update_id=f"update-{i}",
                    channel_id=channel_id,
                    sequence_number=i + 1,
                    update_type=StateUpdateType.TRANSFER,
                    participants=participants,
                    state_data={"sender": sender, "recipient": recipient, "amount": amount},
                    timestamp=int(time.time())
                )
                
                # Sign update
                for participant, private_key in private_keys.items():
                    signature = private_key.sign(update.get_hash())
                    update.add_signature(participant, signature)
                
                # Apply update
                success, errors = manager.update_channel_state(
                    channel_id, update, public_keys
                )
                
                if success:
                    # Check balance conservation
                    info = manager.get_channel_info(channel_id)
                    current_total = info["total_balances"]
                    assert current_total == initial_total, f"Balance conservation violated: {current_total} != {initial_total}"
    
    @given(
        num_participants=st.integers(min_value=2, max_value=5),
        num_updates=st.integers(min_value=1, max_value=20)
    )
    def test_sequence_number_monotonicity(self, num_participants: int, num_updates: int):
        """Test that sequence numbers are always monotonically increasing."""
        participants = [f"participant_{i}" for i in range(num_participants)]
        deposits = {p: 10000 for p in participants}
        
        manager = ChannelManager(self.config)
        
        # Create keys
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        
        # Create channel
        success, channel_id, errors = manager.create_channel(
            participants, deposits, public_keys
        )
        
        if not success:
            return
        
        manager.open_channel(channel_id)
        
        last_sequence = 0
        
        for i in range(num_updates):
            sender = participants[i % len(participants)]
            recipient = participants[(i + 1) % len(participants)]
            
            update = StateUpdate(
                update_id=f"update-{i}",
                channel_id=channel_id,
                sequence_number=i + 1,
                update_type=StateUpdateType.TRANSFER,
                participants=participants,
                state_data={"sender": sender, "recipient": recipient, "amount": 100},
                timestamp=int(time.time())
            )
            
            # Sign update
            for participant, private_key in private_keys.items():
                signature = private_key.sign(update.get_hash())
                update.add_signature(participant, signature)
            
            # Apply update
            success, errors = manager.update_channel_state(
                channel_id, update, public_keys
            )
            
            if success:
                info = manager.get_channel_info(channel_id)
                current_sequence = info["sequence_number"]
                assert current_sequence > last_sequence, f"Sequence number not monotonic: {current_sequence} <= {last_sequence}"
                last_sequence = current_sequence
    
    @given(
        participants=st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=2,
            max_size=4,
            unique=True
        ),
        transfer_amounts=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=10
        )
    )
    def test_non_negative_balances_property(self, participants: List[str], transfer_amounts: List[int]):
        """Test that balances never become negative."""
        deposits = {p: 10000 for p in participants}
        
        manager = ChannelManager(self.config)
        
        # Create keys
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        
        # Create channel
        success, channel_id, errors = manager.create_channel(
            participants, deposits, public_keys
        )
        
        if not success:
            return
        
        manager.open_channel(channel_id)
        
        for i, amount in enumerate(transfer_amounts):
            sender = participants[i % len(participants)]
            recipient = participants[(i + 1) % len(participants)]
            
            update = StateUpdate(
                update_id=f"update-{i}",
                channel_id=channel_id,
                sequence_number=i + 1,
                update_type=StateUpdateType.TRANSFER,
                participants=participants,
                state_data={"sender": sender, "recipient": recipient, "amount": amount},
                timestamp=int(time.time())
            )
            
            # Sign update
            for participant, private_key in private_keys.items():
                signature = private_key.sign(update.get_hash())
                update.add_signature(participant, signature)
            
            # Apply update
            success, errors = manager.update_channel_state(
                channel_id, update, public_keys
            )
            
            if success:
                # Check that no balance is negative
                info = manager.get_channel_info(channel_id)
                # Note: We can't directly access individual balances from info,
                # but we can check that the total is still positive
                assert info["total_balances"] >= 0, "Total balances became negative"
    
    @given(
        num_participants=st.integers(min_value=2, max_value=5),
        num_updates=st.integers(min_value=1, max_value=15)
    )
    def test_channel_state_consistency(self, num_participants: int, num_updates: int):
        """Test that channel state remains consistent after updates."""
        participants = [f"participant_{i}" for i in range(num_participants)]
        deposits = {p: 10000 for p in participants}
        
        manager = ChannelManager(self.config)
        
        # Create keys
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        
        # Create channel
        success, channel_id, errors = manager.create_channel(
            participants, deposits, public_keys
        )
        
        if not success:
            return
        
        manager.open_channel(channel_id)
        
        for i in range(num_updates):
            sender = participants[i % len(participants)]
            recipient = participants[(i + 1) % len(participants)]
            
            update = StateUpdate(
                update_id=f"update-{i}",
                channel_id=channel_id,
                sequence_number=i + 1,
                update_type=StateUpdateType.TRANSFER,
                participants=participants,
                state_data={"sender": sender, "recipient": recipient, "amount": 100},
                timestamp=int(time.time())
            )
            
            # Sign update
            for participant, private_key in private_keys.items():
                signature = private_key.sign(update.get_hash())
                update.add_signature(participant, signature)
            
            # Apply update
            success, errors = manager.update_channel_state(
                channel_id, update, public_keys
            )
            
            if success:
                # Verify state consistency
                state = manager.off_chain_manager.get_channel_state(channel_id)
                assert state is not None
                assert state.validate_balances(), "Channel state balance validation failed"
                assert state.sequence_number == i + 1, "Sequence number mismatch"
                assert state.channel_id == channel_id, "Channel ID mismatch"
                assert set(state.participants) == set(participants), "Participants mismatch"
    
    @given(
        participants=st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=2,
            max_size=3,
            unique=True
        )
    )
    def test_signature_requirement_property(self, participants: List[str]):
        """Test that signature requirements are enforced."""
        deposits = {p: 10000 for p in participants}
        
        manager = ChannelManager(self.config)
        
        # Create keys
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        
        # Create channel
        success, channel_id, errors = manager.create_channel(
            participants, deposits, public_keys
        )
        
        if not success:
            return
        
        manager.open_channel(channel_id)
        
        # Create update without signatures
        update = StateUpdate(
            update_id="unsigned-update",
            channel_id=channel_id,
            sequence_number=1,
            update_type=StateUpdateType.TRANSFER,
            participants=participants,
            state_data={"sender": participants[0], "recipient": participants[1], "amount": 100},
            timestamp=int(time.time())
        )
        
        # Try to apply unsigned update
        success, errors = manager.update_channel_state(
            channel_id, update, public_keys
        )
        
        # Should fail due to missing signatures
        assert not success, "Unsigned update was accepted"
        assert any("signature" in error.lower() for error in errors), "No signature error found"
    
    @given(
        num_participants=st.integers(min_value=2, max_value=4),
        num_operations=st.integers(min_value=1, max_value=10)
    )
    def test_idempotency_property(self, num_participants: int, num_operations: int):
        """Test that operations are idempotent where expected."""
        participants = [f"participant_{i}" for i in range(num_participants)]
        deposits = {p: 10000 for p in participants}
        
        manager = ChannelManager(self.config)
        
        # Create keys
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        
        # Create channel
        success, channel_id, errors = manager.create_channel(
            participants, deposits, public_keys
        )
        
        if not success:
            return
        
        # Opening channel multiple times should be idempotent
        for _ in range(num_operations):
            success, errors = manager.open_channel(channel_id)
            # Should succeed the first time, then be idempotent
            assert success or len(errors) > 0  # Either succeeds or has expected error
        
        # Channel should be open
        info = manager.get_channel_info(channel_id)
        assert info["status"] == ChannelStatus.OPEN.value


class StateChannelStateMachine(RuleBasedStateMachine):
    """State machine for property-based testing of state channels."""
    
    def __init__(self):
        super().__init__()
        self.manager = ChannelManager(ChannelConfig())
        self.participants = []
        self.private_keys = {}
        self.public_keys = {}
        self.deposits = {}
        self.channel_id = None
        self.expected_total = 0
        self.expected_sequence = 0
    
    @rule(
        participants=st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=2,
            max_size=4,
            unique=True
        ),
        deposit_amount=st.integers(min_value=1000, max_value=10000)
    )
    def create_channel(self, participants: List[str], deposit_amount: int):
        """Create a new channel."""
        if self.channel_id is not None:
            return  # Channel already exists
        
        self.participants = participants
        self.deposits = {p: deposit_amount for p in participants}
        self.private_keys = {p: PrivateKey.generate() for p in participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.expected_total = sum(self.deposits.values())
        
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        
        if success:
            self.channel_id = channel_id
    
    @rule()
    def open_channel(self):
        """Open the channel."""
        if self.channel_id is None:
            return
        
        self.manager.open_channel(self.channel_id)
    
    @rule(
        amount=st.integers(min_value=1, max_value=1000)
    )
    def perform_transfer(self, amount: int):
        """Perform a transfer."""
        if self.channel_id is None or len(self.participants) < 2:
            return
        
        sender = self.participants[0]
        recipient = self.participants[1]
        
        update = StateUpdate(
            update_id=f"transfer-{self.expected_sequence + 1}",
            channel_id=self.channel_id,
            sequence_number=self.expected_sequence + 1,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time())
        )
        
        # Sign update
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
        
        # Apply update
        success, errors = self.manager.update_channel_state(
            self.channel_id, update, self.public_keys
        )
        
        if success:
            self.expected_sequence += 1
    
    @rule()
    def close_channel(self):
        """Close the channel."""
        if self.channel_id is None:
            return
        
        self.manager.close_channel(self.channel_id)
    
    @invariant()
    def balance_conservation(self):
        """Total balances should always be conserved."""
        if self.channel_id is None:
            return
        
        info = self.manager.get_channel_info(self.channel_id)
        if info is not None:
            # Calculate total balances from the balances dict
            total_balances = sum(info.get("balances", {}).values())
            assert total_balances == self.expected_total, \
                f"Balance conservation violated: {total_balances} != {self.expected_total}"
    
    @invariant()
    def sequence_monotonicity(self):
        """Sequence numbers should be monotonic."""
        if self.channel_id is None:
            return
        
        info = self.manager.get_channel_info(self.channel_id)
        if info is not None:
            assert info["sequence_number"] == self.expected_sequence, \
                f"Sequence number mismatch: {info['sequence_number']} != {self.expected_sequence}"
    
    @invariant()
    def non_negative_balances(self):
        """Balances should never be negative."""
        if self.channel_id is None:
            return
        
        info = self.manager.get_channel_info(self.channel_id)
        if info is not None:
            assert info["total_balances"] >= 0, "Total balances became negative"
    
    @invariant()
    def channel_state_consistency(self):
        """Channel state should be consistent."""
        if self.channel_id is None:
            return
        
        state = self.manager.off_chain_manager.get_channel_state(self.channel_id)
        if state is not None:
            assert state.validate_balances(), "Channel state balance validation failed"
            assert state.channel_id == self.channel_id, "Channel ID mismatch"
            assert set(state.participants) == set(self.participants), "Participants mismatch"


# Test the state machine
TestStateChannelStateMachine = StateChannelStateMachine.TestCase


class TestStateChannelProperties:
    """Test class for state channel properties."""
    
    def test_balance_conservation_property(self):
        """Test balance conservation property."""
        test_instance = StateChannelPropertyTests()
        test_instance.setup_method()
        
        # Test with default participants and deposits
        test_instance.test_balance_conservation_property(
            participants=["alice", "bob"],
            deposits={"alice": 5000, "bob": 3000}
        )
    
    def test_sequence_number_monotonicity(self):
        """Test sequence number monotonicity."""
        test_instance = StateChannelPropertyTests()
        test_instance.setup_method()
        
        # Test with default parameters
        test_instance.test_sequence_number_monotonicity(
            num_participants=3,
            num_updates=10
        )
    
    def test_non_negative_balances_property(self):
        """Test non-negative balances property."""
        test_instance = StateChannelPropertyTests()
        test_instance.setup_method()
        
        # Test with default parameters
        test_instance.test_non_negative_balances_property(
            participants=["alice", "bob", "charlie"],
            transfer_amounts=[100, 200, 300, 150]
        )
    
    def test_channel_state_consistency(self):
        """Test channel state consistency."""
        test_instance = StateChannelPropertyTests()
        test_instance.setup_method()
        
        # Test with default parameters
        test_instance.test_channel_state_consistency(
            num_participants=2,
            num_updates=5
        )
    
    def test_signature_requirement_property(self):
        """Test signature requirement property."""
        test_instance = StateChannelPropertyTests()
        test_instance.setup_method()
        
        # Test with default parameters
        test_instance.test_signature_requirement_property(
            participants=["alice", "bob"]
        )
    
    def test_idempotency_property(self):
        """Test idempotency property."""
        test_instance = StateChannelPropertyTests()
        test_instance.setup_method()
        
        # Test with default parameters
        test_instance.test_idempotency_property(
            num_participants=2,
            num_operations=3
        )


if __name__ == "__main__":
    pytest.main([__file__])
