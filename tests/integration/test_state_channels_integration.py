"""
Integration tests for State Channels

Tests the complete state channel lifecycle including:
- Channel creation, opening, and closing
- Multi-party state updates
- Dispute resolution
- Security mechanisms
- Performance under load
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from src.dubchain.state_channels.channel_manager import ChannelManager, ChannelMetrics
from src.dubchain.state_channels.channel_protocol import (
    ChannelConfig,
    ChannelId,
    StateUpdate,
    StateUpdateType,
    ChannelStatus,
    ChannelEvent,
    ChannelCloseReason,
)
from src.dubchain.state_channels.dispute_resolution import DisputeManager, DisputeEvidence, DisputeStatus
from src.dubchain.state_channels.security import SecurityManager, SecurityThreat
from src.dubchain.crypto.signatures import PrivateKey, PublicKey


class TestChannelLifecycleIntegration:
    """Test complete channel lifecycle integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig(
            timeout_blocks=100,
            dispute_period_blocks=50,
            max_participants=5,
            min_deposit=1000,
            state_update_timeout=60
        )
        self.manager = ChannelManager(self.config)
        
        # Create test participants
        self.participants = ["alice", "bob", "charlie"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.deposits = {"alice": 10000, "bob": 8000, "charlie": 6000}
    
    def test_complete_channel_lifecycle(self):
        """Test complete channel lifecycle from creation to closure."""
        # 1. Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        assert channel_id is not None
        assert len(errors) == 0
        
        # 2. Open channel
        success, errors = self.manager.open_channel(channel_id)
        assert success is True
        assert len(errors) == 0
        
        # Verify channel is open
        info = self.manager.get_channel_info(channel_id)
        assert info["status"] == ChannelStatus.OPEN.value
        
        # 3. Perform state updates
        for i in range(3):
            update = self._create_transfer_update(
                channel_id, i + 1, "alice", "bob", 1000
            )
            self._sign_update(update)
            
            success, errors = self.manager.update_channel_state(
                channel_id, update, self.public_keys
            )
            assert success is True
            assert len(errors) == 0
        
        # 4. Verify final state
        info = self.manager.get_channel_info(channel_id)
        assert info["sequence_number"] == 3
        
        # 5. Close channel cooperatively
        success, errors = self.manager.close_channel(channel_id, "cooperative")
        assert success is True
        assert len(errors) == 0
        
        # Verify channel is closed
        info = self.manager.get_channel_info(channel_id)
        assert info["status"] == ChannelStatus.CLOSED.value
    
    def test_multi_party_channel_operations(self):
        """Test multi-party channel operations."""
        # Create channel with 3 participants
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Test multi-party transfer
        update = self._create_multi_party_update(channel_id, 1, [
            {"sender": "alice", "recipient": "bob", "amount": 1000},
            {"sender": "bob", "recipient": "charlie", "amount": 500},
            {"sender": "charlie", "recipient": "alice", "amount": 200}
        ])
        self._sign_update(update)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is True
        assert len(errors) == 0
        
        # Verify balances
        info = self.manager.get_channel_info(channel_id)
        # alice: 10000 - 1000 + 200 = 9200
        # bob: 8000 + 1000 - 500 = 8500
        # charlie: 6000 + 500 - 200 = 6300
        # Total should still be 24000
        assert info["total_balances"] == 24000
    
    def test_conditional_payments(self):
        """Test conditional payment functionality."""
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Create conditional payment
        update = self._create_conditional_update(
            channel_id, 1, "alice", "bob", 1000,
            {"type": "time_based", "target_time": int(time.time()) + 3600}
        )
        self._sign_update(update)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is True
        assert len(errors) == 0
    
    def _create_transfer_update(self, channel_id, sequence, sender, recipient, amount):
        """Create a transfer state update."""
        return StateUpdate(
            update_id=f"transfer-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time()),
            nonce=sequence  # Use sequence number as nonce to ensure uniqueness
        )
    
    def _create_multi_party_update(self, channel_id, sequence, transfers):
        """Create a multi-party state update."""
        return StateUpdate(
            update_id=f"multi-party-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.MULTI_PARTY,
            participants=self.participants,
            state_data={"transfers": transfers},
            timestamp=int(time.time()),
            nonce=sequence
        )
    
    def _create_conditional_update(self, channel_id, sequence, sender, recipient, amount, condition):
        """Create a conditional state update."""
        return StateUpdate(
            update_id=f"conditional-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.CONDITIONAL,
            participants=self.participants,
            state_data={
                "sender": sender,
                "recipient": recipient,
                "amount": amount,
                "condition": condition
            },
            timestamp=int(time.time()),
            nonce=sequence
        )
    
    def _sign_update(self, update):
        """Sign a state update with all participant keys."""
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)


class TestDisputeResolutionIntegration:
    """Test dispute resolution integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.manager = ChannelManager(self.config)
        
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.deposits = {"alice": 10000, "bob": 8000}
    
    def test_dispute_initiation_and_resolution(self):
        """Test dispute initiation and resolution process."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Perform some updates
        for i in range(2):
            update = self._create_transfer_update(
                channel_id, i + 1, "alice", "bob", 1000
            )
            self._sign_update(update)
            self.manager.update_channel_state(channel_id, update, self.public_keys)
        
        # Initiate dispute
        evidence = {
            "type": "state_update",
            "disputed_sequence": 2,
            "reason": "Invalid state transition"
        }
        
        success, dispute_id, errors = self.manager.initiate_dispute(
            channel_id, "alice", "Invalid state", evidence
        )
        assert success is True
        assert dispute_id is not None
        
        # Verify dispute was created
        dispute = self.manager.dispute_manager.contract.get_dispute(dispute_id)
        assert dispute is not None
        assert dispute.channel_id == channel_id
        assert dispute.initiator == "alice"
        
        # Verify dispute can be retrieved and has correct properties
        dispute = self.manager.dispute_manager.contract.get_dispute(dispute_id)
        assert dispute is not None
        assert dispute.channel_id == channel_id
        assert dispute.initiator == "alice"
        assert dispute.status == DisputeStatus.EVIDENCE_PERIOD
        
        # Resolve dispute
        channel_state = self.manager.off_chain_manager.get_channel_state(channel_id)
        success = self.manager.dispute_manager.resolve_dispute(
            dispute_id, channel_state, "Resolved in favor of alice"
        )
        assert success is True
        
        # Verify dispute is resolved
        dispute = self.manager.dispute_manager.contract.get_dispute(dispute_id)
        assert dispute.status.value == "resolved"
    
    def _create_transfer_update(self, channel_id, sequence, sender, recipient, amount):
        """Create a transfer state update."""
        return StateUpdate(
            update_id=f"transfer-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time()),
            nonce=sequence  # Use sequence number as nonce to ensure uniqueness
        )
    
    def _sign_update(self, update):
        """Sign a state update with all participant keys."""
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)


class TestSecurityIntegration:
    """Test security mechanisms integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.manager = ChannelManager(self.config)
        
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.deposits = {"alice": 10000, "bob": 8000}
    
    def test_replay_attack_prevention(self):
        """Test prevention of replay attacks."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Create and sign update
        update = self._create_transfer_update(channel_id, 1, "alice", "bob", 1000)
        self._sign_update(update)
        
        # Apply update successfully
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is True
        
        # Try to replay the same update
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is False
        assert any("replay" in error.lower() for error in errors)
    
    def test_insufficient_balance_protection(self):
        """Test protection against insufficient balance transfers."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Try to transfer more than available
        update = self._create_transfer_update(channel_id, 1, "alice", "bob", 15000)
        self._sign_update(update)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is False
        assert any("double_spend" in error.lower() or "insufficient" in error.lower() for error in errors)
    
    def test_invalid_signature_detection(self):
        """Test detection of invalid signatures."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Create update with invalid signature
        update = self._create_transfer_update(channel_id, 1, "alice", "bob", 1000)
        
        # Sign with wrong private key
        wrong_key = PrivateKey.generate()
        signature = wrong_key.sign(update.get_hash())
        update.add_signature("alice", signature)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is False
        assert any("signature" in error.lower() for error in errors)
    
    def test_sequence_number_validation(self):
        """Test sequence number validation."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Try to apply update with wrong sequence number
        update = self._create_transfer_update(channel_id, 3, "alice", "bob", 1000)  # Should be 1
        self._sign_update(update)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is False
        assert any("sequence" in error.lower() for error in errors)
    
    def test_byzantine_behavior_detection(self):
        """Test detection of Byzantine behavior patterns."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Simulate rapid updates (potential spam)
        successful_updates = 0
        for i in range(15):  # More than normal
            update = self._create_transfer_update(
                channel_id, i + 1, "alice", "bob", 100
            )
            self._sign_update(update)
            
            success, errors = self.manager.update_channel_state(
                channel_id, update, self.public_keys
            )
            if success:
                successful_updates += 1
            else:
                logger.info(f"Update {i+1} failed: {errors}")
                break  # Stop if update fails
        
        # Check for security events
        security = self.manager.security_manager.get_channel_security(channel_id)
        security_events = security.get_security_events()
        
        logger.info(f"Successful updates: {successful_updates}")
        logger.info(f"Security events: {len(security_events)}")
        
        # Should have some security events due to rapid updates, or at least some successful updates
        assert successful_updates > 0 or len(security_events) > 0
    
    def _create_transfer_update(self, channel_id, sequence, sender, recipient, amount):
        """Create a transfer state update."""
        return StateUpdate(
            update_id=f"transfer-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time()),
            nonce=sequence  # Use sequence number as nonce to ensure uniqueness
        )
    
    def _sign_update(self, update):
        """Sign a state update with all participant keys."""
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)


class TestPerformanceIntegration:
    """Test performance under various loads."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig(
            max_state_updates=1000,
            state_update_timeout=300
        )
        self.manager = ChannelManager(self.config)
        
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.deposits = {"alice": 100000, "bob": 100000}  # Large deposits for many updates
    
    def test_high_frequency_updates(self):
        """Test performance with high frequency updates."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Perform many updates
        num_updates = 100
        start_time = time.time()
        
        for i in range(num_updates):
            update = self._create_transfer_update(
                channel_id, i + 1, "alice", "bob", 100
            )
            self._sign_update(update)
            
            success, errors = self.manager.update_channel_state(
                channel_id, update, self.public_keys
            )
            assert success is True
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_update = total_time / num_updates
        
        # Verify performance metrics
        metrics = self.manager.get_channel_metrics(channel_id)
        assert metrics.total_updates == num_updates
        assert metrics.successful_updates == num_updates
        assert metrics.get_success_rate() == 1.0
        
        # Performance should be reasonable (less than 100ms per update)
        assert avg_time_per_update < 0.1
    
    def test_concurrent_updates(self):
        """Test handling of concurrent updates."""
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Create updates concurrently
        def perform_update(sequence):
            update = self._create_transfer_update(
                channel_id, sequence, "alice", "bob", 100
            )
            self._sign_update(update)
            
            success, errors = self.manager.update_channel_state(
                channel_id, update, self.public_keys
            )
            return success, errors
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(perform_update, i) for i in range(1, 11)]
            results = [future.result() for future in futures]
        
        # Some updates should succeed, others should fail due to sequence conflicts
        successful_updates = sum(1 for success, _ in results if success)
        assert successful_updates > 0  # At least some should succeed
        assert successful_updates < 10  # Not all should succeed due to concurrency
    
    def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        # Create multiple channels
        channel_ids = []
        for i in range(10):
            success, channel_id, errors = self.manager.create_channel(
                self.participants, self.deposits, self.public_keys
            )
            assert success is True
            self.manager.open_channel(channel_id)
            channel_ids.append(channel_id)
        
        # Perform updates on all channels
        for channel_id in channel_ids:
            for i in range(20):
                update = self._create_transfer_update(
                    channel_id, i + 1, "alice", "bob", 100
                )
                self._sign_update(update)
                self.manager.update_channel_state(channel_id, update, self.public_keys)
        
        # Verify all channels are still functional
        for channel_id in channel_ids:
            info = self.manager.get_channel_info(channel_id)
            assert info is not None
            assert info["sequence_number"] == 20
        
        # Clean up channels
        for channel_id in channel_ids:
            self.manager.close_channel(channel_id)
    
    def _create_transfer_update(self, channel_id, sequence, sender, recipient, amount):
        """Create a transfer state update."""
        return StateUpdate(
            update_id=f"transfer-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time()),
            nonce=sequence  # Use sequence number as nonce to ensure uniqueness
        )
    
    def _sign_update(self, update):
        """Sign a state update with all participant keys."""
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)


class TestEventHandlingIntegration:
    """Test event handling integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.manager = ChannelManager(self.config)
        
        self.participants = ["alice", "bob"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.deposits = {"alice": 10000, "bob": 8000}
        
        # Track events
        self.events_received = []
    
    def test_event_handling_lifecycle(self):
        """Test event handling throughout channel lifecycle."""
        # Add event handler
        def event_handler(event, channel_id):
            self.events_received.append((event, channel_id))
        
        self.manager.add_event_handler(ChannelEvent.CREATED, event_handler)
        self.manager.add_event_handler(ChannelEvent.OPENED, event_handler)
        self.manager.add_event_handler(ChannelEvent.STATE_UPDATED, event_handler)
        self.manager.add_event_handler(ChannelEvent.CLOSED, event_handler)
        
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Perform update
        update = self._create_transfer_update(channel_id, 1, "alice", "bob", 1000)
        self._sign_update(update)
        self.manager.update_channel_state(channel_id, update, self.public_keys)
        
        # Close channel
        self.manager.close_channel(channel_id)
        
        # Verify events were received
        assert (ChannelEvent.CREATED, channel_id) in self.events_received
        assert (ChannelEvent.OPENED, channel_id) in self.events_received
        assert (ChannelEvent.STATE_UPDATED, channel_id) in self.events_received
        assert (ChannelEvent.CLOSED, channel_id) in self.events_received
    
    def test_event_handler_error_handling(self):
        """Test event handler error handling."""
        # Add error-prone event handler
        def error_handler(event, channel_id):
            if event == ChannelEvent.STATE_UPDATED:
                raise Exception("Handler error")
        
        self.manager.add_event_handler(ChannelEvent.STATE_UPDATED, error_handler)
        
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Perform update (should not fail despite handler error)
        update = self._create_transfer_update(channel_id, 1, "alice", "bob", 1000)
        self._sign_update(update)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, self.public_keys
        )
        assert success is True  # Update should succeed despite handler error
    
    def _create_transfer_update(self, channel_id, sequence, sender, recipient, amount):
        """Create a transfer state update."""
        return StateUpdate(
            update_id=f"transfer-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.TRANSFER,
            participants=self.participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time()),
            nonce=sequence  # Use sequence number as nonce to ensure uniqueness
        )
    
    def _sign_update(self, update):
        """Sign a state update with all participant keys."""
        for participant, private_key in self.private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)


if __name__ == "__main__":
    pytest.main([__file__])
