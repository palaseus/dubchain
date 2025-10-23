"""
State Channels Demo

This demo showcases the complete state channels implementation including:
- Channel creation and management
- Multi-party state updates
- Dispute resolution
- Security mechanisms
- Performance monitoring
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import time
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dubchain.state_channels.channel_manager import ChannelManager
from dubchain.state_channels.channel_protocol import (
    ChannelConfig,
    StateUpdate,
    StateUpdateType,
    ChannelEvent,
)
from dubchain.crypto.signatures import PrivateKey, PublicKey


class StateChannelsDemo:
    """Comprehensive demo of state channels functionality."""
    
    def __init__(self):
        self.config = ChannelConfig(
            timeout_blocks=1000,
            dispute_period_blocks=100,
            max_participants=10,
            min_deposit=1000,
            state_update_timeout=300,
            require_all_signatures=True,
            enable_fraud_proofs=True,
            enable_timeout_mechanism=True
        )
        self.manager = ChannelManager(self.config)
        self.events_received = []
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for monitoring."""
        def event_handler(event, channel_id):
            self.events_received.append((event, channel_id, time.time()))
            logger.info(f"📢 Event: {event.value} for channel {channel_id}")
        
        for event in ChannelEvent:
            self.manager.add_event_handler(event, event_handler)
    
    def demo_basic_channel_lifecycle(self):
        """Demo basic channel lifecycle."""
        logger.info("🚀 Demo: Basic Channel Lifecycle")
        logger.info("=" * 50)
        
        # Create participants
        participants = ["alice", "bob", "charlie"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {"alice": 10000, "bob": 8000, "charlie": 6000}
        
        logger.info(f"👥 Participants: {participants}")
        logger.info(f"💰 Deposits: {deposits}")
        
        # 1. Create channel
        logger.info("\n1️⃣ Creating channel...")
        success, channel_id, errors = self.manager.create_channel(
            participants, deposits, public_keys
        )
        
        if success:
            logger.info(f"✅ Channel created: {channel_id}")
        else:
            logger.info(f"❌ Channel creation failed: {errors}")
            return
        
        # 2. Open channel
        logger.info("\n2️⃣ Opening channel...")
        success, errors = self.manager.open_channel(channel_id)
        
        if success:
            logger.info("✅ Channel opened successfully")
        else:
            logger.info(f"❌ Channel opening failed: {errors}")
            return
        
        # 3. Perform state updates
        logger.info("\n3️⃣ Performing state updates...")
        
        # Transfer from alice to bob
        update1 = self._create_transfer_update(
            channel_id, 1, "alice", "bob", 1000, participants
        )
        self._sign_update(update1, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update1, public_keys
        )
        
        if success:
            logger.info("✅ Transfer: Alice → Bob (1000)")
        else:
            logger.info(f"❌ Transfer failed: {errors}")
        
        # Transfer from bob to charlie
        update2 = self._create_transfer_update(
            channel_id, 2, "bob", "charlie", 500, participants
        )
        self._sign_update(update2, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update2, public_keys
        )
        
        if success:
            logger.info("✅ Transfer: Bob → Charlie (500)")
        else:
            logger.info(f"❌ Transfer failed: {errors}")
        
        # Multi-party transfer
        update3 = self._create_multi_party_update(
            channel_id, 3, [
                {"sender": "alice", "recipient": "bob", "amount": 200},
                {"sender": "charlie", "recipient": "alice", "amount": 100}
            ], participants
        )
        self._sign_update(update3, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update3, public_keys
        )
        
        if success:
            logger.info("✅ Multi-party transfer completed")
        else:
            logger.info(f"❌ Multi-party transfer failed: {errors}")
        
        # 4. Show channel info
        logger.info("\n4️⃣ Channel Information:")
        info = self.manager.get_channel_info(channel_id)
        self._print_channel_info(info)
        
        # 5. Close channel
        logger.info("\n5️⃣ Closing channel...")
        success, errors = self.manager.close_channel(channel_id, "cooperative")
        
        if success:
            logger.info("✅ Channel closed cooperatively")
        else:
            logger.info(f"❌ Channel closure failed: {errors}")
        
        return channel_id
    
    def demo_dispute_resolution(self):
        """Demo dispute resolution process."""
        logger.info("\n⚖️ Demo: Dispute Resolution")
        logger.info("=" * 50)
        
        # Create participants
        participants = ["alice", "bob"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {"alice": 10000, "bob": 8000}
        
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            participants, deposits, public_keys
        )
        assert success
        
        self.manager.open_channel(channel_id)
        
        # Perform some updates
        for i in range(3):
            update = self._create_transfer_update(
                channel_id, i + 1, "alice", "bob", 1000, participants
            )
            self._sign_update(update, private_keys)
            self.manager.update_channel_state(channel_id, update, public_keys)
        
        logger.info(f"📊 Channel state after {3} updates:")
        info = self.manager.get_channel_info(channel_id)
        self._print_channel_info(info)
        
        # Initiate dispute
        logger.info("\n🚨 Initiating dispute...")
        evidence = {
            "type": "state_disagreement",
            "disputed_sequence": 2,
            "reason": "Bob claims Alice's last update is invalid"
        }
        
        success, dispute_id, errors = self.manager.initiate_dispute(
            channel_id, "bob", "Invalid state update", evidence
        )
        
        if success:
            logger.info(f"✅ Dispute initiated: {dispute_id}")
        else:
            logger.info(f"❌ Dispute initiation failed: {errors}")
            return
        
        # Show dispute information
        dispute = self.manager.dispute_manager.contract.get_dispute(dispute_id)
        if dispute:
            logger.info(f"📋 Dispute Status: {dispute.status.value}")
            logger.info(f"📋 Evidence Period End: {dispute.evidence_period_end}")
            logger.info(f"📋 Challenge Period End: {dispute.challenge_period_end}")
        
        # Submit additional evidence
        logger.info("\n📝 Submitting additional evidence...")
        from dubchain.state_channels.dispute_resolution import DisputeEvidence
        
        evidence_update = DisputeEvidence(
            evidence_id="evidence-1",
            channel_id=channel_id,
            submitter="alice",
            evidence_type="counter_evidence",
            evidence_data={"counter_claim": "State update is valid"},
            timestamp=int(time.time())
        )
        
        success = self.manager.dispute_manager.submit_evidence(
            dispute_id, evidence_update, public_keys["alice"]
        )
        
        if success:
            logger.info("✅ Evidence submitted successfully")
        else:
            logger.info("❌ Evidence submission failed")
        
        # Resolve dispute
        logger.info("\n🔧 Resolving dispute...")
        channel_state = self.manager.off_chain_manager.get_channel_state(channel_id)
        
        success = self.manager.dispute_manager.resolve_dispute(
            dispute_id, channel_state, "Resolved in favor of Alice"
        )
        
        if success:
            logger.info("✅ Dispute resolved successfully")
        else:
            logger.info("❌ Dispute resolution failed")
        
        return channel_id
    
    def demo_security_mechanisms(self):
        """Demo security mechanisms."""
        logger.info("\n🔒 Demo: Security Mechanisms")
        logger.info("=" * 50)
        
        # Create participants
        participants = ["alice", "bob"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {"alice": 10000, "bob": 8000}
        
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            participants, deposits, public_keys
        )
        assert success
        
        self.manager.open_channel(channel_id)
        
        # Test 1: Replay attack prevention
        logger.info("\n🛡️ Test 1: Replay Attack Prevention")
        
        # Create legitimate update
        update = self._create_transfer_update(
            channel_id, 1, "alice", "bob", 1000, participants
        )
        self._sign_update(update, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, public_keys
        )
        logger.info(f"✅ Legitimate update: {'Success' if success else 'Failed'}")
        
        # Try to replay the same update
        success, errors = self.manager.update_channel_state(
            channel_id, update, public_keys
        )
        logger.info(f"🚫 Replay attempt: {'Blocked' if not success else 'Allowed'}")
        
        # Test 2: Insufficient balance protection
        logger.info("\n🛡️ Test 2: Insufficient Balance Protection")
        
        # Try to transfer more than available
        update2 = self._create_transfer_update(
            channel_id, 2, "alice", "bob", 20000, participants  # More than alice has
        )
        self._sign_update(update2, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update2, public_keys
        )
        logger.info(f"🚫 Overspend attempt: {'Blocked' if not success else 'Allowed'}")
        
        # Test 3: Invalid signature detection
        logger.info("\n🛡️ Test 3: Invalid Signature Detection")
        
        # Create update with invalid signature
        update3 = self._create_transfer_update(
            channel_id, 2, "alice", "bob", 1000, participants
        )
        
        # Sign with wrong private key
        wrong_key = PrivateKey.generate()
        signature = wrong_key.sign(update3.get_hash())
        update3.add_signature("alice", signature)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update3, public_keys
        )
        logger.info(f"🚫 Invalid signature: {'Blocked' if not success else 'Allowed'}")
        
        # Test 4: Sequence number validation
        logger.info("\n🛡️ Test 4: Sequence Number Validation")
        
        # Try to skip sequence numbers
        update4 = self._create_transfer_update(
            channel_id, 5, "alice", "bob", 1000, participants  # Skip to sequence 5
        )
        self._sign_update(update4, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update4, public_keys
        )
        logger.info(f"🚫 Sequence skip: {'Blocked' if not success else 'Allowed'}")
        
        # Show security statistics
        logger.info("\n📊 Security Statistics:")
        security = self.manager.security_manager.get_channel_security(channel_id)
        security_events = security.get_security_events()
        fraud_proofs = security.get_fraud_proofs()
        
        logger.info(f"  Security Events: {len(security_events)}")
        logger.info(f"  Fraud Proofs: {len(fraud_proofs)}")
        
        return channel_id
    
    def demo_performance_monitoring(self):
        """Demo performance monitoring."""
        logger.info("\n📈 Demo: Performance Monitoring")
        logger.info("=" * 50)
        
        # Create participants
        participants = ["alice", "bob", "charlie", "dave"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {p: 100000 for p in participants}  # Large deposits for performance testing
        
        # Create and open channel
        success, channel_id, errors = self.manager.create_channel(
            participants, deposits, public_keys
        )
        assert success
        
        self.manager.open_channel(channel_id)
        
        # Perform many updates and measure performance
        logger.info("🏃‍♂️ Performing 100 state updates...")
        
        start_time = time.time()
        
        for i in range(100):
            update = self._create_transfer_update(
                channel_id, i + 1, 
                participants[i % len(participants)], 
                participants[(i + 1) % len(participants)], 
                100, participants
            )
            self._sign_update(update, private_keys)
            
            success, errors = self.manager.update_channel_state(
                channel_id, update, public_keys
            )
            
            if not success and i % 10 == 0:  # Log every 10th failure
                logger.info(f"  Update {i} failed: {errors}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"✅ Completed 100 updates in {total_time:.2f} seconds")
        logger.info(f"📊 Average time per update: {total_time / 100:.4f} seconds")
        logger.info(f"📊 Throughput: {100 / total_time:.2f} updates/second")
        
        # Show performance metrics
        metrics = self.manager.get_channel_metrics(channel_id)
        if metrics:
            logger.info(f"\n📊 Channel Metrics:")
            logger.info(f"  Total Updates: {metrics.total_updates}")
            logger.info(f"  Successful Updates: {metrics.successful_updates}")
            logger.info(f"  Failed Updates: {metrics.failed_updates}")
            logger.info(f"  Success Rate: {metrics.get_success_rate():.2%}")
            logger.info(f"  Average Update Time: {metrics.average_update_time:.4f}s")
            logger.info(f"  Total Volume: {metrics.total_volume}")
        
        # Show global metrics
        global_metrics = self.manager.get_global_metrics()
        logger.info(f"\n📊 Global Metrics:")
        logger.info(f"  Active Channels: {global_metrics['total_channels']}")
        logger.info(f"  Total Updates: {global_metrics['total_updates']}")
        logger.info(f"  Total Volume: {global_metrics['total_volume']}")
        logger.info(f"  Average Success Rate: {global_metrics['average_success_rate']:.2%}")
        
        return channel_id
    
    def demo_multi_channel_operations(self):
        """Demo operations across multiple channels."""
        logger.info("\n🔄 Demo: Multi-Channel Operations")
        logger.info("=" * 50)
        
        # Create participants
        participants = ["alice", "bob", "charlie"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {p: 10000 for p in participants}
        
        # Create multiple channels
        channel_ids = []
        for i in range(5):
            success, channel_id, errors = self.manager.create_channel(
                participants, deposits, public_keys
            )
            if success:
                channel_ids.append(channel_id)
                self.manager.open_channel(channel_id)
                logger.info(f"✅ Created channel {i + 1}: {channel_id}")
        
        logger.info(f"\n📊 Created {len(channel_ids)} channels")
        
        # Perform operations across all channels
        logger.info("\n🔄 Performing operations across all channels...")
        
        for i, channel_id in enumerate(channel_ids):
            # Perform some updates
            for j in range(3):
                update = self._create_transfer_update(
                    channel_id, j + 1,
                    participants[j % len(participants)],
                    participants[(j + 1) % len(participants)],
                    100, participants
                )
                self._sign_update(update, private_keys)
                
                success, errors = self.manager.update_channel_state(
                    channel_id, update, public_keys
                )
                
                if not success:
                    logger.info(f"  Channel {i + 1}, Update {j + 1} failed")
        
        # Show statistics for all channels
        logger.info("\n📊 Multi-Channel Statistics:")
        for i, channel_id in enumerate(channel_ids):
            info = self.manager.get_channel_info(channel_id)
            if info:
                logger.info(f"  Channel {i + 1}: {info['sequence_number']} updates, "
                      f"Status: {info['status']}")
        
        # Show global statistics
        global_stats = self.manager.get_channel_statistics()
        logger.info(f"\n📊 Global Statistics:")
        logger.info(f"  Total Channels: {global_stats['active_channels']}")
        logger.info(f"  Total Participants: {global_stats['total_participants']}")
        
        return channel_ids
    
    def _create_transfer_update(self, channel_id, sequence, sender, recipient, amount, participants):
        """Create a transfer state update."""
        return StateUpdate(
            update_id=f"transfer-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.TRANSFER,
            participants=participants,
            state_data={"sender": sender, "recipient": recipient, "amount": amount},
            timestamp=int(time.time())
        )
    
    def _create_multi_party_update(self, channel_id, sequence, transfers, participants):
        """Create a multi-party state update."""
        return StateUpdate(
            update_id=f"multi-party-{sequence}",
            channel_id=channel_id,
            sequence_number=sequence,
            update_type=StateUpdateType.MULTI_PARTY,
            participants=participants,
            state_data={"transfers": transfers},
            timestamp=int(time.time())
        )
    
    def _sign_update(self, update, private_keys):
        """Sign a state update with all participant keys."""
        for participant, private_key in private_keys.items():
            signature = private_key.sign(update.get_hash())
            update.add_signature(participant, signature)
    
    def _print_channel_info(self, info):
        """Print formatted channel information."""
        if not info:
            logger.info("  No channel information available")
            return
        
        logger.info(f"  Channel ID: {info['channel_id']}")
        logger.info(f"  Status: {info['status']}")
        logger.info(f"  Participants: {info['participants']}")
        logger.info(f"  Total Deposits: {info['total_deposits']}")
        logger.info(f"  Total Balances: {info['total_balances']}")
        logger.info(f"  Sequence Number: {info['sequence_number']}")
        logger.info(f"  Created At: {info['created_at']}")
        if info.get('opened_at'):
            logger.info(f"  Opened At: {info['opened_at']}")
        if info.get('closed_at'):
            logger.info(f"  Closed At: {info['closed_at']}")
        if info.get('close_reason'):
            logger.info(f"  Close Reason: {info['close_reason']}")
    
    def run_complete_demo(self):
        """Run the complete demo."""
        logger.info("🎯 State Channels Complete Demo")
        logger.info("=" * 60)
        
        try:
            # Run all demos
            self.demo_basic_channel_lifecycle()
            self.demo_dispute_resolution()
            self.demo_security_mechanisms()
            self.demo_performance_monitoring()
            self.demo_multi_channel_operations()
            
            # Show event summary
            logger.info("\n📋 Event Summary:")
            logger.info("=" * 50)
            event_counts = {}
            for event, channel_id, timestamp in self.events_received:
                event_counts[event.value] = event_counts.get(event.value, 0) + 1
            
            for event_type, count in event_counts.items():
                logger.info(f"  {event_type}: {count}")
            
            logger.info(f"\n✅ Demo completed successfully!")
            logger.info(f"📊 Total events received: {len(self.events_received)}")
            
        except Exception as e:
            logger.info(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo."""
    demo = StateChannelsDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
