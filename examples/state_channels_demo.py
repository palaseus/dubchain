"""
State Channels Demo

This demo showcases the complete state channels implementation including:
- Channel creation and management
- Multi-party state updates
- Dispute resolution
- Security mechanisms
- Performance monitoring
"""

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
            print(f"📢 Event: {event.value} for channel {channel_id}")
        
        for event in ChannelEvent:
            self.manager.add_event_handler(event, event_handler)
    
    def demo_basic_channel_lifecycle(self):
        """Demo basic channel lifecycle."""
        print("🚀 Demo: Basic Channel Lifecycle")
        print("=" * 50)
        
        # Create participants
        participants = ["alice", "bob", "charlie"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {"alice": 10000, "bob": 8000, "charlie": 6000}
        
        print(f"👥 Participants: {participants}")
        print(f"💰 Deposits: {deposits}")
        
        # 1. Create channel
        print("\n1️⃣ Creating channel...")
        success, channel_id, errors = self.manager.create_channel(
            participants, deposits, public_keys
        )
        
        if success:
            print(f"✅ Channel created: {channel_id}")
        else:
            print(f"❌ Channel creation failed: {errors}")
            return
        
        # 2. Open channel
        print("\n2️⃣ Opening channel...")
        success, errors = self.manager.open_channel(channel_id)
        
        if success:
            print("✅ Channel opened successfully")
        else:
            print(f"❌ Channel opening failed: {errors}")
            return
        
        # 3. Perform state updates
        print("\n3️⃣ Performing state updates...")
        
        # Transfer from alice to bob
        update1 = self._create_transfer_update(
            channel_id, 1, "alice", "bob", 1000, participants
        )
        self._sign_update(update1, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update1, public_keys
        )
        
        if success:
            print("✅ Transfer: Alice → Bob (1000)")
        else:
            print(f"❌ Transfer failed: {errors}")
        
        # Transfer from bob to charlie
        update2 = self._create_transfer_update(
            channel_id, 2, "bob", "charlie", 500, participants
        )
        self._sign_update(update2, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update2, public_keys
        )
        
        if success:
            print("✅ Transfer: Bob → Charlie (500)")
        else:
            print(f"❌ Transfer failed: {errors}")
        
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
            print("✅ Multi-party transfer completed")
        else:
            print(f"❌ Multi-party transfer failed: {errors}")
        
        # 4. Show channel info
        print("\n4️⃣ Channel Information:")
        info = self.manager.get_channel_info(channel_id)
        self._print_channel_info(info)
        
        # 5. Close channel
        print("\n5️⃣ Closing channel...")
        success, errors = self.manager.close_channel(channel_id, "cooperative")
        
        if success:
            print("✅ Channel closed cooperatively")
        else:
            print(f"❌ Channel closure failed: {errors}")
        
        return channel_id
    
    def demo_dispute_resolution(self):
        """Demo dispute resolution process."""
        print("\n⚖️ Demo: Dispute Resolution")
        print("=" * 50)
        
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
        
        print(f"📊 Channel state after {3} updates:")
        info = self.manager.get_channel_info(channel_id)
        self._print_channel_info(info)
        
        # Initiate dispute
        print("\n🚨 Initiating dispute...")
        evidence = {
            "type": "state_disagreement",
            "disputed_sequence": 2,
            "reason": "Bob claims Alice's last update is invalid"
        }
        
        success, dispute_id, errors = self.manager.initiate_dispute(
            channel_id, "bob", "Invalid state update", evidence
        )
        
        if success:
            print(f"✅ Dispute initiated: {dispute_id}")
        else:
            print(f"❌ Dispute initiation failed: {errors}")
            return
        
        # Show dispute information
        dispute = self.manager.dispute_manager.contract.get_dispute(dispute_id)
        if dispute:
            print(f"📋 Dispute Status: {dispute.status.value}")
            print(f"📋 Evidence Period End: {dispute.evidence_period_end}")
            print(f"📋 Challenge Period End: {dispute.challenge_period_end}")
        
        # Submit additional evidence
        print("\n📝 Submitting additional evidence...")
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
            print("✅ Evidence submitted successfully")
        else:
            print("❌ Evidence submission failed")
        
        # Resolve dispute
        print("\n🔧 Resolving dispute...")
        channel_state = self.manager.off_chain_manager.get_channel_state(channel_id)
        
        success = self.manager.dispute_manager.resolve_dispute(
            dispute_id, channel_state, "Resolved in favor of Alice"
        )
        
        if success:
            print("✅ Dispute resolved successfully")
        else:
            print("❌ Dispute resolution failed")
        
        return channel_id
    
    def demo_security_mechanisms(self):
        """Demo security mechanisms."""
        print("\n🔒 Demo: Security Mechanisms")
        print("=" * 50)
        
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
        print("\n🛡️ Test 1: Replay Attack Prevention")
        
        # Create legitimate update
        update = self._create_transfer_update(
            channel_id, 1, "alice", "bob", 1000, participants
        )
        self._sign_update(update, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update, public_keys
        )
        print(f"✅ Legitimate update: {'Success' if success else 'Failed'}")
        
        # Try to replay the same update
        success, errors = self.manager.update_channel_state(
            channel_id, update, public_keys
        )
        print(f"🚫 Replay attempt: {'Blocked' if not success else 'Allowed'}")
        
        # Test 2: Insufficient balance protection
        print("\n🛡️ Test 2: Insufficient Balance Protection")
        
        # Try to transfer more than available
        update2 = self._create_transfer_update(
            channel_id, 2, "alice", "bob", 20000, participants  # More than alice has
        )
        self._sign_update(update2, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update2, public_keys
        )
        print(f"🚫 Overspend attempt: {'Blocked' if not success else 'Allowed'}")
        
        # Test 3: Invalid signature detection
        print("\n🛡️ Test 3: Invalid Signature Detection")
        
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
        print(f"🚫 Invalid signature: {'Blocked' if not success else 'Allowed'}")
        
        # Test 4: Sequence number validation
        print("\n🛡️ Test 4: Sequence Number Validation")
        
        # Try to skip sequence numbers
        update4 = self._create_transfer_update(
            channel_id, 5, "alice", "bob", 1000, participants  # Skip to sequence 5
        )
        self._sign_update(update4, private_keys)
        
        success, errors = self.manager.update_channel_state(
            channel_id, update4, public_keys
        )
        print(f"🚫 Sequence skip: {'Blocked' if not success else 'Allowed'}")
        
        # Show security statistics
        print("\n📊 Security Statistics:")
        security = self.manager.security_manager.get_channel_security(channel_id)
        security_events = security.get_security_events()
        fraud_proofs = security.get_fraud_proofs()
        
        print(f"  Security Events: {len(security_events)}")
        print(f"  Fraud Proofs: {len(fraud_proofs)}")
        
        return channel_id
    
    def demo_performance_monitoring(self):
        """Demo performance monitoring."""
        print("\n📈 Demo: Performance Monitoring")
        print("=" * 50)
        
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
        print("🏃‍♂️ Performing 100 state updates...")
        
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
                print(f"  Update {i} failed: {errors}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Completed 100 updates in {total_time:.2f} seconds")
        print(f"📊 Average time per update: {total_time / 100:.4f} seconds")
        print(f"📊 Throughput: {100 / total_time:.2f} updates/second")
        
        # Show performance metrics
        metrics = self.manager.get_channel_metrics(channel_id)
        if metrics:
            print(f"\n📊 Channel Metrics:")
            print(f"  Total Updates: {metrics.total_updates}")
            print(f"  Successful Updates: {metrics.successful_updates}")
            print(f"  Failed Updates: {metrics.failed_updates}")
            print(f"  Success Rate: {metrics.get_success_rate():.2%}")
            print(f"  Average Update Time: {metrics.average_update_time:.4f}s")
            print(f"  Total Volume: {metrics.total_volume}")
        
        # Show global metrics
        global_metrics = self.manager.get_global_metrics()
        print(f"\n📊 Global Metrics:")
        print(f"  Active Channels: {global_metrics['total_channels']}")
        print(f"  Total Updates: {global_metrics['total_updates']}")
        print(f"  Total Volume: {global_metrics['total_volume']}")
        print(f"  Average Success Rate: {global_metrics['average_success_rate']:.2%}")
        
        return channel_id
    
    def demo_multi_channel_operations(self):
        """Demo operations across multiple channels."""
        print("\n🔄 Demo: Multi-Channel Operations")
        print("=" * 50)
        
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
                print(f"✅ Created channel {i + 1}: {channel_id}")
        
        print(f"\n📊 Created {len(channel_ids)} channels")
        
        # Perform operations across all channels
        print("\n🔄 Performing operations across all channels...")
        
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
                    print(f"  Channel {i + 1}, Update {j + 1} failed")
        
        # Show statistics for all channels
        print("\n📊 Multi-Channel Statistics:")
        for i, channel_id in enumerate(channel_ids):
            info = self.manager.get_channel_info(channel_id)
            if info:
                print(f"  Channel {i + 1}: {info['sequence_number']} updates, "
                      f"Status: {info['status']}")
        
        # Show global statistics
        global_stats = self.manager.get_channel_statistics()
        print(f"\n📊 Global Statistics:")
        print(f"  Total Channels: {global_stats['active_channels']}")
        print(f"  Total Participants: {global_stats['total_participants']}")
        
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
            print("  No channel information available")
            return
        
        print(f"  Channel ID: {info['channel_id']}")
        print(f"  Status: {info['status']}")
        print(f"  Participants: {info['participants']}")
        print(f"  Total Deposits: {info['total_deposits']}")
        print(f"  Total Balances: {info['total_balances']}")
        print(f"  Sequence Number: {info['sequence_number']}")
        print(f"  Created At: {info['created_at']}")
        if info.get('opened_at'):
            print(f"  Opened At: {info['opened_at']}")
        if info.get('closed_at'):
            print(f"  Closed At: {info['closed_at']}")
        if info.get('close_reason'):
            print(f"  Close Reason: {info['close_reason']}")
    
    def run_complete_demo(self):
        """Run the complete demo."""
        print("🎯 State Channels Complete Demo")
        print("=" * 60)
        
        try:
            # Run all demos
            self.demo_basic_channel_lifecycle()
            self.demo_dispute_resolution()
            self.demo_security_mechanisms()
            self.demo_performance_monitoring()
            self.demo_multi_channel_operations()
            
            # Show event summary
            print("\n📋 Event Summary:")
            print("=" * 50)
            event_counts = {}
            for event, channel_id, timestamp in self.events_received:
                event_counts[event.value] = event_counts.get(event.value, 0) + 1
            
            for event_type, count in event_counts.items():
                print(f"  {event_type}: {count}")
            
            print(f"\n✅ Demo completed successfully!")
            print(f"📊 Total events received: {len(self.events_received)}")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo."""
    demo = StateChannelsDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
