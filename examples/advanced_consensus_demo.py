#!/usr/bin/env python3
"""
Advanced Consensus Demo for DubChain

This demo showcases the sophisticated consensus mechanisms including:
- Proof of Stake (PoS)
- Delegated Proof of Stake (DPoS)
- Practical Byzantine Fault Tolerance (PBFT)
- Hybrid Consensus System

Run this demo to see how DubChain can adapt to different network conditions
and provide enterprise-grade consensus mechanisms.
"""

import asyncio
import time
import json
from typing import Dict, Any

# Import DubChain consensus components
from dubchain.consensus import (
    ConsensusType,
    ConsensusConfig,
    ConsensusEngine,
    Validator,
    ProofOfStake,
    DelegatedProofOfStake,
    PracticalByzantineFaultTolerance,
    HybridConsensus
)
from dubchain.crypto.signatures import PrivateKey


class ConsensusDemo:
    """Demonstrates advanced consensus mechanisms."""
    
    def __init__(self):
        """Initialize the demo."""
        self.validators = []
        self.consensus_engines = {}
        self.demo_data = {
            'block_number': 1,
            'timestamp': time.time(),
            'transactions': [
                {'from': 'alice', 'to': 'bob', 'amount': 100},
                {'from': 'charlie', 'to': 'david', 'amount': 50}
            ],
            'previous_hash': '0x0000000000000000000000000000000000000000000000000000000000000000',
            'gas_used': 21000
        }
    
    def create_validators(self, count: int = 5) -> None:
        """Create validators for the demo."""
        print(f"🔧 Creating {count} validators...")
        
        for i in range(count):
            private_key = PrivateKey.generate()
            validator = Validator(
                validator_id=f"validator_{i}",
                private_key=private_key,
                commission_rate=0.1
            )
            self.validators.append(validator)
            print(f"  ✅ Created {validator.validator_id}")
    
    def setup_consensus_mechanisms(self) -> None:
        """Setup different consensus mechanisms."""
        print("\n🚀 Setting up consensus mechanisms...")
        
        # Proof of Stake
        pos_config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            block_time=2.0,
            max_validators=10,
            min_stake=1000000,
            reward_rate=0.1
        )
        self.consensus_engines['PoS'] = ConsensusEngine(pos_config)
        
        # Delegated Proof of Stake
        dpos_config = ConsensusConfig(
            consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE,
            block_time=1.0,
            max_validators=21,
            min_stake=500000,
            reward_rate=0.08
        )
        self.consensus_engines['DPoS'] = ConsensusEngine(dpos_config)
        
        # PBFT
        pbft_config = ConsensusConfig(
            consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
            block_time=3.0,
            max_validators=7,
            min_stake=2000000,
            pbft_fault_tolerance=1
        )
        self.consensus_engines['PBFT'] = ConsensusEngine(pbft_config)
        
        # Hybrid Consensus
        hybrid_config = ConsensusConfig(
            consensus_type=ConsensusType.HYBRID,
            block_time=2.0,
            max_validators=15,
            min_stake=1000000,
            enable_hybrid=True,
            hybrid_switch_threshold=0.8
        )
        self.consensus_engines['Hybrid'] = ConsensusEngine(hybrid_config)
        
        print("  ✅ All consensus mechanisms initialized")
    
    def register_validators(self) -> None:
        """Register validators with consensus mechanisms."""
        print("\n👥 Registering validators...")
        
        for engine_name, engine in self.consensus_engines.items():
            print(f"  📝 Registering validators with {engine_name}...")
            
            for i, validator in enumerate(self.validators):
                initial_stake = 1000000 + (i * 500000)  # Varying stakes
                success = engine.register_validator(validator, initial_stake)
                if success:
                    print(f"    ✅ {validator.validator_id} registered with {initial_stake} stake")
                else:
                    print(f"    ❌ Failed to register {validator.validator_id}")
    
    def demonstrate_proof_of_stake(self) -> None:
        """Demonstrate Proof of Stake consensus."""
        print("\n🪙 PROOF OF STAKE DEMONSTRATION")
        print("=" * 50)
        
        engine = self.consensus_engines['PoS']
        
        # Show validator information
        print("📊 Validator Information:")
        for validator_id in engine.get_active_validators():
            validator_info = engine.get_validator_info(validator_id)
            if validator_info:
                print(f"  {validator_id}: {validator_info.total_stake} stake, "
                      f"{validator_info.voting_power} voting power")
        
        # Propose blocks
        print("\n⛏️  Block Production:")
        for i in range(3):
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            print(f"  Proposing block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                print(f"    ✅ Block {i + 1} finalized by {result.validator_id}")
                print(f"    🔗 Block hash: {result.block_hash}")
                print(f"    ⏱️  Gas used: {result.gas_used}")
            else:
                print(f"    ❌ Block {i + 1} failed: {result.error_message}")
        
        # Show metrics
        metrics = engine.get_consensus_metrics()
        print(f"\n📈 PoS Metrics:")
        print(f"  Total blocks: {metrics.total_blocks}")
        print(f"  Success rate: {metrics.success_rate:.2%}")
        print(f"  Active validators: {metrics.active_validators}")
    
    def demonstrate_delegated_proof_of_stake(self) -> None:
        """Demonstrate Delegated Proof of Stake consensus."""
        print("\n🗳️  DELEGATED PROOF OF STAKE DEMONSTRATION")
        print("=" * 50)
        
        engine = self.consensus_engines['DPoS']
        
        # Show voting statistics
        print("📊 Voting Statistics:")
        if hasattr(engine.consensus_mechanism, 'get_voting_statistics'):
            stats = engine.consensus_mechanism.get_voting_statistics()
            print(f"  Total voters: {stats['total_voters']}")
            print(f"  Active delegates: {stats['active_delegates']}")
            print(f"  Total voting power: {stats['total_voting_power']}")
        
        # Show delegate rankings
        print("\n🏆 Delegate Rankings:")
        if hasattr(engine.consensus_mechanism, 'get_delegate_rankings'):
            rankings = engine.consensus_mechanism.get_delegate_rankings()
            for i, (delegate_id, power) in enumerate(rankings[:5]):
                print(f"  {i + 1}. {delegate_id}: {power} voting power")
        
        # Propose blocks
        print("\n⛏️  Block Production:")
        for i in range(3):
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            print(f"  Proposing block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                print(f"    ✅ Block {i + 1} produced by {result.validator_id}")
                print(f"    🔗 Block hash: {result.block_hash}")
            else:
                print(f"    ❌ Block {i + 1} failed: {result.error_message}")
    
    def demonstrate_pbft(self) -> None:
        """Demonstrate PBFT consensus."""
        print("\n🛡️  PRACTICAL BYZANTINE FAULT TOLERANCE DEMONSTRATION")
        print("=" * 50)
        
        engine = self.consensus_engines['PBFT']
        
        # Show network status
        print("📊 Network Status:")
        if hasattr(engine.consensus_mechanism, 'get_network_status'):
            status = engine.consensus_mechanism.get_network_status()
            print(f"  Total validators: {status['total_validators']}")
            print(f"  Online validators: {status['online_validators']}")
            print(f"  Primary validator: {status['primary_validator']}")
            print(f"  Current view: {status['current_view']}")
            print(f"  Fault tolerance: {status['fault_tolerance']}")
        
        # Propose blocks through PBFT
        print("\n⛏️  PBFT Consensus Process:")
        for i in range(2):  # Fewer blocks for PBFT demo
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            print(f"  Starting PBFT consensus for block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                print(f"    ✅ Block {i + 1} committed through PBFT")
                print(f"    🔗 Block hash: {result.block_hash}")
                print(f"    👑 Primary: {result.validator_id}")
            else:
                print(f"    ❌ Block {i + 1} failed: {result.error_message}")
    
    def demonstrate_hybrid_consensus(self) -> None:
        """Demonstrate Hybrid consensus system."""
        print("\n🔄 HYBRID CONSENSUS DEMONSTRATION")
        print("=" * 50)
        
        engine = self.consensus_engines['Hybrid']
        
        # Show consensus information
        print("📊 Hybrid Consensus Information:")
        if hasattr(engine.consensus_mechanism, 'get_consensus_info'):
            info = engine.consensus_mechanism.get_consensus_info()
            print(f"  Current consensus: {info['current_consensus']}")
            print(f"  Switch count: {info['switch_count']}")
            print(f"  Can switch: {info['can_switch']}")
        
        # Simulate network conditions
        print("\n🌐 Simulating Network Conditions:")
        network_conditions = {
            'network_size': 15,
            'average_latency': 50.0,
            'fault_tolerance': 0.2
        }
        print(f"  Network size: {network_conditions['network_size']}")
        print(f"  Average latency: {network_conditions['average_latency']}ms")
        print(f"  Fault tolerance needed: {network_conditions['fault_tolerance']}")
        
        # Update network conditions
        if hasattr(engine.consensus_mechanism, 'update_network_conditions'):
            engine.consensus_mechanism.update_network_conditions(network_conditions)
        
        # Propose blocks
        print("\n⛏️  Hybrid Block Production:")
        for i in range(3):
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            print(f"  Proposing block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                print(f"    ✅ Block {i + 1} finalized")
                print(f"    🔗 Block hash: {result.block_hash}")
                print(f"    🎯 Consensus type: {result.consensus_type.value}")
            else:
                print(f"    ❌ Block {i + 1} failed: {result.error_message}")
    
    def show_performance_comparison(self) -> None:
        """Show performance comparison between consensus mechanisms."""
        print("\n📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        
        comparison_data = []
        
        for engine_name, engine in self.consensus_engines.items():
            metrics = engine.get_consensus_metrics()
            performance = engine.get_performance_statistics()
            
            comparison_data.append({
                'consensus': engine_name,
                'total_blocks': metrics.total_blocks,
                'success_rate': metrics.success_rate,
                'avg_block_time': performance.get('average_block_time', 0),
                'active_validators': metrics.active_validators
            })
        
        # Display comparison table
        print(f"{'Consensus':<10} {'Blocks':<8} {'Success':<8} {'Avg Time':<10} {'Validators':<12}")
        print("-" * 60)
        
        for data in comparison_data:
            print(f"{data['consensus']:<10} {data['total_blocks']:<8} "
                  f"{data['success_rate']:<8.2%} {data['avg_block_time']:<10.2f} "
                  f"{data['active_validators']:<12}")
    
    def run_demo(self) -> None:
        """Run the complete consensus demo."""
        print("🚀 DUBCHAIN ADVANCED CONSENSUS DEMO")
        print("=" * 60)
        print("This demo showcases sophisticated consensus mechanisms")
        print("including PoS, DPoS, PBFT, and Hybrid consensus.")
        print("=" * 60)
        
        # Setup
        self.create_validators(5)
        self.setup_consensus_mechanisms()
        self.register_validators()
        
        # Demonstrate each consensus mechanism
        self.demonstrate_proof_of_stake()
        self.demonstrate_delegated_proof_of_stake()
        self.demonstrate_pbft()
        self.demonstrate_hybrid_consensus()
        
        # Show performance comparison
        self.show_performance_comparison()
        
        print("\n🎉 DEMO COMPLETED!")
        print("=" * 60)
        print("DubChain's advanced consensus system provides:")
        print("✅ Multiple consensus mechanisms")
        print("✅ Adaptive consensus selection")
        print("✅ High fault tolerance")
        print("✅ Enterprise-grade security")
        print("✅ Scalable architecture")
        print("=" * 60)


async def main():
    """Main demo function."""
    demo = ConsensusDemo()
    demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
