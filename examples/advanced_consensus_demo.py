#!/usr/bin/env python3
logger = logging.getLogger(__name__)
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

import logging
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
        logger.info(f"🔧 Creating {count} validators...")
        
        for i in range(count):
            private_key = PrivateKey.generate()
            validator = Validator(
                validator_id=f"validator_{i}",
                private_key=private_key,
                commission_rate=0.1
            )
            self.validators.append(validator)
            logger.info(f"  ✅ Created {validator.validator_id}")
    
    def setup_consensus_mechanisms(self) -> None:
        """Setup different consensus mechanisms."""
        logger.info("\n🚀 Setting up consensus mechanisms...")
        
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
        
        logger.info("  ✅ All consensus mechanisms initialized")
    
    def register_validators(self) -> None:
        """Register validators with consensus mechanisms."""
        logger.info("\n👥 Registering validators...")
        
        for engine_name, engine in self.consensus_engines.items():
            logger.info(f"  📝 Registering validators with {engine_name}...")
            
            for i, validator in enumerate(self.validators):
                initial_stake = 1000000 + (i * 500000)  # Varying stakes
                success = engine.register_validator(validator, initial_stake)
                if success:
                    logger.info(f"    ✅ {validator.validator_id} registered with {initial_stake} stake")
                else:
                    logger.info(f"    ❌ Failed to register {validator.validator_id}")
    
    def demonstrate_proof_of_stake(self) -> None:
        """Demonstrate Proof of Stake consensus."""
        logger.info("\n🪙 PROOF OF STAKE DEMONSTRATION")
        logger.info("=" * 50)
        
        engine = self.consensus_engines['PoS']
        
        # Show validator information
        logger.info("📊 Validator Information:")
        for validator_id in engine.get_active_validators():
            validator_info = engine.get_validator_info(validator_id)
            if validator_info:
                logger.info(f"  {validator_id}: {validator_info.total_stake} stake, "
                      f"{validator_info.voting_power} voting power")
        
        # Propose blocks
        logger.info("\n⛏️  Block Production:")
        for i in range(3):
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            logger.info(f"  Proposing block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                logger.info(f"    ✅ Block {i + 1} finalized by {result.validator_id}")
                logger.info(f"    🔗 Block hash: {result.block_hash}")
                logger.info(f"    ⏱️  Gas used: {result.gas_used}")
            else:
                logger.info(f"    ❌ Block {i + 1} failed: {result.error_message}")
        
        # Show metrics
        metrics = engine.get_consensus_metrics()
        logger.info(f"\n📈 PoS Metrics:")
        logger.info(f"  Total blocks: {metrics.total_blocks}")
        logger.info(f"  Success rate: {metrics.success_rate:.2%}")
        logger.info(f"  Active validators: {metrics.active_validators}")
    
    def demonstrate_delegated_proof_of_stake(self) -> None:
        """Demonstrate Delegated Proof of Stake consensus."""
        logger.info("\n🗳️  DELEGATED PROOF OF STAKE DEMONSTRATION")
        logger.info("=" * 50)
        
        engine = self.consensus_engines['DPoS']
        
        # Show voting statistics
        logger.info("📊 Voting Statistics:")
        if hasattr(engine.consensus_mechanism, 'get_voting_statistics'):
            stats = engine.consensus_mechanism.get_voting_statistics()
            logger.info(f"  Total voters: {stats['total_voters']}")
            logger.info(f"  Active delegates: {stats['active_delegates']}")
            logger.info(f"  Total voting power: {stats['total_voting_power']}")
        
        # Show delegate rankings
        logger.info("\n🏆 Delegate Rankings:")
        if hasattr(engine.consensus_mechanism, 'get_delegate_rankings'):
            rankings = engine.consensus_mechanism.get_delegate_rankings()
            for i, (delegate_id, power) in enumerate(rankings[:5]):
                logger.info(f"  {i + 1}. {delegate_id}: {power} voting power")
        
        # Propose blocks
        logger.info("\n⛏️  Block Production:")
        for i in range(3):
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            logger.info(f"  Proposing block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                logger.info(f"    ✅ Block {i + 1} produced by {result.validator_id}")
                logger.info(f"    🔗 Block hash: {result.block_hash}")
            else:
                logger.info(f"    ❌ Block {i + 1} failed: {result.error_message}")
    
    def demonstrate_pbft(self) -> None:
        """Demonstrate PBFT consensus."""
        logger.info("\n🛡️  PRACTICAL BYZANTINE FAULT TOLERANCE DEMONSTRATION")
        logger.info("=" * 50)
        
        engine = self.consensus_engines['PBFT']
        
        # Show network status
        logger.info("📊 Network Status:")
        if hasattr(engine.consensus_mechanism, 'get_network_status'):
            status = engine.consensus_mechanism.get_network_status()
            logger.info(f"  Total validators: {status['total_validators']}")
            logger.info(f"  Online validators: {status['online_validators']}")
            logger.info(f"  Primary validator: {status['primary_validator']}")
            logger.info(f"  Current view: {status['current_view']}")
            logger.info(f"  Fault tolerance: {status['fault_tolerance']}")
        
        # Propose blocks through PBFT
        logger.info("\n⛏️  PBFT Consensus Process:")
        for i in range(2):  # Fewer blocks for PBFT demo
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            logger.info(f"  Starting PBFT consensus for block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                logger.info(f"    ✅ Block {i + 1} committed through PBFT")
                logger.info(f"    🔗 Block hash: {result.block_hash}")
                logger.info(f"    👑 Primary: {result.validator_id}")
            else:
                logger.info(f"    ❌ Block {i + 1} failed: {result.error_message}")
    
    def demonstrate_hybrid_consensus(self) -> None:
        """Demonstrate Hybrid consensus system."""
        logger.info("\n🔄 HYBRID CONSENSUS DEMONSTRATION")
        logger.info("=" * 50)
        
        engine = self.consensus_engines['Hybrid']
        
        # Show consensus information
        logger.info("📊 Hybrid Consensus Information:")
        if hasattr(engine.consensus_mechanism, 'get_consensus_info'):
            info = engine.consensus_mechanism.get_consensus_info()
            logger.info(f"  Current consensus: {info['current_consensus']}")
            logger.info(f"  Switch count: {info['switch_count']}")
            logger.info(f"  Can switch: {info['can_switch']}")
        
        # Simulate network conditions
        logger.info("\n🌐 Simulating Network Conditions:")
        network_conditions = {
            'network_size': 15,
            'average_latency': 50.0,
            'fault_tolerance': 0.2
        }
        logger.info(f"  Network size: {network_conditions['network_size']}")
        logger.info(f"  Average latency: {network_conditions['average_latency']}ms")
        logger.info(f"  Fault tolerance needed: {network_conditions['fault_tolerance']}")
        
        # Update network conditions
        if hasattr(engine.consensus_mechanism, 'update_network_conditions'):
            engine.consensus_mechanism.update_network_conditions(network_conditions)
        
        # Propose blocks
        logger.info("\n⛏️  Hybrid Block Production:")
        for i in range(3):
            block_data = self.demo_data.copy()
            block_data['block_number'] = i + 1
            block_data['timestamp'] = time.time()
            
            logger.info(f"  Proposing block {i + 1}...")
            result = engine.propose_block(block_data)
            
            if result.success:
                logger.info(f"    ✅ Block {i + 1} finalized")
                logger.info(f"    🔗 Block hash: {result.block_hash}")
                logger.info(f"    🎯 Consensus type: {result.consensus_type.value}")
            else:
                logger.info(f"    ❌ Block {i + 1} failed: {result.error_message}")
    
    def show_performance_comparison(self) -> None:
        """Show performance comparison between consensus mechanisms."""
        logger.info("\n📊 PERFORMANCE COMPARISON")
        logger.info("=" * 50)
        
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
        logger.info(f"{'Consensus':<10} {'Blocks':<8} {'Success':<8} {'Avg Time':<10} {'Validators':<12}")
        logger.info("-" * 60)
        
        for data in comparison_data:
            logger.info(f"{data['consensus']:<10} {data['total_blocks']:<8} "
                  f"{data['success_rate']:<8.2%} {data['avg_block_time']:<10.2f} "
                  f"{data['active_validators']:<12}")
    
    def run_demo(self) -> None:
        """Run the complete consensus demo."""
        logger.info("🚀 DUBCHAIN ADVANCED CONSENSUS DEMO")
        logger.info("=" * 60)
        logger.info("This demo showcases sophisticated consensus mechanisms")
        logger.info("including PoS, DPoS, PBFT, and Hybrid consensus.")
        logger.info("=" * 60)
        
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
        
        logger.info("\n🎉 DEMO COMPLETED!")
        logger.info("=" * 60)
        logger.info("DubChain's advanced consensus system provides:")
        logger.info("✅ Multiple consensus mechanisms")
        logger.info("✅ Adaptive consensus selection")
        logger.info("✅ High fault tolerance")
        logger.info("✅ Enterprise-grade security")
        logger.info("✅ Scalable architecture")
        logger.info("=" * 60)


async def main():
    """Main demo function."""
    demo = ConsensusDemo()
    demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
