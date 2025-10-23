#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Cross-Chain Bridge Demo for DubChain

This demo showcases the sophisticated cross-chain bridge system including:
- Multi-chain asset transfers
- Atomic swaps between chains
- Cross-chain message passing
- Universal asset management
- Bridge security and validation

Run this demo to see how DubChain enables seamless interoperability
between different blockchain networks.
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, List

# Import DubChain bridge components
from dubchain.bridge import (
    BridgeType,
    BridgeStatus,
    ChainType,
    AssetType,
    BridgeConfig,
    BridgeManager,
    CrossChainTransaction,
    BridgeAsset,
    BridgeValidator,
    AtomicSwap,
    SwapProposal
)
from dubchain.crypto.signatures import PrivateKey


class CrossChainBridgeDemo:
    """Demonstrates cross-chain bridge capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.bridge_manager = None
        self.atomic_swap = None
        self.demo_chains = {}
        self.demo_assets = {}
        self.demo_validators = []
    
    def setup_bridge_system(self) -> None:
        """Setup the cross-chain bridge system."""
        logger.info("🌉 Setting up cross-chain bridge system...")
        
        # Create bridge configuration
        config = BridgeConfig(
            bridge_type=BridgeType.LOCK_AND_MINT,
            supported_chains=["dubchain", "ethereum", "bitcoin", "polygon"],
            supported_assets=["DUB", "ETH", "BTC", "MATIC"],
            min_transfer_amount=1,
            max_transfer_amount=1000000000,
            transfer_fee_percentage=0.001,
            confirmation_blocks=12,
            timeout_blocks=100,
            enable_atomic_swaps=True,
            enable_cross_chain_messaging=True
        )
        
        # Create bridge manager
        self.bridge_manager = BridgeManager(config)
        
        # Create atomic swap system
        self.atomic_swap = AtomicSwap()
        
        logger.info("  ✅ Cross-chain bridge system initialized")
    
    def setup_demo_chains(self) -> None:
        """Setup demo blockchain networks."""
        logger.info("\n⛓️  Setting up demo blockchain networks...")
        
        # Define demo chains
        chains = [
            {
                "chain_id": "dubchain",
                "name": "DubChain",
                "type": ChainType.MAINNET.value,
                "block_time": 2.0,
                "consensus": "proof_of_stake"
            },
            {
                "chain_id": "ethereum",
                "name": "Ethereum",
                "type": ChainType.MAINNET.value,
                "block_time": 13.0,
                "consensus": "proof_of_stake"
            },
            {
                "chain_id": "bitcoin",
                "name": "Bitcoin",
                "type": ChainType.MAINNET.value,
                "block_time": 600.0,
                "consensus": "proof_of_work"
            },
            {
                "chain_id": "polygon",
                "name": "Polygon",
                "type": ChainType.LAYER2.value,
                "block_time": 2.0,
                "consensus": "proof_of_stake"
            }
        ]
        
        # Add chains to bridge manager
        for chain_info in chains:
            success = self.bridge_manager.add_chain(chain_info["chain_id"], chain_info)
            if success:
                logger.info(f"  ✅ Added {chain_info['name']} ({chain_info['chain_id']})")
                self.demo_chains[chain_info["chain_id"]] = chain_info
            else:
                logger.info(f"  ❌ Failed to add {chain_info['name']}")
    
    def setup_demo_assets(self) -> None:
        """Setup demo assets."""
        logger.info("\n💰 Setting up demo assets...")
        
        # Define demo assets
        assets = [
            {
                "asset_id": "DUB",
                "asset_type": AssetType.NATIVE,
                "chain_id": "dubchain",
                "symbol": "DUB",
                "name": "DubChain Token",
                "decimals": 18,
                "total_supply": 1000000000
            },
            {
                "asset_id": "ETH",
                "asset_type": AssetType.NATIVE,
                "chain_id": "ethereum",
                "symbol": "ETH",
                "name": "Ethereum",
                "decimals": 18,
                "total_supply": 120000000
            },
            {
                "asset_id": "BTC",
                "asset_type": AssetType.NATIVE,
                "chain_id": "bitcoin",
                "symbol": "BTC",
                "name": "Bitcoin",
                "decimals": 8,
                "total_supply": 21000000
            },
            {
                "asset_id": "MATIC",
                "asset_type": AssetType.NATIVE,
                "chain_id": "polygon",
                "symbol": "MATIC",
                "name": "Polygon",
                "decimals": 18,
                "total_supply": 10000000000
            }
        ]
        
        # Register assets
        for asset_info in assets:
            asset = BridgeAsset(**asset_info)
            success = self.bridge_manager.register_asset(asset)
            if success:
                logger.info(f"  ✅ Registered {asset.symbol} on {asset.chain_id}")
                self.demo_assets[asset.asset_id] = asset
            else:
                logger.info(f"  ❌ Failed to register {asset.symbol}")
    
    def setup_demo_validators(self) -> None:
        """Setup demo bridge validators."""
        logger.info("\n🛡️  Setting up demo bridge validators...")
        
        # Create validators for each chain
        for chain_id in self.demo_chains.keys():
            for i in range(3):  # 3 validators per chain
                private_key = PrivateKey.generate()
                validator = BridgeValidator(
                    validator_id=f"validator_{chain_id}_{i}",
                    chain_id=chain_id,
                    public_key=private_key.get_public_key().to_hex(),
                    stake_amount=1000000 + (i * 100000),
                    is_active=True
                )
                
                success = self.bridge_manager.add_validator(validator)
                if success:
                    logger.info(f"  ✅ Added validator {validator.validator_id}")
                    self.demo_validators.append(validator)
                else:
                    logger.info(f"  ❌ Failed to add validator {validator.validator_id}")
    
    def demonstrate_cross_chain_transfers(self) -> None:
        """Demonstrate cross-chain asset transfers."""
        logger.info("\n🔄 CROSS-CHAIN TRANSFERS DEMONSTRATION")
        logger.info("=" * 50)
        
        # Create cross-chain transactions
        logger.info("📝 Creating cross-chain transactions...")
        
        transactions = [
            {
                "source_chain": "dubchain",
                "target_chain": "ethereum",
                "source_asset": "DUB",
                "target_asset": "ETH",
                "sender": "alice",
                "receiver": "bob",
                "amount": 1000
            },
            {
                "source_chain": "ethereum",
                "target_chain": "polygon",
                "source_asset": "ETH",
                "target_asset": "MATIC",
                "sender": "charlie",
                "receiver": "david",
                "amount": 500
            },
            {
                "source_chain": "bitcoin",
                "target_chain": "dubchain",
                "source_asset": "BTC",
                "target_asset": "DUB",
                "sender": "eve",
                "receiver": "frank",
                "amount": 10
            }
        ]
        
        created_transactions = []
        
        for i, tx_data in enumerate(transactions):
            logger.info(f"  Creating transaction {i + 1}...")
            
            transaction = self.bridge_manager.create_cross_chain_transaction(
                source_chain=tx_data["source_chain"],
                target_chain=tx_data["target_chain"],
                source_asset=tx_data["source_asset"],
                target_asset=tx_data["target_asset"],
                sender=tx_data["sender"],
                receiver=tx_data["receiver"],
                amount=tx_data["amount"]
            )
            
            if transaction:
                logger.info(f"    ✅ Transaction {transaction.transaction_id} created")
                created_transactions.append(transaction)
            else:
                logger.info(f"    ❌ Failed to create transaction {i + 1}")
        
        # Process transactions
        logger.info("\n⚙️  Processing cross-chain transactions...")
        for i, transaction in enumerate(created_transactions):
            logger.info(f"  Processing transaction {i + 1}...")
            
            success = self.bridge_manager.process_transaction(transaction.transaction_id)
            if success:
                logger.info(f"    ✅ Transaction {transaction.transaction_id} completed")
                logger.info(f"    📊 {transaction.amount} {transaction.source_asset} → {transaction.target_asset}")
                logger.info(f"    🔗 {transaction.source_chain} → {transaction.target_chain}")
            else:
                logger.info(f"    ❌ Transaction {transaction.transaction_id} failed")
    
    def demonstrate_atomic_swaps(self) -> None:
        """Demonstrate atomic swaps."""
        logger.info("\n⚛️  ATOMIC SWAPS DEMONSTRATION")
        logger.info("=" * 50)
        
        # Create atomic swap proposal
        logger.info("📝 Creating atomic swap proposal...")
        
        proposal = self.atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain",
            target_chain="ethereum",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=5,
            timeout=3600
        )
        
        if proposal:
            logger.info(f"  ✅ Atomic swap proposal created: {proposal.proposal_id}")
            logger.info(f"    Initiator: {proposal.initiator}")
            logger.info(f"    Counterparty: {proposal.counterparty}")
            logger.info(f"    Amount: {proposal.source_amount} {proposal.source_asset} ↔ {proposal.target_amount} {proposal.target_asset}")
            logger.info(f"    Chains: {proposal.source_chain} ↔ {proposal.target_chain}")
            
            # Accept proposal
            logger.info("\n🤝 Accepting atomic swap proposal...")
            accept_success = self.atomic_swap.accept_proposal(proposal.proposal_id, "bob")
            if accept_success:
                logger.info(f"  ✅ Proposal accepted by {proposal.counterparty}")
                
                # Simulate locking funds
                logger.info("\n🔒 Simulating fund locking...")
                lock1_success = self.atomic_swap.lock_funds(proposal.proposal_id, "alice", "tx_hash_1")
                lock2_success = self.atomic_swap.lock_funds(proposal.proposal_id, "bob", "tx_hash_2")
                
                if lock1_success and lock2_success:
                    logger.info("  ✅ Both parties locked funds")
                    
                    # Reveal secret
                    logger.info("\n🔓 Revealing secret...")
                    reveal_success = self.atomic_swap.reveal_secret(proposal.proposal_id, proposal.secret)
                    if reveal_success:
                        logger.info("  ✅ Secret revealed")
                        
                        # Complete swap
                        logger.info("\n✅ Completing atomic swap...")
                        complete_success = self.atomic_swap.complete_swap(proposal.proposal_id)
                        if complete_success:
                            logger.info("  ✅ Atomic swap completed successfully!")
                        else:
                            logger.info("  ❌ Failed to complete atomic swap")
                    else:
                        logger.info("  ❌ Failed to reveal secret")
                else:
                    logger.info("  ❌ Failed to lock funds")
            else:
                logger.info("  ❌ Failed to accept proposal")
        else:
            logger.info("  ❌ Failed to create atomic swap proposal")
    
    def demonstrate_bridge_management(self) -> None:
        """Demonstrate bridge management features."""
        logger.info("\n🎛️  BRIDGE MANAGEMENT DEMONSTRATION")
        logger.info("=" * 50)
        
        # Show bridge status
        logger.info("📊 Bridge Status:")
        status = self.bridge_manager.get_bridge_status()
        logger.info(f"  Status: {status['status']}")
        logger.info(f"  Bridge Type: {status['bridge_type']}")
        logger.info(f"  Supported Chains: {len(status['supported_chains'])}")
        logger.info(f"  Supported Assets: {len(status['supported_assets'])}")
        logger.info(f"  Active Validators: {status['active_validators']}")
        logger.info(f"  Total Transactions: {status['total_transactions']}")
        logger.info(f"  Completed Transactions: {status['completed_transactions']}")
        
        # Show chain information
        logger.info("\n⛓️  Chain Information:")
        for chain_id in status['supported_chains']:
            chain_info = self.bridge_manager.chain_manager.get_chain_info(chain_id)
            if chain_info:
                logger.info(f"  {chain_info['name']} ({chain_id}):")
                logger.info(f"    Type: {chain_info['type']}")
                logger.info(f"    Block Time: {chain_info['block_time']}s")
                logger.info(f"    Consensus: {chain_info['consensus']}")
        
        # Show asset information
        logger.info("\n💰 Asset Information:")
        for asset_id in status['supported_assets']:
            asset = self.bridge_manager.asset_manager.get_asset_info(asset_id)
            if asset:
                logger.info(f"  {asset.symbol} ({asset_id}):")
                logger.info(f"    Name: {asset.name}")
                logger.info(f"    Chain: {asset.chain_id}")
                logger.info(f"    Type: {asset.asset_type.value}")
                logger.info(f"    Decimals: {asset.decimals}")
        
        # Show validator information
        logger.info("\n🛡️  Validator Information:")
        for validator in self.demo_validators[:5]:  # Show first 5 validators
            logger.info(f"  {validator.validator_id}:")
            logger.info(f"    Chain: {validator.chain_id}")
            logger.info(f"    Stake: {validator.stake_amount}")
            logger.info(f"    Active: {validator.is_active}")
            logger.info(f"    Validations: {validator.validation_count}")
    
    def show_bridge_metrics(self) -> None:
        """Show comprehensive bridge metrics."""
        logger.info("\n📊 BRIDGE METRICS")
        logger.info("=" * 50)
        
        # Bridge metrics
        bridge_metrics = self.bridge_manager.get_bridge_metrics()
        logger.info("🌉 Bridge Metrics:")
        logger.info(f"  Total Transactions: {bridge_metrics.total_transactions}")
        logger.info(f"  Successful Transactions: {bridge_metrics.successful_transactions}")
        logger.info(f"  Failed Transactions: {bridge_metrics.failed_transactions}")
        logger.info(f"  Success Rate: {bridge_metrics.success_rate:.2%}")
        logger.info(f"  Total Volume: {bridge_metrics.total_volume}")
        logger.info(f"  Average Transaction Time: {bridge_metrics.average_transaction_time:.2f}s")
        logger.info(f"  Active Validators: {bridge_metrics.active_validators}")
        logger.info(f"  Supported Chains: {bridge_metrics.supported_chains}")
        logger.info(f"  Supported Assets: {bridge_metrics.supported_assets}")
        
        # Atomic swap metrics
        if self.atomic_swap:
            swap_metrics = self.atomic_swap.get_swap_metrics()
            logger.info("\n⚛️  Atomic Swap Metrics:")
            logger.info(f"  Total Proposals: {swap_metrics['total_proposals']}")
            logger.info(f"  Active Proposals: {swap_metrics['active_proposals']}")
            logger.info(f"  Completed Swaps: {swap_metrics['completed_swaps']}")
            logger.info(f"  Failed Swaps: {swap_metrics['failed_swaps']}")
            logger.info(f"  Proposals Created: {swap_metrics['metrics']['proposals_created']}")
            logger.info(f"  Proposals Accepted: {swap_metrics['metrics']['proposals_accepted']}")
            logger.info(f"  Swaps Completed: {swap_metrics['metrics']['swaps_completed']}")
            logger.info(f"  Swaps Failed: {swap_metrics['metrics']['swaps_failed']}")
    
    def run_demo(self) -> None:
        """Run the complete cross-chain bridge demo."""
        logger.info("🌉 DUBCHAIN CROSS-CHAIN BRIDGE DEMO")
        logger.info("=" * 60)
        logger.info("This demo showcases seamless interoperability between")
        logger.info("different blockchain networks through advanced bridge")
        logger.info("technology and atomic swap mechanisms.")
        logger.info("=" * 60)
        
        # Setup
        self.setup_bridge_system()
        self.setup_demo_chains()
        self.setup_demo_assets()
        self.setup_demo_validators()
        
        # Demonstrate features
        self.demonstrate_cross_chain_transfers()
        self.demonstrate_atomic_swaps()
        self.demonstrate_bridge_management()
        self.show_bridge_metrics()
        
        logger.info("\n🎉 DEMO COMPLETED!")
        logger.info("=" * 60)
        logger.info("DubChain's cross-chain bridge system provides:")
        logger.info("✅ Multi-chain asset transfers")
        logger.info("✅ Atomic swaps between chains")
        logger.info("✅ Cross-chain message passing")
        logger.info("✅ Universal asset management")
        logger.info("✅ Bridge security and validation")
        logger.info("✅ Enterprise-grade interoperability")
        logger.info("=" * 60)


async def main():
    """Main demo function."""
    demo = CrossChainBridgeDemo()
    demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
