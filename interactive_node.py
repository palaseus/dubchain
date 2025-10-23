#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Interactive DubChain Node

This script creates an interactive DubChain node that you can experiment with in real-time.
It provides a command-line interface to interact with the blockchain, create transactions,
mine blocks, and experiment with different consensus mechanisms.

Usage: python3 interactive_node.py
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional
from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig
from dubchain.consensus import ConsensusType, ConsensusEngine


class InteractiveNode:
    """Interactive DubChain node for experimentation."""
    
    def __init__(self):
        """Initialize the interactive node."""
        self.blockchain = None
        self.wallets = {}
        self.consensus_engine = None
        self.running = True
        self.node_id = f"node_{int(time.time())}"
        
        logger.info("🚀 DubChain Interactive Node")
        logger.info("=" * 50)
        logger.info("Type 'help' for available commands")
        logger.info("Type 'quit' to exit")
        logger.info("=" * 50)
    
    def setup_blockchain(self, consensus_type: str = "pos") -> None:
        """Set up the blockchain with specified consensus."""
        logger.info(f"\n🔧 Setting up blockchain with {consensus_type.upper()} consensus...")
        
        # Create consensus configuration
        config = ConsensusConfig(
            target_block_time=2,  # 2 seconds for interactive demo
            difficulty_adjustment_interval=5,
            min_difficulty=1,
            max_difficulty=4
        )
        
        # Create blockchain
        self.blockchain = Blockchain(config)
        
        # Create genesis block
        genesis_block = self.blockchain.create_genesis_block(
            coinbase_recipient=self.node_id,
            coinbase_amount=1000000
        )
        
        logger.info(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
        
        # Set up consensus engine
        self.consensus_engine = ConsensusEngine(consensus_type)
        
        logger.info(f"✅ Blockchain initialized with {consensus_type.upper()} consensus")
    
    def create_wallet(self, name: str) -> None:
        """Create a new wallet."""
        if name in self.wallets:
            logger.info(f"❌ Wallet '{name}' already exists")
            return
        
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        address = public_key.to_address()
        
        self.wallets[name] = {
            'private_key': private_key,
            'public_key': public_key,
            'address': address,
            'balance': 0
        }
        
        logger.info(f"✅ Wallet '{name}' created")
        logger.info(f"   Address: {address}")
        logger.info(f"   Private Key: {private_key.to_hex()[:16]}...")
    
    def mine_blocks(self, count: int = 1, miner: str = None) -> None:
        """Mine blocks to get funds."""
        if not self.blockchain:
            logger.info("❌ Blockchain not initialized")
            return
        
        if not miner:
            # Use first available wallet
            if not self.wallets:
                logger.info("❌ No wallets available. Create a wallet first.")
                return
            miner = list(self.wallets.keys())[0]
        
        if miner not in self.wallets:
            logger.info(f"❌ Wallet '{miner}' not found")
            return
        
        miner_address = self.wallets[miner]['address']
        
        logger.info(f"\n⛏️  Mining {count} block(s) with {miner}...")
        
        for i in range(count):
            start_time = time.time()
            block = self.blockchain.mine_block(miner_address, max_transactions=10)
            
            if block:
                mining_time = time.time() - start_time
                logger.info(f"   ✅ Block {i+1} mined in {mining_time:.2f}s")
                logger.info(f"      Hash: {block.get_hash().to_hex()[:16]}...")
                logger.info(f"      Nonce: {block.header.nonce}")
                logger.info(f"      Transactions: {len(block.transactions)}")
            else:
                logger.info(f"   ❌ Failed to mine block {i+1}")
    
    def create_transaction(self, sender: str, recipient: str, amount: int, fee: int = 1000) -> None:
        """Create a transaction between wallets."""
        if not self.blockchain:
            logger.info("❌ Blockchain not initialized")
            return
        
        if sender not in self.wallets:
            logger.info(f"❌ Sender wallet '{sender}' not found")
            return
        
        if recipient not in self.wallets:
            logger.info(f"❌ Recipient wallet '{recipient}' not found")
            return
        
        sender_private = self.wallets[sender]['private_key']
        recipient_address = self.wallets[recipient]['address']
        
        logger.info(f"\n💸 Creating transaction...")
        logger.info(f"   From: {sender} ({self.wallets[sender]['address']})")
        logger.info(f"   To: {recipient} ({recipient_address})")
        logger.info(f"   Amount: {amount}")
        logger.info(f"   Fee: {fee}")
        
        # Create transaction
        tx = self.blockchain.create_transfer_transaction(
            sender_private_key=sender_private,
            recipient_address=recipient_address,
            amount=amount,
            fee=fee
        )
        
        if tx:
            logger.info(f"✅ Transaction created: {tx.get_hash().to_hex()[:16]}...")
            
            # Add to pending pool
            self.blockchain.add_transaction(tx)
            logger.info("   Transaction added to pending pool")
        else:
            logger.info("❌ Failed to create transaction")
    
    def check_balances(self) -> None:
        """Check balances of all wallets."""
        if not self.blockchain:
            logger.info("❌ Blockchain not initialized")
            return
        
        logger.info("\n💰 Wallet Balances:")
        logger.info("-" * 40)
        
        for name, wallet in self.wallets.items():
            balance = self.blockchain.get_balance(wallet['address'])
            wallet['balance'] = balance
            logger.info(f"   {name}: {balance:,} satoshis")
    
    def show_blockchain_info(self) -> None:
        """Show blockchain information."""
        if not self.blockchain:
            logger.info("❌ Blockchain not initialized")
            return
        
        info = self.blockchain.get_chain_info()
        
        logger.info("\n📊 Blockchain Information:")
        logger.info("-" * 40)
        logger.info(f"   Block count: {info['block_count']}")
        logger.info(f"   Block height: {info['block_height']}")
        logger.info(f"   Total difficulty: {info['total_difficulty']}")
        logger.info(f"   Pending transactions: {info['pending_transactions']}")
        logger.info(f"   UTXO count: {info['utxo_count']}")
        
        if 'current_difficulty' in info:
            logger.info(f"   Current difficulty: {info['current_difficulty']}")
            logger.info(f"   Average block time: {info.get('average_block_time', 0):.2f}s")
    
    def show_wallets(self) -> None:
        """Show all wallets."""
        if not self.wallets:
            logger.info("❌ No wallets created")
            return
        
        logger.info("\n👛 Wallets:")
        logger.info("-" * 40)
        
        for name, wallet in self.wallets.items():
            logger.info(f"   {name}:")
            logger.info(f"      Address: {wallet['address']}")
            logger.info(f"      Balance: {wallet.get('balance', 0):,} satoshis")
    
    def stress_test(self, transactions: int = 100) -> None:
        """Run a stress test with many transactions."""
        if not self.blockchain or len(self.wallets) < 2:
            logger.info("❌ Need blockchain and at least 2 wallets for stress test")
            return
        
        logger.info(f"\n🔥 Running stress test with {transactions} transactions...")
        
        wallet_names = list(self.wallets.keys())
        start_time = time.time()
        
        for i in range(transactions):
            sender = wallet_names[i % len(wallet_names)]
            recipient = wallet_names[(i + 1) % len(wallet_names)]
            amount = 1000 + (i % 10000)  # Varying amounts
            
            tx = self.blockchain.create_transfer_transaction(
                sender_private_key=self.wallets[sender]['private_key'],
                recipient_address=self.wallets[recipient]['address'],
                amount=amount,
                fee=100
            )
            
            if tx:
                self.blockchain.add_transaction(tx)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Created {i + 1} transactions...")
        
        end_time = time.time()
        logger.info(f"✅ Stress test completed in {end_time - start_time:.2f}s")
        logger.info(f"   Created {transactions} transactions")
        logger.info(f"   Rate: {transactions / (end_time - start_time):.2f} tx/s")
    
    def chaos_mode(self) -> None:
        """Enter chaos mode - random operations."""
        logger.info("\n🌪️  CHAOS MODE ACTIVATED!")
        logger.info("Random operations will be performed...")
        
        operations = [
            lambda: self.mine_blocks(1),
            lambda: self.create_transaction(
                list(self.wallets.keys())[0] if self.wallets else None,
                list(self.wallets.keys())[1] if len(self.wallets) > 1 else None,
                1000 + (int(time.time()) % 10000)
            ) if len(self.wallets) >= 2 else None,
            lambda: self.check_balances(),
            lambda: self.show_blockchain_info()
        ]
        
        for i in range(10):
            op = operations[i % len(operations)]
            if op:
                try:
                    op()
                except Exception as e:
                    logger.info(f"   💥 Chaos operation failed: {e}")
            
            time.sleep(0.5)
        
        logger.info("🌪️  Chaos mode completed!")
    
    def help(self) -> None:
        """Show help information."""
        logger.info("\n📖 Available Commands:")
        logger.info("-" * 40)
        logger.info("  setup [consensus]     - Set up blockchain (pos/dpos/pbft)")
        logger.info("  wallet <name>         - Create a new wallet")
        logger.info("  mine [count] [miner]  - Mine blocks")
        logger.info("  send <from> <to> <amount> [fee] - Send transaction")
        logger.info("  balance               - Check all balances")
        logger.info("  info                  - Show blockchain info")
        logger.info("  wallets               - Show all wallets")
        logger.info("  stress [count]        - Run stress test")
        logger.info("  chaos                 - Enter chaos mode")
        logger.info("  validate              - Validate blockchain")
        logger.info("  help                  - Show this help")
        logger.info("  quit                  - Exit")
    
    def validate_blockchain(self) -> None:
        """Validate the entire blockchain."""
        if not self.blockchain:
            logger.info("❌ Blockchain not initialized")
            return
        
        logger.info("\n🔍 Validating blockchain...")
        start_time = time.time()
        
        is_valid = self.blockchain.validate_chain()
        
        end_time = time.time()
        logger.info(f"✅ Validation completed in {end_time - start_time:.2f}s")
        logger.info(f"   Result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    def run(self) -> None:
        """Run the interactive node."""
        while self.running:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == "quit" or cmd == "exit":
                    logger.info("👋 Goodbye!")
                    break
                
                elif cmd == "help":
                    self.help()
                
                elif cmd == "setup":
                    consensus = command[1] if len(command) > 1 else "pos"
                    self.setup_blockchain(consensus)
                
                elif cmd == "wallet":
                    if len(command) < 2:
                        logger.info("❌ Usage: wallet <name>")
                        continue
                    self.create_wallet(command[1])
                
                elif cmd == "mine":
                    count = int(command[1]) if len(command) > 1 else 1
                    miner = command[2] if len(command) > 2 else None
                    self.mine_blocks(count, miner)
                
                elif cmd == "send":
                    if len(command) < 4:
                        logger.info("❌ Usage: send <from> <to> <amount> [fee]")
                        continue
                    
                    try:
                        sender = command[1]
                        recipient = command[2]
                        amount = int(command[3])
                        fee = int(command[4]) if len(command) > 4 else 1000
                        
                        self.create_transaction(sender, recipient, amount, fee)
                    except ValueError:
                        logger.info("❌ Invalid amount or fee")
                
                elif cmd == "balance":
                    self.check_balances()
                
                elif cmd == "info":
                    self.show_blockchain_info()
                
                elif cmd == "wallets":
                    self.show_wallets()
                
                elif cmd == "stress":
                    count = int(command[1]) if len(command) > 1 else 100
                    self.stress_test(count)
                
                elif cmd == "chaos":
                    self.chaos_mode()
                
                elif cmd == "validate":
                    self.validate_blockchain()
                
                else:
                    logger.info(f"❌ Unknown command: {cmd}")
                    logger.info("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.info(f"❌ Error: {e}")


def main():
    """Main function."""
    node = InteractiveNode()
    node.run()


if __name__ == "__main__":
    main()
