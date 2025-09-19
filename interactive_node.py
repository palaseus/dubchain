#!/usr/bin/env python3
"""
Interactive DubChain Node

This script creates an interactive DubChain node that you can "fuck with" in real-time.
It provides a command-line interface to interact with the blockchain, create transactions,
mine blocks, and experiment with different consensus mechanisms.

Usage: python3 interactive_node.py
"""

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
        
        print("🚀 DubChain Interactive Node")
        print("=" * 50)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit")
        print("=" * 50)
    
    def setup_blockchain(self, consensus_type: str = "pos") -> None:
        """Set up the blockchain with specified consensus."""
        print(f"\n🔧 Setting up blockchain with {consensus_type.upper()} consensus...")
        
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
        
        print(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
        
        # Set up consensus engine
        self.consensus_engine = ConsensusEngine(consensus_type)
        
        print(f"✅ Blockchain initialized with {consensus_type.upper()} consensus")
    
    def create_wallet(self, name: str) -> None:
        """Create a new wallet."""
        if name in self.wallets:
            print(f"❌ Wallet '{name}' already exists")
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
        
        print(f"✅ Wallet '{name}' created")
        print(f"   Address: {address}")
        print(f"   Private Key: {private_key.to_hex()[:16]}...")
    
    def mine_blocks(self, count: int = 1, miner: str = None) -> None:
        """Mine blocks to get funds."""
        if not self.blockchain:
            print("❌ Blockchain not initialized")
            return
        
        if not miner:
            # Use first available wallet
            if not self.wallets:
                print("❌ No wallets available. Create a wallet first.")
                return
            miner = list(self.wallets.keys())[0]
        
        if miner not in self.wallets:
            print(f"❌ Wallet '{miner}' not found")
            return
        
        miner_address = self.wallets[miner]['address']
        
        print(f"\n⛏️  Mining {count} block(s) with {miner}...")
        
        for i in range(count):
            start_time = time.time()
            block = self.blockchain.mine_block(miner_address, max_transactions=10)
            
            if block:
                mining_time = time.time() - start_time
                print(f"   ✅ Block {i+1} mined in {mining_time:.2f}s")
                print(f"      Hash: {block.get_hash().to_hex()[:16]}...")
                print(f"      Nonce: {block.header.nonce}")
                print(f"      Transactions: {len(block.transactions)}")
            else:
                print(f"   ❌ Failed to mine block {i+1}")
    
    def create_transaction(self, sender: str, recipient: str, amount: int, fee: int = 1000) -> None:
        """Create a transaction between wallets."""
        if not self.blockchain:
            print("❌ Blockchain not initialized")
            return
        
        if sender not in self.wallets:
            print(f"❌ Sender wallet '{sender}' not found")
            return
        
        if recipient not in self.wallets:
            print(f"❌ Recipient wallet '{recipient}' not found")
            return
        
        sender_private = self.wallets[sender]['private_key']
        recipient_address = self.wallets[recipient]['address']
        
        print(f"\n💸 Creating transaction...")
        print(f"   From: {sender} ({self.wallets[sender]['address']})")
        print(f"   To: {recipient} ({recipient_address})")
        print(f"   Amount: {amount}")
        print(f"   Fee: {fee}")
        
        # Create transaction
        tx = self.blockchain.create_transfer_transaction(
            sender_private_key=sender_private,
            recipient_address=recipient_address,
            amount=amount,
            fee=fee
        )
        
        if tx:
            print(f"✅ Transaction created: {tx.get_hash().to_hex()[:16]}...")
            
            # Add to pending pool
            self.blockchain.add_transaction(tx)
            print("   Transaction added to pending pool")
        else:
            print("❌ Failed to create transaction")
    
    def check_balances(self) -> None:
        """Check balances of all wallets."""
        if not self.blockchain:
            print("❌ Blockchain not initialized")
            return
        
        print("\n💰 Wallet Balances:")
        print("-" * 40)
        
        for name, wallet in self.wallets.items():
            balance = self.blockchain.get_balance(wallet['address'])
            wallet['balance'] = balance
            print(f"   {name}: {balance:,} satoshis")
    
    def show_blockchain_info(self) -> None:
        """Show blockchain information."""
        if not self.blockchain:
            print("❌ Blockchain not initialized")
            return
        
        info = self.blockchain.get_chain_info()
        
        print("\n📊 Blockchain Information:")
        print("-" * 40)
        print(f"   Block count: {info['block_count']}")
        print(f"   Block height: {info['block_height']}")
        print(f"   Total difficulty: {info['total_difficulty']}")
        print(f"   Pending transactions: {info['pending_transactions']}")
        print(f"   UTXO count: {info['utxo_count']}")
        
        if 'current_difficulty' in info:
            print(f"   Current difficulty: {info['current_difficulty']}")
            print(f"   Average block time: {info.get('average_block_time', 0):.2f}s")
    
    def show_wallets(self) -> None:
        """Show all wallets."""
        if not self.wallets:
            print("❌ No wallets created")
            return
        
        print("\n👛 Wallets:")
        print("-" * 40)
        
        for name, wallet in self.wallets.items():
            print(f"   {name}:")
            print(f"      Address: {wallet['address']}")
            print(f"      Balance: {wallet.get('balance', 0):,} satoshis")
    
    def stress_test(self, transactions: int = 100) -> None:
        """Run a stress test with many transactions."""
        if not self.blockchain or len(self.wallets) < 2:
            print("❌ Need blockchain and at least 2 wallets for stress test")
            return
        
        print(f"\n🔥 Running stress test with {transactions} transactions...")
        
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
                print(f"   Created {i + 1} transactions...")
        
        end_time = time.time()
        print(f"✅ Stress test completed in {end_time - start_time:.2f}s")
        print(f"   Created {transactions} transactions")
        print(f"   Rate: {transactions / (end_time - start_time):.2f} tx/s")
    
    def chaos_mode(self) -> None:
        """Enter chaos mode - random operations."""
        print("\n🌪️  CHAOS MODE ACTIVATED!")
        print("Random operations will be performed...")
        
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
                    print(f"   💥 Chaos operation failed: {e}")
            
            time.sleep(0.5)
        
        print("🌪️  Chaos mode completed!")
    
    def help(self) -> None:
        """Show help information."""
        print("\n📖 Available Commands:")
        print("-" * 40)
        print("  setup [consensus]     - Set up blockchain (pos/dpos/pbft)")
        print("  wallet <name>         - Create a new wallet")
        print("  mine [count] [miner]  - Mine blocks")
        print("  send <from> <to> <amount> [fee] - Send transaction")
        print("  balance               - Check all balances")
        print("  info                  - Show blockchain info")
        print("  wallets               - Show all wallets")
        print("  stress [count]        - Run stress test")
        print("  chaos                 - Enter chaos mode")
        print("  validate              - Validate blockchain")
        print("  help                  - Show this help")
        print("  quit                  - Exit")
    
    def validate_blockchain(self) -> None:
        """Validate the entire blockchain."""
        if not self.blockchain:
            print("❌ Blockchain not initialized")
            return
        
        print("\n🔍 Validating blockchain...")
        start_time = time.time()
        
        is_valid = self.blockchain.validate_chain()
        
        end_time = time.time()
        print(f"✅ Validation completed in {end_time - start_time:.2f}s")
        print(f"   Result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    def run(self) -> None:
        """Run the interactive node."""
        while self.running:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == "quit" or cmd == "exit":
                    print("👋 Goodbye!")
                    break
                
                elif cmd == "help":
                    self.help()
                
                elif cmd == "setup":
                    consensus = command[1] if len(command) > 1 else "pos"
                    self.setup_blockchain(consensus)
                
                elif cmd == "wallet":
                    if len(command) < 2:
                        print("❌ Usage: wallet <name>")
                        continue
                    self.create_wallet(command[1])
                
                elif cmd == "mine":
                    count = int(command[1]) if len(command) > 1 else 1
                    miner = command[2] if len(command) > 2 else None
                    self.mine_blocks(count, miner)
                
                elif cmd == "send":
                    if len(command) < 4:
                        print("❌ Usage: send <from> <to> <amount> [fee]")
                        continue
                    
                    try:
                        sender = command[1]
                        recipient = command[2]
                        amount = int(command[3])
                        fee = int(command[4]) if len(command) > 4 else 1000
                        
                        self.create_transaction(sender, recipient, amount, fee)
                    except ValueError:
                        print("❌ Invalid amount or fee")
                
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
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function."""
    node = InteractiveNode()
    node.run()


if __name__ == "__main__":
    main()
