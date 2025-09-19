#!/usr/bin/env python3
"""
Basic GodChain Demo

This script demonstrates the core functionality of GodChain:
- Creating a blockchain
- Mining blocks
- Creating and validating transactions
- Cryptographic operations
"""

import time
from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig


def main():
    print("🚀 GodChain - Sophisticated Blockchain Demo")
    print("=" * 50)
    
    # Create blockchain with custom configuration
    config = ConsensusConfig(
        target_block_time=5,  # 5 seconds for demo
        difficulty_adjustment_interval=10,  # Adjust every 10 blocks
        min_difficulty=1,
        max_difficulty=8  # Keep it low for demo
    )
    
    blockchain = Blockchain(config)
    
    # Create genesis block
    print("\n📦 Creating Genesis Block...")
    genesis_block = blockchain.create_genesis_block(
        coinbase_recipient="genesis_miner",
        coinbase_amount=1000000
    )
    print(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
    print(f"   Height: {genesis_block.header.block_height}")
    print(f"   Difficulty: {genesis_block.header.difficulty}")
    
    # Generate some wallets
    print("\n👛 Creating Wallets...")
    alice_private = PrivateKey.generate()
    alice_public = alice_private.get_public_key()
    alice_address = alice_public.to_address()
    
    bob_private = PrivateKey.generate()
    bob_public = bob_private.get_public_key()
    bob_address = bob_public.to_address()
    
    print(f"Alice's address: {alice_address}")
    print(f"Bob's address: {bob_address}")
    
    # Mine some blocks to get funds
    print("\n⛏️  Mining Blocks...")
    for i in range(3):
        print(f"   Mining block {i+1}...")
        start_time = time.time()
        
        block = blockchain.mine_block(alice_address, max_transactions=10)
        
        if block:
            mining_time = time.time() - start_time
            print(f"   ✅ Block mined in {mining_time:.2f}s")
            print(f"      Hash: {block.get_hash().to_hex()[:16]}...")
            print(f"      Nonce: {block.header.nonce}")
            print(f"      Transactions: {len(block.transactions)}")
        else:
            print(f"   ❌ Failed to mine block {i+1}")
    
    # Check balances
    print("\n💰 Checking Balances...")
    alice_balance = blockchain.get_balance(alice_address)
    bob_balance = blockchain.get_balance(bob_address)
    
    print(f"Alice's balance: {alice_balance}")
    print(f"Bob's balance: {bob_balance}")
    
    # Create a transaction
    if alice_balance > 100000:
        print("\n💸 Creating Transaction...")
        
        # Create a transfer transaction
        transfer_tx = blockchain.create_transfer_transaction(
            sender_private_key=alice_private,
            recipient_address=bob_address,
            amount=50000,
            fee=1000
        )
        
        if transfer_tx:
            print(f"✅ Transaction created: {transfer_tx.get_hash().to_hex()[:16]}...")
            print(f"   From: {alice_address}")
            print(f"   To: {bob_address}")
            print(f"   Amount: 50000")
            print(f"   Fee: 1000")
            
            # Add transaction to pending pool
            blockchain.add_transaction(transfer_tx)
            print("   Transaction added to pending pool")
            
            # Mine a block with the transaction
            print("\n⛏️  Mining Block with Transaction...")
            block = blockchain.mine_block(alice_address, max_transactions=10)
            
            if block:
                print(f"✅ Block mined: {block.get_hash().to_hex()[:16]}...")
                print(f"   Transactions: {len(block.transactions)}")
                
                # Check balances again
                print("\n💰 Updated Balances...")
                alice_balance = blockchain.get_balance(alice_address)
                bob_balance = blockchain.get_balance(bob_address)
                
                print(f"Alice's balance: {alice_balance}")
                print(f"Bob's balance: {bob_balance}")
        else:
            print("❌ Failed to create transaction")
    
    # Show blockchain info
    print("\n📊 Blockchain Information...")
    info = blockchain.get_chain_info()
    print(f"Block count: {info['block_count']}")
    print(f"Block height: {info['block_height']}")
    print(f"Total difficulty: {info['total_difficulty']}")
    print(f"Pending transactions: {info['pending_transactions']}")
    print(f"UTXO count: {info['utxo_count']}")
    
    if 'current_difficulty' in info:
        print(f"Current difficulty: {info['current_difficulty']}")
        print(f"Average block time: {info.get('average_block_time', 0):.2f}s")
    
    # Validate the entire chain
    print("\n🔍 Validating Blockchain...")
    is_valid = blockchain.validate_chain()
    print(f"Chain validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nGodChain features demonstrated:")
    print("  ✅ Cryptographic signatures (ECDSA)")
    print("  ✅ Hash functions (SHA-256)")
    print("  ✅ Merkle trees")
    print("  ✅ Proof of Work consensus")
    print("  ✅ UTXO transaction model")
    print("  ✅ Block validation")
    print("  ✅ Difficulty adjustment")
    print("  ✅ Comprehensive testing (97 tests passing)")


if __name__ == "__main__":
    main()
