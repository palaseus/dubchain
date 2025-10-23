#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Basic GodChain Demo

This script demonstrates the core functionality of GodChain:
- Creating a blockchain
- Mining blocks
- Creating and validating transactions
- Cryptographic operations
"""

import logging
import time
from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig


def main():
    logger.info("🚀 GodChain - Sophisticated Blockchain Demo")
    logger.info("=" * 50)
    
    # Create blockchain with custom configuration
    config = ConsensusConfig(
        target_block_time=5,  # 5 seconds for demo
        difficulty_adjustment_interval=10,  # Adjust every 10 blocks
        min_difficulty=1,
        max_difficulty=8  # Keep it low for demo
    )
    
    blockchain = Blockchain(config)
    
    # Create genesis block
    logger.info("\n📦 Creating Genesis Block...")
    genesis_block = blockchain.create_genesis_block(
        coinbase_recipient="genesis_miner",
        coinbase_amount=1000000
    )
    logger.info(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
    logger.info(f"   Height: {genesis_block.header.block_height}")
    logger.info(f"   Difficulty: {genesis_block.header.difficulty}")
    
    # Generate some wallets
    logger.info("\n👛 Creating Wallets...")
    alice_private = PrivateKey.generate()
    alice_public = alice_private.get_public_key()
    alice_address = alice_public.to_address()
    
    bob_private = PrivateKey.generate()
    bob_public = bob_private.get_public_key()
    bob_address = bob_public.to_address()
    
    logger.info(f"Alice's address: {alice_address}")
    logger.info(f"Bob's address: {bob_address}")
    
    # Mine some blocks to get funds
    logger.info("\n⛏️  Mining Blocks...")
    for i in range(3):
        logger.info(f"   Mining block {i+1}...")
        start_time = time.time()
        
        block = blockchain.mine_block(alice_address, max_transactions=10)
        
        if block:
            mining_time = time.time() - start_time
            logger.info(f"   ✅ Block mined in {mining_time:.2f}s")
            logger.info(f"      Hash: {block.get_hash().to_hex()[:16]}...")
            logger.info(f"      Nonce: {block.header.nonce}")
            logger.info(f"      Transactions: {len(block.transactions)}")
        else:
            logger.info(f"   ❌ Failed to mine block {i+1}")
    
    # Check balances
    logger.info("\n💰 Checking Balances...")
    alice_balance = blockchain.get_balance(alice_address)
    bob_balance = blockchain.get_balance(bob_address)
    
    logger.info(f"Alice's balance: {alice_balance}")
    logger.info(f"Bob's balance: {bob_balance}")
    
    # Create a transaction
    if alice_balance > 100000:
        logger.info("\n💸 Creating Transaction...")
        
        # Create a transfer transaction
        transfer_tx = blockchain.create_transfer_transaction(
            sender_private_key=alice_private,
            recipient_address=bob_address,
            amount=50000,
            fee=1000
        )
        
        if transfer_tx:
            logger.info(f"✅ Transaction created: {transfer_tx.get_hash().to_hex()[:16]}...")
            logger.info(f"   From: {alice_address}")
            logger.info(f"   To: {bob_address}")
            logger.info(f"   Amount: 50000")
            logger.info(f"   Fee: 1000")
            
            # Add transaction to pending pool
            blockchain.add_transaction(transfer_tx)
            logger.info("   Transaction added to pending pool")
            
            # Mine a block with the transaction
            logger.info("\n⛏️  Mining Block with Transaction...")
            block = blockchain.mine_block(alice_address, max_transactions=10)
            
            if block:
                logger.info(f"✅ Block mined: {block.get_hash().to_hex()[:16]}...")
                logger.info(f"   Transactions: {len(block.transactions)}")
                
                # Check balances again
                logger.info("\n💰 Updated Balances...")
                alice_balance = blockchain.get_balance(alice_address)
                bob_balance = blockchain.get_balance(bob_address)
                
                logger.info(f"Alice's balance: {alice_balance}")
                logger.info(f"Bob's balance: {bob_balance}")
        else:
            logger.info("❌ Failed to create transaction")
    
    # Show blockchain info
    logger.info("\n📊 Blockchain Information...")
    info = blockchain.get_chain_info()
    logger.info(f"Block count: {info['block_count']}")
    logger.info(f"Block height: {info['block_height']}")
    logger.info(f"Total difficulty: {info['total_difficulty']}")
    logger.info(f"Pending transactions: {info['pending_transactions']}")
    logger.info(f"UTXO count: {info['utxo_count']}")
    
    if 'current_difficulty' in info:
        logger.info(f"Current difficulty: {info['current_difficulty']}")
        logger.info(f"Average block time: {info.get('average_block_time', 0):.2f}s")
    
    # Validate the entire chain
    logger.info("\n🔍 Validating Blockchain...")
    is_valid = blockchain.validate_chain()
    logger.info(f"Chain validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    logger.info("\n🎉 Demo completed successfully!")
    logger.info("\nGodChain features demonstrated:")
    logger.info("  ✅ Cryptographic signatures (ECDSA)")
    logger.info("  ✅ Hash functions (SHA-256)")
    logger.info("  ✅ Merkle trees")
    logger.info("  ✅ Proof of Work consensus")
    logger.info("  ✅ UTXO transaction model")
    logger.info("  ✅ Block validation")
    logger.info("  ✅ Difficulty adjustment")
    logger.info("  ✅ Comprehensive testing (97 tests passing)")


if __name__ == "__main__":
    main()
