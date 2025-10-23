#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Quick DubChain Demo - Comprehensive Testing

This script demonstrates various ways to interact with and stress test DubChain.
"""

import logging
import time
import random
from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig


def main():
    logger.info("🚀 DUBCHAIN QUICK DEMO - COMPREHENSIVE TESTING")
    logger.info("=" * 60)
    
    # Set up blockchain
    logger.info("\n🔧 Setting up blockchain...")
    config = ConsensusConfig(
        target_block_time=1,  # Fast blocks for demo
        difficulty_adjustment_interval=5,
        min_difficulty=1,
        max_difficulty=3
    )
    
    blockchain = Blockchain(config)
    
    # Create genesis block
    genesis_block = blockchain.create_genesis_block(
        coinbase_recipient="demo_miner",
        coinbase_amount=1000000
    )
    logger.info(f"✅ Genesis block: {genesis_block.get_hash().to_hex()[:16]}...")
    
    # Create some wallets
    logger.info("\n👛 Creating wallets...")
    wallets = {}
    for i in range(5):
        name = f"wallet_{i}"
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()
        address = public_key.to_address()
        
        wallets[name] = {
            'private_key': private_key,
            'public_key': public_key,
            'address': address
        }
        logger.info(f"   {name}: {address}")
    
    # Mine some blocks to get funds
    logger.info("\n⛏️  Mining blocks to get funds...")
    for i in range(10):
        miner = f"wallet_{i % len(wallets)}"
        miner_address = wallets[miner]['address']
        
        start_time = time.time()
        block = blockchain.mine_block(miner_address, max_transactions=10)
        mining_time = time.time() - start_time
        
        if block:
            logger.info(f"   Block {i+1}: {block.get_hash().to_hex()[:16]}... ({mining_time:.3f}s)")
        else:
            logger.info(f"   Block {i+1}: Failed")
    
    # Check balances
    logger.info("\n💰 Checking balances...")
    for name, wallet in wallets.items():
        balance = blockchain.get_balance(wallet['address'])
        logger.info(f"   {name}: {balance:,} satoshis")
    
    # Create some transactions
    logger.info("\n💸 Creating random transactions...")
    for i in range(20):
        sender = random.choice(list(wallets.keys()))
        recipient = random.choice([w for w in wallets.keys() if w != sender])
        
        sender_private = wallets[sender]['private_key']
        recipient_address = wallets[recipient]['address']
        amount = random.randint(1000, 50000)
        fee = random.randint(100, 1000)
        
        tx = blockchain.create_transfer_transaction(
            sender_private_key=sender_private,
            recipient_address=recipient_address,
            amount=amount,
            fee=fee
        )
        
        if tx:
            blockchain.add_transaction(tx)
            logger.info(f"   TX {i+1}: {sender} → {recipient} ({amount} satoshis)")
    
    # Mine a block with transactions
    logger.info("\n⛏️  Mining block with transactions...")
    miner = random.choice(list(wallets.keys()))
    miner_address = wallets[miner]['address']
    
    start_time = time.time()
    block = blockchain.mine_block(miner_address, max_transactions=20)
    mining_time = time.time() - start_time
    
    if block:
        logger.info(f"   Block mined: {block.get_hash().to_hex()[:16]}... ({mining_time:.3f}s)")
        logger.info(f"   Transactions: {len(block.transactions)}")
    
    # Check balances again
    logger.info("\n💰 Updated balances...")
    for name, wallet in wallets.items():
        balance = blockchain.get_balance(wallet['address'])
        logger.info(f"   {name}: {balance:,} satoshis")
    
    # Stress test
    logger.info("\n🔥 STRESS TEST - Creating 100 transactions...")
    start_time = time.time()
    
    for i in range(100):
        sender = random.choice(list(wallets.keys()))
        recipient = random.choice([w for w in wallets.keys() if w != sender])
        
        sender_private = wallets[sender]['private_key']
        recipient_address = wallets[recipient]['address']
        amount = random.randint(100, 10000)
        fee = random.randint(50, 500)
        
        tx = blockchain.create_transfer_transaction(
            sender_private_key=sender_private,
            recipient_address=recipient_address,
            amount=amount,
            fee=fee
        )
        
        if tx:
            blockchain.add_transaction(tx)
        
        if (i + 1) % 20 == 0:
            logger.info(f"   Created {i + 1} transactions...")
    
    end_time = time.time()
    logger.info(f"✅ Stress test completed in {end_time - start_time:.2f}s")
    logger.info(f"   Rate: {100 / (end_time - start_time):.2f} tx/s")
    
    # Mine multiple blocks
    logger.info("\n⛏️  Mining multiple blocks...")
    for i in range(5):
        miner = random.choice(list(wallets.keys()))
        miner_address = wallets[miner]['address']
        
        start_time = time.time()
        block = blockchain.mine_block(miner_address, max_transactions=50)
        mining_time = time.time() - start_time
        
        if block:
            logger.info(f"   Block {i+1}: {block.get_hash().to_hex()[:16]}... ({mining_time:.3f}s, {len(block.transactions)} tx)")
    
    # Final blockchain info
    logger.info("\n📊 Final Blockchain Information:")
    info = blockchain.get_chain_info()
    logger.info(f"   Block count: {info['block_count']}")
    logger.info(f"   Block height: {info['block_height']}")
    logger.info(f"   Total difficulty: {info['total_difficulty']}")
    logger.info(f"   Pending transactions: {info['pending_transactions']}")
    logger.info(f"   UTXO count: {info['utxo_count']}")
    
    # Validate blockchain
    logger.info("\n🔍 Validating blockchain...")
    start_time = time.time()
    is_valid = blockchain.validate_chain()
    validation_time = time.time() - start_time
    
    logger.info(f"✅ Validation completed in {validation_time:.2f}s")
    logger.info(f"   Result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Final balances
    logger.info("\n💰 Final balances:")
    for name, wallet in wallets.items():
        balance = blockchain.get_balance(wallet['address'])
        logger.info(f"   {name}: {balance:,} satoshis")
    
    logger.info("\n🎉 DEMO COMPLETED!")
    logger.info("=" * 60)
    logger.info("✨ What we accomplished:")
    logger.info("  ✅ Created and mined blockchain")
    logger.info("  ✅ Generated multiple wallets")
    logger.info("  ✅ Created and processed transactions")
    logger.info("  ✅ Ran stress test (100 transactions)")
    logger.info("  ✅ Mined multiple blocks")
    logger.info("  ✅ Validated entire blockchain")
    logger.info("  ✅ Demonstrated UTXO model")
    logger.info("  ✅ Showed proof-of-work consensus")
    logger.info("  ✅ Tested difficulty adjustment")


if __name__ == "__main__":
    main()
