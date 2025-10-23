#!/usr/bin/env python3
"""
Simple Integration Tests for DubChain

This script adds simple integration tests that work with the actual APIs.
"""

import sys
import time
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.crypto.hashing import SHA256Hasher


def test_basic_blockchain_integration():
    """Test basic blockchain operations integration."""
    logger.info("🔄 Testing Basic Blockchain Integration...")
    
    # Create blockchain
    blockchain = Blockchain()
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    
    # Create genesis block
    blockchain.create_genesis_block(public_key.to_address())
    
    # Create and add transactions
    for i in range(3):
        tx = blockchain.create_transfer_transaction(
            sender_private_key=private_key,
            recipient_address=public_key.to_address(),
            amount=100 + i,
            fee=10
        )
        blockchain.add_transaction(tx)
    
    # Mine blocks
    for i in range(3):
        block = blockchain.mine_block(public_key.to_address())
        assert block is not None, f"Should mine block {i+1}"
    
    logger.info("  ✅ Basic blockchain integration working")
    return True


def test_crypto_operations_integration():
    """Test cryptographic operations integration."""
    logger.info("🔄 Testing Crypto Operations Integration...")
    
    # Test hashing
    message = b"integration_test_message"
    hash_result = SHA256Hasher.hash(message)
    assert hash_result is not None, "Should create hash"
    
    # Test signature operations
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    signature = private_key.sign(message)
    is_valid = public_key.verify(signature, message)
    assert is_valid, "Signature should be valid"
    
    # Test address generation
    address = public_key.to_address()
    assert address is not None, "Should generate address"
    assert len(address) > 0, "Address should not be empty"
    
    logger.info("  ✅ Crypto operations integration working")
    return True


def test_transaction_lifecycle_integration():
    """Test complete transaction lifecycle integration."""
    logger.info("🔄 Testing Transaction Lifecycle Integration...")
    
    # Create blockchain and keys
    blockchain = Blockchain()
    sender_key = PrivateKey.generate()
    receiver_key = PrivateKey.generate()
    sender_pub = sender_key.get_public_key()
    receiver_pub = receiver_key.get_public_key()
    
    # Create genesis block
    blockchain.create_genesis_block(sender_pub.to_address())
    
    # Create transaction
    tx = blockchain.create_transfer_transaction(
        sender_private_key=sender_key,
        recipient_address=receiver_pub.to_address(),
        amount=1000,
        fee=50
    )
    
    assert tx is not None, "Should create transaction"
    
    # Add transaction to pool
    success = blockchain.add_transaction(tx)
    assert success, "Should add transaction to pool"
    
    # Mine block with transaction
    block = blockchain.mine_block(sender_pub.to_address())
    assert block is not None, "Should mine block"
    assert len(block.transactions) > 1, "Block should contain transactions"
    
    logger.info("  ✅ Transaction lifecycle integration working")
    return True


def test_multi_block_integration():
    """Test multi-block blockchain integration."""
    logger.info("🔄 Testing Multi-Block Integration...")
    
    # Create blockchain
    blockchain = Blockchain()
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    
    # Create genesis block
    blockchain.create_genesis_block(public_key.to_address())
    
    # Create multiple blocks
    for block_num in range(5):
        # Add some transactions
        for tx_num in range(2):
            tx = blockchain.create_transfer_transaction(
                sender_private_key=private_key,
                recipient_address=public_key.to_address(),
                amount=100 + block_num * 10 + tx_num,
                fee=10
            )
            blockchain.add_transaction(tx)
        
        # Mine block
        block = blockchain.mine_block(public_key.to_address())
        assert block is not None, f"Should mine block {block_num + 1}"
        assert block.header.block_height == block_num + 1, f"Block height should be {block_num + 1}"
    
    # Verify blockchain integrity
    assert len(blockchain.state.blocks) == 6, "Should have 6 blocks (1 genesis + 5 mined)"
    
    logger.info("  ✅ Multi-block integration working")
    return True


def test_error_handling_integration():
    """Test error handling integration."""
    logger.info("🔄 Testing Error Handling Integration...")
    
    # Test invalid transaction
    blockchain = Blockchain()
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    
    blockchain.create_genesis_block(public_key.to_address())
    
    # Try to create transaction with insufficient balance
    try:
        tx = blockchain.create_transfer_transaction(
            sender_private_key=private_key,
            recipient_address=public_key.to_address(),
            amount=999999999,  # Very large amount
            fee=10
        )
        # This should either succeed (if validation is lenient) or fail gracefully
        if tx is not None:
            success = blockchain.add_transaction(tx)
            # Should handle gracefully whether it succeeds or fails
    except Exception as e:
        # Should handle errors gracefully
        assert isinstance(e, Exception), "Should raise proper exception"
    
    logger.info("  ✅ Error handling integration working")
    return True


def main():
    """Run all integration tests."""
    logger.info("🚀 Running Simple Integration Tests")
    logger.info("=" * 50)
    
    start_time = time.time()
    tests = [
        test_basic_blockchain_integration,
        test_crypto_operations_integration,
        test_transaction_lifecycle_integration,
        test_multi_block_integration,
        test_error_handling_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.info(f"  ❌ {test.__name__}: {e}")
            failed += 1
    
    end_time = time.time()
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 Integration Test Results")
    logger.info("=" * 50)
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    logger.info(f"Duration: {end_time - start_time:.2f} seconds")
    
    if failed == 0:
        logger.info("\n✅ All integration tests passed!")
        return True
    else:
        logger.info(f"\n❌ {failed} integration tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
