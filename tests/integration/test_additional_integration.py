#!/usr/bin/env python3
"""
Additional Integration Tests for DubChain

This script adds comprehensive integration tests to improve
production readiness and test coverage.
"""

import sys
import time
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.consensus import ProofOfStake, ConsensusConfig
from dubchain.sharding.shard_manager import ShardManager
from dubchain.sharding.shard_types import ShardConfig


def test_blockchain_consensus_integration():
    """Test integration between blockchain and consensus mechanisms."""
    logger.info("🔄 Testing Blockchain-Consensus Integration...")
    
    # Create blockchain with consensus
    blockchain = Blockchain()
    config = ConsensusConfig()
    pos = ProofOfStake(config)
    
    # Add validators
    pos.add_validator("validator1", 1000)
    pos.add_validator("validator2", 2000)
    pos.add_validator("validator3", 1500)
    
    # Create genesis block
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    blockchain.create_genesis_block(public_key.to_address())
    
    # Test consensus selection
    proposer = pos.select_proposer(1)
    assert proposer is not None, "Should select a proposer"
    
    # Test block creation with consensus
    for i in range(5):
        block = blockchain.mine_block(public_key.to_address())
        assert block is not None, f"Should create block {i+1}"
    
    logger.info("  ✅ Blockchain-Consensus integration working")
    return True


def test_crypto_integration():
    """Test cryptographic operations integration."""
    logger.info("🔄 Testing Crypto Integration...")
    
    from dubchain.crypto.hashing import SHA256Hasher
    
    # Test hashing
    message = b"test_message"
    hash_result = SHA256Hasher.hash(message)
    assert hash_result is not None, "Should create hash"
    
    # Test signature operations
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    signature = private_key.sign(message)
    is_valid = public_key.verify(signature, message)
    assert is_valid, "Signature should be valid"
    
    logger.info("  ✅ Crypto integration working")
    return True


def test_sharding_integration():
    """Test sharding integration with blockchain."""
    logger.info("🔄 Testing Sharding Integration...")
    
    # Create shard manager
    config = ShardConfig()
    shard_manager = ShardManager(config)
    
    # Create shards
    shards = []
    for i in range(3):
        shard = shard_manager.create_shard(f"shard_{i}")
        shards.append(shard)
        assert shard is not None, f"Should create shard {i}"
    
    assert len(shards) == 3, "Should create 3 shards"
    
    # Test shard state management
    for shard in shards:
        state = shard_manager.get_shard_state(shard.shard_id)
        assert state is not None, f"Should get state for shard {shard.shard_id}"
    
    logger.info("  ✅ Sharding integration working")
    return True


def test_cross_module_integration():
    """Test integration across multiple modules."""
    logger.info("🔄 Testing Cross-Module Integration...")
    
    # Create blockchain with all components
    blockchain = Blockchain()
    config = ConsensusConfig()
    pos = ProofOfStake(config)
    shard_config = ShardConfig()
    shard_manager = ShardManager(shard_config)
    
    # Setup
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    blockchain.create_genesis_block(public_key.to_address())
    
    # Add validators
    pos.add_validator("validator1", 1000)
    pos.add_validator("validator2", 2000)
    
    # Create shard
    shard = shard_manager.create_shard("main_shard")
    
    # Test integrated workflow
    # 1. Create transaction
    tx = blockchain.create_transfer_transaction(
        sender_private_key=private_key,
        recipient_address=public_key.to_address(),
        amount=100,
        fee=10
    )
    
    # 2. Add to blockchain
    blockchain.add_transaction(tx)
    
    # 3. Mine block
    block = blockchain.mine_block(public_key.to_address())
    
    # 4. Verify block
    assert block is not None, "Should mine block"
    assert len(block.transactions) > 0, "Block should have transactions"
    
    # 5. Test consensus
    proposer = pos.select_proposer(2)
    assert proposer is not None, "Should select proposer"
    
    # 6. Test shard state
    shard_state = shard_manager.get_shard_state(shard.shard_id)
    assert shard_state is not None, "Should get shard state"
    
    logger.info("  ✅ Cross-module integration working")
    return True


def test_performance_integration():
    """Test performance monitoring integration."""
    logger.info("🔄 Testing Performance Integration...")
    
    from dubchain.performance.monitoring import PerformanceMonitor
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Test performance tracking
    with monitor.track_operation("test_operation"):
        time.sleep(0.01)  # Simulate work
    
    # Get metrics
    metrics = monitor.get_metrics()
    assert metrics is not None, "Should get performance metrics"
    
    logger.info("  ✅ Performance integration working")
    return True


def test_security_integration():
    """Test security features integration."""
    logger.info("🔄 Testing Security Integration...")
    
    from dubchain.crypto.signatures import PrivateKey, PublicKey
    from dubchain.crypto.hashing import SHA256Hasher
    
    # Test cryptographic operations
    private_key = PrivateKey.generate()
    public_key = private_key.get_public_key()
    
    message = b"security_test_message"
    signature = private_key.sign(message)
    
    # Verify signature
    is_valid = public_key.verify(signature, message)
    assert is_valid, "Signature should be valid"
    
    # Test hashing
    hash_result = SHA256Hasher.hash(message)
    assert hash_result is not None, "Should create hash"
    
    logger.info("  ✅ Security integration working")
    return True


def main():
    """Run all integration tests."""
    logger.info("🚀 Running Additional Integration Tests")
    logger.info("=" * 50)
    
    start_time = time.time()
    tests = [
        test_blockchain_consensus_integration,
        test_crypto_integration,
        test_sharding_integration,
        test_cross_module_integration,
        test_performance_integration,
        test_security_integration,
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
